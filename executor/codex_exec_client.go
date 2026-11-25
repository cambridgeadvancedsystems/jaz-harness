package executor

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"path"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

type rpcFrame struct {
	JSONRPC string          `json:"jsonrpc,omitempty"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// CodexExecClient speaks upstream unforked `codex exec-server`'s WebSocket
// JSON-RPC protocol directly from Go — sending Authorization: Bearer headers
// over cluster-internal ws:// natively (no loopback TCP forwarder required).
type CodexExecClient struct {
	conn         *websocket.Conn
	workspaceDir string
	writeMu      sync.Mutex
	nextID       atomic.Int64
	pendingMu    sync.Mutex
	pending      map[string]chan rpcFrame
	procMu       sync.Mutex
	procs        map[string]*execProcState
	done         chan struct{}
}

const (
	maxWSMessageBytes = 64 << 20
	maxRetainedProcs  = 256
)

type execProcState struct {
	stdout   *cappedBuffer
	stderr   *cappedBuffer
	exitCode int
	exited   bool
	doneCh   chan struct{}
}

func DialCodexExecServer(ctx context.Context, wsURL, bearerToken, workspaceDir string) (*CodexExecClient, error) {
	if workspaceDir == "" {
		workspaceDir = "/workspace"
	}
	headers := http.Header{}
	if bearerToken != "" {
		headers.Set("Authorization", "Bearer "+bearerToken)
	}
	dialer := websocket.Dialer{HandshakeTimeout: 15 * time.Second}
	conn, _, err := dialer.DialContext(ctx, wsURL, headers)
	if err != nil {
		return nil, fmt.Errorf("dial codex exec-server %s: %w", wsURL, err)
	}
	conn.SetReadLimit(maxWSMessageBytes)
	c := &CodexExecClient{
		conn:         conn,
		workspaceDir: workspaceDir,
		pending:      make(map[string]chan rpcFrame),
		procs:        make(map[string]*execProcState),
		done:         make(chan struct{}),
	}
	go c.readLoop()

	if err := c.call(ctx, "initialize", map[string]any{
		"clientName": "jaz-harness",
	}, nil); err != nil {
		_ = c.Close()
		return nil, fmt.Errorf("codex exec-server initialize: %w", err)
	}
	_ = c.notify("initialized", map[string]any{})
	return c, nil
}

func (c *CodexExecClient) WorkspaceDir() string { return c.workspaceDir }

func (c *CodexExecClient) Close() error {
	c.writeMu.Lock()
	err := c.conn.Close()
	c.writeMu.Unlock()
	<-c.done
	return err
}

func (c *CodexExecClient) readLoop() {
	defer close(c.done)
	for {
		_, data, err := c.conn.ReadMessage()
		if err != nil {
			c.pendingMu.Lock()
			for k, ch := range c.pending {
				close(ch)
				delete(c.pending, k)
			}
			c.pendingMu.Unlock()
			return
		}
		var frame rpcFrame
		if json.Unmarshal(data, &frame) != nil {
			continue
		}
		if frame.Method != "" && len(frame.ID) == 0 {
			c.handleNotification(frame.Method, frame.Params)
			continue
		}
		if len(frame.ID) > 0 {
			key := string(frame.ID)
			c.pendingMu.Lock()
			ch, ok := c.pending[key]
			if ok {
				delete(c.pending, key)
			}
			c.pendingMu.Unlock()
			if ok {
				ch <- frame
			}
		}
	}
}

func (c *CodexExecClient) handleNotification(method string, params json.RawMessage) {
	switch method {
	case "process/output":
		var n struct {
			ProcessID string `json:"processId"`
			Stream    string `json:"stream"`
			Chunk     string `json:"chunk"` // base64 bytes
		}
		if json.Unmarshal(params, &n) != nil {
			return
		}
		decoded, err := base64.StdEncoding.DecodeString(n.Chunk)
		if err != nil {
			decoded = []byte(n.Chunk)
		}
		c.procMu.Lock()
		st := c.ensureProcLocked(n.ProcessID)
		if strings.EqualFold(n.Stream, "stderr") {
			_, _ = st.stderr.Write(decoded)
		} else {
			_, _ = st.stdout.Write(decoded)
		}
		c.procMu.Unlock()

	case "process/exited":
		var n struct {
			ProcessID string `json:"processId"`
			ExitCode  int    `json:"exitCode"`
		}
		if json.Unmarshal(params, &n) != nil {
			return
		}
		c.procMu.Lock()
		st := c.ensureProcLocked(n.ProcessID)
		st.exitCode = n.ExitCode
		if !st.exited {
			st.exited = true
			close(st.doneCh)
		}
		c.procMu.Unlock()
	}
}

func (c *CodexExecClient) ensureProcLocked(id string) *execProcState {
	st, ok := c.procs[id]
	if !ok {
		if len(c.procs) >= maxRetainedProcs {
			for k, existing := range c.procs {
				if existing.exited {
					delete(c.procs, k)
				}
			}
		}
		st = &execProcState{
			stdout: newCappedBuffer(DefaultMaxExecOutputBytes),
			stderr: newCappedBuffer(DefaultMaxExecOutputBytes),
			doneCh: make(chan struct{}),
		}
		c.procs[id] = st
	}
	return st
}

func (c *CodexExecClient) sendFrame(frame rpcFrame) error {
	raw, err := json.Marshal(frame)
	if err != nil {
		return err
	}
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	return c.conn.WriteMessage(websocket.TextMessage, raw)
}

func (c *CodexExecClient) notify(method string, params any) error {
	raw, _ := json.Marshal(params)
	return c.sendFrame(rpcFrame{
		JSONRPC: "2.0",
		Method:  method,
		Params:  raw,
	})
}

func (c *CodexExecClient) call(ctx context.Context, method string, params any, out any) error {
	id := c.nextID.Add(1)
	idRaw, _ := json.Marshal(id)
	key := string(idRaw)
	ch := make(chan rpcFrame, 1)

	c.pendingMu.Lock()
	c.pending[key] = ch
	c.pendingMu.Unlock()

	paramsRaw, err := json.Marshal(params)
	if err != nil {
		c.pendingMu.Lock()
		delete(c.pending, key)
		c.pendingMu.Unlock()
		return err
	}
	if err := c.sendFrame(rpcFrame{
		JSONRPC: "2.0",
		ID:      idRaw,
		Method:  method,
		Params:  paramsRaw,
	}); err != nil {
		c.pendingMu.Lock()
		delete(c.pending, key)
		c.pendingMu.Unlock()
		return err
	}

	select {
	case <-ctx.Done():
		c.pendingMu.Lock()
		delete(c.pending, key)
		c.pendingMu.Unlock()
		return ctx.Err()
	case resp, ok := <-ch:
		if !ok {
			return errors.New("codex exec-server closed before response")
		}
		if resp.Error != nil {
			return fmt.Errorf("codex exec-server %s error %d: %s", method, resp.Error.Code, resp.Error.Message)
		}
		if out != nil && len(resp.Result) > 0 {
			return json.Unmarshal(resp.Result, out)
		}
		return nil
	}
}

func (c *CodexExecClient) ExecStart(ctx context.Context, req ExecRequest) (ExecResult, error) {
	procID := "proc_" + uuid.NewString()[:8]
	c.procMu.Lock()
	st := c.ensureProcLocked(procID)
	c.procMu.Unlock()

	argv := req.Args
	if req.Shell || len(argv) == 0 {
		argv = []string{"/bin/sh", "-lc", req.Command}
	} else {
		argv = append([]string{req.Command}, req.Args...)
	}
	cwd := req.Cwd
	if cwd == "" {
		cwd = c.workspaceDir
	}
	var startRes struct {
		ProcessID string `json:"processId"`
	}
	if err := c.call(ctx, "process/start", map[string]any{
		"processId": procID,
		"argv":      argv,
		"cwd":       cwd,
		"env":       req.Env,
	}, &startRes); err != nil {
		return ExecResult{}, err
	}
	if startRes.ProcessID != "" {
		procID = startRes.ProcessID
	}

	waitDur := 30 * time.Second
	if req.TimeoutMS > 0 {
		waitDur = time.Duration(req.TimeoutMS) * time.Millisecond
	}
	select {
	case <-st.doneCh:
	case <-time.After(waitDur):
	case <-ctx.Done():
		return ExecResult{}, ctx.Err()
	}

	c.procMu.Lock()
	defer c.procMu.Unlock()
	return ExecResult{
		ProcessID: procID,
		Stdout:    st.stdout.String(),
		Stderr:    st.stderr.String(),
		ExitCode:  st.exitCode,
		Running:   !st.exited,
	}, nil
}

func (c *CodexExecClient) ExecWriteStdin(ctx context.Context, processID, data string) (ExecResult, error) {
	if err := c.call(ctx, "process/write", map[string]any{
		"processId": processID,
		"chunk":     base64.StdEncoding.EncodeToString([]byte(data)),
	}, nil); err != nil {
		return ExecResult{}, err
	}
	c.procMu.Lock()
	st := c.ensureProcLocked(processID)
	c.procMu.Unlock()
	select {
	case <-st.doneCh:
	case <-time.After(2 * time.Second):
	case <-ctx.Done():
		return ExecResult{}, ctx.Err()
	}
	c.procMu.Lock()
	defer c.procMu.Unlock()
	return ExecResult{
		ProcessID: processID,
		Stdout:    st.stdout.String(),
		Stderr:    st.stderr.String(),
		ExitCode:  st.exitCode,
		Running:   !st.exited,
	}, nil
}

func (c *CodexExecClient) ExecTerminate(ctx context.Context, processID string) error {
	return c.call(ctx, "process/terminate", map[string]any{
		"processId": processID,
	}, nil)
}

func (c *CodexExecClient) fullPath(rel string) string {
	if strings.HasPrefix(rel, "/") {
		return path.Clean(rel)
	}
	return path.Join(c.workspaceDir, rel)
}

func (c *CodexExecClient) ReadFile(ctx context.Context, filePath string) ([]byte, error) {
	var res struct {
		DataBase64 string `json:"dataBase64"`
		Content    string `json:"content"`
	}
	if err := c.call(ctx, "fs/readFile", map[string]any{
		"path": c.fullPath(filePath),
	}, &res); err != nil {
		return nil, err
	}
	if res.DataBase64 != "" {
		return base64.StdEncoding.DecodeString(res.DataBase64)
	}
	return []byte(res.Content), nil
}

func (c *CodexExecClient) WriteFile(ctx context.Context, filePath string, data []byte) error {
	return c.call(ctx, "fs/writeFile", map[string]any{
		"path":       c.fullPath(filePath),
		"dataBase64": base64.StdEncoding.EncodeToString(data),
	}, nil)
}

func (c *CodexExecClient) ListDir(ctx context.Context, dirPath string) ([]FileEntry, error) {
	var res struct {
		Entries []struct {
			FileName    string `json:"fileName"`
			IsDirectory bool   `json:"isDirectory"`
		} `json:"entries"`
	}
	if err := c.call(ctx, "fs/readDirectory", map[string]any{
		"path": c.fullPath(dirPath),
	}, &res); err != nil {
		return nil, err
	}
	out := make([]FileEntry, 0, len(res.Entries))
	for _, e := range res.Entries {
		out = append(out, FileEntry{
			Name:  e.FileName,
			Path:  path.Join(dirPath, e.FileName),
			IsDir: e.IsDirectory,
		})
	}
	return out, nil
}
