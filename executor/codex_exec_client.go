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

const (
	maxWSMessageBytes    = 64 << 20
	maxRetainedProcs     = 256
	rpcErrCodeNotFound   = -32004
	processExitGraceWait = 250 * time.Millisecond
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

type rpcError struct {
	Method  string
	Code    int
	Message string
}

func (e *rpcError) Error() string {
	return fmt.Sprintf("codex exec-server %s error %d: %s", e.Method, e.Code, e.Message)
}

type initializeParams struct {
	ClientName      string `json:"clientName"`
	ResumeSessionID string `json:"resumeSessionId,omitempty"`
}

type initializeResult struct {
	SessionID       string               `json:"sessionId"`
	EnvironmentInfo *execEnvironmentInfo `json:"environmentInfo,omitempty"`
}

type execEnvironmentInfo struct {
	Cwd        string `json:"cwd,omitempty"`
	PlatformOS string `json:"platformOs,omitempty"`
}

type execEnvPolicy struct {
	Inherit               string            `json:"inherit"`
	IgnoreDefaultExcludes bool              `json:"ignoreDefaultExcludes"`
	Exclude               []string          `json:"exclude"`
	Set                   map[string]string `json:"set"`
	IncludeOnly           []string          `json:"includeOnly"`
}

type processStartParams struct {
	ProcessID string            `json:"processId"`
	Argv      []string          `json:"argv"`
	Cwd       string            `json:"cwd"`
	EnvPolicy *execEnvPolicy    `json:"envPolicy,omitempty"`
	Env       map[string]string `json:"env"`
	TTY       bool              `json:"tty"`
	PipeStdin bool              `json:"pipeStdin,omitempty"`
}

type processStartResult struct {
	ProcessID string `json:"processId"`
}

type processReadParams struct {
	ProcessID string  `json:"processId"`
	AfterSeq  *uint64 `json:"afterSeq,omitempty"`
	MaxBytes  *int    `json:"maxBytes,omitempty"`
	WaitMS    *uint64 `json:"waitMs,omitempty"`
}

type processWriteParams struct {
	ProcessID string `json:"processId"`
	Chunk     string `json:"chunk"`
	WriteID   string `json:"writeId"`
}

type processWriteResult struct {
	Status string `json:"status"`
}

type processTerminateParams struct {
	ProcessID string `json:"processId"`
}

type fsPathParams struct {
	Path string `json:"path"`
}

type fsWriteFileParams struct {
	Path       string `json:"path"`
	DataBase64 string `json:"dataBase64"`
}

type fsCreateDirParams struct {
	Path      string `json:"path"`
	Recursive bool   `json:"recursive"`
}

// FileMetadata describes the result of `fs/getMetadata`.
type FileMetadata struct {
	IsDirectory  bool   `json:"isDirectory"`
	IsFile       bool   `json:"isFile"`
	IsSymlink    bool   `json:"isSymlink"`
	Size         uint64 `json:"size"`
	CreatedAtMS  int64  `json:"createdAtMs"`
	ModifiedAtMS int64  `json:"modifiedAtMs"`
}

// ExecOutputChunk is one retained output chunk returned by `process/read`.
type ExecOutputChunk struct {
	Seq    uint64 `json:"seq"`
	Stream string `json:"stream"`
	Data   []byte `json:"data"`
}

// ExecReadResult holds the decoded response from `process/read`.
type ExecReadResult struct {
	Chunks        []ExecOutputChunk
	NextSeq       uint64
	Exited        bool
	ExitCode      *int
	Closed        bool
	Failure       string
	SandboxDenied bool
}

// CodexExecClient speaks upstream unforked `codex exec-server`'s WebSocket
// JSON-RPC protocol (0.171.0+) directly from Go — sending Authorization: Bearer
// headers over cluster-internal ws:// natively (no loopback forwarder required).
type CodexExecClient struct {
	conn         *websocket.Conn
	workspaceDir string
	platformOS   string
	writeMu      sync.Mutex
	nextID       atomic.Int64
	pendingMu    sync.Mutex
	pending      map[string]chan rpcFrame
	procMu       sync.Mutex
	procs        map[string]*execProcState
	done         chan struct{}
}

type execProcState struct {
	stdout   *CappedBuffer
	stderr   *CappedBuffer
	exitCode int
	exited   bool
	closed   bool
	exitedCh chan struct{}
	closedCh chan struct{}
}

func DialCodexExecServer(ctx context.Context, wsURL, bearerToken, workspaceDir string) (*CodexExecClient, error) {
	if strings.TrimSpace(workspaceDir) == "" {
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

	var initRes initializeResult
	if err := c.call(ctx, "initialize", initializeParams{
		ClientName: "jaz-harness",
	}, &initRes); err != nil {
		_ = c.Close()
		return nil, fmt.Errorf("codex exec-server initialize: %w", err)
	}
	if initRes.EnvironmentInfo != nil {
		c.platformOS = strings.ToLower(strings.TrimSpace(initRes.EnvironmentInfo.PlatformOS))
	}
	if err := c.notify("initialized", struct{}{}); err != nil {
		_ = c.Close()
		return nil, fmt.Errorf("codex exec-server initialized: %w", err)
	}
	return c, nil
}

func (c *CodexExecClient) WorkspaceDir() string {
	return c.workspaceDir
}

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
			Chunk     string `json:"chunk"`
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
			close(st.exitedCh)
		}
		c.procMu.Unlock()

	case "process/closed":
		var n struct {
			ProcessID string `json:"processId"`
		}
		if json.Unmarshal(params, &n) != nil {
			return
		}
		c.procMu.Lock()
		st := c.ensureProcLocked(n.ProcessID)
		if !st.closed {
			st.closed = true
			close(st.closedCh)
		}
		c.procMu.Unlock()
	}
}

func (c *CodexExecClient) ensureProcLocked(id string) *execProcState {
	st, ok := c.procs[id]
	if !ok {
		if len(c.procs) >= maxRetainedProcs {
			for k, existing := range c.procs {
				if existing.closed || existing.exited {
					delete(c.procs, k)
				}
			}
		}
		st = &execProcState{
			stdout:   newCappedBuffer(DefaultMaxExecOutputBytes),
			stderr:   newCappedBuffer(DefaultMaxExecOutputBytes),
			exitedCh: make(chan struct{}),
			closedCh: make(chan struct{}),
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
	raw, err := json.Marshal(params)
	if err != nil {
		return err
	}
	return c.sendFrame(rpcFrame{
		Method: method,
		Params: raw,
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
		ID:     idRaw,
		Method: method,
		Params: paramsRaw,
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
			return &rpcError{
				Method:  method,
				Code:    resp.Error.Code,
				Message: resp.Error.Message,
			}
		}
		if out != nil && len(resp.Result) > 0 {
			return json.Unmarshal(resp.Result, out)
		}
		return nil
	}
}

func (c *CodexExecClient) shellArgv(command string) []string {
	if c.platformOS == "windows" {
		return []string{"cmd.exe", "/C", command}
	}
	return []string{"/bin/sh", "-lc", command}
}

func (c *CodexExecClient) waitProcess(ctx context.Context, procID string, st *execProcState, waitDur time.Duration) (ExecResult, error) {
	timer := time.NewTimer(waitDur)
	defer timer.Stop()

	select {
	case <-st.closedCh:
	case <-st.exitedCh:
		grace := time.NewTimer(processExitGraceWait)
		select {
		case <-st.closedCh:
		case <-grace.C:
		case <-timer.C:
		case <-ctx.Done():
			grace.Stop()
			return ExecResult{}, ctx.Err()
		case <-c.done:
			grace.Stop()
			return ExecResult{}, errors.New("codex exec-server closed while waiting for process")
		}
		grace.Stop()
	case <-timer.C:
	case <-ctx.Done():
		return ExecResult{}, ctx.Err()
	case <-c.done:
		return ExecResult{}, errors.New("codex exec-server closed while waiting for process")
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

func (c *CodexExecClient) ExecStart(ctx context.Context, req ExecRequest) (ExecResult, error) {
	procID := "proc_" + uuid.NewString()[:8]
	c.procMu.Lock()
	st := c.ensureProcLocked(procID)
	c.procMu.Unlock()

	argv := req.Args
	if req.Shell || len(argv) == 0 {
		argv = c.shellArgv(req.Command)
	} else {
		argv = append([]string{req.Command}, req.Args...)
	}
	cwd := req.Cwd
	if strings.TrimSpace(cwd) == "" {
		cwd = c.workspaceDir
	}
	env := req.Env
	if env == nil {
		env = map[string]string{}
	}
	var startRes processStartResult
	if err := c.call(ctx, "process/start", processStartParams{
		ProcessID: procID,
		Argv:      argv,
		Cwd:       c.fileURI(cwd),
		EnvPolicy: &execEnvPolicy{
			Inherit:               "all",
			IgnoreDefaultExcludes: false,
			Exclude:               []string{},
			Set:                   map[string]string{},
			IncludeOnly:           []string{},
		},
		Env:       env,
		TTY:       false,
		PipeStdin: req.PipeStdin,
	}, &startRes); err != nil {
		c.procMu.Lock()
		delete(c.procs, procID)
		c.procMu.Unlock()
		return ExecResult{}, err
	}
	if startRes.ProcessID != "" && startRes.ProcessID != procID {
		procID = startRes.ProcessID
		c.procMu.Lock()
		st = c.ensureProcLocked(procID)
		c.procMu.Unlock()
	}

	waitDur := 30 * time.Second
	if req.TimeoutMS > 0 {
		waitDur = time.Duration(req.TimeoutMS) * time.Millisecond
	}
	return c.waitProcess(ctx, procID, st, waitDur)
}

func (c *CodexExecClient) ExecRead(ctx context.Context, processID string, afterSeq uint64, maxBytes int, waitMS uint64) (ExecReadResult, error) {
	params := processReadParams{
		ProcessID: processID,
		AfterSeq:  &afterSeq,
		WaitMS:    &waitMS,
	}
	if maxBytes > 0 {
		params.MaxBytes = &maxBytes
	}
	var raw struct {
		Chunks []struct {
			Seq    uint64 `json:"seq"`
			Stream string `json:"stream"`
			Chunk  string `json:"chunk"`
		} `json:"chunks"`
		NextSeq       uint64 `json:"nextSeq"`
		Exited        bool   `json:"exited"`
		ExitCode      *int   `json:"exitCode"`
		Closed        bool   `json:"closed"`
		Failure       string `json:"failure"`
		SandboxDenied bool   `json:"sandboxDenied"`
	}
	if err := c.call(ctx, "process/read", params, &raw); err != nil {
		return ExecReadResult{}, err
	}
	chunks := make([]ExecOutputChunk, 0, len(raw.Chunks))
	for _, ch := range raw.Chunks {
		decoded, err := base64.StdEncoding.DecodeString(ch.Chunk)
		if err != nil {
			decoded = []byte(ch.Chunk)
		}
		chunks = append(chunks, ExecOutputChunk{
			Seq:    ch.Seq,
			Stream: ch.Stream,
			Data:   decoded,
		})
	}
	return ExecReadResult{
		Chunks:        chunks,
		NextSeq:       raw.NextSeq,
		Exited:        raw.Exited,
		ExitCode:      raw.ExitCode,
		Closed:        raw.Closed,
		Failure:       raw.Failure,
		SandboxDenied: raw.SandboxDenied,
	}, nil
}

func (c *CodexExecClient) ExecWriteStdin(ctx context.Context, processID, data string) (ExecResult, error) {
	var res processWriteResult
	if err := c.call(ctx, "process/write", processWriteParams{
		ProcessID: processID,
		Chunk:     base64.StdEncoding.EncodeToString([]byte(data)),
		WriteID:   "w_" + uuid.NewString()[:8],
	}, &res); err != nil {
		return ExecResult{}, err
	}
	if res.Status != "" && res.Status != "accepted" {
		return ExecResult{}, fmt.Errorf("codex exec-server process/write status: %s", res.Status)
	}
	c.procMu.Lock()
	st := c.ensureProcLocked(processID)
	c.procMu.Unlock()
	return c.waitProcess(ctx, processID, st, 2*time.Second)
}

func (c *CodexExecClient) ExecTerminate(ctx context.Context, processID string) error {
	return c.call(ctx, "process/terminate", processTerminateParams{
		ProcessID: processID,
	}, nil)
}

func (c *CodexExecClient) Stat(ctx context.Context, targetPath string) (FileMetadata, error) {
	var res FileMetadata
	if err := c.call(ctx, "fs/getMetadata", fsPathParams{
		Path: c.fileURI(targetPath),
	}, &res); err != nil {
		return FileMetadata{}, err
	}
	return res, nil
}

func (c *CodexExecClient) ReadFile(ctx context.Context, filePath string) ([]byte, error) {
	var res struct {
		DataBase64 string `json:"dataBase64"`
		Content    string `json:"content"`
	}
	if err := c.call(ctx, "fs/readFile", fsPathParams{
		Path: c.fileURI(filePath),
	}, &res); err != nil {
		return nil, err
	}
	if res.DataBase64 != "" {
		return base64.StdEncoding.DecodeString(res.DataBase64)
	}
	return []byte(res.Content), nil
}

func (c *CodexExecClient) WriteFile(ctx context.Context, filePath string, data []byte) error {
	uri := c.fileURI(filePath)
	params := fsWriteFileParams{
		Path:       uri,
		DataBase64: base64.StdEncoding.EncodeToString(data),
	}
	err := c.call(ctx, "fs/writeFile", params, nil)
	if err == nil {
		return nil
	}
	var rpcErr *rpcError
	if !errors.As(err, &rpcErr) || rpcErr.Code != rpcErrCodeNotFound {
		return err
	}
	parentURI := parentFileURI(uri)
	if parentURI == "" || parentURI == uri {
		return err
	}
	if mkErr := c.call(ctx, "fs/createDirectory", fsCreateDirParams{
		Path:      parentURI,
		Recursive: true,
	}, nil); mkErr != nil {
		return err
	}
	return c.call(ctx, "fs/writeFile", params, nil)
}

func (c *CodexExecClient) ListDir(ctx context.Context, dirPath string) ([]FileEntry, error) {
	var res struct {
		Entries []struct {
			FileName    string `json:"fileName"`
			IsDirectory bool   `json:"isDirectory"`
		} `json:"entries"`
	}
	if err := c.call(ctx, "fs/readDirectory", fsPathParams{
		Path: c.fileURI(dirPath),
	}, &res); err != nil {
		return nil, err
	}
	baseDir := displayDirPath(dirPath)
	out := make([]FileEntry, 0, len(res.Entries))
	for _, e := range res.Entries {
		out = append(out, FileEntry{
			Name:  e.FileName,
			Path:  path.Join(baseDir, e.FileName),
			IsDir: e.IsDirectory,
		})
	}
	return out, nil
}

func (c *CodexExecClient) fileURI(raw string) string {
	return pathToFileURI(c.workspaceDir, raw)
}
