package execserver

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/executor"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

const (
	clientMaxRetainedProcs = 256
	clientExitGraceWait    = 250 * time.Millisecond
	clientWriteStdinWait   = 2 * time.Second
)

// DialConfig configures a WebSocket JSON-RPC client connection to
// jaz-exec-server.
type DialConfig struct {
	URL               string
	BearerToken       string
	WorkspaceDir      string
	ClientName        string
	HandshakeTimeout  time.Duration
	MaxWSMessageBytes int64
	MaxOutputBytes    int
}

type clientProcState struct {
	stdout   *executor.CappedBuffer
	stderr   *executor.CappedBuffer
	exitCode int
	exited   bool
	closed   bool
	exitedCh chan struct{}
	closedCh chan struct{}
}

// Client connects to a remote jaz-exec-server over authenticated WebSocket
// JSON-RPC 2.0 and implements executor.Executor.
type Client struct {
	conn           *websocket.Conn
	workspaceDir   string
	sessionID      string
	platformOS     string
	maxOutputBytes int

	writeMu   sync.Mutex
	closeOnce sync.Once
	nextID    atomic.Int64

	pendingMu sync.Mutex
	pending   map[string]chan rpcFrame

	procMu sync.Mutex
	procs  map[string]*clientProcState

	done chan struct{}
}

var _ executor.Executor = (*Client)(nil)

// Dial connects to a jaz-exec-server WebSocket endpoint using a bearer
// capability token and performs the initialize handshake.
func Dial(ctx context.Context, wsURL, bearerToken, workspaceDir string) (*Client, error) {
	return DialWithConfig(ctx, DialConfig{
		URL:          wsURL,
		BearerToken:  bearerToken,
		WorkspaceDir: workspaceDir,
	})
}

// DialWithConfig connects to jaz-exec-server with explicit dial options.
func DialWithConfig(ctx context.Context, cfg DialConfig) (*Client, error) {
	if strings.TrimSpace(cfg.URL) == "" {
		return nil, errors.New("execserver URL is required")
	}
	handshakeTimeout := cfg.HandshakeTimeout
	if handshakeTimeout <= 0 {
		handshakeTimeout = 15 * time.Second
	}
	maxWS := cfg.MaxWSMessageBytes
	if maxWS <= 0 {
		maxWS = DefaultMaxWSMessageBytes
	}
	maxOut := cfg.MaxOutputBytes
	if maxOut <= 0 {
		maxOut = DefaultMaxOutputBytes
	}
	clientName := strings.TrimSpace(cfg.ClientName)
	if clientName == "" {
		clientName = "jaz-harness"
	}

	headers := http.Header{}
	if cfg.BearerToken != "" {
		headers.Set("Authorization", "Bearer "+cfg.BearerToken)
	}
	dialer := websocket.Dialer{
		HandshakeTimeout: handshakeTimeout,
	}
	conn, resp, err := dialer.DialContext(ctx, cfg.URL, headers)
	if err != nil {
		if resp != nil {
			if resp.Body != nil {
				_ = resp.Body.Close()
			}
			return nil, fmt.Errorf("dial jaz-exec-server %s (HTTP %d): %w", cfg.URL, resp.StatusCode, err)
		}
		return nil, fmt.Errorf("dial jaz-exec-server %s: %w", cfg.URL, err)
	}
	conn.SetReadLimit(maxWS)

	c := &Client{
		conn:           conn,
		workspaceDir:   strings.TrimSpace(cfg.WorkspaceDir),
		maxOutputBytes: maxOut,
		pending:        make(map[string]chan rpcFrame),
		procs:          make(map[string]*clientProcState),
		done:           make(chan struct{}),
	}
	go c.readLoop()

	var initRes initializeResult
	if err := c.call(ctx, MethodInitialize, initializeParams{
		ClientName: clientName,
	}, &initRes); err != nil {
		_ = c.Close()
		return nil, fmt.Errorf("jaz-exec-server initialize: %w", err)
	}
	c.sessionID = initRes.SessionID
	c.platformOS = strings.ToLower(strings.TrimSpace(initRes.PlatformOS))
	if c.workspaceDir == "" {
		c.workspaceDir = initRes.WorkspaceDir
	}
	return c, nil
}

// WorkspaceDir returns the workspace directory associated with this client.
func (c *Client) WorkspaceDir() string {
	return c.workspaceDir
}

// SessionID returns the server-assigned WebSocket session ID.
func (c *Client) SessionID() string {
	return c.sessionID
}

// Ping verifies round-trip liveness over the WebSocket JSON-RPC connection.
func (c *Client) Ping(ctx context.Context) error {
	var res pingResult
	if err := c.call(ctx, MethodPing, struct{}{}, &res); err != nil {
		return err
	}
	if !res.OK {
		return errors.New("jaz-exec-server ping returned ok=false")
	}
	return nil
}

// Close shuts down the WebSocket connection and waits for the reader loop to exit.
func (c *Client) Close() error {
	var closeErr error
	c.closeOnce.Do(func() {
		c.writeMu.Lock()
		closeErr = c.conn.Close()
		c.writeMu.Unlock()
	})
	<-c.done
	return closeErr
}

func (c *Client) readLoop() {
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

func (c *Client) handleNotification(method string, params json.RawMessage) {
	switch method {
	case NotifyProcessOutput:
		var n processOutputNotification
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

	case NotifyProcessExited:
		var n processExitedNotification
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

	case NotifyProcessClosed:
		var n processClosedNotification
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

func (c *Client) ensureProcLocked(id string) *clientProcState {
	st, ok := c.procs[id]
	if !ok {
		if len(c.procs) >= clientMaxRetainedProcs {
			for k, existing := range c.procs {
				if existing.closed || existing.exited {
					delete(c.procs, k)
				}
			}
		}
		st = &clientProcState{
			stdout:   executor.NewCappedBuffer(c.maxOutputBytes),
			stderr:   executor.NewCappedBuffer(c.maxOutputBytes),
			exitedCh: make(chan struct{}),
			closedCh: make(chan struct{}),
		}
		c.procs[id] = st
	}
	return st
}

func (c *Client) sendFrame(frame rpcFrame) error {
	frame.JSONRPC = JSONRPCVersion
	raw, err := json.Marshal(frame)
	if err != nil {
		return err
	}
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	_ = c.conn.SetWriteDeadline(time.Now().Add(wsWriteTimeout))
	return c.conn.WriteMessage(websocket.TextMessage, raw)
}

func (c *Client) call(ctx context.Context, method string, params any, out any) error {
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
	case <-c.done:
		c.pendingMu.Lock()
		delete(c.pending, key)
		c.pendingMu.Unlock()
		return errors.New("jaz-exec-server closed before response")
	case resp, ok := <-ch:
		if !ok {
			return errors.New("jaz-exec-server closed before response")
		}
		if resp.Error != nil {
			return &RPCError{
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

func (c *Client) effectiveCwd(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed != "" {
		return trimmed
	}
	return c.workspaceDir
}

// Exec runs a command to completion on the server with hard timeout enforcement
// and guaranteed process-group termination on timeout or cancellation.
func (c *Client) Exec(ctx context.Context, req executor.ExecRequest) (executor.ExecResult, error) {
	procID := "exec_" + uuid.NewString()[:8]
	var res processExecResult
	err := c.call(ctx, MethodProcessExec, processExecParams{
		ProcessID: procID,
		Command:   req.Command,
		Args:      req.Args,
		Shell:     req.Shell,
		Cwd:       c.effectiveCwd(req.Cwd),
		Env:       req.Env,
		TimeoutMS: req.TimeoutMS,
	}, &res)
	if err != nil {
		if ctx.Err() != nil {
			c.terminateBestEffort(procID)
		}
		return executor.ExecResult{}, err
	}
	out := executor.ExecResult{
		Stdout:   res.Stdout,
		Stderr:   res.Stderr,
		ExitCode: res.ExitCode,
		Running:  false,
	}
	if res.TimedOut {
		return out, fmt.Errorf("jaz-exec-server process/exec: %w", ErrTimedOut)
	}
	return out, nil
}

// ExecStart launches a managed process (supporting PipeStdin and live output
// streaming) and waits up to req.TimeoutMS (default 30s) for completion before
// returning Running=true with a ProcessID. Canceling ctx terminates the remote
// process group so no orphan processes leak.
func (c *Client) ExecStart(ctx context.Context, req executor.ExecRequest) (executor.ExecResult, error) {
	procID := "proc_" + uuid.NewString()[:8]
	c.procMu.Lock()
	st := c.ensureProcLocked(procID)
	c.procMu.Unlock()

	var startRes processStartResult
	if err := c.call(ctx, MethodProcessStart, processStartParams{
		ProcessID: procID,
		Command:   req.Command,
		Args:      req.Args,
		Shell:     req.Shell,
		Cwd:       c.effectiveCwd(req.Cwd),
		Env:       req.Env,
		PipeStdin: req.PipeStdin,
	}, &startRes); err != nil {
		c.procMu.Lock()
		delete(c.procs, procID)
		c.procMu.Unlock()
		return executor.ExecResult{}, err
	}
	if startRes.ProcessID != "" && startRes.ProcessID != procID {
		procID = startRes.ProcessID
		c.procMu.Lock()
		st = c.ensureProcLocked(procID)
		c.procMu.Unlock()
	}

	waitDur := DefaultStartWait
	if req.TimeoutMS > 0 {
		waitDur = time.Duration(req.TimeoutMS) * time.Millisecond
	}
	return c.waitProcess(ctx, procID, st, waitDur, true)
}

func (c *Client) waitProcess(ctx context.Context, procID string, st *clientProcState, waitDur time.Duration, terminateOnCancel bool) (executor.ExecResult, error) {
	timer := time.NewTimer(waitDur)
	defer timer.Stop()

	select {
	case <-st.closedCh:
	case <-st.exitedCh:
		grace := time.NewTimer(clientExitGraceWait)
		select {
		case <-st.closedCh:
		case <-grace.C:
		case <-timer.C:
		case <-ctx.Done():
			grace.Stop()
			return executor.ExecResult{}, ctx.Err()
		case <-c.done:
			grace.Stop()
			return executor.ExecResult{}, errors.New("jaz-exec-server closed while waiting for process")
		}
		grace.Stop()
	case <-timer.C:
	case <-ctx.Done():
		if terminateOnCancel {
			c.terminateBestEffort(procID)
		}
		return executor.ExecResult{}, ctx.Err()
	case <-c.done:
		return executor.ExecResult{}, errors.New("jaz-exec-server closed while waiting for process")
	}

	c.procMu.Lock()
	defer c.procMu.Unlock()
	return executor.ExecResult{
		ProcessID: procID,
		Stdout:    st.stdout.String(),
		Stderr:    st.stderr.String(),
		ExitCode:  st.exitCode,
		Running:   !st.exited,
	}, nil
}

func (c *Client) terminateBestEffort(procID string) {
	termCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = c.ExecTerminate(termCtx, procID)
}

// ExecWriteStdin writes data to a running process's stdin pipe and waits
// briefly for exit or additional streamed output.
func (c *Client) ExecWriteStdin(ctx context.Context, processID, data string) (executor.ExecResult, error) {
	if err := c.ExecWriteStdinWithOptions(ctx, processID, []byte(data), false); err != nil {
		return executor.ExecResult{}, err
	}
	c.procMu.Lock()
	st := c.ensureProcLocked(processID)
	c.procMu.Unlock()
	return c.waitProcess(ctx, processID, st, clientWriteStdinWait, false)
}

// ExecWriteStdinWithOptions writes raw bytes to a running process's stdin pipe
// and optionally closes stdin (sending EOF).
func (c *Client) ExecWriteStdinWithOptions(ctx context.Context, processID string, data []byte, closeStdin bool) error {
	var res processWriteResult
	return c.call(ctx, MethodProcessWrite, processWriteParams{
		ProcessID:  processID,
		Chunk:      base64.StdEncoding.EncodeToString(data),
		CloseStdin: closeStdin,
	}, &res)
}

// ExecRead reads retained output chunks after afterSeq, optionally long-polling
// up to waitMS milliseconds for new output or process exit.
func (c *Client) ExecRead(ctx context.Context, processID string, afterSeq uint64, maxBytes int, waitMS uint64) (executor.ExecReadResult, error) {
	var raw processReadResult
	if err := c.call(ctx, MethodProcessRead, processReadParams{
		ProcessID: processID,
		AfterSeq:  afterSeq,
		MaxBytes:  maxBytes,
		WaitMS:    waitMS,
	}, &raw); err != nil {
		return executor.ExecReadResult{}, err
	}
	chunks := make([]executor.ExecOutputChunk, 0, len(raw.Chunks))
	for _, ch := range raw.Chunks {
		decoded, err := base64.StdEncoding.DecodeString(ch.Chunk)
		if err != nil {
			decoded = []byte(ch.Chunk)
		}
		chunks = append(chunks, executor.ExecOutputChunk{
			Seq:    ch.Seq,
			Stream: ch.Stream,
			Data:   decoded,
		})
	}
	return executor.ExecReadResult{
		Chunks:   chunks,
		NextSeq:  raw.NextSeq,
		Exited:   raw.Exited,
		ExitCode: raw.ExitCode,
		Closed:   raw.Closed,
		Failure:  raw.Failure,
	}, nil
}

// ExecTerminate terminates a managed process and its entire process group.
func (c *Client) ExecTerminate(ctx context.Context, processID string) error {
	return c.call(ctx, MethodProcessTerminate, processTerminateParams{
		ProcessID: processID,
	}, nil)
}

// ReadFile reads a file from the sandbox within the server's bounded read limit.
func (c *Client) ReadFile(ctx context.Context, filePath string) ([]byte, error) {
	return c.ReadFileBounded(ctx, filePath, 0)
}

// ReadFileBounded reads a file with an explicit per-call maximum byte cap.
func (c *Client) ReadFileBounded(ctx context.Context, filePath string, maxBytes int) ([]byte, error) {
	var res fsReadFileResult
	if err := c.call(ctx, MethodFSReadFile, fsReadFileParams{
		Path:     filePath,
		MaxBytes: maxBytes,
	}, &res); err != nil {
		return nil, err
	}
	if res.DataBase64 == "" {
		return []byte{}, nil
	}
	return base64.StdEncoding.DecodeString(res.DataBase64)
}

// WriteFile writes data to filePath, automatically creating missing parent
// directories.
func (c *Client) WriteFile(ctx context.Context, filePath string, data []byte) error {
	createParents := true
	return c.call(ctx, MethodFSWriteFile, fsWriteFileParams{
		Path:          filePath,
		DataBase64:    base64.StdEncoding.EncodeToString(data),
		CreateParents: &createParents,
	}, nil)
}

// ListDir returns sorted directory entries for dirPath.
func (c *Client) ListDir(ctx context.Context, dirPath string) ([]executor.FileEntry, error) {
	var res fsListDirResult
	if err := c.call(ctx, MethodFSListDir, fsPathParams{
		Path: dirPath,
	}, &res); err != nil {
		return nil, err
	}
	return res.Entries, nil
}

// Stat returns file or directory metadata for targetPath.
func (c *Client) Stat(ctx context.Context, targetPath string) (executor.FileMetadata, error) {
	var res executor.FileMetadata
	if err := c.call(ctx, MethodFSStat, fsPathParams{
		Path: targetPath,
	}, &res); err != nil {
		return executor.FileMetadata{}, err
	}
	return res, nil
}

// MkdirAll creates dirPath and any missing parent directories.
func (c *Client) MkdirAll(ctx context.Context, dirPath string) error {
	return c.call(ctx, MethodFSMkdirAll, fsMkdirAllParams{
		Path: dirPath,
	}, nil)
}

// Remove deletes a file or directory (recursively when recursive is true).
func (c *Client) Remove(ctx context.Context, targetPath string, recursive bool) error {
	return c.call(ctx, MethodFSRemove, fsRemoveParams{
		Path:      targetPath,
		Recursive: recursive,
	}, nil)
}

// Rename moves oldPath to newPath, creating parent directories if needed.
func (c *Client) Rename(ctx context.Context, oldPath, newPath string) error {
	createParents := true
	return c.call(ctx, MethodFSRename, fsRenameParams{
		OldPath:       oldPath,
		NewPath:       newPath,
		CreateParents: &createParents,
	}, nil)
}
