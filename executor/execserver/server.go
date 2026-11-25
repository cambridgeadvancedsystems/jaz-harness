package execserver

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

const (
	wsWriteTimeout = 15 * time.Second
)

// Config configures a pure-Go Jaz sandbox exec server.
type Config struct {
	WorkspaceDir         string
	RestrictToWorkspace  bool
	Token                string
	TokenSHA256          string
	AllowUnauthenticated bool
	MaxWSMessageBytes    int64
	MaxReadFileBytes     int
	MaxWriteFileBytes    int
	MaxOutputBytes       int
	MaxProcesses         int
	KillGracePeriod      time.Duration
	KillOnDisconnect     *bool
	CheckOrigin          func(*http.Request) bool
}

// Server exposes filesystem and process-lifecycle operations over authenticated
// WebSocket JSON-RPC 2.0 alongside an unauthenticated /healthz probe.
type Server struct {
	auth             tokenAuthenticator
	fs               *fsService
	procs            *processManager
	upgrader         websocket.Upgrader
	maxWSMessage     int64
	killOnDisconnect bool

	mux *http.ServeMux

	sessMu   sync.Mutex
	sessions map[string]*wsSession
	sessWG   sync.WaitGroup
	closed   bool
}

// New constructs a validated Server. By default, authentication fails closed
// unless Token or TokenSHA256 is provided, or AllowUnauthenticated is true.
func New(cfg Config) (*Server, error) {
	auth, err := newTokenAuthenticator(cfg.Token, cfg.TokenSHA256, cfg.AllowUnauthenticated)
	if err != nil {
		return nil, err
	}
	fsSvc, err := newFSService(cfg.WorkspaceDir, cfg.RestrictToWorkspace, cfg.MaxReadFileBytes, cfg.MaxWriteFileBytes)
	if err != nil {
		return nil, err
	}
	maxWS := cfg.MaxWSMessageBytes
	if maxWS <= 0 {
		maxWS = DefaultMaxWSMessageBytes
	}
	killOnDisconnect := true
	if cfg.KillOnDisconnect != nil {
		killOnDisconnect = *cfg.KillOnDisconnect
	}
	checkOrigin := cfg.CheckOrigin
	if checkOrigin == nil {
		checkOrigin = func(*http.Request) bool { return true }
	}

	s := &Server{
		auth:             auth,
		fs:               fsSvc,
		procs:            newProcessManager(fsSvc, cfg.MaxOutputBytes, cfg.MaxProcesses, cfg.KillGracePeriod),
		maxWSMessage:     maxWS,
		killOnDisconnect: killOnDisconnect,
		upgrader: websocket.Upgrader{
			ReadBufferSize:  32 * 1024,
			WriteBufferSize: 32 * 1024,
			CheckOrigin:     checkOrigin,
		},
		mux:      http.NewServeMux(),
		sessions: make(map[string]*wsSession),
	}
	s.mux.HandleFunc("/healthz", s.handleHealthz)
	s.mux.HandleFunc("/ws", s.handleWS)
	s.mux.HandleFunc("/", s.handleRoot)
	return s, nil
}

// Handler returns the HTTP handler serving /healthz and the WebSocket JSON-RPC
// endpoint on / and /ws.
func (s *Server) Handler() http.Handler {
	return s.mux
}

// WorkspaceDir returns the resolved absolute workspace directory.
func (s *Server) WorkspaceDir() string {
	return s.fs.workspaceDir
}

// ActiveProcesses returns the number of currently running processes.
func (s *Server) ActiveProcesses() int {
	return s.procs.activeCount()
}

// Close terminates all active process groups, closes active WebSocket
// sessions, and waits for session handlers to finish unwinding.
func (s *Server) Close() error {
	s.sessMu.Lock()
	s.closed = true
	sessions := make([]*wsSession, 0, len(s.sessions))
	for _, sess := range s.sessions {
		sessions = append(sessions, sess)
	}
	s.sessMu.Unlock()

	for _, sess := range sessions {
		sess.close()
	}
	s.procs.terminateAll()
	s.sessWG.Wait()
	return nil
}

func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		writeHTTPJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method not allowed"})
		return
	}
	writeHTTPJSON(w, http.StatusOK, map[string]any{
		"ok":               true,
		"protocol_version": ProtocolVersion,
		"workspace_dir":    s.fs.workspaceDir,
	})
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		writeHTTPJSON(w, http.StatusNotFound, map[string]any{"error": "not found"})
		return
	}
	s.handleWS(w, r)
}

func (s *Server) handleWS(w http.ResponseWriter, r *http.Request) {
	if !s.auth.authorizeRequest(r) {
		w.Header().Set("WWW-Authenticate", "Bearer")
		writeHTTPJSON(w, http.StatusUnauthorized, map[string]any{"error": "unauthorized"})
		return
	}
	s.sessMu.Lock()
	isClosed := s.closed
	s.sessMu.Unlock()
	if isClosed {
		writeHTTPJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "server is shutting down"})
		return
	}

	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	conn.SetReadLimit(s.maxWSMessage)

	ctx, cancel := context.WithCancel(context.Background())
	sess := &wsSession{
		id:       "sess_" + uuid.NewString()[:8],
		server:   s,
		conn:     conn,
		ctx:      ctx,
		cancel:   cancel,
		inFlight: make(chan struct{}, DefaultMaxInFlightRequests),
	}

	s.sessMu.Lock()
	if s.closed {
		s.sessMu.Unlock()
		cancel()
		_ = conn.Close()
		return
	}
	s.sessions[sess.id] = sess
	s.sessWG.Add(1)
	s.sessMu.Unlock()

	defer func() {
		defer s.sessWG.Done()
		sess.close()
		if s.killOnDisconnect {
			s.procs.terminateSession(sess.id)
		}
		sess.wg.Wait()
		s.sessMu.Lock()
		delete(s.sessions, sess.id)
		s.sessMu.Unlock()
	}()

	sess.readLoop()
}

type wsSession struct {
	id       string
	server   *Server
	conn     *websocket.Conn
	ctx      context.Context
	cancel   context.CancelFunc
	inFlight chan struct{}

	writeMu   sync.Mutex
	closeOnce sync.Once
	wg        sync.WaitGroup
}

func (sess *wsSession) close() {
	sess.closeOnce.Do(func() {
		sess.cancel()
		sess.writeMu.Lock()
		_ = sess.conn.Close()
		sess.writeMu.Unlock()
	})
}

func (sess *wsSession) sendFrame(frame rpcFrame) error {
	if sess.ctx.Err() != nil {
		return sess.ctx.Err()
	}
	frame.JSONRPC = JSONRPCVersion
	raw, err := json.Marshal(frame)
	if err != nil {
		return err
	}
	sess.writeMu.Lock()
	defer sess.writeMu.Unlock()
	_ = sess.conn.SetWriteDeadline(time.Now().Add(wsWriteTimeout))
	return sess.conn.WriteMessage(websocket.TextMessage, raw)
}

func (sess *wsSession) sendNotify(method string, params any) {
	raw, err := json.Marshal(params)
	if err != nil {
		return
	}
	_ = sess.sendFrame(rpcFrame{
		Method: method,
		Params: raw,
	})
}

func (sess *wsSession) sendResult(id json.RawMessage, result any) {
	if result == nil {
		result = map[string]any{}
	}
	raw, err := json.Marshal(result)
	if err != nil {
		sess.sendError(id, &rpcErrorBody{
			Code:    ErrCodeInternal,
			Message: err.Error(),
		})
		return
	}
	_ = sess.sendFrame(rpcFrame{
		ID:     id,
		Result: raw,
	})
}

func (sess *wsSession) sendError(id json.RawMessage, errBody *rpcErrorBody) {
	_ = sess.sendFrame(rpcFrame{
		ID:    id,
		Error: errBody,
	})
}

func (sess *wsSession) readLoop() {
	for {
		msgType, data, err := sess.conn.ReadMessage()
		if err != nil {
			return
		}
		if msgType != websocket.TextMessage && msgType != websocket.BinaryMessage {
			continue
		}
		var frame rpcFrame
		if err := json.Unmarshal(data, &frame); err != nil {
			sess.sendError(nil, &rpcErrorBody{
				Code:    ErrCodeParse,
				Message: "invalid JSON frame",
			})
			continue
		}
		if strings.TrimSpace(frame.Method) == "" {
			if len(frame.ID) > 0 {
				sess.sendError(frame.ID, &rpcErrorBody{
					Code:    ErrCodeInvalidRequest,
					Message: "method is required",
				})
			}
			continue
		}
		if len(frame.ID) == 0 {
			// Notification from client (no response required).
			continue
		}

		select {
		case sess.inFlight <- struct{}{}:
		case <-sess.ctx.Done():
			return
		}
		sess.wg.Add(1)
		go func(req rpcFrame) {
			defer func() {
				<-sess.inFlight
				sess.wg.Done()
			}()
			sess.dispatch(req)
		}(frame)
	}
}

func (sess *wsSession) dispatch(req rpcFrame) {
	result, errBody := sess.handleMethod(req.Method, req.Params)
	if errBody != nil {
		sess.sendError(req.ID, errBody)
		return
	}
	sess.sendResult(req.ID, result)
}

func (sess *wsSession) handleMethod(method string, params json.RawMessage) (any, *rpcErrorBody) {
	switch method {
	case MethodInitialize:
		var p initializeParams
		if len(params) > 0 {
			if err := json.Unmarshal(params, &p); err != nil {
				return nil, invalidParamsErr(err)
			}
		}
		return initializeResult{
			SessionID:       sess.id,
			ProtocolVersion: ProtocolVersion,
			WorkspaceDir:    sess.server.fs.workspaceDir,
			PlatformOS:      runtime.GOOS,
			Arch:            runtime.GOARCH,
		}, nil

	case MethodPing:
		return pingResult{OK: true}, nil

	case MethodFSReadFile:
		var p fsReadFileParams
		if err := decodeParams(params, &p); err != nil {
			return nil, invalidParamsErr(err)
		}
		data, errBody := sess.server.fs.readFile(p.Path, p.MaxBytes)
		if errBody != nil {
			return nil, errBody
		}
		return fsReadFileResult{
			DataBase64: base64.StdEncoding.EncodeToString(data),
			Size:       int64(len(data)),
		}, nil

	case MethodFSWriteFile:
		var p fsWriteFileParams
		if err := decodeParams(params, &p); err != nil {
			return nil, invalidParamsErr(err)
		}
		if base64.StdEncoding.DecodedLen(len(p.DataBase64)) > sess.server.fs.maxWriteBytes+3 {
			return nil, &rpcErrorBody{
				Code:    ErrCodeTooLarge,
				Message: "write payload exceeds maximum write limit",
			}
		}
		data, err := base64.StdEncoding.DecodeString(p.DataBase64)
		if err != nil {
			return nil, &rpcErrorBody{
				Code:    ErrCodeInvalidParams,
				Message: "invalid base64 data: " + err.Error(),
			}
		}
		createParents := true
		if p.CreateParents != nil {
			createParents = *p.CreateParents
		}
		if errBody := sess.server.fs.writeFile(p.Path, data, p.Mode, createParents); errBody != nil {
			return nil, errBody
		}
		return fsWriteFileResult{BytesWritten: len(data)}, nil

	case MethodFSListDir:
		var p fsPathParams
		if len(params) > 0 {
			if err := json.Unmarshal(params, &p); err != nil {
				return nil, invalidParamsErr(err)
			}
		}
		entries, errBody := sess.server.fs.listDir(p.Path)
		if errBody != nil {
			return nil, errBody
		}
		return fsListDirResult{Entries: entries}, nil

	case MethodFSStat:
		var p fsPathParams
		if err := decodeParams(params, &p); err != nil {
			return nil, invalidParamsErr(err)
		}
		meta, errBody := sess.server.fs.stat(p.Path)
		if errBody != nil {
			return nil, errBody
		}
		return meta, nil

	case MethodFSMkdirAll:
		var p fsMkdirAllParams
		if err := decodeParams(params, &p); err != nil {
			return nil, invalidParamsErr(err)
		}
		if errBody := sess.server.fs.mkdirAll(p.Path, p.Mode); errBody != nil {
			return nil, errBody
		}
		return map[string]any{}, nil

	case MethodFSRemove:
		var p fsRemoveParams
		if err := decodeParams(params, &p); err != nil {
			return nil, invalidParamsErr(err)
		}
		if errBody := sess.server.fs.remove(p.Path, p.Recursive); errBody != nil {
			return nil, errBody
		}
		return map[string]any{}, nil

	case MethodFSRename:
		var p fsRenameParams
		if err := decodeParams(params, &p); err != nil {
			return nil, invalidParamsErr(err)
		}
		createParents := true
		if p.CreateParents != nil {
			createParents = *p.CreateParents
		}
		if errBody := sess.server.fs.rename(p.OldPath, p.NewPath, createParents); errBody != nil {
			return nil, errBody
		}
		return map[string]any{}, nil

	case MethodProcessExec:
		var p processExecParams
		if err := decodeParams(params, &p); err != nil {
			return nil, invalidParamsErr(err)
		}
		res, errBody := sess.server.procs.execSync(sess.ctx, sess.id, p)
		if errBody != nil {
			return nil, errBody
		}
		return res, nil

	case MethodProcessStart:
		var p processStartParams
		if err := decodeParams(params, &p); err != nil {
			return nil, invalidParamsErr(err)
		}
		procID, errBody := sess.server.procs.start(sess.id, p, sess.sendNotify)
		if errBody != nil {
			return nil, errBody
		}
		return processStartResult{ProcessID: procID}, nil

	case MethodProcessWrite:
		var p processWriteParams
		if err := decodeParams(params, &p); err != nil {
			return nil, invalidParamsErr(err)
		}
		if base64.StdEncoding.DecodedLen(len(p.Chunk)) > sess.server.procs.maxOutputBytes+3 {
			return nil, &rpcErrorBody{
				Code:    ErrCodeTooLarge,
				Message: "stdin chunk exceeds maximum payload size",
			}
		}
		proc, errBody := sess.server.procs.get(sess.id, p.ProcessID)
		if errBody != nil {
			return nil, errBody
		}
		data, err := base64.StdEncoding.DecodeString(p.Chunk)
		if err != nil {
			return nil, &rpcErrorBody{
				Code:    ErrCodeInvalidParams,
				Message: "invalid base64 chunk: " + err.Error(),
			}
		}
		n, errBody := proc.writeStdin(data, p.CloseStdin)
		if errBody != nil {
			return nil, errBody
		}
		return processWriteResult{BytesWritten: n}, nil

	case MethodProcessRead:
		var p processReadParams
		if err := decodeParams(params, &p); err != nil {
			return nil, invalidParamsErr(err)
		}
		proc, errBody := sess.server.procs.get(sess.id, p.ProcessID)
		if errBody != nil {
			return nil, errBody
		}
		res, err := proc.readChunks(sess.ctx, p.AfterSeq, p.MaxBytes, p.WaitMS)
		if err != nil {
			return nil, &rpcErrorBody{
				Code:    ErrCodeInternal,
				Message: err.Error(),
			}
		}
		return res, nil

	case MethodProcessTerminate:
		var p processTerminateParams
		if err := decodeParams(params, &p); err != nil {
			return nil, invalidParamsErr(err)
		}
		if errBody := sess.server.procs.terminate(sess.id, p.ProcessID); errBody != nil {
			return nil, errBody
		}
		return map[string]any{}, nil

	default:
		return nil, &rpcErrorBody{
			Code:    ErrCodeMethodNotFound,
			Message: "unknown method: " + method,
		}
	}
}

func decodeParams(raw json.RawMessage, out any) error {
	if len(raw) == 0 {
		return errors.New("params are required")
	}
	return json.Unmarshal(raw, out)
}

func invalidParamsErr(err error) *rpcErrorBody {
	return &rpcErrorBody{
		Code:    ErrCodeInvalidParams,
		Message: err.Error(),
	}
}

func writeHTTPJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
