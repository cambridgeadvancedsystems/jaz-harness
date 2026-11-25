package execserver

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/executor"
)

const (
	ProtocolVersion = "1"
	JSONRPCVersion  = "2.0"

	DefaultMaxWSMessageBytes   = 64 << 20
	DefaultMaxReadFileBytes    = 32 << 20
	DefaultMaxWriteFileBytes   = 32 << 20
	DefaultMaxOutputBytes      = executor.DefaultMaxExecOutputBytes
	DefaultMaxProcesses        = 64
	DefaultMaxInFlightRequests = 64
	DefaultExecTimeout         = 60 * time.Second
	DefaultStartWait           = 30 * time.Second
	DefaultKillGracePeriod     = 500 * time.Millisecond
)

const (
	ErrCodeParse          = -32700
	ErrCodeInvalidRequest = -32600
	ErrCodeMethodNotFound = -32601
	ErrCodeInvalidParams  = -32602
	ErrCodeInternal       = -32603

	ErrCodeNotFound      = -32004
	ErrCodePermission    = -32005
	ErrCodeTooLarge      = -32006
	ErrCodePathEscapes   = -32007
	ErrCodeProcessLimit  = -32008
	ErrCodeAlreadyExists = -32009
	ErrCodeTimeout       = -32010
)

const (
	MethodInitialize       = "initialize"
	MethodPing             = "ping"
	MethodFSReadFile       = "fs/readFile"
	MethodFSWriteFile      = "fs/writeFile"
	MethodFSListDir        = "fs/listDir"
	MethodFSStat           = "fs/stat"
	MethodFSMkdirAll       = "fs/mkdirAll"
	MethodFSRemove         = "fs/remove"
	MethodFSRename         = "fs/rename"
	MethodProcessExec      = "process/exec"
	MethodProcessStart     = "process/start"
	MethodProcessWrite     = "process/write"
	MethodProcessRead      = "process/read"
	MethodProcessTerminate = "process/terminate"

	NotifyProcessOutput = "process/output"
	NotifyProcessExited = "process/exited"
	NotifyProcessClosed = "process/closed"
)

var (
	ErrNotFound     = errors.New("not found")
	ErrPathEscapes  = errors.New("path escapes workspace root")
	ErrFileTooLarge = errors.New("file exceeds maximum allowed size")
	ErrTimedOut     = errors.New("process timed out")
)

type rpcErrorBody struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type rpcFrame struct {
	JSONRPC string          `json:"jsonrpc,omitempty"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcErrorBody   `json:"error,omitempty"`
}

// RPCError represents a JSON-RPC error returned by jaz-exec-server.
type RPCError struct {
	Method  string
	Code    int
	Message string
}

func (e *RPCError) Error() string {
	if e.Method != "" {
		return fmt.Sprintf("jaz-exec-server %s error %d: %s", e.Method, e.Code, e.Message)
	}
	return fmt.Sprintf("jaz-exec-server error %d: %s", e.Code, e.Message)
}

func (e *RPCError) Unwrap() error {
	switch e.Code {
	case ErrCodeNotFound:
		return ErrNotFound
	case ErrCodePathEscapes:
		return ErrPathEscapes
	case ErrCodeTooLarge:
		return ErrFileTooLarge
	case ErrCodeTimeout:
		return ErrTimedOut
	default:
		return nil
	}
}

type initializeParams struct {
	ClientName string `json:"clientName,omitempty"`
}

type initializeResult struct {
	SessionID       string `json:"sessionId"`
	ProtocolVersion string `json:"protocolVersion"`
	WorkspaceDir    string `json:"workspaceDir"`
	PlatformOS      string `json:"platformOs"`
	Arch            string `json:"arch"`
}

type pingResult struct {
	OK bool `json:"ok"`
}

type fsReadFileParams struct {
	Path     string `json:"path"`
	MaxBytes int    `json:"maxBytes,omitempty"`
}

type fsReadFileResult struct {
	DataBase64 string `json:"dataBase64"`
	Size       int64  `json:"size"`
}

type fsWriteFileParams struct {
	Path          string `json:"path"`
	DataBase64    string `json:"dataBase64"`
	Mode          uint32 `json:"mode,omitempty"`
	CreateParents *bool  `json:"createParents,omitempty"`
}

type fsWriteFileResult struct {
	BytesWritten int `json:"bytesWritten"`
}

type fsPathParams struct {
	Path string `json:"path"`
}

type fsListDirResult struct {
	Entries []executor.FileEntry `json:"entries"`
}

type fsMkdirAllParams struct {
	Path string `json:"path"`
	Mode uint32 `json:"mode,omitempty"`
}

type fsRemoveParams struct {
	Path      string `json:"path"`
	Recursive bool   `json:"recursive,omitempty"`
}

type fsRenameParams struct {
	OldPath       string `json:"oldPath"`
	NewPath       string `json:"newPath"`
	CreateParents *bool  `json:"createParents,omitempty"`
}

type processExecParams struct {
	ProcessID string            `json:"processId,omitempty"`
	Command   string            `json:"command"`
	Args      []string          `json:"args,omitempty"`
	Shell     bool              `json:"shell,omitempty"`
	Cwd       string            `json:"cwd,omitempty"`
	Env       map[string]string `json:"env,omitempty"`
	TimeoutMS int64             `json:"timeoutMs,omitempty"`
}

type processExecResult struct {
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
	ExitCode int    `json:"exitCode"`
	TimedOut bool   `json:"timedOut,omitempty"`
}

type processStartParams struct {
	ProcessID string            `json:"processId,omitempty"`
	Command   string            `json:"command"`
	Args      []string          `json:"args,omitempty"`
	Shell     bool              `json:"shell,omitempty"`
	Cwd       string            `json:"cwd,omitempty"`
	Env       map[string]string `json:"env,omitempty"`
	PipeStdin bool              `json:"pipeStdin,omitempty"`
}

type processStartResult struct {
	ProcessID string `json:"processId"`
}

type processWriteParams struct {
	ProcessID  string `json:"processId"`
	Chunk      string `json:"chunk"`
	CloseStdin bool   `json:"closeStdin,omitempty"`
}

type processWriteResult struct {
	BytesWritten int `json:"bytesWritten"`
}

type processReadParams struct {
	ProcessID string `json:"processId"`
	AfterSeq  uint64 `json:"afterSeq,omitempty"`
	MaxBytes  int    `json:"maxBytes,omitempty"`
	WaitMS    uint64 `json:"waitMs,omitempty"`
}

type wireOutputChunk struct {
	Seq    uint64 `json:"seq"`
	Stream string `json:"stream"`
	Chunk  string `json:"chunk"`
}

type processReadResult struct {
	Chunks   []wireOutputChunk `json:"chunks"`
	NextSeq  uint64            `json:"nextSeq"`
	Exited   bool              `json:"exited"`
	ExitCode *int              `json:"exitCode,omitempty"`
	Closed   bool              `json:"closed"`
	Failure  string            `json:"failure,omitempty"`
}

type processTerminateParams struct {
	ProcessID string `json:"processId"`
}

type processOutputNotification struct {
	ProcessID string `json:"processId"`
	Seq       uint64 `json:"seq"`
	Stream    string `json:"stream"`
	Chunk     string `json:"chunk"`
}

type processExitedNotification struct {
	ProcessID string `json:"processId"`
	ExitCode  int    `json:"exitCode"`
}

type processClosedNotification struct {
	ProcessID string `json:"processId"`
}
