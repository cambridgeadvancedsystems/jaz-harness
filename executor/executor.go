package executor

import (
	"context"
	"io"
)

type ExecRequest struct {
	Command   string
	Args      []string
	Shell     bool
	Cwd       string
	Env       map[string]string
	TimeoutMS int64
	PipeStdin bool
}

type ExecResult struct {
	ProcessID string
	Stdout    string
	Stderr    string
	ExitCode  int
	Running   bool
}

type FileEntry struct {
	Name  string `json:"name"`
	Path  string `json:"path"`
	IsDir bool   `json:"is_dir"`
	Size  int64  `json:"size"`
}

// Executor abstracts remote gVisor `codex exec-server` pods, paired-device
// executors, and local dev sandboxes behind one uniform 0-pod-compatible API.
type Executor interface {
	WorkspaceDir() string
	ExecStart(ctx context.Context, req ExecRequest) (ExecResult, error)
	ExecWriteStdin(ctx context.Context, processID, data string) (ExecResult, error)
	ExecTerminate(ctx context.Context, processID string) error
	ReadFile(ctx context.Context, path string) ([]byte, error)
	WriteFile(ctx context.Context, path string, data []byte) error
	ListDir(ctx context.Context, path string) ([]FileEntry, error)
	io.Closer
}
