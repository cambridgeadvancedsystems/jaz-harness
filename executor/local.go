package executor

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// LocalExecutor runs commands and rooted file operations on a local directory
// (used for standalone CLI mode, local development, and deterministic tests).
type LocalExecutor struct {
	root string
}

func NewLocal(workspaceDir string) (*LocalExecutor, error) {
	abs, err := filepath.Abs(workspaceDir)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return nil, err
	}
	return &LocalExecutor{root: abs}, nil
}

func (e *LocalExecutor) WorkspaceDir() string { return e.root }
func (e *LocalExecutor) Close() error         { return nil }

func (e *LocalExecutor) resolvePath(raw string) (string, error) {
	p := strings.TrimSpace(raw)
	if p == "" || p == "." {
		return e.root, nil
	}
	var candidate string
	if filepath.IsAbs(p) {
		candidate = filepath.Clean(p)
	} else {
		candidate = filepath.Clean(filepath.Join(e.root, p))
	}
	rel, err := filepath.Rel(e.root, candidate)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("path %q escapes workspace root", raw)
	}
	return candidate, nil
}

func (e *LocalExecutor) ExecStart(ctx context.Context, req ExecRequest) (ExecResult, error) {
	timeout := 60 * time.Second
	if req.TimeoutMS > 0 {
		timeout = time.Duration(req.TimeoutMS) * time.Millisecond
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var cmd *exec.Cmd
	if req.Shell || len(req.Args) == 0 {
		cmd = exec.CommandContext(runCtx, "/bin/sh", "-lc", req.Command)
	} else {
		cmd = exec.CommandContext(runCtx, req.Command, req.Args...)
	}
	cwd := e.root
	if strings.TrimSpace(req.Cwd) != "" {
		resolved, err := e.resolvePath(req.Cwd)
		if err != nil {
			return ExecResult{}, err
		}
		cwd = resolved
	}
	cmd.Dir = cwd
	if len(req.Env) > 0 {
		env := os.Environ()
		for k, v := range req.Env {
			env = append(env, k+"="+v)
		}
		cmd.Env = env
	}

	stdout := newCappedBuffer(DefaultMaxExecOutputBytes)
	stderr := newCappedBuffer(DefaultMaxExecOutputBytes)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	err := cmd.Run()
	exitCode := 0
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			exitCode = exitErr.ExitCode()
		} else {
			return ExecResult{}, err
		}
	}
	return ExecResult{
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		ExitCode: exitCode,
	}, nil
}

func (e *LocalExecutor) ExecWriteStdin(_ context.Context, _, _ string) (ExecResult, error) {
	return ExecResult{}, errors.New("local executor does not keep interactive PTY sessions open")
}

func (e *LocalExecutor) ExecTerminate(_ context.Context, _ string) error {
	return nil
}

func (e *LocalExecutor) ReadFile(_ context.Context, path string) ([]byte, error) {
	resolved, err := e.resolvePath(path)
	if err != nil {
		return nil, err
	}
	return os.ReadFile(resolved)
}

func (e *LocalExecutor) WriteFile(_ context.Context, path string, data []byte) error {
	resolved, err := e.resolvePath(path)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(resolved), 0o755); err != nil {
		return err
	}
	return os.WriteFile(resolved, data, 0o644)
}

func (e *LocalExecutor) ListDir(_ context.Context, path string) ([]FileEntry, error) {
	resolved, err := e.resolvePath(path)
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(resolved)
	if err != nil {
		return nil, err
	}
	out := make([]FileEntry, 0, len(entries))
	for _, entry := range entries {
		info, _ := entry.Info()
		var size int64
		if info != nil {
			size = info.Size()
		}
		rel, _ := filepath.Rel(e.root, filepath.Join(resolved, entry.Name()))
		out = append(out, FileEntry{
			Name:  entry.Name(),
			Path:  filepath.ToSlash(rel),
			IsDir: entry.IsDir(),
			Size:  size,
		})
	}
	return out, nil
}
