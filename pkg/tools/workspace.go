package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/executor"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
)

const maxToolOutputBytes = 32 * 1024

// LazyExecutorFunc is invoked ONLY when the model actually calls a workspace
// tool, keeping hosted-only turns at 0 sandbox pods with 0 schema mutation.
type LazyExecutorFunc func(ctx context.Context) (executor.Executor, error)

type WorkspaceToolset struct {
	acquire    LazyExecutorFunc
	mu         sync.Mutex
	exec       executor.Executor
	loadedRule bool
}

func NewWorkspaceToolset(acquire LazyExecutorFunc) *WorkspaceToolset {
	return &WorkspaceToolset{acquire: acquire}
}

func (w *WorkspaceToolset) ensure(ctx context.Context) (executor.Executor, string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.exec == nil {
		if w.acquire == nil {
			return nil, "", errors.New("workspace sandbox is unavailable")
		}
		ex, err := w.acquire(ctx)
		if err != nil {
			return nil, "", err
		}
		w.exec = ex
	}
	var agentsPrefix string
	if !w.loadedRule {
		w.loadedRule = true
		if raw, err := w.exec.ReadFile(ctx, "AGENTS.md"); err == nil && strings.TrimSpace(string(raw)) != "" {
			agentsPrefix = "<workspace_agents_md>\n" + truncateOutput(string(raw), 8192) + "\n</workspace_agents_md>\n\n"
		}
	}
	return w.exec, agentsPrefix, nil
}

func (w *WorkspaceToolset) Attached() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.exec != nil
}

func (w *WorkspaceToolset) Close() error {
	w.mu.Lock()
	ex := w.exec
	w.exec = nil
	w.mu.Unlock()
	if ex != nil {
		return ex.Close()
	}
	return nil
}

// RegisterInto declares the static workspace tools (`exec_command`, `read_file`,
// `write_file`, `edit_file`, `list_dir`) so tool definitions never mutate mid-turn.
func (w *WorkspaceToolset) RegisterInto(reg *Registry) {
	reg.Register(schema.ToolDef{
		Name:        "exec_command",
		Description: "Run a shell command inside the isolated workspace sandbox (/workspace). Lazily attaches the sandbox on first call.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"},"cwd":{"type":"string"},"timeout_ms":{"type":"integer"}},"required":["command"]}`),
		AlwaysLoad:  true,
		Category:    "workspace",
	}, func(ctx context.Context, args json.RawMessage) (string, error) {
		var p struct {
			Command   string `json:"command"`
			Cwd       string `json:"cwd"`
			TimeoutMS int64  `json:"timeout_ms"`
		}
		if err := json.Unmarshal(args, &p); err != nil {
			return "", err
		}
		ex, prefix, err := w.ensure(ctx)
		if err != nil {
			return "", err
		}
		res, err := ex.ExecStart(ctx, executor.ExecRequest{
			Command:   p.Command,
			Shell:     true,
			Cwd:       p.Cwd,
			TimeoutMS: p.TimeoutMS,
		})
		if err != nil {
			return "", err
		}
		var b strings.Builder
		b.WriteString(prefix)
		fmt.Fprintf(&b, "exit_code: %d\n", res.ExitCode)
		if res.Stdout != "" {
			b.WriteString("stdout:\n")
			b.WriteString(truncateOutput(res.Stdout, maxToolOutputBytes))
			b.WriteByte('\n')
		}
		if res.Stderr != "" {
			b.WriteString("stderr:\n")
			b.WriteString(truncateOutput(res.Stderr, maxToolOutputBytes/2))
			b.WriteByte('\n')
		}
		return b.String(), nil
	})

	reg.Register(schema.ToolDef{
		Name:        "read_file",
		Description: "Read a UTF-8 file from the workspace sandbox.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
		AlwaysLoad:  true,
		Category:    "workspace",
	}, func(ctx context.Context, args json.RawMessage) (string, error) {
		var p struct {
			Path string `json:"path"`
		}
		if err := json.Unmarshal(args, &p); err != nil {
			return "", err
		}
		ex, prefix, err := w.ensure(ctx)
		if err != nil {
			return "", err
		}
		data, err := ex.ReadFile(ctx, p.Path)
		if err != nil {
			return "", err
		}
		return prefix + truncateOutput(string(data), maxToolOutputBytes), nil
	})

	reg.Register(schema.ToolDef{
		Name:        "write_file",
		Description: "Create or overwrite a file inside the workspace sandbox.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}`),
		AlwaysLoad:  true,
		Category:    "workspace",
	}, func(ctx context.Context, args json.RawMessage) (string, error) {
		var p struct {
			Path    string `json:"path"`
			Content string `json:"content"`
		}
		if err := json.Unmarshal(args, &p); err != nil {
			return "", err
		}
		ex, prefix, err := w.ensure(ctx)
		if err != nil {
			return "", err
		}
		if err := ex.WriteFile(ctx, p.Path, []byte(p.Content)); err != nil {
			return "", err
		}
		return fmt.Sprintf("%swrote %d bytes to %s", prefix, len(p.Content), p.Path), nil
	})

	reg.Register(schema.ToolDef{
		Name:        "edit_file",
		Description: "Replace an exact target string inside a workspace file.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"}},"required":["path","old_text","new_text"]}`),
		AlwaysLoad:  true,
		Category:    "workspace",
	}, func(ctx context.Context, args json.RawMessage) (string, error) {
		var p struct {
			Path    string `json:"path"`
			OldText string `json:"old_text"`
			NewText string `json:"new_text"`
		}
		if err := json.Unmarshal(args, &p); err != nil {
			return "", err
		}
		ex, prefix, err := w.ensure(ctx)
		if err != nil {
			return "", err
		}
		raw, err := ex.ReadFile(ctx, p.Path)
		if err != nil {
			return "", err
		}
		text := string(raw)
		if !strings.Contains(text, p.OldText) {
			return "", fmt.Errorf("old_text not found in %s", p.Path)
		}
		updated := strings.Replace(text, p.OldText, p.NewText, 1)
		if err := ex.WriteFile(ctx, p.Path, []byte(updated)); err != nil {
			return "", err
		}
		return fmt.Sprintf("%sedited %s", prefix, p.Path), nil
	})

	reg.Register(schema.ToolDef{
		Name:        "list_dir",
		Description: "List files and directories inside the workspace sandbox.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`),
		AlwaysLoad:  true,
		Category:    "workspace",
	}, func(ctx context.Context, args json.RawMessage) (string, error) {
		var p struct {
			Path string `json:"path"`
		}
		_ = json.Unmarshal(args, &p)
		ex, prefix, err := w.ensure(ctx)
		if err != nil {
			return "", err
		}
		entries, err := ex.ListDir(ctx, p.Path)
		if err != nil {
			return "", err
		}
		raw, _ := json.Marshal(entries)
		return prefix + string(raw), nil
	})
}

func truncateOutput(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	head := limit * 2 / 3
	tail := limit - head
	return s[:head] + fmt.Sprintf("\n... [truncated %d bytes] ...\n", len(s)-limit) + s[len(s)-tail:]
}
