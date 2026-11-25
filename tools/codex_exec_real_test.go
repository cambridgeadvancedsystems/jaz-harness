package tools_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/executor"
	"github.com/cambridgeadvancedsystems/jaz-harness/skills"
	"github.com/cambridgeadvancedsystems/jaz-harness/tools"
)

func reserveLoopbackAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve loopback port: %v", err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()
	return addr
}

func startRealCodexExecServer(t *testing.T, codexBin, token string) (string, func()) {
	t.Helper()
	addr := reserveLoopbackAddr(t)
	sum := sha256.Sum256([]byte(token))
	tokenHash := hex.EncodeToString(sum[:])

	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(
		ctx,
		codexBin,
		"exec-server",
		"--ws-auth=capability-token",
		"--ws-token-sha256="+tokenHash,
		"--listen=ws://"+addr,
	)
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		cancel()
		t.Fatalf("start codex exec-server: %v", err)
	}

	wsURL := "ws://" + addr
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 150*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			return wsURL, func() {
				cancel()
				_ = cmd.Wait()
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	cancel()
	_ = cmd.Wait()
	t.Fatalf("codex exec-server failed to listen on %s within 10s", addr)
	return "", func() {}
}

func TestWorkspaceToolsAgainstRealCodexExecServer(t *testing.T) {
	codexBin, err := exec.LookPath("codex")
	if err != nil {
		t.Skip("codex binary not installed in PATH")
	}

	const bearerToken = "jaz-harness-real-exec-server-test-token"
	wsURL, stopServer := startRealCodexExecServer(t, codexBin, bearerToken)
	defer stopServer()

	wsDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(wsDir, "AGENTS.md"), []byte("REAL_EXEC_AGENTS_RULE"), 0o644); err != nil {
		t.Fatalf("seed AGENTS.md: %v", err)
	}
	if err := os.WriteFile(filepath.Join(wsDir, "seed.txt"), []byte("seed-content"), 0o644); err != nil {
		t.Fatalf("seed seed.txt: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Verify capability-token auth fails closed with the wrong token.
	if badCli, badErr := executor.DialCodexExecServer(ctx, wsURL, "wrong-token", wsDir); badErr == nil {
		_ = badCli.Close()
		t.Fatal("expected DialCodexExecServer to reject invalid capability token")
	}

	cli, err := executor.DialCodexExecServer(ctx, wsURL, bearerToken, wsDir)
	if err != nil {
		t.Fatalf("DialCodexExecServer against real codex exec-server: %v", err)
	}
	defer cli.Close()

	reg := tools.NewRegistry(skills.NewCatalog(), nil)
	wsToolset := tools.NewWorkspaceToolset(func(context.Context) (executor.Executor, error) {
		return cli, nil
	})
	wsToolset.RegisterInto(reg)

	// 1. list_dir with explicit absolute path (the exact call that failed on staging).
	listArgs, _ := json.Marshal(map[string]string{"path": wsDir})
	listOut, err := reg.Execute(ctx, "list_dir", listArgs)
	if err != nil {
		t.Fatalf("list_dir(%s) failed: %v", wsDir, err)
	}
	if !strings.Contains(listOut, "REAL_EXEC_AGENTS_RULE") {
		t.Fatalf("expected first workspace tool call to prepend AGENTS.md, got:\n%s", listOut)
	}
	if !strings.Contains(listOut, "seed.txt") {
		t.Fatalf("expected list_dir to contain seed.txt, got:\n%s", listOut)
	}

	// 2. list_dir with empty path (defaults to workspace root).
	rootListOut, err := reg.Execute(ctx, "list_dir", json.RawMessage(`{}`))
	if err != nil || !strings.Contains(rootListOut, "seed.txt") {
		t.Fatalf("list_dir({}) failed: err=%v out=%s", err, rootListOut)
	}

	// 3. write_file into a nested directory with spaces & URI-special characters.
	nestedRel := "sub dir/hello #1.txt"
	writeArgs, _ := json.Marshal(map[string]string{
		"path":    nestedRel,
		"content": "alpha line\nbeta line\n",
	})
	writeOut, err := reg.Execute(ctx, "write_file", writeArgs)
	if err != nil || !strings.Contains(writeOut, "wrote") {
		t.Fatalf("write_file failed: err=%v out=%s", err, writeOut)
	}

	// 4. read_file on the written nested file.
	readArgs, _ := json.Marshal(map[string]string{"path": nestedRel})
	readOut, err := reg.Execute(ctx, "read_file", readArgs)
	if err != nil || readOut != "alpha line\nbeta line\n" {
		t.Fatalf("read_file mismatch: err=%v out=%q", err, readOut)
	}

	// 5. edit_file on the nested file and verify via read_file.
	editArgs, _ := json.Marshal(map[string]string{
		"path":     nestedRel,
		"old_text": "beta line",
		"new_text": "gamma edited",
	})
	editOut, err := reg.Execute(ctx, "edit_file", editArgs)
	if err != nil || !strings.Contains(editOut, "edited") {
		t.Fatalf("edit_file failed: err=%v out=%s", err, editOut)
	}
	editedRead, err := reg.Execute(ctx, "read_file", readArgs)
	if err != nil || editedRead != "alpha line\ngamma edited\n" {
		t.Fatalf("read_file after edit mismatch: err=%v out=%q", err, editedRead)
	}

	// 6. exec_command: `ls -la <wsDir>` plus stderr and exit code handling.
	execArgs, _ := json.Marshal(map[string]any{
		"command": fmt.Sprintf("ls -la %q && echo STDERR_MARKER >&2", wsDir),
	})
	execOut, err := reg.Execute(ctx, "exec_command", execArgs)
	if err != nil {
		t.Fatalf("exec_command(ls -la) failed: %v", err)
	}
	if !strings.Contains(execOut, "exit_code: 0") || !strings.Contains(execOut, "seed.txt") || !strings.Contains(execOut, "STDERR_MARKER") {
		t.Fatalf("unexpected exec_command output:\n%s", execOut)
	}

	// 7. Direct protocol coverage: fs/getMetadata (Stat).
	meta, err := cli.Stat(ctx, nestedRel)
	if err != nil || !meta.IsFile || meta.IsDirectory || meta.Size == 0 {
		t.Fatalf("Stat(%s) failed: err=%v meta=%+v", nestedRel, err, meta)
	}

	// 8. Direct protocol coverage: process/start (pipeStdin) + process/write + process/read.
	stdinRes, err := cli.ExecStart(ctx, executor.ExecRequest{
		Command:   "read line; echo GOT:$line",
		Shell:     true,
		PipeStdin: true,
		TimeoutMS: 300,
	})
	if err != nil || !stdinRes.Running {
		t.Fatalf("expected interactive process to stay running waiting for stdin: err=%v res=%+v", err, stdinRes)
	}
	afterWrite, err := cli.ExecWriteStdin(ctx, stdinRes.ProcessID, "HELLO_STDIN\n")
	if err != nil || afterWrite.ExitCode != 0 || !strings.Contains(afterWrite.Stdout, "GOT:HELLO_STDIN") {
		t.Fatalf("ExecWriteStdin failed: err=%v res=%+v", err, afterWrite)
	}
	readRes, err := cli.ExecRead(ctx, stdinRes.ProcessID, 0, 65536, 0)
	if err != nil || !readRes.Exited || readRes.ExitCode == nil || *readRes.ExitCode != 0 {
		t.Fatalf("ExecRead failed: err=%v res=%+v", err, readRes)
	}

	// 9. Direct protocol coverage: process/terminate on a long-running process.
	sleepRes, err := cli.ExecStart(ctx, executor.ExecRequest{
		Command:   "sleep 30",
		Shell:     true,
		TimeoutMS: 200,
	})
	if err != nil || !sleepRes.Running {
		t.Fatalf("expected sleep process to be running: err=%v res=%+v", err, sleepRes)
	}
	if err := cli.ExecTerminate(ctx, sleepRes.ProcessID); err != nil {
		t.Fatalf("ExecTerminate failed: %v", err)
	}
}
