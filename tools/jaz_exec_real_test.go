package tools_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image/color"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/executor"
	"github.com/cambridgeadvancedsystems/jaz-harness/executor/execserver"
	"github.com/cambridgeadvancedsystems/jaz-harness/skills"
	"github.com/cambridgeadvancedsystems/jaz-harness/tools"
)

func buildJazExecServerBinary(t *testing.T) string {
	t.Helper()
	binName := "jaz-exec-server"
	if runtime.GOOS == "windows" {
		binName += ".exe"
	}
	binPath := filepath.Join(t.TempDir(), binName)
	cmd := exec.Command("go", "build", "-o", binPath, "../cmd/jaz-exec-server")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build jaz-exec-server binary: %v\n%s", err, string(out))
	}
	return binPath
}

func startRealJazExecServer(t *testing.T, binPath, wsDir, token string) (httpBase, wsURL string, stop func()) {
	t.Helper()
	addr := reserveLoopbackAddr(t)
	tokenHash := execserver.HashToken(token)

	cmd := exec.Command(
		binPath,
		"--listen="+addr,
		"--workspace="+wsDir,
		"--restrict-workspace",
		"--token-sha256="+tokenHash,
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("start jaz-exec-server: %v", err)
	}

	httpBase = "http://" + addr
	wsURL = "ws://" + addr + "/ws"

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(httpBase + "/healthz")
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				stopped := false
				return httpBase, wsURL, func() {
					if stopped {
						return
					}
					stopped = true
					if runtime.GOOS == "windows" {
						_ = cmd.Process.Kill()
					} else {
						_ = cmd.Process.Signal(syscall.SIGTERM)
					}
					done := make(chan error, 1)
					go func() {
						done <- cmd.Wait()
					}()
					select {
					case <-done:
					case <-time.After(5 * time.Second):
						_ = cmd.Process.Kill()
						<-done
					}
				}
			}
		}
		time.Sleep(40 * time.Millisecond)
	}
	_ = cmd.Process.Kill()
	_ = cmd.Wait()
	t.Fatalf("jaz-exec-server failed to become healthy on %s within 10s\nstderr:\n%s", addr, stderr.String())
	return "", "", func() {}
}

func TestWorkspaceToolsAgainstRealJazExecServer(t *testing.T) {
	binPath := buildJazExecServerBinary(t)
	wsDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(wsDir, "AGENTS.md"), []byte("JAZ_EXEC_SERVER_AGENTS_RULE"), 0o644); err != nil {
		t.Fatalf("seed AGENTS.md: %v", err)
	}
	if err := os.WriteFile(filepath.Join(wsDir, "seed.txt"), []byte("seed-content"), 0o644); err != nil {
		t.Fatalf("seed seed.txt: %v", err)
	}

	const bearerToken = "jaz-exec-server-real-binary-test-token"
	httpBase, wsURL, stopServer := startRealJazExecServer(t, binPath, wsDir, bearerToken)
	defer stopServer()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// 0. /healthz works unauthenticated; invalid tokens fail closed.
	healthResp, err := http.Get(httpBase + "/healthz")
	if err != nil || healthResp.StatusCode != http.StatusOK {
		t.Fatalf("GET /healthz failed: err=%v resp=%v", err, healthResp)
	}
	_ = healthResp.Body.Close()

	if badCli, badErr := execserver.Dial(ctx, wsURL, "wrong-token", wsDir); badErr == nil {
		_ = badCli.Close()
		t.Fatal("expected Dial to reject wrong bearer token")
	}
	if badCli, badErr := execserver.Dial(ctx, wsURL, "", wsDir); badErr == nil {
		_ = badCli.Close()
		t.Fatal("expected Dial to reject missing bearer token")
	}

	cli, err := execserver.Dial(ctx, wsURL, bearerToken, wsDir)
	if err != nil {
		t.Fatalf("execserver.Dial against real jaz-exec-server: %v", err)
	}
	defer cli.Close()

	reg := tools.NewRegistry(skills.NewCatalog(), nil)
	wsToolset := tools.NewWorkspaceToolset(func(context.Context) (executor.Executor, error) {
		return cli, nil
	})
	wsToolset.RegisterInto(reg)

	// 1. list_dir with explicit absolute path + first-call AGENTS.md prefix.
	listArgs, _ := json.Marshal(map[string]string{"path": wsDir})
	listOut, err := reg.Execute(ctx, "list_dir", listArgs)
	if err != nil {
		t.Fatalf("list_dir(%s) failed: %v", wsDir, err)
	}
	if !strings.Contains(listOut, "JAZ_EXEC_SERVER_AGENTS_RULE") {
		t.Fatalf("expected first workspace call to prepend AGENTS.md, got:\n%s", listOut)
	}
	if !strings.Contains(listOut, "seed.txt") {
		t.Fatalf("expected list_dir to contain seed.txt, got:\n%s", listOut)
	}

	// 2. list_dir with empty path.
	rootListOut, err := reg.Execute(ctx, "list_dir", json.RawMessage(`{}`))
	if err != nil || !strings.Contains(rootListOut, "seed.txt") {
		t.Fatalf("list_dir({}) failed: err=%v out=%s", err, rootListOut)
	}

	// 3. write_file into nested directory with spaces & special characters.
	nestedRel := "sub dir/hello #1.txt"
	writeArgs, _ := json.Marshal(map[string]string{
		"path":    nestedRel,
		"content": "alpha line\nbeta line\n",
	})
	writeOut, err := reg.Execute(ctx, "write_file", writeArgs)
	if err != nil || !strings.Contains(writeOut, "wrote") {
		t.Fatalf("write_file failed: err=%v out=%s", err, writeOut)
	}

	// 4. read_file on the nested file.
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

	// 6. exec_command with stdout, stderr, and exit code.
	execArgs, _ := json.Marshal(map[string]any{
		"command": fmt.Sprintf("ls -la %q && echo STDERR_MARKER >&2", wsDir),
	})
	execOut, err := reg.Execute(ctx, "exec_command", execArgs)
	if err != nil {
		t.Fatalf("exec_command failed: %v", err)
	}
	if !strings.Contains(execOut, "exit_code: 0") || !strings.Contains(execOut, "seed.txt") || !strings.Contains(execOut, "STDERR_MARKER") {
		t.Fatalf("unexpected exec_command output:\n%s", execOut)
	}

	// 7. view_image reads a workspace PNG through jaz-exec-server and returns multimodal parts.
	imgPath := "assets/sample.png"
	if err := cli.WriteFile(ctx, imgPath, makeTestPNG(t, color.RGBA{R: 32, G: 128, B: 220, A: 255})); err != nil {
		t.Fatalf("WriteFile sample.png: %v", err)
	}
	viewArgs, _ := json.Marshal(map[string]string{
		"path":   imgPath,
		"detail": "high",
	})
	viewRes, err := reg.ExecuteResult(ctx, "view_image", viewArgs)
	if err != nil {
		t.Fatalf("view_image failed: %v", err)
	}
	if len(viewRes.Images) != 1 || viewRes.Images[0].MediaType != "image/png" || viewRes.Images[0].Base64 == "" || viewRes.Images[0].Path != imgPath {
		t.Fatalf("expected multimodal image/png result from view_image, got: %+v", viewRes)
	}

	// 8. Direct protocol coverage: Stat & MkdirAll.
	if err := cli.MkdirAll(ctx, "deep/a/b"); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	meta, err := cli.Stat(ctx, nestedRel)
	if err != nil || !meta.IsFile || meta.IsDirectory || meta.Size == 0 {
		t.Fatalf("Stat(%s) failed: err=%v meta=%+v", nestedRel, err, meta)
	}

	// 9. Streaming stdin/stdout via ExecStart (PipeStdin) + ExecWriteStdin + ExecRead.
	stdinRes, err := cli.ExecStart(ctx, executor.ExecRequest{
		Command:   "read line; echo GOT:$line",
		Shell:     true,
		PipeStdin: true,
		TimeoutMS: 250,
	})
	if err != nil || !stdinRes.Running {
		t.Fatalf("expected interactive process to stay running: err=%v res=%+v", err, stdinRes)
	}
	afterWrite, err := cli.ExecWriteStdin(ctx, stdinRes.ProcessID, "HELLO_JAZ_EXEC\n")
	if err != nil || afterWrite.ExitCode != 0 || !strings.Contains(afterWrite.Stdout, "GOT:HELLO_JAZ_EXEC") {
		t.Fatalf("ExecWriteStdin failed: err=%v res=%+v", err, afterWrite)
	}
	readRes, err := cli.ExecRead(ctx, stdinRes.ProcessID, 0, 65536, 0)
	if err != nil || !readRes.Exited || readRes.ExitCode == nil || *readRes.ExitCode != 0 {
		t.Fatalf("ExecRead failed: err=%v res=%+v", err, readRes)
	}

	// 10. Hard timeout + grandchild cleanup via Exec.
	pidFile := filepath.Join(wsDir, "real-child.pid")
	_, timeoutErr := cli.Exec(ctx, executor.ExecRequest{
		Command:   fmt.Sprintf("sleep 30 & echo $! > %q; wait", pidFile),
		Shell:     true,
		TimeoutMS: 300,
	})
	if !errors.Is(timeoutErr, execserver.ErrTimedOut) {
		t.Fatalf("expected ErrTimedOut from real binary Exec, got: %v", timeoutErr)
	}
	if runtime.GOOS != "windows" {
		deadline := time.Now().Add(3 * time.Second)
		var childPID int
		for time.Now().Before(deadline) {
			if raw, err := os.ReadFile(pidFile); err == nil {
				if p, err := strconv.Atoi(strings.TrimSpace(string(raw))); err == nil && p > 0 {
					childPID = p
					break
				}
			}
			time.Sleep(25 * time.Millisecond)
		}
		if childPID == 0 {
			t.Fatal("failed to read grandchild pid from real binary timeout test")
		}
		killDeadline := time.Now().Add(3 * time.Second)
		for time.Now().Before(killDeadline) {
			if errors.Is(syscall.Kill(childPID, 0), syscall.ESRCH) {
				childPID = 0
				break
			}
			time.Sleep(25 * time.Millisecond)
		}
		if childPID != 0 {
			t.Fatalf("grandchild pid %d survived real binary timeout", childPID)
		}
	}

	// 11. ExecTerminate on a long-running process.
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
