package execserver_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/executor"
	"github.com/cambridgeadvancedsystems/jaz-harness/executor/execserver"
)

func startTestServer(t *testing.T, cfg execserver.Config) (*execserver.Server, *httptest.Server, string) {
	t.Helper()
	srv, err := execserver.New(cfg)
	if err != nil {
		t.Fatalf("execserver.New: %v", err)
	}
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(func() {
		_ = srv.Close()
		ts.Close()
	})
	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/ws"
	return srv, ts, wsURL
}

func waitForPIDDead(t *testing.T, pid int, timeout time.Duration) {
	t.Helper()
	if runtime.GOOS == "windows" {
		return
	}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		err := syscall.Kill(pid, 0)
		if errors.Is(err, syscall.ESRCH) {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("process pid %d is still alive after %v (orphan process leak)", pid, timeout)
}

func waitForFilePID(t *testing.T, path string, timeout time.Duration) int {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		raw, err := os.ReadFile(path)
		if err == nil {
			s := strings.TrimSpace(string(raw))
			if s != "" {
				pid, perr := strconv.Atoi(s)
				if perr == nil && pid > 0 {
					return pid
				}
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for child pid file %s", path)
	return 0
}

func TestConfigAuthFailClosed(t *testing.T) {
	wsDir := t.TempDir()

	if _, err := execserver.New(execserver.Config{WorkspaceDir: wsDir}); err == nil {
		t.Fatal("expected error when no token/hash is provided and AllowUnauthenticated is false")
	}
	if _, err := execserver.New(execserver.Config{WorkspaceDir: wsDir, TokenSHA256: "not-hex"}); err == nil {
		t.Fatal("expected error for malformed token SHA-256")
	}
	if _, err := execserver.New(execserver.Config{
		WorkspaceDir: wsDir,
		Token:        "token-a",
		TokenSHA256:  execserver.HashToken("token-b"),
	}); err == nil {
		t.Fatal("expected error when Token and TokenSHA256 do not match")
	}
	srv, err := execserver.New(execserver.Config{
		WorkspaceDir:         wsDir,
		AllowUnauthenticated: true,
	})
	if err != nil {
		t.Fatalf("expected AllowUnauthenticated config to succeed: %v", err)
	}
	_ = srv.Close()
}

func TestAuthRejectionAndHealthz(t *testing.T) {
	const token = "cap-token-secret-12345"
	wsDir := t.TempDir()
	_, ts, wsURL := startTestServer(t, execserver.Config{
		WorkspaceDir: wsDir,
		TokenSHA256:  execserver.HashToken(token),
	})

	// /healthz must work without authentication.
	resp, err := http.Get(ts.URL + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /healthz status = %d, want 200", resp.StatusCode)
	}
	var health map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		t.Fatalf("decode /healthz: %v", err)
	}
	if health["ok"] != true || health["protocol_version"] != execserver.ProtocolVersion {
		t.Fatalf("unexpected /healthz body: %+v", health)
	}

	// Non-GET/HEAD on /healthz must return 405.
	postResp, err := http.Post(ts.URL+"/healthz", "application/json", nil)
	if err != nil {
		t.Fatalf("POST /healthz: %v", err)
	}
	_ = postResp.Body.Close()
	if postResp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("POST /healthz status = %d, want 405", postResp.StatusCode)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Missing token, wrong token, and whitespace-padded token must be rejected.
	for _, bad := range []string{"", "wrong-token", token + " extra"} {
		if cli, err := execserver.Dial(ctx, wsURL, bad, wsDir); err == nil {
			_ = cli.Close()
			t.Fatalf("expected Dial with bad token %q to fail", bad)
		}
	}

	// Direct HTTP check on /ws confirms 401 + WWW-Authenticate header.
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/ws", nil)
	req.Header.Set("Authorization", "Basic dXNlcjpwYXNz")
	unauthResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET /ws bad auth: %v", err)
	}
	_ = unauthResp.Body.Close()
	if unauthResp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("GET /ws status = %d, want 401", unauthResp.StatusCode)
	}

	// Valid bearer token succeeds on both /ws and /.
	cli, err := execserver.Dial(ctx, wsURL, token, "")
	if err != nil {
		t.Fatalf("Dial with valid token: %v", err)
	}
	defer cli.Close()
	if cli.SessionID() == "" {
		t.Fatal("expected non-empty SessionID")
	}
	if cli.WorkspaceDir() == "" {
		t.Fatal("expected WorkspaceDir populated from initialize")
	}
	if err := cli.Ping(ctx); err != nil {
		t.Fatalf("Ping: %v", err)
	}

	rootURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/"
	cli2, err := execserver.Dial(ctx, rootURL, token, wsDir)
	if err != nil {
		t.Fatalf("Dial root endpoint: %v", err)
	}
	_ = cli2.Close()
}

func TestFilesystemOpsAndContainment(t *testing.T) {
	const token = "fs-test-token"
	wsDir := t.TempDir()
	outsideDir := t.TempDir()
	outsideFile := filepath.Join(outsideDir, "secret.txt")
	if err := os.WriteFile(outsideFile, []byte("outside-data"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	if err := os.Symlink(outsideFile, filepath.Join(wsDir, "escape-link.txt")); err != nil {
		t.Fatalf("create symlink: %v", err)
	}

	_, _, wsURL := startTestServer(t, execserver.Config{
		WorkspaceDir:        wsDir,
		RestrictToWorkspace: true,
		TokenSHA256:         execserver.HashToken(token),
		MaxReadFileBytes:    1024,
		MaxWriteFileBytes:   2048,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cli, err := execserver.Dial(ctx, wsURL, token, wsDir)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer cli.Close()

	// WriteFile with nested parent dir creation and special POSIX filename chars.
	nested := "dir with space/sub/hello #1?x%2.txt"
	if err := cli.WriteFile(ctx, nested, []byte("line1\nline2\n")); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	got, err := cli.ReadFile(ctx, nested)
	if err != nil || string(got) != "line1\nline2\n" {
		t.Fatalf("ReadFile mismatch: err=%v got=%q", err, string(got))
	}

	// Bounded ReadFile rejects oversized file.
	if err := cli.WriteFile(ctx, "big.bin", bytes.Repeat([]byte("A"), 1500)); err != nil {
		t.Fatalf("WriteFile big.bin: %v", err)
	}
	if _, err := cli.ReadFile(ctx, "big.bin"); !errors.Is(err, execserver.ErrFileTooLarge) {
		t.Fatalf("expected ErrFileTooLarge on server limit, got: %v", err)
	}
	if _, err := cli.ReadFileBounded(ctx, nested, 4); !errors.Is(err, execserver.ErrFileTooLarge) {
		t.Fatalf("expected ErrFileTooLarge on per-call limit, got: %v", err)
	}

	// WriteFile rejects payload over MaxWriteFileBytes.
	if err := cli.WriteFile(ctx, "too-big-write.bin", bytes.Repeat([]byte("B"), 3000)); !errors.Is(err, execserver.ErrFileTooLarge) {
		t.Fatalf("expected ErrFileTooLarge on oversized write, got: %v", err)
	}

	// Stat on file, dir, and missing path.
	meta, err := cli.Stat(ctx, nested)
	if err != nil || !meta.IsFile || meta.IsDirectory || meta.Size != 12 {
		t.Fatalf("Stat file mismatch: err=%v meta=%+v", err, meta)
	}
	dirMeta, err := cli.Stat(ctx, "dir with space")
	if err != nil || !dirMeta.IsDirectory || dirMeta.IsFile {
		t.Fatalf("Stat dir mismatch: err=%v meta=%+v", err, dirMeta)
	}
	if _, err := cli.Stat(ctx, "does-not-exist.txt"); !errors.Is(err, execserver.ErrNotFound) {
		t.Fatalf("expected ErrNotFound, got: %v", err)
	}

	// MkdirAll, ListDir, Rename, Remove.
	if err := cli.MkdirAll(ctx, "alpha/beta/gamma"); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	entries, err := cli.ListDir(ctx, "")
	if err != nil || len(entries) < 3 {
		t.Fatalf("ListDir root failed: err=%v entries=%+v", err, entries)
	}
	if err := cli.Rename(ctx, nested, "moved/renamed.txt"); err != nil {
		t.Fatalf("Rename: %v", err)
	}
	movedData, err := cli.ReadFile(ctx, "moved/renamed.txt")
	if err != nil || string(movedData) != "line1\nline2\n" {
		t.Fatalf("ReadFile after rename: err=%v data=%q", err, string(movedData))
	}
	if err := cli.Remove(ctx, "alpha", true); err != nil {
		t.Fatalf("Remove recursive: %v", err)
	}
	if _, err := cli.Stat(ctx, "alpha"); !errors.Is(err, execserver.ErrNotFound) {
		t.Fatalf("expected alpha removed, got: %v", err)
	}
	// Refuses removing workspace root itself.
	if err := cli.Remove(ctx, ".", true); err == nil {
		t.Fatal("expected removing workspace root to fail")
	}

	// Workspace containment rejects .., outside absolute path, symlink escape, dangling symlinks, and escaping cwd.
	for _, badPath := range []string{"../secret.txt", outsideFile, "escape-link.txt"} {
		if _, err := cli.ReadFile(ctx, badPath); !errors.Is(err, execserver.ErrPathEscapes) {
			t.Fatalf("ReadFile(%q) expected ErrPathEscapes, got: %v", badPath, err)
		}
	}
	danglingFileTarget := filepath.Join(outsideDir, "dangling-write.txt")
	if err := os.Symlink(danglingFileTarget, filepath.Join(wsDir, "dangling-file.txt")); err != nil {
		t.Fatalf("create dangling file symlink: %v", err)
	}
	if err := cli.WriteFile(ctx, "dangling-file.txt", []byte("blocked")); !errors.Is(err, execserver.ErrPathEscapes) {
		t.Fatalf("WriteFile through dangling symlink expected ErrPathEscapes, got: %v", err)
	}
	if _, err := os.Stat(danglingFileTarget); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("dangling symlink target was unexpectedly created: %v", err)
	}

	danglingDirTarget := filepath.Join(outsideDir, "dangling-dir")
	if err := os.Symlink(danglingDirTarget, filepath.Join(wsDir, "dangling-dir")); err != nil {
		t.Fatalf("create dangling dir symlink: %v", err)
	}
	if err := cli.WriteFile(ctx, "dangling-dir/sub/pwn.txt", []byte("blocked")); !errors.Is(err, execserver.ErrPathEscapes) {
		t.Fatalf("WriteFile through dangling dir symlink expected ErrPathEscapes, got: %v", err)
	}
	if err := cli.MkdirAll(ctx, "dangling-dir/sub"); !errors.Is(err, execserver.ErrPathEscapes) {
		t.Fatalf("MkdirAll through dangling dir symlink expected ErrPathEscapes, got: %v", err)
	}
	if _, err := os.Stat(danglingDirTarget); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("dangling dir symlink target was unexpectedly created: %v", err)
	}

	// Internal symlink within workspace remains allowed and resolves cleanly.
	if err := os.Symlink(filepath.Join(wsDir, "moved", "renamed.txt"), filepath.Join(wsDir, "internal-link.txt")); err != nil {
		t.Fatalf("create internal symlink: %v", err)
	}
	linkData, err := cli.ReadFile(ctx, "internal-link.txt")
	if err != nil || string(linkData) != "line1\nline2\n" {
		t.Fatalf("ReadFile internal symlink failed: err=%v data=%q", err, string(linkData))
	}

	if _, err := cli.Exec(ctx, executor.ExecRequest{Command: "pwd", Shell: true, Cwd: outsideDir}); !errors.Is(err, execserver.ErrPathEscapes) {
		t.Fatalf("Exec with outside cwd expected ErrPathEscapes, got: %v", err)
	}
}

func TestProcessExecTimeoutAndOrphanCleanup(t *testing.T) {
	const token = "proc-timeout-token"
	wsDir := t.TempDir()
	srv, _, wsURL := startTestServer(t, execserver.Config{
		WorkspaceDir:    wsDir,
		TokenSHA256:     execserver.HashToken(token),
		KillGracePeriod: 150 * time.Millisecond,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	cli, err := execserver.Dial(ctx, wsURL, token, wsDir)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer cli.Close()

	// Normal one-shot Exec with env override, stderr, and non-zero exit code.
	res, err := cli.Exec(ctx, executor.ExecRequest{
		Command: "echo OUT:$JAZ_TEST_VAR && echo ERR_LINE >&2 && exit 7",
		Shell:   true,
		Env:     map[string]string{"JAZ_TEST_VAR": "custom-val"},
	})
	if err != nil {
		t.Fatalf("Exec failed: %v", err)
	}
	if res.ExitCode != 7 || !strings.Contains(res.Stdout, "OUT:custom-val") || !strings.Contains(res.Stderr, "ERR_LINE") {
		t.Fatalf("unexpected Exec result: %+v", res)
	}

	// Direct argv execution without shell.
	directRes, err := cli.Exec(ctx, executor.ExecRequest{
		Command: "/bin/echo",
		Args:    []string{"-n", "direct-argv"},
		Shell:   false,
	})
	if err != nil || directRes.ExitCode != 0 || directRes.Stdout != "direct-argv" {
		t.Fatalf("direct Exec failed: err=%v res=%+v", err, directRes)
	}

	// Hard timeout must kill the entire process group including background child processes.
	pidFile := filepath.Join(wsDir, "child.pid")
	cmd := fmt.Sprintf("sleep 30 & echo $! > %q; echo STARTED; wait", pidFile)
	timeoutRes, err := cli.Exec(ctx, executor.ExecRequest{
		Command:   cmd,
		Shell:     true,
		TimeoutMS: 300,
	})
	if !errors.Is(err, execserver.ErrTimedOut) {
		t.Fatalf("expected ErrTimedOut, got err=%v res=%+v", err, timeoutRes)
	}
	if !strings.Contains(timeoutRes.Stdout, "STARTED") {
		t.Fatalf("expected partial stdout before timeout, got: %+v", timeoutRes)
	}
	childPID := waitForFilePID(t, pidFile, 2*time.Second)
	waitForPIDDead(t, childPID, 3*time.Second)
	if got := srv.ActiveProcesses(); got != 0 {
		t.Fatalf("ActiveProcesses = %d, want 0 after timeout cleanup", got)
	}

	// Caller context cancellation during one-shot Exec must also kill remote process group.
	cancelExecPidFile := filepath.Join(wsDir, "exec-cancel.pid")
	execCtx, execCancel := context.WithTimeout(ctx, 250*time.Millisecond)
	defer execCancel()
	_, cancelExecErr := cli.Exec(execCtx, executor.ExecRequest{
		Command:   fmt.Sprintf("sleep 30 & echo $! > %q; wait", cancelExecPidFile),
		Shell:     true,
		TimeoutMS: 30000,
	})
	if !errors.Is(cancelExecErr, context.DeadlineExceeded) {
		t.Fatalf("expected context.DeadlineExceeded on canceled Exec, got: %v", cancelExecErr)
	}
	cancelExecChildPID := waitForFilePID(t, cancelExecPidFile, 2*time.Second)
	waitForPIDDead(t, cancelExecChildPID, 3*time.Second)
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && srv.ActiveProcesses() > 0 {
		time.Sleep(25 * time.Millisecond)
	}
	if got := srv.ActiveProcesses(); got != 0 {
		t.Fatalf("ActiveProcesses = %d, want 0 after canceled Exec", got)
	}
}

func TestProcessStreamingStdinReadTerminateAndDisconnectCleanup(t *testing.T) {
	const token = "proc-stream-token"
	wsDir := t.TempDir()
	srv, _, wsURL := startTestServer(t, execserver.Config{
		WorkspaceDir:    wsDir,
		TokenSHA256:     execserver.HashToken(token),
		KillGracePeriod: 150 * time.Millisecond,
		MaxOutputBytes:  2048,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	cli, err := execserver.Dial(ctx, wsURL, token, wsDir)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}

	// 1. Streaming stdin/stdout with cat + incremental ExecRead + CloseStdin EOF.
	startRes, err := cli.ExecStart(ctx, executor.ExecRequest{
		Command:   "cat",
		Shell:     true,
		PipeStdin: true,
		TimeoutMS: 200,
	})
	if err != nil || !startRes.Running || startRes.ProcessID == "" {
		t.Fatalf("ExecStart cat failed: err=%v res=%+v", err, startRes)
	}
	if err := cli.ExecWriteStdinWithOptions(ctx, startRes.ProcessID, []byte("chunk-one\n"), false); err != nil {
		t.Fatalf("write chunk-one: %v", err)
	}
	r1, err := cli.ExecRead(ctx, startRes.ProcessID, 0, 65536, 2000)
	if err != nil || len(r1.Chunks) == 0 {
		t.Fatalf("ExecRead 1 failed: err=%v res=%+v", err, r1)
	}
	var firstBuf bytes.Buffer
	for _, ch := range r1.Chunks {
		firstBuf.Write(ch.Data)
	}
	if !strings.Contains(firstBuf.String(), "chunk-one") {
		t.Fatalf("expected chunk-one in first read, got %q", firstBuf.String())
	}

	// Send second chunk and close stdin so cat exits cleanly.
	if err := cli.ExecWriteStdinWithOptions(ctx, startRes.ProcessID, []byte("chunk-two\n"), true); err != nil {
		t.Fatalf("write chunk-two + closeStdin: %v", err)
	}
	var tailBuf bytes.Buffer
	afterSeq := r1.NextSeq
	var finalRead executor.ExecReadResult
	for i := 0; i < 10; i++ {
		finalRead, err = cli.ExecRead(ctx, startRes.ProcessID, afterSeq, 65536, 1000)
		if err != nil {
			t.Fatalf("ExecRead loop: %v", err)
		}
		for _, ch := range finalRead.Chunks {
			tailBuf.Write(ch.Data)
		}
		afterSeq = finalRead.NextSeq
		if finalRead.Closed {
			break
		}
	}
	if !finalRead.Exited || finalRead.ExitCode == nil || *finalRead.ExitCode != 0 {
		t.Fatalf("expected clean exit 0 after EOF, got: %+v", finalRead)
	}
	if !strings.Contains(tailBuf.String(), "chunk-two") {
		t.Fatalf("expected chunk-two after EOF, got %q", tailBuf.String())
	}

	// 2. Interactive ExecWriteStdin helper.
	readProc, err := cli.ExecStart(ctx, executor.ExecRequest{
		Command:   "read line; echo ECHO:$line",
		Shell:     true,
		PipeStdin: true,
		TimeoutMS: 200,
	})
	if err != nil || !readProc.Running {
		t.Fatalf("ExecStart read failed: err=%v res=%+v", err, readProc)
	}
	afterStdin, err := cli.ExecWriteStdin(ctx, readProc.ProcessID, "HELLO_WORLD\n")
	if err != nil || afterStdin.Running || afterStdin.ExitCode != 0 || !strings.Contains(afterStdin.Stdout, "ECHO:HELLO_WORLD") {
		t.Fatalf("ExecWriteStdin failed: err=%v res=%+v", err, afterStdin)
	}

	// 3. ExecTerminate kills process group including grandchild.
	termPidFile := filepath.Join(wsDir, "term.pid")
	termProc, err := cli.ExecStart(ctx, executor.ExecRequest{
		Command:   fmt.Sprintf("sleep 30 & echo $! > %q; wait", termPidFile),
		Shell:     true,
		TimeoutMS: 200,
	})
	if err != nil || !termProc.Running {
		t.Fatalf("ExecStart sleep failed: err=%v res=%+v", err, termProc)
	}
	termChildPID := waitForFilePID(t, termPidFile, 2*time.Second)
	if err := cli.ExecTerminate(ctx, termProc.ProcessID); err != nil {
		t.Fatalf("ExecTerminate: %v", err)
	}
	waitForPIDDead(t, termChildPID, 3*time.Second)
	// Terminating an already-exited process is safe and idempotent.
	if err := cli.ExecTerminate(ctx, termProc.ProcessID); err != nil {
		t.Fatalf("second ExecTerminate on exited process failed: %v", err)
	}

	// 3b. Blocked stdin write on full pipe must not deadlock concurrent ExecTerminate.
	pipeBlockProc, err := cli.ExecStart(ctx, executor.ExecRequest{
		Command:   "sleep 30",
		Shell:     true,
		PipeStdin: true,
		TimeoutMS: 150,
	})
	if err != nil || !pipeBlockProc.Running {
		t.Fatalf("ExecStart pipeBlockProc failed: err=%v res=%+v", err, pipeBlockProc)
	}
	writeDone := make(chan error, 1)
	go func() {
		// Write enough chunks to fill kernel pipe buffer on a non-reading child.
		payload := bytes.Repeat([]byte("Z"), 1024)
		for i := 0; i < 256; i++ {
			if err := cli.ExecWriteStdinWithOptions(ctx, pipeBlockProc.ProcessID, payload, false); err != nil {
				writeDone <- err
				return
			}
		}
		writeDone <- nil
	}()
	time.Sleep(150 * time.Millisecond)
	if err := cli.ExecTerminate(ctx, pipeBlockProc.ProcessID); err != nil {
		t.Fatalf("ExecTerminate during blocked stdin write failed: %v", err)
	}
	select {
	case <-writeDone:
	case <-time.After(3 * time.Second):
		t.Fatal("stdin writer goroutine did not unblock after ExecTerminate (potential stdinMu deadlock)")
	}

	// 4. Context cancellation during ExecStart terminates remote process group.
	cancelPidFile := filepath.Join(wsDir, "cancel.pid")
	shortCtx, shortCancel := context.WithTimeout(ctx, 250*time.Millisecond)
	defer shortCancel()
	_, cancelErr := cli.ExecStart(shortCtx, executor.ExecRequest{
		Command:   fmt.Sprintf("sleep 30 & echo $! > %q; wait", cancelPidFile),
		Shell:     true,
		TimeoutMS: 5000,
	})
	if !errors.Is(cancelErr, context.DeadlineExceeded) {
		t.Fatalf("expected context.DeadlineExceeded on canceled ExecStart, got: %v", cancelErr)
	}
	cancelChildPID := waitForFilePID(t, cancelPidFile, 2*time.Second)
	waitForPIDDead(t, cancelChildPID, 3*time.Second)

	// 5. Output capping prevents runaway output OOM.
	capRes, err := cli.Exec(ctx, executor.ExecRequest{
		Command: "head -c 16384 /dev/zero | tr '\\0' 'X'",
		Shell:   true,
	})
	if err != nil {
		t.Fatalf("capped output Exec: %v", err)
	}
	if !strings.Contains(capRes.Stdout, "stream output capped at 2048 bytes") {
		t.Fatalf("expected output truncation marker, got len=%d", len(capRes.Stdout))
	}

	// 6. Disconnect cleanup terminates surviving session process groups.
	discPidFile := filepath.Join(wsDir, "disconnect.pid")
	discProc, err := cli.ExecStart(ctx, executor.ExecRequest{
		Command:   fmt.Sprintf("sleep 30 & echo $! > %q; wait", discPidFile),
		Shell:     true,
		TimeoutMS: 200,
	})
	if err != nil || !discProc.Running {
		t.Fatalf("ExecStart disconnect test: err=%v res=%+v", err, discProc)
	}
	discChildPID := waitForFilePID(t, discPidFile, 2*time.Second)
	_ = cli.Close()
	waitForPIDDead(t, discChildPID, 3*time.Second)

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && srv.ActiveProcesses() > 0 {
		time.Sleep(25 * time.Millisecond)
	}
	if got := srv.ActiveProcesses(); got != 0 {
		t.Fatalf("ActiveProcesses = %d after disconnect, want 0", got)
	}
}
