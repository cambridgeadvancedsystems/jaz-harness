package execserver

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/executor"
	"github.com/google/uuid"
)

type notifyFunc func(method string, params any)

type storedChunk struct {
	seq    uint64
	stream string
	data   []byte
}

type managedProcess struct {
	id        string
	sessionID string
	cmd       *exec.Cmd
	pid       int
	pipeStdin bool

	stdinMu     sync.Mutex
	stdin       io.WriteCloser
	stdinClosed bool

	notifyMu sync.Mutex
	emit     notifyFunc

	mu          sync.Mutex
	notifyCh    chan struct{}
	chunks      []storedChunk
	chunkBytes  int
	maxRetained int
	nextSeq     uint64
	exited      bool
	exitCode    int
	closed      bool
	failure     string
	createdAt   time.Time

	waitDoneCh chan struct{}
	killGrace  time.Duration
	killOnce   sync.Once
}

func (p *managedProcess) broadcastLocked() {
	close(p.notifyCh)
	p.notifyCh = make(chan struct{})
}

func (p *managedProcess) appendChunk(stream string, data []byte) {
	if len(data) == 0 {
		return
	}
	copied := make([]byte, len(data))
	copy(copied, data)

	p.notifyMu.Lock()
	defer p.notifyMu.Unlock()

	p.mu.Lock()
	p.nextSeq++
	seq := p.nextSeq
	p.chunks = append(p.chunks, storedChunk{
		seq:    seq,
		stream: stream,
		data:   copied,
	})
	p.chunkBytes += len(copied)
	for len(p.chunks) > 1 && p.chunkBytes > p.maxRetained {
		p.chunkBytes -= len(p.chunks[0].data)
		p.chunks = p.chunks[1:]
	}
	p.broadcastLocked()
	p.mu.Unlock()

	if p.emit != nil {
		p.emit(NotifyProcessOutput, processOutputNotification{
			ProcessID: p.id,
			Seq:       seq,
			Stream:    stream,
			Chunk:     base64.StdEncoding.EncodeToString(copied),
		})
	}
}

func (p *managedProcess) finish(exitCode int, failure string) {
	p.closeStdin()

	p.notifyMu.Lock()
	defer p.notifyMu.Unlock()

	p.mu.Lock()
	p.exitCode = exitCode
	p.exited = true
	p.closed = true
	p.failure = failure
	p.broadcastLocked()
	p.mu.Unlock()

	if p.emit != nil {
		p.emit(NotifyProcessExited, processExitedNotification{
			ProcessID: p.id,
			ExitCode:  exitCode,
		})
		p.emit(NotifyProcessClosed, processClosedNotification{
			ProcessID: p.id,
		})
	}
}

func (p *managedProcess) closeStdin() {
	p.stdinMu.Lock()
	w := p.stdin
	already := p.stdinClosed
	p.stdinClosed = true
	p.stdin = nil
	p.stdinMu.Unlock()
	if !already && w != nil {
		_ = w.Close()
	}
}

func (p *managedProcess) writeStdin(data []byte, closeAfter bool) (int, *rpcErrorBody) {
	if !p.pipeStdin {
		return 0, &rpcErrorBody{
			Code:    ErrCodeInvalidParams,
			Message: "process stdin is not piped (start with pipeStdin=true)",
		}
	}
	p.stdinMu.Lock()
	defer p.stdinMu.Unlock()

	if p.stdinClosed || p.stdin == nil {
		return 0, &rpcErrorBody{
			Code:    ErrCodeInvalidRequest,
			Message: "process stdin is closed",
		}
	}
	written := 0
	if len(data) > 0 {
		n, err := p.stdin.Write(data)
		written = n
		if err != nil {
			return written, &rpcErrorBody{
				Code:    ErrCodeInternal,
				Message: fmt.Sprintf("write stdin: %v", err),
			}
		}
	}
	if closeAfter {
		_ = p.stdin.Close()
		p.stdinClosed = true
		p.stdin = nil
	}
	return written, nil
}

func (p *managedProcess) readChunks(ctx context.Context, afterSeq uint64, maxBytes int, waitMS uint64) (processReadResult, error) {
	var timer *time.Timer
	var timerCh <-chan time.Time
	if waitMS > 0 {
		timer = time.NewTimer(time.Duration(waitMS) * time.Millisecond)
		defer timer.Stop()
		timerCh = timer.C
	}

	for {
		p.mu.Lock()
		res, hasData := p.snapshotLocked(afterSeq, maxBytes)
		if hasData || res.Closed || waitMS == 0 {
			p.mu.Unlock()
			return res, nil
		}
		ch := p.notifyCh
		p.mu.Unlock()

		select {
		case <-ch:
		case <-timerCh:
			p.mu.Lock()
			res, _ = p.snapshotLocked(afterSeq, maxBytes)
			p.mu.Unlock()
			return res, nil
		case <-ctx.Done():
			return processReadResult{}, ctx.Err()
		}
	}
}

func (p *managedProcess) snapshotLocked(afterSeq uint64, maxBytes int) (processReadResult, bool) {
	out := make([]wireOutputChunk, 0)
	total := 0
	nextSeq := afterSeq
	for _, ch := range p.chunks {
		if ch.seq <= afterSeq {
			continue
		}
		if maxBytes > 0 && len(out) > 0 && total+len(ch.data) > maxBytes {
			break
		}
		out = append(out, wireOutputChunk{
			Seq:    ch.seq,
			Stream: ch.stream,
			Chunk:  base64.StdEncoding.EncodeToString(ch.data),
		})
		total += len(ch.data)
		nextSeq = ch.seq
	}
	var exitCodePtr *int
	if p.exited {
		code := p.exitCode
		exitCodePtr = &code
	}
	return processReadResult{
		Chunks:   out,
		NextSeq:  nextSeq,
		Exited:   p.exited,
		ExitCode: exitCodePtr,
		Closed:   p.closed,
		Failure:  p.failure,
	}, len(out) > 0
}

// terminate kills the process group BEFORE closing stdin so a caller blocked
// inside stdin.Write on a full pipe unblocks immediately with EPIPE instead
// of deadlocking on stdinMu.
func (p *managedProcess) terminate() {
	p.killOnce.Do(func() {
		terminateProcessGroup(p.cmd, p.pid, p.killGrace, p.waitDoneCh)
		p.closeStdin()
	})
}

type procStreamWriter struct {
	proc   *managedProcess
	stream string
}

func (w procStreamWriter) Write(p []byte) (int, error) {
	w.proc.appendChunk(w.stream, p)
	return len(p), nil
}

type processManager struct {
	fs             *fsService
	maxOutputBytes int
	maxProcesses   int
	killGrace      time.Duration

	mu     sync.Mutex
	procs  map[string]*managedProcess
	active map[string]struct{}
	closed bool
}

func newProcessManager(fsSvc *fsService, maxOutputBytes, maxProcesses int, killGrace time.Duration) *processManager {
	if maxOutputBytes <= 0 {
		maxOutputBytes = DefaultMaxOutputBytes
	}
	if maxProcesses <= 0 {
		maxProcesses = DefaultMaxProcesses
	}
	if killGrace <= 0 {
		killGrace = DefaultKillGracePeriod
	}
	return &processManager{
		fs:             fsSvc,
		maxOutputBytes: maxOutputBytes,
		maxProcesses:   maxProcesses,
		killGrace:      killGrace,
		procs:          make(map[string]*managedProcess),
		active:         make(map[string]struct{}),
	}
}

func (m *processManager) buildCommand(command string, args []string, shell bool, cwd string, env map[string]string) (*exec.Cmd, *rpcErrorBody) {
	if strings.TrimSpace(command) == "" {
		return nil, &rpcErrorBody{
			Code:    ErrCodeInvalidParams,
			Message: "command is required",
		}
	}
	resolvedCwd, errBody := m.fs.resolvePath(cwd)
	if errBody != nil {
		return nil, errBody
	}
	info, err := os.Stat(resolvedCwd)
	if err != nil {
		return nil, mapFSError(err)
	}
	if !info.IsDir() {
		return nil, &rpcErrorBody{
			Code:    ErrCodeInvalidParams,
			Message: fmt.Sprintf("cwd %q is not a directory", cwd),
		}
	}

	var cmd *exec.Cmd
	if shell || len(args) == 0 {
		if runtime.GOOS == "windows" {
			cmd = exec.Command("cmd.exe", "/C", command)
		} else {
			cmd = exec.Command("/bin/sh", "-lc", command)
		}
	} else {
		cmd = exec.Command(command, args...)
	}
	cmd.Dir = resolvedCwd
	cmd.WaitDelay = 500 * time.Millisecond
	if len(env) > 0 {
		cmd.Env = mergeEnv(os.Environ(), env)
	}
	configureProcessGroup(cmd)
	return cmd, nil
}

func mergeEnv(base []string, overrides map[string]string) []string {
	idx := make(map[string]int, len(base)+len(overrides))
	out := make([]string, 0, len(base)+len(overrides))
	for _, entry := range base {
		k, _, ok := strings.Cut(entry, "=")
		if !ok {
			out = append(out, entry)
			continue
		}
		idx[k] = len(out)
		out = append(out, entry)
	}
	for k, v := range overrides {
		if k == "" || strings.ContainsAny(k, "=\x00") || strings.ContainsRune(v, 0) {
			continue
		}
		pair := k + "=" + v
		if pos, exists := idx[k]; exists {
			out[pos] = pair
		} else {
			idx[k] = len(out)
			out = append(out, pair)
		}
	}
	return out
}

func (m *processManager) registerAndStart(proc *managedProcess) *rpcErrorBody {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return &rpcErrorBody{
			Code:    ErrCodeInternal,
			Message: "server is shutting down",
		}
	}
	if _, exists := m.procs[proc.id]; exists {
		return &rpcErrorBody{
			Code:    ErrCodeAlreadyExists,
			Message: fmt.Sprintf("process %q already exists", proc.id),
		}
	}
	if len(m.procs) >= m.maxProcesses {
		m.evictOldestClosedLocked()
	}
	if len(m.active) >= m.maxProcesses || len(m.procs) >= m.maxProcesses {
		return &rpcErrorBody{
			Code:    ErrCodeProcessLimit,
			Message: fmt.Sprintf("maximum process limit (%d) reached", m.maxProcesses),
		}
	}
	if err := proc.cmd.Start(); err != nil {
		return &rpcErrorBody{
			Code:    ErrCodeInternal,
			Message: fmt.Sprintf("start process: %v", err),
		}
	}
	if proc.cmd.Process != nil {
		proc.pid = proc.cmd.Process.Pid
	}
	m.procs[proc.id] = proc
	m.active[proc.id] = struct{}{}
	return nil
}

func (m *processManager) evictOldestClosedLocked() {
	var oldestID string
	var oldestTime time.Time
	for id, p := range m.procs {
		p.mu.Lock()
		isClosed := p.closed
		created := p.createdAt
		p.mu.Unlock()
		if !isClosed {
			continue
		}
		if oldestID == "" || created.Before(oldestTime) {
			oldestID = id
			oldestTime = created
		}
	}
	if oldestID != "" {
		delete(m.procs, oldestID)
	}
}

func extractExitCode(waitErr error) (int, string) {
	if waitErr == nil {
		return 0, ""
	}
	var exitErr *exec.ExitError
	if errors.As(waitErr, &exitErr) {
		return exitErr.ExitCode(), ""
	}
	return -1, waitErr.Error()
}

func (m *processManager) execSync(ctx context.Context, sessionID string, params processExecParams) (processExecResult, *rpcErrorBody) {
	cmd, errBody := m.buildCommand(params.Command, params.Args, params.Shell, params.Cwd, params.Env)
	if errBody != nil {
		return processExecResult{}, errBody
	}

	timeout := DefaultExecTimeout
	if params.TimeoutMS > 0 {
		timeout = time.Duration(params.TimeoutMS) * time.Millisecond
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	stdoutCap := executor.NewCappedBuffer(m.maxOutputBytes)
	stderrCap := executor.NewCappedBuffer(m.maxOutputBytes)
	cmd.Stdout = stdoutCap
	cmd.Stderr = stderrCap

	procID := strings.TrimSpace(params.ProcessID)
	if procID == "" {
		procID = "exec_" + uuid.NewString()[:8]
	}
	proc := &managedProcess{
		id:         procID,
		sessionID:  sessionID,
		cmd:        cmd,
		notifyCh:   make(chan struct{}),
		createdAt:  time.Now(),
		waitDoneCh: make(chan struct{}),
		killGrace:  m.killGrace,
	}
	if errBody := m.registerAndStart(proc); errBody != nil {
		return processExecResult{}, errBody
	}

	var timedOut atomic.Bool
	watchDone := make(chan struct{})
	go func() {
		defer close(watchDone)
		select {
		case <-proc.waitDoneCh:
		case <-runCtx.Done():
			if errors.Is(runCtx.Err(), context.DeadlineExceeded) && ctx.Err() == nil {
				timedOut.Store(true)
			}
			proc.terminate()
		}
	}()

	waitErr := cmd.Wait()
	close(proc.waitDoneCh)
	<-watchDone

	m.mu.Lock()
	delete(m.active, procID)
	delete(m.procs, procID)
	m.mu.Unlock()

	if timedOut.Load() {
		return processExecResult{
			Stdout:   stdoutCap.String(),
			Stderr:   stderrCap.String(),
			ExitCode: -1,
			TimedOut: true,
		}, nil
	}
	exitCode, _ := extractExitCode(waitErr)
	return processExecResult{
		Stdout:   stdoutCap.String(),
		Stderr:   stderrCap.String(),
		ExitCode: exitCode,
	}, nil
}

func (m *processManager) start(sessionID string, params processStartParams, emit notifyFunc) (string, *rpcErrorBody) {
	cmd, errBody := m.buildCommand(params.Command, params.Args, params.Shell, params.Cwd, params.Env)
	if errBody != nil {
		return "", errBody
	}

	procID := strings.TrimSpace(params.ProcessID)
	if procID == "" {
		procID = "proc_" + uuid.NewString()[:8]
	}

	var stdinPipe io.WriteCloser
	if params.PipeStdin {
		pipe, err := cmd.StdinPipe()
		if err != nil {
			return "", &rpcErrorBody{
				Code:    ErrCodeInternal,
				Message: fmt.Sprintf("create stdin pipe: %v", err),
			}
		}
		stdinPipe = pipe
	}

	proc := &managedProcess{
		id:          procID,
		sessionID:   sessionID,
		cmd:         cmd,
		pipeStdin:   params.PipeStdin,
		stdin:       stdinPipe,
		emit:        emit,
		notifyCh:    make(chan struct{}),
		maxRetained: m.maxOutputBytes * 2,
		createdAt:   time.Now(),
		waitDoneCh:  make(chan struct{}),
		killGrace:   m.killGrace,
	}
	cmd.Stdout = procStreamWriter{proc: proc, stream: "stdout"}
	cmd.Stderr = procStreamWriter{proc: proc, stream: "stderr"}

	if errBody := m.registerAndStart(proc); errBody != nil {
		if stdinPipe != nil {
			_ = stdinPipe.Close()
		}
		return "", errBody
	}

	go m.waitManaged(proc)
	return procID, nil
}

func (m *processManager) waitManaged(proc *managedProcess) {
	waitErr := proc.cmd.Wait()
	close(proc.waitDoneCh)

	exitCode, failure := extractExitCode(waitErr)

	m.mu.Lock()
	delete(m.active, proc.id)
	m.mu.Unlock()

	proc.finish(exitCode, failure)
}

func (m *processManager) get(sessionID, procID string) (*managedProcess, *rpcErrorBody) {
	m.mu.Lock()
	proc, ok := m.procs[procID]
	m.mu.Unlock()
	if !ok || (sessionID != "" && proc.sessionID != sessionID) {
		return nil, &rpcErrorBody{
			Code:    ErrCodeNotFound,
			Message: fmt.Sprintf("process %q not found", procID),
		}
	}
	return proc, nil
}

func (m *processManager) terminate(sessionID, procID string) *rpcErrorBody {
	proc, errBody := m.get(sessionID, procID)
	if errBody != nil {
		return errBody
	}
	proc.terminate()
	<-proc.waitDoneCh
	return nil
}

func (m *processManager) terminateSession(sessionID string) {
	m.mu.Lock()
	var targets []*managedProcess
	for id := range m.active {
		if p, ok := m.procs[id]; ok && p.sessionID == sessionID {
			targets = append(targets, p)
		}
	}
	m.mu.Unlock()

	terminateTargets(targets)
}

func (m *processManager) terminateAll() {
	m.mu.Lock()
	m.closed = true
	var targets []*managedProcess
	for id := range m.active {
		if p, ok := m.procs[id]; ok {
			targets = append(targets, p)
		}
	}
	m.mu.Unlock()

	terminateTargets(targets)
}

func terminateTargets(targets []*managedProcess) {
	var wg sync.WaitGroup
	for _, p := range targets {
		wg.Add(1)
		go func(proc *managedProcess) {
			defer wg.Done()
			proc.terminate()
			<-proc.waitDoneCh
		}(p)
	}
	wg.Wait()
}

func (m *processManager) activeCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.active)
}
