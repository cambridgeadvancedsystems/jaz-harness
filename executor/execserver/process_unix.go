//go:build unix

package execserver

import (
	"errors"
	"os/exec"
	"syscall"
	"time"
)

func configureProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
	}
}

func terminateProcessGroup(cmd *exec.Cmd, pid int, grace time.Duration, exited <-chan struct{}) {
	if pid <= 0 {
		return
	}
	// Never signal a PID after cmd.Wait() has already reaped the process, since
	// the OS may have recycled the PID for an unrelated process.
	select {
	case <-exited:
		return
	default:
	}

	if grace <= 0 {
		grace = DefaultKillGracePeriod
	}

	// Signal the entire process group (-pid) first so child/grandchild processes
	// spawned by a shell receive SIGTERM cleanly.
	err := syscall.Kill(-pid, syscall.SIGTERM)
	if err != nil && !errors.Is(err, syscall.ESRCH) {
		_ = syscall.Kill(pid, syscall.SIGTERM)
	}

	timer := time.NewTimer(grace)
	defer timer.Stop()

	select {
	case <-exited:
		// Kill any remaining stubborn grandchildren still sharing PGID pid.
		_ = syscall.Kill(-pid, syscall.SIGKILL)
		return
	case <-timer.C:
	}

	err = syscall.Kill(-pid, syscall.SIGKILL)
	if err != nil && !errors.Is(err, syscall.ESRCH) && cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
}
