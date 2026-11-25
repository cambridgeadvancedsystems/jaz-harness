//go:build !unix

package execserver

import (
	"os/exec"
	"time"
)

func configureProcessGroup(_ *exec.Cmd) {}

func terminateProcessGroup(cmd *exec.Cmd, _ int, _ time.Duration, _ <-chan struct{}) {
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
}
