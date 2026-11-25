package executor

import (
	"bytes"
	"fmt"
	"unicode/utf8"
)

// DefaultMaxExecOutputBytes caps each captured stdout/stderr stream from a
// command or PTY session (4 MiB) so large builds/tests are preserved while
// runaway output cannot OOM the harness worker.
const DefaultMaxExecOutputBytes = 4 << 20

// CappedBuffer retains up to limit bytes of stream output and records dropped
// bytes so runaway command output cannot OOM the worker or sandbox daemon.
type CappedBuffer struct {
	buf       bytes.Buffer
	limit     int
	dropped   int64
	truncated bool
}

// NewCappedBuffer constructs a CappedBuffer with limit (defaulting to
// DefaultMaxExecOutputBytes when limit <= 0).
func NewCappedBuffer(limit int) *CappedBuffer {
	if limit <= 0 {
		limit = DefaultMaxExecOutputBytes
	}
	return &CappedBuffer{limit: limit}
}

func newCappedBuffer(limit int) *CappedBuffer {
	return NewCappedBuffer(limit)
}

func (c *CappedBuffer) Write(p []byte) (int, error) {
	n := len(p)
	if n == 0 {
		return 0, nil
	}
	rem := c.limit - c.buf.Len()
	if rem > 0 {
		if n <= rem {
			c.buf.Write(p)
			return n, nil
		}
		c.buf.Write(p[:rem])
		c.dropped += int64(n - rem)
		c.truncated = true
		return n, nil
	}
	c.dropped += int64(n)
	c.truncated = true
	return n, nil
}

func (c *CappedBuffer) String() string {
	raw := c.buf.Bytes()
	if !c.truncated {
		return string(raw)
	}
	end := len(raw)
	for end > 0 && !utf8.Valid(raw[:end]) {
		end--
	}
	return string(raw[:end]) + fmt.Sprintf("\n... [stream output capped at %d bytes; %d additional bytes omitted] ...", c.limit, c.dropped)
}

func (c *CappedBuffer) Truncated() bool {
	return c.truncated
}

func (c *CappedBuffer) Dropped() int64 {
	return c.dropped
}
