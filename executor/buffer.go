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

type cappedBuffer struct {
	buf       bytes.Buffer
	limit     int
	dropped   int64
	truncated bool
}

func newCappedBuffer(limit int) *cappedBuffer {
	if limit <= 0 {
		limit = DefaultMaxExecOutputBytes
	}
	return &cappedBuffer{limit: limit}
}

func (c *cappedBuffer) Write(p []byte) (int, error) {
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

func (c *cappedBuffer) String() string {
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
