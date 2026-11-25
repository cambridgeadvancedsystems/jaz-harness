package provider

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
)

// ScanLines reads newline-delimited lines from r without bufio.Scanner's fixed
// small-token failure mode. Memory grows only for lines that need it, bounded
// by maxLineBytes (defaulting to DefaultMaxLineBytes = 64 MiB).
func ScanLines(r io.Reader, maxLineBytes int, fn func(line string) error) error {
	if maxLineBytes <= 0 {
		maxLineBytes = DefaultMaxLineBytes
	}
	br := bufio.NewReaderSize(r, 64*1024)
	var acc bytes.Buffer
	for {
		frag, isPrefix, err := br.ReadLine()
		if len(frag) > 0 {
			if acc.Len()+len(frag) > maxLineBytes {
				return fmt.Errorf("stream line exceeds maximum length (%d bytes)", maxLineBytes)
			}
			acc.Write(frag)
		}
		if isPrefix {
			if err != nil {
				return err
			}
			continue
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				if acc.Len() > 0 {
					return fn(acc.String())
				}
				return nil
			}
			return err
		}
		line := acc.String()
		acc.Reset()
		if cbErr := fn(line); cbErr != nil {
			return cbErr
		}
	}
}
