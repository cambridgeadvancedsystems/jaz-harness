package anthropic

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// readBedrockEventStream decodes AWS's binary application/vnd.amazon.eventstream
// framing and yields inner JSON payloads (Bedrock wraps each Anthropic SSE JSON
// event inside {"bytes": "<base64-json>"} frames).
func readBedrockEventStream(r io.Reader, onEvent func(jsonPayload []byte) error) error {
	var prelude [12]byte
	for {
		_, err := io.ReadFull(r, prelude[:])
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		totalLen := binary.BigEndian.Uint32(prelude[0:4])
		headersLen := binary.BigEndian.Uint32(prelude[4:8])
		if totalLen < 16 || totalLen > 16<<20 {
			return fmt.Errorf("bedrock eventstream invalid frame length %d", totalLen)
		}
		remLen := int(totalLen) - 12
		buf := make([]byte, remLen)
		if _, err := io.ReadFull(r, buf); err != nil {
			return err
		}
		if int(headersLen)+4 > len(buf) {
			return errors.New("bedrock eventstream truncated headers")
		}
		payload := buf[headersLen : len(buf)-4]
		if len(payload) == 0 {
			continue
		}
		var env struct {
			Bytes   []byte `json:"bytes"`
			Message string `json:"message"`
		}
		if json.Unmarshal(payload, &env) == nil {
			if len(env.Bytes) > 0 {
				if err := onEvent(env.Bytes); err != nil {
					return err
				}
				continue
			}
			if env.Message != "" {
				return fmt.Errorf("bedrock stream error: %s", env.Message)
			}
		}
		if err := onEvent(payload); err != nil {
			return err
		}
	}
}
