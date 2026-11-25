package anthropic

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
)

func readAnthropicSSE(r io.Reader, onEvent func([]byte) error) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		if err := onEvent([]byte(data)); err != nil {
			return err
		}
	}
	return sc.Err()
}

type blockAcc struct {
	blockType    string
	id           string
	name         string
	initialInput string
	textBuf      strings.Builder
	thinkingBuf  strings.Builder
	sigBuf       strings.Builder
	inputBuf     strings.Builder
	redactedData string
	rawServer    json.RawMessage
	citations    []schema.Citation
}

type anthropicStreamAcc struct {
	providerID string
	respID     string
	model      string
	stopReason string
	usage      schema.Usage
	blocks     map[int]*blockAcc
	emit       provider.StreamCallback
}

func newAnthropicStreamAcc(providerID, model string, emit provider.StreamCallback) *anthropicStreamAcc {
	return &anthropicStreamAcc{
		providerID: providerID,
		model:      model,
		blocks:     make(map[int]*blockAcc),
		emit:       emit,
	}
}

func (a *anthropicStreamAcc) handleJSONEvent(raw []byte) error {
	var ev struct {
		Type    string `json:"type"`
		Index   int    `json:"index"`
		Message *struct {
			ID    string `json:"id"`
			Model string `json:"model"`
			Usage *struct {
				InputTokens              int64 `json:"input_tokens"`
				CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
				CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
				OutputTokens             int64 `json:"output_tokens"`
			} `json:"usage"`
		} `json:"message"`
		ContentBlock json.RawMessage `json:"content_block"`
		Delta        *struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			Thinking    string `json:"thinking"`
			Signature   string `json:"signature"`
			PartialJSON string `json:"partial_json"`
			StopReason  string `json:"stop_reason"`
			Citation    *struct {
				Type      string `json:"type"`
				URL       string `json:"url"`
				Title     string `json:"title"`
				CitedText string `json:"cited_text"`
			} `json:"citation"`
		} `json:"delta"`
		Usage *struct {
			InputTokens              int64 `json:"input_tokens"`
			CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
			OutputTokens             int64 `json:"output_tokens"`
		} `json:"usage"`
		Error *struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &ev); err != nil {
		return nil
	}

	switch ev.Type {
	case "message_start":
		if ev.Message != nil {
			a.respID = ev.Message.ID
			if ev.Message.Model != "" {
				a.model = ev.Message.Model
			}
			if ev.Message.Usage != nil {
				u := ev.Message.Usage
				a.usage.InputTokens = u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens
				a.usage.CacheCreationTokens = u.CacheCreationInputTokens
				a.usage.CachedInputTokens = u.CacheReadInputTokens
				a.usage.OutputTokens = u.OutputTokens
			}
		}
	case "content_block_start":
		var cb struct {
			Type      string          `json:"type"`
			ID        string          `json:"id"`
			Name      string          `json:"name"`
			Text      string          `json:"text"`
			Thinking  string          `json:"thinking"`
			Signature string          `json:"signature"`
			Data      string          `json:"data"`
			Input     json.RawMessage `json:"input"`
		}
		if json.Unmarshal(ev.ContentBlock, &cb) != nil {
			return nil
		}
		blk := &blockAcc{
			blockType:    cb.Type,
			id:           cb.ID,
			name:         cb.Name,
			initialInput: strings.TrimSpace(string(cb.Input)),
			redactedData: cb.Data,
		}
		if cb.Text != "" {
			blk.textBuf.WriteString(cb.Text)
		}
		if cb.Thinking != "" {
			blk.thinkingBuf.WriteString(cb.Thinking)
		}
		if cb.Signature != "" {
			blk.sigBuf.WriteString(cb.Signature)
		}
		switch cb.Type {
		case "web_search_tool_result", "web_fetch_tool_result", "tool_search_tool_result", "tool_reference":
			blk.rawServer = append(json.RawMessage(nil), ev.ContentBlock...)
		}
		a.blocks[ev.Index] = blk

		if a.emit != nil {
			switch cb.Type {
			case "tool_use":
				a.emit(provider.StreamChunk{Kind: provider.DeltaToolCallStart, CallID: cb.ID, ToolName: cb.Name})
			case "server_tool_use":
				a.emit(provider.StreamChunk{Kind: provider.DeltaHostedToolStart, CallID: cb.ID, ToolName: cb.Name})
			case "web_search_tool_result", "web_fetch_tool_result", "tool_search_tool_result":
				a.emit(provider.StreamChunk{Kind: provider.DeltaHostedToolComplete, CallID: cb.ID, ToolName: cb.Type})
			}
		}
	case "content_block_delta":
		blk := a.blocks[ev.Index]
		if blk == nil || ev.Delta == nil {
			return nil
		}
		switch ev.Delta.Type {
		case "text_delta":
			blk.textBuf.WriteString(ev.Delta.Text)
			if a.emit != nil && ev.Delta.Text != "" {
				a.emit(provider.StreamChunk{Kind: provider.DeltaText, Text: ev.Delta.Text})
			}
		case "thinking_delta":
			blk.thinkingBuf.WriteString(ev.Delta.Thinking)
			if a.emit != nil && ev.Delta.Thinking != "" {
				a.emit(provider.StreamChunk{Kind: provider.DeltaThinking, Text: ev.Delta.Thinking})
			}
		case "signature_delta":
			blk.sigBuf.WriteString(ev.Delta.Signature)
		case "input_json_delta":
			blk.inputBuf.WriteString(ev.Delta.PartialJSON)
			if a.emit != nil && blk.blockType == "tool_use" && ev.Delta.PartialJSON != "" {
				a.emit(provider.StreamChunk{Kind: provider.DeltaToolCallArgs, CallID: blk.id, Arguments: ev.Delta.PartialJSON})
			}
		case "citations_delta":
			if ev.Delta.Citation != nil {
				blk.citations = append(blk.citations, schema.Citation{
					Type:      ev.Delta.Citation.Type,
					URL:       ev.Delta.Citation.URL,
					Title:     ev.Delta.Citation.Title,
					CitedText: ev.Delta.Citation.CitedText,
				})
			}
		}
	case "message_delta":
		if ev.Delta != nil && ev.Delta.StopReason != "" {
			a.stopReason = ev.Delta.StopReason
		}
		if ev.Usage != nil {
			if ev.Usage.OutputTokens > 0 {
				a.usage.OutputTokens = ev.Usage.OutputTokens
			}
			if ev.Usage.InputTokens > 0 {
				a.usage.InputTokens = ev.Usage.InputTokens + ev.Usage.CacheCreationInputTokens + ev.Usage.CacheReadInputTokens
			}
			if ev.Usage.CacheReadInputTokens > 0 {
				a.usage.CachedInputTokens = ev.Usage.CacheReadInputTokens
			}
			if ev.Usage.CacheCreationInputTokens > 0 {
				a.usage.CacheCreationTokens = ev.Usage.CacheCreationInputTokens
			}
		}
	case "error":
		if ev.Error != nil {
			return fmt.Errorf("anthropic stream error (%s): %s", ev.Error.Type, ev.Error.Message)
		}
		return errors.New("anthropic stream error")
	}
	return nil
}

func (a *anthropicStreamAcc) finish() provider.Response {
	indices := make([]int, 0, len(a.blocks))
	for idx := range a.blocks {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	var (
		canonical []schema.ContentBlock
		rawNative []json.RawMessage
	)

	for _, idx := range indices {
		blk := a.blocks[idx]
		switch blk.blockType {
		case "thinking":
			thinking := blk.thinkingBuf.String()
			sig := blk.sigBuf.String()
			nativeMap := map[string]any{
				"type":      "thinking",
				"thinking":  thinking,
				"signature": sig,
			}
			raw, _ := json.Marshal(nativeMap)
			rawNative = append(rawNative, raw)
			canonical = append(canonical, schema.ContentBlock{
				Type:      schema.BlockThinking,
				Thinking:  thinking,
				Signature: sig,
			})
		case "redacted_thinking":
			nativeMap := map[string]any{
				"type": "redacted_thinking",
				"data": blk.redactedData,
			}
			raw, _ := json.Marshal(nativeMap)
			rawNative = append(rawNative, raw)
			canonical = append(canonical, schema.ContentBlock{
				Type:         schema.BlockRedactedThinking,
				RedactedData: blk.redactedData,
			})
		case "text":
			text := blk.textBuf.String()
			nativeMap := map[string]any{
				"type": "text",
				"text": text,
			}
			raw, _ := json.Marshal(nativeMap)
			rawNative = append(rawNative, raw)
			canonical = append(canonical, schema.ContentBlock{
				Type:      schema.BlockText,
				Text:      text,
				Citations: blk.citations,
			})
		case "tool_use":
			rawArgs := strings.TrimSpace(blk.inputBuf.String())
			if rawArgs == "" && blk.initialInput != "" && blk.initialInput != "null" {
				rawArgs = blk.initialInput
			}
			args := provider.SanitizeToolArgsJSON([]byte(rawArgs))
			nativeMap := map[string]any{
				"type":  "tool_use",
				"id":    blk.id,
				"name":  blk.name,
				"input": provider.DecodeToolArgsObject([]byte(args)),
			}
			raw, _ := json.Marshal(nativeMap)
			rawNative = append(rawNative, raw)
			canonical = append(canonical, schema.ContentBlock{
				Type:      schema.BlockToolCall,
				CallID:    blk.id,
				Name:      blk.name,
				Arguments: json.RawMessage(args),
			})
		case "server_tool_use":
			rawArgs := strings.TrimSpace(blk.inputBuf.String())
			if rawArgs == "" && blk.initialInput != "" && blk.initialInput != "null" {
				rawArgs = blk.initialInput
			}
			args := provider.SanitizeToolArgsJSON([]byte(rawArgs))
			nativeMap := map[string]any{
				"type":  "server_tool_use",
				"id":    blk.id,
				"name":  blk.name,
				"input": provider.DecodeToolArgsObject([]byte(args)),
			}
			raw, _ := json.Marshal(nativeMap)
			rawNative = append(rawNative, raw)
			canonical = append(canonical, schema.ContentBlock{
				Type:           schema.BlockHostedToolCall,
				CallID:         blk.id,
				Name:           blk.name,
				Arguments:      json.RawMessage(args),
				ProviderOpaque: raw,
			})
		case "web_search_tool_result", "web_fetch_tool_result", "tool_search_tool_result", "tool_reference":
			if len(blk.rawServer) > 0 {
				rawNative = append(rawNative, blk.rawServer)
				canonical = append(canonical, schema.ContentBlock{
					Type:           schema.BlockHostedToolResult,
					CallID:         blk.id,
					Name:           blk.blockType,
					ProviderOpaque: blk.rawServer,
				})
			}
		}
	}

	a.usage.TotalTokens = a.usage.InputTokens + a.usage.OutputTokens
	return provider.Response{
		ID:         a.respID,
		Model:      a.model,
		Blocks:     canonical,
		Usage:      a.usage,
		StopReason: a.stopReason,
		Native: &schema.NativeEnvelope{
			Provider:   a.providerID,
			WireFormat: WireFormat,
			Model:      a.model,
			ResponseID: a.respID,
			RawItems:   rawNative,
		},
	}
}
