package anthropic

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
)

func readAnthropicSSE(r io.Reader, onEvent func([]byte) error) error {
	return provider.ScanLines(r, provider.DefaultMaxLineBytes, func(rawLine string) error {
		line := strings.TrimSpace(rawLine)
		if !strings.HasPrefix(line, "data:") {
			return nil
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			return nil
		}
		return onEvent([]byte(data))
	})
}

type anthropicUsagePayload struct {
	InputTokens              int64 `json:"input_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	Iterations               []struct {
		Type                     string `json:"type"`
		InputTokens              int64  `json:"input_tokens"`
		CacheCreationInputTokens int64  `json:"cache_creation_input_tokens"`
		CacheReadInputTokens     int64  `json:"cache_read_input_tokens"`
		OutputTokens             int64  `json:"output_tokens"`
	} `json:"iterations"`
}

func applyAnthropicUsage(dst *schema.Usage, u *anthropicUsagePayload, isDelta bool) {
	if u == nil {
		return
	}
	if len(u.Iterations) > 0 {
		var inTok, cacheWrite, cacheRead, outTok int64
		for _, it := range u.Iterations {
			inTok += it.InputTokens + it.CacheCreationInputTokens + it.CacheReadInputTokens
			cacheWrite += it.CacheCreationInputTokens
			cacheRead += it.CacheReadInputTokens
			outTok += it.OutputTokens
		}
		dst.InputTokens = inTok
		dst.CacheCreationTokens = cacheWrite
		dst.CachedInputTokens = cacheRead
		dst.OutputTokens = outTok
		return
	}
	if !isDelta {
		dst.InputTokens = u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens
		dst.CacheCreationTokens = u.CacheCreationInputTokens
		dst.CachedInputTokens = u.CacheReadInputTokens
		dst.OutputTokens = u.OutputTokens
		return
	}
	if u.OutputTokens > 0 {
		dst.OutputTokens = u.OutputTokens
	}
	if u.InputTokens > 0 || u.CacheCreationInputTokens > 0 || u.CacheReadInputTokens > 0 {
		dst.InputTokens = u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens
	}
	if u.CacheReadInputTokens > 0 {
		dst.CachedInputTokens = u.CacheReadInputTokens
	}
	if u.CacheCreationInputTokens > 0 {
		dst.CacheCreationTokens = u.CacheCreationInputTokens
	}
}

type blockAcc struct {
	blockType      string
	id             string
	name           string
	initialInput   string
	textBuf        strings.Builder
	thinkingBuf    strings.Builder
	sigBuf         strings.Builder
	inputBuf       strings.Builder
	compactionBuf  strings.Builder
	compactionNull bool
	redactedData   string
	rawServer      json.RawMessage
	citations      []schema.Citation
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
			ID    string                 `json:"id"`
			Model string                 `json:"model"`
			Usage *anthropicUsagePayload `json:"usage"`
		} `json:"message"`
		ContentBlock json.RawMessage `json:"content_block"`
		Delta        *struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			Thinking    string `json:"thinking"`
			Signature   string `json:"signature"`
			PartialJSON string `json:"partial_json"`
			Content     string `json:"content"`
			StopReason  string `json:"stop_reason"`
			Citation    *struct {
				Type      string `json:"type"`
				URL       string `json:"url"`
				Title     string `json:"title"`
				CitedText string `json:"cited_text"`
			} `json:"citation"`
		} `json:"delta"`
		Usage *anthropicUsagePayload `json:"usage"`
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
			applyAnthropicUsage(&a.usage, ev.Message.Usage, false)
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
			Content   json.RawMessage `json:"content"`
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
		if cb.Type == "compaction" && len(cb.Content) > 0 {
			trimmed := strings.TrimSpace(string(cb.Content))
			if trimmed == "null" {
				blk.compactionNull = true
			} else {
				var s string
				if json.Unmarshal(cb.Content, &s) == nil && s != "" {
					blk.compactionBuf.WriteString(s)
				}
			}
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
		case "compaction_delta":
			if ev.Delta.Content != "" {
				blk.compactionNull = false
				blk.compactionBuf.WriteString(ev.Delta.Content)
				if a.emit != nil {
					a.emit(provider.StreamChunk{Kind: provider.DeltaCompaction, Text: ev.Delta.Content})
				}
			}
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
		applyAnthropicUsage(&a.usage, ev.Usage, true)
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
		case "compaction":
			content := blk.compactionBuf.String()
			var nativeContent any = content
			if content == "" && blk.compactionNull {
				nativeContent = nil
			}
			nativeMap := map[string]any{
				"type":    "compaction",
				"content": nativeContent,
			}
			raw, _ := json.Marshal(nativeMap)
			rawNative = append(rawNative, raw)
			canonical = append(canonical, schema.ContentBlock{
				Type:           schema.BlockCompaction,
				Text:           content,
				ProviderOpaque: raw,
			})
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
