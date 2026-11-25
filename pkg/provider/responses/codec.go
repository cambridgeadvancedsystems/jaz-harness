package responses

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
)

// Dialect parameterizes the shared OpenAI / xAI Responses API wire codec
// without forcing lossy translation onto non-Responses providers.
type Dialect struct {
	ProviderID             string
	WireFormat             string
	StripExternalWebAccess bool
	IsHostedCall           func(callID, name string) bool
}

type UsagePayload struct {
	InputTokens        int64 `json:"input_tokens"`
	OutputTokens       int64 `json:"output_tokens"`
	TotalTokens        int64 `json:"total_tokens"`
	InputTokensDetails *struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails *struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

func (u UsagePayload) ToSchema() schema.Usage {
	out := schema.Usage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
		TotalTokens:  u.TotalTokens,
	}
	if out.TotalTokens == 0 {
		out.TotalTokens = out.InputTokens + out.OutputTokens
	}
	if u.InputTokensDetails != nil {
		out.CachedInputTokens = u.InputTokensDetails.CachedTokens
	}
	if u.OutputTokensDetails != nil {
		out.ReasoningTokens = u.OutputTokensDetails.ReasoningTokens
	}
	return out
}

func BuildInputItems(msgs []schema.Message, wireFormat string) []any {
	out := make([]any, 0, len(msgs)*2)
	for _, m := range msgs {
		if m.Native != nil && m.Native.WireFormat == wireFormat && len(m.Native.RawItems) > 0 {
			for _, raw := range m.Native.RawItems {
				out = append(out, raw)
			}
			continue
		}
		for _, b := range m.Blocks {
			switch b.Type {
			case schema.BlockCompaction:
				if len(b.ProviderOpaque) > 0 && m.Native != nil && m.Native.WireFormat == wireFormat {
					out = append(out, b.ProviderOpaque)
				} else if b.Text != "" {
					out = append(out, map[string]any{
						"type":    "message",
						"role":    "developer",
						"content": []map[string]any{{"type": "input_text", "text": b.Text}},
					})
				}
			case schema.BlockText:
				if b.Text == "" {
					continue
				}
				partType := "input_text"
				if m.Role == schema.RoleAssistant {
					partType = "output_text"
				}
				out = append(out, map[string]any{
					"type":    "message",
					"role":    string(m.Role),
					"content": []map[string]any{{"type": partType, "text": b.Text}},
				})
			case schema.BlockImage:
				if b.Image == nil {
					continue
				}
				imgURL := b.Image.URL
				if imgURL == "" && b.Image.Base64 != "" {
					mt := b.Image.MediaType
					if mt == "" {
						mt = "image/png"
					}
					imgURL = "data:" + mt + ";base64," + b.Image.Base64
				}
				if imgURL != "" {
					out = append(out, map[string]any{
						"type": "message",
						"role": "user",
						"content": []map[string]any{
							{"type": "input_image", "image_url": imgURL},
						},
					})
				}
			case schema.BlockToolCall:
				args := strings.TrimSpace(string(b.Arguments))
				if args == "" {
					args = "{}"
				}
				out = append(out, map[string]any{
					"type":      "function_call",
					"call_id":   b.CallID,
					"name":      b.Name,
					"arguments": args,
				})
			case schema.BlockToolResult:
				out = append(out, map[string]any{
					"type":    "function_call_output",
					"call_id": b.CallID,
					"output":  b.Output,
				})
			}
		}
	}
	return out
}

func BuildTools(defs []schema.ToolDef, hosted []schema.HostedTool, stripExternalWebAccess bool) []any {
	out := make([]any, 0, len(defs)+len(hosted))
	for _, t := range defs {
		params := t.InputSchema
		if len(bytes.TrimSpace(params)) == 0 {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		item := map[string]any{
			"type":       "function",
			"name":       t.Name,
			"parameters": params,
		}
		if t.Description != "" {
			item["description"] = t.Description
		}
		out = append(out, item)
	}
	for _, h := range hosted {
		obj := map[string]any{"type": h.Type}
		for k, v := range h.Config {
			if stripExternalWebAccess && k == "external_web_access" {
				continue
			}
			obj[k] = v
		}
		out = append(out, obj)
	}
	return out
}

func BuildToolChoice(tc *schema.ToolChoice) any {
	if tc == nil || tc.Mode == "" {
		return nil
	}
	switch tc.Mode {
	case "auto", "required", "none":
		return tc.Mode
	case "tool":
		if tc.Name != "" {
			return map[string]any{"type": "function", "name": tc.Name}
		}
	}
	return nil
}

func ParseSSE(r io.Reader, d Dialect, fallbackModel string, emit provider.StreamCallback) (provider.Response, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)

	var (
		respID      string
		model       = fallbackModel
		usage       schema.Usage
		stopReason  = "completed"
		rawItems    []json.RawMessage
		finalBlocks []schema.ContentBlock
		sawDone     bool
		eventName   string
	)

	isHosted := func(callID, name string) bool {
		return d.IsHostedCall != nil && d.IsHostedCall(callID, name)
	}

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			eventName = ""
			continue
		}
		if strings.HasPrefix(line, "event:") {
			eventName = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}

		var ev struct {
			Type     string          `json:"type"`
			Delta    string          `json:"delta"`
			CallID   string          `json:"call_id"`
			Item     json.RawMessage `json:"item"`
			Response *struct {
				ID     string            `json:"id"`
				Model  string            `json:"model"`
				Status string            `json:"status"`
				Output []json.RawMessage `json:"output"`
				Usage  *UsagePayload     `json:"usage"`
				Error  *struct {
					Message string `json:"message"`
				} `json:"error"`
			} `json:"response"`
		}
		if json.Unmarshal([]byte(data), &ev) != nil {
			continue
		}
		evType := ev.Type
		if evType == "" {
			evType = eventName
		}

		switch evType {
		case "response.created":
			if ev.Response != nil {
				respID = ev.Response.ID
				if ev.Response.Model != "" {
					model = ev.Response.Model
				}
			}
		case "response.output_text.delta":
			if emit != nil && ev.Delta != "" {
				emit(provider.StreamChunk{Kind: provider.DeltaText, Text: ev.Delta})
			}
		case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
			if emit != nil && ev.Delta != "" {
				emit(provider.StreamChunk{Kind: provider.DeltaThinking, Text: ev.Delta})
			}
		case "response.output_item.added":
			if emit != nil && len(ev.Item) > 0 {
				var meta struct {
					Type   string `json:"type"`
					ID     string `json:"id"`
					CallID string `json:"call_id"`
					Name   string `json:"name"`
				}
				if json.Unmarshal(ev.Item, &meta) == nil {
					if meta.Type == "function_call" && !isHosted(meta.CallID, meta.Name) {
						emit(provider.StreamChunk{Kind: provider.DeltaToolCallStart, CallID: meta.CallID, ToolName: meta.Name})
					} else if meta.Type == "web_search_call" || meta.Type == "custom_tool_call" || isHosted(meta.CallID, meta.Name) {
						cid := meta.CallID
						if cid == "" {
							cid = meta.ID
						}
						name := meta.Name
						if name == "" {
							name = meta.Type
						}
						emit(provider.StreamChunk{Kind: provider.DeltaHostedToolStart, CallID: cid, ToolName: name})
					}
				}
			}
		case "response.function_call_arguments.delta":
			if emit != nil && ev.Delta != "" {
				emit(provider.StreamChunk{Kind: provider.DeltaToolCallArgs, CallID: ev.CallID, Arguments: ev.Delta})
			}
		case "response.output_item.done":
			if len(ev.Item) > 0 {
				cp := append(json.RawMessage(nil), ev.Item...)
				rawItems = append(rawItems, cp)
				finalBlocks = append(finalBlocks, decodeOutputItem(cp, isHosted)...)
			}
		case "response.completed", "response.incomplete":
			sawDone = true
			if ev.Response != nil {
				if ev.Response.ID != "" {
					respID = ev.Response.ID
				}
				if ev.Response.Model != "" {
					model = ev.Response.Model
				}
				if ev.Response.Status != "" {
					stopReason = ev.Response.Status
				}
				if ev.Response.Usage != nil {
					usage = ev.Response.Usage.ToSchema()
				}
				if len(rawItems) == 0 && len(ev.Response.Output) > 0 {
					for _, item := range ev.Response.Output {
						cp := append(json.RawMessage(nil), item...)
						rawItems = append(rawItems, cp)
						finalBlocks = append(finalBlocks, decodeOutputItem(cp, isHosted)...)
					}
				}
			}
		case "response.failed":
			msg := d.ProviderID + " response failed"
			if ev.Response != nil && ev.Response.Error != nil && ev.Response.Error.Message != "" {
				msg = ev.Response.Error.Message
			}
			return provider.Response{}, errors.New(msg)
		}
	}
	if err := sc.Err(); err != nil {
		return provider.Response{}, err
	}
	if !sawDone && len(rawItems) == 0 {
		return provider.Response{}, errors.New(d.ProviderID + " stream ended before response.completed")
	}

	return provider.Response{
		ID:         respID,
		Model:      model,
		Blocks:     finalBlocks,
		Usage:      usage,
		StopReason: stopReason,
		Native: &schema.NativeEnvelope{
			Provider:   d.ProviderID,
			WireFormat: d.WireFormat,
			Model:      model,
			ResponseID: respID,
			RawItems:   rawItems,
		},
	}, nil
}

func decodeOutputItem(raw json.RawMessage, isHosted func(callID, name string) bool) []schema.ContentBlock {
	var item struct {
		Type             string `json:"type"`
		ID               string `json:"id"`
		CallID           string `json:"call_id"`
		Name             string `json:"name"`
		Arguments        string `json:"arguments"`
		Input            string `json:"input"`
		EncryptedContent string `json:"encrypted_content"`
		Summary          []struct {
			Text string `json:"text"`
		} `json:"summary"`
		Content []struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			Annotations []struct {
				Type  string `json:"type"`
				URL   string `json:"url"`
				Title string `json:"title"`
			} `json:"annotations"`
		} `json:"content"`
	}
	if json.Unmarshal(raw, &item) != nil {
		return nil
	}

	switch item.Type {
	case "message":
		var out []schema.ContentBlock
		for _, p := range item.Content {
			if p.Type == "output_text" || p.Type == "text" {
				var cits []schema.Citation
				for _, ann := range p.Annotations {
					if ann.URL != "" {
						cits = append(cits, schema.Citation{
							Type:  ann.Type,
							URL:   ann.URL,
							Title: ann.Title,
						})
					}
				}
				out = append(out, schema.ContentBlock{
					Type:      schema.BlockText,
					Text:      p.Text,
					Citations: cits,
				})
			}
		}
		return out
	case "reasoning":
		var parts []string
		for _, s := range item.Summary {
			if s.Text != "" {
				parts = append(parts, s.Text)
			}
		}
		return []schema.ContentBlock{{
			Type:           schema.BlockThinking,
			Thinking:       strings.Join(parts, "\n"),
			Signature:      item.EncryptedContent,
			ProviderOpaque: append(json.RawMessage(nil), raw...),
		}}
	case "function_call":
		if isHosted(item.CallID, item.Name) {
			return []schema.ContentBlock{{
				Type:           schema.BlockHostedToolCall,
				CallID:         item.CallID,
				Name:           item.Name,
				ProviderOpaque: append(json.RawMessage(nil), raw...),
			}}
		}
		args := strings.TrimSpace(item.Arguments)
		if args == "" {
			args = "{}"
		}
		return []schema.ContentBlock{{
			Type:      schema.BlockToolCall,
			CallID:    item.CallID,
			Name:      item.Name,
			Arguments: json.RawMessage(args),
		}}
	case "web_search_call", "custom_tool_call":
		cid := item.CallID
		if cid == "" {
			cid = item.ID
		}
		name := item.Name
		if name == "" {
			name = item.Type
		}
		return []schema.ContentBlock{{
			Type:           schema.BlockHostedToolCall,
			CallID:         cid,
			Name:           name,
			ProviderOpaque: append(json.RawMessage(nil), raw...),
		}}
	case "compaction":
		return []schema.ContentBlock{{
			Type:           schema.BlockCompaction,
			ProviderOpaque: append(json.RawMessage(nil), raw...),
		}}
	}
	return nil
}
