package anthropic

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
)

type anthropicMessage struct {
	Role    string           `json:"role"`
	Content []map[string]any `json:"content"`
}

func buildAnthropicMessages(msgs []schema.Message, wireFormat string) []anthropicMessage {
	var out []anthropicMessage
	appendBlocks := func(role string, blocks []map[string]any) {
		if len(blocks) == 0 {
			return
		}
		if len(out) > 0 && out[len(out)-1].Role == role {
			out[len(out)-1].Content = append(out[len(out)-1].Content, blocks...)
			return
		}
		out = append(out, anthropicMessage{Role: role, Content: blocks})
	}

	for _, m := range msgs {
		if m.Native != nil && m.Native.WireFormat == wireFormat && len(m.Native.RawItems) > 0 {
			var nativeBlocks []map[string]any
			for _, raw := range m.Native.RawItems {
				var obj map[string]any
				if json.Unmarshal(raw, &obj) == nil && obj != nil {
					nativeBlocks = append(nativeBlocks, obj)
				}
			}
			role := "assistant"
			if m.Role == schema.RoleUser || m.Role == schema.RoleTool {
				role = "user"
			}
			appendBlocks(role, nativeBlocks)
			continue
		}

		for _, b := range m.Blocks {
			switch b.Type {
			case schema.BlockCompaction:
				if len(b.ProviderOpaque) > 0 && m.Native != nil && m.Native.WireFormat == wireFormat {
					var obj map[string]any
					if json.Unmarshal(b.ProviderOpaque, &obj) == nil && obj != nil {
						appendBlocks("assistant", []map[string]any{obj})
						continue
					}
				}
				if b.Text != "" {
					appendBlocks("user", []map[string]any{{"type": "text", "text": b.Text}})
				}
			case schema.BlockText:
				if b.Text == "" {
					continue
				}
				role := "user"
				if m.Role == schema.RoleAssistant {
					role = "assistant"
				}
				appendBlocks(role, []map[string]any{{"type": "text", "text": b.Text}})
			case schema.BlockImage:
				if b.Image == nil {
					continue
				}
				var src map[string]any
				if b.Image.Base64 != "" {
					mt := b.Image.MediaType
					if mt == "" {
						mt = "image/png"
					}
					src = map[string]any{"type": "base64", "media_type": mt, "data": b.Image.Base64}
				} else if b.Image.URL != "" {
					src = map[string]any{"type": "url", "url": b.Image.URL}
				}
				if src != nil {
					appendBlocks("user", []map[string]any{{"type": "image", "source": src}})
				}
			case schema.BlockThinking:
				// Only replay thinking blocks carrying a genuine Anthropic signature.
				if b.Signature != "" && m.Native != nil && m.Native.WireFormat == wireFormat {
					appendBlocks("assistant", []map[string]any{
						{"type": "thinking", "thinking": b.Thinking, "signature": b.Signature},
					})
				}
			case schema.BlockRedactedThinking:
				if b.RedactedData != "" {
					appendBlocks("assistant", []map[string]any{
						{"type": "redacted_thinking", "data": b.RedactedData},
					})
				}
			case schema.BlockToolCall:
				appendBlocks("assistant", []map[string]any{
					{
						"type":  "tool_use",
						"id":    b.CallID,
						"name":  b.Name,
						"input": provider.DecodeToolArgsObject(b.Arguments),
					},
				})
			case schema.BlockToolResult:
				blk := map[string]any{
					"type":        "tool_result",
					"tool_use_id": b.CallID,
					"content":     b.Output,
				}
				if b.IsError {
					blk["is_error"] = true
				}
				appendBlocks("user", []map[string]any{blk})
			case schema.BlockHostedToolCall, schema.BlockHostedToolResult:
				if len(b.ProviderOpaque) > 0 && m.Native != nil && m.Native.WireFormat == wireFormat {
					var obj map[string]any
					if json.Unmarshal(b.ProviderOpaque, &obj) == nil && obj != nil {
						appendBlocks("assistant", []map[string]any{obj})
					}
				}
			}
		}
	}
	if len(out) > 0 && out[0].Role == "assistant" {
		out = append([]anthropicMessage{{
			Role:    "user",
			Content: []map[string]any{{"type": "text", "text": "[Resumed from compaction checkpoint]"}},
		}}, out...)
	}
	return out
}

// applyConversationCacheBreakpoints sets ephemeral cache_control on up to
// budget (capped at 2) stable user-turn boundaries so total cache_control
// markers never exceed Anthropic's 4-breakpoint limit.
func applyConversationCacheBreakpoints(messages []anthropicMessage, budget int) {
	if budget <= 0 {
		return
	}
	if budget > 2 {
		budget = 2
	}
	var userIndices []int
	for i, m := range messages {
		if m.Role == "user" && len(m.Content) > 0 {
			userIndices = append(userIndices, i)
		}
	}
	if len(userIndices) == 0 {
		return
	}
	if budget >= 2 && len(userIndices) >= 2 {
		idx := userIndices[len(userIndices)-2]
		blocks := messages[idx].Content
		blocks[len(blocks)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
	}
	idx := userIndices[len(userIndices)-1]
	blocks := messages[idx].Content
	blocks[len(blocks)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
}

func buildAnthropicTools(defs []schema.ToolDef, hosted []schema.HostedTool, nativeToolSearch bool) ([]map[string]any, bool, bool) {
	out := make([]map[string]any, 0, len(defs)+len(hosted)+1)
	hasWebFetch := false
	hasDeferred := false

	for _, t := range defs {
		schemaObj := map[string]any{"type": "object"}
		if len(bytes.TrimSpace(t.InputSchema)) > 0 {
			var parsed map[string]any
			if json.Unmarshal(t.InputSchema, &parsed) == nil && parsed != nil {
				if _, ok := parsed["type"]; !ok {
					parsed["type"] = "object"
				}
				schemaObj = parsed
			}
		}
		item := map[string]any{
			"name":         t.Name,
			"input_schema": schemaObj,
		}
		if t.Description != "" {
			item["description"] = t.Description
		}
		if nativeToolSearch && t.DeferLoading && !t.AlwaysLoad {
			item["defer_loading"] = true
			hasDeferred = true
		}
		out = append(out, item)
	}
	if hasDeferred {
		out = append([]map[string]any{{
			"type": "tool_search_tool_bm25_20251119",
			"name": "tool_search_tool_bm25",
		}}, out...)
	}

	for _, h := range hosted {
		switch h.Type {
		case "web_search", "web_search_20250305":
			obj := map[string]any{"type": "web_search_20250305", "name": "web_search"}
			for k, v := range h.Config {
				obj[k] = v
			}
			out = append(out, obj)
		case "web_fetch", "web_fetch_20250910":
			hasWebFetch = true
			obj := map[string]any{"type": "web_fetch_20250910", "name": "web_fetch"}
			for k, v := range h.Config {
				obj[k] = v
			}
			out = append(out, obj)
		default:
			obj := map[string]any{"type": h.Type}
			if h.Name != "" {
				obj["name"] = h.Name
			}
			for k, v := range h.Config {
				obj[k] = v
			}
			out = append(out, obj)
		}
	}
	return out, hasWebFetch, hasDeferred
}

func buildAnthropicToolChoice(tc *schema.ToolChoice) map[string]any {
	if tc == nil || tc.Mode == "" {
		return nil
	}
	switch tc.Mode {
	case "auto":
		return map[string]any{"type": "auto"}
	case "required":
		return map[string]any{"type": "any"}
	case "none":
		return map[string]any{"type": "none"}
	case "tool":
		if tc.Name != "" {
			return map[string]any{"type": "tool", "name": tc.Name}
		}
	}
	return nil
}

func mapThinkingBudget(effort string) int {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "low":
		return 4096
	case "medium":
		return 10000
	case "high":
		return 16000
	case "xhigh", "max":
		return 32000
	default:
		return 0
	}
}
