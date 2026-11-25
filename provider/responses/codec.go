package responses

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
)

// Dialect parameterizes the shared OpenAI / xAI Responses API wire codec
// without forcing lossy translation onto non-Responses providers.
type Dialect struct {
	ProviderID               string
	WireFormat               string
	StripExternalWebAccess   bool
	AllowImageGenerationTool bool
	AllowXSearchTool         bool
	AllowOriginalDetail      bool
	IsHostedCall             func(callID, name string) bool
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

// RawItemType extracts the top-level "type" field from a raw Responses API item.
func RawItemType(raw json.RawMessage) string {
	var meta struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &meta) != nil {
		return ""
	}
	return meta.Type
}

func BuildInputItems(msgs []schema.Message, wireFormat string) []any {
	return BuildInputItemsWithDialect(msgs, Dialect{
		WireFormat:          wireFormat,
		AllowOriginalDetail: wireFormat == "openai_responses",
	})
}

func BuildInputItemsWithDialect(msgs []schema.Message, d Dialect) []any {
	out := make([]any, 0, len(msgs)*2)
	for _, m := range msgs {
		if m.Native != nil && m.Native.WireFormat == d.WireFormat && len(m.Native.RawItems) > 0 {
			for _, raw := range m.Native.RawItems {
				out = append(out, raw)
			}
			continue
		}

		var pendingUserParts []map[string]any
		flushUserParts := func() {
			if len(pendingUserParts) == 0 {
				return
			}
			out = append(out, map[string]any{
				"type":    "message",
				"role":    "user",
				"content": pendingUserParts,
			})
			pendingUserParts = nil
		}

		for _, b := range m.Blocks {
			switch b.Type {
			case schema.BlockCompaction:
				flushUserParts()
				if len(b.ProviderOpaque) > 0 && m.Native != nil && m.Native.WireFormat == d.WireFormat {
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
				if m.Role == schema.RoleUser {
					pendingUserParts = append(pendingUserParts, map[string]any{
						"type": "input_text",
						"text": b.Text,
					})
					continue
				}
				flushUserParts()
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
				// Preserve native OpenAI image_generation_call items verbatim for
				// multi-turn iterative image editing continuity.
				if len(b.ProviderOpaque) > 0 && d.WireFormat == "openai_responses" && RawItemType(b.ProviderOpaque) == "image_generation_call" {
					flushUserParts()
					out = append(out, b.ProviderOpaque)
					continue
				}
				if b.Image == nil {
					continue
				}
				part := buildInputImagePart(*b.Image, d.AllowOriginalDetail)
				if part == nil {
					continue
				}
				if b.Text != "" && m.Role == schema.RoleAssistant {
					pendingUserParts = append(pendingUserParts, map[string]any{
						"type": "input_text",
						"text": "[Generated image prompt: " + b.Text + "]",
					})
				}
				pendingUserParts = append(pendingUserParts, part)
			case schema.BlockToolCall:
				flushUserParts()
				out = append(out, map[string]any{
					"type":      "function_call",
					"call_id":   b.CallID,
					"name":      b.Name,
					"arguments": provider.SanitizeToolArgsJSON(b.Arguments),
				})
			case schema.BlockToolResult:
				flushUserParts()
				out = append(out, buildFunctionCallOutputItem(b, d.AllowOriginalDetail))
			}
		}
		flushUserParts()
	}
	return out
}

func buildInputImagePart(img schema.ImageSource, allowOriginal bool) map[string]any {
	imgURL := img.DataURL()
	if imgURL == "" {
		return nil
	}
	part := map[string]any{
		"type":      "input_image",
		"image_url": imgURL,
	}
	if detail := normalizeImageDetail(img.Detail, allowOriginal); detail != "" {
		part["detail"] = detail
	}
	return part
}

func normalizeImageDetail(detail string, allowOriginal bool) string {
	switch strings.ToLower(strings.TrimSpace(detail)) {
	case "low", "high", "auto":
		return strings.ToLower(strings.TrimSpace(detail))
	case "original":
		if allowOriginal {
			return "original"
		}
		return "high"
	default:
		return ""
	}
}

func buildFunctionCallOutputItem(b schema.ContentBlock, allowOriginal bool) map[string]any {
	if len(b.Images) == 0 {
		return map[string]any{
			"type":    "function_call_output",
			"call_id": b.CallID,
			"output":  b.Output,
		}
	}
	items := make([]map[string]any, 0, len(b.Images)+1)
	for _, img := range b.Images {
		if part := buildInputImagePart(img, allowOriginal); part != nil {
			items = append(items, part)
		}
	}
	if b.Output != "" {
		items = append(items, map[string]any{
			"type": "input_text",
			"text": b.Output,
		})
	}
	if len(items) == 0 {
		return map[string]any{
			"type":    "function_call_output",
			"call_id": b.CallID,
			"output":  b.Output,
		}
	}
	return map[string]any{
		"type":    "function_call_output",
		"call_id": b.CallID,
		"output":  items,
	}
}

func BuildTools(defs []schema.ToolDef, hosted []schema.HostedTool, stripExternalWebAccess bool) []any {
	return BuildToolsForDialect(defs, hosted, Dialect{
		StripExternalWebAccess:   stripExternalWebAccess,
		AllowImageGenerationTool: !stripExternalWebAccess,
		AllowXSearchTool:         stripExternalWebAccess,
	})
}

func BuildToolsForDialect(defs []schema.ToolDef, hosted []schema.HostedTool, d Dialect) []any {
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
	seenHosted := map[string]bool{}
	for _, h := range hosted {
		toolType := normalizeHostedToolType(h.Type, d)
		if toolType == "" {
			continue
		}
		if (toolType == "web_search" || toolType == "x_search" || toolType == "image_generation") && seenHosted[toolType] {
			continue
		}
		seenHosted[toolType] = true
		obj := map[string]any{"type": toolType}
		for k, v := range h.Config {
			if shouldStripHostedConfigKey(toolType, k, d) {
				continue
			}
			obj[k] = v
		}
		out = append(out, obj)
	}
	return out
}

func normalizeHostedToolType(raw string, d Dialect) string {
	switch strings.TrimSpace(raw) {
	case "web_search", "web_search_20250305", "web_search_20260209", "google_search", "googleSearch":
		return "web_search"
	case "web_search_preview", "web_search_preview_2025_03_11":
		if d.StripExternalWebAccess {
			return "web_search"
		}
		return strings.TrimSpace(raw)
	case "x_search":
		if !d.AllowXSearchTool {
			return ""
		}
		return "x_search"
	case "image_generation":
		if !d.AllowImageGenerationTool {
			return ""
		}
		return "image_generation"
	case "url_context", "urlContext", "web_fetch", "web_fetch_20250910", "web_fetch_20260209":
		return ""
	default:
		return strings.TrimSpace(raw)
	}
}

func shouldStripHostedConfigKey(toolType, key string, d Dialect) bool {
	if toolType != "web_search" {
		return false
	}
	if d.StripExternalWebAccess {
		switch key {
		case "external_web_access", "search_context_size", "user_location":
			return true
		}
	}
	if !d.AllowXSearchTool {
		switch key {
		case "allowed_x_handles", "excluded_x_handles", "from_date", "to_date",
			"enable_image_understanding", "enable_video_understanding":
			return true
		}
	}
	return false
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
