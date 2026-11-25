package gemini

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
)

type geminiContent struct {
	Role  string           `json:"role"`
	Parts []map[string]any `json:"parts"`
}

func isGemini4Model(model string) bool {
	norm := strings.ToLower(strings.TrimPrefix(strings.TrimSpace(model), "google/"))
	return strings.HasPrefix(norm, "gemini-4")
}

// mapGeminiThinkingLevel enforces Gemini 4 API constraints:
//   - Gemini 4 returns HTTP 400 on thinkingLevel="MINIMAL"; clamp none/minimal -> "LOW".
//   - Gemini 4 Neon/Hydrogen support "XHIGH" for xhigh/max; earlier models cap at "HIGH".
func mapGeminiThinkingLevel(model, effort string) string {
	norm := strings.ToLower(strings.TrimSpace(effort))
	gem4 := isGemini4Model(model)
	switch norm {
	case "none", "minimal":
		if gem4 {
			return "LOW"
		}
		return "MINIMAL"
	case "low":
		return "LOW"
	case "medium":
		return "MEDIUM"
	case "high":
		return "HIGH"
	case "xhigh", "max":
		if gem4 {
			return "XHIGH"
		}
		return "HIGH"
	default:
		return ""
	}
}

// buildGeminiContents constructs Gemini generateContent `contents[]`.
// On same-provider turns, NativeEnvelope replays original `parts` verbatim
// (preserving thought, text, server-side toolCall/toolResponse, and exact
// thoughtSignature fields). On canonical fallback replay, thoughtSignature is
// placed ONLY on the first parallel functionCall part (duplicating onto sibling
// functionCall parts returns HTTP 400 on Gemini 4) and on text parts.
func buildGeminiContents(msgs []schema.Message, wireFormat string) []any {
	out := make([]any, 0, len(msgs))
	callNames := make(map[string]string)

	for _, m := range msgs {
		for _, b := range m.Blocks {
			if b.Type == schema.BlockToolCall && b.CallID != "" && b.Name != "" {
				callNames[b.CallID] = b.Name
			}
		}
		if m.Native != nil && m.Native.WireFormat == wireFormat && len(m.Native.RawItems) > 0 {
			for _, raw := range m.Native.RawItems {
				out = append(out, raw)
			}
			continue
		}

		role := "user"
		if m.Role == schema.RoleAssistant {
			role = "model"
		}
		var parts []map[string]any
		attachedFirstFuncSig := false

		for _, b := range m.Blocks {
			switch b.Type {
			case schema.BlockCompaction:
				if b.Text != "" {
					parts = append(parts, map[string]any{"text": b.Text})
				}
			case schema.BlockText:
				if b.Text == "" {
					continue
				}
				part := map[string]any{"text": b.Text}
				if b.ThoughtSignBlob != "" {
					part["thoughtSignature"] = b.ThoughtSignBlob
				}
				parts = append(parts, part)
			case schema.BlockImage:
				if b.Image != nil && b.Image.Base64 != "" {
					mt := b.Image.MediaType
					if mt == "" {
						mt = "image/png"
					}
					parts = append(parts, map[string]any{
						"inlineData": map[string]any{
							"mimeType": mt,
							"data":     b.Image.Base64,
						},
					})
				}
			case schema.BlockToolCall:
				fc := map[string]any{
					"name": b.Name,
					"args": provider.DecodeToolArgsObject(b.Arguments),
				}
				if b.CallID != "" {
					fc["id"] = b.CallID
				}
				part := map[string]any{"functionCall": fc}
				if b.ThoughtSignBlob != "" && !attachedFirstFuncSig {
					part["thoughtSignature"] = b.ThoughtSignBlob
					attachedFirstFuncSig = true
				}
				parts = append(parts, part)
			case schema.BlockToolResult:
				name := b.Name
				if name == "" {
					name = callNames[b.CallID]
				}
				if name == "" {
					name = "tool"
				}
				fr := map[string]any{
					"name":     name,
					"response": map[string]any{"result": b.Output, "is_error": b.IsError},
				}
				if b.CallID != "" {
					fr["id"] = b.CallID
				}
				parts = append(parts, map[string]any{"functionResponse": fr})
			case schema.BlockHostedToolCall, schema.BlockHostedToolResult:
				if len(b.ProviderOpaque) > 0 && m.Native != nil && m.Native.WireFormat == wireFormat {
					var obj map[string]any
					if json.Unmarshal(b.ProviderOpaque, &obj) == nil && obj != nil {
						parts = append(parts, obj)
					}
				}
			}
		}
		if len(parts) > 0 {
			if len(out) > 0 {
				if prev, ok := out[len(out)-1].(geminiContent); ok && prev.Role == role {
					prev.Parts = append(prev.Parts, parts...)
					out[len(out)-1] = prev
					continue
				}
			}
			out = append(out, geminiContent{Role: role, Parts: parts})
		}
	}
	return out
}

func buildGeminiTools(defs []schema.ToolDef, hosted []schema.HostedTool) ([]map[string]any, bool, bool) {
	var tools []map[string]any
	hasFunctions := false
	hasHosted := false

	if len(defs) > 0 {
		hasFunctions = true
		decls := make([]map[string]any, 0, len(defs))
		for _, t := range defs {
			decl := map[string]any{
				"name": t.Name,
			}
			if t.Description != "" {
				decl["description"] = t.Description
			}
			if len(bytes.TrimSpace(t.InputSchema)) > 0 {
				var obj map[string]any
				if json.Unmarshal(t.InputSchema, &obj) == nil && obj != nil {
					cleanGeminiSchema(obj)
					decl["parameters"] = obj
				}
			}
			decls = append(decls, decl)
		}
		tools = append(tools, map[string]any{"functionDeclarations": decls})
	}
	for _, h := range hosted {
		hasHosted = true
		switch h.Type {
		case "google_search", "web_search":
			cfg := h.Config
			if cfg == nil {
				cfg = map[string]any{}
			}
			tools = append(tools, map[string]any{"googleSearch": cfg})
		case "url_context", "web_fetch":
			cfg := h.Config
			if cfg == nil {
				cfg = map[string]any{}
			}
			tools = append(tools, map[string]any{"urlContext": cfg})
		default:
			cfg := h.Config
			if cfg == nil {
				cfg = map[string]any{}
			}
			tools = append(tools, map[string]any{h.Type: cfg})
		}
	}
	return tools, hasFunctions, hasHosted
}

func cleanGeminiSchema(obj map[string]any) {
	delete(obj, "additionalProperties")
	delete(obj, "$schema")
	delete(obj, "const")
	if props, ok := obj["properties"].(map[string]any); ok {
		for _, v := range props {
			if child, ok := v.(map[string]any); ok {
				cleanGeminiSchema(child)
			}
		}
	}
	if items, ok := obj["items"].(map[string]any); ok {
		cleanGeminiSchema(items)
	}
}

// buildGeminiToolConfig sets functionCallingConfig and enables
// includeServerSideToolInvocations=true whenever built-in server tools
// (googleSearch / urlContext) are combined with custom functionDeclarations.
func buildGeminiToolConfig(tc *schema.ToolChoice, combineServerAndFunctions bool) map[string]any {
	out := map[string]any{}
	if combineServerAndFunctions {
		out["includeServerSideToolInvocations"] = true
	}
	if tc != nil && tc.Mode != "" {
		cfg := map[string]any{}
		switch tc.Mode {
		case "auto":
			cfg["mode"] = "AUTO"
		case "required":
			cfg["mode"] = "ANY"
		case "none":
			cfg["mode"] = "NONE"
		case "tool":
			cfg["mode"] = "ANY"
			if tc.Name != "" {
				cfg["allowedFunctionNames"] = []string{tc.Name}
			}
		}
		if len(cfg) > 0 {
			out["functionCallingConfig"] = cfg
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
