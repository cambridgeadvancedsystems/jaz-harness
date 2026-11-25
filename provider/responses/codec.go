package responses

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
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
				if len(b.ProviderOpaque) > 0 && d.WireFormat == "openai_responses" && rawItemType(b.ProviderOpaque) == "image_generation_call" {
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
		if toolType == "web_search" && seenHosted[toolType] {
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
	if toolType == "web_search" {
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

func ParseSSE(r io.Reader, d Dialect, fallbackModel string, emit provider.StreamCallback) (provider.Response, error) {
	var (
		respID         string
		model          = fallbackModel
		usage          schema.Usage
		stopReason     = "completed"
		rawItems       []json.RawMessage
		finalBlocks    []schema.ContentBlock
		respCitations  []schema.Citation
		sawDone        bool
		eventName      string
		startedHosted  = map[string]bool{}
		emitHostedOnce = func(callID, name, detail string) {
			if emit == nil {
				return
			}
			key := callID + ":" + name
			if startedHosted[key] {
				return
			}
			startedHosted[key] = true
			emit(provider.StreamChunk{
				Kind:     provider.DeltaHostedToolStart,
				CallID:   callID,
				ToolName: name,
				Detail:   detail,
			})
		}
	)

	isHosted := func(callID, name string) bool {
		return d.IsHostedCall != nil && d.IsHostedCall(callID, name)
	}

	err := provider.ScanLines(r, provider.DefaultMaxLineBytes, func(rawLine string) error {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			eventName = ""
			return nil
		}
		if strings.HasPrefix(line, "event:") {
			eventName = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			return nil
		}
		if !strings.HasPrefix(line, "data:") {
			return nil
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			return nil
		}

		var ev struct {
			Type     string          `json:"type"`
			Delta    string          `json:"delta"`
			CallID   string          `json:"call_id"`
			ItemID   string          `json:"item_id"`
			Item     json.RawMessage `json:"item"`
			Response *struct {
				ID        string            `json:"id"`
				Model     string            `json:"model"`
				Status    string            `json:"status"`
				Output    []json.RawMessage `json:"output"`
				Citations []json.RawMessage `json:"citations"`
				Usage     *UsagePayload     `json:"usage"`
				Error     *struct {
					Message string `json:"message"`
				} `json:"error"`
			} `json:"response"`
		}
		if json.Unmarshal([]byte(data), &ev) != nil {
			return nil
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
		case "response.image_generation_call.in_progress", "response.image_generation_call.generating":
			cid := ev.ItemID
			if cid == "" {
				cid = ev.CallID
			}
			emitHostedOnce(cid, "image_generation", "")
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
					} else if isHostedOutputItemType(meta.Type) || isHosted(meta.CallID, meta.Name) {
						cid := firstNonEmpty(meta.CallID, meta.ID)
						name := hostedDisplayName(meta.Type, meta.Name)
						emitHostedOnce(cid, name, "")
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
				decoded, hostedDone := decodeOutputItemWithActivity(cp, isHosted)
				finalBlocks = append(finalBlocks, decoded...)
				if emit != nil && hostedDone != nil {
					emit(provider.StreamChunk{
						Kind:     provider.DeltaHostedToolComplete,
						CallID:   hostedDone.callID,
						ToolName: hostedDone.name,
						Detail:   hostedDone.detail,
					})
				}
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
				if len(ev.Response.Citations) > 0 {
					respCitations = parseTopLevelCitations(ev.Response.Citations)
				}
				if len(rawItems) == 0 && len(ev.Response.Output) > 0 {
					for _, item := range ev.Response.Output {
						cp := append(json.RawMessage(nil), item...)
						rawItems = append(rawItems, cp)
						decoded, _ := decodeOutputItemWithActivity(cp, isHosted)
						finalBlocks = append(finalBlocks, decoded...)
					}
				}
			}
		case "response.failed":
			msg := d.ProviderID + " response failed"
			if ev.Response != nil && ev.Response.Error != nil && ev.Response.Error.Message != "" {
				msg = ev.Response.Error.Message
			}
			return errors.New(msg)
		}
		return nil
	})
	if err != nil {
		return provider.Response{}, err
	}
	if !sawDone && len(rawItems) == 0 {
		return provider.Response{}, errors.New(d.ProviderID + " stream ended before response.completed")
	}

	if len(respCitations) > 0 {
		mergeCitationsIntoLastTextBlock(finalBlocks, respCitations)
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

type hostedActivity struct {
	callID string
	name   string
	detail string
}

func isHostedOutputItemType(t string) bool {
	switch t {
	case "web_search_call", "x_search_call", "image_generation_call", "custom_tool_call", "code_interpreter_call":
		return true
	default:
		return false
	}
}

func hostedDisplayName(itemType, itemName string) string {
	if itemName != "" {
		return itemName
	}
	if itemType == "image_generation_call" {
		return "image_generation"
	}
	return itemType
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

func rawItemType(raw json.RawMessage) string {
	var meta struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &meta) != nil {
		return ""
	}
	return meta.Type
}

func parseTopLevelCitations(rawList []json.RawMessage) []schema.Citation {
	var out []schema.Citation
	seen := map[string]bool{}
	for _, raw := range rawList {
		var u string
		if json.Unmarshal(raw, &u) == nil && u != "" {
			if !seen[u] {
				seen[u] = true
				out = append(out, schema.Citation{Type: "url_citation", URL: u})
			}
			continue
		}
		var obj struct {
			Type  string `json:"type"`
			URL   string `json:"url"`
			Title string `json:"title"`
		}
		if json.Unmarshal(raw, &obj) == nil && obj.URL != "" && !seen[obj.URL] {
			seen[obj.URL] = true
			ct := obj.Type
			if ct == "" {
				ct = "url_citation"
			}
			out = append(out, schema.Citation{Type: ct, URL: obj.URL, Title: obj.Title})
		}
	}
	return out
}

func mergeCitationsIntoLastTextBlock(blocks []schema.ContentBlock, extra []schema.Citation) {
	lastTextIdx := -1
	for i := range blocks {
		if blocks[i].Type == schema.BlockText {
			lastTextIdx = i
		}
	}
	if lastTextIdx < 0 {
		return
	}
	seen := map[string]bool{}
	for _, c := range blocks[lastTextIdx].Citations {
		if c.URL != "" {
			seen[c.URL] = true
		}
	}
	for _, c := range extra {
		if c.URL != "" && !seen[c.URL] {
			seen[c.URL] = true
			blocks[lastTextIdx].Citations = append(blocks[lastTextIdx].Citations, c)
		}
	}
}

func decodeOutputItemWithActivity(raw json.RawMessage, isHosted func(callID, name string) bool) ([]schema.ContentBlock, *hostedActivity) {
	var item struct {
		Type             string `json:"type"`
		ID               string `json:"id"`
		CallID           string `json:"call_id"`
		Name             string `json:"name"`
		Status           string `json:"status"`
		Arguments        string `json:"arguments"`
		Input            string `json:"input"`
		RevisedPrompt    string `json:"revised_prompt"`
		Result           string `json:"result"`
		EncryptedContent string `json:"encrypted_content"`
		Action           *struct {
			Type    string   `json:"type"`
			Query   string   `json:"query"`
			Queries []string `json:"queries"`
			URL     string   `json:"url"`
			Pattern string   `json:"pattern"`
		} `json:"action"`
		Summary []struct {
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
		return nil, nil
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
		return out, nil
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
		}}, nil
	case "function_call":
		if isHosted(item.CallID, item.Name) {
			return []schema.ContentBlock{{
				Type:           schema.BlockHostedToolCall,
				CallID:         item.CallID,
				Name:           item.Name,
				ProviderOpaque: append(json.RawMessage(nil), raw...),
			}}, &hostedActivity{callID: item.CallID, name: item.Name}
		}
		return []schema.ContentBlock{{
			Type:      schema.BlockToolCall,
			CallID:    item.CallID,
			Name:      item.Name,
			Arguments: json.RawMessage(provider.SanitizeToolArgsJSON([]byte(item.Arguments))),
		}}, nil
	case "image_generation_call":
		cid := firstNonEmpty(item.CallID, item.ID)
		cp := append(json.RawMessage(nil), raw...)
		act := &hostedActivity{
			callID: cid,
			name:   "image_generation",
			detail: item.RevisedPrompt,
		}
		if strings.TrimSpace(item.Result) != "" {
			return []schema.ContentBlock{{
				Type:   schema.BlockImage,
				CallID: cid,
				Name:   "image_generation",
				Text:   item.RevisedPrompt,
				Image: &schema.ImageSource{
					MediaType: schema.DetectBase64ImageMediaType(item.Result),
					Base64:    strings.TrimSpace(item.Result),
				},
				ProviderOpaque: cp,
			}}, act
		}
		return []schema.ContentBlock{{
			Type:           schema.BlockHostedToolResult,
			CallID:         cid,
			Name:           "image_generation",
			Text:           item.RevisedPrompt,
			ProviderOpaque: cp,
		}}, act
	case "web_search_call", "x_search_call", "custom_tool_call", "code_interpreter_call":
		cid := firstNonEmpty(item.CallID, item.ID)
		name := hostedDisplayName(item.Type, item.Name)
		detail := ""
		if item.Action != nil {
			detail = firstNonEmpty(item.Action.Query, item.Action.URL, item.Action.Pattern)
			if detail == "" && len(item.Action.Queries) > 0 {
				detail = strings.Join(item.Action.Queries, ", ")
			}
		}
		return []schema.ContentBlock{{
			Type:           schema.BlockHostedToolCall,
			CallID:         cid,
			Name:           name,
			Text:           detail,
			ProviderOpaque: append(json.RawMessage(nil), raw...),
		}}, &hostedActivity{callID: cid, name: name, detail: detail}
	case "compaction":
		return []schema.ContentBlock{{
			Type:           schema.BlockCompaction,
			ProviderOpaque: append(json.RawMessage(nil), raw...),
		}}, nil
	}
	return nil, nil
}
