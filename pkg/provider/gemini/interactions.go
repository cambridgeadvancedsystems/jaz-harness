package gemini

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
	"github.com/google/uuid"
)

const WireFormatInteractions = "gemini_interactions"

// streamInteractions speaks Google's stateless Vertex v1beta1 Interactions API
// (`store: false`, `stream: true`), which natively combines `google_search`,
// `url_context`, and custom `function` tools with signed step circulation and
// `continuation_token` support for Gemini 4 long decodes.
func (c *Client) streamInteractions(ctx context.Context, req provider.Request, emit provider.StreamCallback) (provider.Response, error) {
	body, err := c.buildInteractionsRequestBody(req)
	if err != nil {
		return provider.Response{}, err
	}
	endpoint, err := c.interactionsEndpoint()
	if err != nil {
		return provider.Response{}, err
	}

	resp, err := c.doHTTP(ctx, endpoint, body)
	if err != nil {
		return provider.Response{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		rawErr, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		return provider.Response{}, &provider.APIError{
			Provider:   c.ID(),
			StatusCode: resp.StatusCode,
			Body:       strings.TrimSpace(string(rawErr)),
			RetryAfter: provider.ParseRetryAfter(resp.Header),
		}
	}

	return parseInteractionsSSE(resp.Body, c.ID(), req.Model, emit)
}

func (c *Client) interactionsEndpoint() (string, error) {
	base := stripLegacyOpenAPISuffix(c.cfg.BaseURL)
	if c.cfg.RelayMode {
		return base + "/interactions", nil
	}
	if c.cfg.Mode == ModeAIStudio {
		return base + "/v1beta/interactions", nil
	}
	if strings.Contains(base, "/locations/") {
		return base + "/interactions", nil
	}
	if c.cfg.Project == "" {
		return "", errors.New("vertex interactions: Project is required")
	}
	loc := c.cfg.Location
	if loc == "" {
		loc = "global"
	}
	return fmt.Sprintf(
		"%s/v1beta1/projects/%s/locations/%s/interactions",
		base,
		url.PathEscape(c.cfg.Project),
		url.PathEscape(loc),
	), nil
}

func (c *Client) buildInteractionsRequestBody(req provider.Request) ([]byte, error) {
	if strings.TrimSpace(req.Model) == "" {
		return nil, errors.New("gemini interactions: model is required")
	}
	cleanModel := strings.TrimPrefix(strings.TrimSpace(req.Model), "google/")

	turns := make([]any, 0, len(req.Messages))
	for _, m := range req.Messages {
		if m.Native != nil && m.Native.WireFormat == WireFormatInteractions && len(m.Native.RawItems) > 0 {
			for _, raw := range m.Native.RawItems {
				turns = append(turns, raw)
			}
			continue
		}
		role := "user"
		if m.Role == schema.RoleAssistant {
			role = "model"
		}
		var content []map[string]any
		attachedFirstFuncSig := false
		for _, b := range m.Blocks {
			switch b.Type {
			case schema.BlockCompaction, schema.BlockText:
				if b.Text == "" {
					continue
				}
				item := map[string]any{"type": "text", "text": b.Text}
				if b.ThoughtSignBlob != "" {
					item["signature"] = b.ThoughtSignBlob
				}
				content = append(content, item)
			case schema.BlockToolCall:
				item := map[string]any{
					"type":      "function_call",
					"id":        b.CallID,
					"name":      b.Name,
					"arguments": provider.DecodeToolArgsObject(b.Arguments),
				}
				if b.ThoughtSignBlob != "" && !attachedFirstFuncSig {
					item["signature"] = b.ThoughtSignBlob
					attachedFirstFuncSig = true
				}
				content = append(content, item)
			case schema.BlockToolResult:
				content = append(content, map[string]any{
					"type":     "function_result",
					"call_id":  b.CallID,
					"name":     b.Name,
					"result":   b.Output,
					"is_error": b.IsError,
				})
			}
		}
		if len(content) > 0 {
			turns = append(turns, map[string]any{"role": role, "content": content})
		}
	}

	var tools []map[string]any
	for _, h := range req.HostedTools {
		switch h.Type {
		case "google_search", "web_search":
			tools = append(tools, map[string]any{"type": "google_search"})
		case "url_context", "web_fetch":
			tools = append(tools, map[string]any{"type": "url_context"})
		default:
			obj := map[string]any{"type": h.Type}
			for k, v := range h.Config {
				obj[k] = v
			}
			tools = append(tools, obj)
		}
	}
	for _, t := range req.Tools {
		toolObj := map[string]any{
			"type": "function",
			"name": t.Name,
		}
		if t.Description != "" {
			toolObj["description"] = t.Description
		}
		if len(bytes.TrimSpace(t.InputSchema)) > 0 {
			var obj map[string]any
			if json.Unmarshal(t.InputSchema, &obj) == nil && obj != nil {
				cleanGeminiSchema(obj)
				toolObj["parameters"] = obj
			}
		}
		tools = append(tools, toolObj)
	}

	genCfg := map[string]any{}
	if req.MaxOutputTokens > 0 {
		genCfg["max_output_tokens"] = req.MaxOutputTokens
	}
	if !isGemini4Model(cleanModel) {
		if req.Temperature != nil {
			genCfg["temperature"] = *req.Temperature
		}
		if req.TopP != nil {
			genCfg["top_p"] = *req.TopP
		}
	}
	if lvl := mapGeminiThinkingLevel(cleanModel, req.ReasoningEffort); lvl != "" {
		genCfg["thinking_level"] = strings.ToLower(lvl)
	}

	payload := map[string]any{
		"model":  cleanModel,
		"input":  turns,
		"store":  false,
		"stream": true,
	}
	if strings.TrimSpace(req.SystemPrompt) != "" {
		payload["system_instruction"] = req.SystemPrompt
	}
	if len(tools) > 0 {
		payload["tools"] = tools
	}
	if len(genCfg) > 0 {
		payload["generation_config"] = genCfg
	}
	if tok := strings.TrimSpace(req.Metadata["continuation_token"]); tok != "" {
		payload["continuation_token"] = tok
	}
	return json.Marshal(payload)
}

func parseInteractionsSSE(r io.Reader, providerID, fallbackModel string, emit provider.StreamCallback) (provider.Response, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)

	var (
		interactionID     string
		model             = fallbackModel
		stopReason        = "completed"
		continuationToken string
		usage             schema.Usage
		nativeOutputs     []json.RawMessage
		canonical         []schema.ContentBlock
		textBuf           strings.Builder
		thoughtBuf        strings.Builder
	)

	flushText := func() {
		if textBuf.Len() == 0 {
			return
		}
		canonical = append(canonical, schema.ContentBlock{Type: schema.BlockText, Text: textBuf.String()})
		textBuf.Reset()
	}
	flushThought := func() {
		if thoughtBuf.Len() == 0 {
			return
		}
		canonical = append(canonical, schema.ContentBlock{Type: schema.BlockThinking, Thinking: thoughtBuf.String()})
		thoughtBuf.Reset()
	}

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}
		var ev struct {
			EventType   string `json:"event_type"`
			Interaction *struct {
				ID                string            `json:"id"`
				Model             string            `json:"model"`
				Status            string            `json:"status"`
				ContinuationToken string            `json:"continuation_token"`
				Outputs           []json.RawMessage `json:"outputs"`
				Usage             *struct {
					TotalInputTokens   int64 `json:"total_input_tokens"`
					TotalCachedTokens  int64 `json:"total_cached_tokens"`
					TotalOutputTokens  int64 `json:"total_output_tokens"`
					TotalThoughtTokens int64 `json:"total_thought_tokens"`
					TotalTokens        int64 `json:"total_tokens"`
				} `json:"usage"`
			} `json:"interaction"`
			Delta *struct {
				Type      string          `json:"type"`
				Text      string          `json:"text"`
				Thought   string          `json:"thought"`
				Signature string          `json:"signature"`
				ID        string          `json:"id"`
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			} `json:"delta"`
		}
		if json.Unmarshal([]byte(data), &ev) != nil {
			continue
		}

		switch ev.EventType {
		case "content.delta":
			if ev.Delta == nil {
				continue
			}
			switch ev.Delta.Type {
			case "thought":
				if ev.Delta.Thought != "" {
					thoughtBuf.WriteString(ev.Delta.Thought)
					if emit != nil {
						emit(provider.StreamChunk{Kind: provider.DeltaThinking, Text: ev.Delta.Thought})
					}
				}
			case "text":
				if ev.Delta.Text != "" {
					flushThought()
					textBuf.WriteString(ev.Delta.Text)
					if emit != nil {
						emit(provider.StreamChunk{Kind: provider.DeltaText, Text: ev.Delta.Text})
					}
				}
			case "function_call":
				flushThought()
				flushText()
				cid := ev.Delta.ID
				if cid == "" {
					cid = "icall_" + uuid.NewString()[:8]
				}
				args := json.RawMessage(provider.SanitizeToolArgsJSON(ev.Delta.Arguments))
				if emit != nil {
					emit(provider.StreamChunk{Kind: provider.DeltaToolCallStart, CallID: cid, ToolName: ev.Delta.Name, Arguments: string(args)})
				}
				canonical = append(canonical, schema.ContentBlock{
					Type:            schema.BlockToolCall,
					CallID:          cid,
					Name:            ev.Delta.Name,
					Arguments:       args,
					ThoughtSignBlob: ev.Delta.Signature,
				})
			case "google_search_call", "url_context_call":
				if emit != nil {
					emit(provider.StreamChunk{Kind: provider.DeltaHostedToolStart, CallID: ev.Delta.ID, ToolName: ev.Delta.Type})
				}
			case "google_search_result", "url_context_result":
				if emit != nil {
					emit(provider.StreamChunk{Kind: provider.DeltaHostedToolComplete, CallID: ev.Delta.ID, ToolName: ev.Delta.Type})
				}
			}
		case "interaction.complete":
			if ev.Interaction != nil {
				interactionID = ev.Interaction.ID
				if ev.Interaction.Model != "" {
					model = ev.Interaction.Model
				}
				if ev.Interaction.Status != "" {
					stopReason = ev.Interaction.Status
				}
				continuationToken = ev.Interaction.ContinuationToken
				if ev.Interaction.Usage != nil {
					u := ev.Interaction.Usage
					usage.InputTokens = u.TotalInputTokens
					usage.CachedInputTokens = u.TotalCachedTokens
					usage.OutputTokens = u.TotalOutputTokens
					usage.ReasoningTokens = u.TotalThoughtTokens
					usage.TotalTokens = u.TotalTokens
					if usage.TotalTokens == 0 {
						usage.TotalTokens = u.TotalInputTokens + u.TotalOutputTokens + u.TotalThoughtTokens
					}
				}
				for _, rawOut := range ev.Interaction.Outputs {
					nativeOutputs = append(nativeOutputs, append(json.RawMessage(nil), rawOut...))
				}
			}
		}
	}
	if err := sc.Err(); err != nil {
		return provider.Response{}, err
	}
	flushThought()
	flushText()

	// If interaction.complete returned authoritative outputs, parse any blocks
	// not already accumulated via deltas (e.g. citations, tool steps, signatures).
	if len(canonical) == 0 && len(nativeOutputs) > 0 {
		for _, raw := range nativeOutputs {
			var item struct {
				Type      string          `json:"type"`
				Text      string          `json:"text"`
				Thought   string          `json:"thought"`
				Signature string          `json:"signature"`
				ID        string          `json:"id"`
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}
			if json.Unmarshal(raw, &item) != nil {
				continue
			}
			switch item.Type {
			case "text":
				canonical = append(canonical, schema.ContentBlock{Type: schema.BlockText, Text: item.Text, ThoughtSignBlob: item.Signature})
			case "thought":
				canonical = append(canonical, schema.ContentBlock{Type: schema.BlockThinking, Thinking: item.Thought, ThoughtSignBlob: item.Signature})
			case "function_call":
				canonical = append(canonical, schema.ContentBlock{
					Type:            schema.BlockToolCall,
					CallID:          item.ID,
					Name:            item.Name,
					Arguments:       json.RawMessage(provider.SanitizeToolArgsJSON(item.Arguments)),
					ThoughtSignBlob: item.Signature,
				})
			case "google_search_call", "url_context_call":
				canonical = append(canonical, schema.ContentBlock{Type: schema.BlockHostedToolCall, CallID: item.ID, Name: item.Type, ProviderOpaque: raw})
			case "google_search_result", "url_context_result":
				canonical = append(canonical, schema.ContentBlock{Type: schema.BlockHostedToolResult, CallID: item.ID, Name: item.Type, ProviderOpaque: raw})
			}
		}
	}

	var rawItems []json.RawMessage
	if len(nativeOutputs) > 0 {
		modelTurnObj, _ := json.Marshal(map[string]any{
			"role":    "model",
			"content": nativeOutputs,
		})
		rawItems = []json.RawMessage{modelTurnObj}
	}
	if interactionID == "" {
		interactionID = "int_" + uuid.NewString()[:12]
	}
	return provider.Response{
		ID:                interactionID,
		Model:             model,
		Blocks:            canonical,
		Usage:             usage,
		StopReason:        stopReason,
		ContinuationToken: continuationToken,
		Native: &schema.NativeEnvelope{
			Provider:   providerID,
			WireFormat: WireFormatInteractions,
			Model:      model,
			ResponseID: interactionID,
			RawItems:   rawItems,
		},
	}, nil
}
