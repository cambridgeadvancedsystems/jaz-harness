package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"sort"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/google/uuid"
)

const WireFormatInteractions = "gemini_interactions"

// streamInteractions speaks Google's stateless Vertex v1beta1 Interactions API
// (`store: false`, `stream: true`) using the steps-based `step_list` wire format,
// natively combining `google_search`, `url_context`, and custom `function` tools
// with signed step circulation and `continuation_token` support for Gemini 4.
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
		rawErr, _ := io.ReadAll(io.LimitReader(resp.Body, provider.DefaultMaxErrorBodyBytes))
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

	steps := make([]any, 0, len(req.Messages)*2)
	callNames := make(map[string]string)

	for _, m := range req.Messages {
		for _, b := range m.Blocks {
			if b.Type == schema.BlockToolCall && b.CallID != "" && b.Name != "" {
				callNames[b.CallID] = b.Name
			}
		}
		if m.Native != nil && m.Native.WireFormat == WireFormatInteractions && len(m.Native.RawItems) > 0 {
			for _, raw := range m.Native.RawItems {
				steps = append(steps, raw)
			}
			continue
		}

		var textContent []map[string]any
		flushTextStep := func() {
			if len(textContent) == 0 {
				return
			}
			stepType := "user_input"
			if m.Role == schema.RoleAssistant {
				stepType = "model_output"
			}
			steps = append(steps, map[string]any{
				"type":    stepType,
				"content": textContent,
			})
			textContent = nil
		}

		for _, b := range m.Blocks {
			switch b.Type {
			case schema.BlockThinking:
				if b.ThoughtSignBlob != "" {
					flushTextStep()
					steps = append(steps, map[string]any{
						"type":      "thought",
						"signature": b.ThoughtSignBlob,
					})
				}
			case schema.BlockCompaction, schema.BlockText:
				if b.Text == "" {
					continue
				}
				textContent = append(textContent, map[string]any{
					"type": "text",
					"text": b.Text,
				})
			case schema.BlockImage:
				if b.Image != nil && b.Image.Base64 != "" {
					mt := b.Image.MediaType
					if mt == "" {
						mt = "image/png"
					}
					textContent = append(textContent, map[string]any{
						"type":      "image",
						"data":      b.Image.Base64,
						"mime_type": mt,
					})
				}
			case schema.BlockToolCall:
				flushTextStep()
				if b.ThoughtSignBlob != "" {
					steps = append(steps, map[string]any{
						"type":      "thought",
						"signature": b.ThoughtSignBlob,
					})
				}
				steps = append(steps, map[string]any{
					"type":      "function_call",
					"id":        b.CallID,
					"name":      b.Name,
					"arguments": provider.DecodeToolArgsObject(b.Arguments),
				})
			case schema.BlockToolResult:
				flushTextStep()
				name := b.Name
				if name == "" {
					name = callNames[b.CallID]
				}
				if name == "" {
					name = "tool"
				}
				steps = append(steps, map[string]any{
					"type":     "function_result",
					"call_id":  b.CallID,
					"name":     name,
					"result":   b.Output,
					"is_error": b.IsError,
				})
			case schema.BlockHostedToolCall, schema.BlockHostedToolResult:
				if len(b.ProviderOpaque) > 0 && m.Native != nil && m.Native.WireFormat == WireFormatInteractions {
					flushTextStep()
					steps = append(steps, b.ProviderOpaque)
				}
			}
		}
		flushTextStep()
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
		"input":  steps,
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

type interactionStepAcc struct {
	stepType    string
	id          string
	name        string
	signature   string
	initialArgs json.RawMessage
	textBuf     strings.Builder
	thoughtBuf  strings.Builder
	argsBuf     strings.Builder
	stepMap     map[string]any
	citations   []schema.Citation
}

func parseInteractionsSSE(r io.Reader, providerID, fallbackModel string, emit provider.StreamCallback) (provider.Response, error) {
	var (
		interactionID     string
		model             = fallbackModel
		stopReason        = "completed"
		continuationToken string
		usage             schema.Usage
		steps             = make(map[int]*interactionStepAcc)
		completedSteps    []json.RawMessage
	)

	err := provider.ScanLines(r, provider.DefaultMaxLineBytes, func(rawLine string) error {
		line := strings.TrimSpace(rawLine)
		if !strings.HasPrefix(line, "data:") {
			return nil
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			return nil
		}
		var ev struct {
			EventType   string          `json:"event_type"`
			Index       int             `json:"index"`
			Step        json.RawMessage `json:"step"`
			Interaction *struct {
				ID                string            `json:"id"`
				Model             string            `json:"model"`
				Status            string            `json:"status"`
				ContinuationToken string            `json:"continuation_token"`
				Steps             []json.RawMessage `json:"steps"`
				Outputs           []json.RawMessage `json:"outputs"`
				Usage             *struct {
					TotalInputTokens   int64 `json:"total_input_tokens"`
					TotalCachedTokens  int64 `json:"total_cached_tokens"`
					TotalOutputTokens  int64 `json:"total_output_tokens"`
					TotalThoughtTokens int64 `json:"total_thought_tokens"`
					TotalToolUseTokens int64 `json:"total_tool_use_tokens"`
					TotalTokens        int64 `json:"total_tokens"`
				} `json:"usage"`
			} `json:"interaction"`
			Delta *struct {
				Type      string          `json:"type"`
				Text      string          `json:"text"`
				Thought   string          `json:"thought"`
				Signature string          `json:"signature"`
				Arguments json.RawMessage `json:"arguments"`
				Result    json.RawMessage `json:"result"`
			} `json:"delta"`
		}
		if json.Unmarshal([]byte(data), &ev) != nil {
			return nil
		}

		switch ev.EventType {
		case "interaction.created":
			if ev.Interaction != nil {
				if ev.Interaction.ID != "" {
					interactionID = ev.Interaction.ID
				}
				if ev.Interaction.Model != "" {
					model = ev.Interaction.Model
				}
			}
		case "step.start":
			var st struct {
				Type      string          `json:"type"`
				ID        string          `json:"id"`
				CallID    string          `json:"call_id"`
				Name      string          `json:"name"`
				Signature string          `json:"signature"`
				Arguments json.RawMessage `json:"arguments"`
			}
			if json.Unmarshal(ev.Step, &st) != nil {
				return nil
			}
			var stepMap map[string]any
			_ = json.Unmarshal(ev.Step, &stepMap)
			id := st.ID
			if id == "" {
				id = st.CallID
			}
			acc := &interactionStepAcc{
				stepType:    st.Type,
				id:          id,
				name:        st.Name,
				signature:   st.Signature,
				initialArgs: append(json.RawMessage(nil), st.Arguments...),
				stepMap:     stepMap,
			}
			steps[ev.Index] = acc
			if emit != nil {
				switch st.Type {
				case "function_call":
					emit(provider.StreamChunk{Kind: provider.DeltaToolCallStart, CallID: id, ToolName: st.Name})
				case "google_search_call", "url_context_call", "code_execution_call":
					emit(provider.StreamChunk{Kind: provider.DeltaHostedToolStart, CallID: id, ToolName: st.Type})
				case "google_search_result", "url_context_result", "code_execution_result":
					emit(provider.StreamChunk{Kind: provider.DeltaHostedToolComplete, CallID: id, ToolName: st.Type})
				}
			}
		case "step.delta":
			acc := steps[ev.Index]
			if acc == nil || ev.Delta == nil {
				return nil
			}
			switch ev.Delta.Type {
			case "thought_signature":
				if ev.Delta.Signature != "" {
					acc.signature = ev.Delta.Signature
				}
			case "thought", "thought_summary":
				txt := ev.Delta.Text
				if txt == "" {
					txt = ev.Delta.Thought
				}
				if txt != "" {
					acc.thoughtBuf.WriteString(txt)
					if emit != nil {
						emit(provider.StreamChunk{Kind: provider.DeltaThinking, Text: txt})
					}
				}
			case "text":
				if ev.Delta.Text != "" {
					acc.textBuf.WriteString(ev.Delta.Text)
					if emit != nil {
						emit(provider.StreamChunk{Kind: provider.DeltaText, Text: ev.Delta.Text})
					}
				}
			case "arguments_delta":
				var frag string
				if json.Unmarshal(ev.Delta.Arguments, &frag) == nil && frag != "" {
					acc.argsBuf.WriteString(frag)
					if emit != nil {
						emit(provider.StreamChunk{Kind: provider.DeltaToolCallArgs, CallID: acc.id, Arguments: frag})
					}
				}
			case "google_search_call", "url_context_call", "code_execution_call":
				if acc.stepMap != nil && len(ev.Delta.Arguments) > 0 {
					var argObj any
					if json.Unmarshal(ev.Delta.Arguments, &argObj) == nil {
						acc.stepMap["arguments"] = argObj
					}
				}
			case "google_search_result", "url_context_result", "code_execution_result":
				if acc.stepMap != nil && len(ev.Delta.Result) > 0 {
					var resObj any
					if json.Unmarshal(ev.Delta.Result, &resObj) == nil {
						acc.stepMap["result"] = resObj
					}
				}
			}
		case "interaction.completed", "interaction.complete":
			if ev.Interaction != nil {
				if ev.Interaction.ID != "" {
					interactionID = ev.Interaction.ID
				}
				if ev.Interaction.Model != "" {
					model = ev.Interaction.Model
				}
				if ev.Interaction.Status != "" {
					stopReason = ev.Interaction.Status
				}
				continuationToken = ev.Interaction.ContinuationToken
				if ev.Interaction.Usage != nil {
					u := ev.Interaction.Usage
					usage.InputTokens = u.TotalInputTokens + u.TotalToolUseTokens
					usage.CachedInputTokens = u.TotalCachedTokens
					usage.OutputTokens = u.TotalOutputTokens
					usage.ReasoningTokens = u.TotalThoughtTokens
					usage.TotalTokens = u.TotalTokens
					if usage.TotalTokens == 0 {
						usage.TotalTokens = usage.InputTokens + u.TotalOutputTokens + u.TotalThoughtTokens
					}
				}
				for _, rawSt := range ev.Interaction.Steps {
					completedSteps = append(completedSteps, append(json.RawMessage(nil), rawSt...))
				}
				if len(completedSteps) == 0 {
					for _, rawOut := range ev.Interaction.Outputs {
						completedSteps = append(completedSteps, append(json.RawMessage(nil), rawOut...))
					}
				}
			}
		}
		return nil
	})
	if err != nil {
		return provider.Response{}, err
	}

	indices := make([]int, 0, len(steps))
	for idx := range steps {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	var (
		canonical []schema.ContentBlock
		rawItems  []json.RawMessage
	)

	for _, idx := range indices {
		st := steps[idx]
		switch st.stepType {
		case "thought":
			stepObj := map[string]any{"type": "thought"}
			if st.signature != "" {
				stepObj["signature"] = st.signature
			}
			if st.thoughtBuf.Len() > 0 {
				stepObj["summary"] = []map[string]any{{"type": "text", "text": st.thoughtBuf.String()}}
			}
			raw, _ := json.Marshal(stepObj)
			rawItems = append(rawItems, raw)
			canonical = append(canonical, schema.ContentBlock{
				Type:            schema.BlockThinking,
				Thinking:        st.thoughtBuf.String(),
				ThoughtSignBlob: st.signature,
				ProviderOpaque:  raw,
			})
		case "model_output":
			text := st.textBuf.String()
			if text == "" {
				continue
			}
			stepObj := map[string]any{
				"type":    "model_output",
				"content": []map[string]any{{"type": "text", "text": text}},
			}
			raw, _ := json.Marshal(stepObj)
			rawItems = append(rawItems, raw)
			canonical = append(canonical, schema.ContentBlock{
				Type:      schema.BlockText,
				Text:      text,
				Citations: st.citations,
			})
		case "function_call":
			cid := st.id
			if cid == "" {
				cid = "icall_" + uuid.NewString()[:8]
			}
			rawArgs := strings.TrimSpace(st.argsBuf.String())
			if rawArgs == "" && len(bytes.TrimSpace(st.initialArgs)) > 0 {
				rawArgs = string(st.initialArgs)
			}
			cleanArgs := provider.SanitizeToolArgsJSON([]byte(rawArgs))
			stepObj := map[string]any{
				"type":      "function_call",
				"id":        cid,
				"name":      st.name,
				"arguments": provider.DecodeToolArgsObject([]byte(cleanArgs)),
			}
			raw, _ := json.Marshal(stepObj)
			rawItems = append(rawItems, raw)
			canonical = append(canonical, schema.ContentBlock{
				Type:            schema.BlockToolCall,
				CallID:          cid,
				Name:            st.name,
				Arguments:       json.RawMessage(cleanArgs),
				ThoughtSignBlob: st.signature,
			})
		case "google_search_call", "url_context_call", "code_execution_call":
			if len(st.stepMap) > 0 {
				raw, _ := json.Marshal(st.stepMap)
				rawItems = append(rawItems, raw)
				canonical = append(canonical, schema.ContentBlock{
					Type:           schema.BlockHostedToolCall,
					CallID:         st.id,
					Name:           st.stepType,
					ProviderOpaque: raw,
				})
			}
		case "google_search_result", "url_context_result", "code_execution_result":
			if len(st.stepMap) > 0 {
				raw, _ := json.Marshal(st.stepMap)
				rawItems = append(rawItems, raw)
				canonical = append(canonical, schema.ContentBlock{
					Type:           schema.BlockHostedToolResult,
					CallID:         st.id,
					Name:           st.stepType,
					ProviderOpaque: raw,
				})
			}
		}
	}

	if len(rawItems) == 0 && len(completedSteps) > 0 {
		rawItems = completedSteps
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
