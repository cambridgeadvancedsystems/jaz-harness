package responses

import (
	"encoding/json"
	"errors"
	"io"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
)

type hostedActivity struct {
	callID string
	name   string
	detail string
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
			emitHostedOnce(firstNonEmpty(ev.ItemID, ev.CallID), "image_generation", "")
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
						emitHostedOnce(firstNonEmpty(meta.CallID, meta.ID), hostedDisplayName(meta.Type, meta.Name), "")
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
		if b64 := strings.TrimSpace(item.Result); b64 != "" {
			return []schema.ContentBlock{{
				Type:   schema.BlockImage,
				CallID: cid,
				Name:   "image_generation",
				Text:   item.RevisedPrompt,
				Image: &schema.ImageSource{
					MediaType: schema.DetectBase64ImageMediaType(b64),
					Base64:    b64,
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
