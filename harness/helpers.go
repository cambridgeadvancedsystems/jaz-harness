package harness

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/cambridgeadvancedsystems/jaz-harness/tools"
)

func (e *Engine) Subscribe(threadID string) (<-chan schema.SessionEvent, func()) {
	ch := make(chan schema.SessionEvent, 128)
	e.subMu.Lock()
	if e.subscribers[threadID] == nil {
		e.subscribers[threadID] = make(map[chan schema.SessionEvent]struct{})
	}
	e.subscribers[threadID][ch] = struct{}{}
	e.subMu.Unlock()

	unsub := func() {
		e.subMu.Lock()
		if set := e.subscribers[threadID]; set != nil {
			delete(set, ch)
			if len(set) == 0 {
				delete(e.subscribers, threadID)
			}
		}
		e.subMu.Unlock()
		close(ch)
	}
	return ch, unsub
}

func (e *Engine) publishEvent(ctx context.Context, threadID string, epoch int64, evType schema.EventType, payload map[string]any) {
	saved, err := e.store.AppendEvent(ctx, schema.SessionEvent{
		ThreadID:  threadID,
		TurnEpoch: epoch,
		Type:      evType,
		Payload:   payload,
		CreatedAt: time.Now().UTC(),
	})
	if err != nil {
		return
	}
	e.subMu.RLock()
	for ch := range e.subscribers[threadID] {
		select {
		case ch <- saved:
		default:
		}
	}
	e.subMu.RUnlock()
}

func (e *Engine) CreateThread(ctx context.Context, th schema.Thread) (schema.Thread, error) {
	if strings.TrimSpace(th.Model) == "" && strings.TrimSpace(th.Provider) != "" {
		th.Model = schema.DefaultModelForProvider(th.Provider)
	}
	return e.store.CreateThread(ctx, th)
}

func (e *Engine) GetThread(ctx context.Context, threadID string) (schema.Thread, error) {
	return e.store.GetThread(ctx, threadID)
}

func (e *Engine) ListThreads(ctx context.Context) ([]schema.Thread, error) {
	return e.store.ListThreads(ctx)
}

func (e *Engine) UpdateThreadModel(ctx context.Context, threadID, providerName, model, effort string) (schema.Thread, error) {
	th, err := e.store.GetThread(ctx, threadID)
	if err != nil {
		return schema.Thread{}, err
	}
	if strings.TrimSpace(providerName) != "" {
		th.Provider = strings.TrimSpace(providerName)
	}
	if strings.TrimSpace(model) != "" {
		th.Model = strings.TrimSpace(model)
	}
	if effort != "" {
		th.ReasoningEffort = strings.TrimSpace(effort)
	}
	if err := e.store.UpdateThread(ctx, th); err != nil {
		return schema.Thread{}, err
	}
	return th, nil
}

// Steer queues a live user steering message into an active turn (delivered
// between model steps without aborting in-flight tool work).
func (e *Engine) Steer(threadID, text string) error {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return errors.New("steer text is required")
	}
	e.runsMu.Lock()
	run, ok := e.runs[threadID]
	e.runsMu.Unlock()
	if !ok {
		return errors.New("no active turn running on thread")
	}
	run.pushSteer(trimmed)
	return nil
}

// Cancel interrupts an active turn cleanly.
func (e *Engine) Cancel(threadID string) bool {
	e.runsMu.Lock()
	run, ok := e.runs[threadID]
	e.runsMu.Unlock()
	if !ok {
		return false
	}
	run.cancel()
	return true
}

func (e *Engine) executeToolCalls(ctx context.Context, threadID string, epoch int64, calls []schema.ContentBlock) []schema.ContentBlock {
	results := make([]schema.ContentBlock, len(calls))
	var wg sync.WaitGroup
	for i, call := range calls {
		i := i
		call := call
		wg.Add(1)
		go func() {
			defer wg.Done()
			var argsAny any
			_ = json.Unmarshal(call.Arguments, &argsAny)
			e.publishEvent(ctx, threadID, epoch, schema.EventToolCallStarted, map[string]any{
				"call_id":   call.CallID,
				"tool":      call.Name,
				"arguments": argsAny,
				"status":    "running",
			})
			callCtx := tools.WithToolCallID(ctx, call.CallID)
			res, err := e.tools.ExecuteResult(callCtx, call.Name, call.Arguments)
			out := res.Text
			isErr := false
			if err != nil {
				isErr = true
				out = err.Error()
			}
			status := "completed"
			if isErr {
				status = "failed"
			}
			evPayload := map[string]any{
				"call_id":  call.CallID,
				"tool":     call.Name,
				"status":   status,
				"output":   out,
				"is_error": isErr,
			}
			if len(res.Images) > 0 {
				evPayload["image_count"] = len(res.Images)
			}
			if len(res.Metadata) > 0 {
				evPayload["metadata"] = res.Metadata
			}
			e.publishEvent(ctx, threadID, epoch, schema.EventToolCallResult, evPayload)
			results[i] = schema.ContentBlock{
				Type:    schema.BlockToolResult,
				CallID:  call.CallID,
				Name:    call.Name,
				Output:  out,
				Images:  res.Images,
				IsError: isErr,
			}
		}()
	}
	wg.Wait()
	return results
}

func (e *Engine) assembleSystemPrompt(th schema.Thread) string {
	var parts []string
	if strings.TrimSpace(e.cfg.BaseSystemPrompt) != "" {
		parts = append(parts, strings.TrimSpace(e.cfg.BaseSystemPrompt))
	}
	if strings.TrimSpace(th.SystemPrompt) != "" {
		parts = append(parts, strings.TrimSpace(th.SystemPrompt))
	}
	if e.skills != nil {
		if idx := e.skills.RenderPromptIndex(); idx != "" {
			parts = append(parts, idx)
		}
	}
	return strings.Join(parts, "\n\n")
}

func (e *Engine) publishHostedToolEvent(ctx context.Context, threadID string, epoch int64, ch provider.StreamChunk, status string) {
	evPayload := map[string]any{
		"call_id": ch.CallID,
		"tool":    ch.ToolName,
		"status":  status,
	}
	if ch.Detail != "" {
		evPayload["detail"] = ch.Detail
	}
	e.publishEvent(ctx, threadID, epoch, schema.EventHostedToolActivity, evPayload)
}

func (e *Engine) resolveHostedTools(providerName string, req TurnRequest) []schema.HostedTool {
	if len(req.HostedTools) > 0 {
		return req.HostedTools
	}
	enableSearch := e.cfg.DefaultHostedTools
	if req.EnableWebSearch != nil {
		enableSearch = *req.EnableWebSearch
	}
	enableImgGen := e.cfg.EnableNativeImageGeneration
	if req.EnableImageGeneration != nil {
		enableImgGen = *req.EnableImageGeneration
	}

	norm := strings.ToLower(strings.TrimSpace(providerName))
	var out []schema.HostedTool
	if enableSearch {
		switch norm {
		case "openai", "codex_oauth", "azure_openai":
			out = append(out, schema.HostedTool{Type: "web_search"})
		case "anthropic":
			out = append(out,
				schema.HostedTool{Type: "web_search_20250305", Name: "web_search"},
				schema.HostedTool{Type: "web_fetch_20250910", Name: "web_fetch"},
			)
		case "vertex_anthropic", "bedrock_anthropic":
			out = append(out, schema.HostedTool{Type: "web_search_20250305", Name: "web_search"})
		case "grok", "xai":
			out = append(out,
				schema.HostedTool{Type: "x_search", Config: map[string]any{"enable_image_understanding": true}},
				schema.HostedTool{Type: "web_search"},
			)
		case "gemini", "vertex", "vertex_gemini", "vertex_interactions":
			out = append(out,
				schema.HostedTool{Type: "google_search"},
				schema.HostedTool{Type: "url_context"},
			)
		}
	}
	if enableImgGen {
		switch norm {
		case "openai", "azure_openai":
			out = append(out, schema.HostedTool{Type: "image_generation"})
		}
	}
	return out
}

func usesHostedImageGeneration(providerName, wireFormat string, hosted []schema.HostedTool) bool {
	if wireFormat != "openai_responses" {
		return false
	}
	if strings.EqualFold(strings.TrimSpace(providerName), "codex_oauth") {
		return false
	}
	for _, h := range hosted {
		if strings.TrimSpace(h.Type) == "image_generation" {
			return true
		}
	}
	return false
}

func omitToolByName(defs []schema.ToolDef, name string) []schema.ToolDef {
	out := make([]schema.ToolDef, 0, len(defs))
	for _, d := range defs {
		if d.Name == name {
			continue
		}
		out = append(out, d)
	}
	return out
}

func collectConversationImages(msgs []schema.Message, currentStepBlocks []schema.ContentBlock) []schema.ImageSource {
	var imgs []schema.ImageSource
	appendFromBlocks := func(blocks []schema.ContentBlock) {
		for _, b := range blocks {
			if b.Type == schema.BlockImage && b.Image != nil && b.Image.DataURL() != "" {
				imgs = append(imgs, *b.Image)
			}
			for _, img := range b.Images {
				if img.DataURL() != "" {
					imgs = append(imgs, img)
				}
			}
		}
	}
	for _, m := range msgs {
		appendFromBlocks(m.Blocks)
	}
	appendFromBlocks(currentStepBlocks)
	return imgs
}

func (e *Engine) buildProviderCompaction(providerName string, spec schema.ModelSpec) *provider.CompactionConfig {
	if !e.cfg.NativeServerCompaction || !spec.SupportsRemoteCompaction {
		return nil
	}
	switch strings.ToLower(strings.TrimSpace(providerName)) {
	case "anthropic", "vertex_anthropic", "bedrock_anthropic", "openai", "azure_openai":
		return &provider.CompactionConfig{
			Enabled:       true,
			TriggerTokens: e.compactor.ResolveAutoCompactThreshold(spec),
		}
	default:
		return nil
	}
}

func buildUserBlocks(prompt string, images []schema.ImageSource) []schema.ContentBlock {
	userBlocks := make([]schema.ContentBlock, 0, 1+len(images))
	if strings.TrimSpace(prompt) != "" {
		userBlocks = append(userBlocks, schema.ContentBlock{
			Type: schema.BlockText,
			Text: prompt,
		})
	}
	for _, img := range images {
		cp := img
		userBlocks = append(userBlocks, schema.ContentBlock{
			Type:  schema.BlockImage,
			Image: &cp,
		})
	}
	return userBlocks
}
