package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/compact"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/skills"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/store"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/tools"
)

type Config struct {
	MaxStepsPerTurn    int
	EnableNativeCache  bool
	DefaultHostedTools bool
	NativeToolSearch   bool // use Anthropic native defer_loading + tool_search_tool_bm25 when on Anthropic
	AutoCompact        bool
	BaseSystemPrompt   string
}

type TurnRequest struct {
	ThreadID        string
	Prompt          string
	Images          []schema.ImageSource
	Provider        string // optional per-turn override (supports mid-thread switching!)
	Model           string // optional per-turn override
	ReasoningEffort string // optional per-turn override
	HostedTools     []schema.HostedTool
	EnableWebSearch *bool
	MaxOutputTokens int
	CachedContent   string              // optional Gemini 2.5/3 cachedContent name
	Heartbeat       func(detail string) // optional liveness callback (e.g. Temporal activity heartbeat)
}

type TurnResult struct {
	ThreadID      string
	TurnEpoch     int64
	Status        string // completed | failed | interrupted
	AssistantText string
	Usage         schema.Usage
	Steps         int
	Compacted     bool
	ErrorMessage  string
}

type activeRun struct {
	cancel  context.CancelFunc
	steerMu sync.Mutex
	steers  []string
}

func (a *activeRun) pushSteer(text string) {
	a.steerMu.Lock()
	a.steers = append(a.steers, text)
	a.steerMu.Unlock()
}

func (a *activeRun) drainSteers() []string {
	a.steerMu.Lock()
	defer a.steerMu.Unlock()
	if len(a.steers) == 0 {
		return nil
	}
	out := a.steers
	a.steers = nil
	return out
}

// Engine is Jaz's lightweight, pure-Go multi-provider agent runtime.
// Hosted turns run as goroutines with 0 subprocesses and 0 sandbox pods;
// workspace tools attach a sandbox lazily without mutating the tool schema.
type Engine struct {
	cfg       Config
	store     store.Store
	providers *provider.Registry
	tools     *tools.Registry
	skills    *skills.Catalog
	plugins   *skills.PluginRegistry
	compactor *compact.Engine

	runsMu sync.Mutex
	runs   map[string]*activeRun

	subMu       sync.RWMutex
	subscribers map[string]map[chan schema.SessionEvent]struct{}
}

func New(
	cfg Config,
	st store.Store,
	providers *provider.Registry,
	toolReg *tools.Registry,
	skillCat *skills.Catalog,
	pluginReg *skills.PluginRegistry,
) *Engine {
	if cfg.MaxStepsPerTurn <= 0 {
		cfg.MaxStepsPerTurn = 32
	}
	if st == nil {
		st = store.NewMemoryStore()
	}
	if providers == nil {
		providers = provider.NewRegistry()
	}
	if skillCat == nil {
		skillCat = skills.NewCatalog()
	}
	if pluginReg == nil {
		pluginReg = skills.NewPluginRegistry()
	}
	if toolReg == nil {
		toolReg = tools.NewRegistry(skillCat, pluginReg)
	}
	return &Engine{
		cfg:         cfg,
		store:       st,
		providers:   providers,
		tools:       toolReg,
		skills:      skillCat,
		plugins:     pluginReg,
		compactor:   compact.NewEngine(compact.Config{}),
		runs:        make(map[string]*activeRun),
		subscribers: make(map[string]map[chan schema.SessionEvent]struct{}),
	}
}

func (e *Engine) Store() store.Store              { return e.store }
func (e *Engine) Providers() *provider.Registry   { return e.providers }
func (e *Engine) Tools() *tools.Registry          { return e.tools }
func (e *Engine) Skills() *skills.Catalog         { return e.skills }
func (e *Engine) Plugins() *skills.PluginRegistry { return e.plugins }

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

// CompactThread runs an explicit provider-aware compaction checkpoint on threadID.
func (e *Engine) CompactThread(ctx context.Context, threadID string) (schema.CompactionCheckpoint, bool, error) {
	th, err := e.store.GetThread(ctx, threadID)
	if err != nil {
		return schema.CompactionCheckpoint{}, false, err
	}
	p, ok := e.providers.Get(th.Provider)
	if !ok {
		return schema.CompactionCheckpoint{}, false, fmt.Errorf("provider %q not registered", th.Provider)
	}
	msgs, err := e.store.ListMessages(ctx, threadID, 0)
	if err != nil {
		return schema.CompactionCheckpoint{}, false, err
	}
	var latestPtr *schema.CompactionCheckpoint
	if cp, found, _ := e.store.LatestCompaction(ctx, threadID); found {
		latestPtr = &cp
	}
	active := compact.ApplyCheckpoint(latestPtr, msgs, p.WireFormat())
	sysPrompt := e.assembleSystemPrompt(th)
	tokensBefore := compact.EstimateTokens(sysPrompt, active)
	cp, didCompact, err := e.compactor.Compact(ctx, p, threadID, th.Model, sysPrompt, active, tokensBefore)
	if err != nil || !didCompact {
		return cp, didCompact, err
	}
	saved, err := e.store.SaveCompaction(ctx, cp)
	if err != nil {
		return schema.CompactionCheckpoint{}, false, err
	}
	e.publishEvent(ctx, threadID, th.TurnEpoch, schema.EventCompacted, map[string]any{
		"compaction_id":      saved.ID,
		"replaces_up_to_seq": saved.ReplacesUpToSeq,
		"tokens_before":      saved.TokensBefore,
		"tokens_after":       saved.TokensAfter,
	})
	return saved, true, nil
}

// RunTurn executes a complete multi-step agent turn synchronously on threadID.
func (e *Engine) RunTurn(ctx context.Context, req TurnRequest) (TurnResult, error) {
	th, err := e.store.GetThread(ctx, req.ThreadID)
	if err != nil {
		return TurnResult{}, err
	}

	if strings.TrimSpace(req.Provider) != "" {
		th.Provider = strings.TrimSpace(req.Provider)
	}
	if strings.TrimSpace(req.Model) != "" {
		th.Model = strings.TrimSpace(req.Model)
	}
	if strings.TrimSpace(req.ReasoningEffort) != "" {
		th.ReasoningEffort = strings.TrimSpace(req.ReasoningEffort)
	}

	p, ok := e.providers.Get(th.Provider)
	if !ok {
		return TurnResult{}, fmt.Errorf("provider %q not registered", th.Provider)
	}

	runCtx, cancel := context.WithCancel(ctx)
	active := &activeRun{cancel: cancel}

	e.runsMu.Lock()
	if _, busy := e.runs[th.ID]; busy {
		e.runsMu.Unlock()
		cancel()
		return TurnResult{}, fmt.Errorf("thread %s already has an active turn", th.ID)
	}
	e.runs[th.ID] = active
	e.runsMu.Unlock()

	defer func() {
		cancel()
		e.runsMu.Lock()
		delete(e.runs, th.ID)
		e.runsMu.Unlock()
	}()

	th.TurnEpoch++
	th.Status = "running"
	if err := e.store.UpdateThread(ctx, th); err != nil {
		return TurnResult{}, err
	}
	epoch := th.TurnEpoch

	e.publishEvent(ctx, th.ID, epoch, schema.EventTurnStarted, map[string]any{
		"provider": th.Provider,
		"model":    th.Model,
		"epoch":    epoch,
	})

	userBlocks := make([]schema.ContentBlock, 0, 1+len(req.Images))
	if strings.TrimSpace(req.Prompt) != "" {
		userBlocks = append(userBlocks, schema.ContentBlock{
			Type: schema.BlockText,
			Text: req.Prompt,
		})
	}
	for _, img := range req.Images {
		cp := img
		userBlocks = append(userBlocks, schema.ContentBlock{
			Type:  schema.BlockImage,
			Image: &cp,
		})
	}
	if len(userBlocks) > 0 {
		if _, err := e.store.AppendMessage(ctx, th.ID, schema.Message{
			TurnEpoch: epoch,
			Role:      schema.RoleUser,
			Blocks:    userBlocks,
		}); err != nil {
			return TurnResult{}, err
		}
	}

	sysPrompt := e.assembleSystemPrompt(th)
	modelSpec := schema.LookupModel(th.Provider, th.Model)
	useNativeToolSearch := e.cfg.NativeToolSearch && modelSpec.SupportsNativeToolSearch && p.WireFormat() == "anthropic_messages"
	activeTools := e.tools.ActiveToolDefs(useNativeToolSearch)
	hostedTools := e.resolveHostedTools(th.Provider, req)

	var (
		totalUsage               schema.Usage
		lastInputTokens          int64
		assistantText            strings.Builder
		compacted                bool
		steps                    int
		pendingContinuationToken string
	)
	heartbeat := func(detail string) {
		if req.Heartbeat != nil {
			req.Heartbeat(detail)
		}
	}

	finishErr := func(status string, runErr error) (TurnResult, error) {
		th.Status = status
		_ = e.store.UpdateThread(context.Background(), th)
		evType := schema.EventTurnFailed
		if status == "interrupted" {
			evType = schema.EventTurnInterrupted
		}
		e.publishEvent(context.Background(), th.ID, epoch, evType, map[string]any{
			"status": status,
			"error":  runErr.Error(),
		})
		return TurnResult{
			ThreadID:      th.ID,
			TurnEpoch:     epoch,
			Status:        status,
			AssistantText: assistantText.String(),
			Usage:         totalUsage,
			Steps:         steps,
			Compacted:     compacted,
			ErrorMessage:  runErr.Error(),
		}, runErr
	}

	for step := 1; step <= e.cfg.MaxStepsPerTurn; step++ {
		steps = step
		if runCtx.Err() != nil {
			return finishErr("interrupted", runCtx.Err())
		}
		heartbeat(fmt.Sprintf("step:%d", step))

		allMsgs, err := e.store.ListMessages(runCtx, th.ID, 0)
		if err != nil {
			return finishErr("failed", err)
		}
		var latestCP *schema.CompactionCheckpoint
		if cp, found, _ := e.store.LatestCompaction(runCtx, th.ID); found {
			latestCP = &cp
		}
		visible := compact.ApplyCheckpoint(latestCP, allMsgs, p.WireFormat())

		// Tier-1 + Tier-2/3 auto-compaction when approaching the model's context window.
		if e.cfg.AutoCompact && e.compactor.ShouldCompact(modelSpec, lastInputTokens, sysPrompt, visible) {
			pruned := e.compactor.PruneOldToolOutputs(visible)
			if e.compactor.ShouldCompact(modelSpec, 0, sysPrompt, pruned) {
				cp, did, cerr := e.compactor.Compact(runCtx, p, th.ID, th.Model, sysPrompt, pruned, lastInputTokens)
				if cerr == nil && did {
					saved, serr := e.store.SaveCompaction(runCtx, cp)
					if serr == nil {
						compacted = true
						visible = compact.ApplyCheckpoint(&saved, allMsgs, p.WireFormat())
						e.publishEvent(runCtx, th.ID, epoch, schema.EventCompacted, map[string]any{
							"compaction_id":      saved.ID,
							"replaces_up_to_seq": saved.ReplacesUpToSeq,
							"tokens_before":      saved.TokensBefore,
							"tokens_after":       saved.TokensAfter,
						})
					}
				}
			} else {
				visible = pruned
			}
		}

		maxOut := req.MaxOutputTokens
		if maxOut <= 0 {
			maxOut = modelSpec.MaxOutputTokens
		}

		var meta map[string]string
		if pendingContinuationToken != "" {
			meta = map[string]string{"continuation_token": pendingContinuationToken}
			pendingContinuationToken = ""
		}

		provReq := provider.Request{
			ThreadID:          th.ID,
			Model:             th.Model,
			SystemPrompt:      sysPrompt,
			Messages:          visible,
			Tools:             activeTools,
			HostedTools:       hostedTools,
			ReasoningEffort:   th.ReasoningEffort,
			MaxOutputTokens:   maxOut,
			CacheKey:          th.ID,
			CachedContentName: req.CachedContent,
			EnableNativeCache: e.cfg.EnableNativeCache,
			NativeToolSearch:  useNativeToolSearch,
			Metadata:          meta,
		}

		resp, err := p.Stream(runCtx, provReq, func(ch provider.StreamChunk) {
			heartbeat("stream:" + string(ch.Kind))
			switch ch.Kind {
			case provider.DeltaText:
				assistantText.WriteString(ch.Text)
				e.publishEvent(runCtx, th.ID, epoch, schema.EventAssistantDelta, map[string]any{
					"delta": ch.Text,
				})
			case provider.DeltaThinking:
				e.publishEvent(runCtx, th.ID, epoch, schema.EventThinkingDelta, map[string]any{
					"delta": ch.Text,
				})
			case provider.DeltaHostedToolStart:
				e.publishEvent(runCtx, th.ID, epoch, schema.EventHostedToolActivity, map[string]any{
					"call_id": ch.CallID,
					"tool":    ch.ToolName,
					"status":  "running",
				})
			case provider.DeltaHostedToolComplete:
				e.publishEvent(runCtx, th.ID, epoch, schema.EventHostedToolActivity, map[string]any{
					"call_id": ch.CallID,
					"tool":    ch.ToolName,
					"status":  "completed",
				})
			}
		})
		if err != nil {
			if runCtx.Err() != nil {
				return finishErr("interrupted", runCtx.Err())
			}
			return finishErr("failed", err)
		}

		if resp.Usage.CostUSD == 0 && (resp.Usage.InputTokens > 0 || resp.Usage.OutputTokens > 0) {
			resp.Usage.CostUSD = modelSpec.EstimateCostUSD(resp.Usage)
		}
		totalUsage.Add(resp.Usage)
		if resp.Usage.InputTokens > 0 {
			lastInputTokens = resp.Usage.InputTokens
		}
		e.publishEvent(runCtx, th.ID, epoch, schema.EventUsageUpdated, map[string]any{
			"input_tokens":          resp.Usage.InputTokens,
			"cached_input_tokens":   resp.Usage.CachedInputTokens,
			"cache_creation_tokens": resp.Usage.CacheCreationTokens,
			"output_tokens":         resp.Usage.OutputTokens,
			"reasoning_tokens":      resp.Usage.ReasoningTokens,
			"cost_usd":              resp.Usage.CostUSD,
		})

		usageCopy := resp.Usage
		if _, err := e.store.AppendMessage(runCtx, th.ID, schema.Message{
			TurnEpoch: epoch,
			Role:      schema.RoleAssistant,
			Blocks:    resp.Blocks,
			Native:    resp.Native,
			Usage:     &usageCopy,
		}); err != nil {
			return finishErr("failed", err)
		}

		// Collect local tool calls (hosted tools were already executed server-side by the LLM provider).
		var calls []schema.ContentBlock
		for _, b := range resp.Blocks {
			if b.Type == schema.BlockToolCall {
				calls = append(calls, b)
			}
		}

		if len(calls) > 0 {
			heartbeat(fmt.Sprintf("tools:%d", len(calls)))
			results := e.executeToolCalls(runCtx, th.ID, epoch, calls)
			if _, err := e.store.AppendMessage(runCtx, th.ID, schema.Message{
				TurnEpoch: epoch,
				Role:      schema.RoleTool,
				Blocks:    results,
			}); err != nil {
				return finishErr("failed", err)
			}
			// Fold in any live user steers queued while tools ran.
			if steers := active.drainSteers(); len(steers) > 0 {
				_, _ = e.store.AppendMessage(runCtx, th.ID, schema.Message{
					TurnEpoch: epoch,
					Role:      schema.RoleUser,
					Blocks:    []schema.ContentBlock{{Type: schema.BlockText, Text: strings.Join(steers, "\n")}},
				})
			}
			continue
		}

		// Gemini 4 long-decode continuation: resume automatically with continuation token.
		if tok := strings.TrimSpace(resp.ContinuationToken); tok != "" {
			pendingContinuationToken = tok
			heartbeat("continuation")
			continue
		}

		// If the user steered just as the model finished text output, run one more step.
		if steers := active.drainSteers(); len(steers) > 0 {
			_, _ = e.store.AppendMessage(runCtx, th.ID, schema.Message{
				TurnEpoch: epoch,
				Role:      schema.RoleUser,
				Blocks:    []schema.ContentBlock{{Type: schema.BlockText, Text: strings.Join(steers, "\n")}},
			})
			continue
		}

		th.Status = "idle"
		_ = e.store.UpdateThread(ctx, th)
		e.publishEvent(ctx, th.ID, epoch, schema.EventTurnCompleted, map[string]any{
			"status": "completed",
			"steps":  steps,
		})
		return TurnResult{
			ThreadID:      th.ID,
			TurnEpoch:     epoch,
			Status:        "completed",
			AssistantText: assistantText.String(),
			Usage:         totalUsage,
			Steps:         steps,
			Compacted:     compacted,
		}, nil
	}

	return finishErr("failed", fmt.Errorf("turn exceeded maximum step budget (%d)", e.cfg.MaxStepsPerTurn))
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
			out, err := e.tools.Execute(ctx, call.Name, call.Arguments)
			isErr := false
			if err != nil {
				isErr = true
				out = err.Error()
			}
			status := "completed"
			if isErr {
				status = "failed"
			}
			e.publishEvent(ctx, threadID, epoch, schema.EventToolCallResult, map[string]any{
				"call_id":  call.CallID,
				"tool":     call.Name,
				"status":   status,
				"output":   out,
				"is_error": isErr,
			})
			results[i] = schema.ContentBlock{
				Type:    schema.BlockToolResult,
				CallID:  call.CallID,
				Name:    call.Name,
				Output:  out,
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

func (e *Engine) resolveHostedTools(providerName string, req TurnRequest) []schema.HostedTool {
	if len(req.HostedTools) > 0 {
		return req.HostedTools
	}
	enable := e.cfg.DefaultHostedTools
	if req.EnableWebSearch != nil {
		enable = *req.EnableWebSearch
	}
	if !enable {
		return nil
	}
	switch strings.ToLower(strings.TrimSpace(providerName)) {
	case "openai", "codex_oauth", "azure_openai":
		return []schema.HostedTool{{Type: "web_search"}}
	case "anthropic", "vertex_anthropic", "bedrock_anthropic":
		return []schema.HostedTool{
			{Type: "web_search_20250305", Name: "web_search"},
			{Type: "web_fetch_20250910", Name: "web_fetch"},
		}
	case "grok", "xai":
		return []schema.HostedTool{
			{Type: "x_search", Config: map[string]any{"enable_image_understanding": true}},
			{Type: "web_search"},
		}
	case "gemini", "vertex", "vertex_gemini":
		return []schema.HostedTool{
			{Type: "google_search"},
			{Type: "url_context"},
		}
	default:
		return nil
	}
}
