package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/cambridgeadvancedsystems/jaz-harness/compact"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/cambridgeadvancedsystems/jaz-harness/skills"
	"github.com/cambridgeadvancedsystems/jaz-harness/store"
	"github.com/cambridgeadvancedsystems/jaz-harness/tools"
)

type Config struct {
	MaxStepsPerTurn        int
	EnableNativeCache      bool
	DefaultHostedTools     bool
	NativeToolSearch       bool // use Anthropic native defer_loading + tool_search_tool_bm25 when on Anthropic
	AutoCompact            bool // automatically commit durable checkpoints near context window
	NativeServerCompaction bool // enable provider-native server compaction (Anthropic compact-2026-01-12 / OpenAI context_management)
	CompactionConfig       compact.Config
	BaseSystemPrompt       string
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
		compactor:   compact.NewEngine(cfg.CompactionConfig),
		runs:        make(map[string]*activeRun),
		subscribers: make(map[string]map[chan schema.SessionEvent]struct{}),
	}
}

func (e *Engine) Store() store.Store              { return e.store }
func (e *Engine) Providers() *provider.Registry   { return e.providers }
func (e *Engine) Tools() *tools.Registry          { return e.tools }
func (e *Engine) Skills() *skills.Catalog         { return e.skills }
func (e *Engine) Plugins() *skills.PluginRegistry { return e.plugins }

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

	userBlocks := buildUserBlocks(req.Prompt, req.Images)
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
	provCompaction := e.buildProviderCompaction(th.Provider, modelSpec)

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
		// Never mutate old messages in-place; prefixes remain byte-identical
		// until a durable compaction checkpoint is committed.
		visible := compact.ApplyCheckpoint(latestCP, allMsgs, p.WireFormat())

		if e.cfg.AutoCompact && e.compactor.ShouldCompact(modelSpec, lastInputTokens, sysPrompt, visible) {
			cp, did, cerr := e.compactor.Compact(runCtx, p, th.ID, th.Model, sysPrompt, visible, lastInputTokens)
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
		}

		var prevSeq int64
		if len(allMsgs) > 0 {
			prevSeq = allMsgs[len(allMsgs)-1].Seq
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
			Compaction:        provCompaction,
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

		// If the provider performed server-side compaction (e.g. Anthropic
		// compact-2026-01-12 or OpenAI context_management), persist a durable
		// checkpoint covering everything prior to this response so future turns
		// and cross-provider switches remain consistent.
		savedBlocks, savedNative, serverCompacted := e.persistServerCompaction(runCtx, th, epoch, p, prevSeq, resp, visible)
		if serverCompacted {
			compacted = true
		}

		if len(savedBlocks) > 0 {
			usageCopy := resp.Usage
			if _, err := e.store.AppendMessage(runCtx, th.ID, schema.Message{
				TurnEpoch: epoch,
				Role:      schema.RoleAssistant,
				Blocks:    savedBlocks,
				Native:    savedNative,
				Usage:     &usageCopy,
			}); err != nil {
				return finishErr("failed", err)
			}
		}

		// Collect local tool calls (hosted tools were already executed server-side by the LLM provider).
		var calls []schema.ContentBlock
		for _, b := range savedBlocks {
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
			if steers := active.drainSteers(); len(steers) > 0 {
				_, _ = e.store.AppendMessage(runCtx, th.ID, schema.Message{
					TurnEpoch: epoch,
					Role:      schema.RoleUser,
					Blocks:    []schema.ContentBlock{{Type: schema.BlockText, Text: strings.Join(steers, "\n")}},
				})
			}
			continue
		}

		// Anthropic native pause_after_compaction: append a user resume cue and continue cleanly.
		if strings.EqualFold(strings.TrimSpace(resp.StopReason), "compaction") {
			_, _ = e.store.AppendMessage(runCtx, th.ID, schema.Message{
				TurnEpoch: epoch,
				Role:      schema.RoleUser,
				Blocks:    []schema.ContentBlock{{Type: schema.BlockText, Text: "Continue."}},
			})
			heartbeat("compaction_resume")
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

func (e *Engine) persistServerCompaction(
	ctx context.Context,
	th schema.Thread,
	epoch int64,
	p provider.Provider,
	prevSeq int64,
	resp provider.Response,
	visibleBefore []schema.Message,
) ([]schema.ContentBlock, *schema.NativeEnvelope, bool) {
	var (
		compactBlock *schema.ContentBlock
		remaining    []schema.ContentBlock
	)
	for i := range resp.Blocks {
		if resp.Blocks[i].Type == schema.BlockCompaction && compactBlock == nil {
			cp := resp.Blocks[i]
			compactBlock = &cp
			continue
		}
		remaining = append(remaining, resp.Blocks[i])
	}
	if compactBlock == nil {
		return resp.Blocks, resp.Native, false
	}

	summaryText := strings.TrimSpace(compactBlock.Text)
	if summaryText == "" {
		summaryText = compact.BuildFallbackSummary(visibleBefore)
	}
	savedCP, err := e.store.SaveCompaction(ctx, schema.CompactionCheckpoint{
		ThreadID:        th.ID,
		ReplacesUpToSeq: prevSeq,
		Provider:        th.Provider,
		WireFormat:      p.WireFormat(),
		SummaryText:     summaryText,
		OpaqueItem:      compactBlock.ProviderOpaque,
		TokensBefore:    resp.Usage.InputTokens,
	})
	if err != nil {
		return resp.Blocks, resp.Native, false
	}
	e.publishEvent(ctx, th.ID, epoch, schema.EventCompacted, map[string]any{
		"compaction_id":      savedCP.ID,
		"replaces_up_to_seq": savedCP.ReplacesUpToSeq,
		"provider_native":    true,
	})

	var trimmedNative *schema.NativeEnvelope
	if resp.Native != nil {
		var keptRaw []json.RawMessage
		for _, raw := range resp.Native.RawItems {
			var meta struct {
				Type string `json:"type"`
			}
			if json.Unmarshal(raw, &meta) == nil && meta.Type == "compaction" {
				continue
			}
			keptRaw = append(keptRaw, raw)
		}
		if len(keptRaw) > 0 {
			cp := *resp.Native
			cp.RawItems = keptRaw
			trimmedNative = &cp
		}
	}
	return remaining, trimmedNative, true
}
