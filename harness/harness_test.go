package harness_test

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/cambridgeadvancedsystems/jaz-harness/harness"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/cambridgeadvancedsystems/jaz-harness/skills"
	"github.com/cambridgeadvancedsystems/jaz-harness/store"
	"github.com/cambridgeadvancedsystems/jaz-harness/tools"
)

type scriptedProvider struct {
	id         string
	wireFormat string
	mu         sync.Mutex
	calls      []provider.Request
	responses  []provider.Response
}

func (p *scriptedProvider) ID() string         { return p.id }
func (p *scriptedProvider) WireFormat() string { return p.wireFormat }

func (p *scriptedProvider) Stream(_ context.Context, req provider.Request, emit provider.StreamCallback) (provider.Response, error) {
	p.mu.Lock()
	p.calls = append(p.calls, req)
	idx := len(p.calls) - 1
	resp := p.responses[idx]
	p.mu.Unlock()

	for _, b := range resp.Blocks {
		if b.Type == schema.BlockText && emit != nil {
			emit(provider.StreamChunk{Kind: provider.DeltaText, Text: b.Text})
		}
	}
	return resp, nil
}

func TestEndToEndMultiProviderSwitchCompactionAndFilePersistence(t *testing.T) {
	dataDir := t.TempDir()
	st, err := store.NewFileStore(dataDir)
	if err != nil {
		t.Fatal(err)
	}

	pGemini := &scriptedProvider{
		id:         "vertex",
		wireFormat: "gemini_generate_content",
		responses: []provider.Response{
			{
				ID:    "g1",
				Model: "gemini-4-neon",
				Blocks: []schema.ContentBlock{
					{Type: schema.BlockThinking, Thinking: "plan", ThoughtSignBlob: "SIG_G1"},
					{Type: schema.BlockToolCall, CallID: "c1", Name: "search_tools", Arguments: json.RawMessage(`{"query":"inventory"}`), ThoughtSignBlob: "SIG_FC1"},
				},
				Native: &schema.NativeEnvelope{
					Provider:   "vertex",
					WireFormat: "gemini_generate_content",
					RawItems:   []json.RawMessage{json.RawMessage(`{"role":"model","parts":[{"functionCall":{"name":"search_tools"},"thoughtSignature":"SIG_FC1"}]}`)},
				},
				Usage: schema.Usage{InputTokens: 200, OutputTokens: 30},
			},
			{
				ID:     "g2",
				Model:  "gemini-4-neon",
				Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Found inventory tool."}},
				Usage:  schema.Usage{InputTokens: 250, CachedInputTokens: 180, OutputTokens: 20},
			},
		},
	}

	pClaude := &scriptedProvider{
		id:         "anthropic",
		wireFormat: "anthropic_messages",
		responses: []provider.Response{
			{
				ID:     "a1",
				Model:  "claude-sonnet-4-5",
				Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Continuing seamlessly on Claude."}},
				Usage:  schema.Usage{InputTokens: 300, CachedInputTokens: 200, OutputTokens: 25},
			},
			// Compaction summary call
			{
				ID:     "a_cmp",
				Model:  "claude-sonnet-4-5",
				Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Summary: User searched inventory on Gemini and continued on Claude."}},
				Usage:  schema.Usage{InputTokens: 200, OutputTokens: 30},
			},
			// Follow-up after compaction
			{
				ID:     "a2",
				Model:  "claude-sonnet-4-5",
				Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Post-compaction reply."}},
				Usage:  schema.Usage{InputTokens: 120, OutputTokens: 15},
			},
		},
	}

	provReg := provider.NewRegistry()
	provReg.Register("vertex", pGemini)
	provReg.Register("anthropic", pClaude)

	cat := skills.NewCatalog()
	plugs := skills.NewPluginRegistry()
	toolReg := tools.NewRegistry(cat, plugs)
	toolReg.Register(schema.ToolDef{
		Name:         "erp_inventory_lookup",
		Description:  "Check warehouse inventory stock",
		DeferLoading: true,
	}, func(_ context.Context, _ json.RawMessage) (string, error) {
		return "42 units", nil
	})

	eng := harness.New(harness.Config{
		EnableNativeCache:  true,
		DefaultHostedTools: true,
	}, st, provReg, toolReg, cat, plugs)

	ctx := context.Background()
	th, err := eng.CreateThread(ctx, schema.Thread{
		Provider: "vertex",
		Model:    "gemini-4-neon",
	})
	if err != nil {
		t.Fatal(err)
	}

	res1, err := eng.RunTurn(ctx, harness.TurnRequest{ThreadID: th.ID, Prompt: "Check inventory tools"})
	if err != nil || res1.Steps != 2 || !strings.Contains(res1.AssistantText, "Found inventory tool") {
		t.Fatalf("turn 1 failed: err=%v res=%+v", err, res1)
	}
	if len(pGemini.calls) != 2 || pGemini.calls[1].Messages[1].Native == nil {
		t.Fatal("expected Gemini step 2 to receive verbatim NativeEnvelope")
	}

	// Simulate worker restart by opening a brand-new Engine on the same FileStore,
	// then switch the thread mid-conversation from Gemini -> Anthropic!
	st2, err := store.NewFileStore(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	eng2 := harness.New(harness.Config{EnableNativeCache: true}, st2, provReg, toolReg, cat, plugs)
	res2, err := eng2.RunTurn(ctx, harness.TurnRequest{
		ThreadID: th.ID,
		Provider: "anthropic",
		Model:    "claude-sonnet-4-5",
		Prompt:   "Now continue on Claude",
	})
	if err != nil || !strings.Contains(res2.AssistantText, "Continuing seamlessly on Claude") {
		t.Fatalf("cross-provider turn 2 failed: err=%v res=%+v", err, res2)
	}
	// Verify Claude received all prior turns from disk!
	if len(pClaude.calls) != 1 || len(pClaude.calls[0].Messages) < 4 {
		t.Fatalf("expected Claude to receive full replayed history after restart, got %d msgs", len(pClaude.calls[0].Messages))
	}

	// Add enough turns and verify compaction checkpoint replaces prefix cleanly.
	for range 4 {
		_, _ = st2.AppendMessage(ctx, th.ID, schema.Message{
			Role:   schema.RoleUser,
			Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "filler"}},
		})
	}
	cp, did, err := eng2.CompactThread(ctx, th.ID)
	if err != nil || !did || !strings.Contains(cp.SummaryText, "Summary:") {
		t.Fatalf("CompactThread failed: err=%v did=%v cp=%+v", err, did, cp)
	}

	res3, err := eng2.RunTurn(ctx, harness.TurnRequest{ThreadID: th.ID, Prompt: "After compact"})
	if err != nil || !strings.Contains(res3.AssistantText, "Post-compaction reply") {
		t.Fatalf("post-compaction turn failed: err=%v res=%+v", err, res3)
	}
	lastClaudeCall := pClaude.calls[len(pClaude.calls)-1]
	if lastClaudeCall.Messages[0].Blocks[0].Type != schema.BlockCompaction {
		t.Fatalf("expected compacted checkpoint as first visible message, got %+v", lastClaudeCall.Messages[0])
	}
}

func TestHeartbeatAndGeminiContinuationLoop(t *testing.T) {
	pGem := &scriptedProvider{
		id:         "vertex",
		wireFormat: "gemini_interactions",
		responses: []provider.Response{
			{
				ID:                "i1",
				Model:             "gemini-4-neon",
				Blocks:            []schema.ContentBlock{{Type: schema.BlockText, Text: "First half. "}},
				StopReason:        "requires_continuation",
				ContinuationToken: "tok_part2",
			},
			{
				ID:         "i2",
				Model:      "gemini-4-neon",
				Blocks:     []schema.ContentBlock{{Type: schema.BlockText, Text: "Second half."}},
				StopReason: "completed",
			},
		},
	}

	provReg := provider.NewRegistry()
	provReg.Register("vertex", pGem)
	eng := harness.New(harness.Config{}, store.NewMemoryStore(), provReg, nil, nil, nil)
	ctx := context.Background()
	th, err := eng.CreateThread(ctx, schema.Thread{Provider: "vertex", Model: "gemini-4-neon"})
	if err != nil {
		t.Fatal(err)
	}

	var beats []string
	res, err := eng.RunTurn(ctx, harness.TurnRequest{
		ThreadID: th.ID,
		Prompt:   "Generate long output",
		Heartbeat: func(detail string) {
			beats = append(beats, detail)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Steps != 2 || res.AssistantText != "First half. Second half." {
		t.Fatalf("expected 2-step continuation, got steps=%d text=%q", res.Steps, res.AssistantText)
	}
	if len(pGem.calls) != 2 || pGem.calls[1].Metadata["continuation_token"] != "tok_part2" {
		t.Fatalf("expected second call to carry continuation_token, got %+v", pGem.calls[1].Metadata)
	}
	if len(beats) < 3 {
		t.Fatalf("expected heartbeat callbacks during stream and continuation, got %v", beats)
	}
}

func TestKVCachePrefixImmutabilityAndZeroMidThreadSchemaMutation(t *testing.T) {
	bigToolOut := strings.Repeat("telemetry-row-0123456789\n", 120)

	pOpenAI := &scriptedProvider{
		id:         "openai",
		wireFormat: "openai_responses",
		responses: []provider.Response{
			{
				ID:    "r1",
				Model: "gpt-5.4",
				Blocks: []schema.ContentBlock{
					{Type: schema.BlockToolCall, CallID: "c1", Name: "search_tools", Arguments: json.RawMessage(`{"query":"metrics"}`)},
				},
				Usage: schema.Usage{InputTokens: 200, OutputTokens: 20},
			},
			{
				ID:    "r2",
				Model: "gpt-5.4",
				Blocks: []schema.ContentBlock{
					{Type: schema.BlockToolCall, CallID: "c2", Name: "call_tool", Arguments: json.RawMessage(`{"name":"obs_dump_metrics","arguments":{}}`)},
				},
				Usage: schema.Usage{InputTokens: 400, OutputTokens: 20},
			},
			{
				ID:     "r3",
				Model:  "gpt-5.4",
				Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "All metrics verified."}},
				Usage:  schema.Usage{InputTokens: 1200, OutputTokens: 15},
			},
		},
	}

	provReg := provider.NewRegistry()
	provReg.Register("openai", pOpenAI)
	cat := skills.NewCatalog()
	plugs := skills.NewPluginRegistry()
	toolReg := tools.NewRegistry(cat, plugs)
	toolReg.RegisterPluginTool("obs", schema.ToolDef{
		Name:        "obs_dump_metrics",
		Description: "Dump live service metrics",
	}, func(_ context.Context, _ json.RawMessage) (string, error) {
		return bigToolOut, nil
	})

	eng := harness.New(harness.Config{
		EnableNativeCache: true,
		AutoCompact:       true,
	}, store.NewMemoryStore(), provReg, toolReg, cat, plugs)

	ctx := context.Background()
	th, err := eng.CreateThread(ctx, schema.Thread{Provider: "openai", Model: "gpt-5.4"})
	if err != nil {
		t.Fatal(err)
	}
	res, err := eng.RunTurn(ctx, harness.TurnRequest{ThreadID: th.ID, Prompt: "Inspect metrics"})
	if err != nil || res.Steps != 3 {
		t.Fatalf("turn failed: err=%v res=%+v", err, res)
	}

	// 1. Tool schemas MUST be byte-identical across every step (zero mid-thread mutation).
	tools0, _ := json.Marshal(pOpenAI.calls[0].Tools)
	for i := 1; i < len(pOpenAI.calls); i++ {
		toolsI, _ := json.Marshal(pOpenAI.calls[i].Tools)
		if string(tools0) != string(toolsI) {
			t.Fatalf("tool schema mutated between step 0 and step %d!\n0=%s\n%d=%s", i, tools0, i, toolsI)
		}
	}

	// 2. Prior messages prefix MUST remain byte-identical across steps (no in-place mutation).
	prefix1, _ := json.Marshal(pOpenAI.calls[1].Messages)
	prefix2, _ := json.Marshal(pOpenAI.calls[2].Messages[:len(pOpenAI.calls[1].Messages)])
	if string(prefix1) != string(prefix2) {
		t.Fatalf("KV prefix mutated between step 1 and step 2!\nstep1=%s\nstep2=%s", prefix1, prefix2)
	}
}

func TestAnthropicServerCompactionPauseResumeAndCrossProviderSwitch(t *testing.T) {
	pAnthropic := &scriptedProvider{
		id:         "anthropic",
		wireFormat: "anthropic_messages",
		responses: []provider.Response{
			{
				ID:         "a_cmp_pause",
				Model:      "claude-sonnet-4-5",
				StopReason: "compaction",
				Blocks: []schema.ContentBlock{
					{
						Type:           schema.BlockCompaction,
						Text:           "Server compacted: user migrating billing DB to Postgres 17.",
						ProviderOpaque: json.RawMessage(`{"type":"compaction","content":"Server compacted: user migrating billing DB to Postgres 17."}`),
					},
				},
				Usage: schema.Usage{InputTokens: 160000, OutputTokens: 200},
			},
			{
				ID:         "a_resume",
				Model:      "claude-sonnet-4-5",
				StopReason: "end_turn",
				Blocks:     []schema.ContentBlock{{Type: schema.BlockText, Text: "Continuing Postgres 17 migration."}},
				Usage:      schema.Usage{InputTokens: 1200, OutputTokens: 25},
			},
		},
	}
	pGemini := &scriptedProvider{
		id:         "vertex",
		wireFormat: "gemini_generate_content",
		responses: []provider.Response{
			{
				ID:     "g_after",
				Model:  "gemini-4-neon",
				Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Gemini picked up migration context."}},
			},
		},
	}

	provReg := provider.NewRegistry()
	provReg.Register("anthropic", pAnthropic)
	provReg.Register("vertex", pGemini)
	eng := harness.New(harness.Config{
		EnableNativeCache:      true,
		NativeServerCompaction: true,
	}, store.NewMemoryStore(), provReg, nil, nil, nil)

	ctx := context.Background()
	th, err := eng.CreateThread(ctx, schema.Thread{Provider: "anthropic", Model: "claude-sonnet-4-5"})
	if err != nil {
		t.Fatal(err)
	}
	res1, err := eng.RunTurn(ctx, harness.TurnRequest{ThreadID: th.ID, Prompt: "Next migration step?"})
	if err != nil || !res1.Compacted || res1.Steps != 2 || !strings.Contains(res1.AssistantText, "Continuing Postgres 17 migration") {
		t.Fatalf("expected auto-resume after Anthropic compaction pause: err=%v res=%+v", err, res1)
	}
	if pAnthropic.calls[0].Compaction == nil || !pAnthropic.calls[0].Compaction.Enabled {
		t.Fatalf("expected NativeServerCompaction request config on Anthropic, got %+v", pAnthropic.calls[0].Compaction)
	}

	// Switch thread mid-conversation from Anthropic -> Gemini and verify compacted summary carried over.
	res2, err := eng.RunTurn(ctx, harness.TurnRequest{
		ThreadID: th.ID,
		Provider: "vertex",
		Model:    "gemini-4-neon",
		Prompt:   "Verify rollback plan",
	})
	if err != nil || !strings.Contains(res2.AssistantText, "Gemini picked up migration context") {
		t.Fatalf("cross-provider turn failed: err=%v res=%+v", err, res2)
	}
	if len(pGemini.calls) != 1 || !strings.Contains(pGemini.calls[0].Messages[0].Blocks[0].Text, "Postgres 17") {
		t.Fatalf("expected Gemini to receive readable compacted summary, got %+v", pGemini.calls[0].Messages)
	}
}
