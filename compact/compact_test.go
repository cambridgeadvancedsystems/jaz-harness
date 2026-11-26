package compact

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
)

type fakeProvider struct {
	id         string
	wireFormat string
	calls      []provider.Request
	responses  []provider.Response
	compactRes *provider.CompactResponse
}

func (f *fakeProvider) ID() string         { return f.id }
func (f *fakeProvider) WireFormat() string { return f.wireFormat }

func (f *fakeProvider) Stream(_ context.Context, req provider.Request, _ provider.StreamCallback) (provider.Response, error) {
	f.calls = append(f.calls, req)
	idx := len(f.calls) - 1
	return f.responses[idx], nil
}

func (f *fakeProvider) Compact(_ context.Context, _ provider.CompactRequest) (provider.CompactResponse, error) {
	if f.compactRes == nil {
		return provider.CompactResponse{}, provider.ErrCompactionUnsupported
	}
	return *f.compactRes, nil
}

func TestSafeSplitNeverBreaksToolPairs(t *testing.T) {
	eng := NewEngine(Config{TailMessages: 2})
	msgs := []schema.Message{
		{Seq: 1, Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "u1"}}},
		{Seq: 2, Role: schema.RoleAssistant, Blocks: []schema.ContentBlock{{Type: schema.BlockToolCall, CallID: "c1", Name: "read_file"}}},
		{Seq: 3, Role: schema.RoleTool, Blocks: []schema.ContentBlock{{Type: schema.BlockToolResult, CallID: "c1", Output: "data"}}},
		{Seq: 4, Role: schema.RoleAssistant, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "done"}}},
	}

	idx := eng.SafeSplitIndex(msgs)
	if idx != 1 {
		t.Fatalf("SafeSplitIndex = %d, want 1 (must not split tool pair)", idx)
	}
}

func TestSummaryMaxOutputTokensAndTruncationDetection(t *testing.T) {
	eng := NewEngine(Config{TailMessages: 1})
	msgs := []schema.Message{
		{Seq: 1, Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Design architecture"}}},
		{Seq: 2, Role: schema.RoleAssistant, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Plan ready"}}},
		{Seq: 3, Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Latest"}}},
	}

	// 1. Truncated first pass continues once and succeeds with generous 12288 token cap.
	p1 := &fakeProvider{
		id:         "anthropic",
		wireFormat: "anthropic_messages",
		responses: []provider.Response{
			{
				Blocks:     []schema.ContentBlock{{Type: schema.BlockText, Text: "Part 1 goals."}},
				StopReason: "max_tokens",
			},
			{
				Blocks:     []schema.ContentBlock{{Type: schema.BlockText, Text: "Part 2 next steps."}},
				StopReason: "end_turn",
			},
		},
	}
	cp, did, err := eng.Compact(context.Background(), p1, "t1", "claude-sonnet-4-5", "sys", msgs, 150000)
	if err != nil || !did {
		t.Fatalf("expected continued compaction to succeed: err=%v did=%v", err, did)
	}
	if len(p1.calls) != 2 || p1.calls[0].MaxOutputTokens != DefaultSummaryMaxOutputTokens {
		t.Fatalf("expected 2 calls with MaxOutputTokens=%d, got calls=%d max=%d", DefaultSummaryMaxOutputTokens, len(p1.calls), p1.calls[0].MaxOutputTokens)
	}
	if !strings.Contains(cp.SummaryText, "Part 1 goals.") || !strings.Contains(cp.SummaryText, "Part 2 next steps.") {
		t.Fatalf("expected stitched summary, got %q", cp.SummaryText)
	}

	// 2. Smaller model clamps MaxOutputTokens safely to model cap (Llama 3.3 = 8192).
	pClamp := &fakeProvider{
		id:         "meta",
		wireFormat: "openai_chat_completions",
		responses: []provider.Response{
			{Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Complete"}}, StopReason: "stop"},
		},
	}
	_, did, err = eng.Compact(context.Background(), pClamp, "t1", "Llama-3.3-70B-Instruct", "sys", msgs, 90000)
	if err != nil || !did || pClamp.calls[0].MaxOutputTokens != 8192 {
		t.Fatalf("expected Llama 3.3 clamp to 8192, got err=%v did=%v max=%d", err, did, pClamp.calls[0].MaxOutputTokens)
	}

	// 3. Double truncation fails closed with ErrSummaryTruncated (never silently clips).
	pClip := &fakeProvider{
		id:         "anthropic",
		wireFormat: "anthropic_messages",
		responses: []provider.Response{
			{Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "half 1"}}, StopReason: "max_tokens"},
			{Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "half 2"}}, StopReason: "max_tokens"},
		},
	}
	_, _, err = eng.Compact(context.Background(), pClip, "t1", "claude-sonnet-4-5", "sys", msgs, 150000)
	if !errors.Is(err, ErrSummaryTruncated) {
		t.Fatalf("expected ErrSummaryTruncated on double truncation, got %v", err)
	}
}

func TestOpaqueCompactionCrossProviderFallbackGuarantee(t *testing.T) {
	eng := NewEngine(Config{TailMessages: 1})
	msgs := []schema.Message{
		{Seq: 1, Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Build payment webhook for invoice #99"}}},
		{Seq: 2, Role: schema.RoleAssistant, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Implemented webhook handler in api/pay.go"}}},
		{Seq: 3, Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Now add tests"}}},
	}

	// OpenAI /responses/compact returns ONLY an encrypted opaque item (no SummaryText).
	pOpenAI := &fakeProvider{
		id:         "openai",
		wireFormat: "openai_responses",
		compactRes: &provider.CompactResponse{
			OpaqueItem:  json.RawMessage(`{"type":"compaction","encrypted_content":"enc_blob_1"}`),
			OpaqueItems: []json.RawMessage{json.RawMessage(`{"type":"compaction","encrypted_content":"enc_blob_1"}`)},
		},
	}
	cp, did, err := eng.Compact(context.Background(), pOpenAI, "t1", "gpt-5.4", "sys", msgs, 300000)
	if err != nil || !did {
		t.Fatalf("remote compact failed: err=%v did=%v", err, did)
	}
	if !strings.Contains(cp.SummaryText, "invoice #99") || !strings.Contains(cp.SummaryText, "api/pay.go") {
		t.Fatalf("expected readable fallback summary stored alongside opaque item, got %q", cp.SummaryText)
	}

	// Same wire format -> replays opaque native item verbatim.
	same := ApplyCheckpoint(&cp, msgs, "openai_responses")
	if len(same) != 2 || same[0].Native == nil || len(same[0].Native.RawItems) != 1 {
		t.Fatalf("expected opaque native replay on same wire format: %+v", same)
	}

	// Cross-provider switch -> projects readable fallback summary.
	cross := ApplyCheckpoint(&cp, msgs, "anthropic_messages")
	if len(cross) != 2 || cross[0].Native != nil || !strings.Contains(cross[0].Blocks[0].Text, "invoice #99") {
		t.Fatalf("expected readable fallback summary on cross-provider switch: %+v", cross)
	}

	// Even a legacy opaque-only checkpoint with empty SummaryText synthesizes a readable digest from allMsgs!
	opaqueOnly := cp
	opaqueOnly.SummaryText = ""
	crossLegacy := ApplyCheckpoint(&opaqueOnly, msgs, "gemini_generate_content")
	if len(crossLegacy) != 2 || !strings.Contains(crossLegacy[0].Blocks[0].Text, "invoice #99") {
		t.Fatalf("expected synthesized fallback digest for opaque-only checkpoint, got %+v", crossLegacy)
	}
}

func TestAutoCompactThresholdRecipeAndOverrides(t *testing.T) {
	eng := NewEngine(Config{})

	// 1M Anthropic / Fable models intentionally compact well below 1M (380k).
	for _, m := range []string{"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2", "claude-sonnet-4-5", "anthropic/claude-opus-4-5"} {
		spec := schema.LookupModel("anthropic", m)
		if got := eng.ResolveAutoCompactThreshold(spec); got != 380_000 {
			t.Fatalf("%s default auto-compact threshold = %d, want 380000", m, got)
		}
	}
	// Codex / OpenAI effective windows: 245k on 272k/400k class, 450k on 500k class.
	if got := eng.ResolveAutoCompactThreshold(schema.LookupModel("openai", "gpt-5.4")); got != 245_000 {
		t.Fatalf("gpt-5.4 threshold = %d, want 245000", got)
	}
	if got := eng.ResolveAutoCompactThreshold(schema.LookupModel("openai", "gpt-6.2-astra")); got != 244_800 {
		t.Fatalf("gpt-6.2-astra threshold = %d, want 244800", got)
	}
	for _, m := range []string{"gpt-6.2-sol", "gpt-6.1-sol", "gpt-6.1-terra"} {
		if got := eng.ResolveAutoCompactThreshold(schema.LookupModel("openai", m)); got != 450_000 {
			t.Fatalf("%s threshold = %d, want 450000", m, got)
		}
	}
	// 1M Gemini 4 / 3.5 and 1M/2M Grok models stay capped at 400k; 500k Grok 4.7/4.6 use 85% (425k); 256k Grok 4/build use 85% (217.6k).
	for _, m := range []string{"google/gemini-4-neon", "google/gemini-4-carbon", "gemini-4-hydrogen", "gemini-3.5-flash-lite"} {
		if got := eng.ResolveAutoCompactThreshold(schema.LookupModel("vertex", m)); got != 400_000 {
			t.Fatalf("%s threshold = %d, want 400000", m, got)
		}
	}
	for _, m := range []string{"grok-4.7", "grok-4.6"} {
		if got := eng.ResolveAutoCompactThreshold(schema.LookupModel("grok", m)); got != 425_000 {
			t.Fatalf("%s threshold = %d, want 425000", m, got)
		}
	}
	for _, m := range []string{"grok-4", "grok-build-0.1"} {
		if got := eng.ResolveAutoCompactThreshold(schema.LookupModel("grok", m)); got != 217_600 {
			t.Fatalf("%s threshold = %d, want 217600", m, got)
		}
	}
	for _, m := range []string{"grok-4-fast", "grok-4.3", "grok-4.20-0309-reasoning"} {
		if got := eng.ResolveAutoCompactThreshold(schema.LookupModel("grok", m)); got != 400_000 {
			t.Fatalf("%s threshold = %d, want 400000", m, got)
		}
	}

	// Per-model override > global token override > fraction override, clamped to <=90% of window.
	custom := NewEngine(Config{
		AutoCompactTokenLimit: 190_000,
		AutoCompactFraction:   0.70,
		ModelTokenLimits: map[string]int64{
			"claude-opus-5-5": 300_000,
			"grok-4":          999_999, // must clamp to 90% of 256k = 230400
		},
	})
	if got := custom.ResolveAutoCompactThreshold(schema.LookupModel("anthropic", "claude-opus-5-5")); got != 300_000 {
		t.Fatalf("per-model override failed: got %d, want 300000", got)
	}
	if got := custom.ResolveAutoCompactThreshold(schema.LookupModel("grok", "grok-4")); got != 230_400 {
		t.Fatalf("per-model override must clamp to 90%% of context window (230400), got %d", got)
	}
	if got := custom.ResolveAutoCompactThreshold(schema.LookupModel("openai", "gpt-5.4")); got != 190_000 {
		t.Fatalf("global token override failed: got %d, want 190000", got)
	}

	fracOnly := NewEngine(Config{AutoCompactFraction: 0.60})
	if got := fracOnly.ResolveAutoCompactThreshold(schema.LookupModel("openai", "gpt-5.4")); got != 240_000 {
		t.Fatalf("fraction override failed: got %d, want 240000", got)
	}
}
