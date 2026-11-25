package compact

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
)

func TestSafeSplitNeverBreaksToolPairs(t *testing.T) {
	eng := NewEngine(Config{TailMessages: 2})
	msgs := []schema.Message{
		{Seq: 1, Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "u1"}}},
		{Seq: 2, Role: schema.RoleAssistant, Blocks: []schema.ContentBlock{{Type: schema.BlockToolCall, CallID: "c1", Name: "read_file"}}},
		{Seq: 3, Role: schema.RoleTool, Blocks: []schema.ContentBlock{{Type: schema.BlockToolResult, CallID: "c1", Output: "data"}}},
		{Seq: 4, Role: schema.RoleAssistant, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "done"}}},
	}

	// Naive tail=2 would split at index 2 (between tool_call at 1 and tool_result at 2).
	// SafeSplitIndex must walk back to index 1 so the tool_call/tool_result pair stays intact!
	idx := eng.SafeSplitIndex(msgs)
	if idx != 1 {
		t.Fatalf("SafeSplitIndex = %d, want 1 (must not split tool pair)", idx)
	}
}

func TestPruneOldToolOutputsAndCheckpointCrossProviderFallback(t *testing.T) {
	eng := NewEngine(Config{
		PruneToolOutputAfter:   1,
		MaxPrunedToolOutputLen: 500,
	})
	big := strings.Repeat("x", 2000)
	msgs := []schema.Message{
		{Seq: 1, Role: schema.RoleTool, Blocks: []schema.ContentBlock{{Type: schema.BlockToolResult, CallID: "c1", Output: big}}},
		{Seq: 2, Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "latest"}}},
	}
	pruned := eng.PruneOldToolOutputs(msgs)
	if len(pruned[0].Blocks[0].Output) >= len(big) || !strings.Contains(pruned[0].Blocks[0].Output, "[compacted") {
		t.Fatalf("expected old tool output to be pruned, got len=%d", len(pruned[0].Blocks[0].Output))
	}

	cp := &schema.CompactionCheckpoint{
		ReplacesUpToSeq: 1,
		Provider:        "openai",
		WireFormat:      "openai_responses",
		SummaryText:     "Canonical summary",
		OpaqueItem:      json.RawMessage(`{"type":"compaction","encrypted_content":"enc"}`),
	}
	// Same wire format -> replays opaque native item.
	same := ApplyCheckpoint(cp, msgs, "openai_responses")
	if len(same) != 2 || same[0].Native == nil || len(same[0].Native.RawItems) != 1 {
		t.Fatalf("expected opaque native replay on same wire format: %+v", same)
	}
	// Different wire format -> falls back cleanly to canonical summary text.
	cross := ApplyCheckpoint(cp, msgs, "anthropic_messages")
	if len(cross) != 2 || cross[0].Native != nil || cross[0].Blocks[0].Text != "Canonical summary" {
		t.Fatalf("expected canonical summary fallback on cross-provider switch: %+v", cross)
	}
}
