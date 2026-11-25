package compact

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
)

const compactionPrompt = `Summarize the conversation history so far into a concise, high-signal continuation checkpoint.
Preserve:
- User goals, explicit constraints, and decisions made
- Key file paths, symbols, commands, and observed outputs/errors
- Unfinished tasks and next steps
Omit verbose tool dumps and chit-chat.`

type Config struct {
	TailMessages           int // recent messages kept verbatim after compaction
	PruneToolOutputAfter   int // prune bulky tool outputs older than this many messages from the tail
	MaxPrunedToolOutputLen int
}

type Engine struct {
	cfg Config
}

func NewEngine(cfg Config) *Engine {
	if cfg.TailMessages <= 0 {
		cfg.TailMessages = 6
	}
	if cfg.PruneToolOutputAfter <= 0 {
		cfg.PruneToolOutputAfter = 4
	}
	if cfg.MaxPrunedToolOutputLen <= 0 {
		cfg.MaxPrunedToolOutputLen = 1200
	}
	return &Engine{cfg: cfg}
}

// EstimateTokens provides a fast deterministic token estimate (~4 chars/token)
// when provider-reported input tokens are not yet available.
func EstimateTokens(systemPrompt string, msgs []schema.Message) int64 {
	chars := len(systemPrompt)
	for _, m := range msgs {
		for _, b := range m.Blocks {
			chars += len(b.Text) + len(b.Thinking) + len(b.Output) + len(b.Arguments) + len(b.Name)
		}
	}
	return int64(chars/4) + 1
}

// ShouldCompact checks whether lastInputTokens (or estimated tokens) crossed
// the model's true auto-compaction threshold.
func (e *Engine) ShouldCompact(spec schema.ModelSpec, lastInputTokens int64, systemPrompt string, msgs []schema.Message) bool {
	tokens := lastInputTokens
	if tokens <= 0 {
		tokens = EstimateTokens(systemPrompt, msgs)
	}
	return tokens >= spec.AutoCompactThreshold()
}

// PruneOldToolOutputs performs Tier-1 cheap micro-compaction in-memory on old
// bulky tool outputs beyond the recent tail without invalidating tool pairings.
func (e *Engine) PruneOldToolOutputs(msgs []schema.Message) []schema.Message {
	if len(msgs) <= e.cfg.PruneToolOutputAfter {
		return msgs
	}
	cutoff := len(msgs) - e.cfg.PruneToolOutputAfter
	out := make([]schema.Message, len(msgs))
	copy(out, msgs)

	for i := 0; i < cutoff; i++ {
		changed := false
		blocks := make([]schema.ContentBlock, len(out[i].Blocks))
		copy(blocks, out[i].Blocks)
		for j, b := range blocks {
			if b.Type == schema.BlockToolResult && len(b.Output) > e.cfg.MaxPrunedToolOutputLen {
				blocks[j].Output = b.Output[:400] + fmt.Sprintf("\n... [compacted %d bytes of earlier tool output] ...", len(b.Output)-400)
				changed = true
			}
		}
		if changed {
			out[i].Blocks = blocks
			out[i].Native = nil // force canonical projection for modified tool output
		}
	}
	return out
}

// SafeSplitIndex chooses a compaction boundary that keeps at least TailMessages
// and NEVER splits between an assistant tool_call and its matching tool_result.
func (e *Engine) SafeSplitIndex(msgs []schema.Message) int {
	if len(msgs) <= e.cfg.TailMessages {
		return 0
	}
	idx := len(msgs) - e.cfg.TailMessages
	for idx > 0 && isToolResultMessage(msgs[idx]) {
		idx--
	}
	if idx > 0 && hasToolCalls(msgs[idx-1]) {
		idx--
	}
	return idx
}

func isToolResultMessage(m schema.Message) bool {
	if m.Role == schema.RoleTool {
		return true
	}
	for _, b := range m.Blocks {
		if b.Type == schema.BlockToolResult {
			return true
		}
	}
	return false
}

func hasToolCalls(m schema.Message) bool {
	for _, b := range m.Blocks {
		if b.Type == schema.BlockToolCall {
			return true
		}
	}
	return false
}

// Compact builds a durable CompactionCheckpoint replacing msgs[:splitIdx].
// It prefers native remote compaction (OpenAI /responses/compact) when supported
// by the provider and falls back to structured summarization for Anthropic,
// Gemini, Grok, Meta, Bedrock, OpenRouter, and cross-provider handoffs.
func (e *Engine) Compact(
	ctx context.Context,
	p provider.Provider,
	threadID, model, systemPrompt string,
	msgs []schema.Message,
	tokensBefore int64,
) (schema.CompactionCheckpoint, bool, error) {
	splitIdx := e.SafeSplitIndex(msgs)
	if splitIdx <= 0 {
		return schema.CompactionCheckpoint{}, false, nil
	}
	prefix := msgs[:splitIdx]
	replacesUpTo := prefix[len(prefix)-1].Seq

	if rc, ok := p.(provider.RemoteCompactor); ok {
		res, err := rc.Compact(ctx, provider.CompactRequest{
			Model:        model,
			SystemPrompt: systemPrompt,
			Messages:     prefix,
			CacheKey:     threadID,
		})
		if err == nil && (len(res.OpaqueItem) > 0 || res.SummaryText != "") {
			cp := schema.CompactionCheckpoint{
				ThreadID:        threadID,
				ReplacesUpToSeq: replacesUpTo,
				Provider:        p.ID(),
				WireFormat:      p.WireFormat(),
				SummaryText:     res.SummaryText,
				OpaqueItem:      res.OpaqueItem,
				TokensBefore:    tokensBefore,
				TokensAfter:     EstimateTokens(systemPrompt, msgs[splitIdx:]),
			}
			return cp, true, nil
		}
		if err != nil && !errors.Is(err, provider.ErrCompactionUnsupported) {
			// Fall back to structured summary if remote compaction endpoint fails.
		}
	}

	summaryMsgs := make([]schema.Message, 0, len(prefix)+1)
	summaryMsgs = append(summaryMsgs, prefix...)
	summaryMsgs = append(summaryMsgs, schema.Message{
		Role:   schema.RoleUser,
		Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: compactionPrompt}},
	})
	resp, err := p.Stream(ctx, provider.Request{
		ThreadID:        threadID,
		Model:           model,
		SystemPrompt:    systemPrompt,
		Messages:        summaryMsgs,
		MaxOutputTokens: 2048,
	}, nil)
	if err != nil {
		return schema.CompactionCheckpoint{}, false, fmt.Errorf("compaction summary failed: %w", err)
	}
	var summaryParts []string
	for _, b := range resp.Blocks {
		if b.Type == schema.BlockText && b.Text != "" {
			summaryParts = append(summaryParts, b.Text)
		}
	}
	summary := strings.TrimSpace(strings.Join(summaryParts, "\n"))
	if summary == "" {
		return schema.CompactionCheckpoint{}, false, errors.New("compaction produced empty summary")
	}

	cp := schema.CompactionCheckpoint{
		ThreadID:        threadID,
		ReplacesUpToSeq: replacesUpTo,
		Provider:        p.ID(),
		WireFormat:      p.WireFormat(),
		SummaryText:     "[Compacted conversation summary]\n" + summary,
		TokensBefore:    tokensBefore,
		TokensAfter:     int64(len(summary)/4) + EstimateTokens(systemPrompt, msgs[splitIdx:]),
	}
	return cp, true, nil
}

// ApplyCheckpoint rebuilds the model-visible message slice from the latest
// compaction checkpoint plus subsequent messages.
func ApplyCheckpoint(cp *schema.CompactionCheckpoint, allMsgs []schema.Message, activeWireFormat string) []schema.Message {
	if cp == nil {
		return allMsgs
	}
	var out []schema.Message
	if len(cp.OpaqueItem) > 0 && cp.WireFormat == activeWireFormat {
		out = append(out, schema.Message{
			Role: schema.RoleDeveloper,
			Blocks: []schema.ContentBlock{{
				Type:           schema.BlockCompaction,
				Text:           cp.SummaryText,
				ProviderOpaque: cp.OpaqueItem,
			}},
			Native: &schema.NativeEnvelope{
				Provider:   cp.Provider,
				WireFormat: cp.WireFormat,
				RawItems:   []json.RawMessage{cp.OpaqueItem},
			},
		})
	} else if cp.SummaryText != "" {
		out = append(out, schema.Message{
			Role: schema.RoleUser,
			Blocks: []schema.ContentBlock{{
				Type: schema.BlockCompaction,
				Text: cp.SummaryText,
			}},
		})
	}
	for _, m := range allMsgs {
		if m.Seq > cp.ReplacesUpToSeq {
			out = append(out, m)
		}
	}
	return out
}
