package compact

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
)

const (
	DefaultTailMessages           = 6
	DefaultSummaryMaxOutputTokens = 12288
	maxFallbackDigestBytes        = 24 * 1024
	maxFallbackEntryBytes         = 1200

	compactionPrompt = `Summarize the conversation history so far into a complete, high-signal continuation checkpoint.
Preserve:
- User goals, explicit constraints, and decisions made
- Key file paths, symbols, commands, URLs, and observed outputs/errors
- Unfinished tasks and next steps
Omit verbose raw tool dumps and chit-chat.`

	compactionContinuePrompt = `Your previous compaction summary was truncated before finishing. Continue from where it stopped and finish any remaining unfinished sections concisely without repeating earlier sections.`
)

var ErrSummaryTruncated = errors.New("compaction summary truncated at max_tokens")

type Config struct {
	TailMessages           int              // recent messages kept verbatim after compaction
	SummaryMaxOutputTokens int              // generous output cap for compaction summaries (default 12288)
	AutoCompactTokenLimit  int64            // optional global token threshold override (0 = use model default)
	AutoCompactFraction    float64          // optional global context fraction override in (0, 1)
	ModelTokenLimits       map[string]int64 // optional per-model/slug token threshold overrides
}

type Engine struct {
	cfg Config
}

func NewEngine(cfg Config) *Engine {
	if cfg.TailMessages <= 0 {
		cfg.TailMessages = DefaultTailMessages
	}
	if cfg.SummaryMaxOutputTokens <= 0 {
		cfg.SummaryMaxOutputTokens = DefaultSummaryMaxOutputTokens
	}
	return &Engine{cfg: cfg}
}

// ResolveAutoCompactThreshold computes the effective compaction trigger in
// tokens using precedence: ModelTokenLimits -> AutoCompactTokenLimit ->
// AutoCompactFraction -> spec.AutoCompactThreshold(), clamped to at most 90%
// of the model's context window so safety headroom is preserved.
func (e *Engine) ResolveAutoCompactThreshold(spec schema.ModelSpec) int64 {
	ceiling := int64(0)
	if spec.ContextWindow > 0 {
		ceiling = (spec.ContextWindow * 9) / 10
	}
	clamp := func(v int64) int64 {
		if ceiling > 0 && v > ceiling {
			return ceiling
		}
		return v
	}

	if len(e.cfg.ModelTokenLimits) > 0 {
		if v, ok := e.cfg.ModelTokenLimits[spec.ID]; ok && v > 0 {
			return clamp(v)
		}
		normTarget := schema.NormalizeModelSlug(spec.ID)
		bestLen := -1
		var bestVal int64
		for k, v := range e.cfg.ModelTokenLimits {
			if v <= 0 {
				continue
			}
			nk := schema.NormalizeModelSlug(k)
			if normTarget == nk || strings.HasPrefix(normTarget, nk+"-") || strings.HasPrefix(normTarget, nk+"@") {
				if len(nk) > bestLen {
					bestLen = len(nk)
					bestVal = v
				}
			}
		}
		if bestLen >= 0 {
			return clamp(bestVal)
		}
	}
	if e.cfg.AutoCompactTokenLimit > 0 {
		return clamp(e.cfg.AutoCompactTokenLimit)
	}
	if e.cfg.AutoCompactFraction > 0 && e.cfg.AutoCompactFraction < 1 && spec.ContextWindow > 0 {
		return clamp(int64(float64(spec.ContextWindow) * e.cfg.AutoCompactFraction))
	}
	return spec.AutoCompactThreshold()
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
// the resolved auto-compaction threshold for spec.
func (e *Engine) ShouldCompact(spec schema.ModelSpec, lastInputTokens int64, systemPrompt string, msgs []schema.Message) bool {
	tokens := lastInputTokens
	if tokens <= 0 {
		tokens = EstimateTokens(systemPrompt, msgs)
	}
	return tokens >= e.ResolveAutoCompactThreshold(spec)
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

func (e *Engine) resolveSummaryMaxTokens(spec schema.ModelSpec) int {
	capTokens := e.cfg.SummaryMaxOutputTokens
	if capTokens <= 0 {
		capTokens = DefaultSummaryMaxOutputTokens
	}
	if spec.MaxOutputTokens > 0 && capTokens > spec.MaxOutputTokens {
		return spec.MaxOutputTokens
	}
	return capTokens
}

func isTruncatedStopReason(reason string) bool {
	switch strings.ToLower(strings.TrimSpace(reason)) {
	case "max_tokens", "length", "incomplete", "max_output_tokens":
		return true
	default:
		return false
	}
}

// Compact builds a durable CompactionCheckpoint replacing msgs[:splitIdx].
// It never mutates msgs in-place, preserving byte-identical KV prefixes until a
// checkpoint is durably committed. Native remote compaction (OpenAI
// /responses/compact) is preferred when supported and always accompanied by a
// readable fallback digest so mid-thread provider switches never lose context.
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
	modelSpec := schema.LookupModel(p.ID(), model)

	if rc, ok := p.(provider.RemoteCompactor); ok {
		res, err := rc.Compact(ctx, provider.CompactRequest{
			Model:        model,
			SystemPrompt: systemPrompt,
			Messages:     prefix,
			CacheKey:     threadID,
		})
		if err == nil && (len(res.OpaqueItem) > 0 || len(res.OpaqueItems) > 0 || res.SummaryText != "") {
			summaryText := strings.TrimSpace(res.SummaryText)
			if summaryText == "" {
				// Guarantee a readable fallback summary so switching away from an
				// opaque compaction provider (e.g. OpenAI -> Anthropic/Gemini)
				// never silently drops compacted conversation history.
				summaryText = BuildFallbackSummary(prefix)
			}
			opaqueItem := res.OpaqueItem
			if len(opaqueItem) == 0 && len(res.OpaqueItems) > 0 {
				opaqueItem = res.OpaqueItems[0]
			}
			cp := schema.CompactionCheckpoint{
				ThreadID:        threadID,
				ReplacesUpToSeq: replacesUpTo,
				Provider:        p.ID(),
				WireFormat:      p.WireFormat(),
				SummaryText:     summaryText,
				OpaqueItem:      opaqueItem,
				OpaqueItems:     res.OpaqueItems,
				TokensBefore:    tokensBefore,
				TokensAfter:     EstimateTokens(systemPrompt, msgs[splitIdx:]),
			}
			return cp, true, nil
		}
		if err != nil && !errors.Is(err, provider.ErrCompactionUnsupported) {
			// Fall back to structured summary if remote compaction endpoint fails.
		}
	}

	maxSummaryTokens := e.resolveSummaryMaxTokens(modelSpec)
	summary, err := e.generateSummary(ctx, p, threadID, model, systemPrompt, prefix, maxSummaryTokens)
	if err != nil {
		return schema.CompactionCheckpoint{}, false, err
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

func (e *Engine) generateSummary(
	ctx context.Context,
	p provider.Provider,
	threadID, model, systemPrompt string,
	prefix []schema.Message,
	maxTokens int,
) (string, error) {
	summaryMsgs := make([]schema.Message, 0, len(prefix)+1)
	summaryMsgs = append(summaryMsgs, prefix...)
	summaryMsgs = append(summaryMsgs, schema.Message{
		Role:   schema.RoleUser,
		Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: compactionPrompt}},
	})

	resp, err := p.Stream(ctx, provider.Request{
		ThreadID:          threadID,
		Model:             model,
		SystemPrompt:      systemPrompt,
		Messages:          summaryMsgs,
		MaxOutputTokens:   maxTokens,
		CacheKey:          threadID,
		EnableNativeCache: true,
	}, nil)
	if err != nil {
		return "", fmt.Errorf("compaction summary failed: %w", err)
	}
	firstPart := extractSummaryText(resp.Blocks)
	if firstPart == "" {
		return "", errors.New("compaction produced empty summary")
	}
	if !isTruncatedStopReason(resp.StopReason) {
		return firstPart, nil
	}

	// Single bounded continuation pass when a large summary hits max_tokens.
	contMsgs := append(summaryMsgs,
		schema.Message{
			Role:   schema.RoleAssistant,
			Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: firstPart}},
		},
		schema.Message{
			Role:   schema.RoleUser,
			Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: compactionContinuePrompt}},
		},
	)
	contResp, err := p.Stream(ctx, provider.Request{
		ThreadID:          threadID,
		Model:             model,
		SystemPrompt:      systemPrompt,
		Messages:          contMsgs,
		MaxOutputTokens:   maxTokens,
		CacheKey:          threadID,
		EnableNativeCache: true,
	}, nil)
	if err != nil {
		return "", fmt.Errorf("%w: continuation failed: %v", ErrSummaryTruncated, err)
	}
	if isTruncatedStopReason(contResp.StopReason) {
		return "", fmt.Errorf("%w (stop_reason=%s, max_output_tokens=%d)", ErrSummaryTruncated, contResp.StopReason, maxTokens)
	}
	secondPart := extractSummaryText(contResp.Blocks)
	if secondPart == "" {
		return "", fmt.Errorf("%w: continuation produced empty output", ErrSummaryTruncated)
	}
	return firstPart + "\n" + secondPart, nil
}

func extractSummaryText(blocks []schema.ContentBlock) string {
	var parts []string
	for _, b := range blocks {
		if (b.Type == schema.BlockText || b.Type == schema.BlockCompaction) && b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

// BuildFallbackSummary builds a deterministic, readable, UTF-8-safe digest of
// replaced messages so cross-provider handoffs away from opaque compaction
// checkpoints (e.g. OpenAI encrypted compaction -> Anthropic/Gemini/Grok)
// always preserve user goals, decisions, tool names, and key outcomes.
func BuildFallbackSummary(msgs []schema.Message) string {
	var b strings.Builder
	b.WriteString("[Compacted conversation summary — cross-provider fallback digest]\n")
	for _, m := range msgs {
		for _, blk := range m.Blocks {
			var line string
			switch blk.Type {
			case schema.BlockCompaction:
				if blk.Text != "" {
					line = "Prior summary: " + clipUTF8(strings.TrimSpace(blk.Text), maxFallbackEntryBytes)
				}
			case schema.BlockText:
				if t := strings.TrimSpace(blk.Text); t != "" {
					line = fmt.Sprintf("%s: %s", m.Role, clipUTF8(t, maxFallbackEntryBytes))
				}
			case schema.BlockToolCall:
				line = fmt.Sprintf("tool_call(%s): %s", blk.Name, clipUTF8(strings.TrimSpace(string(blk.Arguments)), 400))
			case schema.BlockToolResult:
				status := "ok"
				if blk.IsError {
					status = "error"
				}
				line = fmt.Sprintf("tool_result(%s, %s): %s", blk.Name, status, clipUTF8(strings.TrimSpace(blk.Output), 400))
			}
			if line == "" {
				continue
			}
			if b.Len()+len(line)+1 > maxFallbackDigestBytes {
				b.WriteString("... [earlier digest clipped to budget]")
				return b.String()
			}
			b.WriteString("- ")
			b.WriteString(line)
			b.WriteByte('\n')
		}
	}
	return strings.TrimSpace(b.String())
}

func clipUTF8(s string, maxBytes int) string {
	if len(s) <= maxBytes {
		return s
	}
	n := maxBytes
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n] + "..."
}

// ApplyCheckpoint rebuilds the model-visible message slice from the latest
// committed compaction checkpoint plus subsequent messages. On cross-provider
// switches away from an opaque checkpoint, it guarantees a readable summary
// fallback (using cp.SummaryText or synthesizing a digest of replaced messages).
func ApplyCheckpoint(cp *schema.CompactionCheckpoint, allMsgs []schema.Message, activeWireFormat string) []schema.Message {
	if cp == nil {
		return allMsgs
	}
	var out []schema.Message
	nativeItems := cp.NativeRawItems()
	if len(nativeItems) > 0 && cp.WireFormat == activeWireFormat {
		role := schema.RoleDeveloper
		if activeWireFormat == "anthropic_messages" {
			role = schema.RoleAssistant
		}
		out = append(out, schema.Message{
			Role: role,
			Blocks: []schema.ContentBlock{{
				Type:           schema.BlockCompaction,
				Text:           cp.SummaryText,
				ProviderOpaque: nativeItems[0],
			}},
			Native: &schema.NativeEnvelope{
				Provider:   cp.Provider,
				WireFormat: cp.WireFormat,
				RawItems:   nativeItems,
			},
		})
	} else {
		summaryText := strings.TrimSpace(cp.SummaryText)
		if summaryText == "" {
			var replaced []schema.Message
			for _, m := range allMsgs {
				if m.Seq <= cp.ReplacesUpToSeq {
					replaced = append(replaced, m)
				}
			}
			summaryText = BuildFallbackSummary(replaced)
		}
		if summaryText != "" {
			out = append(out, schema.Message{
				Role: schema.RoleUser,
				Blocks: []schema.ContentBlock{{
					Type: schema.BlockCompaction,
					Text: summaryText,
				}},
			})
		}
	}
	for _, m := range allMsgs {
		if m.Seq > cp.ReplacesUpToSeq {
			out = append(out, m)
		}
	}
	return out
}
