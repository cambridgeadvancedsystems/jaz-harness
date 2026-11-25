package schema

import (
	"encoding/json"
	"time"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type BlockType string

const (
	BlockText             BlockType = "text"
	BlockImage            BlockType = "image"
	BlockThinking         BlockType = "thinking"
	BlockRedactedThinking BlockType = "redacted_thinking"
	BlockToolCall         BlockType = "tool_call"
	BlockToolResult       BlockType = "tool_result"
	BlockHostedToolCall   BlockType = "hosted_tool_call"
	BlockHostedToolResult BlockType = "hosted_tool_result"
	BlockCompaction       BlockType = "compaction"
)

type Citation struct {
	Type      string `json:"type,omitempty"`
	URL       string `json:"url,omitempty"`
	Title     string `json:"title,omitempty"`
	CitedText string `json:"cited_text,omitempty"`
}

type ImageSource struct {
	URL       string `json:"url,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	Base64    string `json:"base64,omitempty"`
}

type ContentBlock struct {
	Type            BlockType       `json:"type"`
	Text            string          `json:"text,omitempty"`
	Image           *ImageSource    `json:"image,omitempty"`
	Thinking        string          `json:"thinking,omitempty"`
	Signature       string          `json:"signature,omitempty"`
	RedactedData    string          `json:"redacted_data,omitempty"`
	CallID          string          `json:"call_id,omitempty"`
	Name            string          `json:"name,omitempty"`
	Arguments       json.RawMessage `json:"arguments,omitempty"`
	Output          string          `json:"output,omitempty"`
	IsError         bool            `json:"is_error,omitempty"`
	Citations       []Citation      `json:"citations,omitempty"`
	ProviderOpaque  json.RawMessage `json:"provider_opaque,omitempty"`
	ThoughtSignBlob string          `json:"thought_signature,omitempty"`
}

// NativeEnvelope stores provider-native wire items verbatim so same-provider
// replay is 100% lossless (preserving cryptographic thinking signatures,
// redacted thinking, encrypted reasoning, server tool results, and prefix bytes
// for KV cache reuse) while cross-provider turns fall back cleanly to Canonical.
type NativeEnvelope struct {
	Provider   string            `json:"provider"`
	WireFormat string            `json:"wire_format"`
	Model      string            `json:"model,omitempty"`
	ResponseID string            `json:"response_id,omitempty"`
	RawItems   []json.RawMessage `json:"raw_items,omitempty"`
}

type Message struct {
	ID         string          `json:"id"`
	Seq        int64           `json:"seq"`
	TurnEpoch  int64           `json:"turn_epoch,omitempty"`
	Role       Role            `json:"role"`
	Blocks     []ContentBlock  `json:"blocks"`
	Native     *NativeEnvelope `json:"native,omitempty"`
	Usage      *Usage          `json:"usage,omitempty"`
	CreatedAt  time.Time       `json:"created_at"`
}

func (m Message) TextContent() string {
	out := ""
	for _, b := range m.Blocks {
		if b.Type == BlockText && b.Text != "" {
			if out != "" {
				out += "\n"
			}
			out += b.Text
		}
	}
	return out
}

func (m Message) ToolCalls() []ContentBlock {
	var calls []ContentBlock
	for _, b := range m.Blocks {
		if b.Type == BlockToolCall {
			calls = append(calls, b)
		}
	}
	return calls
}

type ToolDef struct {
	Name         string          `json:"name"`
	Description  string          `json:"description,omitempty"`
	InputSchema  json.RawMessage `json:"input_schema"`
	DeferLoading bool            `json:"defer_loading,omitempty"`
	AlwaysLoad   bool            `json:"always_load,omitempty"`
	ServerID     string          `json:"server_id,omitempty"`
	PluginID     string          `json:"plugin_id,omitempty"`
	Category     string          `json:"category,omitempty"`
}

type HostedTool struct {
	Type   string         `json:"type"`
	Name   string         `json:"name,omitempty"`
	Config map[string]any `json:"config,omitempty"`
}

type ToolChoice struct {
	Mode string `json:"mode,omitempty"` // auto | required | none | tool
	Name string `json:"name,omitempty"`
}

type Usage struct {
	InputTokens         int64   `json:"input_tokens"`
	CachedInputTokens   int64   `json:"cached_input_tokens,omitempty"`
	CacheCreationTokens int64   `json:"cache_creation_tokens,omitempty"`
	OutputTokens        int64   `json:"output_tokens"`
	ReasoningTokens     int64   `json:"reasoning_tokens,omitempty"`
	TotalTokens         int64   `json:"total_tokens,omitempty"`
	CostUSD             float64 `json:"cost_usd,omitempty"`
}

func (u *Usage) Add(other Usage) {
	u.InputTokens += other.InputTokens
	u.CachedInputTokens += other.CachedInputTokens
	u.CacheCreationTokens += other.CacheCreationTokens
	u.OutputTokens += other.OutputTokens
	u.ReasoningTokens += other.ReasoningTokens
	u.TotalTokens += other.TotalTokens
	u.CostUSD += other.CostUSD
}

type CompactionCheckpoint struct {
	ID              string          `json:"id"`
	ThreadID        string          `json:"thread_id"`
	CompactionSeq   int64           `json:"compaction_seq"`
	ReplacesUpToSeq int64           `json:"replaces_up_to_seq"`
	Provider        string          `json:"provider"`
	WireFormat      string          `json:"wire_format"`
	SummaryText     string          `json:"summary_text,omitempty"`
	OpaqueItem      json.RawMessage `json:"opaque_item,omitempty"`
	TokensBefore    int64           `json:"tokens_before,omitempty"`
	TokensAfter     int64           `json:"tokens_after,omitempty"`
	CreatedAt       time.Time       `json:"created_at"`
}

type EventType string

const (
	EventTurnStarted        EventType = "turn_started"
	EventAssistantDelta     EventType = "assistant_delta"
	EventThinkingDelta      EventType = "thinking_delta"
	EventToolCallStarted    EventType = "tool_call"
	EventToolCallDelta      EventType = "tool_delta"
	EventToolCallResult     EventType = "tool_result"
	EventHostedToolActivity EventType = "hosted_tool"
	EventCompacted          EventType = "compacted"
	EventUsageUpdated       EventType = "usage_updated"
	EventTurnCompleted      EventType = "turn_completed"
	EventTurnFailed         EventType = "turn_failed"
	EventTurnInterrupted    EventType = "turn_interrupted"
)

type SessionEvent struct {
	Seq       int64          `json:"seq"`
	ThreadID  string         `json:"thread_id"`
	TurnEpoch int64          `json:"turn_epoch,omitempty"`
	Type      EventType      `json:"type"`
	Payload   map[string]any `json:"payload,omitempty"`
	CreatedAt time.Time      `json:"created_at"`
}

type Thread struct {
	ID              string            `json:"id"`
	Title           string            `json:"title,omitempty"`
	Provider        string            `json:"provider"`
	Model           string            `json:"model"`
	ReasoningEffort string            `json:"reasoning_effort,omitempty"`
	Status          string            `json:"status"` // idle | running | failed | interrupted
	TurnEpoch       int64             `json:"turn_epoch"`
	SystemPrompt    string            `json:"system_prompt,omitempty"`
	Metadata        map[string]string `json:"metadata,omitempty"`
	CreatedAt       time.Time         `json:"created_at"`
	UpdatedAt       time.Time         `json:"updated_at"`
}
