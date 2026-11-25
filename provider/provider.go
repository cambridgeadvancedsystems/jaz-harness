package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
)

type StreamDeltaKind string

const (
	DeltaText               StreamDeltaKind = "text"
	DeltaThinking           StreamDeltaKind = "thinking"
	DeltaToolCallStart      StreamDeltaKind = "tool_call_start"
	DeltaToolCallArgs       StreamDeltaKind = "tool_call_args"
	DeltaHostedToolStart    StreamDeltaKind = "hosted_tool_start"
	DeltaHostedToolComplete StreamDeltaKind = "hosted_tool_complete"
	DeltaCompaction         StreamDeltaKind = "compaction"

	DefaultMaxLineBytes      = 64 << 20 // 64 MiB per SSE/JSONL line ceiling
	DefaultMaxBodyBytes      = 64 << 20 // 64 MiB response payload ceiling
	DefaultMaxErrorBodyBytes = 64 << 10 // 64 KiB error body diagnostic ceiling
)

// CompactionConfig enables provider-native server-side context compaction
// (Anthropic compact-2026-01-12 or OpenAI Responses context_management).
type CompactionConfig struct {
	Enabled              bool
	TriggerTokens        int64  // optional input token trigger threshold
	PauseAfterCompaction bool   // Anthropic pause_after_compaction
	Instructions         string // optional custom compaction instructions
}

type StreamChunk struct {
	Kind      StreamDeltaKind
	Text      string
	Detail    string
	CallID    string
	ToolName  string
	Arguments string
	Citations []schema.Citation
}

type StreamCallback func(StreamChunk)

type Request struct {
	ThreadID          string
	Model             string
	SystemPrompt      string
	Messages          []schema.Message
	Tools             []schema.ToolDef
	HostedTools       []schema.HostedTool
	ToolChoice        *schema.ToolChoice
	ReasoningEffort   string
	MaxOutputTokens   int
	Temperature       *float64
	TopP              *float64
	CacheKey          string
	CachedContentName string // Gemini 2.5/3 explicit cachedContents resource name (omitted on Gemini 4)
	EnableNativeCache bool
	NativeToolSearch  bool              // Anthropic native defer_loading + tool_search_tool_bm25
	Compaction        *CompactionConfig // Provider-native server-side compaction (Anthropic / OpenAI)
	Metadata          map[string]string
}

type Response struct {
	ID                string
	Model             string
	Blocks            []schema.ContentBlock
	Native            *schema.NativeEnvelope
	Usage             schema.Usage
	StopReason        string
	ContinuationToken string // Gemini 4 long-decode continuation token when status requires continuation
}

type CompactRequest struct {
	Model        string
	SystemPrompt string
	Messages     []schema.Message
	CacheKey     string
}

type CompactResponse struct {
	SummaryText string
	OpaqueItem  json.RawMessage
	OpaqueItems []json.RawMessage
	Usage       schema.Usage
}

var ErrCompactionUnsupported = errors.New("provider does not support native remote compaction")

// Provider is the native wire-protocol client contract. Each implementation
// speaks its vendor's real API without lossy cross-protocol translation.
type Provider interface {
	ID() string
	WireFormat() string
	Stream(ctx context.Context, req Request, emit StreamCallback) (Response, error)
}

// RemoteCompactor is optionally implemented by providers offering native
// server-side compaction (e.g. OpenAI /v1/responses/compact).
type RemoteCompactor interface {
	Compact(ctx context.Context, req CompactRequest) (CompactResponse, error)
}

// ImageGenerationRequest describes a native or fallback image generation or
// reference-image editing request.
type ImageGenerationRequest struct {
	Prompt                string
	Model                 string
	ReferenceImages       []schema.ImageSource
	Size                  string
	Quality               string
	Background            string
	OutputFormat          string
	InputFidelity         string
	TransparentBackground bool
}

// ImageGenerationResponse carries generated images and optional provider metadata.
type ImageGenerationResponse struct {
	ID            string
	Model         string
	RevisedPrompt string
	Images        []schema.ImageSource
	Metadata      map[string]any
}

// ImageGenerator is optionally implemented by providers that expose a native
// image generation and reference-image editing API, and can also be supplied
// as a configurable fallback backend for providers without native imagegen.
type ImageGenerator interface {
	GenerateImage(ctx context.Context, req ImageGenerationRequest) (ImageGenerationResponse, error)
}

// ImageGeneratorFunc adapts a plain function to the ImageGenerator interface.
type ImageGeneratorFunc func(ctx context.Context, req ImageGenerationRequest) (ImageGenerationResponse, error)

func (fn ImageGeneratorFunc) GenerateImage(ctx context.Context, req ImageGenerationRequest) (ImageGenerationResponse, error) {
	return fn(ctx, req)
}

type APIError struct {
	Provider   string
	StatusCode int
	Body       string
	RetryAfter time.Duration
}

func (e *APIError) Error() string {
	return fmt.Sprintf("%s api error (status %d): %s", e.Provider, e.StatusCode, e.Body)
}

func (e *APIError) Retryable() bool {
	return e.StatusCode == http.StatusTooManyRequests ||
		e.StatusCode == http.StatusRequestTimeout ||
		e.StatusCode == 529 ||
		e.StatusCode >= 500
}

func ParseRetryAfter(h http.Header) time.Duration {
	raw := strings.TrimSpace(h.Get("Retry-After"))
	if raw == "" {
		return 0
	}
	if secs, err := strconv.Atoi(raw); err == nil && secs > 0 {
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(raw); err == nil {
		d := time.Until(t)
		if d > 0 {
			return d
		}
	}
	return 0
}

// SanitizeToolArgsJSON guarantees valid JSON object text for tool arguments so
// malformed model emissions never cause HTTP 400 errors on replay turns.
func SanitizeToolArgsJSON(raw []byte) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "{}"
	}
	if json.Valid(trimmed) {
		return string(trimmed)
	}
	return "{}"
}

// DecodeToolArgsObject parses raw tool arguments into a JSON object map,
// falling back to an empty object on malformed JSON.
func DecodeToolArgsObject(raw []byte) map[string]any {
	clean := SanitizeToolArgsJSON(raw)
	var obj map[string]any
	if json.Unmarshal([]byte(clean), &obj) == nil && obj != nil {
		return obj
	}
	return map[string]any{}
}

// DoWithRetry executes an HTTP request builder with bounded exponential backoff
// on transient 429/529/5xx errors while respecting Retry-After, x-should-retry,
// and context cancellation.
func DoWithRetry(ctx context.Context, maxAttempts int, fn func(attempt int) (*http.Response, error)) (*http.Response, error) {
	if maxAttempts <= 0 {
		maxAttempts = 3
	}
	var lastErr error
	backoff := 400 * time.Millisecond
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		resp, err := fn(attempt)
		if err == nil {
			if strings.EqualFold(strings.TrimSpace(resp.Header.Get("x-should-retry")), "false") {
				return resp, nil
			}
			if resp.StatusCode != http.StatusTooManyRequests && resp.StatusCode != 529 && resp.StatusCode < 500 {
				return resp, nil
			}
			wait := ParseRetryAfter(resp.Header)
			if attempt == maxAttempts {
				return resp, nil
			}
			resp.Body.Close()
			if wait <= 0 {
				wait = backoff
				backoff *= 2
			}
			if wait > 15*time.Second {
				wait = 15 * time.Second
			}
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(wait):
			}
			continue
		}
		lastErr = err
		if attempt == maxAttempts || ctx.Err() != nil {
			return nil, lastErr
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(backoff):
			backoff *= 2
		}
	}
	return nil, lastErr
}

type Registry struct {
	mu        sync.RWMutex
	providers map[string]Provider
}

func NewRegistry() *Registry {
	return &Registry{
		providers: make(map[string]Provider),
	}
}

func (r *Registry) Register(name string, p Provider) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.providers[strings.ToLower(strings.TrimSpace(name))] = p
}

func (r *Registry) Get(name string) (Provider, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	p, ok := r.providers[strings.ToLower(strings.TrimSpace(name))]
	return p, ok
}

func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.providers))
	for k := range r.providers {
		out = append(out, k)
	}
	return out
}
