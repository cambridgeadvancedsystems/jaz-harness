package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/auth"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
)

const (
	DefaultBaseURL       = "https://api.anthropic.com"
	DefaultModel         = "claude-opus-5-5"
	DefaultAPIVersion    = "2023-06-01"
	VertexAPIVersion     = "vertex-2023-10-16"
	BedrockAPIVersion    = "bedrock-2023-05-31"
	WireFormat           = "anthropic_messages"
	BetaWebFetch         = "web-fetch-2025-09-10"
	BetaAdvancedToolUse  = "advanced-tool-use-2025-11-20"
	BetaCompaction       = "compact-2026-01-12"
	BetaInterleavedThink = "interleaved-thinking-2025-05-14"
	defaultMaxTokens     = 16384
	maxCacheBreakpoints  = 4
)

type Mode string

const (
	ModeDirect  Mode = "direct"
	ModeVertex  Mode = "vertex"
	ModeBedrock Mode = "bedrock"
)

type TokenSource func(ctx context.Context) (string, error)

type Config struct {
	ProviderID     string // "anthropic" | "vertex_anthropic" | "bedrock_anthropic"
	Mode           Mode
	BaseURL        string
	APIKey         string // Anthropic API key, or relay Bearer token when BearerAuth/RelayMode is true
	BearerAuth     bool   // Send Authorization: Bearer instead of x-api-key (for OAuth or LLM relays)
	RelayMode      bool   // Talk through a native LLM relay using Bearer auth without requiring VertexProject
	VertexProject  string
	VertexLocation string
	VertexToken    TokenSource
	AWS            auth.AWSCredentials
	APIVersion     string
	ExtraHeaders   map[string]string
	HTTPClient     *http.Client
	Now            func() time.Time
}

type Client struct {
	cfg        Config
	httpClient *http.Client
	nowFn      func() time.Time
}

func New(cfg Config) *Client {
	if cfg.Mode == "" {
		cfg.Mode = ModeDirect
	}
	if cfg.ProviderID == "" {
		switch cfg.Mode {
		case ModeVertex:
			cfg.ProviderID = "vertex_anthropic"
		case ModeBedrock:
			cfg.ProviderID = "bedrock_anthropic"
		default:
			cfg.ProviderID = "anthropic"
		}
	}
	if cfg.BaseURL == "" {
		switch cfg.Mode {
		case ModeDirect:
			cfg.BaseURL = DefaultBaseURL
		case ModeVertex:
			loc := cfg.VertexLocation
			if loc == "" {
				loc = "us-east5"
			}
			if loc == "global" {
				cfg.BaseURL = "https://aiplatform.googleapis.com"
			} else {
				cfg.BaseURL = fmt.Sprintf("https://%s-aiplatform.googleapis.com", loc)
			}
		case ModeBedrock:
			reg := cfg.AWS.Region
			if reg == "" {
				reg = "us-east-1"
			}
			cfg.BaseURL = fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", reg)
		}
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	if cfg.APIVersion == "" {
		switch cfg.Mode {
		case ModeVertex:
			cfg.APIVersion = VertexAPIVersion
		case ModeBedrock:
			cfg.APIVersion = BedrockAPIVersion
		default:
			cfg.APIVersion = DefaultAPIVersion
		}
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: 5 * time.Minute}
	}
	nowFn := cfg.Now
	if nowFn == nil {
		nowFn = time.Now
	}
	return &Client{
		cfg:        cfg,
		httpClient: hc,
		nowFn:      nowFn,
	}
}

func (c *Client) ID() string         { return c.cfg.ProviderID }
func (c *Client) WireFormat() string { return WireFormat }

func (c *Client) buildRequestBody(req provider.Request) ([]byte, []string, error) {
	if strings.TrimSpace(req.Model) == "" {
		return nil, nil, errors.New("anthropic: model is required")
	}
	enableCache := req.EnableNativeCache
	messages := buildAnthropicMessages(req.Messages, WireFormat)

	var betas []string
	addBeta := func(b string) {
		for _, existing := range betas {
			if existing == b {
				return
			}
		}
		betas = append(betas, b)
	}

	tools, hasWebFetch, hasDeferred := buildAnthropicTools(req.Tools, req.HostedTools, req.NativeToolSearch)
	if hasWebFetch {
		addBeta(BetaWebFetch)
	}
	if hasDeferred {
		addBeta(BetaAdvancedToolUse)
	}

	usedBreakpoints := 0
	if enableCache && strings.TrimSpace(req.SystemPrompt) != "" {
		usedBreakpoints++
	}
	if enableCache && len(tools) > 0 && usedBreakpoints < maxCacheBreakpoints {
		tools[len(tools)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
		usedBreakpoints++
	}
	if enableCache {
		remaining := maxCacheBreakpoints - usedBreakpoints
		applyConversationCacheBreakpoints(messages, remaining)
	}

	maxTokens := req.MaxOutputTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}

	thinkingBudget := mapThinkingBudget(req.ReasoningEffort)
	forcedTool := req.ToolChoice != nil && (req.ToolChoice.Mode == "required" || req.ToolChoice.Mode == "tool")
	if forcedTool {
		thinkingBudget = 0
	}
	if thinkingBudget > 0 {
		addBeta(BetaInterleavedThink)
		if maxTokens <= thinkingBudget {
			maxTokens = thinkingBudget + 8192
		}
	}
	if req.Compaction != nil && req.Compaction.Enabled {
		addBeta(BetaCompaction)
	}

	out := map[string]any{
		"max_tokens": maxTokens,
		"messages":   messages,
	}
	switch c.cfg.Mode {
	case ModeDirect:
		out["model"] = req.Model
		out["stream"] = true
	case ModeVertex:
		out["anthropic_version"] = c.cfg.APIVersion
		out["stream"] = true
	case ModeBedrock:
		out["anthropic_version"] = c.cfg.APIVersion
	}

	if strings.TrimSpace(req.SystemPrompt) != "" {
		sysBlock := map[string]any{
			"type": "text",
			"text": req.SystemPrompt,
		}
		if enableCache {
			sysBlock["cache_control"] = map[string]any{"type": "ephemeral"}
		}
		out["system"] = []map[string]any{sysBlock}
	}
	if len(tools) > 0 {
		out["tools"] = tools
	}
	if tc := buildAnthropicToolChoice(req.ToolChoice); tc != nil {
		out["tool_choice"] = tc
	}
	if thinkingBudget > 0 {
		out["thinking"] = map[string]any{
			"type":          "enabled",
			"budget_tokens": thinkingBudget,
		}
	} else {
		if req.Temperature != nil {
			out["temperature"] = *req.Temperature
		}
		if req.TopP != nil {
			out["top_p"] = *req.TopP
		}
	}
	if req.Compaction != nil && req.Compaction.Enabled {
		edit := map[string]any{
			"type": "compact_20260112",
		}
		if req.Compaction.TriggerTokens > 0 {
			edit["trigger"] = map[string]any{
				"type":  "input_tokens",
				"value": req.Compaction.TriggerTokens,
			}
		}
		if req.Compaction.PauseAfterCompaction {
			edit["pause_after_compaction"] = true
		}
		if instr := strings.TrimSpace(req.Compaction.Instructions); instr != "" {
			edit["instructions"] = instr
		}
		out["context_management"] = map[string]any{
			"edits": []map[string]any{edit},
		}
	}
	if len(betas) > 0 && c.cfg.Mode == ModeBedrock {
		out["anthropic_beta"] = betas
	}

	encoded, err := json.Marshal(out)
	return encoded, betas, err
}

func (c *Client) endpointForModel(model string) (string, error) {
	switch c.cfg.Mode {
	case ModeDirect:
		if strings.HasSuffix(c.cfg.BaseURL, "/v1") {
			return c.cfg.BaseURL + "/messages", nil
		}
		return c.cfg.BaseURL + "/v1/messages", nil
	case ModeVertex:
		if c.cfg.RelayMode || strings.Contains(c.cfg.BaseURL, "/publishers/anthropic") {
			return fmt.Sprintf("%s/models/%s:streamRawPredict", c.cfg.BaseURL, url.PathEscape(model)), nil
		}
		loc := c.cfg.VertexLocation
		if loc == "" {
			loc = "us-east5"
		}
		if c.cfg.VertexProject == "" {
			return "", errors.New("vertex_anthropic: VertexProject is required")
		}
		return fmt.Sprintf(
			"%s/v1/projects/%s/locations/%s/publishers/anthropic/models/%s:streamRawPredict",
			c.cfg.BaseURL,
			url.PathEscape(c.cfg.VertexProject),
			url.PathEscape(loc),
			url.PathEscape(model),
		), nil
	case ModeBedrock:
		return fmt.Sprintf("%s/model/%s/invoke-with-response-stream", c.cfg.BaseURL, url.PathEscape(model)), nil
	default:
		return "", fmt.Errorf("unsupported anthropic mode %q", c.cfg.Mode)
	}
}

func (c *Client) Stream(ctx context.Context, req provider.Request, emit provider.StreamCallback) (provider.Response, error) {
	body, betas, err := c.buildRequestBody(req)
	if err != nil {
		return provider.Response{}, err
	}
	endpoint, err := c.endpointForModel(req.Model)
	if err != nil {
		return provider.Response{}, err
	}

	resp, err := provider.DoWithRetry(ctx, 3, func(_ int) (*http.Response, error) {
		httpReq, rerr := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if rerr != nil {
			return nil, rerr
		}
		httpReq.Header.Set("Content-Type", "application/json")
		for k, v := range c.cfg.ExtraHeaders {
			httpReq.Header.Set(k, v)
		}
		switch c.cfg.Mode {
		case ModeDirect:
			if c.cfg.BearerAuth || c.cfg.RelayMode {
				if c.cfg.APIKey != "" {
					httpReq.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
				}
			} else if c.cfg.APIKey != "" {
				httpReq.Header.Set("x-api-key", c.cfg.APIKey)
			}
			httpReq.Header.Set("anthropic-version", c.cfg.APIVersion)
			httpReq.Header.Set("Accept", "text/event-stream")
			if len(betas) > 0 {
				httpReq.Header.Set("anthropic-beta", strings.Join(betas, ","))
			}
		case ModeVertex:
			if c.cfg.RelayMode && c.cfg.APIKey != "" {
				httpReq.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
			} else {
				if c.cfg.VertexToken == nil {
					return nil, errors.New("vertex_anthropic: VertexToken source not configured")
				}
				tok, terr := c.cfg.VertexToken(ctx)
				if terr != nil {
					return nil, terr
				}
				httpReq.Header.Set("Authorization", "Bearer "+tok)
			}
			httpReq.Header.Set("Accept", "text/event-stream")
			if len(betas) > 0 {
				httpReq.Header.Set("anthropic-beta", strings.Join(betas, ","))
			}
		case ModeBedrock:
			httpReq.Header.Set("Accept", "application/vnd.amazon.eventstream")
			auth.SignAWSV4(httpReq, body, c.cfg.AWS, "bedrock", c.nowFn())
		}
		return c.httpClient.Do(httpReq)
	})
	if err != nil {
		return provider.Response{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		rawErr, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		return provider.Response{}, &provider.APIError{
			Provider:   c.ID(),
			StatusCode: resp.StatusCode,
			Body:       strings.TrimSpace(string(rawErr)),
			RetryAfter: provider.ParseRetryAfter(resp.Header),
		}
	}

	acc := newAnthropicStreamAcc(c.ID(), req.Model, emit)
	if c.cfg.Mode == ModeBedrock {
		err = readBedrockEventStream(resp.Body, func(payload []byte) error {
			return acc.handleJSONEvent(payload)
		})
	} else {
		err = readAnthropicSSE(resp.Body, acc.handleJSONEvent)
	}
	if err != nil {
		return provider.Response{}, err
	}
	return acc.finish(), nil
}
