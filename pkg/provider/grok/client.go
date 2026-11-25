package grok

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider/responses"
)

const (
	DefaultBaseURL = "https://api.x.ai/v1"
	WireFormat     = "xai_responses"
)

type Config struct {
	ProviderID   string
	BaseURL      string
	APIKey       string
	ExtraHeaders map[string]string
	HTTPClient   *http.Client
}

// Client speaks xAI's native /v1/responses API directly with native x_search
// and web_search hosted tools, x-grok-conv-id KV cache affinity, and verbatim
// xs_*/ws_* custom_tool_call preservation (zero Codex fork required).
type Client struct {
	cfg        Config
	httpClient *http.Client
}

func New(cfg Config) *Client {
	if strings.TrimSpace(cfg.ProviderID) == "" {
		cfg.ProviderID = "grok"
	}
	if strings.TrimSpace(cfg.BaseURL) == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: 5 * time.Minute}
	}
	return &Client{
		cfg:        cfg,
		httpClient: hc,
	}
}

func (c *Client) ID() string         { return c.cfg.ProviderID }
func (c *Client) WireFormat() string { return WireFormat }

func (c *Client) buildRequestBody(req provider.Request) ([]byte, error) {
	if strings.TrimSpace(req.Model) == "" {
		return nil, errors.New("grok: model is required")
	}
	payload := map[string]any{
		"model":  req.Model,
		"input":  responses.BuildInputItems(req.Messages, WireFormat),
		"stream": true,
		"store":  false,
	}
	if strings.TrimSpace(req.SystemPrompt) != "" {
		payload["instructions"] = req.SystemPrompt
	}
	if req.MaxOutputTokens > 0 {
		payload["max_output_tokens"] = req.MaxOutputTokens
	}
	if req.Temperature != nil {
		payload["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		payload["top_p"] = *req.TopP
	}
	if eff := normalizeEffort(req.ReasoningEffort); eff != "" {
		payload["reasoning"] = map[string]any{"effort": eff}
	}
	if tools := responses.BuildTools(req.Tools, req.HostedTools, true); len(tools) > 0 {
		payload["tools"] = tools
	}
	if tc := responses.BuildToolChoice(req.ToolChoice); tc != nil {
		payload["tool_choice"] = tc
	}
	return json.Marshal(payload)
}

func (c *Client) Stream(ctx context.Context, req provider.Request, emit provider.StreamCallback) (provider.Response, error) {
	body, err := c.buildRequestBody(req)
	if err != nil {
		return provider.Response{}, err
	}
	endpoint := c.cfg.BaseURL + "/responses"
	convID := req.CacheKey
	if convID == "" {
		convID = req.ThreadID
	}

	resp, err := provider.DoWithRetry(ctx, 3, func(_ int) (*http.Response, error) {
		httpReq, rerr := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if rerr != nil {
			return nil, rerr
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")
		for k, v := range c.cfg.ExtraHeaders {
			httpReq.Header.Set(k, v)
		}
		if c.cfg.APIKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
		}
		if convID != "" {
			httpReq.Header.Set("x-grok-conv-id", convID)
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

	return responses.ParseSSE(resp.Body, responses.Dialect{
		ProviderID:             c.ID(),
		WireFormat:             WireFormat,
		StripExternalWebAccess: true,
		IsHostedCall:           isGrokHostedCall,
	}, req.Model, emit)
}

func isGrokHostedCall(callID, name string) bool {
	if strings.HasPrefix(callID, "xs_") || strings.HasPrefix(callID, "ws_") {
		return true
	}
	switch name {
	case "x_search", "x_keyword_search", "x_semantic_search", "x_user_search", "x_thread_fetch",
		"web_search", "web_search_with_snippets", "browse_page",
		"code_interpreter", "code_execution":
		return true
	}
	return false
}

func normalizeEffort(effort string) string {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "low", "medium", "high":
		return strings.ToLower(strings.TrimSpace(effort))
	case "xhigh", "max":
		return "high"
	default:
		return ""
	}
}
