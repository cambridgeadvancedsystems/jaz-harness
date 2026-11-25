package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/auth"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider/responses"
)

const (
	DefaultBaseURL = "https://api.openai.com/v1"
	WireFormat     = "openai_responses"
)

type Config struct {
	ProviderID   string // "openai" | "codex_oauth" | "azure_openai"
	BaseURL      string
	APIKey       string
	CodexOAuth   *auth.CodexOAuthManager
	ExtraHeaders map[string]string
	HTTPClient   *http.Client
	Originator   string
}

type Client struct {
	cfg        Config
	httpClient *http.Client
}

func New(cfg Config) *Client {
	if strings.TrimSpace(cfg.ProviderID) == "" {
		if cfg.CodexOAuth != nil {
			cfg.ProviderID = "codex_oauth"
		} else {
			cfg.ProviderID = "openai"
		}
	}
	if strings.TrimSpace(cfg.BaseURL) == "" {
		if cfg.CodexOAuth != nil {
			cfg.BaseURL = auth.CodexDefaultBackendURL
		} else {
			cfg.BaseURL = DefaultBaseURL
		}
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
func (c *Client) isCodexOAuth() bool { return c.cfg.CodexOAuth != nil }

func (c *Client) buildRequestBody(req provider.Request, stream bool) ([]byte, error) {
	if strings.TrimSpace(req.Model) == "" {
		return nil, errors.New("openai: model is required")
	}

	payload := map[string]any{
		"model":  req.Model,
		"input":  responses.BuildInputItems(req.Messages, WireFormat),
		"stream": stream,
		"store":  false,
	}
	if strings.TrimSpace(req.SystemPrompt) != "" {
		payload["instructions"] = req.SystemPrompt
	} else if c.isCodexOAuth() {
		payload["instructions"] = "You are a helpful assistant."
	}
	if req.CacheKey != "" && !c.isCodexOAuth() {
		payload["prompt_cache_key"] = req.CacheKey
	}
	if req.MaxOutputTokens > 0 && !c.isCodexOAuth() {
		payload["max_output_tokens"] = req.MaxOutputTokens
	}
	if req.Temperature != nil && req.ReasoningEffort == "" {
		payload["temperature"] = *req.Temperature
	}
	if req.TopP != nil && req.ReasoningEffort == "" {
		payload["top_p"] = *req.TopP
	}

	if effort := normalizeEffort(req.ReasoningEffort); effort != "" {
		payload["reasoning"] = map[string]any{"effort": effort, "summary": "auto"}
		payload["include"] = []string{"reasoning.encrypted_content"}
	}
	if tools := responses.BuildTools(req.Tools, req.HostedTools, false); len(tools) > 0 {
		payload["tools"] = tools
		payload["parallel_tool_calls"] = true
	}
	if tc := responses.BuildToolChoice(req.ToolChoice); tc != nil {
		payload["tool_choice"] = tc
	}
	return json.Marshal(payload)
}

func (c *Client) applyHeaders(ctx context.Context, httpReq *http.Request, accept string, forceRefresh bool) error {
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", accept)
	for k, v := range c.cfg.ExtraHeaders {
		httpReq.Header.Set(k, v)
	}
	if c.isCodexOAuth() {
		var creds auth.CodexCredentials
		var err error
		if forceRefresh {
			creds, err = c.cfg.CodexOAuth.ForceRefresh(ctx)
		} else {
			creds, err = c.cfg.CodexOAuth.Credentials(ctx)
		}
		if err != nil {
			return err
		}
		httpReq.Header.Set("Authorization", "Bearer "+creds.AccessToken)
		if creds.AccountID != "" {
			httpReq.Header.Set("chatgpt-account-id", creds.AccountID)
		}
		originator := c.cfg.Originator
		if originator == "" {
			originator = "codex_cli_rs"
		}
		httpReq.Header.Set("originator", originator)
		return nil
	}
	if c.cfg.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}
	return nil
}

func (c *Client) doAuthenticated(ctx context.Context, endpoint string, body []byte, accept string) (*http.Response, error) {
	refreshed := false
	return provider.DoWithRetry(ctx, 3, func(_ int) (*http.Response, error) {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		if err := c.applyHeaders(ctx, httpReq, accept, refreshed); err != nil {
			return nil, err
		}
		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode == http.StatusUnauthorized && c.isCodexOAuth() && !refreshed {
			resp.Body.Close()
			refreshed = true
			retryReq, rerr := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
			if rerr != nil {
				return nil, rerr
			}
			if rerr := c.applyHeaders(ctx, retryReq, accept, true); rerr != nil {
				return nil, rerr
			}
			return c.httpClient.Do(retryReq)
		}
		return resp, nil
	})
}

func (c *Client) Stream(ctx context.Context, req provider.Request, emit provider.StreamCallback) (provider.Response, error) {
	body, err := c.buildRequestBody(req, true)
	if err != nil {
		return provider.Response{}, err
	}
	resp, err := c.doAuthenticated(ctx, c.cfg.BaseURL+"/responses", body, "text/event-stream")
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
		ProviderID: c.ID(),
		WireFormat: WireFormat,
	}, req.Model, emit)
}

func (c *Client) Compact(ctx context.Context, req provider.CompactRequest) (provider.CompactResponse, error) {
	payload := map[string]any{
		"model": req.Model,
		"input": responses.BuildInputItems(req.Messages, WireFormat),
	}
	if strings.TrimSpace(req.SystemPrompt) != "" {
		payload["instructions"] = req.SystemPrompt
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return provider.CompactResponse{}, err
	}
	resp, err := c.doAuthenticated(ctx, c.cfg.BaseURL+"/responses/compact", body, "application/json")
	if err != nil {
		return provider.CompactResponse{}, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return provider.CompactResponse{}, err
	}
	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusNotImplemented {
		return provider.CompactResponse{}, provider.ErrCompactionUnsupported
	}
	if resp.StatusCode >= 300 {
		return provider.CompactResponse{}, &provider.APIError{
			Provider:   c.ID(),
			StatusCode: resp.StatusCode,
			Body:       strings.TrimSpace(string(raw)),
		}
	}

	var parsed struct {
		Output []json.RawMessage       `json:"output"`
		Usage  *responses.UsagePayload `json:"usage"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return provider.CompactResponse{}, fmt.Errorf("decode responses/compact: %w", err)
	}
	var out provider.CompactResponse
	if parsed.Usage != nil {
		out.Usage = parsed.Usage.ToSchema()
	}
	for _, rawItem := range parsed.Output {
		var meta struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(rawItem, &meta) == nil && meta.Type == "compaction" {
			out.OpaqueItem = append(json.RawMessage(nil), rawItem...)
			break
		}
	}
	if len(out.OpaqueItem) == 0 && len(parsed.Output) > 0 {
		out.OpaqueItem = append(json.RawMessage(nil), parsed.Output[len(parsed.Output)-1]...)
	}
	return out, nil
}

func normalizeEffort(effort string) string {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "low", "medium", "high", "xhigh":
		return strings.ToLower(strings.TrimSpace(effort))
	case "max":
		return "xhigh"
	default:
		return ""
	}
}
