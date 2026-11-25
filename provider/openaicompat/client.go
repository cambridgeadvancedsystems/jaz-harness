package openaicompat

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/auth"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/google/uuid"
)

const (
	MetaLlamaBaseURL  = "https://api.llama.com/compat/v1"
	OpenRouterBaseURL = "https://openrouter.ai/api/v1"
	WireFormat        = "openai_chat_completions"
)

type Flavor string

const (
	FlavorGeneric    Flavor = "generic"
	FlavorMeta       Flavor = "meta"
	FlavorOpenRouter Flavor = "openrouter"
	FlavorBedrock    Flavor = "bedrock"
)

type Config struct {
	ProviderID   string // "meta" | "openrouter" | "bedrock_openai" | custom
	Flavor       Flavor
	BaseURL      string
	APIKey       string
	AWS          *auth.AWSCredentials // optional AWS SigV4 for Bedrock endpoints
	HTTPReferer  string               // OpenRouter optional HTTP-Referer
	AppTitle     string               // OpenRouter optional X-Title
	ExtraHeaders map[string]string
	HTTPClient   *http.Client
	Now          func() time.Time
}

// Client speaks /chat/completions natively for Meta Llama API, OpenRouter,
// Bedrock OpenAI-compatible endpoints, and custom inference gateways.
type Client struct {
	cfg        Config
	httpClient *http.Client
	nowFn      func() time.Time
}

func New(cfg Config) *Client {
	if cfg.Flavor == "" {
		cfg.Flavor = FlavorGeneric
	}
	if cfg.ProviderID == "" {
		switch cfg.Flavor {
		case FlavorMeta:
			cfg.ProviderID = "meta"
		case FlavorOpenRouter:
			cfg.ProviderID = "openrouter"
		case FlavorBedrock:
			cfg.ProviderID = "bedrock_openai"
		default:
			cfg.ProviderID = "openai_compat"
		}
	}
	if cfg.BaseURL == "" {
		switch cfg.Flavor {
		case FlavorMeta:
			cfg.BaseURL = MetaLlamaBaseURL
		case FlavorOpenRouter:
			cfg.BaseURL = OpenRouterBaseURL
		}
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
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

func (c *Client) buildRequestBody(req provider.Request) ([]byte, error) {
	if strings.TrimSpace(req.Model) == "" {
		return nil, errors.New("openaicompat: model is required")
	}
	messages := buildChatMessages(req.SystemPrompt, req.Messages)
	payload := map[string]any{
		"model":          req.Model,
		"messages":       messages,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	}
	if req.MaxOutputTokens > 0 {
		payload["max_tokens"] = req.MaxOutputTokens
	}
	if req.Temperature != nil {
		payload["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		payload["top_p"] = *req.TopP
	}
	if c.cfg.Flavor == FlavorOpenRouter && req.ReasoningEffort != "" {
		payload["reasoning"] = map[string]any{
			"effort": strings.ToLower(strings.TrimSpace(req.ReasoningEffort)),
		}
	}
	if tools := buildChatTools(req.Tools); len(tools) > 0 {
		payload["tools"] = tools
		if tc := c.buildChatToolChoice(req.ToolChoice); tc != nil {
			payload["tool_choice"] = tc
		}
	}
	return json.Marshal(payload)
}

func (c *Client) buildChatToolChoice(tc *schema.ToolChoice) any {
	if tc == nil || tc.Mode == "" {
		return nil
	}
	// Meta Llama hosted endpoints commonly only accept "auto" for tool_choice.
	if c.cfg.Flavor == FlavorMeta {
		return "auto"
	}
	switch tc.Mode {
	case "auto", "required", "none":
		return tc.Mode
	case "tool":
		if tc.Name != "" {
			return map[string]any{
				"type":     "function",
				"function": map[string]any{"name": tc.Name},
			}
		}
	}
	return nil
}

func (c *Client) Stream(ctx context.Context, req provider.Request, emit provider.StreamCallback) (provider.Response, error) {
	if c.cfg.BaseURL == "" {
		return provider.Response{}, errors.New("openaicompat: BaseURL is required")
	}
	body, err := c.buildRequestBody(req)
	if err != nil {
		return provider.Response{}, err
	}
	endpoint := c.cfg.BaseURL + "/chat/completions"

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
		if c.cfg.Flavor == FlavorOpenRouter {
			if c.cfg.HTTPReferer != "" {
				httpReq.Header.Set("HTTP-Referer", c.cfg.HTTPReferer)
			}
			if c.cfg.AppTitle != "" {
				httpReq.Header.Set("X-Title", c.cfg.AppTitle)
			}
		}
		if c.cfg.AWS != nil && c.cfg.AWS.AccessKeyID != "" {
			auth.SignAWSV4(httpReq, body, *c.cfg.AWS, "bedrock", c.nowFn())
		} else if c.cfg.APIKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
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

	return parseChatSSE(resp.Body, c.ID(), req.Model, emit)
}

type tcAcc struct {
	index int
	id    string
	name  string
	args  strings.Builder
}

func parseChatSSE(r io.Reader, providerID, fallbackModel string, emit provider.StreamCallback) (provider.Response, error) {
	var (
		respID     string
		model      = fallbackModel
		stopReason = "stop"
		usage      schema.Usage
		textBuf    strings.Builder
		reasonBuf  strings.Builder
		calls      = map[int]*tcAcc{}
		done       bool
	)

	err := provider.ScanLines(r, provider.DefaultMaxLineBytes, func(rawLine string) error {
		if done {
			return nil
		}
		line := strings.TrimSpace(rawLine)
		if !strings.HasPrefix(line, "data:") {
			return nil
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			return nil
		}
		if data == "[DONE]" {
			done = true
			return nil
		}
		var chunk struct {
			ID      string `json:"id"`
			Model   string `json:"model"`
			Choices []struct {
				FinishReason string `json:"finish_reason"`
				Delta        struct {
					Content          string `json:"content"`
					Reasoning        string `json:"reasoning"`
					ReasoningContent string `json:"reasoning_content"`
					ToolCalls        []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens        int64   `json:"prompt_tokens"`
				CompletionTokens    int64   `json:"completion_tokens"`
				TotalTokens         int64   `json:"total_tokens"`
				Cost                float64 `json:"cost"`
				PromptTokensDetails *struct {
					CachedTokens int64 `json:"cached_tokens"`
				} `json:"prompt_tokens_details"`
				CompletionTokensDetails *struct {
					ReasoningTokens int64 `json:"reasoning_tokens"`
				} `json:"completion_tokens_details"`
			} `json:"usage"`
		}
		if json.Unmarshal([]byte(data), &chunk) != nil {
			return nil
		}
		if chunk.ID != "" {
			respID = chunk.ID
		}
		if chunk.Model != "" {
			model = chunk.Model
		}
		if chunk.Usage != nil {
			u := chunk.Usage
			usage.InputTokens = u.PromptTokens
			usage.OutputTokens = u.CompletionTokens
			usage.TotalTokens = u.TotalTokens
			if usage.TotalTokens == 0 {
				usage.TotalTokens = u.PromptTokens + u.CompletionTokens
			}
			if u.Cost > 0 {
				usage.CostUSD = u.Cost
			}
			if u.PromptTokensDetails != nil {
				usage.CachedInputTokens = u.PromptTokensDetails.CachedTokens
			}
			if u.CompletionTokensDetails != nil {
				usage.ReasoningTokens = u.CompletionTokensDetails.ReasoningTokens
			}
		}

		for _, ch := range chunk.Choices {
			if ch.FinishReason != "" {
				stopReason = ch.FinishReason
			}
			thought := ch.Delta.ReasoningContent
			if thought == "" {
				thought = ch.Delta.Reasoning
			}
			if thought != "" {
				reasonBuf.WriteString(thought)
				if emit != nil {
					emit(provider.StreamChunk{Kind: provider.DeltaThinking, Text: thought})
				}
			}
			if ch.Delta.Content != "" {
				textBuf.WriteString(ch.Delta.Content)
				if emit != nil {
					emit(provider.StreamChunk{Kind: provider.DeltaText, Text: ch.Delta.Content})
				}
			}
			for _, tc := range ch.Delta.ToolCalls {
				acc, ok := calls[tc.Index]
				if !ok {
					id := tc.ID
					if id == "" {
						id = "call_" + uuid.NewString()[:8]
					}
					acc = &tcAcc{
						index: tc.Index,
						id:    id,
						name:  tc.Function.Name,
					}
					calls[tc.Index] = acc
					if emit != nil {
						emit(provider.StreamChunk{Kind: provider.DeltaToolCallStart, CallID: acc.id, ToolName: acc.name})
					}
				} else {
					if acc.name == "" && tc.Function.Name != "" {
						acc.name = tc.Function.Name
					}
				}
				if tc.Function.Arguments != "" {
					acc.args.WriteString(tc.Function.Arguments)
					if emit != nil {
						emit(provider.StreamChunk{Kind: provider.DeltaToolCallArgs, CallID: acc.id, Arguments: tc.Function.Arguments})
					}
				}
			}
		}
		return nil
	})
	if err != nil {
		return provider.Response{}, err
	}

	var blocks []schema.ContentBlock
	if reasonBuf.Len() > 0 {
		blocks = append(blocks, schema.ContentBlock{
			Type:     schema.BlockThinking,
			Thinking: reasonBuf.String(),
		})
	}
	if textBuf.Len() > 0 {
		blocks = append(blocks, schema.ContentBlock{
			Type: schema.BlockText,
			Text: textBuf.String(),
		})
	}
	indices := make([]int, 0, len(calls))
	for idx := range calls {
		indices = append(indices, idx)
	}
	sort.Ints(indices)
	for _, idx := range indices {
		acc := calls[idx]
		args := provider.SanitizeToolArgsJSON([]byte(acc.args.String()))
		blocks = append(blocks, schema.ContentBlock{
			Type:      schema.BlockToolCall,
			CallID:    acc.id,
			Name:      acc.name,
			Arguments: json.RawMessage(args),
		})
	}

	return provider.Response{
		ID:         respID,
		Model:      model,
		Blocks:     blocks,
		Usage:      usage,
		StopReason: stopReason,
		Native: &schema.NativeEnvelope{
			Provider:   providerID,
			WireFormat: WireFormat,
			Model:      model,
			ResponseID: respID,
		},
	}, nil
}

func buildChatMessages(systemPrompt string, msgs []schema.Message) []map[string]any {
	var out []map[string]any
	if strings.TrimSpace(systemPrompt) != "" {
		out = append(out, map[string]any{"role": "system", "content": systemPrompt})
	}
	for _, m := range msgs {
		var (
			textParts   []string
			imageParts  []map[string]any
			toolCalls   []map[string]any
			toolResults []map[string]any
			toolImages  []map[string]any
		)
		for _, b := range m.Blocks {
			switch b.Type {
			case schema.BlockCompaction:
				if b.Text != "" {
					out = append(out, map[string]any{"role": "system", "content": b.Text})
				}
			case schema.BlockText:
				if b.Text != "" {
					textParts = append(textParts, b.Text)
				}
			case schema.BlockImage:
				if b.Image != nil {
					if part := buildChatImagePart(*b.Image); part != nil {
						if b.Text != "" && m.Role == schema.RoleAssistant {
							imageParts = append(imageParts, map[string]any{
								"type": "text",
								"text": "[Generated image prompt: " + b.Text + "]",
							})
						}
						imageParts = append(imageParts, part)
					}
				}
			case schema.BlockToolCall:
				toolCalls = append(toolCalls, map[string]any{
					"id":   b.CallID,
					"type": "function",
					"function": map[string]any{
						"name":      b.Name,
						"arguments": provider.SanitizeToolArgsJSON(b.Arguments),
					},
				})
			case schema.BlockToolResult:
				toolResults = append(toolResults, map[string]any{
					"role":         "tool",
					"tool_call_id": b.CallID,
					"content":      b.Output,
				})
				if len(b.Images) > 0 {
					label := b.Name
					if label == "" {
						label = "tool"
					}
					toolImages = append(toolImages, map[string]any{
						"type": "text",
						"text": "[Image output from " + label + "]",
					})
					for _, img := range b.Images {
						if part := buildChatImagePart(img); part != nil {
							toolImages = append(toolImages, part)
						}
					}
				}
			}
		}
		if len(textParts) > 0 || len(toolCalls) > 0 || (len(imageParts) > 0 && m.Role != schema.RoleAssistant) {
			role := string(m.Role)
			if role == string(schema.RoleDeveloper) {
				role = "system"
			}
			msg := map[string]any{"role": role}
			if len(imageParts) > 0 && role == "user" {
				content := make([]map[string]any, 0, len(textParts)+len(imageParts))
				if len(textParts) > 0 {
					content = append(content, map[string]any{
						"type": "text",
						"text": strings.Join(textParts, "\n"),
					})
				}
				content = append(content, imageParts...)
				msg["content"] = content
			} else if len(textParts) > 0 {
				msg["content"] = strings.Join(textParts, "\n")
			}
			if len(toolCalls) > 0 {
				msg["tool_calls"] = toolCalls
			}
			out = append(out, msg)
		}
		if len(imageParts) > 0 && m.Role == schema.RoleAssistant {
			out = append(out, map[string]any{
				"role":    "user",
				"content": imageParts,
			})
		}
		out = append(out, toolResults...)
		if len(toolImages) > 0 {
			out = append(out, map[string]any{
				"role":    "user",
				"content": toolImages,
			})
		}
	}
	return out
}

func buildChatImagePart(img schema.ImageSource) map[string]any {
	u := img.DataURL()
	if u == "" {
		return nil
	}
	imageURL := map[string]any{"url": u}
	switch strings.ToLower(strings.TrimSpace(img.Detail)) {
	case "low", "high", "auto":
		imageURL["detail"] = strings.ToLower(strings.TrimSpace(img.Detail))
	case "original":
		imageURL["detail"] = "high"
	}
	return map[string]any{
		"type":      "image_url",
		"image_url": imageURL,
	}
}

func buildChatTools(defs []schema.ToolDef) []map[string]any {
	out := make([]map[string]any, 0, len(defs))
	for _, t := range defs {
		params := t.InputSchema
		if len(bytes.TrimSpace(params)) == 0 {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		fn := map[string]any{
			"name":       t.Name,
			"parameters": params,
		}
		if t.Description != "" {
			fn["description"] = t.Description
		}
		out = append(out, map[string]any{
			"type":     "function",
			"function": fn,
		})
	}
	return out
}
