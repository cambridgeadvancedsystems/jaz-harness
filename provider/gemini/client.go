package gemini

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

	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/google/uuid"
)

const (
	DefaultAIStudioURL = "https://generativelanguage.googleapis.com"
	WireFormat         = "gemini_generate_content"
)

type Mode string

const (
	ModeVertex   Mode = "vertex"
	ModeAIStudio Mode = "ai_studio"
)

type APISurface string

const (
	SurfaceGenerateContent APISurface = "generate_content" // Vertex v1beta1 streamGenerateContent (default)
	SurfaceInteractions    APISurface = "interactions"     // Stateless Vertex v1beta1 Interactions API (store=false)
)

type TokenSource func(ctx context.Context) (string, error)

type Config struct {
	ProviderID   string // "vertex" | "gemini"
	Mode         Mode
	Surface      APISurface
	BaseURL      string
	APIKey       string      // AI Studio key or relay Bearer token when RelayMode=true
	Project      string      // Vertex project
	Location     string      // Vertex location ("global" | "us-central1" ...)
	TokenSource  TokenSource // Vertex OAuth / Workload Identity Bearer token
	RelayMode    bool        // Route through a native LLM relay using Bearer auth without requiring Project
	ExtraHeaders map[string]string
	HTTPClient   *http.Client
}

// Client speaks Google Gemini's native Vertex v1beta1 API (streamGenerateContent
// with toolConfig.includeServerSideToolInvocations, or stateless Interactions API
// with store=false), preserving thought parts and thoughtSignature verbatim.
type Client struct {
	cfg        Config
	httpClient *http.Client
}

func New(cfg Config) *Client {
	if cfg.Mode == "" {
		if cfg.APIKey != "" && cfg.Project == "" && !cfg.RelayMode {
			cfg.Mode = ModeAIStudio
		} else {
			cfg.Mode = ModeVertex
		}
	}
	if cfg.Surface == "" {
		cfg.Surface = SurfaceGenerateContent
	}
	if cfg.ProviderID == "" {
		if cfg.Mode == ModeVertex {
			cfg.ProviderID = "vertex"
		} else {
			cfg.ProviderID = "gemini"
		}
	}
	if cfg.BaseURL == "" {
		if cfg.Mode == ModeAIStudio {
			cfg.BaseURL = DefaultAIStudioURL
		} else {
			loc := cfg.Location
			if loc == "" {
				loc = "global"
			}
			if loc == "global" {
				cfg.BaseURL = "https://aiplatform.googleapis.com"
			} else {
				cfg.BaseURL = fmt.Sprintf("https://%s-aiplatform.googleapis.com", loc)
			}
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

func (c *Client) ID() string { return c.cfg.ProviderID }

func (c *Client) WireFormat() string {
	if c.cfg.Surface == SurfaceInteractions {
		return WireFormatInteractions
	}
	return WireFormat
}

func stripLegacyOpenAPISuffix(raw string) string {
	trimmed := strings.TrimRight(strings.TrimSpace(raw), "/")
	return strings.TrimSuffix(trimmed, "/endpoints/openapi")
}

func (c *Client) buildRequestBody(req provider.Request) ([]byte, error) {
	if strings.TrimSpace(req.Model) == "" {
		return nil, errors.New("gemini: model is required")
	}

	gem4 := isGemini4Model(req.Model)
	contents := buildGeminiContents(req.Messages, WireFormat)
	payload := map[string]any{
		"contents": contents,
	}
	if strings.TrimSpace(req.SystemPrompt) != "" {
		payload["systemInstruction"] = map[string]any{
			"parts": []map[string]any{{"text": req.SystemPrompt}},
		}
	}
	// Explicit cachedContent is supported on Gemini 2.5/3 regional only;
	// Gemini 4 uses implicit/durable prefix caching and rejects explicit handles.
	if !gem4 && strings.TrimSpace(req.CachedContentName) != "" {
		payload["cachedContent"] = strings.TrimSpace(req.CachedContentName)
	}

	genCfg := map[string]any{}
	if req.MaxOutputTokens > 0 {
		genCfg["maxOutputTokens"] = req.MaxOutputTokens
	}
	// Gemini 4 recommends omitting temperature and topP for thinking models.
	if !gem4 {
		if req.Temperature != nil {
			genCfg["temperature"] = *req.Temperature
		}
		if req.TopP != nil {
			genCfg["topP"] = *req.TopP
		}
	}
	thinkingCfg := map[string]any{"includeThoughts": true}
	if level := mapGeminiThinkingLevel(req.Model, req.ReasoningEffort); level != "" {
		thinkingCfg["thinkingLevel"] = level
	}
	genCfg["thinkingConfig"] = thinkingCfg
	payload["generationConfig"] = genCfg

	tools, hasFunctions, hasHosted := buildGeminiTools(req.Tools, req.HostedTools)
	if len(tools) > 0 {
		payload["tools"] = tools
	}
	if tc := buildGeminiToolConfig(req.ToolChoice, hasFunctions && hasHosted); tc != nil {
		payload["toolConfig"] = tc
	}
	if tok := strings.TrimSpace(req.Metadata["continuation_token"]); tok != "" {
		payload["continuationToken"] = tok
	}
	return json.Marshal(payload)
}

func (c *Client) endpointForModelAction(model, action string) (string, error) {
	cleanModel := strings.TrimPrefix(strings.TrimSpace(model), "google/")
	base := stripLegacyOpenAPISuffix(c.cfg.BaseURL)
	if c.cfg.RelayMode {
		return fmt.Sprintf("%s/models/%s:%s", base, url.PathEscape(cleanModel), action), nil
	}
	if c.cfg.Mode == ModeAIStudio {
		return fmt.Sprintf("%s/v1beta/models/%s:%s", base, url.PathEscape(cleanModel), action), nil
	}
	if strings.Contains(base, "/publishers/google") {
		return fmt.Sprintf("%s/models/%s:%s", base, url.PathEscape(cleanModel), action), nil
	}
	if c.cfg.Project == "" {
		return "", errors.New("vertex gemini: Project is required")
	}
	loc := c.cfg.Location
	if loc == "" {
		loc = "global"
	}
	return fmt.Sprintf(
		"%s/v1beta1/projects/%s/locations/%s/publishers/google/models/%s:%s",
		base,
		url.PathEscape(c.cfg.Project),
		url.PathEscape(loc),
		url.PathEscape(cleanModel),
		action,
	), nil
}

func (c *Client) endpointForModel(model string) (string, error) {
	return c.endpointForModelAction(model, "streamGenerateContent?alt=sse")
}

func (c *Client) doHTTP(ctx context.Context, endpoint string, body []byte) (*http.Response, error) {
	return c.doHTTPWithAccept(ctx, endpoint, body, "text/event-stream")
}

func (c *Client) doHTTPWithAccept(ctx context.Context, endpoint string, body []byte, accept string) (*http.Response, error) {
	return provider.DoWithRetry(ctx, 3, func(_ int) (*http.Response, error) {
		httpReq, rerr := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if rerr != nil {
			return nil, rerr
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", accept)
		for k, v := range c.cfg.ExtraHeaders {
			httpReq.Header.Set(k, v)
		}
		switch {
		case c.cfg.RelayMode:
			if c.cfg.APIKey != "" {
				httpReq.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
			} else if c.cfg.TokenSource != nil {
				tok, terr := c.cfg.TokenSource(ctx)
				if terr != nil {
					return nil, terr
				}
				httpReq.Header.Set("Authorization", "Bearer "+tok)
			}
		case c.cfg.Mode == ModeAIStudio:
			if c.cfg.APIKey != "" {
				httpReq.Header.Set("x-goog-api-key", c.cfg.APIKey)
			}
		default:
			if c.cfg.TokenSource == nil {
				return nil, errors.New("vertex gemini: TokenSource not configured")
			}
			tok, terr := c.cfg.TokenSource(ctx)
			if terr != nil {
				return nil, terr
			}
			httpReq.Header.Set("Authorization", "Bearer "+tok)
		}
		return c.httpClient.Do(httpReq)
	})
}

func (c *Client) Stream(ctx context.Context, req provider.Request, emit provider.StreamCallback) (provider.Response, error) {
	if c.cfg.Surface == SurfaceInteractions {
		return c.streamInteractions(ctx, req, emit)
	}
	body, err := c.buildRequestBody(req)
	if err != nil {
		return provider.Response{}, err
	}
	endpoint, err := c.endpointForModel(req.Model)
	if err != nil {
		return provider.Response{}, err
	}

	resp, err := c.doHTTP(ctx, endpoint, body)
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

	return parseGeminiSSE(resp.Body, c.ID(), req.Model, emit)
}

func parseGeminiSSE(r io.Reader, providerID, fallbackModel string, emit provider.StreamCallback) (provider.Response, error) {
	var (
		model             = fallbackModel
		stopReason        = "STOP"
		continuationToken string
		usage             schema.Usage
		nativeParts       []json.RawMessage
		canonical         []schema.ContentBlock
		textBuf           strings.Builder
		textSig           string
		thoughtBuf        strings.Builder
		thoughtSig        string
		citations         []schema.Citation
		seenCits          = map[string]bool{}
	)

	flushText := func() {
		if textBuf.Len() == 0 && textSig == "" {
			return
		}
		text := textBuf.String()
		partMap := map[string]any{"text": text}
		if textSig != "" {
			partMap["thoughtSignature"] = textSig
		}
		if raw, err := json.Marshal(partMap); err == nil {
			nativeParts = append(nativeParts, raw)
		}
		canonical = append(canonical, schema.ContentBlock{
			Type:            schema.BlockText,
			Text:            text,
			Citations:       citations,
			ThoughtSignBlob: textSig,
		})
		textBuf.Reset()
		textSig = ""
		citations = nil
	}
	flushThought := func() {
		if thoughtBuf.Len() == 0 && thoughtSig == "" {
			return
		}
		thought := thoughtBuf.String()
		partMap := map[string]any{"thought": true, "text": thought}
		if thoughtSig != "" {
			partMap["thoughtSignature"] = thoughtSig
		}
		if raw, err := json.Marshal(partMap); err == nil {
			nativeParts = append(nativeParts, raw)
		}
		canonical = append(canonical, schema.ContentBlock{
			Type:            schema.BlockThinking,
			Thinking:        thought,
			ThoughtSignBlob: thoughtSig,
		})
		thoughtBuf.Reset()
		thoughtSig = ""
	}

	err := provider.ScanLines(r, provider.DefaultMaxLineBytes, func(rawLine string) error {
		line := strings.TrimSpace(rawLine)
		if !strings.HasPrefix(line, "data:") {
			return nil
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			return nil
		}
		var chunk struct {
			ModelVersion      string `json:"modelVersion"`
			ContinuationToken string `json:"continuationToken"`
			Candidates        []struct {
				Content *struct {
					Role  string            `json:"role"`
					Parts []json.RawMessage `json:"parts"`
				} `json:"content"`
				FinishReason      string `json:"finishReason"`
				GroundingMetadata *struct {
					WebSearchQueries []string `json:"webSearchQueries"`
					GroundingChunks  []struct {
						Web *struct {
							URI   string `json:"uri"`
							Title string `json:"title"`
						} `json:"web"`
					} `json:"groundingChunks"`
				} `json:"groundingMetadata"`
				URLContextMetadata *struct {
					URLMetadata []struct {
						RetrievedURL string `json:"retrievedUrl"`
					} `json:"urlMetadata"`
				} `json:"urlContextMetadata"`
			} `json:"candidates"`
			UsageMetadata *struct {
				PromptTokenCount        int64 `json:"promptTokenCount"`
				ToolUsePromptTokenCount int64 `json:"toolUsePromptTokenCount"`
				CachedContentTokenCount int64 `json:"cachedContentTokenCount"`
				CandidatesTokenCount    int64 `json:"candidatesTokenCount"`
				ThoughtsTokenCount      int64 `json:"thoughtsTokenCount"`
				TotalTokenCount         int64 `json:"totalTokenCount"`
			} `json:"usageMetadata"`
		}
		if json.Unmarshal([]byte(data), &chunk) != nil {
			return nil
		}
		if chunk.ModelVersion != "" {
			model = chunk.ModelVersion
		}
		if chunk.ContinuationToken != "" {
			continuationToken = chunk.ContinuationToken
		}
		if chunk.UsageMetadata != nil {
			u := chunk.UsageMetadata
			usage.InputTokens = u.PromptTokenCount + u.ToolUsePromptTokenCount
			usage.CachedInputTokens = u.CachedContentTokenCount
			usage.OutputTokens = u.CandidatesTokenCount
			usage.ReasoningTokens = u.ThoughtsTokenCount
			usage.TotalTokens = u.TotalTokenCount
			if usage.TotalTokens == 0 {
				usage.TotalTokens = usage.InputTokens + u.CandidatesTokenCount + u.ThoughtsTokenCount
			}
		}

		for _, cand := range chunk.Candidates {
			if cand.FinishReason != "" {
				stopReason = cand.FinishReason
			}
			if cand.GroundingMetadata != nil {
				for _, gc := range cand.GroundingMetadata.GroundingChunks {
					if gc.Web != nil && gc.Web.URI != "" && !seenCits[gc.Web.URI] {
						seenCits[gc.Web.URI] = true
						citations = append(citations, schema.Citation{
							Type:  "url_citation",
							URL:   gc.Web.URI,
							Title: gc.Web.Title,
						})
					}
				}
			}
			if cand.URLContextMetadata != nil {
				for _, um := range cand.URLContextMetadata.URLMetadata {
					if um.RetrievedURL != "" && !seenCits[um.RetrievedURL] {
						seenCits[um.RetrievedURL] = true
						citations = append(citations, schema.Citation{
							Type: "url_citation",
							URL:  um.RetrievedURL,
						})
					}
				}
			}
			if cand.Content == nil {
				continue
			}
			for _, rawPart := range cand.Content.Parts {
				cp := append(json.RawMessage(nil), rawPart...)

				var part struct {
					Text             string `json:"text"`
					Thought          bool   `json:"thought"`
					ThoughtSignature string `json:"thoughtSignature"`
					InlineData       *struct {
						MimeType string `json:"mimeType"`
						Data     string `json:"data"`
					} `json:"inlineData"`
					FunctionCall *struct {
						ID   string          `json:"id"`
						Name string          `json:"name"`
						Args json.RawMessage `json:"args"`
					} `json:"functionCall"`
					ToolCall *struct {
						ID       string `json:"id"`
						ToolType string `json:"toolType"`
					} `json:"toolCall"`
					ToolResponse *struct {
						ID       string `json:"id"`
						ToolType string `json:"toolType"`
					} `json:"toolResponse"`
				}
				if json.Unmarshal(cp, &part) != nil {
					nativeParts = append(nativeParts, cp)
					continue
				}

				if part.InlineData != nil && part.InlineData.Data != "" {
					flushThought()
					flushText()
					nativeParts = append(nativeParts, cp)
					mt := part.InlineData.MimeType
					if mt == "" {
						mt = schema.DetectBase64ImageMediaType(part.InlineData.Data)
					}
					canonical = append(canonical, schema.ContentBlock{
						Type: schema.BlockImage,
						Image: &schema.ImageSource{
							MediaType: mt,
							Base64:    part.InlineData.Data,
						},
						ThoughtSignBlob: part.ThoughtSignature,
						ProviderOpaque:  cp,
					})
					continue
				}

				if part.Thought {
					flushText()
					if part.ThoughtSignature != "" {
						thoughtSig = part.ThoughtSignature
					}
					if part.Text != "" {
						thoughtBuf.WriteString(part.Text)
						if emit != nil {
							emit(provider.StreamChunk{Kind: provider.DeltaThinking, Text: part.Text})
						}
					}
					continue
				}
				if part.ToolCall != nil {
					flushThought()
					flushText()
					nativeParts = append(nativeParts, cp)
					if emit != nil {
						emit(provider.StreamChunk{Kind: provider.DeltaHostedToolStart, CallID: part.ToolCall.ID, ToolName: part.ToolCall.ToolType})
					}
					canonical = append(canonical, schema.ContentBlock{
						Type:            schema.BlockHostedToolCall,
						CallID:          part.ToolCall.ID,
						Name:            part.ToolCall.ToolType,
						ThoughtSignBlob: part.ThoughtSignature,
						ProviderOpaque:  cp,
					})
					continue
				}
				if part.ToolResponse != nil {
					flushThought()
					flushText()
					nativeParts = append(nativeParts, cp)
					if emit != nil {
						emit(provider.StreamChunk{Kind: provider.DeltaHostedToolComplete, CallID: part.ToolResponse.ID, ToolName: part.ToolResponse.ToolType})
					}
					canonical = append(canonical, schema.ContentBlock{
						Type:           schema.BlockHostedToolResult,
						CallID:         part.ToolResponse.ID,
						Name:           part.ToolResponse.ToolType,
						ProviderOpaque: cp,
					})
					continue
				}
				if part.Text != "" || (part.FunctionCall == nil && part.ThoughtSignature != "") {
					flushThought()
					if part.ThoughtSignature != "" {
						textSig = part.ThoughtSignature
					}
					if part.Text != "" {
						textBuf.WriteString(part.Text)
						if emit != nil {
							emit(provider.StreamChunk{Kind: provider.DeltaText, Text: part.Text})
						}
					}
				}
				if part.FunctionCall != nil {
					flushThought()
					flushText()
					nativeParts = append(nativeParts, cp)
					callID := part.FunctionCall.ID
					if callID == "" {
						callID = "gcall_" + uuid.NewString()[:8]
					}
					args := json.RawMessage(provider.SanitizeToolArgsJSON(part.FunctionCall.Args))
					if emit != nil {
						emit(provider.StreamChunk{
							Kind:      provider.DeltaToolCallStart,
							CallID:    callID,
							ToolName:  part.FunctionCall.Name,
							Arguments: string(args),
						})
					}
					canonical = append(canonical, schema.ContentBlock{
						Type:            schema.BlockToolCall,
						CallID:          callID,
						Name:            part.FunctionCall.Name,
						Arguments:       args,
						ThoughtSignBlob: part.ThoughtSignature,
					})
				}
			}
		}
		return nil
	})
	if err != nil {
		return provider.Response{}, err
	}
	flushThought()
	flushText()

	var rawItems []json.RawMessage
	if len(nativeParts) > 0 {
		modelContentObj, _ := json.Marshal(map[string]any{
			"role":  "model",
			"parts": nativeParts,
		})
		rawItems = []json.RawMessage{modelContentObj}
	}
	return provider.Response{
		ID:                "gem_" + uuid.NewString()[:12],
		Model:             model,
		Blocks:            canonical,
		Usage:             usage,
		StopReason:        stopReason,
		ContinuationToken: continuationToken,
		Native: &schema.NativeEnvelope{
			Provider:   providerID,
			WireFormat: WireFormat,
			Model:      model,
			RawItems:   rawItems,
		},
	}, nil
}
