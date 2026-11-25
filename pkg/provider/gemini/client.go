package gemini

import (
	"bufio"
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

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
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

type TokenSource func(ctx context.Context) (string, error)

type Config struct {
	ProviderID  string // "vertex" | "gemini"
	Mode        Mode
	BaseURL     string
	APIKey      string      // AI Studio key
	Project     string      // Vertex project
	Location    string      // Vertex location ("global" | "us-central1" ...)
	TokenSource TokenSource // Vertex OAuth/Workload Identity Bearer token
	HTTPClient  *http.Client
}

// Client speaks Google Gemini's native generateContent / streamGenerateContent
// wire format directly, preserving thought parts and cryptographic
// thoughtSignature fields verbatim across turns and restarts, plus native
// googleSearch, urlContext, and explicit cachedContent.
type Client struct {
	cfg        Config
	httpClient *http.Client
}

func New(cfg Config) *Client {
	if cfg.Mode == "" {
		if cfg.APIKey != "" && cfg.Project == "" {
			cfg.Mode = ModeAIStudio
		} else {
			cfg.Mode = ModeVertex
		}
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

func (c *Client) ID() string         { return c.cfg.ProviderID }
func (c *Client) WireFormat() string { return WireFormat }

type geminiContent struct {
	Role  string           `json:"role"`
	Parts []map[string]any `json:"parts"`
}

func (c *Client) buildRequestBody(req provider.Request) ([]byte, error) {
	if strings.TrimSpace(req.Model) == "" {
		return nil, errors.New("gemini: model is required")
	}

	contents := buildGeminiContents(req.Messages, WireFormat)
	payload := map[string]any{
		"contents": contents,
	}
	if strings.TrimSpace(req.SystemPrompt) != "" {
		payload["systemInstruction"] = map[string]any{
			"parts": []map[string]any{{"text": req.SystemPrompt}},
		}
	}
	if strings.TrimSpace(req.CachedContentName) != "" {
		payload["cachedContent"] = strings.TrimSpace(req.CachedContentName)
	}

	genCfg := map[string]any{}
	if req.MaxOutputTokens > 0 {
		genCfg["maxOutputTokens"] = req.MaxOutputTokens
	}
	if req.Temperature != nil {
		genCfg["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		genCfg["topP"] = *req.TopP
	}
	thinkingCfg := map[string]any{"includeThoughts": true}
	if level := mapGeminiThinkingLevel(req.ReasoningEffort); level != "" {
		thinkingCfg["thinkingLevel"] = level
	}
	genCfg["thinkingConfig"] = thinkingCfg
	payload["generationConfig"] = genCfg

	if tools := buildGeminiTools(req.Tools, req.HostedTools); len(tools) > 0 {
		payload["tools"] = tools
	}
	if tc := buildGeminiToolConfig(req.ToolChoice); tc != nil {
		payload["toolConfig"] = tc
	}
	return json.Marshal(payload)
}

func (c *Client) endpointForModel(model string) (string, error) {
	cleanModel := strings.TrimPrefix(strings.TrimSpace(model), "google/")
	if c.cfg.Mode == ModeAIStudio {
		return fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse", c.cfg.BaseURL, url.PathEscape(cleanModel)), nil
	}
	if strings.Contains(c.cfg.BaseURL, "/publishers/google") {
		return fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse", c.cfg.BaseURL, url.PathEscape(cleanModel)), nil
	}
	if c.cfg.Project == "" {
		return "", errors.New("vertex gemini: Project is required")
	}
	loc := c.cfg.Location
	if loc == "" {
		loc = "global"
	}
	return fmt.Sprintf(
		"%s/v1/projects/%s/locations/%s/publishers/google/models/%s:streamGenerateContent?alt=sse",
		c.cfg.BaseURL,
		url.PathEscape(c.cfg.Project),
		url.PathEscape(loc),
		url.PathEscape(cleanModel),
	), nil
}

func (c *Client) Stream(ctx context.Context, req provider.Request, emit provider.StreamCallback) (provider.Response, error) {
	body, err := c.buildRequestBody(req)
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
		httpReq.Header.Set("Accept", "text/event-stream")
		if c.cfg.Mode == ModeAIStudio {
			if c.cfg.APIKey != "" {
				httpReq.Header.Set("x-goog-api-key", c.cfg.APIKey)
			}
		} else {
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
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)

	var (
		model       = fallbackModel
		stopReason  = "STOP"
		usage       schema.Usage
		nativeParts []json.RawMessage
		canonical   []schema.ContentBlock
		textBuf     strings.Builder
		thoughtBuf  strings.Builder
		citations   []schema.Citation
		seenCits    = map[string]bool{}
	)

	flushText := func() {
		if textBuf.Len() == 0 {
			return
		}
		canonical = append(canonical, schema.ContentBlock{
			Type:      schema.BlockText,
			Text:      textBuf.String(),
			Citations: citations,
		})
		textBuf.Reset()
		citations = nil
	}
	flushThought := func() {
		if thoughtBuf.Len() == 0 {
			return
		}
		canonical = append(canonical, schema.ContentBlock{
			Type:     schema.BlockThinking,
			Thinking: thoughtBuf.String(),
		})
		thoughtBuf.Reset()
	}

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}
		var chunk struct {
			ModelVersion string `json:"modelVersion"`
			Candidates   []struct {
				Content *struct {
					Role  string            `json:"role"`
					Parts []json.RawMessage `json:"parts"`
				} `json:"content"`
				FinishReason      string `json:"finishReason"`
				GroundingMetadata *struct {
					GroundingChunks []struct {
						Web *struct {
							URI   string `json:"uri"`
							Title string `json:"title"`
						} `json:"web"`
					} `json:"groundingChunks"`
				} `json:"groundingMetadata"`
			} `json:"candidates"`
			UsageMetadata *struct {
				PromptTokenCount        int64 `json:"promptTokenCount"`
				CachedContentTokenCount int64 `json:"cachedContentTokenCount"`
				CandidatesTokenCount    int64 `json:"candidatesTokenCount"`
				ThoughtsTokenCount      int64 `json:"thoughtsTokenCount"`
				TotalTokenCount         int64 `json:"totalTokenCount"`
			} `json:"usageMetadata"`
		}
		if json.Unmarshal([]byte(data), &chunk) != nil {
			continue
		}
		if chunk.ModelVersion != "" {
			model = chunk.ModelVersion
		}
		if chunk.UsageMetadata != nil {
			u := chunk.UsageMetadata
			usage.InputTokens = u.PromptTokenCount
			usage.CachedInputTokens = u.CachedContentTokenCount
			usage.OutputTokens = u.CandidatesTokenCount
			usage.ReasoningTokens = u.ThoughtsTokenCount
			usage.TotalTokens = u.TotalTokenCount
			if usage.TotalTokens == 0 {
				usage.TotalTokens = u.PromptTokenCount + u.CandidatesTokenCount + u.ThoughtsTokenCount
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
			if cand.Content == nil {
				continue
			}
			for _, rawPart := range cand.Content.Parts {
				cp := append(json.RawMessage(nil), rawPart...)
				nativeParts = append(nativeParts, cp)

				var part struct {
					Text             string `json:"text"`
					Thought          bool   `json:"thought"`
					ThoughtSignature string `json:"thoughtSignature"`
					FunctionCall     *struct {
						ID   string          `json:"id"`
						Name string          `json:"name"`
						Args json.RawMessage `json:"args"`
					} `json:"functionCall"`
				}
				if json.Unmarshal(cp, &part) != nil {
					continue
				}

				if part.Thought && part.Text != "" {
					thoughtBuf.WriteString(part.Text)
					if emit != nil {
						emit(provider.StreamChunk{Kind: provider.DeltaThinking, Text: part.Text})
					}
					continue
				}
				if part.Text != "" {
					flushThought()
					textBuf.WriteString(part.Text)
					if emit != nil {
						emit(provider.StreamChunk{Kind: provider.DeltaText, Text: part.Text})
					}
				}
				if part.FunctionCall != nil {
					flushThought()
					flushText()
					callID := part.FunctionCall.ID
					if callID == "" {
						callID = "gcall_" + uuid.NewString()[:8]
					}
					args := part.FunctionCall.Args
					if len(bytes.TrimSpace(args)) == 0 {
						args = json.RawMessage(`{}`)
					}
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
	}
	if err := sc.Err(); err != nil {
		return provider.Response{}, err
	}
	flushThought()
	flushText()

	// Wrap native model content object verbatim so replay preserves every
	// thoughtSignature and thought part byte-for-byte.
	modelContentObj, _ := json.Marshal(map[string]any{
		"role":  "model",
		"parts": nativeParts,
	})
	return provider.Response{
		ID:         "gem_" + uuid.NewString()[:12],
		Model:      model,
		Blocks:     canonical,
		Usage:      usage,
		StopReason: stopReason,
		Native: &schema.NativeEnvelope{
			Provider:   providerID,
			WireFormat: WireFormat,
			Model:      model,
			RawItems:   []json.RawMessage{modelContentObj},
		},
	}, nil
}

func buildGeminiContents(msgs []schema.Message, wireFormat string) []any {
	out := make([]any, 0, len(msgs))
	callNames := make(map[string]string)

	for _, m := range msgs {
		for _, b := range m.Blocks {
			if b.Type == schema.BlockToolCall && b.CallID != "" && b.Name != "" {
				callNames[b.CallID] = b.Name
			}
		}
		if m.Native != nil && m.Native.WireFormat == wireFormat && len(m.Native.RawItems) > 0 {
			for _, raw := range m.Native.RawItems {
				out = append(out, raw)
			}
			continue
		}

		role := "user"
		if m.Role == schema.RoleAssistant {
			role = "model"
		}
		var parts []map[string]any
		for _, b := range m.Blocks {
			switch b.Type {
			case schema.BlockCompaction:
				if b.Text != "" {
					parts = append(parts, map[string]any{"text": b.Text})
				}
			case schema.BlockText:
				if b.Text != "" {
					parts = append(parts, map[string]any{"text": b.Text})
				}
			case schema.BlockImage:
				if b.Image != nil && b.Image.Base64 != "" {
					mt := b.Image.MediaType
					if mt == "" {
						mt = "image/png"
					}
					parts = append(parts, map[string]any{
						"inlineData": map[string]any{
							"mimeType": mt,
							"data":     b.Image.Base64,
						},
					})
				}
			case schema.BlockToolCall:
				var argsObj map[string]any
				if json.Unmarshal(b.Arguments, &argsObj) != nil || argsObj == nil {
					argsObj = map[string]any{}
				}
				fc := map[string]any{
					"name": b.Name,
					"args": argsObj,
				}
				if b.CallID != "" {
					fc["id"] = b.CallID
				}
				part := map[string]any{"functionCall": fc}
				if b.ThoughtSignBlob != "" {
					part["thoughtSignature"] = b.ThoughtSignBlob
				}
				parts = append(parts, part)
			case schema.BlockToolResult:
				name := b.Name
				if name == "" {
					name = callNames[b.CallID]
				}
				if name == "" {
					name = "tool"
				}
				fr := map[string]any{
					"name":     name,
					"response": map[string]any{"result": b.Output, "is_error": b.IsError},
				}
				if b.CallID != "" {
					fr["id"] = b.CallID
				}
				parts = append(parts, map[string]any{"functionResponse": fr})
			}
		}
		if len(parts) > 0 {
			out = append(out, geminiContent{Role: role, Parts: parts})
		}
	}
	return out
}

func buildGeminiTools(defs []schema.ToolDef, hosted []schema.HostedTool) []map[string]any {
	var tools []map[string]any
	if len(defs) > 0 {
		decls := make([]map[string]any, 0, len(defs))
		for _, t := range defs {
			decl := map[string]any{
				"name": t.Name,
			}
			if t.Description != "" {
				decl["description"] = t.Description
			}
			if len(bytes.TrimSpace(t.InputSchema)) > 0 {
				var obj map[string]any
				if json.Unmarshal(t.InputSchema, &obj) == nil && obj != nil {
					cleanGeminiSchema(obj)
					decl["parameters"] = obj
				}
			}
			decls = append(decls, decl)
		}
		tools = append(tools, map[string]any{"functionDeclarations": decls})
	}
	for _, h := range hosted {
		switch h.Type {
		case "google_search", "web_search":
			tools = append(tools, map[string]any{"googleSearch": map[string]any{}})
		case "url_context", "web_fetch":
			tools = append(tools, map[string]any{"urlContext": map[string]any{}})
		default:
			cfg := h.Config
			if cfg == nil {
				cfg = map[string]any{}
			}
			tools = append(tools, map[string]any{h.Type: cfg})
		}
	}
	return tools
}

func cleanGeminiSchema(obj map[string]any) {
	delete(obj, "additionalProperties")
	delete(obj, "$schema")
	if props, ok := obj["properties"].(map[string]any); ok {
		for _, v := range props {
			if child, ok := v.(map[string]any); ok {
				cleanGeminiSchema(child)
			}
		}
	}
	if items, ok := obj["items"].(map[string]any); ok {
		cleanGeminiSchema(items)
	}
}

func buildGeminiToolConfig(tc *schema.ToolChoice) map[string]any {
	if tc == nil || tc.Mode == "" {
		return nil
	}
	cfg := map[string]any{}
	switch tc.Mode {
	case "auto":
		cfg["mode"] = "AUTO"
	case "required":
		cfg["mode"] = "ANY"
	case "none":
		cfg["mode"] = "NONE"
	case "tool":
		cfg["mode"] = "ANY"
		if tc.Name != "" {
			cfg["allowedFunctionNames"] = []string{tc.Name}
		}
	default:
		return nil
	}
	return map[string]any{"functionCallingConfig": cfg}
}

func mapGeminiThinkingLevel(effort string) string {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "none":
		return "MINIMAL"
	case "minimal":
		return "MINIMAL"
	case "low":
		return "LOW"
	case "medium":
		return "MEDIUM"
	case "high":
		return "HIGH"
	case "xhigh", "max":
		return "HIGH"
	default:
		return ""
	}
}
