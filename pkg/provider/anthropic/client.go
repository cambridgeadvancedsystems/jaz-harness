package anthropic

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
	"sort"
	"strings"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/auth"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
)

const (
	DefaultBaseURL       = "https://api.anthropic.com"
	DefaultAPIVersion    = "2023-06-01"
	VertexAPIVersion     = "vertex-2023-10-16"
	BedrockAPIVersion    = "bedrock-2023-05-31"
	WireFormat           = "anthropic_messages"
	BetaWebFetch         = "web-fetch-2025-09-10"
	BetaAdvancedToolUse  = "advanced-tool-use-2025-11-20"
	BetaCompaction       = "compact-2026-01-12"
	BetaInterleavedThink = "interleaved-thinking-2025-05-14"
	defaultMaxTokens     = 8192
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
	APIKey         string
	VertexProject  string
	VertexLocation string
	VertexToken    TokenSource
	AWS            auth.AWSCredentials
	APIVersion     string
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

type anthropicMessage struct {
	Role    string           `json:"role"`
	Content []map[string]any `json:"content"`
}

func (c *Client) buildRequestBody(req provider.Request) ([]byte, []string, error) {
	if strings.TrimSpace(req.Model) == "" {
		return nil, nil, errors.New("anthropic: model is required")
	}
	enableCache := req.EnableNativeCache
	messages := buildAnthropicMessages(req.Messages, WireFormat)
	if enableCache {
		applyConversationCacheBreakpoints(messages)
	}

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
	if len(tools) > 0 && enableCache {
		tools[len(tools)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
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
			maxTokens = thinkingBudget + 4096
		}
	}
	if req.EnableCompaction {
		addBeta(BetaCompaction)
	}

	out := map[string]any{
		"max_tokens": maxTokens,
		"messages":   messages,
	}
	if c.cfg.Mode == ModeDirect {
		out["model"] = req.Model
		out["stream"] = true
	} else if c.cfg.Mode == ModeVertex {
		out["anthropic_version"] = c.cfg.APIVersion
		out["stream"] = true
	} else if c.cfg.Mode == ModeBedrock {
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
		loc := c.cfg.VertexLocation
		if loc == "" {
			loc = "us-east5"
		}
		if strings.Contains(c.cfg.BaseURL, "/publishers/anthropic") {
			return fmt.Sprintf("%s/models/%s:streamRawPredict", c.cfg.BaseURL, url.PathEscape(model)), nil
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
		switch c.cfg.Mode {
		case ModeDirect:
			httpReq.Header.Set("x-api-key", c.cfg.APIKey)
			httpReq.Header.Set("anthropic-version", c.cfg.APIVersion)
			httpReq.Header.Set("Accept", "text/event-stream")
			if len(betas) > 0 {
				httpReq.Header.Set("anthropic-beta", strings.Join(betas, ","))
			}
		case ModeVertex:
			if c.cfg.VertexToken == nil {
				return nil, errors.New("vertex_anthropic: VertexToken source not configured")
			}
			tok, terr := c.cfg.VertexToken(ctx)
			if terr != nil {
				return nil, terr
			}
			httpReq.Header.Set("Authorization", "Bearer "+tok)
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

func readAnthropicSSE(r io.Reader, onEvent func([]byte) error) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		if err := onEvent([]byte(data)); err != nil {
			return err
		}
	}
	return sc.Err()
}

type blockAcc struct {
	blockType    string
	id           string
	name         string
	initialInput string
	textBuf      strings.Builder
	thinkingBuf  strings.Builder
	sigBuf       strings.Builder
	inputBuf     strings.Builder
	redactedData string
	rawServer    json.RawMessage
	citations    []schema.Citation
}

type anthropicStreamAcc struct {
	providerID string
	respID     string
	model      string
	stopReason string
	usage      schema.Usage
	blocks     map[int]*blockAcc
	emit       provider.StreamCallback
}

func newAnthropicStreamAcc(providerID, model string, emit provider.StreamCallback) *anthropicStreamAcc {
	return &anthropicStreamAcc{
		providerID: providerID,
		model:      model,
		blocks:     make(map[int]*blockAcc),
		emit:       emit,
	}
}

func (a *anthropicStreamAcc) handleJSONEvent(raw []byte) error {
	var ev struct {
		Type    string `json:"type"`
		Index   int    `json:"index"`
		Message *struct {
			ID    string `json:"id"`
			Model string `json:"model"`
			Usage *struct {
				InputTokens              int64 `json:"input_tokens"`
				CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
				CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
				OutputTokens             int64 `json:"output_tokens"`
			} `json:"usage"`
		} `json:"message"`
		ContentBlock json.RawMessage `json:"content_block"`
		Delta        *struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			Thinking    string `json:"thinking"`
			Signature   string `json:"signature"`
			PartialJSON string `json:"partial_json"`
			StopReason  string `json:"stop_reason"`
			Citation    *struct {
				Type      string `json:"type"`
				URL       string `json:"url"`
				Title     string `json:"title"`
				CitedText string `json:"cited_text"`
			} `json:"citation"`
		} `json:"delta"`
		Usage *struct {
			InputTokens              int64 `json:"input_tokens"`
			CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
			OutputTokens             int64 `json:"output_tokens"`
		} `json:"usage"`
		Error *struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &ev); err != nil {
		return nil
	}

	switch ev.Type {
	case "message_start":
		if ev.Message != nil {
			a.respID = ev.Message.ID
			if ev.Message.Model != "" {
				a.model = ev.Message.Model
			}
			if ev.Message.Usage != nil {
				u := ev.Message.Usage
				a.usage.InputTokens = u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens
				a.usage.CacheCreationTokens = u.CacheCreationInputTokens
				a.usage.CachedInputTokens = u.CacheReadInputTokens
				a.usage.OutputTokens = u.OutputTokens
			}
		}
	case "content_block_start":
		var cb struct {
			Type      string          `json:"type"`
			ID        string          `json:"id"`
			Name      string          `json:"name"`
			Text      string          `json:"text"`
			Thinking  string          `json:"thinking"`
			Signature string          `json:"signature"`
			Data      string          `json:"data"`
			Input     json.RawMessage `json:"input"`
		}
		if json.Unmarshal(ev.ContentBlock, &cb) != nil {
			return nil
		}
		blk := &blockAcc{
			blockType:    cb.Type,
			id:           cb.ID,
			name:         cb.Name,
			initialInput: strings.TrimSpace(string(cb.Input)),
			redactedData: cb.Data,
		}
		if cb.Text != "" {
			blk.textBuf.WriteString(cb.Text)
		}
		if cb.Thinking != "" {
			blk.thinkingBuf.WriteString(cb.Thinking)
		}
		if cb.Signature != "" {
			blk.sigBuf.WriteString(cb.Signature)
		}
		if cb.Type == "web_search_tool_result" || cb.Type == "web_fetch_tool_result" || cb.Type == "tool_search_tool_result" {
			blk.rawServer = append(json.RawMessage(nil), ev.ContentBlock...)
		}
		a.blocks[ev.Index] = blk

		if a.emit != nil {
			switch cb.Type {
			case "tool_use":
				a.emit(provider.StreamChunk{Kind: provider.DeltaToolCallStart, CallID: cb.ID, ToolName: cb.Name})
			case "server_tool_use":
				a.emit(provider.StreamChunk{Kind: provider.DeltaHostedToolStart, CallID: cb.ID, ToolName: cb.Name})
			case "web_search_tool_result", "web_fetch_tool_result":
				a.emit(provider.StreamChunk{Kind: provider.DeltaHostedToolComplete, CallID: cb.ID, ToolName: cb.Type})
			}
		}
	case "content_block_delta":
		blk := a.blocks[ev.Index]
		if blk == nil || ev.Delta == nil {
			return nil
		}
		switch ev.Delta.Type {
		case "text_delta":
			blk.textBuf.WriteString(ev.Delta.Text)
			if a.emit != nil && ev.Delta.Text != "" {
				a.emit(provider.StreamChunk{Kind: provider.DeltaText, Text: ev.Delta.Text})
			}
		case "thinking_delta":
			blk.thinkingBuf.WriteString(ev.Delta.Thinking)
			if a.emit != nil && ev.Delta.Thinking != "" {
				a.emit(provider.StreamChunk{Kind: provider.DeltaThinking, Text: ev.Delta.Thinking})
			}
		case "signature_delta":
			blk.sigBuf.WriteString(ev.Delta.Signature)
		case "input_json_delta":
			blk.inputBuf.WriteString(ev.Delta.PartialJSON)
			if a.emit != nil && blk.blockType == "tool_use" && ev.Delta.PartialJSON != "" {
				a.emit(provider.StreamChunk{Kind: provider.DeltaToolCallArgs, CallID: blk.id, Arguments: ev.Delta.PartialJSON})
			}
		case "citations_delta":
			if ev.Delta.Citation != nil {
				blk.citations = append(blk.citations, schema.Citation{
					Type:      ev.Delta.Citation.Type,
					URL:       ev.Delta.Citation.URL,
					Title:     ev.Delta.Citation.Title,
					CitedText: ev.Delta.Citation.CitedText,
				})
			}
		}
	case "message_delta":
		if ev.Delta != nil && ev.Delta.StopReason != "" {
			a.stopReason = ev.Delta.StopReason
		}
		if ev.Usage != nil {
			if ev.Usage.OutputTokens > 0 {
				a.usage.OutputTokens = ev.Usage.OutputTokens
			}
			if ev.Usage.InputTokens > 0 {
				a.usage.InputTokens = ev.Usage.InputTokens + ev.Usage.CacheCreationInputTokens + ev.Usage.CacheReadInputTokens
			}
			if ev.Usage.CacheReadInputTokens > 0 {
				a.usage.CachedInputTokens = ev.Usage.CacheReadInputTokens
			}
			if ev.Usage.CacheCreationInputTokens > 0 {
				a.usage.CacheCreationTokens = ev.Usage.CacheCreationInputTokens
			}
		}
	case "error":
		if ev.Error != nil {
			return fmt.Errorf("anthropic stream error (%s): %s", ev.Error.Type, ev.Error.Message)
		}
		return errors.New("anthropic stream error")
	}
	return nil
}

func (a *anthropicStreamAcc) finish() provider.Response {
	indices := make([]int, 0, len(a.blocks))
	for idx := range a.blocks {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	var (
		canonical []schema.ContentBlock
		rawNative []json.RawMessage
	)

	for _, idx := range indices {
		blk := a.blocks[idx]
		switch blk.blockType {
		case "thinking":
			thinking := blk.thinkingBuf.String()
			sig := blk.sigBuf.String()
			nativeMap := map[string]any{
				"type":      "thinking",
				"thinking":  thinking,
				"signature": sig,
			}
			raw, _ := json.Marshal(nativeMap)
			rawNative = append(rawNative, raw)
			canonical = append(canonical, schema.ContentBlock{
				Type:      schema.BlockThinking,
				Thinking:  thinking,
				Signature: sig,
			})
		case "redacted_thinking":
			nativeMap := map[string]any{
				"type": "redacted_thinking",
				"data": blk.redactedData,
			}
			raw, _ := json.Marshal(nativeMap)
			rawNative = append(rawNative, raw)
			canonical = append(canonical, schema.ContentBlock{
				Type:         schema.BlockRedactedThinking,
				RedactedData: blk.redactedData,
			})
		case "text":
			text := blk.textBuf.String()
			nativeMap := map[string]any{
				"type": "text",
				"text": text,
			}
			raw, _ := json.Marshal(nativeMap)
			rawNative = append(rawNative, raw)
			canonical = append(canonical, schema.ContentBlock{
				Type:      schema.BlockText,
				Text:      text,
				Citations: blk.citations,
			})
		case "tool_use":
			args := strings.TrimSpace(blk.inputBuf.String())
			if args == "" && blk.initialInput != "" && blk.initialInput != "null" {
				args = blk.initialInput
			}
			if args == "" {
				args = "{}"
			}
			var inputObj any
			if json.Unmarshal([]byte(args), &inputObj) != nil {
				inputObj = map[string]any{}
			}
			nativeMap := map[string]any{
				"type":  "tool_use",
				"id":    blk.id,
				"name":  blk.name,
				"input": inputObj,
			}
			raw, _ := json.Marshal(nativeMap)
			rawNative = append(rawNative, raw)
			canonical = append(canonical, schema.ContentBlock{
				Type:      schema.BlockToolCall,
				CallID:    blk.id,
				Name:      blk.name,
				Arguments: json.RawMessage(args),
			})
		case "server_tool_use":
			args := strings.TrimSpace(blk.inputBuf.String())
			if args == "" && blk.initialInput != "" && blk.initialInput != "null" {
				args = blk.initialInput
			}
			if args == "" {
				args = "{}"
			}
			var inputObj any
			if json.Unmarshal([]byte(args), &inputObj) != nil {
				inputObj = map[string]any{}
			}
			nativeMap := map[string]any{
				"type":  "server_tool_use",
				"id":    blk.id,
				"name":  blk.name,
				"input": inputObj,
			}
			raw, _ := json.Marshal(nativeMap)
			rawNative = append(rawNative, raw)
			canonical = append(canonical, schema.ContentBlock{
				Type:           schema.BlockHostedToolCall,
				CallID:         blk.id,
				Name:           blk.name,
				Arguments:      json.RawMessage(args),
				ProviderOpaque: raw,
			})
		case "web_search_tool_result", "web_fetch_tool_result", "tool_search_tool_result":
			if len(blk.rawServer) > 0 {
				rawNative = append(rawNative, blk.rawServer)
				canonical = append(canonical, schema.ContentBlock{
					Type:           schema.BlockHostedToolResult,
					CallID:         blk.id,
					Name:           blk.blockType,
					ProviderOpaque: blk.rawServer,
				})
			}
		}
	}

	a.usage.TotalTokens = a.usage.InputTokens + a.usage.OutputTokens
	return provider.Response{
		ID:         a.respID,
		Model:      a.model,
		Blocks:     canonical,
		Usage:      a.usage,
		StopReason: a.stopReason,
		Native: &schema.NativeEnvelope{
			Provider:   a.providerID,
			WireFormat: WireFormat,
			Model:      a.model,
			ResponseID: a.respID,
			RawItems:   rawNative,
		},
	}
}

func buildAnthropicMessages(msgs []schema.Message, wireFormat string) []anthropicMessage {
	var out []anthropicMessage
	appendBlocks := func(role string, blocks []map[string]any) {
		if len(blocks) == 0 {
			return
		}
		if len(out) > 0 && out[len(out)-1].Role == role {
			out[len(out)-1].Content = append(out[len(out)-1].Content, blocks...)
			return
		}
		out = append(out, anthropicMessage{Role: role, Content: blocks})
	}

	for _, m := range msgs {
		if m.Native != nil && m.Native.WireFormat == wireFormat && len(m.Native.RawItems) > 0 {
			var nativeBlocks []map[string]any
			for _, raw := range m.Native.RawItems {
				var obj map[string]any
				if json.Unmarshal(raw, &obj) == nil && obj != nil {
					nativeBlocks = append(nativeBlocks, obj)
				}
			}
			role := "assistant"
			if m.Role == schema.RoleUser || m.Role == schema.RoleTool {
				role = "user"
			}
			appendBlocks(role, nativeBlocks)
			continue
		}

		for _, b := range m.Blocks {
			switch b.Type {
			case schema.BlockCompaction:
				if b.Text != "" {
					appendBlocks("user", []map[string]any{{"type": "text", "text": b.Text}})
				}
			case schema.BlockText:
				if b.Text == "" {
					continue
				}
				role := "user"
				if m.Role == schema.RoleAssistant {
					role = "assistant"
				}
				appendBlocks(role, []map[string]any{{"type": "text", "text": b.Text}})
			case schema.BlockImage:
				if b.Image == nil {
					continue
				}
				var src map[string]any
				if b.Image.Base64 != "" {
					mt := b.Image.MediaType
					if mt == "" {
						mt = "image/png"
					}
					src = map[string]any{"type": "base64", "media_type": mt, "data": b.Image.Base64}
				} else if b.Image.URL != "" {
					src = map[string]any{"type": "url", "url": b.Image.URL}
				}
				if src != nil {
					appendBlocks("user", []map[string]any{{"type": "image", "source": src}})
				}
			case schema.BlockThinking:
				// Only replay thinking blocks carrying a genuine Anthropic signature
				// (cross-provider thinking blocks without Anthropic signature are skipped).
				if b.Signature != "" && m.Native != nil && m.Native.WireFormat == wireFormat {
					appendBlocks("assistant", []map[string]any{
						{"type": "thinking", "thinking": b.Thinking, "signature": b.Signature},
					})
				}
			case schema.BlockRedactedThinking:
				if b.RedactedData != "" {
					appendBlocks("assistant", []map[string]any{
						{"type": "redacted_thinking", "data": b.RedactedData},
					})
				}
			case schema.BlockToolCall:
				var inputObj map[string]any
				if json.Unmarshal(b.Arguments, &inputObj) != nil || inputObj == nil {
					inputObj = map[string]any{}
				}
				appendBlocks("assistant", []map[string]any{
					{"type": "tool_use", "id": b.CallID, "name": b.Name, "input": inputObj},
				})
			case schema.BlockToolResult:
				blk := map[string]any{
					"type":        "tool_result",
					"tool_use_id": b.CallID,
					"content":     b.Output,
				}
				if b.IsError {
					blk["is_error"] = true
				}
				appendBlocks("user", []map[string]any{blk})
			}
		}
	}
	return out
}

// applyConversationCacheBreakpoints sets ephemeral cache_control on up to two
// stable user-turn boundaries (penultimate + latest user message) in addition
// to system prompt and last tool (totaling Anthropic's 4 cache breakpoints).
func applyConversationCacheBreakpoints(messages []anthropicMessage) {
	var userIndices []int
	for i, m := range messages {
		if m.Role == "user" && len(m.Content) > 0 {
			userIndices = append(userIndices, i)
		}
	}
	if len(userIndices) >= 2 {
		idx := userIndices[len(userIndices)-2]
		blocks := messages[idx].Content
		blocks[len(blocks)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
	}
	if len(userIndices) >= 1 {
		idx := userIndices[len(userIndices)-1]
		blocks := messages[idx].Content
		blocks[len(blocks)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
	}
}

func buildAnthropicTools(defs []schema.ToolDef, hosted []schema.HostedTool, nativeToolSearch bool) ([]map[string]any, bool, bool) {
	out := make([]map[string]any, 0, len(defs)+len(hosted)+1)
	hasWebFetch := false
	hasDeferred := false

	for _, t := range defs {
		schemaObj := map[string]any{"type": "object"}
		if len(bytes.TrimSpace(t.InputSchema)) > 0 {
			var parsed map[string]any
			if json.Unmarshal(t.InputSchema, &parsed) == nil && parsed != nil {
				if _, ok := parsed["type"]; !ok {
					parsed["type"] = "object"
				}
				schemaObj = parsed
			}
		}
		item := map[string]any{
			"name":         t.Name,
			"input_schema": schemaObj,
		}
		if t.Description != "" {
			item["description"] = t.Description
		}
		if nativeToolSearch && t.DeferLoading && !t.AlwaysLoad {
			item["defer_loading"] = true
			hasDeferred = true
		}
		out = append(out, item)
	}
	if hasDeferred {
		out = append([]map[string]any{{
			"type": "tool_search_tool_bm25_20251119",
			"name": "tool_search_tool_bm25",
		}}, out...)
	}

	for _, h := range hosted {
		switch h.Type {
		case "web_search", "web_search_20250305":
			obj := map[string]any{"type": "web_search_20250305", "name": "web_search"}
			for k, v := range h.Config {
				obj[k] = v
			}
			out = append(out, obj)
		case "web_fetch", "web_fetch_20250910":
			hasWebFetch = true
			obj := map[string]any{"type": "web_fetch_20250910", "name": "web_fetch"}
			for k, v := range h.Config {
				obj[k] = v
			}
			out = append(out, obj)
		default:
			obj := map[string]any{"type": h.Type}
			if h.Name != "" {
				obj["name"] = h.Name
			}
			for k, v := range h.Config {
				obj[k] = v
			}
			out = append(out, obj)
		}
	}
	return out, hasWebFetch, hasDeferred
}

func buildAnthropicToolChoice(tc *schema.ToolChoice) map[string]any {
	if tc == nil || tc.Mode == "" {
		return nil
	}
	switch tc.Mode {
	case "auto":
		return map[string]any{"type": "auto"}
	case "required":
		return map[string]any{"type": "any"}
	case "none":
		return map[string]any{"type": "none"}
	case "tool":
		if tc.Name != "" {
			return map[string]any{"type": "tool", "name": tc.Name}
		}
	}
	return nil
}

func mapThinkingBudget(effort string) int {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "low":
		return 4096
	case "medium":
		return 10000
	case "high":
		return 16000
	case "xhigh", "max":
		return 32000
	default:
		return 0
	}
}
