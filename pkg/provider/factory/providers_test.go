package factory_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/auth"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider/factory"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider/gemini"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
)

func makeBedrockEventFrame(t *testing.T, innerJSON string) []byte {
	t.Helper()
	env, err := json.Marshal(map[string]string{
		"bytes": base64.StdEncoding.EncodeToString([]byte(innerJSON)),
	})
	if err != nil {
		t.Fatal(err)
	}
	totalLen := uint32(12 + len(env) + 4)
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.BigEndian, totalLen)
	_ = binary.Write(&buf, binary.BigEndian, uint32(0)) // headers length
	_ = binary.Write(&buf, binary.BigEndian, uint32(0)) // prelude crc
	buf.Write(env)
	_ = binary.Write(&buf, binary.BigEndian, uint32(0)) // message crc
	return buf.Bytes()
}

func TestOpenAIAndCodexOAuthNativeWireAndCompaction(t *testing.T) {
	var sawCacheKey bool
	var sawCompact bool
	var sawEncryptedReplay bool
	var sawStickyTurnState bool
	var sawSessionID bool

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if r.URL.Path == "/v1/responses/compact" {
			sawCompact = true
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"output": []map[string]any{
					{"type": "compaction", "id": "cmp_1", "encrypted_content": "opaque_cmp_blob"},
				},
				"usage": map[string]any{"input_tokens": 500, "output_tokens": 40},
			})
			return
		}
		if r.Header.Get("session_id") == "thr_1" {
			sawSessionID = true
		}
		if r.Header.Get("x-codex-turn-state") == "sticky_state_xyz" {
			sawStickyTurnState = true
		}
		if strings.Contains(string(body), `"prompt_cache_key":"thr_1"`) {
			sawCacheKey = true
		}
		if strings.Contains(string(body), `"encrypted_content":"enc_rs_1"`) {
			sawEncryptedReplay = true
		}
		w.Header().Set("x-codex-turn-state", "sticky_state_xyz")
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5.4\"}}\n\n")
		fmt.Fprint(w, "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"encrypted_content\":\"enc_rs_1\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"checked\"}]}}\n\n")
		fmt.Fprint(w, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello OpenAI\"}\n\n")
		fmt.Fprint(w, "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello OpenAI\"}]}}\n\n")
		fmt.Fprint(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5.4\",\"status\":\"completed\",\"usage\":{\"input_tokens\":100,\"output_tokens\":20,\"input_tokens_details\":{\"cached_tokens\":80}}}}\n\n")
	}))
	defer srv.Close()

	p, err := factory.New(factory.ProfileConfig{
		Provider:   "openai",
		BaseURL:    srv.URL + "/v1",
		APIKey:     "sk-test",
		HTTPClient: srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}

	resp1, err := p.Stream(context.Background(), provider.Request{
		ThreadID:        "thr_1",
		Model:           "gpt-5.4",
		CacheKey:        "thr_1",
		ReasoningEffort: "high",
		Messages:        []schema.Message{{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "hi"}}}},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !sawCacheKey || !sawSessionID || resp1.Usage.CachedInputTokens != 80 {
		t.Fatalf("expected cache key, session_id, and 80 cached tokens, got %+v", resp1.Usage)
	}

	// Second turn must replay encrypted reasoning verbatim AND echo x-codex-turn-state.
	_, err = p.Stream(context.Background(), provider.Request{
		ThreadID: "thr_1",
		Model:    "gpt-5.4",
		CacheKey: "thr_1",
		Messages: []schema.Message{
			{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "hi"}}},
			{Role: schema.RoleAssistant, Blocks: resp1.Blocks, Native: resp1.Native},
			{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "next"}}},
		},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !sawEncryptedReplay || !sawStickyTurnState {
		t.Fatalf("expected encrypted_content replay (%v) and x-codex-turn-state sticky header (%v)", sawEncryptedReplay, sawStickyTurnState)
	}

	rc, ok := p.(provider.RemoteCompactor)
	if !ok {
		t.Fatal("openai provider must implement RemoteCompactor")
	}
	cmpRes, err := rc.Compact(context.Background(), provider.CompactRequest{
		Model:    "gpt-5.4",
		Messages: []schema.Message{{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "old"}}}},
	})
	if err != nil || !sawCompact || !strings.Contains(string(cmpRes.OpaqueItem), "opaque_cmp_blob") {
		t.Fatalf("remote compact failed: err=%v res=%+v", err, cmpRes)
	}
}

func TestCodexOAuth401RetryAndAccountHeader(t *testing.T) {
	var attempts atomic.Int32
	var sawAccountHeader bool

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/oauth/token" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"access_token":  "fresh_tok",
				"refresh_token": "rt_2",
			})
			return
		}
		if attempts.Add(1) == 1 {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		if r.Header.Get("chatgpt-account-id") == "acct_codex_1" && r.Header.Get("Authorization") == "Bearer fresh_tok" {
			sawAccountHeader = true
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r_codex\",\"model\":\"gpt-6.1-sol\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"OK\"}]}]}}\n\n")
	}))
	defer srv.Close()

	mgr := auth.NewCodexOAuthManager(auth.CodexOAuthConfig{
		Issuer:     srv.URL,
		HTTPClient: srv.Client(),
	})
	mgr.LoadFromMemory(auth.CodexAuthFile{
		Tokens: &auth.CodexTokenSet{
			AccessToken:  "stale_tok",
			RefreshToken: "rt_1",
			AccountID:    "acct_codex_1",
		},
	})

	p, err := factory.New(factory.ProfileConfig{
		Provider:   "codex_oauth",
		BaseURL:    srv.URL,
		CodexOAuth: mgr,
		HTTPClient: srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := p.Stream(context.Background(), provider.Request{
		Model:    "gpt-6.1-sol",
		Messages: []schema.Message{{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "ping"}}}},
	}, nil)
	if err != nil {
		t.Fatalf("codex oauth stream failed: %v", err)
	}
	if !sawAccountHeader || len(res.Blocks) == 0 || res.Blocks[0].Text != "OK" {
		t.Fatalf("unexpected codex oauth response: sawAcct=%v blocks=%+v", sawAccountHeader, res.Blocks)
	}
}

func TestAnthropicDirectThinkingSignaturesCacheAndNativeDeferredTools(t *testing.T) {
	var reqCount atomic.Int32
	var secondReqBody string
	var sawBetaHeader string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		sawBetaHeader = r.Header.Get("anthropic-beta")
		n := reqCount.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		if n == 1 {
			fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_a1\",\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":50,\"cache_read_input_tokens\":400}}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Need weather.\"}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"SIG_CRYPTO_123\"}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"srvtoolu_1\",\"name\":\"web_search\"}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"web_search_tool_result\",\"tool_use_id\":\"srvtoolu_1\",\"content\":[{\"type\":\"web_search_result\",\"url\":\"https://example.com\"}]}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":3,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\"}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"index\":3,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"London\\\"}\"}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":35}}\n\n")
			return
		}
		secondReqBody = string(body)
		fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_a2\",\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":500}}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Rainy in London\"}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":12}}\n\n")
	}))
	defer srv.Close()

	p, err := factory.New(factory.ProfileConfig{
		Provider:   "anthropic",
		BaseURL:    srv.URL,
		APIKey:     "test-anthropic-key",
		HTTPClient: srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}

	req1 := provider.Request{
		Model:             "claude-sonnet-4-5",
		SystemPrompt:      "Be concise.",
		ReasoningEffort:   "high",
		EnableNativeCache: true,
		NativeToolSearch:  true,
		HostedTools:       []schema.HostedTool{{Type: "web_search"}, {Type: "web_fetch"}},
		Tools: []schema.ToolDef{
			{Name: "get_weather", Description: "Weather lookup", DeferLoading: true},
		},
		Messages: []schema.Message{
			{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Weather?"}}},
		},
	}
	res1, err := p.Stream(context.Background(), req1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(sawBetaHeader, "web-fetch-2025-09-10") || !strings.Contains(sawBetaHeader, "advanced-tool-use-2025-11-20") {
		t.Fatalf("missing expected anthropic beta headers: %q", sawBetaHeader)
	}

	// Tool continuation MUST keep thinking enabled AND replay signed thinking + server_tool_use + web_search_tool_result verbatim!
	req2 := req1
	req2.Messages = append(req2.Messages,
		schema.Message{Role: schema.RoleAssistant, Blocks: res1.Blocks, Native: res1.Native},
		schema.Message{Role: schema.RoleTool, Blocks: []schema.ContentBlock{{Type: schema.BlockToolResult, CallID: "toolu_1", Output: "12C rain"}}},
	)
	res2, err := p.Stream(context.Background(), req2, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`"thinking":{"budget_tokens":16000,"type":"enabled"}`,
		`"signature":"SIG_CRYPTO_123"`,
		`"type":"server_tool_use"`,
		`"type":"web_search_tool_result"`,
		`"tool_search_tool_bm25_20251119"`,
		`"defer_loading":true`,
		`"cache_control":{"type":"ephemeral"}`,
	} {
		if !strings.Contains(secondReqBody, want) {
			t.Fatalf("second Anthropic request missing %s\nbody=%s", want, secondReqBody)
		}
	}
	if res2.Usage.CachedInputTokens != 500 {
		t.Fatalf("expected 500 cached tokens, got %+v", res2.Usage)
	}
	if got := strings.Count(secondReqBody, `"cache_control":{"type":"ephemeral"}`); got > 4 {
		t.Fatalf("anthropic cache_control breakpoints must never exceed 4, got %d", got)
	}
}

func TestAnthropicRelayBearerAndNoRetryHeader(t *testing.T) {
	var attempts atomic.Int32
	var sawBearer string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		sawBearer = r.Header.Get("Authorization")
		if r.Header.Get("x-api-key") != "" {
			t.Errorf("relay bearer mode must not send x-api-key")
		}
		w.Header().Set("x-should-retry", "false")
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"error":{"type":"rate_limit_error","message":"stop retrying"}}`)
	}))
	defer srv.Close()

	p, err := factory.New(factory.ProfileConfig{
		Provider:   "anthropic",
		BaseURL:    srv.URL,
		APIKey:     "turn_llm_tok_1",
		RelayMode:  true,
		HTTPClient: srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Stream(context.Background(), provider.Request{
		Model: "claude-sonnet-4-5",
		Messages: []schema.Message{
			{Role: schema.RoleAssistant, Blocks: []schema.ContentBlock{{Type: schema.BlockToolCall, CallID: "t1", Name: "fn", Arguments: json.RawMessage(`{malformed`)}}},
			{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "hi"}}},
		},
	}, nil)
	if err == nil {
		t.Fatal("expected 429 error")
	}
	if attempts.Load() != 1 {
		t.Fatalf("x-should-retry: false must prevent retries, got %d attempts", attempts.Load())
	}
	if sawBearer != "Bearer turn_llm_tok_1" {
		t.Fatalf("expected relay Bearer auth, got %q", sawBearer)
	}
}

func TestAnthropicBedrockAndVertexModes(t *testing.T) {
	var sawSigV4 bool
	bedrockSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.Header.Get("Authorization"), "AWS4-HMAC-SHA256 ") {
			sawSigV4 = true
		}
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		_, _ = w.Write(makeBedrockEventFrame(t, `{"type":"message_start","message":{"id":"b1","model":"claude-sonnet-4-5","usage":{"input_tokens":20}}}`))
		_, _ = w.Write(makeBedrockEventFrame(t, `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`))
		_, _ = w.Write(makeBedrockEventFrame(t, `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Bedrock OK"}}`))
		_, _ = w.Write(makeBedrockEventFrame(t, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`))
	}))
	defer bedrockSrv.Close()

	bp, err := factory.New(factory.ProfileConfig{
		Provider:   "bedrock_anthropic",
		BaseURL:    bedrockSrv.URL,
		AWS:        auth.AWSCredentials{AccessKeyID: "TESTKEYID", SecretAccessKey: "TESTSECRET", Region: "us-east-1"},
		HTTPClient: bedrockSrv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	bres, err := bp.Stream(context.Background(), provider.Request{
		Model:    "us.anthropic.claude-sonnet-4-5-v1:0",
		Messages: []schema.Message{{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "hi"}}}},
	}, nil)
	if err != nil || !sawSigV4 || len(bres.Blocks) == 0 || bres.Blocks[0].Text != "Bedrock OK" {
		t.Fatalf("bedrock anthropic failed: err=%v sigv4=%v blocks=%+v", err, sawSigV4, bres.Blocks)
	}

	var sawVertexPath bool
	vertexSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, ":streamRawPredict") && r.Header.Get("Authorization") == "Bearer gcp-tok" {
			sawVertexPath = true
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"Vertex Claude OK\"}}\n\n")
	}))
	defer vertexSrv.Close()

	vp, err := factory.New(factory.ProfileConfig{
		Provider:       "vertex_anthropic",
		BaseURL:        vertexSrv.URL,
		VertexProject:  "proj-1",
		VertexLocation: "us-east5",
		GCPTokenSource: func(_ context.Context) (string, error) { return "gcp-tok", nil },
		HTTPClient:     vertexSrv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	vres, err := vp.Stream(context.Background(), provider.Request{
		Model:    "claude-sonnet-4-5",
		Messages: []schema.Message{{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "hi"}}}},
	}, nil)
	if err != nil || !sawVertexPath || len(vres.Blocks) == 0 || vres.Blocks[0].Text != "Vertex Claude OK" {
		t.Fatalf("vertex anthropic failed: err=%v path=%v blocks=%+v", err, sawVertexPath, vres.Blocks)
	}
}

func TestGrokNativeXSearchAndConvID(t *testing.T) {
	var sawConvID string
	var sawXSearch bool

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		sawConvID = r.Header.Get("x-grok-conv-id")
		if strings.Contains(string(body), `"type":"x_search"`) {
			sawXSearch = true
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"ctc_1\",\"type\":\"custom_tool_call\",\"call_id\":\"xs_1\",\"name\":\"x_keyword_search\",\"input\":\"{\\\"query\\\":\\\"jaz\\\"}\"}}\n\n")
		fmt.Fprint(w, "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"msg_g1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Found posts\",\"annotations\":[{\"type\":\"url_citation\",\"url\":\"https://x.com/status/1\",\"title\":\"Post\"}]}]}}\n\n")
		fmt.Fprint(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"rg_1\",\"model\":\"grok-4\",\"status\":\"completed\"}}\n\n")
	}))
	defer srv.Close()

	p, err := factory.New(factory.ProfileConfig{
		Provider:   "grok",
		BaseURL:    srv.URL,
		APIKey:     "xai-test",
		HTTPClient: srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := p.Stream(context.Background(), provider.Request{
		ThreadID:    "thr_grok_1",
		CacheKey:    "thr_grok_1",
		Model:       "grok-4",
		HostedTools: []schema.HostedTool{{Type: "x_search"}, {Type: "web_search"}},
		Messages:    []schema.Message{{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "latest?"}}}},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if sawConvID != "thr_grok_1" || !sawXSearch {
		t.Fatalf("grok headers/tools mismatch: convID=%q xSearch=%v", sawConvID, sawXSearch)
	}
	for _, b := range res.Blocks {
		if b.Type == schema.BlockToolCall {
			t.Fatalf("xs_* hosted tool must NOT be emitted as local BlockToolCall: %+v", b)
		}
	}
	if len(res.Blocks) < 2 || len(res.Blocks[1].Citations) != 1 || res.Blocks[1].Citations[0].URL != "https://x.com/status/1" {
		t.Fatalf("expected hosted block + cited text block, got %+v", res.Blocks)
	}
}

func TestGemini4VertexV1Beta1CombinedToolsSignaturesAndRelay(t *testing.T) {
	var reqNum atomic.Int32
	var firstPath string
	var firstBody string
	var secondBody string
	var thirdBody string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		n := reqNum.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		switch n {
		case 1:
			firstPath = r.URL.Path
			firstBody = string(body)
			fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"thought\":true,\"text\":\"Thinking step\",\"thoughtSignature\":\"GEM_SIG_999\"},{\"functionCall\":{\"id\":\"fc_1\",\"name\":\"lookup\",\"args\":{\"q\":\"price\"}},\"thoughtSignature\":\"GEM_SIG_FC_1\"}]}}],\"usageMetadata\":{\"promptTokenCount\":200,\"toolUsePromptTokenCount\":40,\"cachedContentTokenCount\":150,\"candidatesTokenCount\":30}}\n\n")
		case 2:
			secondBody = string(body)
			fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"$42\",\"thoughtSignature\":\"GEM_SIG_TXT\"}]},\"finishReason\":\"STOP\",\"groundingMetadata\":{\"groundingChunks\":[{\"web\":{\"uri\":\"https://docs.example.com\",\"title\":\"Pricing\"}}]},\"urlContextMetadata\":{\"urlMetadata\":[{\"retrievedUrl\":\"https://status.example.com\",\"urlRetrievalStatus\":\"URL_RETRIEVAL_STATUS_SUCCESS\"}]}}],\"usageMetadata\":{\"promptTokenCount\":260,\"cachedContentTokenCount\":200,\"candidatesTokenCount\":10}}\n\n")
		default:
			thirdBody = string(body)
			fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"done\"}]}}]}\n\n")
		}
	}))
	defer srv.Close()

	// Strip legacy /endpoints/openapi suffix automatically and route to Vertex v1beta1.
	p, err := factory.New(factory.ProfileConfig{
		Provider:       "vertex",
		BaseURL:        srv.URL + "/endpoints/openapi",
		VertexProject:  "proj-1",
		VertexLocation: "global",
		GCPTokenSource: func(_ context.Context) (string, error) { return "gcp-tok", nil },
		HTTPClient:     srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	temp := 0.7
	topP := 0.9
	res1, err := p.Stream(context.Background(), provider.Request{
		Model:             "google/gemini-4-neon",
		ReasoningEffort:   "xhigh",
		Temperature:       &temp,
		TopP:              &topP,
		CachedContentName: "projects/proj-1/locations/global/cachedContents/cc_1",
		HostedTools:       []schema.HostedTool{{Type: "google_search"}, {Type: "url_context"}},
		Tools:             []schema.ToolDef{{Name: "lookup", Description: "Lookup"}},
		Messages:          []schema.Message{{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "price?"}}}},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(firstPath, "/v1beta1/projects/proj-1/locations/global/publishers/google/models/gemini-4-neon:streamGenerateContent") {
		t.Fatalf("expected Vertex v1beta1 path without /endpoints/openapi, got %q", firstPath)
	}
	for _, want := range []string{
		`"thinkingLevel":"XHIGH"`,
		`"includeServerSideToolInvocations":true`,
		`"googleSearch":{}`,
		`"urlContext":{}`,
		`"functionDeclarations"`,
	} {
		if !strings.Contains(firstBody, want) {
			t.Fatalf("first Gemini 4 request missing %s\nbody=%s", want, firstBody)
		}
	}
	for _, forbid := range []string{`"temperature"`, `"topP"`, `"cachedContent"`} {
		if strings.Contains(firstBody, forbid) {
			t.Fatalf("Gemini 4 request must omit %s\nbody=%s", forbid, firstBody)
		}
	}
	if res1.Usage.InputTokens != 240 {
		t.Fatalf("expected promptTokenCount(200)+toolUsePromptTokenCount(40)=240, got %+v", res1.Usage)
	}

	res2, err := p.Stream(context.Background(), provider.Request{
		Model:           "google/gemini-4-neon",
		ReasoningEffort: "minimal", // must clamp to LOW on Gemini 4 (never MINIMAL)
		HostedTools:     []schema.HostedTool{{Type: "google_search"}, {Type: "url_context"}},
		Tools:           []schema.ToolDef{{Name: "lookup", Description: "Lookup"}},
		Messages: []schema.Message{
			{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "price?"}}},
			{Role: schema.RoleAssistant, Blocks: res1.Blocks, Native: res1.Native},
			{Role: schema.RoleTool, Blocks: []schema.ContentBlock{{Type: schema.BlockToolResult, CallID: "fc_1", Name: "lookup", Output: "42"}}},
		},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`"thinkingLevel":"LOW"`,
		`"thoughtSignature":"GEM_SIG_999"`,
		`"thoughtSignature":"GEM_SIG_FC_1"`,
	} {
		if !strings.Contains(secondBody, want) {
			t.Fatalf("second Gemini 4 request missing %s\nbody=%s", want, secondBody)
		}
	}
	if strings.Contains(secondBody, `"MINIMAL"`) {
		t.Fatalf("Gemini 4 must never emit MINIMAL thinkingLevel: %s", secondBody)
	}
	if len(res2.Blocks) == 0 || len(res2.Blocks[0].Citations) != 2 || res2.Blocks[0].ThoughtSignBlob != "GEM_SIG_TXT" || res2.Usage.CachedInputTokens != 200 {
		t.Fatalf("unexpected Gemini 4 response (grounding+urlContext+text sig): blocks=%+v usage=%+v", res2.Blocks, res2.Usage)
	}

	// Canonical projection of parallel functionCalls must attach thoughtSignature ONLY to the first functionCall,
	// while explicit cachedContent remains allowed on Gemini 2.5.
	_, err = p.Stream(context.Background(), provider.Request{
		Model:             "gemini-2.5-pro",
		CachedContentName: "projects/proj-1/locations/us-central1/cachedContents/cc_25",
		Messages: []schema.Message{
			{
				Role: schema.RoleAssistant,
				Blocks: []schema.ContentBlock{
					{Type: schema.BlockToolCall, CallID: "c1", Name: "fn_a", Arguments: json.RawMessage(`{"a":1}`), ThoughtSignBlob: "SIG_PARALLEL"},
					{Type: schema.BlockToolCall, CallID: "c2", Name: "fn_b", Arguments: json.RawMessage(`{"b":2}`), ThoughtSignBlob: "SIG_PARALLEL"},
				},
			},
		},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(thirdBody, `"thoughtSignature":"SIG_PARALLEL"`); got != 1 {
		t.Fatalf("parallel functionCalls must emit thoughtSignature only on first call, got %d in %s", got, thirdBody)
	}
	if !strings.Contains(thirdBody, `"cachedContent":"projects/proj-1/locations/us-central1/cachedContents/cc_25"`) {
		t.Fatalf("Gemini 2.5 should preserve explicit cachedContent: %s", thirdBody)
	}
}

func TestGeminiInteractionsStatelessAPIAndRelay(t *testing.T) {
	var sawPath string
	var sawAuth string
	var reqBody string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		sawPath = r.URL.Path
		sawAuth = r.Header.Get("Authorization")
		reqBody = string(body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"event_type\":\"content.delta\",\"delta\":{\"type\":\"thought\",\"thought\":\"Checking\"}}\n\n")
		fmt.Fprint(w, "data: {\"event_type\":\"content.delta\",\"delta\":{\"type\":\"google_search_call\",\"id\":\"gs_1\"}}\n\n")
		fmt.Fprint(w, "data: {\"event_type\":\"content.delta\",\"delta\":{\"type\":\"text\",\"text\":\"Part 1\"}}\n\n")
		fmt.Fprint(w, "data: {\"event_type\":\"interaction.complete\",\"interaction\":{\"id\":\"int_1\",\"model\":\"gemini-4-neon\",\"status\":\"requires_continuation\",\"continuation_token\":\"ctok_abc\",\"outputs\":[{\"type\":\"text\",\"text\":\"Part 1\",\"signature\":\"INT_SIG_1\"}],\"usage\":{\"total_input_tokens\":100,\"total_cached_tokens\":80,\"total_output_tokens\":25,\"total_thought_tokens\":15}}}\n\n")
	}))
	defer srv.Close()

	p, err := factory.New(factory.ProfileConfig{
		Provider:      "vertex",
		GeminiSurface: gemini.SurfaceInteractions,
		BaseURL:       srv.URL,
		APIKey:        "relay_tok_9",
		RelayMode:     true,
		HTTPClient:    srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := p.Stream(context.Background(), provider.Request{
		Model:           "google/gemini-4-neon",
		ReasoningEffort: "xhigh",
		HostedTools:     []schema.HostedTool{{Type: "google_search"}, {Type: "url_context"}},
		Tools:           []schema.ToolDef{{Name: "lookup", Description: "Lookup"}},
		Messages:        []schema.Message{{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "research"}}}},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if sawPath != "/interactions" || sawAuth != "Bearer relay_tok_9" {
		t.Fatalf("unexpected relay interactions path=%q auth=%q", sawPath, sawAuth)
	}
	for _, want := range []string{
		`"store":false`,
		`"stream":true`,
		`"type":"google_search"`,
		`"type":"url_context"`,
		`"type":"function"`,
		`"thinking_level":"xhigh"`,
	} {
		if !strings.Contains(reqBody, want) {
			t.Fatalf("interactions request missing %s\nbody=%s", want, reqBody)
		}
	}
	if res.ContinuationToken != "ctok_abc" || res.Usage.CachedInputTokens != 80 || res.Usage.ReasoningTokens != 15 {
		t.Fatalf("unexpected interactions response: token=%q usage=%+v", res.ContinuationToken, res.Usage)
	}
}

func TestMetaAndOpenRouterAndBedrockOpenAICompat(t *testing.T) {
	var lastReq string
	var lastReferer string
	var lastAuth string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		lastReq = string(body)
		lastReferer = r.Header.Get("HTTP-Referer")
		lastAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"cmpl_1\",\"model\":\"meta-llama/llama-4-maverick\",\"choices\":[{\"delta\":{\"reasoning\":\"step\",\"content\":\"Llama OK\"}}],\"usage\":{\"prompt_tokens\":30,\"completion_tokens\":8,\"prompt_tokens_details\":{\"cached_tokens\":10},\"cost\":0.0004}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	// Meta normalizes forced tool_choice -> "auto".
	mp, err := factory.New(factory.ProfileConfig{
		Provider:   "meta",
		BaseURL:    srv.URL,
		APIKey:     "llama-key",
		HTTPClient: srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = mp.Stream(context.Background(), provider.Request{
		Model:      "Llama-4-Maverick-17B-128E-Instruct-FP8",
		Tools:      []schema.ToolDef{{Name: "ping"}},
		ToolChoice: &schema.ToolChoice{Mode: "required"},
		Messages:   []schema.Message{{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "hi"}}}},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(lastReq, `"tool_choice":"auto"`) {
		t.Fatalf("Meta should normalize tool_choice to auto: %s", lastReq)
	}

	// OpenRouter forwards referer + reasoning effort + captures cost & cached tokens.
	orp, err := factory.New(factory.ProfileConfig{
		Provider:    "openrouter",
		BaseURL:     srv.URL,
		APIKey:      "or-key",
		HTTPReferer: "https://jaz.chat",
		AppTitle:    "jaz-harness",
		HTTPClient:  srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	ores, err := orp.Stream(context.Background(), provider.Request{
		Model:           "anthropic/claude-sonnet-4-5",
		ReasoningEffort: "medium",
		Messages:        []schema.Message{{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "hi"}}}},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if lastReferer != "https://jaz.chat" || ores.Usage.CachedInputTokens != 10 || ores.Usage.CostUSD <= 0 {
		t.Fatalf("openrouter mismatch: ref=%q usage=%+v", lastReferer, ores.Usage)
	}

	// Bedrock OpenAI uses AWS SigV4 when AWS creds are provided.
	bop, err := factory.New(factory.ProfileConfig{
		Provider:   "bedrock_openai",
		BaseURL:    srv.URL,
		AWS:        auth.AWSCredentials{AccessKeyID: "TESTKEYID", SecretAccessKey: "TESTSECRET", Region: "us-west-2"},
		HTTPClient: srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = bop.Stream(context.Background(), provider.Request{
		Model:    "openai.gpt-oss-120b",
		Messages: []schema.Message{{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "hi"}}}},
	}, nil)
	if err != nil || !strings.HasPrefix(lastAuth, "AWS4-HMAC-SHA256 ") {
		t.Fatalf("bedrock_openai failed: err=%v auth=%q", err, lastAuth)
	}
	_ = time.Now()
}
