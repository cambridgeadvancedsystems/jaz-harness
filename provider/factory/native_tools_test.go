package factory_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cambridgeadvancedsystems/jaz-harness/auth"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider/factory"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
)

func tinyPNGBytes(t *testing.T, c color.RGBA) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	for y := range 2 {
		for x := range 2 {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func tinyPNGBase64(t *testing.T, c color.RGBA) string {
	t.Helper()
	return base64.StdEncoding.EncodeToString(tinyPNGBytes(t, c))
}

func TestGrokNativeXSearchCitationsMultimodalToolOutputAndImageGen(t *testing.T) {
	var streamReqBody string
	var imgGenReqBody string
	var imgEditReqBody string

	redB64 := tinyPNGBase64(t, color.RGBA{R: 255, A: 255})
	blueB64 := tinyPNGBase64(t, color.RGBA{B: 255, A: 255})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/images/generations":
			imgGenReqBody = string(body)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"b64_json": blueB64, "revised_prompt": "A blue square"}},
			})
			return
		case "/images/edits":
			imgEditReqBody = string(body)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"b64_json": blueB64}},
			})
			return
		}

		streamReqBody = string(body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"xs_item_1\",\"type\":\"x_search_call\",\"call_id\":\"xs_1\",\"name\":\"x_keyword_search\"}}\n\n")
		fmt.Fprint(w, "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"xs_item_1\",\"type\":\"x_search_call\",\"call_id\":\"xs_1\",\"name\":\"x_keyword_search\",\"action\":{\"type\":\"search\",\"query\":\"golang release\"}}}\n\n")
		fmt.Fprint(w, "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Verified red image and X posts.\"}]}}\n\n")
		fmt.Fprint(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"rg_2\",\"model\":\"grok-4\",\"status\":\"completed\",\"citations\":[\"https://x.com/golang/status/1\",\"https://go.dev/blog\"]}}\n\n")
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

	var hostedStarts, hostedCompletes int
	res, err := p.Stream(context.Background(), provider.Request{
		ThreadID: "thr_grok_mm",
		Model:    "grok-4",
		HostedTools: []schema.HostedTool{
			{Type: "x_search", Config: map[string]any{"allowed_x_handles": []string{"golang"}, "from_date": "2026-01-01", "enable_image_understanding": true}},
			{Type: "web_search", Config: map[string]any{"allowed_domains": []string{"go.dev"}, "external_web_access": true}},
			{Type: "image_generation"}, // OpenAI-only hosted tool must be stripped on Grok
		},
		Messages: []schema.Message{
			{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Inspect image and search X"}}},
			{Role: schema.RoleAssistant, Blocks: []schema.ContentBlock{{Type: schema.BlockToolCall, CallID: "c_img", Name: "view_image", Arguments: json.RawMessage(`{"path":"red.png"}`)}}},
			{Role: schema.RoleTool, Blocks: []schema.ContentBlock{{
				Type:   schema.BlockToolResult,
				CallID: "c_img",
				Name:   "view_image",
				Output: "Loaded image red.png",
				Images: []schema.ImageSource{{MediaType: "image/png", Base64: redB64, Detail: "original"}},
			}}},
		},
	}, func(ch provider.StreamChunk) {
		if ch.Kind == provider.DeltaHostedToolStart {
			hostedStarts++
		}
		if ch.Kind == provider.DeltaHostedToolComplete {
			hostedCompletes++
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`"type":"x_search"`,
		`"allowed_x_handles":["golang"]`,
		`"allowed_domains":["go.dev"]`,
		`"type":"input_image"`,
		`"detail":"high"`, // Grok clamps "original" -> "high"
	} {
		if !strings.Contains(streamReqBody, want) {
			t.Fatalf("grok request missing %s\nbody=%s", want, streamReqBody)
		}
	}
	for _, forbid := range []string{`"external_web_access"`, `"image_generation"`, `"detail":"original"`} {
		if strings.Contains(streamReqBody, forbid) {
			t.Fatalf("grok request must strip %s\nbody=%s", forbid, streamReqBody)
		}
	}
	if hostedStarts != 1 || hostedCompletes != 1 {
		t.Fatalf("expected 1 hosted start + 1 complete, got start=%d complete=%d", hostedStarts, hostedCompletes)
	}
	if len(res.Blocks) < 2 || len(res.Blocks[1].Citations) != 2 {
		t.Fatalf("expected 2 merged top-level citations on Grok text block, got %+v", res.Blocks)
	}

	gen, ok := p.(provider.ImageGenerator)
	if !ok {
		t.Fatal("grok client must implement provider.ImageGenerator")
	}
	imgResp, err := gen.GenerateImage(context.Background(), provider.ImageGenerationRequest{
		Prompt: "A blue square",
	})
	if err != nil || len(imgResp.Images) != 1 || imgResp.Images[0].Base64 != blueB64 || !strings.Contains(imgGenReqBody, `"response_format":"b64_json"`) {
		t.Fatalf("grok GenerateImage failed: err=%v resp=%+v req=%s", err, imgResp, imgGenReqBody)
	}
	editResp, err := gen.GenerateImage(context.Background(), provider.ImageGenerationRequest{
		Prompt:          "Turn red into blue",
		ReferenceImages: []schema.ImageSource{{MediaType: "image/png", Base64: redB64}},
	})
	if err != nil || len(editResp.Images) != 1 || !strings.Contains(imgEditReqBody, `"image":{"url":"data:image/png;base64,`) {
		t.Fatalf("grok edit failed: err=%v resp=%+v req=%s", err, editResp, imgEditReqBody)
	}
}

func TestOpenAIAndCodexImageGenerationCallContinuityAndImagesAPI(t *testing.T) {
	redB64 := tinyPNGBase64(t, color.RGBA{R: 255, A: 255})
	greenB64 := tinyPNGBase64(t, color.RGBA{G: 255, A: 255})

	var reqCount atomic.Int32
	var req1Body string
	var req2Body string
	var req3Body string
	var directEditContentType string
	var codexEditBody string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/images/edits":
			directEditContentType = r.Header.Get("Content-Type")
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"b64_json": greenB64, "revised_prompt": "Edited green"}},
			})
			return
		case "/codex/images/edits":
			codexEditBody = string(body)
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("x-codex-imagegen-request-id", "ig_req_9")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"b64_json": greenB64, "generation_id": "gen_9"}},
			})
			return
		}

		n := reqCount.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		switch n {
		case 1:
			req1Body = string(body)
			fmt.Fprint(w, "event: response.image_generation_call.in_progress\ndata: {\"type\":\"response.image_generation_call.in_progress\",\"item_id\":\"ig_123\"}\n\n")
			fmt.Fprintf(w, "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"ig_123\",\"type\":\"image_generation_call\",\"status\":\"completed\",\"revised_prompt\":\"A bright green icon\",\"result\":\"%s\"}}\n\n", greenB64)
			fmt.Fprint(w, "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Created the green icon.\"}]}}\n\n")
			fmt.Fprint(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r_ig_1\",\"model\":\"gpt-5.4\",\"status\":\"completed\"}}\n\n")
		case 2:
			req2Body = string(body)
			fmt.Fprint(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r_ig_2\",\"model\":\"gpt-5.4\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Edited.\"}]}]}}\n\n")
		default:
			req3Body = string(body)
			fmt.Fprint(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r_ig_3\",\"model\":\"gpt-5.4\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Canonical replayed.\"}]}]}}\n\n")
		}
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

	var starts, completes int
	res1, err := p.Stream(context.Background(), provider.Request{
		Model: "gpt-5.4",
		HostedTools: []schema.HostedTool{
			{Type: "web_search", Config: map[string]any{"external_web_access": true}},
			{Type: "image_generation", Config: map[string]any{"input_fidelity": "high", "partial_images": 1}},
			{Type: "x_search"}, // xAI-only must be stripped on OpenAI
		},
		Messages: []schema.Message{
			{Role: schema.RoleUser, Blocks: []schema.ContentBlock{
				{Type: schema.BlockText, Text: "Use this reference image to create a green icon."},
				{Type: schema.BlockImage, Image: &schema.ImageSource{MediaType: "image/png", Base64: redB64, Detail: "original"}},
			}},
		},
	}, func(ch provider.StreamChunk) {
		if ch.Kind == provider.DeltaHostedToolStart && ch.ToolName == "image_generation" {
			starts++
		}
		if ch.Kind == provider.DeltaHostedToolComplete && ch.ToolName == "image_generation" {
			completes++
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"type":"web_search"`, `"type":"image_generation"`, `"input_fidelity":"high"`, `"detail":"original"`} {
		if !strings.Contains(req1Body, want) {
			t.Fatalf("openai req1 missing %s\nbody=%s", want, req1Body)
		}
	}
	if strings.Contains(req1Body, `"x_search"`) {
		t.Fatalf("openai must strip x_search: %s", req1Body)
	}
	if starts != 1 || completes != 1 {
		t.Fatalf("expected 1 image_generation start+complete, got starts=%d completes=%d", starts, completes)
	}
	if len(res1.Blocks) < 2 || res1.Blocks[0].Type != schema.BlockImage || res1.Blocks[0].Image == nil || res1.Blocks[0].Image.Base64 != greenB64 || res1.Blocks[0].Text != "A bright green icon" {
		t.Fatalf("expected BlockImage decoded from image_generation_call, got %+v", res1.Blocks)
	}

	// Turn 2 (native replay) + multimodal view_image tool output: replays image_generation_call verbatim!
	_, err = p.Stream(context.Background(), provider.Request{
		Model: "gpt-5.4",
		Messages: []schema.Message{
			{Role: schema.RoleAssistant, Blocks: res1.Blocks, Native: res1.Native},
			{Role: schema.RoleTool, Blocks: []schema.ContentBlock{{
				Type:   schema.BlockToolResult,
				CallID: "call_vi",
				Name:   "view_image",
				Output: "Loaded ref.png",
				Images: []schema.ImageSource{{MediaType: "image/png", Base64: redB64, Detail: "original"}},
			}}},
		},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"type":"image_generation_call"`, `"id":"ig_123"`, `"type":"function_call_output"`, `"type":"input_image"`, `"detail":"original"`} {
		if !strings.Contains(req2Body, want) {
			t.Fatalf("openai turn 2 missing %s\nbody=%s", want, req2Body)
		}
	}

	// Turn 3 (canonical replay without NativeEnvelope): still replays image_generation_call via ProviderOpaque!
	_, err = p.Stream(context.Background(), provider.Request{
		Model: "gpt-5.4",
		Messages: []schema.Message{
			{Role: schema.RoleAssistant, Blocks: res1.Blocks},
		},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(req3Body, `"type":"image_generation_call"`) || !strings.Contains(req3Body, `"id":"ig_123"`) {
		t.Fatalf("canonical replay must preserve image_generation_call continuity: %s", req3Body)
	}

	// Direct OpenAI GenerateImage with reference images uses multipart/form-data on /v1/images/edits.
	imgGen := p.(provider.ImageGenerator)
	editRes, err := imgGen.GenerateImage(context.Background(), provider.ImageGenerationRequest{
		Prompt:          "Make it green",
		ReferenceImages: []schema.ImageSource{{MediaType: "image/png", Base64: redB64}},
	})
	if err != nil || len(editRes.Images) != 1 || !strings.HasPrefix(directEditContentType, "multipart/form-data;") {
		t.Fatalf("direct openai multipart edit failed: err=%v ct=%q res=%+v", err, directEditContentType, editRes)
	}

	// Codex OAuth GenerateImage with reference images uses Codex JSON ImageEditRequest on /images/edits.
	mgr := auth.NewCodexOAuthManager(auth.CodexOAuthConfig{HTTPClient: srv.Client()})
	mgr.LoadFromMemory(auth.CodexAuthFile{
		Tokens: &auth.CodexTokenSet{AccessToken: "codex_tok", AccountID: "acct_1"},
	})
	cp, err := factory.New(factory.ProfileConfig{
		Provider:   "codex_oauth",
		BaseURL:    srv.URL + "/codex",
		CodexOAuth: mgr,
		HTTPClient: srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	codexRes, err := cp.(provider.ImageGenerator).GenerateImage(context.Background(), provider.ImageGenerationRequest{
		Prompt:                "Transparent green cutout",
		TransparentBackground: true,
		ReferenceImages:       []schema.ImageSource{{MediaType: "image/png", Base64: redB64}},
	})
	if err != nil || codexRes.ID != "gen_9" || codexRes.Metadata["imagegen_request_id"] != "ig_req_9" {
		t.Fatalf("codex image edit failed: err=%v res=%+v", err, codexRes)
	}
	if !strings.Contains(codexEditBody, `"images":[{"image_url":"data:image/png;base64,`) || !strings.Contains(codexEditBody, `"background":"transparent"`) {
		t.Fatalf("unexpected codex edit payload: %s", codexEditBody)
	}
}

func TestAnthropicWebSearchFetchToolUseIDAndMultimodalToolResult(t *testing.T) {
	redB64 := tinyPNGBase64(t, color.RGBA{R: 255, A: 255})
	var reqBody string
	var betaHeader string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		reqBody = string(body)
		betaHeader = r.Header.Get("anthropic-beta")
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_wf\",\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":40}}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"srvtoolu_wf1\",\"name\":\"web_fetch\"}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"web_fetch_tool_result\",\"tool_use_id\":\"srvtoolu_wf1\",\"content\":{\"type\":\"web_fetch_result\",\"url\":\"https://example.com\"}}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"text\",\"text\":\"Saw red image and fetched page.\"}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":10}}\n\n")
	}))
	defer srv.Close()

	p, err := factory.New(factory.ProfileConfig{
		Provider:   "anthropic",
		BaseURL:    srv.URL,
		APIKey:     "test-key",
		HTTPClient: srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}

	var completedCallID string
	res, err := p.Stream(context.Background(), provider.Request{
		Model: "claude-sonnet-4-5",
		HostedTools: []schema.HostedTool{
			{Type: "web_search_20260209"},
			{Type: "web_fetch"},
			{Type: "x_search"},         // foreign: must be skipped
			{Type: "image_generation"}, // foreign: must be skipped
		},
		Messages: []schema.Message{
			{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Inspect tool image"}}},
			{Role: schema.RoleAssistant, Blocks: []schema.ContentBlock{{Type: schema.BlockToolCall, CallID: "toolu_vi", Name: "view_image", Arguments: json.RawMessage(`{"path":"red.png"}`)}}},
			{Role: schema.RoleTool, Blocks: []schema.ContentBlock{{
				Type:   schema.BlockToolResult,
				CallID: "toolu_vi",
				Name:   "view_image",
				Output: "Loaded red.png",
				Images: []schema.ImageSource{{MediaType: "image/png", Base64: redB64}},
			}}},
		},
	}, func(ch provider.StreamChunk) {
		if ch.Kind == provider.DeltaHostedToolComplete {
			completedCallID = ch.CallID
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if completedCallID != "srvtoolu_wf1" {
		t.Fatalf("web_fetch_tool_result must preserve tool_use_id in stream event, got %q", completedCallID)
	}
	if len(res.Blocks) < 2 || res.Blocks[1].CallID != "srvtoolu_wf1" {
		t.Fatalf("web_fetch_tool_result block must preserve tool_use_id, got %+v", res.Blocks)
	}
	if !strings.Contains(betaHeader, "web-fetch-2025-09-10") {
		t.Fatalf("expected web-fetch beta header, got %q", betaHeader)
	}
	for _, want := range []string{
		`"type":"web_search_20260209"`,
		`"type":"web_fetch_20250910"`,
		`"type":"tool_result"`,
		`"type":"image"`,
		`"media_type":"image/png"`,
	} {
		if !strings.Contains(reqBody, want) {
			t.Fatalf("anthropic request missing %s\nbody=%s", want, reqBody)
		}
	}
	for _, forbid := range []string{`"x_search"`, `"image_generation"`} {
		if strings.Contains(reqBody, forbid) {
			t.Fatalf("anthropic request must skip %s\nbody=%s", forbid, reqBody)
		}
	}
}

func TestGeminiMultimodalToolResponseInteractionsAndImageGen(t *testing.T) {
	redB64 := tinyPNGBase64(t, color.RGBA{R: 255, A: 255})
	greenB64 := tinyPNGBase64(t, color.RGBA{G: 255, A: 255})

	var genContentBody string
	var interactionsBody string
	var imgGenBody string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch {
		case strings.Contains(r.URL.Path, ":generateContent"):
			imgGenBody = string(body)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"modelVersion": "gemini-2.5-flash-image",
				"candidates": []map[string]any{{
					"content": map[string]any{
						"role": "model",
						"parts": []map[string]any{
							{"text": "Generated green variant"},
							{"inlineData": map[string]any{"mimeType": "image/png", "data": greenB64}},
						},
					},
				}},
			})
			return
		case strings.Contains(r.URL.Path, "/interactions"):
			interactionsBody = string(body)
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"event_type\":\"step.start\",\"index\":0,\"step\":{\"type\":\"google_search_result\",\"id\":\"gsr_1\",\"result\":[{\"url\":\"https://go.dev\",\"title\":\"Go\"}]}}\n\n")
			fmt.Fprint(w, "data: {\"event_type\":\"step.start\",\"index\":1,\"step\":{\"type\":\"model_output\"}}\n\n")
			fmt.Fprint(w, "data: {\"event_type\":\"step.delta\",\"index\":1,\"delta\":{\"type\":\"text\",\"text\":\"Interactions saw image.\"}}\n\n")
			fmt.Fprint(w, "data: {\"event_type\":\"interaction.completed\",\"interaction\":{\"id\":\"int_mm\",\"status\":\"completed\"}}\n\n")
			return
		default:
			genContentBody = string(body)
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintf(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Red image confirmed.\"},{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":\"%s\"}}]},\"finishReason\":\"STOP\"}]}\n\n", greenB64)
		}
	}))
	defer srv.Close()

	// 1. Vertex streamGenerateContent multimodal functionResponse.parts + inlineData response decoding.
	p, err := factory.New(factory.ProfileConfig{
		Provider:       "vertex",
		BaseURL:        srv.URL,
		VertexProject:  "proj-1",
		VertexLocation: "global",
		GCPTokenSource: func(_ context.Context) (string, error) { return "gcp-tok", nil },
		HTTPClient:     srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := p.Stream(context.Background(), provider.Request{
		Model: "gemini-4-neon",
		HostedTools: []schema.HostedTool{
			{Type: "google_search"},
			{Type: "url_context"},
			{Type: "x_search"},
		},
		Tools: []schema.ToolDef{{Name: "view_image"}},
		Messages: []schema.Message{
			{Role: schema.RoleUser, Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "View red.png"}}},
			{Role: schema.RoleAssistant, Blocks: []schema.ContentBlock{{Type: schema.BlockToolCall, CallID: "fc_1", Name: "view_image", Arguments: json.RawMessage(`{"path":"red.png"}`)}}},
			{Role: schema.RoleTool, Blocks: []schema.ContentBlock{{
				Type:   schema.BlockToolResult,
				CallID: "fc_1",
				Name:   "view_image",
				Output: "Loaded red.png",
				Images: []schema.ImageSource{{MediaType: "image/png", Base64: redB64}},
			}}},
		},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`"includeServerSideToolInvocations":true`,
		`"googleSearch":{}`,
		`"urlContext":{}`,
		`"functionResponse"`,
		`"parts":[{"inlineData":{"data":"` + redB64,
	} {
		if !strings.Contains(genContentBody, want) {
			t.Fatalf("gemini generateContent missing %s\nbody=%s", want, genContentBody)
		}
	}
	if strings.Contains(genContentBody, `"x_search"`) {
		t.Fatalf("gemini must skip x_search: %s", genContentBody)
	}
	if len(res.Blocks) != 2 || res.Blocks[1].Type != schema.BlockImage || res.Blocks[1].Image.Base64 != greenB64 {
		t.Fatalf("expected decoded inlineData BlockImage from Gemini SSE, got %+v", res.Blocks)
	}

	// 2. Gemini native GenerateImage with reference image.
	imgRes, err := p.(provider.ImageGenerator).GenerateImage(context.Background(), provider.ImageGenerationRequest{
		Prompt:          "Recolor to green",
		ReferenceImages: []schema.ImageSource{{MediaType: "image/png", Base64: redB64}},
	})
	if err != nil || len(imgRes.Images) != 1 || imgRes.Images[0].Base64 != greenB64 || imgRes.RevisedPrompt != "Generated green variant" {
		t.Fatalf("gemini GenerateImage failed: err=%v res=%+v", err, imgRes)
	}
	if !strings.Contains(imgGenBody, `"responseModalities":["TEXT","IMAGE"]`) || !strings.Contains(imgGenBody, redB64) {
		t.Fatalf("unexpected gemini image gen payload: %s", imgGenBody)
	}

	// 3. Vertex Interactions API multimodal function_result + search citation extraction.
	pi, err := factory.New(factory.ProfileConfig{
		Provider:       "vertex_interactions",
		BaseURL:        srv.URL,
		VertexProject:  "proj-1",
		VertexLocation: "global",
		GCPTokenSource: func(_ context.Context) (string, error) { return "gcp-tok", nil },
		HTTPClient:     srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	ires, err := pi.Stream(context.Background(), provider.Request{
		Model:       "gemini-4-neon",
		HostedTools: []schema.HostedTool{{Type: "google_search"}, {Type: "url_context"}, {Type: "x_search"}},
		Messages: []schema.Message{
			{Role: schema.RoleTool, Blocks: []schema.ContentBlock{{
				Type:   schema.BlockToolResult,
				CallID: "ic_1",
				Name:   "view_image",
				Output: "Loaded red.png",
				Images: []schema.ImageSource{{MediaType: "image/png", Base64: redB64}},
			}}},
		},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(interactionsBody, `"type":"function_result"`) || !strings.Contains(interactionsBody, `"mime_type":"image/png"`) {
		t.Fatalf("interactions multimodal function_result missing image item: %s", interactionsBody)
	}
	if strings.Contains(interactionsBody, `"x_search"`) {
		t.Fatalf("interactions must skip x_search: %s", interactionsBody)
	}
	if len(ires.Blocks) < 2 || len(ires.Blocks[1].Citations) != 1 || ires.Blocks[1].Citations[0].URL != "https://go.dev" {
		t.Fatalf("expected citation extracted from google_search_result, got %+v", ires.Blocks)
	}
}

func TestOpenAICompatMultimodalUserAndToolResultImages(t *testing.T) {
	redB64 := tinyPNGBase64(t, color.RGBA{R: 255, A: 255})
	var reqBody string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		reqBody = string(body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"cmpl_mm\",\"model\":\"Llama-4-Maverick\",\"choices\":[{\"delta\":{\"content\":\"Saw red.\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	p, err := factory.New(factory.ProfileConfig{
		Provider:   "meta",
		BaseURL:    srv.URL,
		APIKey:     "llama-key",
		HTTPClient: srv.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Stream(context.Background(), provider.Request{
		Model: "Llama-4-Maverick-17B-128E-Instruct-FP8",
		Messages: []schema.Message{
			{Role: schema.RoleUser, Blocks: []schema.ContentBlock{
				{Type: schema.BlockText, Text: "Compare images"},
				{Type: schema.BlockImage, Image: &schema.ImageSource{MediaType: "image/png", Base64: redB64}},
			}},
			{Role: schema.RoleTool, Blocks: []schema.ContentBlock{{
				Type:   schema.BlockToolResult,
				CallID: "tc_1",
				Name:   "view_image",
				Output: "Loaded diagram.png",
				Images: []schema.ImageSource{{MediaType: "image/png", Base64: redB64}},
			}}},
		},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(reqBody, `"type":"image_url"`) != 2 || !strings.Contains(reqBody, `"[Image output from view_image]"`) {
		t.Fatalf("expected user + companion tool image_url blocks in openaicompat payload: %s", reqBody)
	}
}
