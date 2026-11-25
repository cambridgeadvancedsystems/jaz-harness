package harness_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cambridgeadvancedsystems/jaz-harness/executor"
	"github.com/cambridgeadvancedsystems/jaz-harness/harness"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/cambridgeadvancedsystems/jaz-harness/store"
	"github.com/cambridgeadvancedsystems/jaz-harness/tools"
)

func makePNGBytes(t *testing.T, c color.RGBA) []byte {
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

func TestHarnessUniversalViewImageAndFallbackGenerateImageLoop(t *testing.T) {
	wsRoot := t.TempDir()
	redPNG := makePNGBytes(t, color.RGBA{R: 255, A: 255})
	bluePNG := makePNGBytes(t, color.RGBA{B: 255, A: 255})
	blueB64 := base64.StdEncoding.EncodeToString(bluePNG)
	if err := os.WriteFile(filepath.Join(wsRoot, "input.png"), redPNG, 0o644); err != nil {
		t.Fatal(err)
	}

	// Simulate Anthropic (no native ImageGenerator): calls view_image -> generate_image(num_last_images_to_include:1) -> finishes.
	pClaude := &scriptedProvider{
		id:         "anthropic",
		wireFormat: "anthropic_messages",
		responses: []provider.Response{
			{
				ID:    "a1",
				Model: "claude-sonnet-4-5",
				Blocks: []schema.ContentBlock{
					{Type: schema.BlockToolCall, CallID: "t_vi", Name: "view_image", Arguments: json.RawMessage(`{"path":"input.png"}`)},
				},
			},
			{
				ID:    "a2",
				Model: "claude-sonnet-4-5",
				Blocks: []schema.ContentBlock{
					{Type: schema.BlockToolCall, CallID: "t_gi", Name: "generate_image", Arguments: json.RawMessage(`{"prompt":"Turn it blue","num_last_images_to_include":1}`)},
				},
			},
			{
				ID:     "a3",
				Model:  "claude-sonnet-4-5",
				Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Generated blue version at generated_images/t_gi.png."}},
			},
		},
	}

	var receivedRefs int
	fallbackGen := provider.ImageGeneratorFunc(func(_ context.Context, req provider.ImageGenerationRequest) (provider.ImageGenerationResponse, error) {
		receivedRefs = len(req.ReferenceImages)
		return provider.ImageGenerationResponse{
			ID:     "fb_gen_1",
			Model:  "fallback-img",
			Images: []schema.ImageSource{{MediaType: "image/png", Base64: blueB64}},
		}, nil
	})

	ws := tools.NewWorkspaceToolset(func(_ context.Context) (executor.Executor, error) {
		return executor.NewLocal(wsRoot)
	})
	toolReg := tools.NewRegistry(nil, nil)
	ws.RegisterInto(toolReg)
	tools.RegisterImageGenerationTool(toolReg, tools.ImageGenerationConfig{
		ReadFile:  ws.ReadWorkspaceFile,
		WriteFile: ws.WriteWorkspaceFile,
		Fallback:  fallbackGen,
	})

	provReg := provider.NewRegistry()
	provReg.Register("anthropic", pClaude)

	st := store.NewMemoryStore()
	eng := harness.New(harness.Config{
		DefaultHostedTools: true,
		ImageGenerator:     fallbackGen,
	}, st, provReg, toolReg, nil, nil)

	ctx := context.Background()
	th, err := eng.CreateThread(ctx, schema.Thread{Provider: "anthropic", Model: "claude-sonnet-4-5"})
	if err != nil {
		t.Fatal(err)
	}
	res, err := eng.RunTurn(ctx, harness.TurnRequest{ThreadID: th.ID, Prompt: "Inspect input.png and generate a blue variant."})
	if err != nil || res.Steps != 3 || !strings.Contains(res.AssistantText, "Generated blue version") {
		t.Fatalf("turn failed: err=%v res=%+v", err, res)
	}
	if receivedRefs != 1 {
		t.Fatalf("expected generate_image to pull 1 conversation image from view_image history, got %d", receivedRefs)
	}
	// Step 2 must have received multimodal BlockToolResult with image from view_image.
	step2Msgs := pClaude.calls[1].Messages
	lastToolMsg := step2Msgs[len(step2Msgs)-1]
	if len(lastToolMsg.Blocks) != 1 || len(lastToolMsg.Blocks[0].Images) != 1 {
		t.Fatalf("expected multimodal tool result image on step 2, got %+v", lastToolMsg.Blocks)
	}
	// Step 3 must have received multimodal BlockToolResult with generated image from generate_image.
	step3Msgs := pClaude.calls[2].Messages
	lastGenMsg := step3Msgs[len(step3Msgs)-1]
	if len(lastGenMsg.Blocks) != 1 || len(lastGenMsg.Blocks[0].Images) != 1 || lastGenMsg.Blocks[0].Images[0].Base64 != blueB64 {
		t.Fatalf("expected generated image delivered to step 3, got %+v", lastGenMsg.Blocks)
	}
	// Verify EventToolCallResult carries image_count and never leaks raw base64.
	events, err := st.ListEventsAfter(ctx, th.ID, 0)
	if err != nil {
		t.Fatal(err)
	}
	sawImageCount := false
	for _, ev := range events {
		raw, _ := json.Marshal(ev)
		if strings.Contains(string(raw), blueB64) {
			t.Fatalf("session event must never leak raw base64 image bytes: %s", raw)
		}
		if ev.Type == schema.EventToolCallResult {
			if cnt, ok := ev.Payload["image_count"].(int); ok && cnt == 1 {
				sawImageCount = true
			}
		}
	}
	if !sawImageCount {
		t.Fatal("expected EventToolCallResult with image_count=1")
	}
}

func TestHarnessOpenAIHostedImageGenerationSuppressesDuplicateLocalTool(t *testing.T) {
	pOpenAI := &scriptedProvider{
		id:         "openai",
		wireFormat: "openai_responses",
		responses: []provider.Response{
			{
				ID:     "r1",
				Model:  "gpt-5.4",
				Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: "Generated natively."}},
			},
		},
	}

	provReg := provider.NewRegistry()
	provReg.Register("openai", pOpenAI)
	eng := harness.New(harness.Config{
		DefaultHostedTools:          true,
		EnableNativeImageGeneration: true,
		ImageGenerator: provider.ImageGeneratorFunc(func(_ context.Context, _ provider.ImageGenerationRequest) (provider.ImageGenerationResponse, error) {
			return provider.ImageGenerationResponse{}, nil
		}),
	}, store.NewMemoryStore(), provReg, nil, nil, nil)

	ctx := context.Background()
	th, err := eng.CreateThread(ctx, schema.Thread{Provider: "openai", Model: "gpt-5.4"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := eng.RunTurn(ctx, harness.TurnRequest{ThreadID: th.ID, Prompt: "Draw a mountain"}); err != nil {
		t.Fatal(err)
	}
	call := pOpenAI.calls[0]
	hasHostedIG := false
	for _, h := range call.HostedTools {
		if h.Type == "image_generation" {
			hasHostedIG = true
		}
	}
	if !hasHostedIG {
		t.Fatalf("expected hosted image_generation tool on OpenAI, got %+v", call.HostedTools)
	}
	for _, td := range call.Tools {
		if td.Name == "generate_image" {
			t.Fatal("local generate_image tool must be suppressed when OpenAI native hosted image_generation is active")
		}
	}
}
