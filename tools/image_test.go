package tools_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cambridgeadvancedsystems/jaz-harness/executor"
	"github.com/cambridgeadvancedsystems/jaz-harness/mcp"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/cambridgeadvancedsystems/jaz-harness/tools"
)

func makeTestPNG(t *testing.T, c color.RGBA) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 4, 3))
	for y := range 3 {
		for x := range 4 {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestViewImageValidationLazyWorkspaceAndMetadata(t *testing.T) {
	wsRoot := t.TempDir()
	pngBytes := makeTestPNG(t, color.RGBA{R: 255, A: 255})
	if err := os.WriteFile(filepath.Join(wsRoot, "AGENTS.md"), []byte("Inspect images carefully."), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wsRoot, "sample.png"), pngBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wsRoot, "fake.png"), []byte("not an image"), 0o644); err != nil {
		t.Fatal(err)
	}

	reg := tools.NewRegistry(nil, nil)
	ws := tools.NewWorkspaceToolset(func(_ context.Context) (executor.Executor, error) {
		return executor.NewLocal(wsRoot)
	})
	ws.RegisterInto(reg)

	if ws.Attached() {
		t.Fatal("view_image registration must not attach workspace eagerly")
	}

	// Valid image with original detail attaches workspace lazily and returns multimodal Result.
	res, err := reg.ExecuteResult(context.Background(), "view_image", json.RawMessage(`{"path":"sample.png","detail":"original"}`))
	if err != nil {
		t.Fatalf("view_image failed: %v", err)
	}
	if !ws.Attached() {
		t.Fatal("expected lazy workspace attach on first view_image call")
	}
	if !strings.Contains(res.Text, "Inspect images carefully.") || !strings.Contains(res.Text, "4x3") || !strings.Contains(res.Text, "detail=original") {
		t.Fatalf("unexpected view_image text: %q", res.Text)
	}
	if len(res.Images) != 1 || res.Images[0].MediaType != "image/png" || res.Images[0].Detail != "original" || res.Images[0].Path != "sample.png" {
		t.Fatalf("unexpected view_image image result: %+v", res.Images)
	}
	if strings.Contains(res.Text, res.Images[0].Base64) {
		t.Fatal("view_image text must never inline raw base64")
	}

	// Invalid image bytes rejected cleanly.
	if _, err := reg.ExecuteResult(context.Background(), "view_image", json.RawMessage(`{"path":"fake.png"}`)); err == nil || !strings.Contains(err.Error(), "invalid or unsupported image data") {
		t.Fatalf("expected invalid image error, got %v", err)
	}

	// Unsupported detail rejected cleanly (matching Codex).
	if _, err := reg.ExecuteResult(context.Background(), "view_image", json.RawMessage(`{"path":"sample.png","detail":"low"}`)); err == nil || !strings.Contains(err.Error(), "only supports") {
		t.Fatalf("expected unsupported detail error, got %v", err)
	}
}

func TestGenerateImageNativeFallbackAndReferenceInputs(t *testing.T) {
	wsRoot := t.TempDir()
	refPNG := makeTestPNG(t, color.RGBA{R: 255, A: 255})
	outPNG := makeTestPNG(t, color.RGBA{G: 255, A: 255})
	outB64 := base64.StdEncoding.EncodeToString(outPNG)
	if err := os.WriteFile(filepath.Join(wsRoot, "ref.png"), refPNG, 0o644); err != nil {
		t.Fatal(err)
	}

	ws := tools.NewWorkspaceToolset(func(_ context.Context) (executor.Executor, error) {
		return executor.NewLocal(wsRoot)
	})
	reg := tools.NewRegistry(nil, nil)
	ws.RegisterInto(reg)

	var fallbackCalls int
	var lastFallbackReq provider.ImageGenerationRequest
	fallbackGen := provider.ImageGeneratorFunc(func(_ context.Context, req provider.ImageGenerationRequest) (provider.ImageGenerationResponse, error) {
		fallbackCalls++
		lastFallbackReq = req
		return provider.ImageGenerationResponse{
			ID:            "fb_1",
			Model:         "fallback-image-1",
			RevisedPrompt: "Fallback edited image",
			Images:        []schema.ImageSource{{MediaType: "image/png", Base64: outB64}},
		}, nil
	})

	var nativeCalls int
	nativeGen := provider.ImageGeneratorFunc(func(_ context.Context, req provider.ImageGenerationRequest) (provider.ImageGenerationResponse, error) {
		nativeCalls++
		return provider.ImageGenerationResponse{
			ID:     "nat_1",
			Model:  "native-image-1",
			Images: []schema.ImageSource{{MediaType: "image/png", Base64: outB64}},
		}, nil
	})

	tools.RegisterImageGenerationTool(reg, tools.ImageGenerationConfig{
		ReadFile:  ws.ReadWorkspaceFile,
		WriteFile: ws.WriteWorkspaceFile,
		Fallback:  fallbackGen,
	})

	// 1. Without active provider generator -> uses Fallback + loads referenced_image_paths + saves to workspace!
	ctx := tools.WithToolCallID(context.Background(), "call_edit_1")
	res, err := reg.ExecuteResult(ctx, "generate_image", json.RawMessage(`{"prompt":"Make it green","referenced_image_paths":["ref.png"],"transparent_background":true}`))
	if err != nil {
		t.Fatalf("generate_image fallback failed: %v", err)
	}
	if fallbackCalls != 1 || len(lastFallbackReq.ReferenceImages) != 1 || lastFallbackReq.Background != "transparent" {
		t.Fatalf("expected fallback with 1 ref image + transparent bg, calls=%d req=%+v", fallbackCalls, lastFallbackReq)
	}
	if len(res.Images) != 1 || res.Images[0].Path != "generated_images/call_edit_1.png" {
		t.Fatalf("expected saved generated image path, got %+v", res.Images)
	}
	savedBytes, err := os.ReadFile(filepath.Join(wsRoot, "generated_images", "call_edit_1.png"))
	if err != nil || !bytes.Equal(savedBytes, outPNG) {
		t.Fatalf("generated image file not saved cleanly: err=%v len=%d", err, len(savedBytes))
	}

	// 2. With active native provider generator + num_last_images_to_include -> prefers native!
	histCtx := tools.WithActiveImageGenerator(
		tools.WithConversationImages(context.Background(), []schema.ImageSource{{MediaType: "image/png", Base64: base64.StdEncoding.EncodeToString(refPNG)}}),
		nativeGen,
	)
	res2, err := reg.ExecuteResult(histCtx, "generate_image", json.RawMessage(`{"prompt":"Recolor last image","num_last_images_to_include":1,"output_path":"out/custom.png"}`))
	if err != nil {
		t.Fatalf("generate_image native failed: %v", err)
	}
	if nativeCalls != 1 || fallbackCalls != 1 || len(res2.Images) != 1 || res2.Images[0].Path != "out/custom.png" {
		t.Fatalf("expected native generator + custom output path, native=%d fallback=%d imgs=%+v", nativeCalls, fallbackCalls, res2.Images)
	}

	// 3. Rejects passing BOTH referenced_image_paths and num_last_images_to_include (matching Codex).
	if _, err := reg.ExecuteResult(histCtx, "generate_image", json.RawMessage(`{"prompt":"bad","referenced_image_paths":["ref.png"],"num_last_images_to_include":1}`)); err == nil || !strings.Contains(err.Error(), "only one of") {
		t.Fatalf("expected mutual exclusivity error, got %v", err)
	}

	// 4. Rejects requesting more history images than exist.
	if _, err := reg.ExecuteResult(histCtx, "generate_image", json.RawMessage(`{"prompt":"bad","num_last_images_to_include":3}`)); err == nil || !strings.Contains(err.Error(), "only 1 were available") {
		t.Fatalf("expected history bounds error, got %v", err)
	}
}

func TestMCPImageResultPreservedThroughDeferredCallTool(t *testing.T) {
	pngB64 := base64.StdEncoding.EncodeToString(makeTestPNG(t, color.RGBA{B: 255, A: 255}))
	mcpSrv := mcp.NewServer("viz", "1.0")
	mcpSrv.RegisterTool(mcp.Tool{
		Name:        "render_chart",
		Description: "Render chart PNG",
	}, func(_ context.Context, _ json.RawMessage) (mcp.CallToolResult, error) {
		return mcp.CallToolResult{
			Content: []mcp.ToolContent{
				{Type: "text", Text: "Chart ready"},
				{Type: "image", MimeType: "image/png", Data: pngB64},
			},
		}, nil
	})
	ts := httptest.NewServer(mcpSrv)
	defer ts.Close()
	cli := mcp.NewHTTPClient("viz", ts.URL, nil, ts.Client())
	defer cli.Close()

	reg := tools.NewRegistry(nil, nil)
	if n, _, err := reg.RegisterMCPServer(context.Background(), cli); err != nil || n != 1 {
		t.Fatalf("RegisterMCPServer failed: n=%d err=%v", n, err)
	}
	res, err := reg.ExecuteResult(context.Background(), "call_tool", json.RawMessage(`{"name":"viz_render_chart","arguments":{}}`))
	if err != nil {
		t.Fatalf("call_tool failed: %v", err)
	}
	if res.Text != "Chart ready" || len(res.Images) != 1 || res.Images[0].Base64 != pngB64 {
		t.Fatalf("expected MCP image preserved through call_tool, got %+v", res)
	}
}
