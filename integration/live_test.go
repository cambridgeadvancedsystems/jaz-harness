package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/auth"
	"github.com/cambridgeadvancedsystems/jaz-harness/executor"
	"github.com/cambridgeadvancedsystems/jaz-harness/harness"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider/factory"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider/gemini"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/cambridgeadvancedsystems/jaz-harness/store"
	"github.com/cambridgeadvancedsystems/jaz-harness/tools"
)

func gcloudAccessToken(ctx context.Context) (string, error) {
	out, err := exec.CommandContext(ctx, "gcloud", "auth", "print-access-token").Output()
	if err != nil {
		return "", err
	}
	tok := strings.TrimSpace(string(out))
	if tok == "" {
		return "", fmt.Errorf("empty gcloud access token")
	}
	return tok, nil
}

func gcloudProject(ctx context.Context) string {
	if p := strings.TrimSpace(os.Getenv("GCP_PROJECT")); p != "" {
		return p
	}
	out, err := exec.CommandContext(ctx, "gcloud", "config", "get-value", "project").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// buildStablePrefix generates a deterministic technical specification prefix of
// approximately targetTokens tokens (~4 chars/token) so prompt/KV caching
// thresholds (>=1,024 on OpenAI/Codex, >=4,096 on Gemini 4) are cleanly exceeded.
func buildStablePrefix(targetTokens int, tag string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "You are a precise engineering verification assistant (suite=%s).\n", tag)
	b.WriteString("Reference specification corpus (read-only context):\n")
	for i := 1; b.Len() < targetTokens*4; i++ {
		fmt.Fprintf(&b, "SPEC-%04d: Subsystem module-%03d enforces deterministic state transitions, idempotent retry tokens, monotonic sequence numbering, strict schema validation, and auditable latency budgets under 250ms p99 across regional failover nodes.\n", i, i%97)
	}
	return b.String()
}

// TestLiveProviders exercises real provider endpoints when JAZ_HARNESS_LIVE=1
// is set alongside provider credentials (or ~/.codex/auth.json for codex_oauth).
func TestLiveProviders(t *testing.T) {
	if os.Getenv("JAZ_HARNESS_LIVE") != "1" {
		t.Skip("set JAZ_HARNESS_LIVE=1 to run live provider smoke tests")
	}

	type liveCase struct {
		name   string
		cfg    factory.ProfileConfig
		model  string
		effort string
	}
	var cases []liveCase

	if p := auth.DefaultCodexAuthPath(); p != "" {
		if _, err := os.Stat(p); err == nil {
			cases = append(cases, liveCase{
				name:   "codex_oauth",
				cfg:    factory.ProfileConfig{Provider: "codex_oauth", CodexOAuth: auth.NewCodexOAuthManager(auth.CodexOAuthConfig{AuthPath: p})},
				model:  schema.DefaultModelForProvider("codex_oauth"),
				effort: "low",
			})
		}
	}
	if key := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")); key != "" {
		cases = append(cases, liveCase{name: "openai", cfg: factory.ProfileConfig{Provider: "openai", APIKey: key}, model: schema.DefaultModelForProvider("openai"), effort: "low"})
	}
	if key := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")); key != "" {
		cases = append(cases, liveCase{name: "anthropic", cfg: factory.ProfileConfig{Provider: "anthropic", APIKey: key}, model: schema.DefaultModelForProvider("anthropic"), effort: "low"})
	}
	if key := strings.TrimSpace(os.Getenv("XAI_API_KEY")); key != "" {
		cases = append(cases, liveCase{name: "grok", cfg: factory.ProfileConfig{Provider: "grok", APIKey: key}, model: schema.DefaultModelForProvider("grok"), effort: "low"})
	}
	if key := strings.TrimSpace(os.Getenv("GEMINI_API_KEY")); key != "" {
		cases = append(cases, liveCase{name: "gemini", cfg: factory.ProfileConfig{Provider: "gemini", APIKey: key}, model: schema.DefaultModelForProvider("gemini"), effort: "low"})
	}
	if key := strings.TrimSpace(os.Getenv("LLAMA_API_KEY")); key != "" {
		cases = append(cases, liveCase{name: "meta", cfg: factory.ProfileConfig{Provider: "meta", APIKey: key}, model: schema.DefaultModelForProvider("meta")})
	}
	if key := strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY")); key != "" {
		cases = append(cases, liveCase{name: "openrouter", cfg: factory.ProfileConfig{Provider: "openrouter", APIKey: key}, model: schema.DefaultModelForProvider("openrouter")})
	}

	if len(cases) == 0 {
		t.Skip("no live provider credentials found in environment or ~/.codex/auth.json")
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			defer cancel()

			p, err := factory.New(tc.cfg)
			if err != nil {
				t.Fatal(err)
			}
			reg := provider.NewRegistry()
			reg.Register(tc.name, p)
			eng := harness.New(harness.Config{EnableNativeCache: true}, store.NewMemoryStore(), reg, nil, nil, nil)
			th, err := eng.CreateThread(ctx, schema.Thread{
				Provider:        tc.name,
				Model:           tc.model,
				ReasoningEffort: tc.effort,
			})
			if err != nil {
				t.Fatal(err)
			}
			res1, err := eng.RunTurn(ctx, harness.TurnRequest{
				ThreadID:        th.ID,
				Prompt:          "Remember the exact codeword ORBITAL-7741 and reply with only ACK.",
				MaxOutputTokens: 128,
			})
			if err != nil {
				t.Fatalf("turn 1 failed: %v", err)
			}
			res2, err := eng.RunTurn(ctx, harness.TurnRequest{
				ThreadID:        th.ID,
				Prompt:          "What was the exact codeword? Reply with only the codeword.",
				MaxOutputTokens: 128,
			})
			if err != nil {
				t.Fatalf("turn 2 failed: %v", err)
			}
			if !strings.Contains(res2.AssistantText, "ORBITAL-7741") {
				t.Fatalf("turn 2 lost conversation continuity: turn1=%q turn2=%q", res1.AssistantText, res2.AssistantText)
			}
			t.Logf("%s PASS: turn2=%q usage=%+v", tc.name, strings.TrimSpace(res2.AssistantText), res2.Usage)
		})
	}
}

// TestLiveCacheAndContinuity verifies real multi-turn continuation, cold
// FileStore reload, KV/prompt-cache hits (CachedInputTokens > 0), and combined
// hosted + custom tools across Codex OAuth and Vertex Gemini 4 surfaces.
func TestLiveCacheAndContinuity(t *testing.T) {
	if os.Getenv("JAZ_HARNESS_LIVE") != "1" {
		t.Skip("set JAZ_HARNESS_LIVE=1 to run live cache and continuity tests")
	}

	t.Run("codex_oauth", func(t *testing.T) {
		authPath := auth.DefaultCodexAuthPath()
		if authPath == "" {
			t.Skip("codex auth path unavailable")
		}
		if _, err := os.Stat(authPath); err != nil {
			t.Skip("~/.codex/auth.json not present")
		}

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		p, err := factory.New(factory.ProfileConfig{
			Provider:   "codex_oauth",
			CodexOAuth: auth.NewCodexOAuthManager(auth.CodexOAuthConfig{AuthPath: authPath}),
		})
		if err != nil {
			t.Fatal(err)
		}
		dataDir := t.TempDir()
		sysPrompt := buildStablePrefix(1800, fmt.Sprintf("codex-%d", time.Now().UnixNano()))

		newEng := func() *harness.Engine {
			st, serr := store.NewFileStore(dataDir)
			if serr != nil {
				t.Fatal(serr)
			}
			reg := provider.NewRegistry()
			reg.Register("codex_oauth", p)
			return harness.New(harness.Config{EnableNativeCache: true}, st, reg, nil, nil, nil)
		}

		eng1 := newEng()
		th, err := eng1.CreateThread(ctx, schema.Thread{
			Provider:        "codex_oauth",
			Model:           schema.DefaultModelForProvider("codex_oauth"),
			ReasoningEffort: "low",
			SystemPrompt:    sysPrompt,
		})
		if err != nil {
			t.Fatal(err)
		}
		r1, err := eng1.RunTurn(ctx, harness.TurnRequest{
			ThreadID: th.ID,
			Prompt:   "Remember the exact codeword ORBITAL-7741. Reply with only ACK.",
		})
		if err != nil {
			t.Fatalf("turn 1 failed: %v", err)
		}

		// Cold FileStore reload into a fresh Engine instance.
		eng2 := newEng()
		r2, err := eng2.RunTurn(ctx, harness.TurnRequest{
			ThreadID: th.ID,
			Prompt:   "What was the exact codeword? Reply with only the codeword.",
		})
		if err != nil {
			t.Fatalf("turn 2 failed: %v", err)
		}
		if !strings.Contains(r2.AssistantText, "ORBITAL-7741") {
			t.Fatalf("turn 2 lost recall after cold reload: %q", r2.AssistantText)
		}

		eng3 := newEng()
		r3, err := eng3.RunTurn(ctx, harness.TurnRequest{
			ThreadID: th.ID,
			Prompt:   "Repeat the exact codeword once more and add -OK.",
		})
		if err != nil {
			t.Fatalf("turn 3 failed: %v", err)
		}
		if !strings.Contains(r3.AssistantText, "ORBITAL-7741") {
			t.Fatalf("turn 3 lost recall: %q", r3.AssistantText)
		}

		maxCached := r2.Usage.CachedInputTokens
		if r3.Usage.CachedInputTokens > maxCached {
			maxCached = r3.Usage.CachedInputTokens
		}
		t.Logf("codex_oauth LIVE: turn1(in=%d cached=%d out=%d) turn2(in=%d cached=%d out=%d text=%q) turn3(in=%d cached=%d out=%d text=%q)",
			r1.Usage.InputTokens, r1.Usage.CachedInputTokens, r1.Usage.OutputTokens,
			r2.Usage.InputTokens, r2.Usage.CachedInputTokens, r2.Usage.OutputTokens, strings.TrimSpace(r2.AssistantText),
			r3.Usage.InputTokens, r3.Usage.CachedInputTokens, r3.Usage.OutputTokens, strings.TrimSpace(r3.AssistantText))
		if maxCached <= 0 {
			t.Fatalf("expected non-zero cached_input_tokens on turn 2/3, got turn2=%d turn3=%d", r2.Usage.CachedInputTokens, r3.Usage.CachedInputTokens)
		}
	})

	runVertexGeminiCase := func(t *testing.T, surface gemini.APISurface, model, codeword string) {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
		defer cancel()

		project := gcloudProject(ctx)
		if project == "" {
			t.Skip("gcloud project not configured")
		}
		if _, err := gcloudAccessToken(ctx); err != nil {
			t.Skipf("gcloud auth token unavailable: %v", err)
		}

		p, err := factory.New(factory.ProfileConfig{
			Provider:       "vertex",
			GeminiSurface:  surface,
			VertexProject:  project,
			VertexLocation: "global",
			GCPTokenSource: gcloudAccessToken,
		})
		if err != nil {
			t.Fatal(err)
		}

		dataDir := t.TempDir()
		sysPrompt := buildStablePrefix(4800, fmt.Sprintf("gem-%s-%d", surface, time.Now().UnixNano()))
		var toolInvocations atomic.Int32

		newEng := func() *harness.Engine {
			st, serr := store.NewFileStore(dataDir)
			if serr != nil {
				t.Fatal(serr)
			}
			reg := provider.NewRegistry()
			reg.Register("vertex", p)
			toolReg := tools.NewRegistry(nil, nil)
			toolReg.Register(schema.ToolDef{
				Name:        "lookup_warehouse_sku",
				Description: "Look up warehouse stock quantity and aisle for a SKU code.",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"sku":{"type":"string"}},"required":["sku"]}`),
				AlwaysLoad:  true,
			}, func(_ context.Context, args json.RawMessage) (string, error) {
				toolInvocations.Add(1)
				return fmt.Sprintf(`{"sku_args":%s,"qty":147,"aisle":"B-12"}`, strings.TrimSpace(string(args))), nil
			})
			return harness.New(harness.Config{
				EnableNativeCache:  true,
				DefaultHostedTools: true,
			}, st, reg, toolReg, nil, nil)
		}

		eng1 := newEng()
		th, err := eng1.CreateThread(ctx, schema.Thread{
			Provider:        "vertex",
			Model:           model,
			ReasoningEffort: "low",
			SystemPrompt:    sysPrompt,
		})
		if err != nil {
			t.Fatal(err)
		}

		// Turn 1: combined googleSearch + urlContext + custom functionDeclarations -> trigger custom tool + remember codeword.
		r1, err := eng1.RunTurn(ctx, harness.TurnRequest{
			ThreadID:        th.ID,
			Prompt:          fmt.Sprintf("Remember the exact codeword %s. Call lookup_warehouse_sku for sku='SKU-99' right now and report the quantity and aisle.", codeword),
			MaxOutputTokens: 512,
		})
		if err != nil {
			t.Fatalf("%s/%s turn 1 failed: %v", surface, model, err)
		}
		if toolInvocations.Load() < 1 || !strings.Contains(r1.AssistantText, "147") {
			t.Fatalf("%s/%s turn 1 did not execute custom tool cleanly: calls=%d text=%q", surface, model, toolInvocations.Load(), r1.AssistantText)
		}

		// Cold FileStore reload -> Turn 2: recall codeword AND exercise hosted google_search alongside custom tool schema.
		eng2 := newEng()
		r2, err := eng2.RunTurn(ctx, harness.TurnRequest{
			ThreadID:        th.ID,
			Prompt:          "What was the exact codeword from Turn 1, and what was the aisle returned by lookup_warehouse_sku? Also briefly check via web search what Go's official website domain is. Keep your answer under 2 sentences.",
			MaxOutputTokens: 512,
		})
		if err != nil {
			t.Fatalf("%s/%s turn 2 failed: %v", surface, model, err)
		}
		if !strings.Contains(r2.AssistantText, codeword) || !strings.Contains(r2.AssistantText, "B-12") {
			t.Fatalf("%s/%s turn 2 lost recall after cold FileStore reload: text=%q", surface, model, r2.AssistantText)
		}

		// Cold FileStore reload -> Turn 3: short recall to verify steady-state implicit/durable cache hit.
		eng3 := newEng()
		r3, err := eng3.RunTurn(ctx, harness.TurnRequest{
			ThreadID:        th.ID,
			Prompt:          "Reply with only the codeword from Turn 1.",
			MaxOutputTokens: 128,
		})
		if err != nil {
			t.Fatalf("%s/%s turn 3 failed: %v", surface, model, err)
		}
		if !strings.Contains(r3.AssistantText, codeword) {
			t.Fatalf("%s/%s turn 3 lost recall: %q", surface, model, r3.AssistantText)
		}

		maxCached := r1.Usage.CachedInputTokens
		if r2.Usage.CachedInputTokens > maxCached {
			maxCached = r2.Usage.CachedInputTokens
		}
		if r3.Usage.CachedInputTokens > maxCached {
			maxCached = r3.Usage.CachedInputTokens
		}
		t.Logf("vertex/%s/%s LIVE: toolCalls=%d turn1(steps=%d in=%d cached=%d out=%d) turn2(in=%d cached=%d out=%d text=%q) turn3(in=%d cached=%d out=%d text=%q)",
			surface, model, toolInvocations.Load(),
			r1.Steps, r1.Usage.InputTokens, r1.Usage.CachedInputTokens, r1.Usage.OutputTokens,
			r2.Usage.InputTokens, r2.Usage.CachedInputTokens, r2.Usage.OutputTokens, strings.TrimSpace(r2.AssistantText),
			r3.Usage.InputTokens, r3.Usage.CachedInputTokens, r3.Usage.OutputTokens, strings.TrimSpace(r3.AssistantText))
		if maxCached <= 0 {
			t.Fatalf("expected non-zero cached_input_tokens on %s/%s (turn1=%d, turn2=%d, turn3=%d)", surface, model, r1.Usage.CachedInputTokens, r2.Usage.CachedInputTokens, r3.Usage.CachedInputTokens)
		}
	}

	t.Run("vertex_gemini4_hydrogen_generate_content", func(t *testing.T) {
		runVertexGeminiCase(t, gemini.SurfaceGenerateContent, "gemini-4-hydrogen", "NEBULA-9042")
	})

	t.Run("vertex_gemini4_hydrogen_interactions", func(t *testing.T) {
		runVertexGeminiCase(t, gemini.SurfaceInteractions, "gemini-4-hydrogen", "QUASAR-3188")
	})

	t.Run("vertex_gemini4_neon_generate_content", func(t *testing.T) {
		runVertexGeminiCase(t, gemini.SurfaceGenerateContent, "gemini-4-neon", "PULSAR-5519")
	})
}

func makeLiveSolidPNG(t *testing.T, c color.RGBA) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for y := range 64 {
		for x := range 64 {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// TestLiveNativeToolsAndViewImage verifies universal `view_image` multimodal
// tool responses and native hosted web search against live provider endpoints
// whenever JAZ_HARNESS_LIVE=1 and local credentials are present.
func TestLiveNativeToolsAndViewImage(t *testing.T) {
	if os.Getenv("JAZ_HARNESS_LIVE") != "1" {
		t.Skip("set JAZ_HARNESS_LIVE=1 to run live native tool tests")
	}

	runViewImageCheck := func(t *testing.T, provName string, cfg factory.ProfileConfig, model, effort string) {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		p, err := factory.New(cfg)
		if err != nil {
			t.Fatal(err)
		}
		wsRoot := t.TempDir()
		if err := os.WriteFile(filepath.Join(wsRoot, "swatch.png"), makeLiveSolidPNG(t, color.RGBA{R: 255, G: 0, B: 0, A: 255}), 0o644); err != nil {
			t.Fatal(err)
		}

		reg := provider.NewRegistry()
		reg.Register(provName, p)
		toolReg := tools.NewRegistry(nil, nil)
		ws := tools.NewWorkspaceToolset(func(_ context.Context) (executor.Executor, error) {
			return executor.NewLocal(wsRoot)
		})
		ws.RegisterInto(toolReg)

		eng := harness.New(harness.Config{
			EnableNativeCache:  true,
			DefaultHostedTools: true,
		}, store.NewMemoryStore(), reg, toolReg, nil, nil)
		th, err := eng.CreateThread(ctx, schema.Thread{
			Provider:        provName,
			Model:           model,
			ReasoningEffort: effort,
		})
		if err != nil {
			t.Fatal(err)
		}
		res, err := eng.RunTurn(ctx, harness.TurnRequest{
			ThreadID:        th.ID,
			Prompt:          "Call view_image on path='swatch.png' and reply with ONLY the single dominant color word (for example: red, green, or blue).",
			MaxOutputTokens: 256,
		})
		if err != nil {
			t.Fatalf("%s view_image turn failed: %v", provName, err)
		}
		if res.Steps < 2 || !strings.Contains(strings.ToLower(res.AssistantText), "red") {
			t.Fatalf("%s view_image did not visually identify red swatch: steps=%d text=%q", provName, res.Steps, res.AssistantText)
		}
		t.Logf("%s LIVE view_image PASS: steps=%d text=%q usage=%+v", provName, res.Steps, strings.TrimSpace(res.AssistantText), res.Usage)
	}

	t.Run("vertex_gemini_view_image", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		project := gcloudProject(ctx)
		if project == "" {
			t.Skip("gcloud project not configured")
		}
		if _, err := gcloudAccessToken(ctx); err != nil {
			t.Skipf("gcloud auth token unavailable: %v", err)
		}
		runViewImageCheck(t, "vertex", factory.ProfileConfig{
			Provider:       "vertex",
			GeminiSurface:  gemini.SurfaceGenerateContent,
			VertexProject:  project,
			VertexLocation: "global",
			GCPTokenSource: gcloudAccessToken,
		}, "gemini-4-hydrogen", "low")
	})

	t.Run("vertex_interactions_view_image", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		project := gcloudProject(ctx)
		if project == "" {
			t.Skip("gcloud project not configured")
		}
		if _, err := gcloudAccessToken(ctx); err != nil {
			t.Skipf("gcloud auth token unavailable: %v", err)
		}
		runViewImageCheck(t, "vertex", factory.ProfileConfig{
			Provider:       "vertex",
			GeminiSurface:  gemini.SurfaceInteractions,
			VertexProject:  project,
			VertexLocation: "global",
			GCPTokenSource: gcloudAccessToken,
		}, "gemini-4-hydrogen", "low")
	})

	t.Run("openai_view_image", func(t *testing.T) {
		key := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
		if key == "" {
			t.Skip("OPENAI_API_KEY not set")
		}
		runViewImageCheck(t, "openai", factory.ProfileConfig{
			Provider: "openai",
			APIKey:   key,
		}, schema.DefaultModelForProvider("openai"), "low")
	})

	t.Run("codex_oauth_view_image", func(t *testing.T) {
		authPath := auth.DefaultCodexAuthPath()
		if authPath == "" {
			t.Skip("codex auth path unavailable")
		}
		if _, err := os.Stat(authPath); err != nil {
			t.Skip("~/.codex/auth.json not present")
		}
		runViewImageCheck(t, "codex_oauth", factory.ProfileConfig{
			Provider:   "codex_oauth",
			CodexOAuth: auth.NewCodexOAuthManager(auth.CodexOAuthConfig{AuthPath: authPath}),
		}, schema.DefaultModelForProvider("codex_oauth"), "low")
	})
}
