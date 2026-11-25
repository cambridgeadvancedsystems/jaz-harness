package integration

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/auth"
	"github.com/cambridgeadvancedsystems/jaz-harness/harness"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider/factory"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/cambridgeadvancedsystems/jaz-harness/store"
)

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
				model:  "gpt-6.1-sol",
				effort: "low",
			})
		}
	}
	if key := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")); key != "" {
		cases = append(cases, liveCase{name: "openai", cfg: factory.ProfileConfig{Provider: "openai", APIKey: key}, model: "gpt-5.4", effort: "low"})
	}
	if key := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")); key != "" {
		cases = append(cases, liveCase{name: "anthropic", cfg: factory.ProfileConfig{Provider: "anthropic", APIKey: key}, model: "claude-sonnet-4-5", effort: "low"})
	}
	if key := strings.TrimSpace(os.Getenv("XAI_API_KEY")); key != "" {
		cases = append(cases, liveCase{name: "grok", cfg: factory.ProfileConfig{Provider: "grok", APIKey: key}, model: "grok-4", effort: "low"})
	}
	if key := strings.TrimSpace(os.Getenv("GEMINI_API_KEY")); key != "" {
		cases = append(cases, liveCase{name: "gemini", cfg: factory.ProfileConfig{Provider: "gemini", APIKey: key}, model: "gemini-2.5-flash", effort: "low"})
	}
	if key := strings.TrimSpace(os.Getenv("LLAMA_API_KEY")); key != "" {
		cases = append(cases, liveCase{name: "meta", cfg: factory.ProfileConfig{Provider: "meta", APIKey: key}, model: "Llama-4-Maverick-17B-128E-Instruct-FP8"})
	}
	if key := strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY")); key != "" {
		cases = append(cases, liveCase{name: "openrouter", cfg: factory.ProfileConfig{Provider: "openrouter", APIKey: key}, model: "openai/gpt-5.4"})
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
