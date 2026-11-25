package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/auth"
	"github.com/cambridgeadvancedsystems/jaz-harness/executor"
	"github.com/cambridgeadvancedsystems/jaz-harness/harness"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/provider/factory"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/cambridgeadvancedsystems/jaz-harness/server"
	"github.com/cambridgeadvancedsystems/jaz-harness/skills"
	"github.com/cambridgeadvancedsystems/jaz-harness/store"
	"github.com/cambridgeadvancedsystems/jaz-harness/tools"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "serve":
		runServe(os.Args[2:])
	case "run":
		runPrompt(os.Args[2:])
	case "login":
		runLogin(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "Usage: jaz-harness <serve|run|login> [flags]")
}

func registerEnvProviders() *provider.Registry {
	reg := provider.NewRegistry()

	if key := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")); key != "" {
		if p, err := factory.New(factory.ProfileConfig{Provider: "openai", APIKey: key, BaseURL: os.Getenv("OPENAI_BASE_URL")}); err == nil {
			reg.Register("openai", p)
		}
	}
	codexAuth := auth.DefaultCodexAuthPath()
	if codexAuth != "" {
		if _, err := os.Stat(codexAuth); err == nil {
			mgr := auth.NewCodexOAuthManager(auth.CodexOAuthConfig{AuthPath: codexAuth})
			if p, err := factory.New(factory.ProfileConfig{Provider: "codex_oauth", CodexOAuth: mgr}); err == nil {
				reg.Register("codex_oauth", p)
			}
		}
	}
	if key := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")); key != "" {
		if p, err := factory.New(factory.ProfileConfig{Provider: "anthropic", APIKey: key, BaseURL: os.Getenv("ANTHROPIC_BASE_URL")}); err == nil {
			reg.Register("anthropic", p)
		}
	}
	if key := strings.TrimSpace(os.Getenv("XAI_API_KEY")); key != "" {
		if p, err := factory.New(factory.ProfileConfig{Provider: "grok", APIKey: key, BaseURL: os.Getenv("XAI_BASE_URL")}); err == nil {
			reg.Register("grok", p)
		}
	}
	if key := strings.TrimSpace(os.Getenv("GEMINI_API_KEY")); key != "" {
		if p, err := factory.New(factory.ProfileConfig{Provider: "gemini", APIKey: key}); err == nil {
			reg.Register("gemini", p)
		}
	}
	if key := strings.TrimSpace(os.Getenv("LLAMA_API_KEY")); key != "" {
		if p, err := factory.New(factory.ProfileConfig{Provider: "meta", APIKey: key, BaseURL: os.Getenv("LLAMA_BASE_URL")}); err == nil {
			reg.Register("meta", p)
		}
	}
	if key := strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY")); key != "" {
		if p, err := factory.New(factory.ProfileConfig{Provider: "openrouter", APIKey: key, AppTitle: "jaz-harness"}); err == nil {
			reg.Register("openrouter", p)
		}
	}
	if ak := strings.TrimSpace(os.Getenv("AWS_ACCESS_KEY_ID")); ak != "" && os.Getenv("AWS_SECRET_ACCESS_KEY") != "" {
		awsCreds := auth.AWSCredentials{
			AccessKeyID:     ak,
			SecretAccessKey: os.Getenv("AWS_SECRET_ACCESS_KEY"),
			SessionToken:    os.Getenv("AWS_SESSION_TOKEN"),
			Region:          os.Getenv("AWS_REGION"),
		}
		if p, err := factory.New(factory.ProfileConfig{Provider: "bedrock_anthropic", AWS: awsCreds}); err == nil {
			reg.Register("bedrock_anthropic", p)
		}
	}
	return reg
}

func resolveFallbackImageGenerator(reg *provider.Registry) provider.ImageGenerator {
	for _, name := range []string{"openai", "codex_oauth", "gemini", "vertex", "grok"} {
		if p, ok := reg.Get(name); ok {
			if gen, ok := p.(provider.ImageGenerator); ok {
				return gen
			}
		}
	}
	return nil
}

func runServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", ":8090", "HTTP listen address")
	dataDir := fs.String("data-dir", ".jaz-harness", "Durable thread storage directory")
	workspaceDir := fs.String("workspace", "", "Optional local workspace directory for lazy sandbox tools")
	skillsDir := fs.String("skills-dir", "", "Optional directory of SKILL.md skills to load")
	authToken := fs.String("auth-token", os.Getenv("JAZ_HARNESS_TOKEN"), "Optional Bearer auth token")
	_ = fs.Parse(args)

	st, err := store.NewFileStore(*dataDir)
	if err != nil {
		log.Fatalf("init store: %v", err)
	}
	skillCat := skills.NewCatalog()
	if *skillsDir != "" {
		if _, err := skillCat.LoadDir(*skillsDir, ""); err != nil {
			log.Fatalf("load skills: %v", err)
		}
	}
	pluginReg := skills.NewPluginRegistry()
	toolReg := tools.NewRegistry(skillCat, pluginReg)
	provReg := registerEnvProviders()
	fallbackImgGen := resolveFallbackImageGenerator(provReg)

	imgCfg := tools.ImageGenerationConfig{Fallback: fallbackImgGen}
	if *workspaceDir != "" {
		wsDir := *workspaceDir
		wsTools := tools.NewWorkspaceToolset(func(_ context.Context) (executor.Executor, error) {
			return executor.NewLocal(wsDir)
		})
		wsTools.RegisterInto(toolReg)
		imgCfg.ReadFile = wsTools.ReadWorkspaceFile
		imgCfg.WriteFile = wsTools.WriteWorkspaceFile
	}
	tools.RegisterImageGenerationTool(toolReg, imgCfg)

	eng := harness.New(harness.Config{
		EnableNativeCache:  true,
		DefaultHostedTools: true,
		AutoCompact:        true,
		ImageGenerator:     fallbackImgGen,
	}, st, provReg, toolReg, skillCat, pluginReg)

	srv := server.New(server.Config{AuthToken: *authToken}, eng)
	httpSrv := &http.Server{
		Addr:              *addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 15 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(shutCtx)
	}()

	log.Printf("jaz-harness listening on %s (providers: %v)", *addr, eng.Providers().Names())
	if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("serve: %v", err)
	}
}

func runPrompt(args []string) {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	serverURL := fs.String("server", "", "Remote jaz-harness server URL (empty = embedded)")
	authToken := fs.String("auth-token", os.Getenv("JAZ_HARNESS_TOKEN"), "Optional Bearer auth token")
	provName := fs.String("provider", "openai", "Provider ID")
	model := fs.String("model", "gpt-5.4", "Model ID")
	effort := fs.String("effort", "", "Reasoning effort (low|medium|high|xhigh)")
	_ = fs.Parse(args)

	prompt := strings.TrimSpace(strings.Join(fs.Args(), " "))
	if prompt == "" {
		log.Fatal("prompt is required")
	}

	ctx := context.Background()
	if *serverURL != "" {
		cli := server.NewClient(*serverURL, *authToken, nil)
		th, err := cli.CreateThread(ctx, schema.Thread{
			Provider:        *provName,
			Model:           *model,
			ReasoningEffort: *effort,
		})
		if err != nil {
			log.Fatalf("create thread: %v", err)
		}
		res, err := cli.RunTurn(ctx, th.ID, prompt, false)
		if err != nil {
			log.Fatalf("run turn: %v", err)
		}
		fmt.Println(res.AssistantText)
		return
	}

	provReg := registerEnvProviders()
	fallbackImgGen := resolveFallbackImageGenerator(provReg)
	eng := harness.New(harness.Config{
		EnableNativeCache:  true,
		DefaultHostedTools: true,
		AutoCompact:        true,
		ImageGenerator:     fallbackImgGen,
	}, store.NewMemoryStore(), provReg, nil, nil, nil)

	th, err := eng.CreateThread(ctx, schema.Thread{
		Provider:        *provName,
		Model:           *model,
		ReasoningEffort: *effort,
	})
	if err != nil {
		log.Fatalf("create thread: %v", err)
	}
	res, err := eng.RunTurn(ctx, harness.TurnRequest{
		ThreadID: th.ID,
		Prompt:   prompt,
	})
	if err != nil {
		log.Fatalf("run turn: %v", err)
	}
	fmt.Println(res.AssistantText)
}

func runLogin(args []string) {
	if len(args) == 0 || args[0] != "codex" {
		log.Fatal("Usage: jaz-harness login codex")
	}
	mgr := auth.NewCodexOAuthManager(auth.CodexOAuthConfig{
		AuthPath: auth.DefaultCodexAuthPath(),
	})
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	dc, err := mgr.RequestDeviceCode(ctx)
	if err != nil {
		log.Fatalf("request device code: %v", err)
	}
	fmt.Printf("Open %s and enter code: %s\n", dc.VerificationURL, dc.UserCode)
	if _, err := mgr.CompleteDeviceCodeLogin(ctx, dc); err != nil {
		log.Fatalf("device login failed: %v", err)
	}
	fmt.Printf("Saved Codex OAuth credentials to %s\n", auth.DefaultCodexAuthPath())
}
