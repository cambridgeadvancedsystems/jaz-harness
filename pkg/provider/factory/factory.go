package factory

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/auth"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider/anthropic"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider/gemini"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider/grok"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider/openai"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider/openaicompat"
)

type ProfileConfig struct {
	Provider       string // openai | codex_oauth | azure_openai | anthropic | vertex_anthropic | bedrock_anthropic | grok | xai | gemini | vertex | vertex_gemini | meta | llama | openrouter | bedrock_openai | openai_compat
	BaseURL        string
	APIKey         string
	CodexOAuth     *auth.CodexOAuthManager
	VertexProject  string
	VertexLocation string
	GCPTokenSource anthropic.TokenSource
	AWS            auth.AWSCredentials
	HTTPReferer    string
	AppTitle       string
	ExtraHeaders   map[string]string
	HTTPClient     *http.Client
}

// New instantiates the exact native wire-protocol provider for ProfileConfig.
func New(cfg ProfileConfig) (provider.Provider, error) {
	kind := strings.ToLower(strings.TrimSpace(cfg.Provider))
	switch kind {
	case "openai", "azure_openai":
		return openai.New(openai.Config{
			ProviderID:   kind,
			BaseURL:      cfg.BaseURL,
			APIKey:       cfg.APIKey,
			ExtraHeaders: cfg.ExtraHeaders,
			HTTPClient:   cfg.HTTPClient,
		}), nil
	case "codex_oauth", "codex":
		mgr := cfg.CodexOAuth
		if mgr == nil {
			mgr = auth.NewCodexOAuthManager(auth.CodexOAuthConfig{
				AuthPath: auth.DefaultCodexAuthPath(),
			})
		}
		return openai.New(openai.Config{
			ProviderID:   "codex_oauth",
			BaseURL:      cfg.BaseURL,
			CodexOAuth:   mgr,
			ExtraHeaders: cfg.ExtraHeaders,
			HTTPClient:   cfg.HTTPClient,
		}), nil
	case "anthropic":
		return anthropic.New(anthropic.Config{
			ProviderID: "anthropic",
			Mode:       anthropic.ModeDirect,
			BaseURL:    cfg.BaseURL,
			APIKey:     cfg.APIKey,
			HTTPClient: cfg.HTTPClient,
		}), nil
	case "vertex_anthropic":
		return anthropic.New(anthropic.Config{
			ProviderID:     "vertex_anthropic",
			Mode:           anthropic.ModeVertex,
			BaseURL:        cfg.BaseURL,
			VertexProject:  cfg.VertexProject,
			VertexLocation: cfg.VertexLocation,
			VertexToken:    cfg.GCPTokenSource,
			HTTPClient:     cfg.HTTPClient,
		}), nil
	case "bedrock_anthropic":
		return anthropic.New(anthropic.Config{
			ProviderID: "bedrock_anthropic",
			Mode:       anthropic.ModeBedrock,
			BaseURL:    cfg.BaseURL,
			AWS:        cfg.AWS,
			HTTPClient: cfg.HTTPClient,
		}), nil
	case "grok", "xai":
		return grok.New(grok.Config{
			ProviderID: "grok",
			BaseURL:    cfg.BaseURL,
			APIKey:     cfg.APIKey,
			HTTPClient: cfg.HTTPClient,
		}), nil
	case "gemini":
		return gemini.New(gemini.Config{
			ProviderID: "gemini",
			Mode:       gemini.ModeAIStudio,
			BaseURL:    cfg.BaseURL,
			APIKey:     cfg.APIKey,
			HTTPClient: cfg.HTTPClient,
		}), nil
	case "vertex", "vertex_gemini":
		var ts gemini.TokenSource
		if cfg.GCPTokenSource != nil {
			ts = gemini.TokenSource(cfg.GCPTokenSource)
		}
		return gemini.New(gemini.Config{
			ProviderID:  "vertex",
			Mode:        gemini.ModeVertex,
			BaseURL:     cfg.BaseURL,
			Project:     cfg.VertexProject,
			Location:    cfg.VertexLocation,
			TokenSource: ts,
			HTTPClient:  cfg.HTTPClient,
		}), nil
	case "meta", "llama":
		return openaicompat.New(openaicompat.Config{
			ProviderID:   "meta",
			Flavor:       openaicompat.FlavorMeta,
			BaseURL:      cfg.BaseURL,
			APIKey:       cfg.APIKey,
			ExtraHeaders: cfg.ExtraHeaders,
			HTTPClient:   cfg.HTTPClient,
		}), nil
	case "openrouter":
		return openaicompat.New(openaicompat.Config{
			ProviderID:   "openrouter",
			Flavor:       openaicompat.FlavorOpenRouter,
			BaseURL:      cfg.BaseURL,
			APIKey:       cfg.APIKey,
			HTTPReferer:  cfg.HTTPReferer,
			AppTitle:     cfg.AppTitle,
			ExtraHeaders: cfg.ExtraHeaders,
			HTTPClient:   cfg.HTTPClient,
		}), nil
	case "bedrock_openai", "bedrock":
		awsCopy := cfg.AWS
		return openaicompat.New(openaicompat.Config{
			ProviderID:   "bedrock_openai",
			Flavor:       openaicompat.FlavorBedrock,
			BaseURL:      cfg.BaseURL,
			APIKey:       cfg.APIKey,
			AWS:          &awsCopy,
			ExtraHeaders: cfg.ExtraHeaders,
			HTTPClient:   cfg.HTTPClient,
		}), nil
	case "openai_compat":
		return openaicompat.New(openaicompat.Config{
			ProviderID:   "openai_compat",
			Flavor:       openaicompat.FlavorGeneric,
			BaseURL:      cfg.BaseURL,
			APIKey:       cfg.APIKey,
			ExtraHeaders: cfg.ExtraHeaders,
			HTTPClient:   cfg.HTTPClient,
		}), nil
	default:
		return nil, fmt.Errorf("unsupported provider %q", cfg.Provider)
	}
}
