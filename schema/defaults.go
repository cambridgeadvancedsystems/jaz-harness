package schema

import "strings"

// Canonical default chat/reasoning and native image generation model IDs.
// Every provider package re-exports these exact constants so compile-time
// single-source-of-truth parity is guaranteed across schema, provider clients,
// CLI defaults, and server discovery responses.
const (
	DefaultOpenAIModel        = "gpt-6.2-sol"
	DefaultOpenAIImageModel   = "gpt-image-2.6-sunburst"
	DefaultCodexImageModel    = "gpt-image-2"
	DefaultAnthropicModel     = "claude-opus-5-5"
	DefaultGrokModel          = "grok-4.7"
	DefaultGrokImageModel     = "grok-imagine-image-2.0"
	DefaultGeminiModel        = "gemini-4-neon"
	DefaultGeminiImageModel   = "gemini-nano-banana-2.1"
	DefaultMetaModel          = "Llama-4-Maverick-17B-128E-Instruct-FP8"
	DefaultOpenRouterModel    = "openai/gpt-6.2-sol"
	DefaultBedrockOpenAIModel = "openai.gpt-oss-120b"
)

type ProviderModelDefaults struct {
	Provider          string `json:"provider"`
	DefaultModel      string `json:"default_model"`
	DefaultImageModel string `json:"default_image_model,omitempty"`
	UpstreamBasis     string `json:"upstream_basis"`
}

var builtinProviderDefaults = []ProviderModelDefaults{
	{
		Provider:          "openai",
		DefaultModel:      DefaultOpenAIModel,
		DefaultImageModel: DefaultOpenAIImageModel,
		UpstreamBasis:     "Codex models.json priority 1 (gpt-6.2-sol) + OpenAI Images API 2.6 flagship (gpt-image-2.6-sunburst)",
	},
	{
		Provider:          "azure_openai",
		DefaultModel:      DefaultOpenAIModel,
		DefaultImageModel: DefaultOpenAIImageModel,
		UpstreamBasis:     "OpenAI API parity (gpt-6.2-sol + gpt-image-2.6-sunburst)",
	},
	{
		Provider:          "codex_oauth",
		DefaultModel:      DefaultOpenAIModel,
		DefaultImageModel: DefaultCodexImageModel,
		UpstreamBasis:     "Codex models.json priority 1 (gpt-6.2-sol) + codex-rs/ext/image-generation IMAGE_MODEL (gpt-image-2)",
	},
	{
		Provider:      "anthropic",
		DefaultModel:  DefaultAnthropicModel,
		UpstreamBasis: "Claude Code cc catalog default + Anthropic API flagship (claude-opus-5-5)",
	},
	{
		Provider:      "vertex_anthropic",
		DefaultModel:  DefaultAnthropicModel,
		UpstreamBasis: "Claude Code provider_alias_targets.opus.per_provider.vertex (claude-opus-5-5)",
	},
	{
		Provider:      "bedrock_anthropic",
		DefaultModel:  DefaultAnthropicModel,
		UpstreamBasis: "Claude Code provider_alias_targets.opus.per_provider.bedrock (claude-opus-5-5)",
	},
	{
		Provider:          "grok",
		DefaultModel:      DefaultGrokModel,
		DefaultImageModel: DefaultGrokImageModel,
		UpstreamBasis:     "Grok CLI default_model (grok-4.7) + xAI Images API active model (grok-imagine-image-2.0)",
	},
	{
		Provider:          "gemini",
		DefaultModel:      DefaultGeminiModel,
		DefaultImageModel: DefaultGeminiImageModel,
		UpstreamBasis:     "Antigravity/Gemini 4 balanced coding flagship (gemini-4-neon) + Google AI GA image model (gemini-nano-banana-2.1)",
	},
	{
		Provider:          "vertex",
		DefaultModel:      DefaultGeminiModel,
		DefaultImageModel: DefaultGeminiImageModel,
		UpstreamBasis:     "Antigravity/Vertex Gemini 4 balanced coding flagship (gemini-4-neon) + Vertex AI GA image model (gemini-nano-banana-2.1)",
	},
	{
		Provider:          "vertex_interactions",
		DefaultModel:      DefaultGeminiModel,
		DefaultImageModel: DefaultGeminiImageModel,
		UpstreamBasis:     "Vertex v1beta1 Interactions API Gemini 4 flagship (gemini-4-neon) + Vertex AI GA image model (gemini-nano-banana-2.1)",
	},
	{
		Provider:      "meta",
		DefaultModel:  DefaultMetaModel,
		UpstreamBasis: "Meta Llama API latest official flagship (Llama-4-Maverick-17B-128E-Instruct-FP8)",
	},
	{
		Provider:      "openrouter",
		DefaultModel:  DefaultOpenRouterModel,
		UpstreamBasis: "OpenRouter latest OpenAI flagship route (openai/gpt-6.2-sol)",
	},
	{
		Provider:      "bedrock_openai",
		DefaultModel:  DefaultBedrockOpenAIModel,
		UpstreamBasis: "AWS Bedrock OpenAI-compatible open-weight default (openai.gpt-oss-120b)",
	},
}

// ProviderDefaults returns a copy of the canonical default model table across
// all supported providers.
func ProviderDefaults() []ProviderModelDefaults {
	out := make([]ProviderModelDefaults, len(builtinProviderDefaults))
	copy(out, builtinProviderDefaults)
	return out
}

func normalizeProviderKey(providerName string) string {
	switch strings.ToLower(strings.TrimSpace(providerName)) {
	case "codex":
		return "codex_oauth"
	case "xai":
		return "grok"
	case "vertex_gemini":
		return "vertex"
	case "llama":
		return "meta"
	case "bedrock":
		return "bedrock_openai"
	default:
		return strings.ToLower(strings.TrimSpace(providerName))
	}
}

// DefaultModelForProvider returns the latest official or upstream-harness-matched
// default chat/reasoning model ID for providerName, or empty string if unknown.
func DefaultModelForProvider(providerName string) string {
	key := normalizeProviderKey(providerName)
	for _, d := range builtinProviderDefaults {
		if d.Provider == key {
			return d.DefaultModel
		}
	}
	return ""
}

// DefaultImageModelForProvider returns the latest official or upstream-harness-matched
// native image generation model ID for providerName, or empty string if the
// provider relies on a fallback image generator.
func DefaultImageModelForProvider(providerName string) string {
	key := normalizeProviderKey(providerName)
	for _, d := range builtinProviderDefaults {
		if d.Provider == key {
			return d.DefaultImageModel
		}
	}
	return ""
}
