package schema

import "strings"

type ModelSpec struct {
	ID                       string  `json:"id"`
	Provider                 string  `json:"provider"`
	ContextWindow            int64   `json:"context_window"`
	MaxOutputTokens          int     `json:"max_output_tokens"`
	AutoCompactFraction      float64 `json:"auto_compact_fraction"`
	AutoCompactTokenLimit    int64   `json:"auto_compact_token_limit,omitempty"`
	SupportsReasoning        bool    `json:"supports_reasoning"`
	SupportsPromptCache      bool    `json:"supports_prompt_cache"`
	SupportsRemoteCompaction bool    `json:"supports_remote_compaction"`
	SupportsNativeToolSearch bool    `json:"supports_native_tool_search"`
	InputCostPerMillion      float64 `json:"input_cost_per_million"`
	CachedCostPerMillion     float64 `json:"cached_cost_per_million"`
	OutputCostPerMillion     float64 `json:"output_cost_per_million"`
}

// AutoCompactThreshold returns the quality-tuned default auto-compaction
// threshold in input tokens. Explicit AutoCompactTokenLimit overrides keep
// 1M/2M-window models well below raw capacity so reasoning quality stays sharp,
// while always clamping to at most the percentage ceiling of ContextWindow.
func (m ModelSpec) AutoCompactThreshold() int64 {
	frac := m.AutoCompactFraction
	if frac <= 0 || frac >= 1 {
		frac = 0.85
	}
	pctLimit := int64(float64(m.ContextWindow) * frac)
	if m.AutoCompactTokenLimit > 0 {
		if pctLimit > 0 && m.AutoCompactTokenLimit > pctLimit {
			return pctLimit
		}
		return m.AutoCompactTokenLimit
	}
	return pctLimit
}

func (m ModelSpec) EstimateCostUSD(u Usage) float64 {
	uncached := u.InputTokens - u.CachedInputTokens
	if uncached < 0 {
		uncached = 0
	}
	inCost := float64(uncached) * m.InputCostPerMillion
	cacheCost := float64(u.CachedInputTokens) * m.CachedCostPerMillion
	cacheWriteCost := float64(u.CacheCreationTokens) * m.InputCostPerMillion * 1.25
	outCost := float64(u.OutputTokens) * m.OutputCostPerMillion
	return (inCost + cacheCost + cacheWriteCost + outCost) / 1_000_000.0
}

type ProviderModelDefaults struct {
	Provider          string `json:"provider"`
	DefaultModel      string `json:"default_model"`
	DefaultImageModel string `json:"default_image_model,omitempty"`
	UpstreamBasis     string `json:"upstream_basis"`
}

var builtinProviderDefaults = []ProviderModelDefaults{
	{
		Provider:          "openai",
		DefaultModel:      "gpt-6.2-sol",
		DefaultImageModel: "gpt-image-2.6-sunburst",
		UpstreamBasis:     "Codex models.json priority 1 (gpt-6.2-sol) + OpenAI Images API 2.6 flagship (gpt-image-2.6-sunburst)",
	},
	{
		Provider:          "azure_openai",
		DefaultModel:      "gpt-6.2-sol",
		DefaultImageModel: "gpt-image-2.6-sunburst",
		UpstreamBasis:     "OpenAI API parity (gpt-6.2-sol + gpt-image-2.6-sunburst)",
	},
	{
		Provider:          "codex_oauth",
		DefaultModel:      "gpt-6.2-sol",
		DefaultImageModel: "gpt-image-2",
		UpstreamBasis:     "Codex models.json priority 1 (gpt-6.2-sol) + codex-rs/ext/image-generation IMAGE_MODEL (gpt-image-2)",
	},
	{
		Provider:      "anthropic",
		DefaultModel:  "claude-opus-5-5",
		UpstreamBasis: "Claude Code cc catalog default + Anthropic API flagship (claude-opus-5-5)",
	},
	{
		Provider:      "vertex_anthropic",
		DefaultModel:  "claude-opus-5-5",
		UpstreamBasis: "Claude Code provider_alias_targets.opus.per_provider.vertex (claude-opus-5-5)",
	},
	{
		Provider:      "bedrock_anthropic",
		DefaultModel:  "claude-opus-5-5",
		UpstreamBasis: "Claude Code provider_alias_targets.opus.per_provider.bedrock (claude-opus-5-5)",
	},
	{
		Provider:          "grok",
		DefaultModel:      "grok-4.7",
		DefaultImageModel: "grok-imagine-image-2.0",
		UpstreamBasis:     "Grok CLI default_model (grok-4.7) + xAI Images API active model (grok-imagine-image-2.0)",
	},
	{
		Provider:          "gemini",
		DefaultModel:      "gemini-4-neon",
		DefaultImageModel: "gemini-nano-banana-2.1",
		UpstreamBasis:     "Antigravity/Gemini 4 balanced coding flagship (gemini-4-neon) + Google AI GA image model (gemini-nano-banana-2.1)",
	},
	{
		Provider:          "vertex",
		DefaultModel:      "gemini-4-neon",
		DefaultImageModel: "gemini-nano-banana-2.1",
		UpstreamBasis:     "Antigravity/Vertex Gemini 4 balanced coding flagship (gemini-4-neon) + Vertex AI GA image model (gemini-nano-banana-2.1)",
	},
	{
		Provider:          "vertex_interactions",
		DefaultModel:      "gemini-4-neon",
		DefaultImageModel: "gemini-nano-banana-2.1",
		UpstreamBasis:     "Vertex v1beta1 Interactions API Gemini 4 flagship (gemini-4-neon) + Vertex AI GA image model (gemini-nano-banana-2.1)",
	},
	{
		Provider:      "meta",
		DefaultModel:  "Llama-4-Maverick-17B-128E-Instruct-FP8",
		UpstreamBasis: "Meta Llama API latest official flagship (Llama-4-Maverick-17B-128E-Instruct-FP8)",
	},
	{
		Provider:      "openrouter",
		DefaultModel:  "openai/gpt-6.2-sol",
		UpstreamBasis: "OpenRouter latest OpenAI flagship route (openai/gpt-6.2-sol)",
	},
	{
		Provider:      "bedrock_openai",
		DefaultModel:  "openai.gpt-oss-120b",
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

var builtinModelSpecs = []ModelSpec{
	// OpenAI & Codex OAuth (aligned with Codex models.json effective windows and OpenAI API specs)
	{
		ID:                       "gpt-6.2-sol",
		Provider:                 "openai",
		ContextWindow:            500_000,
		MaxOutputTokens:          128_000,
		AutoCompactFraction:      0.90,
		AutoCompactTokenLimit:    450_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		InputCostPerMillion:      2.00,
		CachedCostPerMillion:     0.10,
		OutputCostPerMillion:     10.00,
	},
	{
		ID:                       "gpt-6.2-astra",
		Provider:                 "openai",
		ContextWindow:            272_000,
		MaxOutputTokens:          128_000,
		AutoCompactFraction:      0.90,
		AutoCompactTokenLimit:    244_800,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		InputCostPerMillion:      8.00,
		CachedCostPerMillion:     0.40,
		OutputCostPerMillion:     40.00,
	},
	{
		ID:                       "gpt-6.1-terra",
		Provider:                 "openai",
		ContextWindow:            500_000,
		MaxOutputTokens:          128_000,
		AutoCompactFraction:      0.90,
		AutoCompactTokenLimit:    450_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		InputCostPerMillion:      2.00,
		CachedCostPerMillion:     0.20,
		OutputCostPerMillion:     10.00,
	},
	{
		ID:                       "gpt-6.1-sol",
		Provider:                 "openai",
		ContextWindow:            500_000,
		MaxOutputTokens:          128_000,
		AutoCompactFraction:      0.90,
		AutoCompactTokenLimit:    450_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		InputCostPerMillion:      2.00,
		CachedCostPerMillion:     0.20,
		OutputCostPerMillion:     10.00,
	},
	{
		ID:                       "gpt-5.4",
		Provider:                 "openai",
		ContextWindow:            400_000,
		MaxOutputTokens:          32_768,
		AutoCompactFraction:      0.85,
		AutoCompactTokenLimit:    245_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		InputCostPerMillion:      2.50,
		CachedCostPerMillion:     0.25,
		OutputCostPerMillion:     10.00,
	},
	{
		ID:                       "gpt-5",
		Provider:                 "openai",
		ContextWindow:            400_000,
		MaxOutputTokens:          32_768,
		AutoCompactFraction:      0.85,
		AutoCompactTokenLimit:    245_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		InputCostPerMillion:      1.25,
		CachedCostPerMillion:     0.125,
		OutputCostPerMillion:     10.00,
	},
	{
		ID:                       "o3",
		Provider:                 "openai",
		ContextWindow:            200_000,
		MaxOutputTokens:          32_768,
		AutoCompactFraction:      0.85,
		AutoCompactTokenLimit:    170_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		InputCostPerMillion:      2.00,
		CachedCostPerMillion:     0.50,
		OutputCostPerMillion:     8.00,
	},
	// Anthropic (direct, Vertex Anthropic, Bedrock Anthropic)
	// 1M models intentionally compact around 380k (matching Claude Code 400k window override minus reserve).
	{
		ID:                       "claude-opus-5-5",
		Provider:                 "anthropic",
		ContextWindow:            1_000_000,
		MaxOutputTokens:          128_000,
		AutoCompactFraction:      0.82,
		AutoCompactTokenLimit:    380_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      4.00,
		CachedCostPerMillion:     0.20,
		OutputCostPerMillion:     20.00,
	},
	{
		ID:                       "claude-sonnet-5-5",
		Provider:                 "anthropic",
		ContextWindow:            1_000_000,
		MaxOutputTokens:          128_000,
		AutoCompactFraction:      0.82,
		AutoCompactTokenLimit:    380_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      2.00,
		CachedCostPerMillion:     0.10,
		OutputCostPerMillion:     10.00,
	},
	{
		ID:                       "claude-haiku-5-5",
		Provider:                 "anthropic",
		ContextWindow:            1_000_000,
		MaxOutputTokens:          128_000,
		AutoCompactFraction:      0.82,
		AutoCompactTokenLimit:    380_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      0.10,
		CachedCostPerMillion:     0.01,
		OutputCostPerMillion:     0.50,
	},
	{
		ID:                       "claude-fable-5",
		Provider:                 "anthropic",
		ContextWindow:            1_000_000,
		MaxOutputTokens:          128_000,
		AutoCompactFraction:      0.82,
		AutoCompactTokenLimit:    380_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      15.00,
		CachedCostPerMillion:     1.50,
		OutputCostPerMillion:     75.00,
	},
	{
		ID:                       "claude-opus-5",
		Provider:                 "anthropic",
		ContextWindow:            1_000_000,
		MaxOutputTokens:          128_000,
		AutoCompactFraction:      0.82,
		AutoCompactTokenLimit:    380_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      4.00,
		CachedCostPerMillion:     0.20,
		OutputCostPerMillion:     20.00,
	},
	{
		ID:                       "claude-sonnet-5",
		Provider:                 "anthropic",
		ContextWindow:            1_000_000,
		MaxOutputTokens:          128_000,
		AutoCompactFraction:      0.82,
		AutoCompactTokenLimit:    380_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      2.00,
		CachedCostPerMillion:     0.10,
		OutputCostPerMillion:     10.00,
	},
	{
		ID:                       "claude-opus-4-8",
		Provider:                 "anthropic",
		ContextWindow:            1_000_000,
		MaxOutputTokens:          128_000,
		AutoCompactFraction:      0.82,
		AutoCompactTokenLimit:    380_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      5.00,
		CachedCostPerMillion:     0.50,
		OutputCostPerMillion:     25.00,
	},
	{
		ID:                       "claude-opus-4-7",
		Provider:                 "anthropic",
		ContextWindow:            1_000_000,
		MaxOutputTokens:          128_000,
		AutoCompactFraction:      0.82,
		AutoCompactTokenLimit:    380_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      5.00,
		CachedCostPerMillion:     0.50,
		OutputCostPerMillion:     25.00,
	},
	{
		ID:                       "claude-opus-4-6",
		Provider:                 "anthropic",
		ContextWindow:            1_000_000,
		MaxOutputTokens:          128_000,
		AutoCompactFraction:      0.82,
		AutoCompactTokenLimit:    380_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      5.00,
		CachedCostPerMillion:     0.50,
		OutputCostPerMillion:     25.00,
	},
	{
		ID:                       "claude-sonnet-4-6",
		Provider:                 "anthropic",
		ContextWindow:            1_000_000,
		MaxOutputTokens:          128_000,
		AutoCompactFraction:      0.82,
		AutoCompactTokenLimit:    380_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      3.00,
		CachedCostPerMillion:     0.30,
		OutputCostPerMillion:     15.00,
	},
	{
		ID:                       "claude-opus-4-5",
		Provider:                 "anthropic",
		ContextWindow:            1_000_000,
		MaxOutputTokens:          32_000,
		AutoCompactFraction:      0.82,
		AutoCompactTokenLimit:    380_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      15.00,
		CachedCostPerMillion:     1.50,
		OutputCostPerMillion:     75.00,
	},
	{
		ID:                       "claude-opus-4-1",
		Provider:                 "anthropic",
		ContextWindow:            200_000,
		MaxOutputTokens:          32_000,
		AutoCompactFraction:      0.82,
		AutoCompactTokenLimit:    165_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      15.00,
		CachedCostPerMillion:     1.50,
		OutputCostPerMillion:     75.00,
	},
	{
		ID:                       "claude-sonnet-4-5",
		Provider:                 "anthropic",
		ContextWindow:            1_000_000,
		MaxOutputTokens:          16_384,
		AutoCompactFraction:      0.82,
		AutoCompactTokenLimit:    380_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      3.00,
		CachedCostPerMillion:     0.30,
		OutputCostPerMillion:     15.00,
	},
	{
		ID:                       "claude-haiku-4-5",
		Provider:                 "anthropic",
		ContextWindow:            200_000,
		MaxOutputTokens:          64_000,
		AutoCompactFraction:      0.85,
		AutoCompactTokenLimit:    165_000,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      1.00,
		CachedCostPerMillion:     0.10,
		OutputCostPerMillion:     5.00,
	},
	// Google Gemini (Vertex AI & Gemini API)
	// 1M models compact at 400k by default to preserve reasoning sharpness and cost efficiency.
	{
		ID:                    "gemini-4-carbon",
		Provider:              "gemini",
		ContextWindow:         1_048_576,
		MaxOutputTokens:       65_536,
		AutoCompactFraction:   0.80,
		AutoCompactTokenLimit: 400_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   3.00,
		CachedCostPerMillion:  0.30,
		OutputCostPerMillion:  15.00,
	},
	{
		ID:                    "gemini-4-neon",
		Provider:              "gemini",
		ContextWindow:         1_048_576,
		MaxOutputTokens:       65_536,
		AutoCompactFraction:   0.80,
		AutoCompactTokenLimit: 400_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   3.00,
		CachedCostPerMillion:  0.30,
		OutputCostPerMillion:  15.00,
	},
	{
		ID:                    "gemini-4-hydrogen",
		Provider:              "gemini",
		ContextWindow:         1_048_576,
		MaxOutputTokens:       65_536,
		AutoCompactFraction:   0.80,
		AutoCompactTokenLimit: 400_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   1.50,
		CachedCostPerMillion:  0.15,
		OutputCostPerMillion:  7.50,
	},
	{
		ID:                    "gemini-4-argon",
		Provider:              "gemini",
		ContextWindow:         1_048_576,
		MaxOutputTokens:       65_536,
		AutoCompactFraction:   0.80,
		AutoCompactTokenLimit: 400_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   3.00,
		CachedCostPerMillion:  0.30,
		OutputCostPerMillion:  15.00,
	},
	{
		ID:                    "gemini-3.5-flash-lite",
		Provider:              "gemini",
		ContextWindow:         1_048_576,
		MaxOutputTokens:       65_536,
		AutoCompactFraction:   0.80,
		AutoCompactTokenLimit: 400_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   0.30,
		CachedCostPerMillion:  0.03,
		OutputCostPerMillion:  2.50,
	},
	{
		ID:                    "gemini-3.1-pro",
		Provider:              "gemini",
		ContextWindow:         1_048_576,
		MaxOutputTokens:       65_536,
		AutoCompactFraction:   0.80,
		AutoCompactTokenLimit: 400_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   2.00,
		CachedCostPerMillion:  0.20,
		OutputCostPerMillion:  12.00,
	},
	{
		ID:                    "gemini-2.5-pro",
		Provider:              "gemini",
		ContextWindow:         1_048_576,
		MaxOutputTokens:       65_536,
		AutoCompactFraction:   0.80,
		AutoCompactTokenLimit: 400_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   1.25,
		CachedCostPerMillion:  0.31,
		OutputCostPerMillion:  10.00,
	},
	{
		ID:                    "gemini-2.5-flash",
		Provider:              "gemini",
		ContextWindow:         1_048_576,
		MaxOutputTokens:       65_536,
		AutoCompactFraction:   0.80,
		AutoCompactTokenLimit: 400_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   0.30,
		CachedCostPerMillion:  0.075,
		OutputCostPerMillion:  2.50,
	},
	// xAI / Grok (Grok CLI defaults to 85% of context window; 1M/2M models capped at 400k by default)
	{
		ID:                    "grok-4.7",
		Provider:              "grok",
		ContextWindow:         500_000,
		MaxOutputTokens:       32_768,
		AutoCompactFraction:   0.85,
		AutoCompactTokenLimit: 425_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   2.00,
		CachedCostPerMillion:  0.50,
		OutputCostPerMillion:  6.00,
	},
	{
		ID:                    "grok-4.6",
		Provider:              "grok",
		ContextWindow:         500_000,
		MaxOutputTokens:       32_768,
		AutoCompactFraction:   0.85,
		AutoCompactTokenLimit: 425_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   2.00,
		CachedCostPerMillion:  0.50,
		OutputCostPerMillion:  6.00,
	},
	{
		ID:                    "grok-4.5",
		Provider:              "grok",
		ContextWindow:         500_000,
		MaxOutputTokens:       32_768,
		AutoCompactFraction:   0.85,
		AutoCompactTokenLimit: 425_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   2.00,
		CachedCostPerMillion:  0.50,
		OutputCostPerMillion:  6.00,
	},
	{
		ID:                    "grok-4.3",
		Provider:              "grok",
		ContextWindow:         1_000_000,
		MaxOutputTokens:       32_768,
		AutoCompactFraction:   0.85,
		AutoCompactTokenLimit: 400_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   1.25,
		CachedCostPerMillion:  0.31,
		OutputCostPerMillion:  5.00,
	},
	{
		ID:                    "grok-4.20",
		Provider:              "grok",
		ContextWindow:         1_000_000,
		MaxOutputTokens:       32_768,
		AutoCompactFraction:   0.85,
		AutoCompactTokenLimit: 400_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   2.00,
		CachedCostPerMillion:  0.50,
		OutputCostPerMillion:  6.00,
	},
	{
		ID:                    "grok-build-0.1",
		Provider:              "grok",
		ContextWindow:         256_000,
		MaxOutputTokens:       32_768,
		AutoCompactFraction:   0.85,
		AutoCompactTokenLimit: 217_600,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   1.00,
		CachedCostPerMillion:  0.20,
		OutputCostPerMillion:  2.00,
	},
	{
		ID:                    "grok-4",
		Provider:              "grok",
		ContextWindow:         256_000,
		MaxOutputTokens:       32_768,
		AutoCompactFraction:   0.85,
		AutoCompactTokenLimit: 217_600,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   3.00,
		CachedCostPerMillion:  0.75,
		OutputCostPerMillion:  15.00,
	},
	{
		ID:                    "grok-4-fast",
		Provider:              "grok",
		ContextWindow:         2_000_000,
		MaxOutputTokens:       32_768,
		AutoCompactFraction:   0.85,
		AutoCompactTokenLimit: 400_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   0.20,
		CachedCostPerMillion:  0.05,
		OutputCostPerMillion:  0.50,
	},
	{
		ID:                    "grok-code",
		Provider:              "grok",
		ContextWindow:         256_000,
		MaxOutputTokens:       32_768,
		AutoCompactFraction:   0.85,
		AutoCompactTokenLimit: 217_600,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   0.20,
		CachedCostPerMillion:  0.05,
		OutputCostPerMillion:  1.50,
	},
	// Meta Llama (1M Llama 4 models capped at 300k by default for high recall)
	{
		ID:                    "Llama-4-Maverick-17B-128E-Instruct-FP8",
		Provider:              "meta",
		ContextWindow:         1_000_000,
		MaxOutputTokens:       16_384,
		AutoCompactFraction:   0.82,
		AutoCompactTokenLimit: 300_000,
		SupportsReasoning:     false,
		SupportsPromptCache:   false,
		InputCostPerMillion:   0.27,
		CachedCostPerMillion:  0.27,
		OutputCostPerMillion:  0.85,
	},
	{
		ID:                    "Llama-4-Scout-17B-16E-Instruct",
		Provider:              "meta",
		ContextWindow:         1_000_000,
		MaxOutputTokens:       16_384,
		AutoCompactFraction:   0.82,
		AutoCompactTokenLimit: 300_000,
		SupportsReasoning:     false,
		SupportsPromptCache:   false,
		InputCostPerMillion:   0.15,
		CachedCostPerMillion:  0.15,
		OutputCostPerMillion:  0.60,
	},
	{
		ID:                    "Llama-3.3-70B-Instruct",
		Provider:              "meta",
		ContextWindow:         128_000,
		MaxOutputTokens:       8_192,
		AutoCompactFraction:   0.80,
		AutoCompactTokenLimit: 100_000,
		SupportsReasoning:     false,
		SupportsPromptCache:   false,
		InputCostPerMillion:   0.40,
		CachedCostPerMillion:  0.40,
		OutputCostPerMillion:  0.60,
	},
}

func BuiltinModels() []ModelSpec {
	out := make([]ModelSpec, len(builtinModelSpecs))
	copy(out, builtinModelSpecs)
	return out
}

// LookupModel resolves true per-model context window and capabilities across
// direct IDs, provider-prefixed IDs (google/, anthropic/, meta-llama/, us.anthropic.),
// and dated suffixes.
func LookupModel(providerName, modelID string) ModelSpec {
	norm := NormalizeModelSlug(modelID)
	bestIdx := -1
	bestLen := -1
	for i, spec := range builtinModelSpecs {
		specNorm := NormalizeModelSlug(spec.ID)
		if norm == specNorm {
			bestIdx = i
			break
		}
		if strings.HasPrefix(norm, specNorm+"-") || strings.HasPrefix(norm, specNorm+"@") {
			if len(specNorm) > bestLen {
				bestLen = len(specNorm)
				bestIdx = i
			}
		}
	}
	if bestIdx >= 0 {
		out := builtinModelSpecs[bestIdx]
		out.ID = modelID
		if providerName != "" {
			out.Provider = providerName
		}
		return out
	}
	return fallbackModelSpec(providerName, modelID, norm)
}

// NormalizeModelSlug strips provider routing prefixes and Bedrock/Vertex
// version suffixes so model capability and threshold overrides match cleanly.
func NormalizeModelSlug(raw string) string {
	s := strings.ToLower(strings.TrimSpace(raw))
	if idx := strings.LastIndex(s, "/"); idx >= 0 {
		s = s[idx+1:]
	}
	for _, prefix := range []string{"us.anthropic.", "eu.anthropic.", "global.anthropic.", "anthropic.", "meta.", "us.meta."} {
		if strings.HasPrefix(s, prefix) {
			s = strings.TrimPrefix(s, prefix)
		}
	}
	if idx := strings.Index(s, ":0"); idx > 0 {
		s = s[:idx]
	}
	if idx := strings.Index(s, "-v1"); idx > 0 && strings.Contains(s, "claude") {
		s = s[:idx]
	}
	if idx := strings.Index(s, "["); idx > 0 {
		s = s[:idx]
	}
	return s
}

func fallbackModelSpec(providerName, modelID, norm string) ModelSpec {
	p := strings.ToLower(strings.TrimSpace(providerName))
	switch {
	case strings.Contains(norm, "claude") || strings.Contains(norm, "fable") || strings.Contains(p, "anthropic"):
		return ModelSpec{
			ID:                       modelID,
			Provider:                 providerName,
			ContextWindow:            200_000,
			MaxOutputTokens:          16_384,
			AutoCompactFraction:      0.82,
			AutoCompactTokenLimit:    165_000,
			SupportsReasoning:        true,
			SupportsPromptCache:      true,
			SupportsRemoteCompaction: true,
			SupportsNativeToolSearch: true,
			InputCostPerMillion:      3.00,
			CachedCostPerMillion:     0.30,
			OutputCostPerMillion:     15.00,
		}
	case strings.Contains(norm, "gemini") || strings.Contains(p, "gemini") || p == "vertex":
		return ModelSpec{
			ID:                    modelID,
			Provider:              providerName,
			ContextWindow:         1_048_576,
			MaxOutputTokens:       65_536,
			AutoCompactFraction:   0.80,
			AutoCompactTokenLimit: 400_000,
			SupportsReasoning:     true,
			SupportsPromptCache:   true,
			InputCostPerMillion:   1.25,
			CachedCostPerMillion:  0.31,
			OutputCostPerMillion:  10.00,
		}
	case strings.Contains(norm, "grok") || p == "grok" || p == "xai":
		return ModelSpec{
			ID:                    modelID,
			Provider:              providerName,
			ContextWindow:         256_000,
			MaxOutputTokens:       32_768,
			AutoCompactFraction:   0.85,
			AutoCompactTokenLimit: 217_600,
			SupportsReasoning:     true,
			SupportsPromptCache:   true,
			InputCostPerMillion:   3.00,
			CachedCostPerMillion:  0.75,
			OutputCostPerMillion:  15.00,
		}
	case strings.Contains(norm, "llama") || p == "meta":
		ctxWin := int64(128_000)
		limit := int64(100_000)
		if strings.Contains(norm, "llama-4") {
			ctxWin = 1_000_000
			limit = 300_000
		}
		return ModelSpec{
			ID:                    modelID,
			Provider:              providerName,
			ContextWindow:         ctxWin,
			MaxOutputTokens:       16_384,
			AutoCompactFraction:   0.82,
			AutoCompactTokenLimit: limit,
			InputCostPerMillion:   0.30,
			CachedCostPerMillion:  0.30,
			OutputCostPerMillion:  0.80,
		}
	default:
		return ModelSpec{
			ID:                       modelID,
			Provider:                 providerName,
			ContextWindow:            200_000,
			MaxOutputTokens:          16_384,
			AutoCompactFraction:      0.85,
			AutoCompactTokenLimit:    170_000,
			SupportsReasoning:        true,
			SupportsPromptCache:      true,
			SupportsRemoteCompaction: p == "openai" || p == "codex_oauth",
			InputCostPerMillion:      2.50,
			CachedCostPerMillion:     0.25,
			OutputCostPerMillion:     10.00,
		}
	}
}
