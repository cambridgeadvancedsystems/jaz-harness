package schema

import "strings"

type ModelSpec struct {
	ID                       string  `json:"id"`
	Provider                 string  `json:"provider"`
	ContextWindow            int64   `json:"context_window"`
	MaxOutputTokens          int     `json:"max_output_tokens"`
	AutoCompactFraction      float64 `json:"auto_compact_fraction"`
	SupportsReasoning        bool    `json:"supports_reasoning"`
	SupportsPromptCache      bool    `json:"supports_prompt_cache"`
	SupportsRemoteCompaction bool    `json:"supports_remote_compaction"`
	SupportsNativeToolSearch bool    `json:"supports_native_tool_search"`
	InputCostPerMillion      float64 `json:"input_cost_per_million"`
	CachedCostPerMillion     float64 `json:"cached_cost_per_million"`
	OutputCostPerMillion     float64 `json:"output_cost_per_million"`
}

func (m ModelSpec) AutoCompactThreshold() int64 {
	frac := m.AutoCompactFraction
	if frac <= 0 || frac >= 1 {
		frac = 0.85
	}
	return int64(float64(m.ContextWindow) * frac)
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

var builtinModelSpecs = []ModelSpec{
	// OpenAI & Codex OAuth
	{
		ID:                       "gpt-6.1-sol",
		Provider:                 "openai",
		ContextWindow:            400_000,
		MaxOutputTokens:          32_768,
		AutoCompactFraction:      0.85,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		InputCostPerMillion:      2.50,
		CachedCostPerMillion:     0.25,
		OutputCostPerMillion:     10.00,
	},
	{
		ID:                       "gpt-5.4",
		Provider:                 "openai",
		ContextWindow:            400_000,
		MaxOutputTokens:          32_768,
		AutoCompactFraction:      0.85,
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
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		InputCostPerMillion:      2.00,
		CachedCostPerMillion:     0.50,
		OutputCostPerMillion:     8.00,
	},
	// Anthropic (direct, Vertex Anthropic, Bedrock Anthropic)
	{
		ID:                       "claude-opus-4-5",
		Provider:                 "anthropic",
		ContextWindow:            1_000_000,
		MaxOutputTokens:          32_000,
		AutoCompactFraction:      0.82,
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
		MaxOutputTokens:          16_384,
		AutoCompactFraction:      0.85,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      1.00,
		CachedCostPerMillion:     0.10,
		OutputCostPerMillion:     5.00,
	},
	// Google Gemini (Vertex AI & Gemini API)
	{
		ID:                   "gemini-4-neon",
		Provider:             "gemini",
		ContextWindow:        1_048_576,
		MaxOutputTokens:      65_536,
		AutoCompactFraction:  0.80,
		SupportsReasoning:    true,
		SupportsPromptCache:  true,
		InputCostPerMillion:  2.50,
		CachedCostPerMillion: 0.25,
		OutputCostPerMillion: 15.00,
	},
	{
		ID:                   "gemini-4-hydrogen",
		Provider:             "gemini",
		ContextWindow:        1_048_576,
		MaxOutputTokens:      65_536,
		AutoCompactFraction:  0.80,
		SupportsReasoning:    true,
		SupportsPromptCache:  true,
		InputCostPerMillion:  0.50,
		CachedCostPerMillion: 0.05,
		OutputCostPerMillion: 3.00,
	},
	{
		ID:                   "gemini-2.5-pro",
		Provider:             "gemini",
		ContextWindow:        1_048_576,
		MaxOutputTokens:      65_536,
		AutoCompactFraction:  0.80,
		SupportsReasoning:    true,
		SupportsPromptCache:  true,
		InputCostPerMillion:  1.25,
		CachedCostPerMillion: 0.31,
		OutputCostPerMillion: 10.00,
	},
	{
		ID:                   "gemini-2.5-flash",
		Provider:             "gemini",
		ContextWindow:        1_048_576,
		MaxOutputTokens:      65_536,
		AutoCompactFraction:  0.80,
		SupportsReasoning:    true,
		SupportsPromptCache:  true,
		InputCostPerMillion:  0.30,
		CachedCostPerMillion: 0.075,
		OutputCostPerMillion: 2.50,
	},
	// xAI / Grok
	{
		ID:                   "grok-4",
		Provider:             "grok",
		ContextWindow:        256_000,
		MaxOutputTokens:      32_768,
		AutoCompactFraction:  0.85,
		SupportsReasoning:    true,
		SupportsPromptCache:  true,
		InputCostPerMillion:  3.00,
		CachedCostPerMillion: 0.75,
		OutputCostPerMillion: 15.00,
	},
	{
		ID:                   "grok-4-fast",
		Provider:             "grok",
		ContextWindow:        2_000_000,
		MaxOutputTokens:      32_768,
		AutoCompactFraction:  0.80,
		SupportsReasoning:    true,
		SupportsPromptCache:  true,
		InputCostPerMillion:  0.20,
		CachedCostPerMillion: 0.05,
		OutputCostPerMillion: 0.50,
	},
	{
		ID:                   "grok-code",
		Provider:             "grok",
		ContextWindow:        256_000,
		MaxOutputTokens:      32_768,
		AutoCompactFraction:  0.85,
		SupportsReasoning:    true,
		SupportsPromptCache:  true,
		InputCostPerMillion:  0.20,
		CachedCostPerMillion: 0.05,
		OutputCostPerMillion: 1.50,
	},
	// Meta Llama
	{
		ID:                   "Llama-4-Maverick-17B-128E-Instruct-FP8",
		Provider:             "meta",
		ContextWindow:        1_000_000,
		MaxOutputTokens:      16_384,
		AutoCompactFraction:  0.82,
		SupportsReasoning:    false,
		SupportsPromptCache:  false,
		InputCostPerMillion:  0.27,
		CachedCostPerMillion: 0.27,
		OutputCostPerMillion: 0.85,
	},
	{
		ID:                   "Llama-4-Scout-17B-16E-Instruct",
		Provider:             "meta",
		ContextWindow:        1_000_000,
		MaxOutputTokens:      16_384,
		AutoCompactFraction:  0.82,
		SupportsReasoning:    false,
		SupportsPromptCache:  false,
		InputCostPerMillion:  0.15,
		CachedCostPerMillion: 0.15,
		OutputCostPerMillion: 0.60,
	},
	{
		ID:                   "Llama-3.3-70B-Instruct",
		Provider:             "meta",
		ContextWindow:        128_000,
		MaxOutputTokens:      8_192,
		AutoCompactFraction:  0.85,
		SupportsReasoning:    false,
		SupportsPromptCache:  false,
		InputCostPerMillion:  0.40,
		CachedCostPerMillion: 0.40,
		OutputCostPerMillion: 0.60,
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
	norm := normalizeModelSlug(modelID)
	for _, spec := range builtinModelSpecs {
		specNorm := normalizeModelSlug(spec.ID)
		if norm == specNorm || strings.HasPrefix(norm, specNorm+"-") || strings.HasPrefix(norm, specNorm+"@") {
			out := spec
			out.ID = modelID
			if providerName != "" {
				out.Provider = providerName
			}
			return out
		}
	}
	return fallbackModelSpec(providerName, modelID, norm)
}

func normalizeModelSlug(raw string) string {
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
	return s
}

func fallbackModelSpec(providerName, modelID, norm string) ModelSpec {
	p := strings.ToLower(strings.TrimSpace(providerName))
	switch {
	case strings.Contains(norm, "claude") || strings.Contains(p, "anthropic"):
		return ModelSpec{
			ID:                       modelID,
			Provider:                 providerName,
			ContextWindow:            200_000,
			MaxOutputTokens:          16_384,
			AutoCompactFraction:      0.82,
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
			ID:                   modelID,
			Provider:             providerName,
			ContextWindow:        1_048_576,
			MaxOutputTokens:      65_536,
			AutoCompactFraction:  0.80,
			SupportsReasoning:    true,
			SupportsPromptCache:  true,
			InputCostPerMillion:  1.25,
			CachedCostPerMillion: 0.31,
			OutputCostPerMillion: 10.00,
		}
	case strings.Contains(norm, "grok") || p == "grok" || p == "xai":
		return ModelSpec{
			ID:                   modelID,
			Provider:             providerName,
			ContextWindow:        256_000,
			MaxOutputTokens:      32_768,
			AutoCompactFraction:  0.85,
			SupportsReasoning:    true,
			SupportsPromptCache:  true,
			InputCostPerMillion:  3.00,
			CachedCostPerMillion: 0.75,
			OutputCostPerMillion: 15.00,
		}
	case strings.Contains(norm, "llama") || p == "meta":
		ctxWin := int64(128_000)
		if strings.Contains(norm, "llama-4") {
			ctxWin = 1_000_000
		}
		return ModelSpec{
			ID:                   modelID,
			Provider:             providerName,
			ContextWindow:        ctxWin,
			MaxOutputTokens:      16_384,
			AutoCompactFraction:  0.82,
			InputCostPerMillion:  0.30,
			CachedCostPerMillion: 0.30,
			OutputCostPerMillion: 0.80,
		}
	default:
		return ModelSpec{
			ID:                       modelID,
			Provider:                 providerName,
			ContextWindow:            200_000,
			MaxOutputTokens:          16_384,
			AutoCompactFraction:      0.85,
			SupportsReasoning:        true,
			SupportsPromptCache:      true,
			SupportsRemoteCompaction: p == "openai" || p == "codex_oauth",
			InputCostPerMillion:      2.50,
			CachedCostPerMillion:     0.25,
			OutputCostPerMillion:     10.00,
		}
	}
}
