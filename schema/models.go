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

func openaiSpec(id string, ctx int64, maxOut int, frac float64, limit int64, inCost, cachedCost, outCost float64) ModelSpec {
	return ModelSpec{
		ID:                       id,
		Provider:                 "openai",
		ContextWindow:            ctx,
		MaxOutputTokens:          maxOut,
		AutoCompactFraction:      frac,
		AutoCompactTokenLimit:    limit,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		InputCostPerMillion:      inCost,
		CachedCostPerMillion:     cachedCost,
		OutputCostPerMillion:     outCost,
	}
}

func anthropicSpec(id string, ctx int64, maxOut int, frac float64, limit int64, inCost, cachedCost, outCost float64) ModelSpec {
	return ModelSpec{
		ID:                       id,
		Provider:                 "anthropic",
		ContextWindow:            ctx,
		MaxOutputTokens:          maxOut,
		AutoCompactFraction:      frac,
		AutoCompactTokenLimit:    limit,
		SupportsReasoning:        true,
		SupportsPromptCache:      true,
		SupportsRemoteCompaction: true,
		SupportsNativeToolSearch: true,
		InputCostPerMillion:      inCost,
		CachedCostPerMillion:     cachedCost,
		OutputCostPerMillion:     outCost,
	}
}

func geminiSpec(id string, inCost, cachedCost, outCost float64) ModelSpec {
	return ModelSpec{
		ID:                    id,
		Provider:              "gemini",
		ContextWindow:         1_048_576,
		MaxOutputTokens:       65_536,
		AutoCompactFraction:   0.80,
		AutoCompactTokenLimit: 400_000,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   inCost,
		CachedCostPerMillion:  cachedCost,
		OutputCostPerMillion:  outCost,
	}
}

func grokSpec(id string, ctx, limit int64, inCost, cachedCost, outCost float64) ModelSpec {
	return ModelSpec{
		ID:                    id,
		Provider:              "grok",
		ContextWindow:         ctx,
		MaxOutputTokens:       32_768,
		AutoCompactFraction:   0.85,
		AutoCompactTokenLimit: limit,
		SupportsReasoning:     true,
		SupportsPromptCache:   true,
		InputCostPerMillion:   inCost,
		CachedCostPerMillion:  cachedCost,
		OutputCostPerMillion:  outCost,
	}
}

func llamaSpec(id string, ctx int64, maxOut int, frac float64, limit int64, inCost, outCost float64) ModelSpec {
	return ModelSpec{
		ID:                    id,
		Provider:              "meta",
		ContextWindow:         ctx,
		MaxOutputTokens:       maxOut,
		AutoCompactFraction:   frac,
		AutoCompactTokenLimit: limit,
		InputCostPerMillion:   inCost,
		CachedCostPerMillion:  inCost,
		OutputCostPerMillion:  outCost,
	}
}

var builtinModelSpecs = []ModelSpec{
	// OpenAI & Codex OAuth (aligned with Codex models.json effective windows and OpenAI API specs)
	openaiSpec("gpt-6.2-sol", 500_000, 128_000, 0.90, 450_000, 2.00, 0.10, 10.00),
	openaiSpec("gpt-6.2-astra", 272_000, 128_000, 0.90, 244_800, 8.00, 0.40, 40.00),
	openaiSpec("gpt-6.1-terra", 500_000, 128_000, 0.90, 450_000, 2.00, 0.20, 10.00),
	openaiSpec("gpt-6.1-sol", 500_000, 128_000, 0.90, 450_000, 2.00, 0.20, 10.00),
	openaiSpec("gpt-5.4", 400_000, 32_768, 0.85, 245_000, 2.50, 0.25, 10.00),
	openaiSpec("gpt-5", 400_000, 32_768, 0.85, 245_000, 1.25, 0.125, 10.00),
	openaiSpec("o3", 200_000, 32_768, 0.85, 170_000, 2.00, 0.50, 8.00),

	// Anthropic (direct, Vertex Anthropic, Bedrock Anthropic)
	// 1M models intentionally compact around 380k (matching Claude Code 400k window override minus reserve).
	anthropicSpec("claude-opus-5-5", 1_000_000, 128_000, 0.82, 380_000, 4.00, 0.20, 20.00),
	anthropicSpec("claude-sonnet-5-5", 1_000_000, 128_000, 0.82, 380_000, 2.00, 0.10, 10.00),
	anthropicSpec("claude-haiku-5-5", 1_000_000, 128_000, 0.82, 380_000, 0.10, 0.01, 0.50),
	anthropicSpec("claude-fable-5", 1_000_000, 128_000, 0.82, 380_000, 15.00, 1.50, 75.00),
	anthropicSpec("claude-opus-5", 1_000_000, 128_000, 0.82, 380_000, 4.00, 0.20, 20.00),
	anthropicSpec("claude-sonnet-5", 1_000_000, 128_000, 0.82, 380_000, 2.00, 0.10, 10.00),
	anthropicSpec("claude-opus-4-8", 1_000_000, 128_000, 0.82, 380_000, 5.00, 0.50, 25.00),
	anthropicSpec("claude-opus-4-7", 1_000_000, 128_000, 0.82, 380_000, 5.00, 0.50, 25.00),
	anthropicSpec("claude-opus-4-6", 1_000_000, 128_000, 0.82, 380_000, 5.00, 0.50, 25.00),
	anthropicSpec("claude-sonnet-4-6", 1_000_000, 128_000, 0.82, 380_000, 3.00, 0.30, 15.00),
	anthropicSpec("claude-opus-4-5", 1_000_000, 32_000, 0.82, 380_000, 15.00, 1.50, 75.00),
	anthropicSpec("claude-opus-4-1", 200_000, 32_000, 0.82, 165_000, 15.00, 1.50, 75.00),
	anthropicSpec("claude-sonnet-4-5", 1_000_000, 16_384, 0.82, 380_000, 3.00, 0.30, 15.00),
	anthropicSpec("claude-haiku-4-5", 200_000, 64_000, 0.85, 165_000, 1.00, 0.10, 5.00),

	// Google Gemini (Vertex AI & Gemini API; 1M models compact at 400k by default)
	geminiSpec("gemini-4-carbon", 3.00, 0.30, 15.00),
	geminiSpec("gemini-4-neon", 3.00, 0.30, 15.00),
	geminiSpec("gemini-4-hydrogen", 1.50, 0.15, 7.50),
	geminiSpec("gemini-4-argon", 3.00, 0.30, 15.00),
	geminiSpec("gemini-3.5-flash-lite", 0.30, 0.03, 2.50),
	geminiSpec("gemini-3.1-pro", 2.00, 0.20, 12.00),
	geminiSpec("gemini-2.5-pro", 1.25, 0.31, 10.00),
	geminiSpec("gemini-2.5-flash", 0.30, 0.075, 2.50),

	// xAI / Grok (Grok CLI defaults to 85% of context window; 1M/2M models capped at 400k by default)
	grokSpec("grok-4.7", 500_000, 425_000, 2.00, 0.50, 6.00),
	grokSpec("grok-4.6", 500_000, 425_000, 2.00, 0.50, 6.00),
	grokSpec("grok-4.5", 500_000, 425_000, 2.00, 0.50, 6.00),
	grokSpec("grok-4.3", 1_000_000, 400_000, 1.25, 0.31, 5.00),
	grokSpec("grok-4.20", 1_000_000, 400_000, 2.00, 0.50, 6.00),
	grokSpec("grok-build-0.1", 256_000, 217_600, 1.00, 0.20, 2.00),
	grokSpec("grok-4", 256_000, 217_600, 3.00, 0.75, 15.00),
	grokSpec("grok-4-fast", 2_000_000, 400_000, 0.20, 0.05, 0.50),
	grokSpec("grok-code", 256_000, 217_600, 0.20, 0.05, 1.50),

	// Meta Llama (1M Llama 4 models capped at 300k by default for high recall)
	llamaSpec("Llama-4-Maverick-17B-128E-Instruct-FP8", 1_000_000, 16_384, 0.82, 300_000, 0.27, 0.85),
	llamaSpec("Llama-4-Scout-17B-16E-Instruct", 1_000_000, 16_384, 0.82, 300_000, 0.15, 0.60),
	llamaSpec("Llama-3.3-70B-Instruct", 128_000, 8_192, 0.80, 100_000, 0.40, 0.60),
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
		spec := anthropicSpec(modelID, 200_000, 16_384, 0.82, 165_000, 3.00, 0.30, 15.00)
		spec.Provider = providerName
		return spec
	case strings.Contains(norm, "gemini") || strings.Contains(p, "gemini") || p == "vertex":
		spec := geminiSpec(modelID, 1.25, 0.31, 10.00)
		spec.Provider = providerName
		return spec
	case strings.Contains(norm, "grok") || p == "grok" || p == "xai":
		spec := grokSpec(modelID, 256_000, 217_600, 3.00, 0.75, 15.00)
		spec.Provider = providerName
		return spec
	case strings.Contains(norm, "llama") || p == "meta":
		ctxWin := int64(128_000)
		limit := int64(100_000)
		if strings.Contains(norm, "llama-4") {
			ctxWin = 1_000_000
			limit = 300_000
		}
		spec := llamaSpec(modelID, ctxWin, 16_384, 0.82, limit, 0.30, 0.80)
		spec.Provider = providerName
		return spec
	default:
		spec := openaiSpec(modelID, 200_000, 16_384, 0.85, 170_000, 2.50, 0.25, 10.00)
		spec.Provider = providerName
		spec.SupportsRemoteCompaction = p == "openai" || p == "codex_oauth"
		return spec
	}
}
