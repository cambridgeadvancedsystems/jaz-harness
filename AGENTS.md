# jaz-harness Engineering & Compaction Guide

Standing project rules and maintained runtime recipes for `jaz-harness`.

## Upstream Parity Rule (Always Re-Check Before Changing Harness Behavior)

Whenever modifying wire protocols, compaction thresholds, prompt caching, tool calling, streaming codecs, or model metadata in `jaz-harness`, **re-check the current upstream coding agents first** rather than relying on stale assumptions:

- **Codex** (`codex-rs`: `models.json`, `openai_models.rs`, `client.rs`, `compact_remote.rs`) — sticky routing (`x-codex-turn-state`), `prompt_cache_key`, `session_id`, `/responses/compact` (`configuration_update` handling), `context_management`, and `auto_compact_token_limit`.
- **Claude Code** (`claude` CLI / Messages API) — `compact-2026-01-12` / `compact_20260112`, `compaction_delta`, `stop_reason: "compaction"`, `usage.iterations`, `cache_control` 4-breakpoint cap, `tool_search_tool_bm25_20251119` + `defer_loading`, `x-should-retry: false`, and `CLAUDE_CODE_AUTO_COMPACT_WINDOW`.
- **Grok CLI** (`grok`) — native `/v1/responses`, `x-grok-conv-id`, `x_search`/`web_search`/`code_interpreter`, invalid tool-arg JSON sanitization, and `auto_compact_threshold_percent`.
- **Antigravity / Gemini Vertex `v1beta1`** (`agy` / Vertex AI) — `:streamGenerateContent` (`toolConfig.includeServerSideToolInvocations: true`) and stateless `Interactions API` (`store: false`), Gemini 4 `XHIGH` / no `MINIMAL` / no `temperature`/`topP`/`cachedContent`, first-parallel `thoughtSignature` placement, and `continuation_token` loops.

Any intentional divergence from upstream defaults (such as lower quality-preserving compaction limits on 1M/2M-window models) must be explicit, documented below, and overridable in config.

## Maintained Per-Model Compaction-Threshold Recipe

1M/2M raw context windows suffer noticeable quality degradation ("context rot") and cost spikes long before hitting 85–90% of nominal capacity. `schema/models.go` therefore sets explicit quality-tuned `AutoCompactTokenLimit` defaults aligned with Jaz's overrides and upstream coding agents:

| Model Family | Raw Context Window | Upstream Reference | `jaz-harness` Default Trigger | Notes |
| --- | --- | --- | --- | --- |
| `gpt-6.2-sol`, `gpt-6.1-sol` | 500k (872k max) | Codex `models.json`: `context_window=500k`, 90% auto-compact | **450,000** | Uses native `/responses/compact` + readable cross-provider fallback digest |
| `gpt-6.2-astra` | 272k (872k max) | Codex `models.json`: `context_window=272k`, 90% = 244,800 | **244,800** | Matches Codex effective working window |
| `gpt-5.4`, `gpt-5` | 400k | Codex ~245k–250k working-window cap | **245,000** | Keeps multi-step coding accuracy sharp well below 400k |
| `o3` | 200k | 85% of 200k | **170,000** | Native `/responses/compact` supported |
| `claude-opus-5-5`, `claude-fable-5*`, `claude-opus-4-5`, `claude-sonnet-4-5` | 1,000k (1M) | Jaz `CLAUDE_CODE_AUTO_COMPACT_WINDOW=400000` minus reserve (Anthropic API default: 150k) | **380,000** | Intentionally well below 1M; supports Anthropic `compact-2026-01-12` (`compact_20260112`) |
| `claude-opus-4-1`, `claude-haiku-4-5` | 200k | Claude Code 200k window minus buffer (Jaz Cloud 180k ceiling) | **165,000** | Safe headroom for thinking + output |
| `gemini-4-neon`, `gemini-4-hydrogen`, `gemini-2.5-pro`, `gemini-2.5-flash` | 1,048,576 (1M) | Jaz 1M quality cap | **400,000** | Keeps implicit/durable prefix caching fast and recall high |
| `grok-4`, `grok-code` | 256k | Grok CLI `auto_compact_threshold_percent = 85` | **217,600** | Matches upstream Grok CLI 85% default |
| `grok-4-fast` | 2,000k (2M) | Jaz long-window quality cap | **400,000** | Prevents 1.6M+ bloat while remaining caller-overridable |
| `Llama-4-Maverick`, `Llama-4-Scout` | 1,000k (1M) | Long-context recall cap | **300,000** | `Llama-3.3-70B` (128k) defaults to **100,000** |

### Override Precedence (`compact.Config`)

`compact.Engine.ResolveAutoCompactThreshold(spec)` is the single source of truth for both local auto-compaction and provider-native server compaction (`compact_20260112.trigger.value` on Anthropic and `context_management.compact_threshold` on OpenAI):

1. `compact.Config.ModelTokenLimits[modelOrSlug]` (exact ID or normalized slug prefix match)
2. `compact.Config.AutoCompactTokenLimit` (global token override)
3. `compact.Config.AutoCompactFraction` (global fraction in `(0, 1)`)
4. `schema.ModelSpec.AutoCompactThreshold()` (table above)

All overrides are clamped to at most 90% of `spec.ContextWindow` to preserve generation headroom.

## Core Architectural Invariants

- **Idiomatic Go Root Layout:** Packages live directly at module root (`harness`, `provider/...`, `compact`, `tools`, `mcp`, `store`, `schema`, `executor`, `server`, `auth`, `skills`, `integration`) with CLI entrypoint in `cmd/jaz-harness`.
- **KV-Cache Prefix Immutability:** Never mutate old messages or tool outputs in-place turn-by-turn before a durable `CompactionCheckpoint` is committed. Prefix bytes must stay byte-identical across steps until compaction commits.
- **Compaction Output Safety:** Summary generation uses `DefaultSummaryMaxOutputTokens = 12288` (clamped to `spec.MaxOutputTokens`), detects `max_tokens`/`length` truncation, performs at most one bounded continuation pass, and fails closed (`compact.ErrSummaryTruncated`) rather than committing a half-clipped summary.
- **Cross-Provider Compaction Safety:** Opaque provider compaction checkpoints (e.g., OpenAI `/responses/compact`) always carry a readable fallback digest (`compact.BuildFallbackSummary`), and `compact.ApplyCheckpoint` synthesizes one from replaced messages if missing, so switching providers mid-thread never drops context.
- **Eager vs. Deferred Tool Tiering:** Keep a small, deterministic, immutable eager core (native hosted tools, `search_tools`/`call_tool`/`plugins_search`/`skills_read`, and core workspace tools). All external MCP (`ServerID != ""`) and plugin (`PluginID != ""`) tools are strictly deferred (`DeferLoading: true`, `AlwaysLoad: false`) and must never mutate active tool schemas mid-thread.
- **Length & Buffer Safety:** Stream SSE/JSONL lines via dynamically growing readers (`provider.ScanLines` / `bufio.Reader`, 64 MiB ceiling) rather than fixed small `bufio.Scanner` limits; cap command/PTY streams via `executor.DefaultMaxExecOutputBytes` (4 MiB) and tool outputs via UTF-8-rune-safe `tools.TruncateUTF8` with explicit truncation markers.
- **Maintainability:** One statement per line; keep files modular and well under 1,000 lines; keep dependencies minimal; zero PII, secrets, or private internal references in repository content or commit messages.
