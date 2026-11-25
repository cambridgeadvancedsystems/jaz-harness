# jaz-harness

Lightweight, zero-subprocess, multi-provider Go agent runtime and client/server daemon.

## Package Layout

Idiomatic Go root packages under `github.com/cambridgeadvancedsystems/jaz-harness`:

- `harness` — core multi-step agent loop, steering, cancellation, heartbeats, and server-compaction persistence
- `provider/...` — native wire-protocol clients (`openai`, `anthropic`, `grok`, `gemini`, `openaicompat`, `responses`, `factory`)
- `compact` — KV-cache-safe durable compaction, truncation-aware summarization, configurable thresholds, and cross-provider fallback digests
- `tools` — immutable eager core (`view_image`, `generate_image`, workspace tools) + strictly deferred MCP/plugin BM25 gateway + lazy workspace toolset
- `mcp` — MCP 2025-11-25 Streamable HTTP / SSE / stdio client + embeddable SEP-2640 Skills over MCP server
- `executor` — local rooted executor + unforked `codex exec-server` WebSocket JSON-RPC client with capped output buffers
- `schema` — canonical messages, lossless `NativeEnvelope`, multimodal tool results, and per-model specs / compaction defaults
- `store` — pluggable `Store` interface with in-memory and durable JSONL `FileStore`
- `server` — HTTP + SSE daemon (`?after_seq=` replay) and Go SDK client
- `auth` — Codex OAuth device-code / refresh-token rotation + pure-Go AWS SigV4
- `skills` — local, cloud snapshot, and SEP-2640 MCP skills catalog + plugin registry
- `cmd/jaz-harness` — standalone CLI (`serve`, `run`, `login`)

## Native Provider Tool Matrix

All providers speak their vendor's native wire protocol directly. Universal image tools (`view_image` and `generate_image`) work across **every** provider: `view_image` injects validated workspace/local images into each provider's native multimodal tool-response format, and `generate_image` uses the active provider's native image backend when available or a configurable fallback `provider.ImageGenerator` otherwise (with full reference/input image support via `referenced_image_paths` and `num_last_images_to_include`).

| Provider ID | Wire Protocol | Native Web Search | Native Web Fetch / URL | Native X / Social Search | Image Generation (Native & Fallback) | Multimodal Tool Inputs (`view_image`) | Deferred Tool Search | Native Compaction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `openai`, `azure_openai` | OpenAI Responses (`/v1/responses`, `/v1/responses/compact`) | `web_search` (`external_web_access`, `search_context_size`, `user_location`) | — | — | Native hosted `image_generation` (`image_generation_call` SSE + verbatim replay continuity) **and** native `/v1/images/generations` & `/v1/images/edits` (multipart + reference images) | Native `function_call_output` content array (`input_image` with `high`/`original` detail + `input_text`) | Static BM25 gateway (`search_tools`/`call_tool`) | `/v1/responses/compact` + `context_management` |
| `codex_oauth` | ChatGPT Codex Responses (`/backend-api/codex/responses`) | `web_search` (`x-codex-turn-state`, `session_id`, `chatgpt-account-id`) | — | — | Native `/images/generations` & JSON `/images/edits` (`images: [{image_url}]`) via `generate_image` + `image_generation_call` replay continuity | Native `function_call_output` content array (`input_image` with `high`/`original` detail + `input_text`) | Static BM25 gateway (`search_tools`/`call_tool`) | `/responses/compact` (with `configuration_update` stripping) |
| `anthropic` | Anthropic Messages (`/v1/messages`) | `web_search_20250305` / `web_search_20260209` (full `server_tool_use` + `web_search_tool_result` replay with `tool_use_id`) | `web_fetch_20250910` (`web-fetch-2025-09-10` beta) / `web_fetch_20260209` | — | Universal `generate_image` via configurable fallback `provider.ImageGenerator` (supports reference/input images) | Native `tool_result` content array (`text` + `image` base64/URL blocks) | Native `defer_loading` + `tool_search_tool_bm25_20251119` (`advanced-tool-use-2025-11-20` beta) | Native `compact_20260112` (`compact-2026-01-12` beta, `compaction_delta`, `stop_reason: "compaction"`, `usage.iterations`) |
| `vertex_anthropic` | Vertex Anthropic (`:streamRawPredict`) | `web_search_20250305` / `web_search_20260209` | `web_fetch_20250910` / `web_fetch_20260209` (when enabled on endpoint) | — | Universal `generate_image` via configurable fallback `provider.ImageGenerator` | Native `tool_result` content array (`text` + `image` blocks) | Native `defer_loading` + `tool_search_tool_bm25_20251119` | Native `compact_20260112` |
| `bedrock_anthropic` | AWS Bedrock (`invoke-with-response-stream`, SigV4 + binary eventstream) | `web_search_20250305` (pass-through when supported by region/account) | `web_fetch_20250910` (pass-through when supported) | — | Universal `generate_image` via configurable fallback `provider.ImageGenerator` | Native `tool_result` content array (`text` + `image` blocks) | Native `defer_loading` + `tool_search_tool_bm25_20251119` | Native `compact_20260112` |
| `grok` (`xai`) | xAI Responses (`/v1/responses`, `x-grok-conv-id`) | `web_search` (`allowed_domains`, `excluded_domains`, `enable_image_understanding`; strips OpenAI-only fields) | — | `x_search` (`allowed_x_handles`, `excluded_x_handles`, `from_date`, `to_date`, `enable_image_understanding`, `enable_video_understanding`; preserves `xs_*`/`ws_*` & `response.citations`) | Native `/v1/images/generations` & `/v1/images/edits` (`grok-imagine-image` + reference images) via `generate_image`, or configurable fallback | Native `function_call_output` content array (`input_image` + `input_text`, `original` normalized to `high`) | Static BM25 gateway (`search_tools`/`call_tool`) | Local KV-safe compaction |
| `gemini`, `vertex` | Google Gemini Vertex `v1beta1` / AI Studio `v1beta` (`:streamGenerateContent?alt=sse`) | `googleSearch` (combined with functions via `toolConfig.includeServerSideToolInvocations: true`; `groundingMetadata` citations) | `urlContext` (`urlContextMetadata` citations) | — | Native `:generateContent` (`responseModalities: ["TEXT", "IMAGE"]`, default `gemini-2.5-flash-image` + reference images) via `generate_image`, plus inline `inlineData` SSE decoding | Native `functionResponse.parts` (`inlineData` / `fileData`) | Static BM25 gateway (`search_tools`/`call_tool`) | Local KV-safe compaction (`continuation_token` supported) |
| `vertex_interactions` | Google Gemini Vertex `v1beta1` Stateless Interactions API (`store: false`, `step_list`) | `google_search` (`google_search_call` / `google_search_result` + citation extraction) | `url_context` (`url_context_call` / `url_context_result` + citation extraction) | — | Native `:generateContent` via `generate_image` (or configurable fallback) | Native `function_result.result` multimodal content array (`text` + `image` items) | Static BM25 gateway (`search_tools`/`call_tool`) | Local KV-safe compaction (`continuation_token` supported) |
| `meta` (`llama`) | Meta Llama API (`/compat/v1/chat/completions`) | — | — | — | Universal `generate_image` via configurable fallback `provider.ImageGenerator` | `role: "tool"` + companion `role: "user"` multimodal `image_url` message | Static BM25 gateway (`search_tools`/`call_tool`) | Local KV-safe compaction |
| `openrouter`, `bedrock_openai`, `openai_compat` | `/chat/completions` (OpenRouter headers / optional AWS SigV4) | — | — | — | Universal `generate_image` via configurable fallback `provider.ImageGenerator` | `role: "tool"` + companion `role: "user"` multimodal `image_url` message | Static BM25 gateway (`search_tools`/`call_tool`) | Local KV-safe compaction |

### Provider Wire Highlights

- **OpenAI & Codex OAuth (`openai`, `azure_openai`, `codex_oauth`):** `prompt_cache_key`, `session_id`, `x-codex-turn-state` sticky routing, encrypted reasoning replay, `~/.codex/auth.json` interop with serialized refresh-token rotation, hosted `web_search` & `image_generation` (`image_generation_call` continuity), and `/responses/compact` (`configuration_update` stripping).
- **Anthropic (`anthropic`, `vertex_anthropic`, `bedrock_anthropic`):** Verbatim signed `thinking` + `redacted_thinking` across tool loops, `<=4` `cache_control` breakpoints, `web_search` + `web_fetch` full replay preserving `tool_use_id`, native `defer_loading` + BM25 tool search, native `compact_20260112`, and Bearer/Relay mode.
- **Grok (`grok`, `xai`):** Native `/v1/responses` with `x_search` + `web_search` + `code_interpreter`, `x-grok-conv-id` KV cache affinity, verbatim `xs_*`/`ws_*`/`ci_*` hosted-call preservation, and top-level `response.citations` capture.
- **Google Gemini (`gemini`, `vertex`, `vertex_interactions`):** Vertex `v1beta1` `:streamGenerateContent` (`toolConfig.includeServerSideToolInvocations: true`) and stateless `Interactions API` (`store: false`), Gemini 4 `XHIGH` / safe `LOW` thinking (no `MINIMAL`, no `temperature`/`topP`/`cachedContent`), first-parallel `thoughtSignature` placement, multimodal `functionResponse.parts`, `continuation_token` loops, and Relay mode.

## Core Features

- **0-Pod Goroutine Runtime:** Hosted turns (chat, search, deferred MCP, `skills_read`, plugins) run as pure Go goroutines with zero child processes and zero sandbox pods.
- **Immutable Eager Core & Strictly Deferred External Tools:** Core workspace tools (`exec_command`, `read_file`, `write_file`, `edit_file`, `list_dir`, `view_image`), image generation (`generate_image`), and discovery tools (`search_tools`, `call_tool`, `plugins_search`, `skills_read`) form a small, immutable schema; all external MCP and plugin tools stay strictly deferred (zero mid-thread schema mutation) while preserving multimodal image results through `call_tool`.
- **KV-Cache-Safe Compaction & Cross-Provider Fallback:** Prefix bytes are never mutated in-place turn-by-turn before a committed checkpoint; quality-tuned thresholds compact 1M/2M models well below raw limits (see [AGENTS.md](AGENTS.md)) with per-model/global overrides, 12,288-token summary caps with truncation detection, native Anthropic/OpenAI server compaction, and guaranteed readable fallback digests when switching providers mid-thread.
- **Latest MCP + SEP-2640 Skills over MCP (`io.modelcontextprotocol/skills`):** Streamable HTTP, SSE, and stdio MCP client plus embeddable server supporting `skills/list`, `skills/get`, `skill://` URIs, `resources/read`, `resources/directory/read`, SHA-256 verification, multimodal tool outputs, and untrusted-origin framing.
- **Cloud-Ready Interfaces:** Pluggable `store.Store`, zero-secret `RelayMode` / `BearerAuth` for native LLM relays, configurable `provider.ImageGenerator`, and `TurnRequest.Heartbeat` hooks for durable workers.

## Quick Start

```bash
go test -race -count=1 ./...
go run ./cmd/jaz-harness serve --addr :8090
```

## License

MIT — see [LICENSE](LICENSE).
