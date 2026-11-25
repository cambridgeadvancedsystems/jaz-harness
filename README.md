# jaz-harness

Lightweight, zero-subprocess, multi-provider Go agent runtime and client/server daemon.

## Package Layout

Idiomatic Go root packages under `github.com/cambridgeadvancedsystems/jaz-harness`:

- `harness` — core multi-step agent loop, steering, cancellation, heartbeats, and server-compaction persistence
- `provider/...` — native wire-protocol clients (`openai`, `anthropic`, `grok`, `gemini`, `openaicompat`, `responses`, `factory`)
- `compact` — KV-cache-safe durable compaction, truncation-aware summarization, configurable thresholds, and cross-provider fallback digests
- `tools` — immutable eager core + strictly deferred MCP/plugin BM25 gateway + lazy workspace toolset
- `mcp` — MCP 2025-11-25 Streamable HTTP / SSE / stdio client + embeddable SEP-2640 Skills over MCP server
- `executor` — local rooted executor + unforked `codex exec-server` WebSocket JSON-RPC client with capped output buffers
- `schema` — canonical messages, lossless `NativeEnvelope`, and per-model specs / compaction defaults
- `store` — pluggable `Store` interface with in-memory and durable JSONL `FileStore`
- `server` — HTTP + SSE daemon (`?after_seq=` replay) and Go SDK client
- `auth` — Codex OAuth device-code / refresh-token rotation + pure-Go AWS SigV4
- `skills` — local, cloud snapshot, and SEP-2640 MCP skills catalog + plugin registry
- `cmd/jaz-harness` — standalone CLI (`serve`, `run`, `login`)

## Supported Providers (100% Native Wire Protocols)

| Provider ID | Wire Protocol | Native Capabilities |
| --- | --- | --- |
| `openai`, `azure_openai` | OpenAI Responses API (`/v1/responses`, `/v1/responses/compact`) | `prompt_cache_key`, `session_id`, `x-codex-turn-state` sticky routing, encrypted reasoning replay, `web_search`, `context_management` server compaction, `/responses/compact` (`configuration_update` stripping) |
| `codex_oauth` | ChatGPT Codex Responses (`/backend-api/codex/responses`) | `~/.codex/auth.json` interop, device-code login, serialized refresh-token rotation, `chatgpt-account-id`, `x-codex-turn-state` |
| `anthropic` | Anthropic Messages API (`/v1/messages`) | Verbatim signed `thinking` + `redacted_thinking` across tool loops, <=4 `cache_control` breakpoints, `web_search_20250305` + `web_fetch_20250910` full replay, native `defer_loading` + `tool_search_tool_bm25_20251119`, native compaction (`compact-2026-01-12` / `compact_20260112`, `compaction_delta`, `stop_reason: "compaction"`, `usage.iterations`), Bearer/Relay mode |
| `vertex_anthropic` | Vertex Anthropic (`:streamRawPredict`) | Full Anthropic Messages fidelity over GCP Bearer auth or native LLM relay |
| `bedrock_anthropic` | AWS Bedrock (`invoke-with-response-stream`) | Full Anthropic Messages fidelity over pure-Go AWS SigV4 + binary AWS eventstream decoder |
| `grok` (`xai`) | xAI Responses API (`/v1/responses`) | Native `x_search` + `web_search` + `code_interpreter`, `x-grok-conv-id` KV cache affinity, verbatim `xs_*`/`ws_*` preservation |
| `gemini`, `vertex`, `vertex_interactions` | Google Gemini Vertex `v1beta1` (`:streamGenerateContent` & stateless `Interactions API`) | Combined `googleSearch` + `urlContext` + `functionDeclarations` (`toolConfig.includeServerSideToolInvocations: true` or stateless `store=false` Interactions API), Gemini 4 `XHIGH`/safe `LOW` thinking, parallel `thoughtSignature` placement, `continuation_token` long-decode loop, Relay mode |
| `meta` (`llama`) | Meta Llama API (`/compat/v1/chat/completions`) | Llama 4 Maverick/Scout & Llama 3.3 streaming, tool calls, `tool_choice` normalization |
| `openrouter`, `bedrock_openai`, `openai_compat` | `/chat/completions` | OpenRouter headers, reasoning & cached token usage, optional AWS SigV4 |

## Core Features

- **0-Pod Goroutine Runtime:** Hosted turns (chat, search, deferred MCP, `skills_read`, plugins) run as pure Go goroutines with zero child processes and zero sandbox pods.
- **Immutable Eager Core & Strictly Deferred External Tools:** Core workspace tools (`exec_command`, `read_file`, `write_file`, `edit_file`, `list_dir`) and discovery tools (`search_tools`, `call_tool`, `plugins_search`, `skills_read`) form a small, immutable schema; all external MCP and plugin tools stay strictly deferred (zero mid-thread schema mutation).
- **KV-Cache-Safe Compaction & Cross-Provider Fallback:** Prefix bytes are never mutated in-place turn-by-turn before a committed checkpoint; quality-tuned thresholds compact 1M/2M models well below raw limits (see [AGENTS.md](AGENTS.md)) with per-model/global overrides, 12,288-token summary caps with truncation detection, native Anthropic/OpenAI server compaction, and guaranteed readable fallback digests when switching providers mid-thread.
- **Latest MCP + SEP-2640 Skills over MCP (`io.modelcontextprotocol/skills`):** Streamable HTTP, SSE, and stdio MCP client plus embeddable server supporting `skills/list`, `skills/get`, `skill://` URIs, `resources/read`, `resources/directory/read`, SHA-256 verification, and untrusted-origin framing.
- **Cloud-Ready Interfaces:** Pluggable `store.Store`, zero-secret `RelayMode` / `BearerAuth` for native LLM relays, and `TurnRequest.Heartbeat` hooks for durable workers.

## Quick Start

```bash
go test -race -count=1 ./...
go run ./cmd/jaz-harness serve --addr :8090
```

## License

MIT — see [LICENSE](LICENSE).
