# jaz-harness

Lightweight, zero-subprocess, multi-provider Go agent runtime and client/server daemon.

## Supported Providers (100% Native Wire Protocols)

| Provider ID | Wire Protocol | Native Capabilities |
| --- | --- | --- |
| `openai`, `azure_openai` | OpenAI Responses API (`/v1/responses`, `/v1/responses/compact`) | `prompt_cache_key`, `session_id`, `x-codex-turn-state` sticky routing, encrypted reasoning replay, `web_search`, native remote compaction |
| `codex_oauth` | ChatGPT Codex Responses (`/backend-api/codex/responses`) | `~/.codex/auth.json` interop, device-code login, serialized refresh-token rotation, `chatgpt-account-id`, `x-codex-turn-state` |
| `anthropic` | Anthropic Messages API (`/v1/messages`) | Verbatim signed `thinking` + `redacted_thinking` across tool loops, <=4 `cache_control` breakpoints, `web_search_20250305` + `web_fetch_20250910` full replay, native `defer_loading` + `tool_search_tool_bm25`, server compaction beta, Bearer/Relay mode |
| `vertex_anthropic` | Vertex Anthropic (`:streamRawPredict`) | Full Anthropic Messages fidelity over GCP Bearer auth or native LLM relay |
| `bedrock_anthropic` | AWS Bedrock (`invoke-with-response-stream`) | Full Anthropic Messages fidelity over pure-Go AWS SigV4 + binary AWS eventstream decoder |
| `grok` (`xai`) | xAI Responses API (`/v1/responses`) | Native `x_search` + `web_search` + `code_interpreter`, `x-grok-conv-id` KV cache affinity, verbatim `xs_*`/`ws_*` preservation |
| `gemini`, `vertex`, `vertex_interactions` | Google Gemini Vertex `v1beta1` (`:streamGenerateContent` & stateless `Interactions API`) | Combined `googleSearch` + `urlContext` + `functionDeclarations` (`toolConfig.includeServerSideToolInvocations: true` or stateless `store=false` Interactions API), Gemini 4 `XHIGH`/safe `LOW` thinking, parallel `thoughtSignature` placement, `continuation_token` long-decode loop, Relay mode |
| `meta` (`llama`) | Meta Llama API (`/compat/v1/chat/completions`) | Llama 4 Maverick/Scout & Llama 3.3 streaming, tool calls, `tool_choice` normalization |
| `openrouter`, `bedrock_openai`, `openai_compat` | `/chat/completions` | OpenRouter headers, reasoning & cached token usage, optional AWS SigV4 |

## Core Features

- **0-Pod Goroutine Runtime:** Hosted turns (chat, search, deferred MCP, `skills_read`, plugins) run as pure Go goroutines with zero child processes and zero sandbox pods.
- **Lazy Workspace Sandbox:** Workspace tools (`exec_command`, `read_file`, `write_file`, `edit_file`, `list_dir`) have a static schema (0 KV-cache invalidation) and attach either a local rooted executor or an unforked upstream `codex exec-server` over WebSocket JSON-RPC on first call.
- **Deferred Tool Calling:** Supports both a universal BM25 gateway (`search_tools`, `call_tool`, `plugins_search`) across all providers and Anthropic's native `defer_loading` + `tool_search_tool_bm25_20251119`.
- **Latest MCP + SEP-2640 Skills over MCP (`io.modelcontextprotocol/skills`):** Streamable HTTP, SSE, and stdio MCP client plus embeddable server supporting `skills/list`, `skills/get`, `skill://` URIs, `resources/read`, `resources/directory/read`, SHA-256 verification, and untrusted-origin framing.
- **Canonical + Lossless Provider-Native Replay:** Stores canonical messages alongside raw provider-native envelopes so same-provider turns replay verbatim (preserving signatures and KV-cache prefixes) while mid-thread model switching projects cleanly across providers.
- **Cloud-Ready Interfaces:** Pluggable `store.Store` (Postgres/SQL/JSONL), zero-secret `RelayMode` / `BearerAuth` for LLM proxies, and `TurnRequest.Heartbeat` hooks for Temporal/durable workers.
- **Client/Server Daemon + Go SDK:** Run `jaz-harness serve` for HTTP + SSE (`?after_seq=` replay), steering, cancellation, and manual compaction, or embed `pkg/harness` directly in Go.

## Quick Start

```bash
go test -race -count=1 ./...
go run ./cmd/jaz-harness serve --addr :8090
```

## License

MIT — see [LICENSE](LICENSE).
