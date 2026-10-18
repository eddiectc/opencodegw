# opencodegw

Multi-dialect gateway for [OpenCode Go](https://opencode.ai/v2/docs/console/go).

Every client dialect reaches every Go model. Native calls pass through losslessly (raw body + surgical fixes); cross-dialect fans out through canonical Chat and wraps back:

| Client dialect | Ingress | Native upstream |
|---|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions` | `{base}/openai/v1/chat/completions` |
| OpenAI Responses | `POST /v1/responses` | `{base}/openai/v1/responses` |
| Anthropic Messages | `POST /anthropic/v1/messages` | `{base}/anthropic/v1/messages` |
| Gemini GenerateContent | `POST /google/v1beta/models/<m>:generateContent` | `{base}/google/v1beta/models/<m>:generateContent` |

Thinking effort (`none|minimal|low|medium|high|xhigh|max`) is accepted in every client dialect and mapped to each backend's native knob.

## Quickstart

```bash
export OPENCODE_API_KEY="your-opencode-go-key"
go run ./cmd/opencodegw -addr :8080

# OpenAI-compatible usage
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer dummy"

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"hello"}],"reasoning_effort":"max","stream":false}'

curl http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"grok-4.7","input":"hello","reasoning":{"effort":"high"}}'

# Anthropic-native usage (exact budgets, cache_control, beta headers preserved)
curl http://localhost:8080/anthropic/v1/messages \
  -H "Content-Type: application/json" -H "x-api-key: dummy" \
  -d '{"model":"claude-haiku-5-5","max_tokens":100,"thinking":{"type":"enabled","budget_tokens":10000},"messages":[{"role":"user","content":"hello"}]}'

# Gemini-native usage (safety, grounding, responseSchema preserved)
curl http://localhost:8080/google/v1beta/models/gemini-4-argon-contributor:generateContent \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"role":"user","parts":[{"text":"hello"}]}],"generationConfig":{"thinkingConfig":{"thinkingLevel":"HIGH"}}}'
```

## Thinking effort

Accepted from any of these client fields (native knob wins on native paths, otherwise first valid wins):

- `reasoning_effort: "max"`
- `reasoningEffort: "xhigh"`
- `reasoning: {"effort": "high"}`
- `effort: "medium"`
- `thinking: {"effort": "low"}` or `thinking: {"budget_tokens": 32000}` (string form `"high"` also works)
- `thinking_effort: "minimal"`
- `generationConfig.thinkingConfig.thinkingLevel: "HIGH"` (Gemini native)

Canonical levels: `none | low | medium | high | xhigh | max` (`minimal` aliases to `low`).

Native paths preserve exact knobs: an Anthropic `thinking.budget_tokens` goes upstream verbatim (including non-bucket values), and a Gemini `thinkingLevel` is never overwritten. Canonical mapping applies only when injecting from aliases/defaults or translating cross-dialect:

| Level | OpenAI `reasoning.effort` | Anthropic `thinking.budget_tokens` | Gemini `thinkingLevel` |
|---|---|---|---|
| `none` | `none` | omitted | `MINIMAL` |
| `low` | `low` | 2048 | `LOW` |
| `medium` | `medium` | 8192 | `MEDIUM` |
| `high` | `high` | 16000 | `HIGH` |
| `xhigh` | `xhigh` | 32000 | `HIGH` |
| `max` | `max` | 64000 | `HIGH` (Gemini tops out at HIGH) |

Set a server-side default with `DEFAULT_EFFORT=max` or `-default-effort max`.

## Model routing

Explicit table in `internal/models/registry.go` (from Go docs + live `/v1/models`), with prefix heuristics for future models and chat/completions as the default:

- **responses**: `grok-4.7/4.6`, `gpt-6-luna`, `gpt-5.6-luna`, `muse-spark-1.x-contributor`, `omen-alpha`
- **messages**: `claude-haiku-5-5`, `minimax-m*`, `qwen3.*`
- **gemini**: `gemini-4-argon-contributor`
- **chat**: everything else (`glm-*`, `kimi-*`, `deepseek-*`, `mimo-*`, `longcat-*`, `hy*`, `space-bunny`, `step-5-preview`, `beam-free`, ...)

Model names are normalized (lowercased, `opencode-go/` prefix stripped).

## Endpoints

- `GET /healthz`, `GET /health`
- `GET /v1/models` (proxied + 5-min cache), alias `GET /models`
- `POST /v1/chat/completions` (+ alias `POST /chat/completions`), streaming via SSE
- `POST /v1/responses` (+ alias `POST /responses`), streaming supported
- `POST /anthropic/v1/messages` (+ aliases `/v1/messages`, `/messages`), `stream: true` via SSE
- `POST /google/v1beta/models/<m>:generateContent` (+ `/v1beta/...` short forms), `...:streamGenerateContent?alt=sse` for streaming

Native calls pass through losslessly: only the model is normalized, effort is merged into the native knob, and gateway-only aliases are stripped. Non-native backends translate through canonical Chat (multimodal content, tools, and sampling mapped; native-only knobs like `cache_control`, thought signatures, `safetySettings`, grounding, and `responseSchema` cannot transfer and are dropped).

Auth: `GATEWAY_API_KEY` accepts `Authorization: Bearer`, Anthropic `x-api-key`, and Google `x-goog-api-key` / `?key=`. 401s are shaped per dialect. Upstream always receives `Authorization: Bearer <OPENCODE_API_KEY>`, a real `User-Agent`, `x-opencode-session` (forwarded or generated), and native version headers (`Anthropic-Beta`, `anthropic-version`, `OpenAI-Beta`). Client credentials never go upstream.

Upstream 400s are proxied with no retry. Two refinements on the Responses path (verified by live probes against `muse-spark-1.3-contributor`):

- **Tool pairing**: every `function_call` needs exactly one matching `function_call_output` (upstream rejects orphans and duplicates with 400). The gateway drops unmatched calls/outputs and keeps the first output per `call_id` before sending.
- **Context overflow**: upstream answers over-window input with a detail-free 400. When the request estimates at or above the model's known window (`muse-spark-1.x-contributor`: 1M tokens), the gateway rewrites it to `code: context_length_exceeded` with the estimate, so clients compact and retry instead of guessing.

## Config

| Env | Flag | Default |
|---|---|---|
| `ADDR` / `PORT` | `-addr` | `:8080` |
| `OPENCODE_BASE` | `-upstream` | `https://opencode.ai/inference/go` |
| `OPENCODE_API_KEY` | `-api-key` | (required) |
| `GATEWAY_API_KEY` | `-gateway-key` | (empty = no auth) |
| `USER_AGENT` | `-user-agent` | `opencodegw/0.1.0` |
| `DEFAULT_EFFORT` | `-default-effort` | (empty) |
| `UPSTREAM_TIMEOUT_SEC` | `-timeout` | `300` |
| `DEBUG` | `-debug` | `false` |

## Layout

```
cmd/opencodegw/      binary entrypoint
internal/config/     env + flags
internal/models/     routing table + heuristics
internal/openai/     OpenAI API types
internal/effort/     none|low|medium|high|xhigh|max normalization
internal/translate/  native passthrough + chat/responses/messages/gemini converters + SSE
internal/upstream/   HTTP forwarding
internal/server/     ingress handlers (openai, responses, anthropic, gemini)
```
