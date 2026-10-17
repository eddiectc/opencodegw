# opencodegw

OpenAI-compatible gateway for [OpenCode Go](https://opencode.ai/v2/docs/console/go).

One OpenAI API (`/v1/chat/completions`, `/v1/responses`, `/v1/models`) routes to all Go models across their four native upstream dialects:

| Upstream dialect | Path |
|---|---|
| OpenAI Chat Completions | `{base}/openai/v1/chat/completions` |
| OpenAI Responses | `{base}/openai/v1/responses` |
| Anthropic Messages | `{base}/anthropic/v1/messages` |
| Gemini GenerateContent | `{base}/google/v1beta/models/<model>:generateContent` |

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
```

## Thinking effort

Accepted from any of these client fields (first valid wins):

- `reasoning_effort: "max"`
- `reasoningEffort: "xhigh"`
- `reasoning: {"effort": "high"}`
- `effort: "medium"`
- `thinking: {"effort": "low"}` or `thinking: {"budget_tokens": 32000}`
- `thinking_effort: "minimal"`

Canonical levels: `none | low | medium | high | xhigh | max` (`minimal` aliases to `low`).

Mapping downstream:

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

Upstream always receives `Authorization: Bearer <OPENCODE_API_KEY>`, a real `User-Agent`, and an `x-opencode-session` (forwarded if the client sent one, else generated) for routing/prompt caching.

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
internal/openai/     gateway API types
internal/effort/     none|low|medium|high|xhigh|max normalization
internal/translate/  chat/responses/messages/gemini converters + SSE
internal/upstream/   HTTP forwarding
internal/server/     handlers
```
