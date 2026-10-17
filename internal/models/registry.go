// Package models holds the upstream routing table for OpenCode Go.
//
// Upstream exposes four different dialects behind one base URL:
//   - OpenAI Chat Completions:  {base}/openai/v1/chat/completions
//   - OpenAI Responses:         {base}/openai/v1/responses
//   - Anthropic Messages:       {base}/anthropic/v1/messages
//   - Gemini GenerateContent:   {base}/google/v1beta/models/<model>:generateContent
//     (+ :streamGenerateContent for streaming)
//
// The docs table (https://opencode.ai/v2/docs/console/go) pins each known
// model to one dialect, but the live /v1/models catalog drifts over time
// (new models appear before docs update). So the registry combines an
// explicit table with prefix heuristics and defaults to chat/completions.
package models

import "strings"

// Backend identifies which upstream dialect a model speaks.
type Backend string

const (
	BackendChat      Backend = "chat"      // OpenAI Chat Completions
	BackendResponses Backend = "responses" // OpenAI Responses
	BackendMessages  Backend = "messages"  // Anthropic Messages
	BackendGemini    Backend = "gemini"    // Gemini GenerateContent
)

func (b Backend) String() string { return string(b) }

// Route describes where to send a model.
type Route struct {
	Backend Backend
	// Path is the request path relative to the upstream base, without leading slash.
	// For Gemini this is a template: the model id is interpolated.
	Path string
}

// Explicit per-model routing taken from the Go docs "Endpoints" table,
// refreshed against the live /v1/models catalog (2026-10-17).
var explicit = map[string]Backend{
	// Responses-native
	"grok-4.7":                   BackendResponses,
	"grok-4.6":                   BackendResponses,
	"gpt-6-luna":                 BackendResponses,
	"gpt-5.6-luna":               BackendResponses,
	"muse-spark-1.3-contributor": BackendResponses,
	"muse-spark-1.2-contributor": BackendResponses,
	"omen-alpha":                 BackendResponses,

	// Anthropic Messages-native
	"claude-haiku-5-5": BackendMessages,
	"minimax-m3":       BackendMessages,
	"minimax-m2.5":     BackendMessages,
	"minimax-m2.7":     BackendMessages,
	"qwen3.8-max":      BackendMessages,
	"qwen3.8-flash":    BackendMessages,
	"qwen3.7-plus":     BackendMessages,
	"qwen3.7-max":      BackendMessages,
	"qwen3.6-plus":     BackendMessages,

	// Gemini-native
	"gemini-4-argon-contributor": BackendGemini,
}

// Lookup returns the backend for a model id. Matching is case-insensitive
// and also strips common prefixes like "opencode-go/".
func Lookup(model string) Backend {
	m := Normalize(model)
	if b, ok := explicit[m]; ok {
		return b
	}
	// Heuristics for future models not yet in the explicit table.
	switch {
	case strings.HasPrefix(m, "gemini-"):
		return BackendGemini
	case strings.HasPrefix(m, "claude-"):
		return BackendMessages
	case strings.HasPrefix(m, "grok-"):
		return BackendResponses
	case strings.HasSuffix(m, "-luna"):
		return BackendResponses
	case strings.HasPrefix(m, "muse-spark-"):
		return BackendResponses
	case strings.HasPrefix(m, "qwen"):
		return BackendMessages
	case strings.HasPrefix(m, "minimax-m"):
		return BackendMessages
	default:
		return BackendChat
	}
}

// Normalize lowercases and strips provider prefixes.
func Normalize(model string) string {
	m := strings.TrimSpace(model)
	if i := strings.LastIndex(m, "/"); i >= 0 {
		m = m[i+1:]
	}
	return strings.ToLower(m)
}

// UpstreamPath returns the path (relative to base) for a backend + model.
func UpstreamPath(backend Backend, model string, stream bool) string {
	switch backend {
	case BackendResponses:
		return "openai/v1/responses"
	case BackendMessages:
		return "anthropic/v1/messages"
	case BackendGemini:
		m := Normalize(model)
		if stream {
			return "google/v1beta/models/" + m + ":streamGenerateContent?alt=sse"
		}
		return "google/v1beta/models/" + m + ":generateContent"
	default:
		return "openai/v1/chat/completions"
	}
}
