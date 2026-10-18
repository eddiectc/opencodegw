package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/eddiectc/opencodegw/internal/openai"
	"github.com/eddiectc/opencodegw/internal/translate"
)

// Native-shaped errors for Anthropic and Gemini ingress. The OpenAI variants
// live in errors.go; behavior is mirrored: preserve status + retry headers,
// pass native-shaped bodies verbatim, normalize anything else, and rewrite
// the detail-free over-window 400 to a clear context error.

// ---- Anthropic ----

// writeAnthropicError writes {"type":"error","error":{"type","message"}}.
func writeAnthropicError(w http.ResponseWriter, status int, typ, msg string) {
	writeJSON(w, status, map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    typ,
			"message": msg,
		},
	})
}

// proxyUpstreamAnthropicErrorWith proxies an upstream failure to an Anthropic
// client, normalizing to Anthropic error shape.
func proxyUpstreamAnthropicErrorWith(w http.ResponseWriter, status int, raw []byte, hdr http.Header, fail upstreamFailure) {
	if status < 400 {
		status = http.StatusBadGateway
	}
	forwardRetryHeaders(w, hdr)
	if status == http.StatusBadRequest && fail.likelyOverflow() && translate.IsGenericInvalidParams(raw) {
		writeAnthropicError(w, status, "invalid_request_error", fmt.Sprintf(
			"input tokens (~%d estimated) exceed %s context window (%d tokens); compact history or tool outputs and retry",
			fail.estTokens, fail.model, fail.window,
		))
		return
	}
	var top map[string]any
	if err := json.Unmarshal(raw, &top); err == nil {
		if t, _ := top["type"].(string); t == "error" {
			if em, ok := top["error"].(map[string]any); ok {
				if _, hasMsg := em["message"]; hasMsg {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(status)
					_, _ = w.Write(raw)
					return
				}
			}
		}
	}
	if oe, ok := translate.ErrorFromBytes(raw); ok {
		writeAnthropicError(w, status, anthropicTypeFromOpenAI(oe, status), oe.Error.Message)
		return
	}
	msg := strings.TrimSpace(string(raw))
	if msg == "" {
		msg = http.StatusText(status)
		if msg == "" {
			msg = "upstream error"
		}
	}
	if len(msg) > 2000 {
		msg = msg[:2000] + "...(truncated)"
	}
	writeAnthropicError(w, status, anthropicTypeForStatus(status), msg)
}

// proxyUpstreamAnthropicBodyIfError detects a 200-with-error body and proxies
// it downstream in Anthropic shape.
func proxyUpstreamAnthropicBodyIfError(w http.ResponseWriter, status int, raw []byte, hdr http.Header) bool {
	if status >= 400 {
		return false
	}
	if _, ok := translate.ErrorFromBytes(raw); ok {
		proxyUpstreamAnthropicErrorWith(w, http.StatusBadGateway, raw, hdr, upstreamFailure{})
		return true
	}
	return false
}

// emitAnthropicStreamError forwards a mid-stream failure as event: error.
func emitAnthropicStreamError(emitRaw func(event, data string) bool, oe openai.ErrorResponse) {
	b, _ := json.Marshal(map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    anthropicTypeFromOpenAI(oe, 0),
			"message": oe.Error.Message,
		},
	})
	emitRaw("error", string(b))
}

func anthropicTypeFromOpenAI(oe openai.ErrorResponse, status int) string {
	t := strings.ToLower(strings.TrimSpace(oe.Error.Type))
	if strings.HasSuffix(t, "_error") {
		return oe.Error.Type
	}
	if code, ok := oe.Error.Code.(string); ok && strings.HasSuffix(strings.ToLower(code), "_error") {
		return code
	}
	joined := t + " " + strings.ToLower(fmt.Sprintf("%v", oe.Error.Code))
	switch {
	case strings.Contains(joined, "rate_limit") || strings.Contains(joined, "quota") || strings.Contains(joined, "resource_exhausted"):
		return "rate_limit_error"
	case strings.Contains(joined, "overload") || strings.Contains(joined, "unavailable"):
		return "overloaded_error"
	case strings.Contains(joined, "auth"):
		return "authentication_error"
	case strings.Contains(joined, "permission") || strings.Contains(joined, "forbidden"):
		return "permission_error"
	case strings.Contains(joined, "not_found"):
		return "not_found_error"
	case strings.Contains(joined, "invalid_request") || strings.Contains(joined, "invalid_argument") || strings.Contains(joined, "context_length"):
		return "invalid_request_error"
	}
	return anthropicTypeForStatus(status)
}

func anthropicTypeForStatus(status int) string {
	switch status {
	case http.StatusBadRequest:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusServiceUnavailable, 529:
		return "overloaded_error"
	default:
		return "api_error"
	}
}

// ---- Gemini ----

// writeGeminiError writes {"error":{"code","message","status"}} with code set
// to the HTTP status, matching Google API shape.
func writeGeminiError(w http.ResponseWriter, status int, gstatus, msg string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"code":    status,
			"message": msg,
			"status":  gstatus,
		},
	})
}

// geminiErrorBody builds a Google-shaped error payload for mid-stream
// failures, where the HTTP status is already committed.
func geminiErrorBody(gstatus, msg string) map[string]any {
	if gstatus == "" {
		gstatus = "INTERNAL"
	}
	return map[string]any{
		"error": map[string]any{
			"code":    geminiCodeForStatus(gstatus),
			"message": msg,
			"status":  gstatus,
		},
	}
}

// geminiErrorFromOpenAI maps a normalized error to Google shape.
func geminiErrorFromOpenAI(oe openai.ErrorResponse) map[string]any {
	return geminiErrorBody(geminiStatusFromText(oe.Error.Type, fmt.Sprintf("%v", oe.Error.Code)), oe.Error.Message)
}

// proxyUpstreamGeminiErrorWith proxies an upstream failure to a Gemini
// client, normalizing to Google error shape.
func proxyUpstreamGeminiErrorWith(w http.ResponseWriter, status int, raw []byte, hdr http.Header, fail upstreamFailure) {
	if status < 400 {
		status = http.StatusBadGateway
	}
	forwardRetryHeaders(w, hdr)
	if status == http.StatusBadRequest && fail.likelyOverflow() && translate.IsGenericInvalidParams(raw) {
		writeGeminiError(w, status, "INVALID_ARGUMENT", fmt.Sprintf(
			"input tokens (~%d estimated) exceed %s context window (%d tokens); compact history or tool outputs and retry",
			fail.estTokens, fail.model, fail.window,
		))
		return
	}
	var top map[string]any
	if err := json.Unmarshal(raw, &top); err == nil {
		if em, ok := top["error"].(map[string]any); ok {
			// Google shape: numeric code + status string. OpenAI shape has
			// string code + type instead.
			_, codeIsNum := em["code"].(float64)
			_, hasStatus := em["status"].(string)
			if _, hasMsg := em["message"]; hasMsg && codeIsNum && hasStatus {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				_, _ = w.Write(raw)
				return
			}
		}
	}
	if oe, ok := translate.ErrorFromBytes(raw); ok {
		gstatus := geminiStatusFromText(oe.Error.Type, fmt.Sprintf("%v", oe.Error.Code))
		if gstatus == "INTERNAL" || gstatus == "UNKNOWN" {
			gstatus = geminiStatusForHTTP(status)
		}
		writeGeminiError(w, status, gstatus, oe.Error.Message)
		return
	}
	msg := strings.TrimSpace(string(raw))
	if msg == "" {
		msg = http.StatusText(status)
		if msg == "" {
			msg = "upstream error"
		}
	}
	if len(msg) > 2000 {
		msg = msg[:2000] + "...(truncated)"
	}
	writeGeminiError(w, status, geminiStatusForHTTP(status), msg)
}

// proxyUpstreamGeminiBodyIfError detects a 200-with-error body and proxies it
// downstream in Google shape.
func proxyUpstreamGeminiBodyIfError(w http.ResponseWriter, status int, raw []byte, hdr http.Header) bool {
	if status >= 400 {
		return false
	}
	if _, ok := translate.ErrorFromBytes(raw); ok {
		proxyUpstreamGeminiErrorWith(w, http.StatusBadGateway, raw, hdr, upstreamFailure{})
		return true
	}
	return false
}

func geminiStatusFromText(typ, code string) string {
	joined := strings.ToLower(typ + " " + code)
	switch {
	case strings.Contains(joined, "invalid_request") || strings.Contains(joined, "invalid_argument") || strings.Contains(joined, "context_length"):
		return "INVALID_ARGUMENT"
	case strings.Contains(joined, "unauthenticated") || strings.Contains(joined, "authentication") || strings.Contains(joined, "invalid_api_key"):
		return "UNAUTHENTICATED"
	case strings.Contains(joined, "permission") || strings.Contains(joined, "forbidden"):
		return "PERMISSION_DENIED"
	case strings.Contains(joined, "not_found"):
		return "NOT_FOUND"
	case strings.Contains(joined, "rate_limit") || strings.Contains(joined, "resource_exhausted") || strings.Contains(joined, "quota"):
		return "RESOURCE_EXHAUSTED"
	case strings.Contains(joined, "overload") || strings.Contains(joined, "unavailable"):
		return "UNAVAILABLE"
	case strings.Contains(joined, "timeout") || strings.Contains(joined, "deadline"):
		return "DEADLINE_EXCEEDED"
	case strings.Contains(joined, "failed_precondition"):
		return "FAILED_PRECONDITION"
	}
	upper := strings.ToUpper(strings.TrimSpace(typ))
	switch upper {
	case "INVALID_ARGUMENT", "UNAUTHENTICATED", "PERMISSION_DENIED", "NOT_FOUND",
		"RESOURCE_EXHAUSTED", "INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED":
		return upper
	}
	return "INTERNAL"
}

func geminiStatusForHTTP(status int) string {
	switch status {
	case http.StatusBadRequest:
		return "INVALID_ARGUMENT"
	case http.StatusUnauthorized:
		return "UNAUTHENTICATED"
	case http.StatusForbidden:
		return "PERMISSION_DENIED"
	case http.StatusNotFound:
		return "NOT_FOUND"
	case http.StatusTooManyRequests:
		return "RESOURCE_EXHAUSTED"
	case http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout, 529:
		return "UNAVAILABLE"
	default:
		if status >= 500 {
			return "INTERNAL"
		}
		return "UNKNOWN"
	}
}

func geminiCodeForStatus(gstatus string) int {
	switch gstatus {
	case "INVALID_ARGUMENT":
		return 400
	case "UNAUTHENTICATED":
		return 401
	case "PERMISSION_DENIED":
		return 403
	case "NOT_FOUND":
		return 404
	case "RESOURCE_EXHAUSTED":
		return 429
	case "FAILED_PRECONDITION":
		return 400
	case "UNAVAILABLE":
		return 503
	case "DEADLINE_EXCEEDED":
		return 504
	default:
		return 500
	}
}
