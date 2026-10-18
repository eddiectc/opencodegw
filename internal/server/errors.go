package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/eddiectc/opencodegw/internal/openai"
	"github.com/eddiectc/opencodegw/internal/translate"
)

// upstreamFailure describes the request that failed upstream so error
// classification can consider request size against the model's window.
type upstreamFailure struct {
	// estTokens estimates the upstream request body in tokens (bytes/4).
	estTokens int
	// window is the model's context window in tokens, 0 when unknown.
	window int
	// model echoes the requested model id for error messages.
	model string
}

func (f upstreamFailure) likelyOverflow() bool {
	return f.window > 0 && f.estTokens >= f.window
}

// proxyUpstreamError forwards an upstream failure downstream with no retry.
// It preserves the upstream status code and Retry-After headers where present,
// passes OpenAI-shaped bodies through verbatim, and normalizes Anthropic/Gemini
// error shapes into OpenAI shape so downstream clients can parse them.
func proxyUpstreamError(w http.ResponseWriter, status int, raw []byte, hdr http.Header) {
	proxyUpstreamErrorWith(w, status, raw, hdr, upstreamFailure{})
}

// proxyUpstreamErrorWith behaves like proxyUpstreamError, except that a
// detail-free upstream 400 on a request estimated at or above the model's
// known context window is rewritten to a context_length_exceeded error so
// downstream clients can act on it (upstream returns the same generic
// message for several unrelated causes).
func proxyUpstreamErrorWith(w http.ResponseWriter, status int, raw []byte, hdr http.Header, fail upstreamFailure) {
	if status < 400 {
		status = http.StatusBadGateway
	}
	forwardRetryHeaders(w, hdr)
	if status == http.StatusBadRequest && fail.likelyOverflow() && translate.IsGenericInvalidParams(raw) {
		// Upstream's detail-free 400 on an over-window request: rewrite to
		// the standard context error (status preserved) with the window
		// and estimate so clients can compact instead of guessing.
		writeJSON(w, status, openai.ErrorResponse{
			Error: openai.ErrorBody{
				Message: fmt.Sprintf(
					"input tokens (~%d estimated) exceed %s context window (%d tokens); compact history or tool outputs and retry",
					fail.estTokens, fail.model, fail.window,
				),
				Type: "invalid_request_error",
				Code: "context_length_exceeded",
			},
		})
		return
	}
	var top map[string]any
	if err := json.Unmarshal(raw, &top); err == nil {
		if _, hasErr := top["error"]; hasErr {
			// Already OpenAI-shaped ({error:{message,...}}) with no
			// Anthropic top-level type marker: pass through verbatim.
			if t, _ := top["type"].(string); t != "error" {
				if em, ok := top["error"].(map[string]any); ok {
					if _, hasMsg := em["message"]; hasMsg {
						w.Header().Set("Content-Type", "application/json")
						w.WriteHeader(status)
						_, _ = w.Write(raw)
						return
					}
				} else if _, ok := top["error"].(string); ok {
					// String error is still meaningful; normalize below.
				} else {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(status)
					_, _ = w.Write(raw)
					return
				}
			}
		}
	}
	if oe, ok := translate.ErrorFromBytes(raw); ok {
		writeJSON(w, status, oe)
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
	writeError(w, status, "upstream_error", msg)
}

// proxyUpstreamBodyIfError detects a 200-with-error body (HTTP 200 carrying an
// error payload) and proxies it downstream. It reports whether it handled the
// response. Callers must not continue normal translation when it returns true.
func proxyUpstreamBodyIfError(w http.ResponseWriter, status int, raw []byte, hdr http.Header) bool {
	if status >= 400 {
		return false
	}
	if _, ok := translate.ErrorFromBytes(raw); ok {
		proxyUpstreamError(w, http.StatusBadGateway, raw, hdr)
		return true
	}
	return false
}

func forwardRetryHeaders(w http.ResponseWriter, hdr http.Header) {
	if len(hdr) == 0 {
		return
	}
	for _, k := range []string{"Retry-After", "Retry-After-Ms", "X-Request-Id", "X-Request-ID"} {
		if v := hdr.Get(k); v != "" {
			w.Header().Set(k, v)
		}
	}
}

// emitChatStreamError forwards a mid-stream upstream failure to a
// /v1/chat/completions SSE client as data: {"error":...} with no retry.
// Headers are already sent (HTTP 200), so the error must ride the stream.
func emitChatStreamError(w http.ResponseWriter, fl http.Flusher, oe openai.ErrorResponse) {
	_, _ = io.WriteString(w, "data: "+translate.OpenAIErrorBytes(oe)+"\n\n")
	if fl != nil {
		fl.Flush()
	}
}

// emitResponsesStreamError forwards a mid-stream upstream failure to a
// /v1/responses SSE client as a response.failed event plus a generic
// error event. Headers are already sent (HTTP 200).
func emitResponsesStreamError(emitRaw func(event, data string) bool, respID string, oe openai.ErrorResponse) {
	errObj, _ := json.Marshal(oe.Error)
	payload := `{"type":"response.failed","response":{"id":"` + respID + `","object":"response","status":"failed","error":` + string(errObj) + `}}`
	emitRaw("response.failed", payload)
	emitRaw("error", translate.OpenAIErrorBytes(oe))
}
