package server

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/eddiectc/opencodegw/internal/openai"
	"github.com/eddiectc/opencodegw/internal/translate"
)

// proxyUpstreamError forwards an upstream failure downstream with no retry.
// It preserves the upstream status code and Retry-After headers where present,
// passes OpenAI-shaped bodies through verbatim, and normalizes Anthropic/Gemini
// error shapes into OpenAI shape so downstream clients can parse them.
func proxyUpstreamError(w http.ResponseWriter, status int, raw []byte, hdr http.Header) {
	if status < 400 {
		status = http.StatusBadGateway
	}
	forwardRetryHeaders(w, hdr)
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
