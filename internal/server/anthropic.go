package server

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/eddiectc/opencodegw/internal/effort"
	"github.com/eddiectc/opencodegw/internal/models"
	"github.com/eddiectc/opencodegw/internal/openai"
	"github.com/eddiectc/opencodegw/internal/translate"
	"github.com/eddiectc/opencodegw/internal/upstream"
)

// ---- /anthropic/v1/messages, /v1/messages, /messages ----

// handleAnthropicMessages serves Anthropic Messages ingress. Messages-native
// models pass through losslessly from the raw body; other backends fan out
// through the canonical chat translators and wrap back to Anthropic shape.
func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 32<<20))
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "cannot read body")
		return
	}
	var probe struct {
		Model  string `json:"model"`
		Stream *bool  `json:"stream,omitempty"`
	}
	if err := json.Unmarshal(body, &probe); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON: "+err.Error())
		return
	}
	if probe.Model == "" {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "model is required")
		return
	}
	defEff := effort.Normalize(s.cfg.DefaultEffort)
	stream := probe.Stream != nil && *probe.Stream

	backend := models.Lookup(probe.Model)
	model := models.Normalize(probe.Model)
	extra := sessionHeaders(r)

	if stream {
		s.serveAnthropicStream(w, r, backend, model, probe.Model, body, defEff, extra)
		return
	}

	if backend == models.BackendMessages {
		upBody := translate.AnthropicNativeBody(body, model, defEff)
		path := models.UpstreamPath(backend, model, false)
		fail := classifyUpstreamBody(upBody, probe.Model, model)
		status, raw, hdr, err := s.upstream.DoJSON(r.Context(), "POST", path, upBody, extra, false)
		if err != nil {
			writeAnthropicError(w, http.StatusBadGateway, "api_error", err.Error())
			return
		}
		if status >= 400 {
			proxyUpstreamAnthropicErrorWith(w, status, raw, hdr, fail)
			return
		}
		if proxyUpstreamAnthropicBodyIfError(w, status, raw, hdr) {
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(normalizeAnthropicModelEcho(raw, probe.Model))
		return
	}

	chatReq := translate.AnthropicToChatRequest(body, probe.Model, defEff)
	chat, perr := s.doChatBackend(r, backend, model, chatReq, extra)
	if perr != nil {
		if ue, ok := perr.(*upstreamErr); ok {
			proxyUpstreamAnthropicErrorWith(w, ue.status, ue.body, ue.header, ue.fail)
			return
		}
		writeAnthropicError(w, http.StatusBadGateway, "api_error", perr.Error())
		return
	}
	writeJSON(w, http.StatusOK, translate.ChatToAnthropicResult(chat, probe.Model))
}

// serveAnthropicStream serves Anthropic Messages with stream=true. Native
// backends relay upstream SSE verbatim; others frame canonical chat deltas
// as Anthropic events. Setup failures proxy as JSON before SSE headers;
// mid-stream failures ride the stream as event: error.
func (s *Server) serveAnthropicStream(w http.ResponseWriter, r *http.Request, backend models.Backend, model, display string, rawBody []byte, defEff effort.Level, extra http.Header) {
	upstreamBody, path, _ := anthropicStreamUpstream(rawBody, display, backend, model, defEff)
	upResp, err := s.upstream.DoStream(r.Context(), "POST", path, upstreamBody, extra)
	if err != nil {
		if ue, ok := err.(*upstream.UpstreamError); ok {
			proxyUpstreamAnthropicErrorWith(w, ue.Status, ue.Body, ue.Header, classifyUpstreamBody(upstreamBody, display, model))
			return
		}
		writeAnthropicError(w, http.StatusBadGateway, "api_error", err.Error())
		return
	}
	defer upResp.Body.Close()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	fl, _ := w.(http.Flusher)
	emitRaw := func(event, data string) bool {
		if event != "" {
			_, _ = io.WriteString(w, "event: "+event+"\n")
		}
		_, _ = io.WriteString(w, "data: "+data+"\n\n")
		fl.Flush()
		return r.Context().Err() == nil
	}

	if backend == models.BackendMessages {
		scanErr := translate.ScanSSE(upResp.Body, func(ev translate.SSEEvent) bool {
			return emitRaw(ev.Event, ev.Data)
		})
		if scanErr != nil && r.Context().Err() == nil {
			emitAnthropicStreamError(emitRaw, translate.UpstreamFailure(scanErr))
		}
		return
	}

	// Cross-dialect: translate upstream SSE to canonical chat chunks and
	// frame directly as Anthropic events (no intermediate pipe: the framer
	// consumes chunk JSON synchronously, so client disconnects cannot
	// deadlock a writer goroutine).
	fr := translate.NewAnthropicStreamFramer(display)
	emitEv := func(ev translate.SSEEvent) bool { return emitRaw(ev.Event, ev.Data) }
	emitChunk := func(chunkJSON string) bool {
		if chunkJSON == "[DONE]" {
			return true
		}
		return fr.Push(chunkJSON, emitEv)
	}
	var terr error
	switch backend {
	case models.BackendGemini:
		terr = translate.TranslateGeminiStreamToChat(upResp.Body, display, emitChunk)
	case models.BackendResponses:
		terr = translate.TranslateResponsesStreamToChat(upResp.Body, display, emitChunk)
	default:
		terr = translate.TranslateChatStream(upResp.Body, display, emitChunk)
	}
	if terr != nil {
		if se, ok := translate.AsStreamError(terr); ok {
			emitAnthropicStreamError(emitRaw, se.Err)
		} else if r.Context().Err() == nil {
			emitAnthropicStreamError(emitRaw, translate.UpstreamFailure(terr))
		}
		return
	}
	fr.Finish(emitEv)
}

// anthropicStreamUpstream builds the upstream request for Anthropic
// streaming: native bodies pass through, cross-dialect goes canonical chat.
func anthropicStreamUpstream(rawBody []byte, display string, backend models.Backend, model string, defEff effort.Level) (any, string, openai.ChatRequest) {
	if backend == models.BackendMessages {
		upBody := translate.AnthropicNativeBody(rawBody, model, defEff)
		upBody["stream"] = true
		return upBody, models.UpstreamPath(backend, model, false), openai.ChatRequest{}
	}
	chatReq := translate.AnthropicToChatRequest(rawBody, display, defEff)
	switch backend {
	case models.BackendGemini:
		return translate.ChatToGemini(chatReq, model), models.UpstreamPath(backend, model, true), chatReq
	case models.BackendResponses:
		m := translate.ChatToResponses(chatReq, model)
		m["stream"] = true
		return m, models.UpstreamPath(backend, model, false), chatReq
	default:
		b, _ := json.Marshal(chatReq)
		var req openai.ChatRequest
		_ = json.Unmarshal(b, &req)
		m := translate.ChatNativeBody(b, req, model, defEff)
		m["stream"] = true
		if chatReq.StreamOptions != nil && chatReq.StreamOptions.IncludeUsage {
			m["stream_options"] = map[string]any{"include_usage": true}
		}
		return m, models.UpstreamPath(backend, model, false), chatReq
	}
}

// normalizeAnthropicModelEcho fixes the model echo when upstream omits it.
func normalizeAnthropicModelEcho(raw []byte, display string) []byte {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return raw
	}
	if s, _ := m["model"].(string); s == "" {
		m["model"] = display
		b, err := json.Marshal(m)
		if err != nil {
			return raw
		}
		return b
	}
	return raw
}
