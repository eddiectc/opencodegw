package server

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/eddiectc/opencodegw/internal/effort"
	"github.com/eddiectc/opencodegw/internal/models"
	"github.com/eddiectc/opencodegw/internal/openai"
	"github.com/eddiectc/opencodegw/internal/translate"
)

// ---- /v1/responses ----

func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 32<<20))
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "cannot read body")
		return
	}
	var req openai.ResponsesRequest
	dec := json.NewDecoder(strings.NewReader(string(body)))
	dec.UseNumber()
	if err := dec.Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "invalid JSON: "+err.Error())
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "model is required")
		return
	}
	applyDefaultEffortResponses(&req, s.cfg.DefaultEffort)
	defEff := effort.Normalize(s.cfg.DefaultEffort)
	stream := req.Stream != nil && *req.Stream

	backend := models.Lookup(req.Model)
	model := models.Normalize(req.Model)
	extra := sessionHeaders(r)

	if stream {
		s.serveResponsesStream(w, r, backend, model, req, defEff, extra)
		return
	}

	// Non-streaming: responses-native goes direct; others go via chat translators.
	if backend == models.BackendResponses {
		upBody := translate.ResponsesUpstreamBody(req, model, defEff)
		path := models.UpstreamPath(backend, model, false)
		status, raw, _, err := s.upstream.DoJSON(r.Context(), "POST", path, upBody, extra, false)
		if err != nil {
			writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
			return
		}
		if status >= 400 {
			proxyUpstreamError(w, status, raw)
			return
		}
		rr, err := translate.ResponsesPassthrough(raw, req.Model)
		if err != nil {
			writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, rr)
		return
	}

	// Fan out through chat path, then wrap as Responses.
	chatReq := translate.ResponsesToChatRequest(req, defEff)
	chatReq.Model = req.Model
	var chat openai.ChatResponse
	var perr error
	switch backend {
	case models.BackendMessages:
		chat, perr = s.doChatBackend(r, backend, model, chatReq, extra)
	case models.BackendGemini:
		chat, perr = s.doChatBackend(r, backend, model, chatReq, extra)
	default:
		chat, perr = s.doChatBackend(r, backend, model, chatReq, extra)
	}
	if perr != nil {
		if ue, ok := perr.(*upstreamErr); ok {
			proxyUpstreamError(w, ue.status, ue.body)
			return
		}
		writeError(w, http.StatusBadGateway, "upstream_error", perr.Error())
		return
	}
	rr := translate.ChatToResponsesResult(chat, req.Model)
	writeJSON(w, http.StatusOK, rr)
}

type upstreamErr struct {
	status int
	body   []byte
}

func (e *upstreamErr) Error() string { return http.StatusText(e.status) }

// doChatBackend runs a chat request against a backend and returns the ChatResponse.
func (s *Server) doChatBackend(r *http.Request, backend models.Backend, model string, chatReq openai.ChatRequest, extra http.Header) (openai.ChatResponse, error) {
	var upstreamBody any
	var path string
	switch backend {
	case models.BackendMessages:
		upstreamBody = translate.ChatToAnthropic(chatReq, model)
		path = models.UpstreamPath(backend, model, false)
	case models.BackendGemini:
		upstreamBody = translate.ChatToGemini(chatReq, model)
		path = models.UpstreamPath(backend, model, false)
	default:
		upstreamBody = chatPassthroughBody(chatReq, model)
		path = models.UpstreamPath(backend, model, false)
	}
	status, raw, _, err := s.upstream.DoJSON(r.Context(), "POST", path, upstreamBody, extra, false)
	if err != nil {
		return openai.ChatResponse{}, err
	}
	if status >= 400 {
		return openai.ChatResponse{}, &upstreamErr{status: status, body: raw}
	}
	switch backend {
	case models.BackendMessages:
		return translate.AnthropicToChat(raw, chatReq.Model)
	case models.BackendGemini:
		return translate.GeminiToChat(raw, chatReq.Model)
	case models.BackendResponses:
		return translate.ResponsesToChat(raw, chatReq.Model)
	default:
		return translate.ChatPassthrough(raw, chatReq.Model)
	}
}

// serveResponsesStream serves /v1/responses with stream=true.
// For responses-native backends we relay upstream SSE directly (translating
// model echo is unnecessary since bodies are already Responses-shaped).
// For other backends we run the chat stream translators but frame output
// as Responses SSE events.
func (s *Server) serveResponsesStream(w http.ResponseWriter, r *http.Request, backend models.Backend, model string, req openai.ResponsesRequest, defEff effort.Level, extra http.Header) {
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

	if backend == models.BackendResponses {
		upBody := translate.ResponsesUpstreamBody(req, model, defEff)
		upBody["stream"] = true
		path := models.UpstreamPath(backend, model, false)
		upResp, err := s.upstream.DoStream(r.Context(), "POST", path, upBody, extra)
		if err != nil {
			return
		}
		defer upResp.Body.Close()
		// Relay upstream events verbatim.
		_ = translate.ScanSSE(upResp.Body, func(ev translate.SSEEvent) bool {
			if ev.Data == "[DONE]" {
				_, _ = io.WriteString(w, "data: [DONE]\n\n")
				fl.Flush()
				return true
			}
			return emitRaw(ev.Event, ev.Data)
		})
		return
	}

	// Non-responses backends: synthesize a minimal Responses event sequence
	// from chat chunks. We run the upstream chat stream, convert to chat
	// chunks in memory via the translators is complex; simpler: run the
	// non-streaming chat call... no — instead relay translated chat deltas
	// wrapped as response.output_text.delta events.
	chatReq := translate.ResponsesToChatRequest(req, defEff)
	chatReq.Model = req.Model
	var upstreamBody any
	var path string
	isGeminiStream := false
	switch backend {
	case models.BackendMessages:
		m := translate.ChatToAnthropic(chatReq, model)
		m["stream"] = true
		upstreamBody = m
		path = models.UpstreamPath(backend, model, false)
	case models.BackendGemini:
		upstreamBody = translate.ChatToGemini(chatReq, model)
		path = models.UpstreamPath(backend, model, true)
		isGeminiStream = true
		_ = isGeminiStream
	default:
		m := chatPassthroughBody(chatReq, model)
		m["stream"] = true
		upstreamBody = m
		path = models.UpstreamPath(backend, model, false)
	}
	upResp, err := s.upstream.DoStream(r.Context(), "POST", path, upstreamBody, extra)
	if err != nil {
		return
	}
	defer upResp.Body.Close()

	respID := "resp_" + newReqID()
	emitRaw("response.created", `{"type":"response.created","response":{"id":"`+respID+`","object":"response","model":"`+req.Model+`"}}`)
	// Collect text via the chat translators by capturing their emitted chunks.
	emit := func(chunkJSON string) bool {
		if chunkJSON == "[DONE]" {
			return true
		}
		var ch openai.ChatStreamChunk
		if err := json.Unmarshal([]byte(chunkJSON), &ch); err != nil {
			return true
		}
		for _, c := range ch.Choices {
			if c.Delta.Content != "" {
				d, _ := json.Marshal(map[string]any{
					"type":  "response.output_text.delta",
					"delta": c.Delta.Content,
				})
				if !emitRaw("", string(d)) {
					return false
				}
			}
			for _, tc := range c.Delta.ToolCalls {
				if tc.Function != nil && tc.Function.Name != "" {
					d, _ := json.Marshal(map[string]any{
						"type": "response.output_item.added",
						"item": map[string]any{
							"type": "function_call", "name": tc.Function.Name, "call_id": tc.ID,
						},
					})
					if !emitRaw("", string(d)) {
						return false
					}
				}
				if tc.Function != nil && tc.Function.Arguments != "" {
					d, _ := json.Marshal(map[string]any{
						"type":  "response.function_call_arguments.delta",
						"delta": tc.Function.Arguments,
					})
					if !emitRaw("", string(d)) {
						return false
					}
				}
			}
		}
		return true
	}
	switch backend {
	case models.BackendMessages:
		_ = translate.TranslateAnthropicStreamToChat(upResp.Body, req.Model, emit)
	case models.BackendGemini:
		_ = translate.TranslateGeminiStreamToChat(upResp.Body, req.Model, emit)
	default:
		_ = translate.TranslateChatStream(upResp.Body, req.Model, emit)
	}
	emitRaw("response.completed", `{"type":"response.completed","response":{"id":"`+respID+`","status":"completed"}}`)
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	fl.Flush()
}
