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
	"github.com/eddiectc/opencodegw/internal/upstream"
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
	applyDefaultEffortResponses(&req, body, s.cfg.DefaultEffort)
	defEff := effort.Normalize(s.cfg.DefaultEffort)
	stream := req.Stream != nil && *req.Stream

	backend := models.Lookup(req.Model)
	model := models.Normalize(req.Model)
	extra := sessionHeaders(r)

	if stream {
		s.serveResponsesStream(w, r, backend, model, req, body, defEff, extra)
		return
	}

	// Non-streaming: responses-native goes direct; others go via chat translators.
	if backend == models.BackendResponses {
		upBody := translate.ResponsesNativeBody(body, req, model, defEff)
		path := models.UpstreamPath(backend, model, false)
		fail := classifyUpstreamBody(upBody, req.Model, model)
		status, raw, hdr, err := s.upstream.DoJSON(r.Context(), "POST", path, upBody, extra, false)
		if err != nil {
			writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
			return
		}
		if status >= 400 {
			proxyUpstreamErrorWith(w, status, raw, hdr, fail)
			return
		}
		if proxyUpstreamBodyIfError(w, status, raw, hdr) {
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
			proxyUpstreamErrorWith(w, ue.status, ue.body, ue.header, ue.fail)
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
	header http.Header
	fail   upstreamFailure
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
	status, raw, hdr, err := s.upstream.DoJSON(r.Context(), "POST", path, upstreamBody, extra, false)
	if err != nil {
		return openai.ChatResponse{}, err
	}
	fail := classifyUpstreamBody(upstreamBody, chatReq.Model, model)
	if status >= 400 {
		return openai.ChatResponse{}, &upstreamErr{status: status, body: raw, header: hdr, fail: fail}
	}
	if oe, ok := translate.ErrorFromBytes(raw); ok {
		return openai.ChatResponse{}, &upstreamErr{
			status: http.StatusBadGateway,
			body:   []byte(translate.OpenAIErrorBytes(oe)),
			header: hdr,
			fail:   fail,
		}
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
//
// Upstream setup failures are proxied as JSON errors before SSE headers are
// sent. Mid-stream failures ride the stream as response.failed + error events
// because headers are already committed. No retries anywhere.
func (s *Server) serveResponsesStream(w http.ResponseWriter, r *http.Request, backend models.Backend, model string, req openai.ResponsesRequest, rawBody []byte, defEff effort.Level, extra http.Header) {
	sendSSEHeaders := func() http.Flusher {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		return fl
	}

	// Build the upstream request without touching the downstream headers so
	// setup errors can still be proxied as JSON with the upstream status.
	var upstreamBody any
	var path string
	if backend == models.BackendResponses {
		upBody := translate.ResponsesNativeBody(rawBody, req, model, defEff)
		upBody["stream"] = true
		upstreamBody = upBody
		path = models.UpstreamPath(backend, model, false)
	} else {
		chatReq := translate.ResponsesToChatRequest(req, defEff)
		chatReq.Model = req.Model
		switch backend {
		case models.BackendMessages:
			m := translate.ChatToAnthropic(chatReq, model)
			m["stream"] = true
			upstreamBody = m
			path = models.UpstreamPath(backend, model, false)
		case models.BackendGemini:
			upstreamBody = translate.ChatToGemini(chatReq, model)
			path = models.UpstreamPath(backend, model, true)
		default:
			m := chatPassthroughBody(chatReq, model)
			m["stream"] = true
			upstreamBody = m
			path = models.UpstreamPath(backend, model, false)
		}
	}
	upResp, err := s.upstream.DoStream(r.Context(), "POST", path, upstreamBody, extra)
	if err != nil {
		if ue, ok := err.(*upstream.UpstreamError); ok {
			proxyUpstreamErrorWith(w, ue.Status, ue.Body, ue.Header, classifyUpstreamBody(upstreamBody, req.Model, model))
			return
		}
		writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
		return
	}
	defer upResp.Body.Close()

	fl := sendSSEHeaders()
	emitRaw := func(event, data string) bool {
		if event != "" {
			_, _ = io.WriteString(w, "event: "+event+"\n")
		}
		_, _ = io.WriteString(w, "data: "+data+"\n\n")
		fl.Flush()
		return r.Context().Err() == nil
	}

	if backend == models.BackendResponses {
		respID := "resp_" + newReqID()
		scanErr := translate.ScanSSE(upResp.Body, func(ev translate.SSEEvent) bool {
			if ev.Data == "[DONE]" {
				_, _ = io.WriteString(w, "data: [DONE]\n\n")
				fl.Flush()
				return true
			}
			if se, isErr := translate.DetectSSEError(ev); isErr {
				if id := extractResponsesID(ev.Data); id != "" {
					respID = id
				}
				// Relay the raw error event verbatim, then the normalized form.
				emitRaw(ev.Event, ev.Data)
				emitResponsesStreamError(emitRaw, respID, se.Err)
				return false
			}
			if id := extractResponsesID(ev.Data); id != "" {
				respID = id
			}
			return emitRaw(ev.Event, ev.Data)
		})
		if scanErr != nil && r.Context().Err() == nil {
			emitResponsesStreamError(emitRaw, respID, translate.UpstreamFailure(scanErr))
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		fl.Flush()
		return
	}

	// Non-responses backends: synthesize a minimal Responses event sequence
	// from chat chunks by relaying translated chat deltas wrapped as
	// response.output_text.delta events.
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
	var terr error
	switch backend {
	case models.BackendMessages:
		terr = translate.TranslateAnthropicStreamToChat(upResp.Body, req.Model, emit)
	case models.BackendGemini:
		terr = translate.TranslateGeminiStreamToChat(upResp.Body, req.Model, emit)
	default:
		terr = translate.TranslateChatStream(upResp.Body, req.Model, emit)
	}
	if terr != nil {
		if se, ok := translate.AsStreamError(terr); ok {
			emitResponsesStreamError(emitRaw, respID, se.Err)
		} else if r.Context().Err() == nil {
			emitResponsesStreamError(emitRaw, respID, translate.UpstreamFailure(terr))
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		fl.Flush()
		return
	}
	emitRaw("response.completed", `{"type":"response.completed","response":{"id":"`+respID+`","status":"completed"}}`)
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	fl.Flush()
}

func extractResponsesID(data string) string {
	var probe struct {
		Response *struct {
			ID string `json:"id"`
		} `json:"response"`
		ID string `json:"id"`
	}
	if err := json.Unmarshal([]byte(data), &probe); err != nil {
		return ""
	}
	if probe.Response != nil && probe.Response.ID != "" {
		return probe.Response.ID
	}
	return probe.ID
}
