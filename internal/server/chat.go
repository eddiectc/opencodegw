package server

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/eddiectc/opencodegw/internal/effort"
	"github.com/eddiectc/opencodegw/internal/models"
	"github.com/eddiectc/opencodegw/internal/openai"
	"github.com/eddiectc/opencodegw/internal/translate"
	"github.com/eddiectc/opencodegw/internal/upstream"
)

func newReqID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// ---- /v1/models ----

type modelsCache struct {
	mu   sync.Mutex
	at   time.Time
	data []byte
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	s.cache.mu.Lock()
	if time.Since(s.cache.at) < 5*time.Minute && len(s.cache.data) > 0 {
		data := s.cache.data
		s.cache.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(data)
		return
	}
	s.cache.mu.Unlock()

	status, raw, hdr, err := s.upstream.DoRawGet(r.Context(), "v1/models")
	if err != nil {
		writeError(w, http.StatusBadGateway, "upstream_error", "models fetch failed: "+err.Error())
		return
	}
	if status >= 400 {
		proxyUpstreamError(w, status, raw, hdr)
		return
	}
	if proxyUpstreamBodyIfError(w, status, raw, hdr) {
		return
	}
	s.cache.mu.Lock()
	s.cache.at = time.Now()
	s.cache.data = raw
	s.cache.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(raw)
}

// ---- /v1/chat/completions ----

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 32<<20))
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "cannot read body")
		return
	}
	var req openai.ChatRequest
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
	applyDefaultEffortChat(&req, s.cfg.DefaultEffort)
	stream := req.Stream != nil && *req.Stream

	backend := models.Lookup(req.Model)
	model := models.Normalize(req.Model)

	var upstreamBody any
	var path string
	switch backend {
	case models.BackendResponses:
		upstreamBody = translate.ChatToResponses(req, model)
		path = models.UpstreamPath(backend, model, false)
	case models.BackendMessages:
		upstreamBody = translate.ChatToAnthropic(req, model)
		path = models.UpstreamPath(backend, model, false)
	case models.BackendGemini:
		upstreamBody = translate.ChatToGemini(req, model)
		path = models.UpstreamPath(backend, model, stream)
	default:
		// Chat backend: forward nearly as-is, but normalize effort to
		// reasoning_effort and drop gateway-only aliases.
		upstreamBody = chatPassthroughBody(req, model)
		path = models.UpstreamPath(backend, model, false)
	}

	extra := sessionHeaders(r)
	if stream {
		s.serveChatStream(w, r, backend, path, upstreamBody, extra, model, req)
		return
	}
	status, raw, hdr, err := s.upstream.DoJSON(r.Context(), "POST", path, upstreamBody, extra, false)
	if err != nil {
		writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
		return
	}
	if status >= 400 {
		proxyUpstreamError(w, status, raw, hdr)
		return
	}
	if proxyUpstreamBodyIfError(w, status, raw, hdr) {
		return
	}
	var chat openai.ChatResponse
	switch backend {
	case models.BackendResponses:
		chat, err = translate.ResponsesToChat(raw, req.Model)
	case models.BackendMessages:
		chat, err = translate.AnthropicToChat(raw, req.Model)
	case models.BackendGemini:
		chat, err = translate.GeminiToChat(raw, req.Model)
	default:
		chat, err = translate.ChatPassthrough(raw, req.Model)
	}
	if err != nil {
		writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, chat)
}

// chatPassthroughBody rebuilds the chat body with a normalized model + effort.
func chatPassthroughBody(req openai.ChatRequest, model string) map[string]any {
	b, _ := json.Marshal(req)
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	m["model"] = model
	eff := translate.EffortFromChat(req)
	if e := eff.OpenAIEffort(); e != "" {
		m["reasoning_effort"] = e
	} else {
		delete(m, "reasoning_effort")
	}
	// Drop gateway-only aliases so strict upstreams do not choke.
	delete(m, "reasoningEffort")
	delete(m, "effort")
	delete(m, "thinking")
	delete(m, "thinking_effort")
	if req.Stream != nil && !*req.Stream {
		delete(m, "stream")
		delete(m, "stream_options")
	}
	return m
}

func applyDefaultEffortChat(req *openai.ChatRequest, def string) {
	if translate.EffortFromChat(*req).Valid() {
		return
	}
	if l := effort.Normalize(def); l.Valid() {
		req.ReasoningEffort = l.String()
	}
}

func applyDefaultEffortResponses(req *openai.ResponsesRequest, def string) {
	if translate.EffortFromResponses(*req).Valid() {
		return
	}
	if l := effort.Normalize(def); l.Valid() {
		req.ReasoningEffort = l.String()
	}
}

// ---- streaming ----

func (s *Server) serveChatStream(w http.ResponseWriter, r *http.Request, backend models.Backend, path string, upstreamBody any, extra http.Header, model string, req openai.ChatRequest) {
	// Mark upstream body as streaming where the dialect supports it.
	switch b := upstreamBody.(type) {
	case map[string]any:
		switch backend {
		case models.BackendResponses, models.BackendMessages:
			b["stream"] = true
		case models.BackendChat:
			b["stream"] = true
			// Ask for usage on final chunk if client wants it.
			if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
				b["stream_options"] = map[string]any{"include_usage": true}
			}
		case models.BackendGemini:
			// streaming is encoded in the path (:streamGenerateContent)
		}
	}

	upResp, err := s.upstream.DoStream(r.Context(), "POST", path, upstreamBody, extra)
	if err != nil {
		if ue, ok := err.(*upstream.UpstreamError); ok {
			proxyUpstreamError(w, ue.Status, ue.Body, ue.Header)
			return
		}
		writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
		return
	}
	defer upResp.Body.Close()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	fl, _ := w.(http.Flusher)

	emit := func(chunkJSON string) bool {
		if chunkJSON == "[DONE]" {
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			fl.Flush()
			return true
		}
		_, _ = io.WriteString(w, "data: "+chunkJSON+"\n\n")
		fl.Flush()
		return r.Context().Err() == nil
	}

	var terr error
	displayModel := req.Model
	switch backend {
	case models.BackendResponses:
		terr = translate.TranslateResponsesStreamToChat(upResp.Body, displayModel, emit)
	case models.BackendMessages:
		terr = translate.TranslateAnthropicStreamToChat(upResp.Body, displayModel, emit)
	case models.BackendGemini:
		terr = translate.TranslateGeminiStreamToChat(upResp.Body, displayModel, emit)
	default:
		terr = translate.TranslateChatStream(upResp.Body, displayModel, emit)
	}
	if terr != nil {
		// Headers are already sent: forward the failure as an SSE error
		// event. No silent truncation, no retry.
		if se, ok := translate.AsStreamError(terr); ok {
			emitChatStreamError(w, fl, se.Err)
		} else if r.Context().Err() == nil {
			emitChatStreamError(w, fl, translate.UpstreamFailure(terr))
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		fl.Flush()
		return
	}
	// If the translator swallowed upstream DONE (all but chat passthrough),
	// close with a finish chunk + DONE.
	if backend != models.BackendChat {
		finish := "stop"
		b, _ := json.Marshal(translate.NewChatChunk(displayModel, openai.ChatStreamDelta{}, &finish))
		_, _ = io.WriteString(w, "data: "+string(b)+"\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		fl.Flush()
	}
}
