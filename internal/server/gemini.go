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

// ---- /google/v1beta/models/<model>:generateContent etc ----

// Gemini ingress mirrors the Google AI shape:
//
//	POST /google/v1beta/models/<model>:generateContent
//	POST /google/v1beta/models/<model>:streamGenerateContent?alt=sse
//	POST /v1beta/models/<model>:generateContent (+ stream variant)
//
// Gemini-native models pass through losslessly from the raw body; other
// backends fan out through the canonical chat translators and wrap back to
// Gemini shape.
func (s *Server) handleGemini(w http.ResponseWriter, r *http.Request) {
	model, stream, ok := parseGeminiPath(r.URL.Path)
	if !ok || model == "" {
		writeGeminiError(w, http.StatusNotFound, "NOT_FOUND", "unknown Gemini path: "+r.URL.Path)
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 32<<20))
	if err != nil {
		writeGeminiError(w, http.StatusBadRequest, "INVALID_ARGUMENT", "cannot read body")
		return
	}
	var probe struct {
		Stream *bool `json:"stream,omitempty"`
	}
	_ = json.Unmarshal(body, &probe)
	if probe.Stream != nil {
		stream = *probe.Stream
	}
	defEff := effort.Normalize(s.cfg.DefaultEffort)

	backend := models.Lookup(model)
	normalized := models.Normalize(model)
	extra := sessionHeaders(r)

	if stream {
		s.serveGeminiStream(w, r, backend, normalized, model, body, defEff, extra)
		return
	}

	if backend == models.BackendGemini {
		upBody := translate.GeminiNativeBody(body, normalized, defEff)
		path := models.UpstreamPath(backend, normalized, false)
		fail := classifyUpstreamBody(upBody, model, normalized)
		status, raw, hdr, err := s.upstream.DoJSON(r.Context(), "POST", path, upBody, extra, false)
		if err != nil {
			writeGeminiError(w, http.StatusBadGateway, "UNAVAILABLE", err.Error())
			return
		}
		if status >= 400 {
			proxyUpstreamGeminiErrorWith(w, status, raw, hdr, fail)
			return
		}
		if proxyUpstreamGeminiBodyIfError(w, status, raw, hdr) {
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(raw)
		return
	}

	// Cross-dialect: Gemini -> canonical chat -> backend -> Gemini result.
	// generateContent is unary, so the chat request must be unary too.
	var rawMap map[string]any
	_ = json.Unmarshal(body, &rawMap)
	if rawMap == nil {
		rawMap = map[string]any{}
	}
	rawMap["stream"] = false
	xbody, _ := json.Marshal(rawMap)
	chatReq := translate.GeminiToChatRequest(xbody, model, defEff)
	chat, perr := s.doChatBackend(r, backend, normalized, chatReq, extra)
	if perr != nil {
		if ue, ok := perr.(*upstreamErr); ok {
			proxyUpstreamGeminiErrorWith(w, ue.status, ue.body, ue.header, ue.fail)
			return
		}
		writeGeminiError(w, http.StatusBadGateway, "INTERNAL", perr.Error())
		return
	}
	writeJSON(w, http.StatusOK, translate.ChatToGeminiResult(chat, model))
}

// parseGeminiPath extracts the model and stream flag from a Gemini ingress
// path. Accepts /google/v1beta/models/<m>:generateContent,
// ...:streamGenerateContent, and the /v1beta/... short forms (with or without
// the models/ segment).
func parseGeminiPath(p string) (model string, stream bool, ok bool) {
	rest := strings.TrimPrefix(p, "/google/v1beta/")
	if rest == p {
		rest = strings.TrimPrefix(p, "/v1beta/")
		if rest == p {
			return "", false, false
		}
	}
	rest = strings.TrimPrefix(rest, "models/")
	switch {
	case strings.HasSuffix(rest, ":generateContent"):
		model = strings.TrimSuffix(rest, ":generateContent")
	case strings.HasSuffix(rest, ":streamGenerateContent"):
		model = strings.TrimSuffix(rest, ":streamGenerateContent")
		stream = true
	default:
		return "", false, false
	}
	model = strings.Trim(model, "/")
	if model == "" || strings.Contains(model, "/") || strings.Contains(model, ":") {
		return "", false, false
	}
	return model, stream, true
}

// serveGeminiStream serves streamGenerateContent. Native backends relay
// upstream SSE verbatim; others frame canonical chat deltas as Gemini
// payloads. Setup failures proxy as JSON before SSE headers; mid-stream
// failures ride the stream as data: {"error":...}.
func (s *Server) serveGeminiStream(w http.ResponseWriter, r *http.Request, backend models.Backend, model, display string, rawBody []byte, defEff effort.Level, extra http.Header) {
	var upstreamBody any
	var path string
	if backend == models.BackendGemini {
		upBody := translate.GeminiNativeBody(rawBody, model, defEff)
		upstreamBody = upBody
		path = models.UpstreamPath(backend, model, true)
	} else {
		chatReq := translate.GeminiToChatRequest(rawBody, display, defEff)
		switch backend {
		case models.BackendMessages:
			m := translate.ChatToAnthropic(chatReq, model)
			m["stream"] = true
			upstreamBody = m
			path = models.UpstreamPath(backend, model, false)
		case models.BackendResponses:
			m := translate.ChatToResponses(chatReq, model)
			m["stream"] = true
			upstreamBody = m
			path = models.UpstreamPath(backend, model, false)
		default:
			b, _ := json.Marshal(chatReq)
			var req openai.ChatRequest
			_ = json.Unmarshal(b, &req)
			m := translate.ChatNativeBody(b, req, model, defEff)
			m["stream"] = true
			upstreamBody = m
			path = models.UpstreamPath(backend, model, false)
		}
	}
	upResp, err := s.upstream.DoStream(r.Context(), "POST", path, upstreamBody, extra)
	if err != nil {
		if ue, ok := err.(*upstream.UpstreamError); ok {
			proxyUpstreamGeminiErrorWith(w, ue.Status, ue.Body, ue.Header, classifyUpstreamBody(upstreamBody, display, model))
			return
		}
		writeGeminiError(w, http.StatusBadGateway, "UNAVAILABLE", err.Error())
		return
	}
	defer upResp.Body.Close()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	fl, _ := w.(http.Flusher)
	emitData := func(data string) bool {
		_, _ = io.WriteString(w, "data: "+data+"\n\n")
		fl.Flush()
		return r.Context().Err() == nil
	}

	if backend == models.BackendGemini {
		scanErr := translate.ScanSSE(upResp.Body, func(ev translate.SSEEvent) bool {
			return emitData(ev.Data)
		})
		if scanErr != nil && r.Context().Err() == nil {
			b, _ := json.Marshal(geminiErrorBody("UNAVAILABLE", scanErr.Error()))
			emitData(string(b))
		}
		return
	}

	// Cross-dialect: translate upstream SSE to canonical chat chunks and
	// frame directly as Gemini payloads (no intermediate pipe: the framer
	// consumes chunk JSON synchronously, so client disconnects cannot
	// deadlock a writer goroutine).
	fr := translate.NewGeminiStreamFramer()
	emitChunk := func(chunkJSON string) bool {
		if chunkJSON == "[DONE]" {
			return true
		}
		return fr.Push(chunkJSON, emitData)
	}
	var terr error
	switch backend {
	case models.BackendMessages:
		terr = translate.TranslateAnthropicStreamToChat(upResp.Body, display, emitChunk)
	case models.BackendResponses:
		terr = translate.TranslateResponsesStreamToChat(upResp.Body, display, emitChunk)
	default:
		terr = translate.TranslateChatStream(upResp.Body, display, emitChunk)
	}
	if terr != nil {
		if se, ok := translate.AsStreamError(terr); ok {
			b, _ := json.Marshal(geminiErrorFromOpenAI(se.Err))
			emitData(string(b))
		} else if r.Context().Err() == nil {
			b, _ := json.Marshal(geminiErrorBody("UNAVAILABLE", terr.Error()))
			emitData(string(b))
		}
		return
	}
	fr.Finish(emitData)
}
