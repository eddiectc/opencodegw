// Package server implements the OpenAI/Anthropic/Gemini-compatible gateway.
package server

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/eddiectc/opencodegw/internal/config"
	"github.com/eddiectc/opencodegw/internal/models"
	"github.com/eddiectc/opencodegw/internal/openai"
	"github.com/eddiectc/opencodegw/internal/translate"
	"github.com/eddiectc/opencodegw/internal/upstream"
)

// Server is the HTTP gateway.
type Server struct {
	cfg      config.Config
	upstream *upstream.Client
	mux      *http.ServeMux
	cache    modelsCache
}

// New creates a Server.
func New(cfg config.Config) *Server {
	s := &Server{
		cfg:      cfg,
		upstream: upstream.New(cfg),
		mux:      http.NewServeMux(),
	}
	s.routes()
	return s
}

func (s *Server) routes() {
	s.mux.HandleFunc("GET /healthz", s.handleHealth)
	s.mux.HandleFunc("GET /health", s.handleHealth)
	s.mux.HandleFunc("GET /v1/models", s.handleModels)
	s.mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	s.mux.HandleFunc("POST /v1/responses", s.handleResponses)
	// Anthropic Messages ingress.
	s.mux.HandleFunc("POST /anthropic/v1/messages", s.handleAnthropicMessages)
	s.mux.HandleFunc("POST /v1/messages", s.handleAnthropicMessages)
	s.mux.HandleFunc("POST /messages", s.handleAnthropicMessages)
	// Gemini GenerateContent ingress (model in path, verb suffix).
	s.mux.HandleFunc("POST /google/v1beta/", s.handleGemini)
	s.mux.HandleFunc("POST /v1beta/", s.handleGemini)
	// Aliases without /v1 for convenience (some clients omit version).
	s.mux.HandleFunc("GET /models", s.handleModels)
	s.mux.HandleFunc("POST /chat/completions", s.handleChatCompletions)
	s.mux.HandleFunc("POST /responses", s.handleResponses)
}

func (s *Server) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		// CORS (permissive; gateway is usually local or self-hosted).
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type, x-opencode-session, x-session-id, x-request-id, OpenAI-Beta, Anthropic-Beta, anthropic-version, anthropic-dangerous-direct-browser-access, x-api-key, x-goog-api-key")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		// Auth (skip health). 401s are shaped per ingress dialect so native
		// SDKs parse them.
		if r.URL.Path != "/healthz" && r.URL.Path != "/health" {
			if !s.authorize(r) {
				writeAuthError(w, r.URL.Path)
				return
			}
		}
		s.mux.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
	})
}

func (s *Server) authorize(r *http.Request) bool {
	if s.cfg.GatewayAPIKey == "" {
		return true
	}
	// OpenAI-style Authorization header.
	if auth := r.Header.Get("Authorization"); auth != "" {
		// Accept "Bearer <key>" or raw key.
		tok := strings.TrimSpace(strings.TrimPrefix(auth, "Bearer "))
		if tok == s.cfg.GatewayAPIKey {
			return true
		}
	}
	// Anthropic SDKs send x-api-key; accept it when it carries the gateway key.
	if v := strings.TrimSpace(r.Header.Get("x-api-key")); v != "" && v == s.cfg.GatewayAPIKey {
		return true
	}
	// Google SDKs send x-goog-api-key or ?key=.
	if v := strings.TrimSpace(r.Header.Get("x-goog-api-key")); v != "" && v == s.cfg.GatewayAPIKey {
		return true
	}
	if v := strings.TrimSpace(r.URL.Query().Get("key")); v != "" && v == s.cfg.GatewayAPIKey {
		return true
	}
	return false
}

// writeAuthError emits a 401 in the ingress dialect's error shape.
func writeAuthError(w http.ResponseWriter, path string) {
	const msg = "Invalid or missing gateway API key"
	switch ingressDialect(path) {
	case "anthropic":
		writeAnthropicError(w, http.StatusUnauthorized, "authentication_error", msg)
	case "gemini":
		writeGeminiError(w, http.StatusUnauthorized, "UNAUTHENTICATED", msg)
	default:
		writeError(w, http.StatusUnauthorized, "invalid_api_key", msg)
	}
}

// ingressDialect classifies a request path by ingress dialect for 401
// shaping: anthropic, gemini, or openai (default).
func ingressDialect(path string) string {
	switch {
	case path == "/anthropic/v1/messages" || path == "/v1/messages" || path == "/messages":
		return "anthropic"
	case strings.HasPrefix(path, "/google/v1beta/") || strings.HasPrefix(path, "/v1beta/"):
		return "gemini"
	default:
		return "openai"
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, code, msg string) {
	writeJSON(w, status, openai.ErrorResponse{
		Error: openai.ErrorBody{Message: msg, Type: code, Code: code},
	})
}

// classifyUpstreamBody snapshots the failure context for an upstream request
// body: estimated input tokens plus the model's known window. displayModel
// is echoed in rewritten errors; normalized is used for window lookup.
func classifyUpstreamBody(upstreamBody any, displayModel, normalized string) upstreamFailure {
	return upstreamFailure{
		estTokens: translate.EstimateTokens(upstreamBody),
		window:    models.ContextWindow(normalized),
		model:     displayModel,
	}
}

// sessionHeaders builds the extra headers forwarded upstream:
// always ensure an x-opencode-session is present for routing/prompt caching.
func sessionHeaders(r *http.Request) http.Header {
	h := http.Header{}
	if v := r.Header.Get("x-opencode-session"); v != "" {
		h.Set("X-Opencode-Session", v)
	} else if v := r.Header.Get("x-session-id"); v != "" {
		h.Set("X-Opencode-Session", v)
	} else {
		h.Set("X-Opencode-Session", "gw-"+newReqID())
	}
	if v := r.Header.Get("x-request-id"); v != "" {
		h.Set("X-Request-Id", v)
	}
	if v := r.Header.Get("Anthropic-Beta"); v != "" {
		h.Set("Anthropic-Beta", v)
	}
	// Native version/beta headers ride upstream so beta features survive
	// passthrough. Client auth keys (x-api-key, x-goog-api-key, ?key) are
	// gateway credentials and are never forwarded: upstream uses Bearer.
	if v := r.Header.Get("anthropic-version"); v != "" {
		h.Set("Anthropic-Version", v)
	}
	if v := r.Header.Get("OpenAI-Beta"); v != "" {
		h.Set("OpenAI-Beta", v)
	}
	return h
}
