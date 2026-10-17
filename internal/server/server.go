// Package server implements the OpenAI-compatible HTTP gateway.
package server

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/eddiectc/opencodegw/internal/config"
	"github.com/eddiectc/opencodegw/internal/openai"
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
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type, x-opencode-session, x-session-id, x-request-id, OpenAI-Beta")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		// Auth (skip health).
		if r.URL.Path != "/healthz" && r.URL.Path != "/health" {
			if !s.authorize(r) {
				writeError(w, http.StatusUnauthorized, "invalid_api_key", "Invalid or missing gateway API key")
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
	auth := r.Header.Get("Authorization")
	if auth == "" {
		return false
	}
	// Accept "Bearer <key>" or raw key.
	tok := strings.TrimPrefix(auth, "Bearer ")
	tok = strings.TrimSpace(tok)
	return tok == s.cfg.GatewayAPIKey
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
	return h
}
