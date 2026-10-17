package upstream

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/eddiectc/opencodegw/internal/config"
)

func TestDoStreamNonSSEJSONSurfacesUpstreamError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"error":{"message":"stream rejected","type":"invalid_request"}}`))
	}))
	defer srv.Close()

	cfg := config.Default()
	cfg.UpstreamBase = srv.URL
	cfg.UpstreamAPIKey = "k"
	c := New(cfg)
	_, err := c.DoStream(context.Background(), "POST", "openai/v1/chat/completions", map[string]any{"stream": true}, nil)
	ue, ok := err.(*UpstreamError)
	if !ok || ue == nil {
		t.Fatalf("expected *UpstreamError, got %T %v", err, err)
	}
	if ue.Status != http.StatusOK {
		t.Errorf("status = %d, want 200 passthrough", ue.Status)
	}
	if len(ue.Body) == 0 {
		t.Fatalf("expected body preserved")
	}
}

func TestDoStreamSSEPassesThrough(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	cfg := config.Default()
	cfg.UpstreamBase = srv.URL
	cfg.UpstreamAPIKey = "k"
	c := New(cfg)
	resp, err := c.DoStream(context.Background(), "POST", "openai/v1/chat/completions", map[string]any{"stream": true}, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
}
