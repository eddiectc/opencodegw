package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/eddiectc/opencodegw/internal/openai"
)

func TestProxyUpstreamErrorVerbatim(t *testing.T) {
	raw := `{"error":{"message":"quota exceeded","type":"insufficient_quota","code":"quota"}}`
	w := httptest.NewRecorder()
	proxyUpstreamError(w, 429, []byte(raw), http.Header{"Retry-After": []string{"7"}})
	res := w.Result()
	defer res.Body.Close()
	if res.StatusCode != 429 {
		t.Fatalf("status = %d, want 429", res.StatusCode)
	}
	if got := res.Header.Get("Retry-After"); got != "7" {
		t.Errorf("Retry-After = %q, want 7", got)
	}
	var body map[string]any
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	errObj, _ := body["error"].(map[string]any)
	if errObj["message"] != "quota exceeded" {
		t.Errorf("message not preserved: %v", body)
	}
}

func TestProxyUpstreamErrorAnthropicNormalized(t *testing.T) {
	raw := `{"type":"error","error":{"type":"overloaded_error","message":"busy"}}`
	w := httptest.NewRecorder()
	proxyUpstreamError(w, 529, []byte(raw), nil)
	res := w.Result()
	defer res.Body.Close()
	if res.StatusCode != 529 {
		t.Fatalf("status = %d, want 529", res.StatusCode)
	}
	var oe openai.ErrorResponse
	if err := json.NewDecoder(res.Body).Decode(&oe); err != nil {
		t.Fatal(err)
	}
	if oe.Error.Message != "busy" {
		t.Errorf("message = %q", oe.Error.Message)
	}
	if oe.Error.Type != "overloaded_error" {
		t.Errorf("type = %q", oe.Error.Type)
	}
}

func TestProxyUpstreamErrorGeminiNormalized(t *testing.T) {
	raw := `{"error":{"code":429,"message":"limit","status":"RESOURCE_EXHAUSTED"}}`
	w := httptest.NewRecorder()
	proxyUpstreamError(w, 429, []byte(raw), nil)
	var oe openai.ErrorResponse
	if err := json.NewDecoder(w.Result().Body).Decode(&oe); err != nil {
		t.Fatal(err)
	}
	if oe.Error.Message != "limit" {
		t.Errorf("message = %q", oe.Error.Message)
	}
}

func TestProxyUpstreamErrorPlainText(t *testing.T) {
	w := httptest.NewRecorder()
	proxyUpstreamError(w, 502, []byte("upstream exploded"), nil)
	var oe openai.ErrorResponse
	if err := json.NewDecoder(w.Result().Body).Decode(&oe); err != nil {
		t.Fatal(err)
	}
	if oe.Error.Message != "upstream exploded" {
		t.Errorf("message = %q", oe.Error.Message)
	}
}

func TestProxyUpstreamBodyIfError(t *testing.T) {
	w := httptest.NewRecorder()
	handled := proxyUpstreamBodyIfError(w, 200, []byte(`{"error":{"message":"late failure"}}`), nil)
	if !handled {
		t.Fatalf("expected 200-with-error to be handled")
	}
	if w.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", w.Code)
	}
	w2 := httptest.NewRecorder()
	if proxyUpstreamBodyIfError(w2, 200, []byte(`{"id":"ok"}`), nil) {
		t.Fatalf("false positive on success body")
	}
}
