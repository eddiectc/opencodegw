package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/eddiectc/opencodegw/internal/config"
)

func newStubGateway(t *testing.T, h http.HandlerFunc) (*Server, func()) {
	t.Helper()
	up := httptest.NewServer(h)
	cfg := config.Default()
	cfg.UpstreamBase = up.URL
	cfg.UpstreamAPIKey = "test-key"
	s := New(cfg)
	return s, up.Close
}

func doGateway(t *testing.T, s *Server, method, path, body string, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	var rdr io.Reader
	if body != "" {
		rdr = strings.NewReader(body)
	}
	req := httptest.NewRequest(method, path, rdr)
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)
	return w
}

// Anthropic native: raw body passes through with model normalized and native
// thinking budget exact; upstream reply relays with model echo fixed.
func TestAnthropicNativePassthrough(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	s, done := newStubGateway(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}`))
	})
	defer done()

	w := doGateway(t, s, "POST", "/anthropic/v1/messages", `{
		"model": "CLAUDE-HAIKU-5-5",
		"max_tokens": 100,
		"thinking": {"type": "enabled", "budget_tokens": 12345},
		"temperature": 0.5,
		"messages": [{"role": "user", "content": "hi"}]
	}`, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	if gotPath != "/anthropic/v1/messages" {
		t.Errorf("upstream path = %q", gotPath)
	}
	if gotBody["model"] != "claude-haiku-5-5" {
		t.Errorf("model = %v, want normalized", gotBody["model"])
	}
	th, _ := gotBody["thinking"].(map[string]any)
	if th["budget_tokens"] != float64(12345) {
		t.Errorf("budget = %v, want exact 12345", th["budget_tokens"])
	}
	if _, ok := gotBody["temperature"]; ok {
		t.Errorf("temperature must drop with thinking: %v", gotBody)
	}
	var out map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &out)
	if out["model"] != "CLAUDE-HAIKU-5-5" {
		t.Errorf("model echo = %v, want display id", out["model"])
	}
}

// Anthropic cross-dialect: responses backend fanned via canonical chat and
// wrapped as an Anthropic message.
func TestAnthropicCrossDialectWrapsChat(t *testing.T) {
	var gotPath string
	s, done := newStubGateway(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","model":"grok-4.7","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`))
	})
	defer done()

	w := doGateway(t, s, "POST", "/v1/messages", `{
		"model": "grok-4.7",
		"max_tokens": 50,
		"messages": [{"role": "user", "content": "hi"}]
	}`, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	if gotPath != "/openai/v1/responses" {
		t.Errorf("upstream path = %q, want responses backend", gotPath)
	}
	var out map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &out)
	if out["type"] != "message" || out["stop_reason"] != "end_turn" {
		t.Errorf("wrapped result = %v", out)
	}
	content, _ := out["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("content = %v", out["content"])
	}
	if cm, _ := content[0].(map[string]any); cm["text"] != "hello" {
		t.Errorf("text = %v", content[0])
	}
}

// Anthropic errors come back in Anthropic shape, verbatim when native.
func TestAnthropicErrorShape(t *testing.T) {
	s, done := newStubGateway(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(529)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"busy"}}`))
	})
	defer done()

	w := doGateway(t, s, "POST", "/messages", `{"model":"claude-haiku-5-5","max_tokens":5,"messages":[]}`, nil)
	if w.Code != 529 {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	var out map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &out)
	if out["type"] != "error" {
		t.Errorf("error shape = %v", out)
	}
	if em, _ := out["error"].(map[string]any); em["type"] != "overloaded_error" || em["message"] != "busy" {
		t.Errorf("error body = %v", out)
	}
}

// Gemini native: body without model key, native level preserved, reply raw.
func TestGeminiNativePassthrough(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	s, done := newStubGateway(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`))
	})
	defer done()

	w := doGateway(t, s, "POST", "/google/v1beta/models/gemini-4-argon-contributor:generateContent", `{
		"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
		"generationConfig": {"thinkingConfig": {"thinkingLevel": "HIGH"}},
		"safetySettings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
		"effort": "low"
	}`, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	if gotPath != "/google/v1beta/models/gemini-4-argon-contributor:generateContent" {
		t.Errorf("upstream path = %q", gotPath)
	}
	if _, ok := gotBody["model"]; ok {
		t.Errorf("model key must not reach GenerateContent: %v", gotBody)
	}
	gc, _ := gotBody["generationConfig"].(map[string]any)
	tc, _ := gc["thinkingConfig"].(map[string]any)
	if tc["thinkingLevel"] != "HIGH" {
		t.Errorf("native level overwritten: %v", tc)
	}
	if gotBody["safetySettings"] == nil {
		t.Errorf("safetySettings lost: %v", gotBody)
	}
	if _, ok := gotBody["effort"]; ok {
		t.Errorf("effort alias leaked: %v", gotBody)
	}
	var out map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &out)
	if out["candidates"] == nil {
		t.Errorf("reply not relayed: %v", out)
	}
}

// Gemini cross-dialect wraps a chat result as generateContent.
func TestGeminiCrossDialectWrapsChat(t *testing.T) {
	var gotPath string
	s, done := newStubGateway(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"kimi-k3","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	})
	defer done()

	w := doGateway(t, s, "POST", "/v1beta/models/kimi-k3:generateContent", `{
		"contents": [{"role": "user", "parts": [{"text": "hi"}]}]
	}`, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	if gotPath != "/openai/v1/chat/completions" {
		t.Errorf("upstream path = %q", gotPath)
	}
	var out map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &out)
	cands, _ := out["candidates"].([]any)
	if len(cands) != 1 {
		t.Fatalf("candidates = %v", out)
	}
}

// Gemini verb parsing covers stream + short forms and rejects junk.
func TestParseGeminiPath(t *testing.T) {
	cases := []struct {
		in     string
		model  string
		stream bool
		ok     bool
	}{
		{"/google/v1beta/models/m1:generateContent", "m1", false, true},
		{"/google/v1beta/models/m1:streamGenerateContent", "m1", true, true},
		{"/v1beta/models/m1:generateContent", "m1", false, true},
		{"/v1beta/models/m1:streamGenerateContent", "m1", true, true},
		{"/v1beta/m1:generateContent", "m1", false, true},
		{"/google/v1beta/models/:generateContent", "", false, false},
		{"/google/v1beta/models/a/b:generateContent", "", false, false},
		{"/google/v1beta/models/m1:embedContent", "", false, false},
		{"/openai/v1/chat/completions", "", false, false},
	}
	for _, c := range cases {
		m, s, ok := parseGeminiPath(c.in)
		if m != c.model || s != c.stream || ok != c.ok {
			t.Errorf("%s -> (%q,%v,%v), want (%q,%v,%v)", c.in, m, s, ok, c.model, c.stream, c.ok)
		}
	}
}

// Gemini errors come back in Google shape.
func TestGeminiErrorShape(t *testing.T) {
	s, done := newStubGateway(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"slow down","type":"rate_limit_error"}}`))
	})
	defer done()

	w := doGateway(t, s, "POST", "/google/v1beta/models/kimi-k3:generateContent", `{"contents":[]}`, nil)
	if w.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	var out map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &out)
	em, _ := out["error"].(map[string]any)
	if em["message"] != "slow down" || em["status"] != "RESOURCE_EXHAUSTED" || em["code"] != float64(429) {
		t.Errorf("gemini error = %v", out)
	}
}

// doChatBackend must route Responses models to the responses endpoint with a
// valid Responses body (regression: the case previously fell to chat default).
func TestDoChatBackendResponsesRoute(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	s, done := newStubGateway(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","model":"grok-4.7","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	})
	defer done()

	// Reach doChatBackend via Anthropic ingress on a responses-native model.
	w := doGateway(t, s, "POST", "/anthropic/v1/messages", `{
		"model": "grok-4.7", "max_tokens": 10,
		"messages": [{"role": "user", "content": "hi"}]
	}`, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	if gotPath != "/openai/v1/responses" {
		t.Errorf("upstream path = %q, want openai/v1/responses", gotPath)
	}
	if gotBody["input"] == nil || gotBody["messages"] != nil {
		t.Errorf("upstream body not Responses-shaped: %v", gotBody)
	}
}

// Gateway auth accepts native headers and shapes 401s per dialect.
func TestNativeAuthAnd401Shapes(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{}`))
	}))
	defer up.Close()
	cfg := config.Default()
	cfg.UpstreamBase = up.URL
	cfg.UpstreamAPIKey = "test-key"
	cfg.GatewayAPIKey = "gw-secret"
	s := New(cfg)

	// x-api-key accepted on Anthropic ingress
	w := doGateway(t, s, "POST", "/anthropic/v1/messages", `{"model":"m","max_tokens":1,"messages":[]}`, map[string]string{"x-api-key": "gw-secret"})
	if w.Code == http.StatusUnauthorized {
		t.Errorf("x-api-key rejected: %s", w.Body.String())
	}
	// x-goog-api-key accepted on Gemini ingress
	w = doGateway(t, s, "POST", "/google/v1beta/models/m:generateContent", `{"contents":[]}`, map[string]string{"x-goog-api-key": "gw-secret"})
	if w.Code == http.StatusUnauthorized {
		t.Errorf("x-goog-api-key rejected: %s", w.Body.String())
	}
	// ?key= accepted on Gemini ingress
	w = doGateway(t, s, "POST", "/google/v1beta/models/m:generateContent?key=gw-secret", `{"contents":[]}`, nil)
	if w.Code == http.StatusUnauthorized {
		t.Errorf("?key rejected: %s", w.Body.String())
	}
	// Wrong key: Anthropic-shaped 401
	w = doGateway(t, s, "POST", "/anthropic/v1/messages", `{}`, map[string]string{"x-api-key": "wrong"})
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("anthropic 401 = %d", w.Code)
	}
	var ao map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &ao)
	if ao["type"] != "error" {
		t.Errorf("anthropic 401 shape = %v", ao)
	}
	// Wrong key: Gemini-shaped 401
	w = doGateway(t, s, "POST", "/google/v1beta/models/m:generateContent", `{}`, nil)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("gemini 401 = %d", w.Code)
	}
	var go_ map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &go_)
	gem, _ := go_["error"].(map[string]any)
	if gem["status"] != "UNAUTHENTICATED" || gem["code"] != float64(401) {
		t.Errorf("gemini 401 shape = %v", go_)
	}
	// Wrong key: OpenAI-shaped 401 (unchanged)
	w = doGateway(t, s, "POST", "/v1/chat/completions", `{}`, nil)
	var oo map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &oo)
	if _, ok := oo["error"]; !ok {
		t.Errorf("openai 401 shape = %v", oo)
	}
}

// Anthropic native streaming relays upstream SSE verbatim and forwards
// native version headers.
func TestAnthropicNativeStreamRelay(t *testing.T) {
	var gotBeta, gotVer string
	s, done := newStubGateway(t, func(w http.ResponseWriter, r *http.Request) {
		gotBeta = r.Header.Get("Anthropic-Beta")
		gotVer = r.Header.Get("Anthropic-Version")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: message_start\ndata: {\"type\":\"message_start\"}\n\n"))
		_, _ = w.Write([]byte("event: ping\ndata: {\"type\":\"ping\"}\n\n"))
	})
	defer done()

	w := doGateway(t, s, "POST", "/anthropic/v1/messages", `{
		"model": "claude-haiku-5-5", "max_tokens": 10, "stream": true,
		"messages": [{"role": "user", "content": "hi"}]
	}`, map[string]string{"Anthropic-Beta": "b1", "anthropic-version": "2023-06-01"})
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	if ct := w.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("content-type = %q", ct)
	}
	if body := w.Body.String(); !strings.Contains(body, "message_start") || !strings.Contains(body, `"type":"ping"`) {
		t.Errorf("stream not relayed verbatim: %q", body)
	}
	if gotBeta != "b1" || gotVer != "2023-06-01" {
		t.Errorf("native headers not forwarded: beta=%q ver=%q", gotBeta, gotVer)
	}
}

// Anthropic cross-dialect streaming frames chat deltas as Anthropic events.
func TestAnthropicCrossDialectStream(t *testing.T) {
	s, done := newStubGateway(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"kimi-k3\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hel\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"kimi-k3\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	})
	defer done()

	w := doGateway(t, s, "POST", "/v1/messages", `{
		"model": "kimi-k3", "max_tokens": 10, "stream": true,
		"messages": [{"role": "user", "content": "hi"}]
	}`, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	body := w.Body.String()
	for _, want := range []string{"message_start", "text_delta", "message_delta", "message_stop"} {
		if !strings.Contains(body, want) {
			t.Errorf("framed stream missing %q:\n%s", want, body)
		}
	}
}

// Anthropic setup failure proxies as Anthropic JSON before SSE headers.
func TestAnthropicStreamSetupError(t *testing.T) {
	s, done := newStubGateway(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"bad input"}}`))
	})
	defer done()

	w := doGateway(t, s, "POST", "/messages", `{
		"model": "claude-haiku-5-5", "max_tokens": 10, "stream": true, "messages": []
	}`, nil)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	var out map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &out)
	if out["type"] != "error" {
		t.Errorf("setup error shape = %v", out)
	}
}

// Gemini native streaming relays and strips the gateway stream hint.
func TestGeminiNativeStreamRelay(t *testing.T) {
	var gotBody map[string]any
	s, done := newStubGateway(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		if !strings.HasSuffix(r.URL.Path, ":streamGenerateContent") {
			t.Errorf("upstream path = %q, want streamGenerateContent", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]}}]}\n\n"))
	})
	defer done()

	w := doGateway(t, s, "POST", "/google/v1beta/models/gemini-4-argon-contributor:streamGenerateContent?alt=sse", `{
		"contents": [{"role": "user", "parts": [{"text": "hi"}]}]
	}`, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	if body := w.Body.String(); !strings.Contains(body, `"text":"hi"`) {
		t.Errorf("stream not relayed: %q", body)
	}
	if _, ok := gotBody["stream"]; ok {
		t.Errorf("gateway stream hint leaked upstream: %v", gotBody)
	}
}

// Gemini cross-dialect streaming frames chat deltas as Gemini payloads.
func TestGeminiCrossDialectStream(t *testing.T) {
	s, done := newStubGateway(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"kimi-k3\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	})
	defer done()

	w := doGateway(t, s, "POST", "/v1beta/models/kimi-k3:streamGenerateContent", `{
		"contents": [{"role": "user", "parts": [{"text": "hi"}]}]
	}`, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	body := w.Body.String()
	if !strings.Contains(body, `"text":"hi"`) || !strings.Contains(body, `"finishReason":"STOP"`) {
		t.Errorf("framed stream wrong:\n%s", body)
	}
}
