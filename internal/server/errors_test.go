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

const generic400 = `{"model":"muse-spark-1.3-contributor","error":{"code":null,"message":"The request contains invalid parameters. Check the request body for any errors or inconsistencies.","param":null,"type":"invalid_request_error"}}`

// An over-window request hitting upstream's detail-free 400 must surface as
// context_length_exceeded (status preserved) so clients compact and retry.
func TestProxyUpstreamErrorRewritesLikelyOverflow(t *testing.T) {
	w := httptest.NewRecorder()
	proxyUpstreamErrorWith(w, 400, []byte(generic400), nil, upstreamFailure{
		estTokens: 1200000,
		window:    1000000,
		model:     "muse-spark-1.3-contributor",
	})
	res := w.Result()
	defer res.Body.Close()
	if res.StatusCode != 400 {
		t.Fatalf("status = %d, want 400", res.StatusCode)
	}
	var oe openai.ErrorResponse
	if err := json.NewDecoder(res.Body).Decode(&oe); err != nil {
		t.Fatal(err)
	}
	if oe.Error.Code != "context_length_exceeded" {
		t.Errorf("code = %v, want context_length_exceeded: %+v", oe.Error.Code, oe.Error)
	}
	if oe.Error.Type != "invalid_request_error" {
		t.Errorf("type = %q, want invalid_request_error", oe.Error.Type)
	}
}

// Same generic 400 on a small request (or unknown window) must pass through
// verbatim: the cause is genuinely unknown (e.g. an unmatched tool call the
// client must fix).
func TestProxyUpstreamErrorKeepsGenericSmallBody(t *testing.T) {
	for _, fail := range []upstreamFailure{
		{estTokens: 1000, window: 1000000, model: "muse-spark-1.3-contributor"},
		{estTokens: 5000000, window: 0, model: "some-future-model"},
		{},
	} {
		w := httptest.NewRecorder()
		proxyUpstreamErrorWith(w, 400, []byte(generic400), nil, fail)
		var body map[string]any
		if err := json.NewDecoder(w.Result().Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		errObj, _ := body["error"].(map[string]any)
		if errObj["message"] != "The request contains invalid parameters. Check the request body for any errors or inconsistencies." {
			t.Errorf("fail=%+v: body rewritten: %v", fail, body)
		}
	}
}

// Specific upstream errors are never rewritten, even over window.
func TestProxyUpstreamErrorNeverRewritesSpecific(t *testing.T) {
	raw := `{"error":{"message":"No function call found for function call output with call_id 'x'","param":"call_id","type":"invalid_request_error"}}`
	w := httptest.NewRecorder()
	proxyUpstreamErrorWith(w, 400, []byte(raw), nil, upstreamFailure{
		estTokens: 2000000,
		window:    1000000,
		model:     "muse-spark-1.3-contributor",
	})
	var oe openai.ErrorResponse
	if err := json.NewDecoder(w.Result().Body).Decode(&oe); err != nil {
		t.Fatal(err)
	}
	if oe.Error.Code == "context_length_exceeded" {
		t.Errorf("specific error rewritten: %+v", oe.Error)
	}
}
