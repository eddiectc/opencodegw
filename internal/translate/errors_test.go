package translate

import (
	"strings"
	"testing"
)

func TestErrorFromBytesOpenAI(t *testing.T) {
	raw := `{"error":{"message":"quota exceeded","type":"insufficient_quota","code":"quota"}}`
	oe, ok := ErrorFromBytes([]byte(raw))
	if !ok {
		t.Fatalf("expected error detection")
	}
	if oe.Error.Message != "quota exceeded" {
		t.Errorf("message = %q", oe.Error.Message)
	}
	if oe.Error.Type != "insufficient_quota" {
		t.Errorf("type = %q", oe.Error.Type)
	}
}

func TestErrorFromBytesAnthropic(t *testing.T) {
	raw := `{"type":"error","error":{"type":"overloaded_error","message":"busy"}}`
	oe, ok := ErrorFromBytes([]byte(raw))
	if !ok {
		t.Fatalf("expected error detection")
	}
	if oe.Error.Message != "busy" {
		t.Errorf("message = %q", oe.Error.Message)
	}
	if oe.Error.Type != "overloaded_error" {
		t.Errorf("type = %q", oe.Error.Type)
	}
}

func TestErrorFromBytesGemini(t *testing.T) {
	raw := `{"error":{"code":429,"message":"limit","status":"RESOURCE_EXHAUSTED"}}`
	oe, ok := ErrorFromBytes([]byte(raw))
	if !ok {
		t.Fatalf("expected error detection")
	}
	if oe.Error.Message != "limit" {
		t.Errorf("message = %q", oe.Error.Message)
	}
}

func TestErrorFromBytesNonError(t *testing.T) {
	raw := `{"id":"chatcmpl-1","choices":[]}`
	if _, ok := ErrorFromBytes([]byte(raw)); ok {
		t.Fatalf("false positive on success body")
	}
}

func TestDetectSSEErrorEvent(t *testing.T) {
	se, ok := DetectSSEError(SSEEvent{Event: "error", Data: `{"error":{"message":"boom"}}`})
	if !ok || se == nil {
		t.Fatalf("expected error event detection")
	}
	if se.Err.Error.Message != "boom" {
		t.Errorf("message = %q", se.Err.Error.Message)
	}
}

func TestDetectSSEErrorAnthropic(t *testing.T) {
	se, ok := DetectSSEError(SSEEvent{Data: `{"type":"error","error":{"type":"overloaded_error","message":"busy"}}`})
	if !ok || se == nil {
		t.Fatalf("expected anthropic stream error detection")
	}
	if se.Err.Error.Message != "busy" {
		t.Errorf("message = %q", se.Err.Error.Message)
	}
}

func TestDetectSSEErrorResponsesFailed(t *testing.T) {
	se, ok := DetectSSEError(SSEEvent{Data: `{"type":"response.failed","response":{"id":"resp_1","status":"failed","error":{"message":"nope","type":"server_error"}}}`})
	if !ok || se == nil {
		t.Fatalf("expected response.failed detection")
	}
	if se.Err.Error.Message != "nope" {
		t.Errorf("message = %q", se.Err.Error.Message)
	}
}

func TestDetectSSEErrorIgnoresDeltas(t *testing.T) {
	if _, ok := DetectSSEError(SSEEvent{Data: `{"type":"response.output_text.delta","delta":"hi"}`}); ok {
		t.Fatalf("false positive on delta")
	}
	if _, ok := DetectSSEError(SSEEvent{Data: "[DONE]"}); ok {
		t.Fatalf("false positive on DONE")
	}
}

func TestTranslateChatStreamSurfacesError(t *testing.T) {
	sse := "data: {\"error\":{\"message\":\"mid-stream boom\",\"type\":\"server_error\"}}\n\n"
	err := TranslateChatStream(strings.NewReader(sse), "m", func(string) bool { return true })
	se, ok := AsStreamError(err)
	if !ok || se == nil {
		t.Fatalf("expected StreamError, got %v", err)
	}
	if se.Err.Error.Message != "mid-stream boom" {
		t.Errorf("message = %q", se.Err.Error.Message)
	}
}

func TestTranslateAnthropicStreamSurfacesError(t *testing.T) {
	sse := "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n"
	var emitted int
	err := TranslateAnthropicStreamToChat(strings.NewReader(sse), "m", func(string) bool {
		emitted++
		return true
	})
	se, ok := AsStreamError(err)
	if !ok || se == nil {
		t.Fatalf("expected StreamError, got %v", err)
	}
	if emitted != 1 {
		t.Errorf("expected only preamble before error, got %d chunks", emitted)
	}
}

func TestTranslateResponsesStreamSurfacesFailed(t *testing.T) {
	sse := "data: {\"type\":\"response.failed\",\"response\":{\"id\":\"r1\",\"status\":\"failed\"}}\n\n"
	err := TranslateResponsesStreamToChat(strings.NewReader(sse), "m", func(string) bool { return true })
	if _, ok := AsStreamError(err); !ok {
		t.Fatalf("expected StreamError, got %v", err)
	}
}

func TestTranslateGeminiStreamSurfacesError(t *testing.T) {
	sse := "data: {\"error\":{\"code\":500,\"message\":\"gemini down\",\"status\":\"INTERNAL\"}}\n\n"
	err := TranslateGeminiStreamToChat(strings.NewReader(sse), "m", func(string) bool { return true })
	se, ok := AsStreamError(err)
	if !ok || se == nil {
		t.Fatalf("expected StreamError, got %v", err)
	}
	if se.Err.Error.Message != "gemini down" {
		t.Errorf("message = %q", se.Err.Error.Message)
	}
}
