package translate

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/eddiectc/opencodegw/internal/openai"
)

// StreamError is an upstream error detected mid-stream (HTTP 200 SSE carrying
// an error event). It carries an OpenAI-shaped error for downstream emission.
// The gateway never retries upstream; this type exists so callers can forward
// the failure to downstream clients instead of truncating the stream.
type StreamError struct {
	Status int
	Err    openai.ErrorResponse
	Raw    string
}

func (e *StreamError) Error() string {
	if e == nil {
		return "upstream stream error"
	}
	if e.Err.Error.Message != "" {
		return fmt.Sprintf("upstream stream error (status %d): %s", e.Status, e.Err.Error.Message)
	}
	if e.Raw != "" {
		return fmt.Sprintf("upstream stream error (status %d): %s", e.Status, e.Raw)
	}
	return fmt.Sprintf("upstream stream error (status %d)", e.Status)
}

// AsStreamError unwraps err to a *StreamError.
func AsStreamError(err error) (*StreamError, bool) {
	var se *StreamError
	if errors.As(err, &se) {
		return se, true
	}
	return nil, false
}

// ErrorFromBytes normalizes an upstream error body from any backend dialect
// (OpenAI, Anthropic, Gemini) into an OpenAI-shaped ErrorResponse.
// It reports false when the payload does not look like an error.
func ErrorFromBytes(raw []byte) (openai.ErrorResponse, bool) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return openai.ErrorResponse{}, false
	}
	var top map[string]any
	if err := json.Unmarshal(raw, &top); err != nil {
		return openai.ErrorResponse{}, false
	}
	errVal, hasErr := top["error"]
	if !hasErr {
		// Anthropic always sets type=error alongside the error object.
		if t, _ := top["type"].(string); t == "error" {
			msg := fmt.Sprintf("%v", top)
			return openai.ErrorResponse{
				Error: openai.ErrorBody{Message: msg, Type: "upstream_error", Code: "upstream_error"},
			}, true
		}
		return openai.ErrorResponse{}, false
	}
	switch e := errVal.(type) {
	case string:
		msg := strings.TrimSpace(e)
		if msg == "" {
			return openai.ErrorResponse{}, false
		}
		return openai.ErrorResponse{
			Error: openai.ErrorBody{Message: msg, Type: "upstream_error", Code: "upstream_error"},
		}, true
	case map[string]any:
		msg, _ := e["message"].(string)
		if msg == "" {
			// Some dialects use "text" or nest differently.
			msg, _ = e["text"].(string)
		}
		if msg == "" {
			// Fall back to raw error object.
			b, _ := json.Marshal(e)
			msg = string(b)
		}
		typ := firstString(e, "type", "status")
		if typ == "" {
			typ = "upstream_error"
		}
		code := anyOf(e, "code")
		if code == nil {
			code = typ
		}
		return openai.ErrorResponse{
			Error: openai.ErrorBody{Message: msg, Type: typ, Code: code},
		}, true
	default:
		// error is present but in an unexpected shape; preserve it.
		b, _ := json.Marshal(errVal)
		if len(b) == 0 || string(b) == "null" {
			return openai.ErrorResponse{}, false
		}
		return openai.ErrorResponse{
			Error: openai.ErrorBody{Message: string(b), Type: "upstream_error", Code: "upstream_error"},
		}, true
	}
}

// OpenAIErrorBytes serializes an ErrorResponse for SSE emission.
func OpenAIErrorBytes(err openai.ErrorResponse) string {
	b, merr := json.Marshal(err)
	if merr != nil {
		return `{"error":{"message":"upstream error","type":"upstream_error"}}`
	}
	return string(b)
}

// DetectSSEError inspects one upstream SSE event for an embedded error.
// It handles:
//   - event: error (all dialects)
//   - OpenAI chat chunks: {"error": {...}}
//   - Responses stream: {"type":"error",...} / {"type":"response.failed","response":{"error":...}}
//   - Anthropic stream: {"type":"error","error":{...}}
//   - Gemini stream: {"error":{...}}
func DetectSSEError(ev SSEEvent) (*StreamError, bool) {
	if strings.TrimSpace(ev.Data) == "" || ev.Data == "[DONE]" {
		return nil, false
	}
	if strings.EqualFold(strings.TrimSpace(ev.Event), "error") {
		if oe, ok := ErrorFromBytes([]byte(ev.Data)); ok {
			return &StreamError{Status: 502, Err: oe, Raw: ev.Data}, true
		}
		msg := strings.TrimSpace(ev.Data)
		if msg == "" {
			msg = "upstream stream error"
		}
		return &StreamError{
			Status: 502,
			Err: openai.ErrorResponse{
				Error: openai.ErrorBody{Message: msg, Type: "upstream_error", Code: "upstream_error"},
			},
			Raw: ev.Data,
		}, true
	}
	var probe map[string]any
	if err := json.Unmarshal([]byte(ev.Data), &probe); err != nil {
		return nil, false
	}
	// Direct {"error": ...} (OpenAI chat, Gemini).
	if _, hasErr := probe["error"]; hasErr {
		if oe, ok := ErrorFromBytes([]byte(ev.Data)); ok {
			return &StreamError{Status: 502, Err: oe, Raw: ev.Data}, true
		}
	}
	t, _ := probe["type"].(string)
	switch t {
	case "error":
		if oe, ok := ErrorFromBytes([]byte(ev.Data)); ok {
			return &StreamError{Status: 502, Err: oe, Raw: ev.Data}, true
		}
		b, _ := json.Marshal(probe)
		return &StreamError{
			Status: 502,
			Err: openai.ErrorResponse{
				Error: openai.ErrorBody{Message: string(b), Type: "upstream_error", Code: "upstream_error"},
			},
			Raw: ev.Data,
		}, true
	case "response.failed", "response.incomplete":
		// Responses failures may nest the error under response.error.
		if resp, ok := probe["response"].(map[string]any); ok {
			if _, hasErr := resp["error"]; hasErr {
				rb, _ := json.Marshal(map[string]any{"error": resp["error"]})
				if oe, ok := ErrorFromBytes(rb); ok {
					return &StreamError{Status: 502, Err: oe, Raw: ev.Data}, true
				}
			}
			// Status-only failure without details still must surface.
			if st, _ := resp["status"].(string); st == "failed" || t == "response.failed" {
				msg := "upstream response failed"
				if t == "response.incomplete" {
					msg = "upstream response incomplete"
				}
				return &StreamError{
					Status: 502,
					Err: openai.ErrorResponse{
						Error: openai.ErrorBody{Message: msg, Type: t, Code: t},
					},
					Raw: ev.Data,
				}, true
			}
			return nil, false
		}
		if t == "response.failed" {
			if oe, ok := ErrorFromBytes([]byte(ev.Data)); ok {
				return &StreamError{Status: 502, Err: oe, Raw: ev.Data}, true
			}
			return &StreamError{
				Status: 502,
				Err: openai.ErrorResponse{
					Error: openai.ErrorBody{Message: "upstream response failed", Type: t, Code: t},
				},
				Raw: ev.Data,
			}, true
		}
		return nil, false
	default:
		return nil, false
	}
}

// UpstreamFailure converts a mid-stream transport error (broken pipe, timeout,
// truncated body) into an OpenAI-shaped error for downstream SSE emission.
func UpstreamFailure(err error) openai.ErrorResponse {
	msg := "upstream stream interrupted"
	if err != nil && strings.TrimSpace(err.Error()) != "" {
		msg = "upstream stream interrupted: " + strings.TrimSpace(err.Error())
	}
	if len(msg) > 2000 {
		msg = msg[:2000] + "...(truncated)"
	}
	return openai.ErrorResponse{
		Error: openai.ErrorBody{Message: msg, Type: "upstream_error", Code: "upstream_error"},
	}
}

func firstString(m map[string]any, keys ...string) string {
	for _, k := range keys {
		if s, ok := m[k].(string); ok && strings.TrimSpace(s) != "" {
			return s
		}
	}
	return ""
}

func anyOf(m map[string]any, keys ...string) any {
	for _, k := range keys {
		if v, ok := m[k]; ok && v != nil {
			return v
		}
	}
	return nil
}
