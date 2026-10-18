package translate

import (
	"encoding/json"
	"testing"

	"github.com/eddiectc/opencodegw/internal/effort"
	"github.com/eddiectc/opencodegw/internal/openai"
)

func decodeChat(t *testing.T, raw string) openai.ChatRequest {
	t.Helper()
	var req openai.ChatRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatal(err)
	}
	return req
}

func decodeResponses(t *testing.T, raw string) openai.ResponsesRequest {
	t.Helper()
	var req openai.ResponsesRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatal(err)
	}
	return req
}

// Chat native must preserve unknown/native fields verbatim: logprobs,
// response_format, seed, stop, modalities, custom keys.
func TestChatNativeBodyPreservesUnknowns(t *testing.T) {
	raw := `{
		"model": "KIMI-K3",
		"messages": [{"role":"user","content":"hi"}],
		"temperature": 0.7,
		"logprobs": true,
		"top_logprobs": 5,
		"seed": 42,
		"stop": ["END"],
		"response_format": {"type":"json_object"},
		"modalities": ["text"],
		"custom_future_key": {"nested": [1,2,3]},
		"stream": false,
		"stream_options": {"include_usage": true}
	}`
	req := decodeChat(t, raw)
	body := ChatNativeBody([]byte(raw), req, "kimi-k3", effort.Unspecified)

	if body["model"] != "kimi-k3" {
		t.Errorf("model = %v, want kimi-k3", body["model"])
	}
	// verbatim preservation
	if body["logprobs"] != true {
		t.Errorf("logprobs lost: %v", body)
	}
	if body["top_logprobs"] != float64(5) {
		t.Errorf("top_logprobs lost: %v", body["top_logprobs"])
	}
	if body["seed"] != float64(42) {
		t.Errorf("seed lost: %v", body["seed"])
	}
	if body["custom_future_key"] == nil {
		t.Errorf("custom_future_key lost: %v", body)
	}
	rf, _ := body["response_format"].(map[string]any)
	if rf["type"] != "json_object" {
		t.Errorf("response_format lost: %v", body["response_format"])
	}
	// stream:false drops stream + stream_options
	if _, ok := body["stream"]; ok {
		t.Errorf("stream:false not dropped: %v", body)
	}
	if _, ok := body["stream_options"]; ok {
		t.Errorf("stream_options not dropped on stream:false: %v", body)
	}
}

// Chat native must map canonical effort to reasoning_effort and drop all
// gateway aliases plus the Responses-only reasoning object. Precedence is
// top-level aliases first, then the reasoning object, then thinking.
func TestChatNativeBodyEffortAndAliases(t *testing.T) {
	raw := `{
		"model": "kimi-k3",
		"messages": [{"role":"user","content":"hi"}],
		"effort": "max",
		"thinking": {"effort": "low"},
		"reasoning": {"effort": "high", "summary": "auto"}
	}`
	req := decodeChat(t, raw)
	body := ChatNativeBody([]byte(raw), req, "kimi-k3", effort.Unspecified)

	// struct path: top-level effort beats reasoning object and thinking.
	if body["reasoning_effort"] != "max" {
		t.Errorf("reasoning_effort = %v, want max (top-level alias first)", body["reasoning_effort"])
	}
	for _, k := range []string{"reasoningEffort", "effort", "thinking", "thinking_effort", "reasoning"} {
		if _, ok := body[k]; ok {
			t.Errorf("alias %q leaked upstream: %v", k, body)
		}
	}
}

// Default effort applies when the client sent none.
func TestChatNativeBodyDefaultEffort(t *testing.T) {
	raw := `{"model":"kimi-k3","messages":[{"role":"user","content":"hi"}]}`
	req := decodeChat(t, raw)
	body := ChatNativeBody([]byte(raw), req, "kimi-k3", effort.Max)
	if body["reasoning_effort"] != "max" {
		t.Errorf("default effort not applied: %v", body)
	}
	body = ChatNativeBody([]byte(raw), req, "kimi-k3", effort.Unspecified)
	if _, ok := body["reasoning_effort"]; ok {
		t.Errorf("reasoning_effort should be absent: %v", body)
	}
}

// Responses native must preserve reasoning subfields while merging effort.
func TestResponsesNativeBodyPreservesReasoningSubfields(t *testing.T) {
	raw := `{
		"model": "muse-spark-1.3-contributor",
		"input": "hi",
		"reasoning": {"effort": "low", "summary": "auto", "max_tokens": 5000, "exclude": true, "future_key": "keep"}
	}`
	req := decodeResponses(t, raw)
	body := ResponsesNativeBody([]byte(raw), req, "muse-spark-1.3-contributor", effort.Unspecified)

	rm, ok := body["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning lost: %v", body)
	}
	if rm["effort"] != "low" {
		t.Errorf("effort = %v, want low", rm["effort"])
	}
	if rm["summary"] != "auto" {
		t.Errorf("summary lost: %v", rm)
	}
	if rm["max_tokens"] != float64(5000) {
		t.Errorf("max_tokens lost: %v", rm)
	}
	if rm["exclude"] != true {
		t.Errorf("exclude lost: %v", rm)
	}
	if rm["future_key"] != "keep" {
		t.Errorf("future_key lost: %v", rm)
	}
}

// Effort alias merges into reasoning.effort, preserving other reasoning keys.
func TestResponsesNativeBodyMergesEffortAlias(t *testing.T) {
	raw := `{
		"model": "grok-4.7",
		"input": "hi",
		"reasoning": {"summary": "detailed"},
		"reasoning_effort": "max"
	}`
	req := decodeResponses(t, raw)
	body := ResponsesNativeBody([]byte(raw), req, "grok-4.7", effort.Unspecified)

	rm, ok := body["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning lost: %v", body)
	}
	if rm["effort"] != "max" {
		t.Errorf("effort = %v, want max", rm["effort"])
	}
	if rm["summary"] != "detailed" {
		t.Errorf("summary lost: %v", rm)
	}
	if _, ok := body["reasoning_effort"]; ok {
		t.Errorf("reasoning_effort alias leaked: %v", body)
	}
}

// effort=none drops the reasoning block; unspecified leaves it untouched.
// Explicit client effort always wins over the server default.
func TestResponsesNativeBodyNoneAndUnspecified(t *testing.T) {
	// Default none with no explicit effort: drop client reasoning block.
	raw := `{"model":"grok-4.7","input":"hi","reasoning":{"summary":"auto"}}`
	req := decodeResponses(t, raw)
	body := ResponsesNativeBody([]byte(raw), req, "grok-4.7", effort.None)
	if _, ok := body["reasoning"]; ok {
		t.Errorf("default none should drop reasoning: %v", body)
	}

	// Explicit effort wins over default none.
	rawExp := `{"model":"grok-4.7","input":"hi","reasoning":{"effort":"high","summary":"auto"}}`
	reqExp := decodeResponses(t, rawExp)
	bodyExp := ResponsesNativeBody([]byte(rawExp), reqExp, "grok-4.7", effort.None)
	rmExp, _ := bodyExp["reasoning"].(map[string]any)
	if rmExp == nil || rmExp["effort"] != "high" {
		t.Errorf("explicit high should win over default none: %v", bodyExp)
	}

	// Unspecified default + client block without effort: untouched.
	raw2 := `{"model":"grok-4.7","input":"hi","reasoning":{"summary":"auto"}}`
	req2 := decodeResponses(t, raw2)
	body2 := ResponsesNativeBody([]byte(raw2), req2, "grok-4.7", effort.Unspecified)
	rm, ok := body2["reasoning"].(map[string]any)
	if !ok || rm["summary"] != "auto" {
		t.Errorf("unspecified should leave reasoning untouched: %v", body2)
	}
	if _, ok := rm["effort"]; ok {
		t.Errorf("unspecified should not inject effort: %v", rm)
	}
}

// Responses native must preserve advanced top-level fields and unknowns.
func TestResponsesNativeBodyPreservesAdvancedFields(t *testing.T) {
	raw := `{
		"model": "GPT-6-LUNA",
		"input": "hi",
		"instructions": "be nice",
		"text": {"format": {"type": "json_schema", "name": "x", "strict": true}},
		"max_tool_calls": 5,
		"previous_response_id": "resp_123",
		"store": false,
		"user": "u1",
		"temperature": 0.5,
		"top_p": 0.9,
		"max_output_tokens": 100,
		"parallel_tool_calls": false,
		"tool_choice": "required",
		"future_top_key": 123
	}`
	req := decodeResponses(t, raw)
	body := ResponsesNativeBody([]byte(raw), req, "gpt-6-luna", effort.Unspecified)

	if body["model"] != "gpt-6-luna" {
		t.Errorf("model = %v", body["model"])
	}
	for _, k := range []string{"instructions", "text", "max_tool_calls", "previous_response_id", "store", "user", "temperature", "top_p", "max_output_tokens", "parallel_tool_calls", "tool_choice", "future_top_key"} {
		if _, ok := body[k]; !ok {
			t.Errorf("field %q lost: %v", k, body)
		}
	}
	tx, _ := body["text"].(map[string]any)
	if tx["format"] == nil {
		t.Errorf("text.format lost: %v", body["text"])
	}
}

// Non-function tools pass through verbatim; chat-shaped functions flatten.
func TestResponsesNativeBodyTools(t *testing.T) {
	raw := `{
		"model": "grok-4.7",
		"input": "hi",
		"tools": [
			{"type": "web_search", "search_context_size": "high", "future": 1},
			{"type": "function", "function": {"name": "ls", "description": "list", "parameters": {"type": "object"}, "strict": true}},
			{"type": "function", "name": "flat", "description": null, "parameters": null, "strict": null}
		]
	}`
	req := decodeResponses(t, raw)
	body := ResponsesNativeBody([]byte(raw), req, "grok-4.7", effort.Unspecified)

	traw, _ := json.Marshal(body["tools"])
	var tools []map[string]any
	if err := json.Unmarshal(traw, &tools); err != nil {
		t.Fatal(err)
	}
	if len(tools) != 3 {
		t.Fatalf("tools = %d, want 3: %s", len(tools), traw)
	}
	// web_search verbatim
	if tools[0]["type"] != "web_search" || tools[0]["search_context_size"] != "high" || tools[0]["future"] != float64(1) {
		t.Errorf("non-function tool not verbatim: %v", tools[0])
	}
	// chat shape flattened
	if _, ok := tools[1]["function"]; ok {
		t.Errorf("nested function not flattened: %v", tools[1])
	}
	if tools[1]["name"] != "ls" || tools[1]["strict"] != true {
		t.Errorf("flattened tool wrong: %v", tools[1])
	}
	// nulls stripped
	if _, ok := tools[2]["description"]; ok {
		t.Errorf("null description not stripped: %v", tools[2])
	}
	if _, ok := tools[2]["parameters"]; ok {
		t.Errorf("null parameters not stripped: %v", tools[2])
	}
	if tools[2]["name"] != "flat" {
		t.Errorf("flat tool name lost: %v", tools[2])
	}
}

// input_audio and future input_* parts survive sanitization.
func TestResponsesNativeBodyPreservesAudioParts(t *testing.T) {
	raw := `{
		"model": "grok-4.7",
		"input": [{"type": "message", "role": "user", "content": [
			{"type": "input_text", "text": "hear this"},
			{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
			{"type": "input_future_x", "blob": 1}
		]}]
	}`
	req := decodeResponses(t, raw)
	body := ResponsesNativeBody([]byte(raw), req, "grok-4.7", effort.Unspecified)

	iraw, _ := json.Marshal(body["input"])
	var items []map[string]any
	if err := json.Unmarshal(iraw, &items); err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 {
		t.Fatalf("items = %d: %s", len(items), iraw)
	}
	craw, _ := json.Marshal(items[0]["content"])
	var parts []map[string]any
	_ = json.Unmarshal(craw, &parts)
	if len(parts) != 3 {
		t.Fatalf("parts = %d, want 3: %s", len(parts), craw)
	}
	types := []string{parts[0]["type"].(string), parts[1]["type"].(string), parts[2]["type"].(string)}
	for i, want := range []string{"input_text", "input_audio", "input_future_x"} {
		if types[i] != want {
			t.Errorf("part types = %v, want audio+future preserved", types)
		}
	}
}

// Raw-only effort aliases (thinking budget, effort string) are honored.
func TestResponsesNativeBodyRawAliases(t *testing.T) {
	raw := `{"model":"grok-4.7","input":"hi","thinking":{"budget_tokens":32000}}`
	req := decodeResponses(t, raw)
	body := ResponsesNativeBody([]byte(raw), req, "grok-4.7", effort.Unspecified)
	rm, _ := body["reasoning"].(map[string]any)
	if rm == nil || rm["effort"] != "xhigh" {
		t.Errorf("budget 32000 -> %v, want xhigh", rm)
	}
	if _, ok := body["thinking"]; ok {
		t.Errorf("thinking alias leaked: %v", body)
	}

	raw2 := `{"model":"grok-4.7","input":"hi","effort":"max"}`
	req2 := decodeResponses(t, raw2)
	body2 := ResponsesNativeBody([]byte(raw2), req2, "grok-4.7", effort.Unspecified)
	rm2, _ := body2["reasoning"].(map[string]any)
	if rm2 == nil || rm2["effort"] != "max" {
		t.Errorf("effort=max -> %v", rm2)
	}
}
