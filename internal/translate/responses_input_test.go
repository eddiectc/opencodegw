package translate

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/eddiectc/opencodegw/internal/effort"
	"github.com/eddiectc/opencodegw/internal/openai"
)

// Regression: upstream rejects `input_text` on assistant messages with 400
// ("content type `input_text` is not valid on `assistant` messages").
// Assistant history must use `output_text`.
func TestChatToResponsesAssistantUsesOutputText(t *testing.T) {
	req := openai.ChatRequest{
		Model: "muse-spark-1.3-contributor",
		Messages: []openai.ChatMessage{
			{Role: "user", Content: "knock knock."},
			{Role: "assistant", Content: "Who's there?"},
			{Role: "user", Content: "Orange."},
			{Role: "assistant", Content: []any{
				map[string]any{"type": "text", "text": "Orange who?"},
			}},
		},
	}
	body := ChatToResponses(req, "muse-spark-1.3-contributor")
	items, ok := body["input"].([]any)
	if !ok {
		t.Fatalf("input = %T, want []any", body["input"])
	}
	if len(items) != 4 {
		t.Fatalf("items = %d, want 4: %v", len(items), items)
	}
	for _, it := range items {
		m, _ := it.(map[string]any)
		if m["type"] != "message" {
			t.Fatalf("item type = %v, want message: %v", m["type"], m)
		}
		role, _ := m["role"].(string)
		content, _ := m["content"].([]any)
		if len(content) == 0 {
			t.Fatalf("role %q has empty content: %v", role, m)
		}
		part, _ := content[0].(map[string]any)
		switch role {
		case "user":
			if part["type"] != "input_text" {
				t.Errorf("user part type = %v, want input_text", part["type"])
			}
		case "assistant":
			if part["type"] != "output_text" {
				t.Errorf("assistant part type = %v, want output_text", part["type"])
			}
		}
	}
}

// Assistant text must precede its function calls in history order, and the
// text part must be output_text.
func TestChatToResponsesAssistantTextBeforeToolCalls(t *testing.T) {
	req := openai.ChatRequest{
		Model: "muse-spark-1.3-contributor",
		Messages: []openai.ChatMessage{
			{Role: "user", Content: "what time is it?"},
			{Role: "assistant", Content: "checking", ToolCalls: []openai.ToolCall{
				{ID: "call_1", Type: "function", Function: openai.FunctionCall{Name: "get_time", Arguments: `{}`}},
			}},
			{Role: "tool", Content: "noon", ToolCallID: "call_1"},
		},
	}
	body := ChatToResponses(req, "muse-spark-1.3-contributor")
	items, _ := body["input"].([]any)
	if len(items) != 4 {
		t.Fatalf("items = %d, want 4: %v", len(items), items)
	}
	types := []string{}
	for _, it := range items {
		m, _ := it.(map[string]any)
		types = append(types, m["type"].(string)+"/"+roleOf(m))
	}
	want := []string{"message/user", "message/assistant", "function_call/", "function_call_output/"}
	for i := range want {
		if types[i] != want[i] {
			t.Fatalf("item order = %v, want %v", types, want)
		}
	}
	am, _ := items[1].(map[string]any)
	ac, _ := am["content"].([]any)
	part, _ := ac[0].(map[string]any)
	if part["type"] != "output_text" || part["text"] != "checking" {
		t.Errorf("assistant content = %v, want output_text/checking", ac)
	}
}

// Empty assistant/user messages are dropped instead of producing empty parts.
func TestChatToResponsesDropsEmptyMessages(t *testing.T) {
	req := openai.ChatRequest{
		Model: "muse-spark-1.3-contributor",
		Messages: []openai.ChatMessage{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: ""},
			{Role: "assistant", Content: nil},
			{Role: "user", Content: "still there?"},
		},
	}
	body := ChatToResponses(req, "muse-spark-1.3-contributor")
	items, _ := body["input"].([]any)
	if len(items) != 2 {
		t.Fatalf("items = %d, want 2: %v", len(items), items)
	}
	for _, it := range items {
		m, _ := it.(map[string]any)
		if m["role"] != "user" {
			t.Errorf("unexpected item: %v", m)
		}
	}
}

// Tools must be flattened to the Responses shape with empty fields omitted
// (no nested "function", no null description/parameters/strict).
func TestChatToResponsesFlattensTools(t *testing.T) {
	req := openai.ChatRequest{
		Model:    "muse-spark-1.3-contributor",
		Messages: []openai.ChatMessage{{Role: "user", Content: "hi"}},
		Tools: []openai.Tool{{
			Type: "function",
			Function: openai.ToolFunction{
				Name:       "ls",
				Parameters: map[string]any{"type": "object"},
			},
		}},
	}
	body := ChatToResponses(req, "muse-spark-1.3-contributor")
	raw, _ := json.Marshal(body["tools"])
	var tools []map[string]any
	if err := json.Unmarshal(raw, &tools); err != nil {
		t.Fatal(err)
	}
	if len(tools) != 1 {
		t.Fatalf("tools = %v", tools)
	}
	tm := tools[0]
	if tm["type"] != "function" || tm["name"] != "ls" {
		t.Errorf("tool = %v, want flat function/ls", tm)
	}
	if _, ok := tm["function"]; ok {
		t.Errorf("tool has nested function key: %v", tm)
	}
	for _, k := range []string{"description", "strict"} {
		if _, ok := tm[k]; ok {
			t.Errorf("tool has empty %q key: %v", k, tm)
		}
	}
	if tm["parameters"] == nil {
		t.Errorf("tool lost parameters: %v", tm)
	}
}

// Responses passthrough must rewrite assistant input_text to output_text,
// drop empty messages, drop unknown item types, and flatten tools.
func TestResponsesUpstreamBodySanitizesInput(t *testing.T) {
	var req openai.ResponsesRequest
	rawReq := `{
		"model": "muse-spark-1.3-contributor",
		"input": [
			{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
			{"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": "hello"}]},
			{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ""}]},
			{"type": "message", "role": "assistant", "content": [{"type": "input_image", "image_url": "https://x/y.png"}]},
			{"type": "reasoning", "summary": [{"type": "summary_text", "text": "thought"}]},
			{"type": "function_call", "call_id": "c1", "name": "ls", "arguments": {}},
			{"type": "function_call_output", "call_id": "c1", "output": {"ok": true}}
		],
		"tools": [{"type": "function", "name": "ls", "parameters": {"type": "object"}}]
	}`
	if err := json.Unmarshal([]byte(rawReq), &req); err != nil {
		t.Fatal(err)
	}
	body := ResponsesUpstreamBody(req, "muse-spark-1.3-contributor", effort.Unspecified)
	raw, _ := json.Marshal(body["input"])
	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil {
		t.Fatal(err)
	}
	if len(items) != 4 {
		t.Fatalf("items = %d, want 4: %s", len(items), raw)
	}
	// user untouched
	if items[0]["role"] != "user" {
		t.Errorf("item0 = %v, want user message", items[0])
	}
	// assistant input_text rewritten to output_text
	ac, _ := items[1]["content"].([]any)
	ap, _ := ac[0].(map[string]any)
	if items[1]["role"] != "assistant" || ap["type"] != "output_text" || ap["text"] != "hello" {
		t.Errorf("item1 = %v, want assistant output_text/hello", items[1])
	}
	// reasoning + empty + image-only assistant messages dropped
	if items[2]["type"] != "function_call" {
		t.Errorf("item2 = %v, want function_call", items[2])
	}
	if items[2]["arguments"] != "{}" {
		t.Errorf("function_call arguments = %v, want {}", items[2]["arguments"])
	}
	if items[3]["type"] != "function_call_output" {
		t.Errorf("item3 = %v, want function_call_output", items[3])
	}
	// tools flattened
	traw, _ := json.Marshal(body["tools"])
	var tools []map[string]any
	if err := json.Unmarshal(traw, &tools); err != nil {
		t.Fatal(err)
	}
	if len(tools) != 1 || tools[0]["name"] != "ls" {
		t.Fatalf("tools = %s, want flat ls", traw)
	}
	if _, ok := tools[0]["function"]; ok {
		t.Errorf("tool has nested function key: %s", traw)
	}
}

// User image parts keep their URL and detail; empty/unknown parts are
// dropped instead of producing invalid input items.
func TestChatToResponsesUserParts(t *testing.T) {
	req := openai.ChatRequest{
		Model: "muse-spark-1.3-contributor",
		Messages: []openai.ChatMessage{
			{Role: "user", Content: []any{
				map[string]any{"type": "text", "text": ""},
				map[string]any{"type": "text", "text": "look"},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://x/y.png", "detail": "high"}},
				map[string]any{"type": "image_url", "image_url": map[string]any{}},
				map[string]any{"type": "input_audio", "input_audio": map[string]any{}},
			}},
		},
	}
	body := ChatToResponses(req, "muse-spark-1.3-contributor")
	items, _ := body["input"].([]any)
	if len(items) != 1 {
		t.Fatalf("items = %d, want 1: %v", len(items), items)
	}
	m, _ := items[0].(map[string]any)
	content, _ := m["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("content = %v, want text+image", content)
	}
	p0, _ := content[0].(map[string]any)
	if p0["type"] != "input_text" || p0["text"] != "look" {
		t.Errorf("part0 = %v, want input_text/look", p0)
	}
	p1, _ := content[1].(map[string]any)
	if p1["type"] != "input_image" || p1["image_url"] != "https://x/y.png" || p1["detail"] != "high" {
		t.Errorf("part1 = %v, want input_image with url+detail", p1)
	}
}

// Flat Responses tools must survive gateway decode so passthrough keeps names.
func TestToolUnmarshalFlatResponsesShape(t *testing.T) {
	var tl openai.Tool
	if err := json.Unmarshal([]byte(`{"type":"function","name":"ls","description":"list"}`), &tl); err != nil {
		t.Fatal(err)
	}
	if tl.Function.Name != "ls" || tl.Function.Description != "list" {
		t.Errorf("flat tool decoded as %+v", tl)
	}
	var tc openai.Tool
	if err := json.Unmarshal([]byte(`{"type":"function","function":{"name":"ls"}}`), &tc); err != nil {
		t.Fatal(err)
	}
	if tc.Function.Name != "ls" {
		t.Errorf("chat tool decoded as %+v", tc)
	}
}

func roleOf(m map[string]any) string {
	r, _ := m["role"].(string)
	return r
}

// Upstream rejects any function_call without exactly one matching
// function_call_output (live probes 2026-10-18: unmatched call fails even on
// a 68KB body; orphan outputs and duplicate outputs fail with explicit
// messages). The sanitizer must drop orphans on both sides and keep only
// the first output per call_id.
func TestMatchToolPairsDropsOrphans(t *testing.T) {
	items := []any{
		map[string]any{"type": "message", "role": "user", "content": []any{map[string]any{"type": "input_text", "text": "hi"}}},
		map[string]any{"type": "function_call", "call_id": "lonely", "name": "bash", "arguments": "{}"},
		map[string]any{"type": "function_call", "call_id": "ok", "name": "bash", "arguments": "{}"},
		map[string]any{"type": "function_call_output", "call_id": "ok", "output": "fine"},
		map[string]any{"type": "function_call_output", "call_id": "orphan", "output": "x"},
		map[string]any{"type": "function_call", "call_id": "dup", "name": "bash", "arguments": "{}"},
		map[string]any{"type": "function_call_output", "call_id": "dup", "output": "first"},
		map[string]any{"type": "function_call_output", "call_id": "dup", "output": "second"},
	}
	got := matchToolPairs(items)
	// order: lonely dropped, orphan dropped, second dup output dropped
	want := []string{"message", "function_call", "function_call_output", "function_call", "function_call_output"}
	var types []string
	var outs []string
	for _, it := range got {
		m, _ := it.(map[string]any)
		types = append(types, m["type"].(string))
		if m["type"] == "function_call_output" {
			outs = append(outs, m["output"].(string))
		}
	}
	if len(got) != len(want) {
		t.Fatalf("items = %d, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if types[i] != want[i] {
			t.Fatalf("types = %v, want %v", types, want)
		}
	}
	if len(outs) != 2 || outs[0] != "fine" || outs[1] != "first" {
		t.Errorf("outputs = %v, want [fine first]", outs)
	}
}

// Order must not matter: output-before-call passes upstream, so the matcher
// keeps the pair regardless of positions.
func TestMatchToolPairsKeepsReordered(t *testing.T) {
	items := []any{
		map[string]any{"type": "function_call_output", "call_id": "a", "output": "early"},
		map[string]any{"type": "function_call", "call_id": "a", "name": "bash", "arguments": "{}"},
	}
	got := matchToolPairs(items)
	if len(got) != 2 {
		t.Fatalf("items = %d, want 2: %v", len(got), got)
	}
}

// The matcher runs on both Responses entry points: the chat translator and
// the Responses passthrough sanitizer.
func TestToolPairMatchingOnBothEntryPoints(t *testing.T) {
	chat := openai.ChatRequest{
		Model: "muse-spark-1.3-contributor",
		Messages: []openai.ChatMessage{
			{Role: "user", Content: "hi"},
			{Role: "assistant", ToolCalls: []openai.ToolCall{
				{ID: "lonely", Type: "function", Function: openai.FunctionCall{Name: "bash", Arguments: `{}`}},
				{ID: "ok", Type: "function", Function: openai.FunctionCall{Name: "bash", Arguments: `{}`}},
			}},
			{Role: "tool", Content: "done", ToolCallID: "ok"},
		},
	}
	body := ChatToResponses(chat, "muse-spark-1.3-contributor")
	raw, _ := json.Marshal(body["input"])
	var citems []map[string]any
	if err := json.Unmarshal(raw, &citems); err != nil {
		t.Fatal(err)
	}
	for _, it := range citems {
		if it["type"] == "function_call" && it["call_id"] == "lonely" {
			t.Fatalf("chat path kept unmatched call: %s", raw)
		}
	}

	var rreq openai.ResponsesRequest
	rawReq := `{"model": "muse-spark-1.3-contributor", "input": [
		{"type": "function_call", "call_id": "lonely", "name": "bash", "arguments": "{}"},
		{"type": "function_call", "call_id": "ok", "name": "bash", "arguments": "{}"},
		{"type": "function_call_output", "call_id": "ok", "output": "done"}
	]}`
	if err := json.Unmarshal([]byte(rawReq), &rreq); err != nil {
		t.Fatal(err)
	}
	rbody := ResponsesUpstreamBody(rreq, "muse-spark-1.3-contributor", effort.Unspecified)
	rraw, _ := json.Marshal(rbody["input"])
	var ritems []map[string]any
	if err := json.Unmarshal(rraw, &ritems); err != nil {
		t.Fatal(err)
	}
	if len(ritems) != 2 {
		t.Fatalf("responses path items = %d, want 2: %s", len(ritems), rraw)
	}
}

// The generic-400 matcher must fire only on upstream's exact detail-free
// shape: type invalid_request_error, null/missing param, and the known
// message prefix.
func TestIsGenericInvalidParams(t *testing.T) {
	generic := `{"model":"muse-spark-1.3-contributor","error":{"code":null,"message":"The request contains invalid parameters. Check the request body for any errors or inconsistencies.","param":null,"type":"invalid_request_error"}}`
	if !IsGenericInvalidParams([]byte(generic)) {
		t.Errorf("generic upstream 400 not matched")
	}
	specific := `{"error":{"message":"No function call found for function call output with call_id 'x'","param":"call_id","type":"invalid_request_error"}}`
	if IsGenericInvalidParams([]byte(specific)) {
		t.Errorf("specific param error matched as generic")
	}
	other := `{"error":{"message":"quota exceeded","type":"insufficient_quota","code":"quota"}}`
	if IsGenericInvalidParams([]byte(other)) {
		t.Errorf("quota error matched as generic")
	}
	if IsGenericInvalidParams([]byte(`not json`)) {
		t.Errorf("garbage matched as generic")
	}
}

// The gateway-only reasoning_effort alias must never reach upstream
// Responses bodies: upstream rejects it as an unknown parameter. This bit
// when -default-effort injects the alias and the request resolves valid.
func TestResponsesUpstreamBodyDropsEffortAlias(t *testing.T) {
	var req openai.ResponsesRequest
	if err := json.Unmarshal([]byte(`{
		"model": "muse-spark-1.3-contributor",
		"reasoning_effort": "max",
		"input": "hi"
	}`), &req); err != nil {
		t.Fatal(err)
	}
	body := ResponsesUpstreamBody(req, "muse-spark-1.3-contributor", effort.Unspecified)
	if _, ok := body["reasoning_effort"]; ok {
		t.Fatalf("reasoning_effort leaked upstream: %v", body)
	}
	r, ok := body["reasoning"].(map[string]any)
	if !ok || r["effort"] != "max" {
		t.Fatalf("reasoning block = %v, want effort/max", body["reasoning"])
	}
}

// bytes/3 estimator sanity: the real 3.62MB failing body must estimate above
// the 1M muse-spark window, small bodies far below.
func TestEstimateTokensScalesWithBody(t *testing.T) {
	if got := EstimateTokens(nil); got != 0 {
		t.Errorf("nil = %d, want 0", got)
	}
	small := map[string]any{"model": "m", "input": "hi"}
	if got := EstimateTokens(small); got > 100 {
		t.Errorf("small = %d, want tiny", got)
	}
	// The real failing body was 3.62MB of dense tool output: it must
	// estimate at/above the 1M muse-spark window.
	big := map[string]any{"model": "m", "input": strings.Repeat("0123456789abcdef", 226250)}
	if got := EstimateTokens(big); got < 1000000 {
		t.Errorf("3.62MB body = %d, want >= 1000000", got)
	}
}
