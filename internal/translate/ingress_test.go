package translate

import (
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/eddiectc/opencodegw/internal/effort"
	"github.com/eddiectc/opencodegw/internal/openai"
)

// Anthropic native must preserve exact budgets, cache control, beta features
// and unknown keys verbatim, while enforcing thinking constraints.
func TestAnthropicNativeBodyPreservesNative(t *testing.T) {
	raw := `{
		"model": "claude-haiku-5-5",
		"max_tokens": 50000,
		"system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
		"messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}],
		"thinking": {"type": "enabled", "budget_tokens": 12345},
		"temperature": 0.7,
		"top_k": 40,
		"top_p": 0.9,
		"stop_sequences": ["END"],
		"metadata": {"user_id": "u1"},
		"service_tier": "priority",
		"future_native_key": {"keep": true}
	}`
	body := AnthropicNativeBody([]byte(raw), "claude-haiku-5-5", effort.Unspecified)

	th, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking lost: %v", body)
	}
	// exact budget preserved, not snapped to canonical buckets
	if th["budget_tokens"] != float64(12345) {
		t.Errorf("budget = %v, want exact 12345", th["budget_tokens"])
	}
	// cache control verbatim
	sys, _ := body["system"].([]any)
	if len(sys) != 1 {
		t.Fatalf("system lost: %v", body["system"])
	}
	if sm, _ := sys[0].(map[string]any); sm["cache_control"] == nil {
		t.Errorf("system cache_control lost: %v", sys[0])
	}
	msgs, _ := body["messages"].([]any)
	mm, _ := msgs[0].(map[string]any)
	cc, _ := mm["content"].([]any)
	if cm, _ := cc[0].(map[string]any); cm["cache_control"] == nil {
		t.Errorf("content cache_control lost: %v", cc[0])
	}
	// natives preserved
	if body["service_tier"] != "priority" || body["future_native_key"] == nil {
		t.Errorf("native keys lost: %v", body)
	}
	if body["metadata"] == nil {
		t.Errorf("metadata lost: %v", body)
	}
	// thinking constraints enforced
	if _, ok := body["temperature"]; ok {
		t.Errorf("temperature must drop with thinking: %v", body)
	}
	if _, ok := body["top_p"]; ok {
		t.Errorf("top_p must drop with thinking: %v", body)
	}
	if _, ok := body["top_k"]; ok {
		t.Errorf("top_k must drop with thinking: %v", body)
	}
	if intJSON(body["max_tokens"]) != 50000 {
		t.Errorf("max_tokens = %v, want 50000 (already above budget)", body["max_tokens"])
	}
}

// thinking disabled leaves sampling params alone; enabled without budget
// fills from aliases/default.
func TestAnthropicNativeBodyDisabledAndFill(t *testing.T) {
	raw := `{"model":"m","max_tokens":100,"messages":[],"thinking":{"type":"disabled"},"temperature":0.5}`
	body := AnthropicNativeBody([]byte(raw), "m", effort.Max)
	if th, _ := body["thinking"].(map[string]any); th["type"] != "disabled" {
		t.Errorf("disabled not preserved: %v", body["thinking"])
	}
	if body["temperature"] != float64(0.5) {
		t.Errorf("temperature must stay with disabled thinking: %v", body)
	}

	raw2 := `{"model":"m","messages":[],"thinking":{"type":"enabled"},"effort":"xhigh"}`
	body2 := AnthropicNativeBody([]byte(raw2), "m", effort.Unspecified)
	th2, _ := body2["thinking"].(map[string]any)
	if th2["budget_tokens"] != 32000 {
		t.Errorf("budget filled = %v, want 32000", th2["budget_tokens"])
	}
	if _, ok := body2["effort"]; ok {
		t.Errorf("effort alias leaked: %v", body2)
	}

	raw3 := `{"model":"m","messages":[]}`
	body3 := AnthropicNativeBody([]byte(raw3), "m", effort.High)
	th3, _ := body3["thinking"].(map[string]any)
	if th3["budget_tokens"] != 16000 {
		t.Errorf("default high -> %v, want 16000", th3["budget_tokens"])
	}
	// max_tokens defaults above budget
	if body3["max_tokens"] != 16000+4096 {
		t.Errorf("max_tokens = %v, want %d", body3["max_tokens"], 16000+4096)
	}
}

// max_tokens defaults to 8192 with no thinking anywhere.
func TestAnthropicNativeBodyMaxTokensDefault(t *testing.T) {
	raw := `{"model":"m","messages":[]}`
	body := AnthropicNativeBody([]byte(raw), "m", effort.Unspecified)
	if body["max_tokens"] != 8192 {
		t.Errorf("max_tokens = %v, want 8192", body["max_tokens"])
	}
	if _, ok := body["thinking"]; ok {
		t.Errorf("thinking should be absent: %v", body)
	}
}

// String thinking is a lenient effort alias, not a native block.
func TestAnthropicNativeBodyStringThinking(t *testing.T) {
	raw := `{"model":"m","messages":[],"thinking":"high"}`
	body := AnthropicNativeBody([]byte(raw), "m", effort.Unspecified)
	th, _ := body["thinking"].(map[string]any)
	if th == nil || th["type"] != "enabled" || th["budget_tokens"] != 16000 {
		t.Errorf("string thinking -> %v, want enabled/16000", body["thinking"])
	}
}

// Anthropic -> chat: messages, images, documents, tool_use/tool_result,
// thinking dropped, tools + tool_choice + metadata mapped.
func TestAnthropicToChatRequestFull(t *testing.T) {
	raw := `{
		"model": "claude-haiku-5-5",
		"system": "sys prompt",
		"max_tokens": 100,
		"temperature": 0.3,
		"top_p": 0.8,
		"stop_sequences": ["END"],
		"metadata": {"user_id": "u9"},
		"messages": [
			{"role": "user", "content": [
				{"type": "text", "text": "look "},
				{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
				{"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "doc text"}}
			]},
			{"role": "assistant", "content": [
				{"type": "thinking", "thinking": "hmm", "signature": "sig"},
				{"type": "text", "text": "checking"},
				{"type": "tool_use", "id": "tu_1", "name": "get_time", "input": {"tz": "utc"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "tu_1", "content": "noon"}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "tu_2", "is_error": true, "content": "boom"}
			]}
		],
		"tools": [
			{"name": "get_time", "description": "clock", "input_schema": {"type": "object"}},
			{"type": "computer_20251122", "name": "computer"},
			{"name": "", "description": "nameless"}
		],
		"tool_choice": {"type": "any"}
	}`
	chat := AnthropicToChatRequest([]byte(raw), "fallback", effort.Unspecified)

	if chat.Model != "claude-haiku-5-5" {
		t.Errorf("model = %q", chat.Model)
	}
	if len(chat.Messages) < 5 {
		t.Fatalf("messages = %d, want >=5: %+v", len(chat.Messages), chat.Messages)
	}
	if chat.Messages[0].Role != "system" || chat.Messages[0].Content != "sys prompt" {
		t.Errorf("system = %+v", chat.Messages[0])
	}
	// user multimodal: text + image data URL + doc text
	u := chat.Messages[1]
	if u.Role != "user" {
		t.Fatalf("msg1 = %+v, want user", u)
	}
	ub, _ := json.Marshal(u.Content)
	us := string(ub)
	if !strings.Contains(us, "look") || !strings.Contains(us, "data:image/png;base64,QUJD") || !strings.Contains(us, "doc text") {
		t.Errorf("user content incomplete: %s", us)
	}
	// assistant: thinking dropped, text + tool_use kept
	a := chat.Messages[2]
	if a.Role != "assistant" || len(a.ToolCalls) != 1 {
		t.Fatalf("assistant = %+v", a)
	}
	if a.ToolCalls[0].ID != "tu_1" || a.ToolCalls[0].Function.Name != "get_time" {
		t.Errorf("tool call = %+v", a.ToolCalls[0])
	}
	if !strings.Contains(a.ToolCalls[0].Function.Arguments, "utc") {
		t.Errorf("tool args = %q", a.ToolCalls[0].Function.Arguments)
	}
	if s, _ := a.Content.(string); !strings.Contains(s, "checking") {
		t.Errorf("assistant text = %v", a.Content)
	}
	// tool results mapped, error prefixed
	tr1 := chat.Messages[3]
	if tr1.Role != "tool" || tr1.ToolCallID != "tu_1" || tr1.Content != "noon" {
		t.Errorf("tool result = %+v", tr1)
	}
	tr2 := chat.Messages[4]
	if c, _ := tr2.Content.(string); !strings.HasPrefix(c, "Error: ") {
		t.Errorf("error result not prefixed: %+v", tr2)
	}
	// tools: only custom with name survive
	if len(chat.Tools) != 1 || chat.Tools[0].Function.Name != "get_time" {
		t.Errorf("tools = %+v", chat.Tools)
	}
	if chat.Tools[0].Function.Parameters["type"] != "object" {
		t.Errorf("input_schema lost: %+v", chat.Tools[0].Function.Parameters)
	}
	if chat.ToolChoice != "required" {
		t.Errorf("tool_choice any -> %v, want required", chat.ToolChoice)
	}
	// sampling + limits + user
	if chat.Temperature == nil || *chat.Temperature != 0.3 {
		t.Errorf("temperature = %v", chat.Temperature)
	}
	if chat.TopP == nil || *chat.TopP != 0.8 {
		t.Errorf("top_p = %v", chat.TopP)
	}
	if chat.MaxCompletionTokens == nil || *chat.MaxCompletionTokens != 100 {
		t.Errorf("max tokens = %v", chat.MaxCompletionTokens)
	}
	if chat.User != "u9" {
		t.Errorf("user = %q", chat.User)
	}
}

// Native thinking budget maps back to canonical effort for cross-dialect.
func TestAnthropicToChatRequestThinkingBudget(t *testing.T) {
	chat := AnthropicToChatRequest([]byte(`{"model":"m","messages":[],"thinking":{"type":"enabled","budget_tokens":32000}}`), "m", effort.Unspecified)
	if chat.ReasoningEffort != "xhigh" {
		t.Errorf("budget 32000 -> %q, want xhigh", chat.ReasoningEffort)
	}
	chat2 := AnthropicToChatRequest([]byte(`{"model":"m","messages":[]}`), "m", effort.Max)
	if chat2.ReasoningEffort != "max" {
		t.Errorf("default max -> %q", chat2.ReasoningEffort)
	}
}

// Gemini native must preserve safety, generation knobs, tools verbatim and
// map aliases to thinkingLevel without touching native levels.
func TestGeminiNativeBodyPreservesNative(t *testing.T) {
	raw := `{
		"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
		"systemInstruction": {"parts": [{"text": "sys"}]},
		"generationConfig": {"temperature": 0.5, "topP": 0.9, "maxOutputTokens": 100, "thinkingConfig": {"thinkingLevel": "HIGH"}},
		"safetySettings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
		"tools": [{"googleSearch": {}}, {"functionDeclarations": [{"name": "f", "parameters": {"type": "object"}}]}],
		"cachedContent": "cached/abc",
		"effort": "low",
		"response_format": {"type": "json_object"},
		"model": "should-drop"
	}`
	body := GeminiNativeBody([]byte(raw), "gemini-4-argon-contributor", effort.Max)

	// natives verbatim
	if body["safetySettings"] == nil || body["cachedContent"] != "cached/abc" {
		t.Errorf("native keys lost: %v", body)
	}
	gc, _ := body["generationConfig"].(map[string]any)
	if gc["temperature"] != float64(0.5) || gc["maxOutputTokens"] != float64(100) {
		t.Errorf("generationConfig lost: %v", gc)
	}
	tc, _ := gc["thinkingConfig"].(map[string]any)
	// native HIGH wins over effort=low alias
	if tc["thinkingLevel"] != "HIGH" {
		t.Errorf("native level overwritten: %v", tc)
	}
	tools, _ := body["tools"].([]any)
	if len(tools) != 2 {
		t.Fatalf("tools = %v", body["tools"])
	}
	if _, ok := body["model"]; ok {
		t.Errorf("model key must not reach GenerateContent: %v", body)
	}
	for _, k := range []string{"effort", "response_format"} {
		if _, ok := body[k]; ok {
			t.Errorf("alias %q leaked: %v", k, body)
		}
	}
}

// Alias/default maps to thinkingLevel only when no native level exists.
func TestGeminiNativeBodyEffortInjection(t *testing.T) {
	raw := `{"contents": [], "generationConfig": {"temperature": 0.5}, "reasoning_effort": "medium"}`
	body := GeminiNativeBody([]byte(raw), "m", effort.Unspecified)
	gc, _ := body["generationConfig"].(map[string]any)
	tc, _ := gc["thinkingConfig"].(map[string]any)
	if tc["thinkingLevel"] != "MEDIUM" {
		t.Errorf("medium -> %v, want MEDIUM", tc)
	}
	if gc["temperature"] != float64(0.5) {
		t.Errorf("sibling key lost: %v", gc)
	}

	raw2 := `{"contents": []}`
	body2 := GeminiNativeBody([]byte(raw2), "m", effort.High)
	gc2, _ := body2["generationConfig"].(map[string]any)
	tc2, _ := gc2["thinkingConfig"].(map[string]any)
	if tc2["thinkingLevel"] != "HIGH" {
		t.Errorf("default high -> %v", tc2)
	}
}

// Gemini -> chat: system, contents, inline/file data, functionCall/Response
// pairing by name, tools + toolConfig.
func TestGeminiToChatRequestFull(t *testing.T) {
	raw := `{
		"systemInstruction": {"parts": [{"text": "sys"}]},
		"contents": [
			{"role": "user", "parts": [
				{"text": "see "},
				{"inlineData": {"mimeType": "image/png", "data": "QUJD"}},
				{"inlineData": {"mimeType": "text/plain", "data": "aGk="}},
				{"fileData": {"mimeType": "image/png", "fileUri": "gs://b/f.png"}}
			]},
			{"role": "model", "parts": [
				{"text": "checking"},
				{"functionCall": {"name": "get_time", "args": {"tz": "utc"}}}
			]},
			{"role": "function", "parts": [
				{"functionResponse": {"name": "get_time", "response": {"content": "noon"}}}
			]}
		],
		"generationConfig": {"temperature": 0.4, "topP": 0.7, "maxOutputTokens": 50, "stopSequences": ["END"], "thinkingConfig": {"thinkingLevel": "LOW"}},
		"tools": [{"functionDeclarations": [{"name": "get_time", "description": "clock", "parameters": {"type": "object"}}]}, {"googleSearch": {}}],
		"toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_time"]}}
	}`
	chat := GeminiToChatRequest([]byte(raw), "gemini-4-argon-contributor", effort.Unspecified)

	if chat.Model != "gemini-4-argon-contributor" {
		t.Errorf("model = %q", chat.Model)
	}
	if len(chat.Messages) != 4 {
		t.Fatalf("messages = %d: %+v", len(chat.Messages), chat.Messages)
	}
	if chat.Messages[0].Role != "system" || chat.Messages[0].Content != "sys" {
		t.Errorf("system = %+v", chat.Messages[0])
	}
	ub, _ := json.Marshal(chat.Messages[1].Content)
	us := string(ub)
	for _, want := range []string{"see", "data:image/png;base64,QUJD", "hi", "gs://b/f.png"} {
		if !strings.Contains(us, want) {
			t.Errorf("user content missing %q: %s", want, us)
		}
	}
	a := chat.Messages[2]
	if a.Role != "assistant" || len(a.ToolCalls) != 1 {
		t.Fatalf("assistant = %+v", a)
	}
	if a.ToolCalls[0].Function.Name != "get_time" || !strings.Contains(a.ToolCalls[0].Function.Arguments, "utc") {
		t.Errorf("functionCall = %+v", a.ToolCalls[0])
	}
	tr := chat.Messages[3]
	if tr.Role != "tool" || tr.Content != "noon" || tr.Name != "get_time" {
		t.Errorf("functionResponse = %+v", tr)
	}
	if tr.ToolCallID != a.ToolCalls[0].ID {
		t.Errorf("tool pairing broken: %q vs %q", tr.ToolCallID, a.ToolCalls[0].ID)
	}
	if len(chat.Tools) != 1 || chat.Tools[0].Function.Name != "get_time" {
		t.Errorf("tools = %+v (googleSearch must drop)", chat.Tools)
	}
	tc, ok := chat.ToolChoice.(map[string]any)
	if !ok {
		t.Fatalf("tool_choice single name -> %v, want function object", chat.ToolChoice)
	}
	if fn, _ := tc["function"].(map[string]any); fn["name"] != "get_time" {
		t.Errorf("tool_choice = %v", chat.ToolChoice)
	}
	if chat.Temperature == nil || *chat.Temperature != 0.4 {
		t.Errorf("temperature = %v", chat.Temperature)
	}
	if chat.MaxCompletionTokens == nil || *chat.MaxCompletionTokens != 50 {
		t.Errorf("max tokens = %v", chat.MaxCompletionTokens)
	}
	if chat.ReasoningEffort != "low" {
		t.Errorf("thinkingLevel LOW -> %q", chat.ReasoningEffort)
	}
}

// Result wrappers: chat -> Anthropic / Gemini shapes.
func TestChatToAnthropicAndGeminiResult(t *testing.T) {
	chat := openai.ChatResponse{
		ID: "chatcmpl-1", Model: "kimi-k3",
		Choices: []openai.ChatChoice{{
			Message: openai.ChatMessage{
				Role:    "assistant",
				Content: "hello",
				ToolCalls: []openai.ToolCall{{
					ID: "call_1", Type: "function",
					Function: openai.FunctionCall{Name: "get_time", Arguments: `{"tz":"utc"}`},
				}},
			},
			FinishReason: "tool_calls",
		}},
		Usage: &openai.Usage{PromptTokens: 5, CompletionTokens: 7, TotalTokens: 12},
	}
	ar := ChatToAnthropicResult(chat, "kimi-k3")
	if ar["type"] != "message" || ar["role"] != "assistant" || ar["model"] != "kimi-k3" {
		t.Errorf("anthropic envelope = %v", ar)
	}
	if ar["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason = %v", ar["stop_reason"])
	}
	content, _ := ar["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("content blocks = %d: %v", len(content), content)
	}
	tu, _ := content[1].(map[string]any)
	if tu["type"] != "tool_use" || tu["id"] != "call_1" {
		t.Errorf("tool_use = %v", tu)
	}
	usage, _ := ar["usage"].(map[string]any)
	if usage["input_tokens"] != 5 || usage["output_tokens"] != 7 {
		t.Errorf("usage = %v", usage)
	}

	gr := ChatToGeminiResult(chat, "kimi-k3")
	cands, _ := gr["candidates"].([]any)
	if len(cands) != 1 {
		t.Fatalf("candidates = %v", gr)
	}
	cm, _ := cands[0].(map[string]any)
	if cm["finishReason"] != "STOP" {
		t.Errorf("finishReason = %v", cm["finishReason"])
	}
	cont, _ := cm["content"].(map[string]any)
	parts, _ := cont["parts"].([]any)
	if len(parts) != 2 {
		t.Fatalf("parts = %v", parts)
	}
	fc, _ := parts[1].(map[string]any)["functionCall"].(map[string]any)
	if fc["name"] != "get_time" {
		t.Errorf("functionCall = %v", parts[1])
	}
	um, _ := gr["usageMetadata"].(map[string]any)
	if um["promptTokenCount"] != 5 || um["totalTokenCount"] != 12 {
		t.Errorf("usageMetadata = %v", um)
	}
}

// EffortFromRawMap covers every alias spelling across dialects.
func TestEffortFromRawMap(t *testing.T) {
	cases := []struct {
		in   string
		want effort.Level
	}{
		{`{"reasoning_effort":"max"}`, effort.Max},
		{`{"reasoningEffort":"xhigh"}`, effort.XHigh},
		{`{"effort":"high"}`, effort.High},
		{`{"effort":2}`, effort.Medium},
		{`{"thinking_effort":"low"}`, effort.Low},
		{`{"reasoning":{"effort":"medium"}}`, effort.Medium},
		{`{"thinking":"high"}`, effort.High},
		{`{"thinking":{"effort":"low"}}`, effort.Low},
		{`{"thinking":{"budget_tokens":64000}}`, effort.Max},
		{`{"thinking":{"budget_tokens":2048}}`, effort.Low},
		{`{"thinking":{"type":"disabled"}}`, effort.None},
		{`{"generationConfig":{"thinkingConfig":{"thinkingLevel":"HIGH"}}}`, effort.High},
		{`{"generationConfig":{"thinkingConfig":{"thinkingLevel":"LOW"}}}`, effort.Low},
		{`{"generationConfig":{"thinkingConfig":{"thinkingLevel":"MINIMAL"}}}`, effort.None},
		{`{"unrelated":true}`, effort.Unspecified},
	}
	for _, c := range cases {
		var m map[string]any
		_ = json.Unmarshal([]byte(c.in), &m)
		if got := EffortFromRawMap(m); got != c.want {
			t.Errorf("%s -> %q, want %q", c.in, got, c.want)
		}
	}
}

// Chat -> Anthropic stream framing carries text deltas, thinking, tool_use.
func TestTranslateChatStreamToAnthropic(t *testing.T) {
	chunks := []string{
		mustChunk(t, openai.ChatStreamDelta{Role: "assistant"}, nil),
		mustChunk(t, openai.ChatStreamDelta{Content: "hel"}, nil),
		mustChunk(t, openai.ChatStreamDelta{Reasoning: "think"}, nil),
		mustChunkTool(t, 0, "tu_1", "get_time", ""),
		mustChunkTool(t, 0, "", "", `{"tz"`),
		mustChunkTool(t, 0, "", "", `"}`),
		mustChunk(t, openai.ChatStreamDelta{}, strPtr("tool_calls")),
	}
	sse := ""
	for _, c := range chunks {
		sse += "data: " + c + "\n\n"
	}
	sse += "data: [DONE]\n\n"
	var evs []SSEEvent
	if err := TranslateChatStreamToAnthropic(strings.NewReader(sse), "m", func(ev SSEEvent) bool {
		evs = append(evs, ev)
		return true
	}); err != nil {
		t.Fatal(err)
	}
	joined := ""
	for _, e := range evs {
		joined += e.Event + "|" + e.Data + "\n"
	}
	for _, want := range []string{
		"message_start", `"type":"message"`,
		"content_block_start", `"type":"text"`,
		"text_delta", `"text":"hel"`,
		"thinking_delta", `"thinking":"think"`,
		`"type":"tool_use"`, `"name":"get_time"`,
		"input_json_delta", `"partial_json":"{\"tz\""`,
		"content_block_stop",
		"message_delta", `"stop_reason":"tool_use"`,
		"message_stop",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("anthropic stream missing %q:\n%s", want, joined)
		}
	}
}

func mustChunk(t *testing.T, d openai.ChatStreamDelta, f *string) string {
	t.Helper()
	b, _ := json.Marshal(NewChatChunk("m", d, f))
	return string(b)
}

func mustChunkTool(t *testing.T, idx int, id, name, args string) string {
	t.Helper()
	fn := &openai.FuncDelta{}
	if name != "" {
		fn.Name = name
	}
	if args != "" {
		fn.Arguments = args
	}
	if name == "" && args == "" {
		fn = nil
	}
	b, _ := json.Marshal(NewChatChunk("m", openai.ChatStreamDelta{
		ToolCalls: []openai.ToolCallDelta{{Index: idx, ID: id, Type: "function", Function: fn}},
	}, nil))
	return string(b)
}

// Chat -> Gemini stream framing carries text and functionCall parts.
func TestTranslateChatStreamToGemini(t *testing.T) {
	chunks := []string{
		mustChunk(t, openai.ChatStreamDelta{Role: "assistant"}, nil),
		mustChunk(t, openai.ChatStreamDelta{Content: "hi"}, nil),
		mustChunkTool(t, 0, "call_1", "get_time", ""),
		mustChunkTool(t, 0, "", "", `{"a":1}`),
	}
	sse := ""
	for _, c := range chunks {
		sse += "data: " + c + "\n\n"
	}
	sse += "data: [DONE]\n\n"
	var got []string
	if err := TranslateChatStreamToGemini(strings.NewReader(sse), func(s string) bool {
		got = append(got, s)
		return true
	}); err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, `"text":"hi"`) {
		t.Errorf("text missing: %s", joined)
	}
	if !strings.Contains(joined, `"functionCall":{"name":"get_time","args":{"a":1}}`) {
		t.Errorf("functionCall missing: %s", joined)
	}
	if !strings.Contains(joined, `"finishReason":"STOP"`) {
		t.Errorf("finishReason missing: %s", joined)
	}
	var _ = io.Discard
}
