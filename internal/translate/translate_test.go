package translate

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/eddiectc/opencodegw/internal/openai"
)

func chatReq(model, eff string) openai.ChatRequest {
	return openai.ChatRequest{
		Model:           model,
		ReasoningEffort: eff,
		Messages: []openai.ChatMessage{
			{Role: "system", Content: "you are helpful"},
			{Role: "user", Content: "hello"},
			{Role: "assistant", Content: "hi", ToolCalls: []openai.ToolCall{
				{ID: "call_1", Type: "function", Function: openai.FunctionCall{Name: "get_time", Arguments: `{}`}},
			}},
			{Role: "tool", Content: "noon", ToolCallID: "call_1"},
		},
	}
}

func TestChatToResponsesEffort(t *testing.T) {
	for _, eff := range []string{"low", "medium", "high", "xhigh", "max"} {
		body := ChatToResponses(chatReq("grok-4.7", eff), "grok-4.7")
		r, ok := body["reasoning"].(map[string]any)
		if !ok {
			t.Fatalf("effort %q: missing reasoning block", eff)
		}
		if r["effort"] != eff {
			t.Errorf("effort %q: got %v", eff, r["effort"])
		}
	}
}

func TestChatToAnthropicEffort(t *testing.T) {
	body := ChatToAnthropic(chatReq("claude-haiku-5-5", "max"), "claude-haiku-5-5")
	th, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("max: missing thinking block")
	}
	if th["budget_tokens"] != 64000 {
		t.Errorf("max budget = %v, want 64000", th["budget_tokens"])
	}
	body = ChatToAnthropic(chatReq("claude-haiku-5-5", "xhigh"), "claude-haiku-5-5")
	th, _ = body["thinking"].(map[string]any)
	if th["budget_tokens"] != 32000 {
		t.Errorf("xhigh budget = %v, want 32000", th["budget_tokens"])
	}
	// system collapsed
	if _, ok := body["system"]; !ok {
		t.Errorf("missing system block")
	}
	msgs, _ := body["messages"].([]any)
	if len(msgs) == 0 {
		t.Errorf("missing messages")
	}
}

func TestChatToGeminiEffort(t *testing.T) {
	body := ChatToGemini(chatReq("gemini-4-argon-contributor", "max"), "x")
	gen, ok := body["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("missing generationConfig")
	}
	tc, ok := gen["thinkingConfig"].(map[string]any)
	if !ok || tc["thinkingLevel"] != "HIGH" {
		t.Errorf("max gemini level = %v, want HIGH", gen["thinkingConfig"])
	}
}

func TestResponsesToChat(t *testing.T) {
	raw := `{"id":"resp_1","object":"response","model":"grok-4.7","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]},{"type":"function_call","call_id":"c1","name":"get_time","arguments":"{}"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`
	chat, err := ResponsesToChat([]byte(raw), "grok-4.7")
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Choices) != 1 {
		t.Fatalf("choices = %d", len(chat.Choices))
	}
	if chat.Choices[0].Message.Content != "hello" {
		t.Errorf("content = %v", chat.Choices[0].Message.Content)
	}
	if len(chat.Choices[0].Message.ToolCalls) != 1 {
		t.Errorf("tool calls missing")
	}
	if chat.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("finish = %q", chat.Choices[0].FinishReason)
	}
}

func TestAnthropicToChat(t *testing.T) {
	raw := `{"id":"msg_1","model":"claude-haiku-5-5","stop_reason":"tool_use","content":[{"type":"text","text":"hi"},{"type":"tool_use","id":"tu_1","name":"get_time","input":{}}],"usage":{"input_tokens":5,"output_tokens":7}}`
	chat, err := AnthropicToChat([]byte(raw), "claude-haiku-5-5")
	if err != nil {
		t.Fatal(err)
	}
	if chat.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("finish = %q", chat.Choices[0].FinishReason)
	}
	if chat.Usage.TotalTokens != 12 {
		t.Errorf("usage = %+v", chat.Usage)
	}
}

func TestGeminiToChat(t *testing.T) {
	raw := `{"candidates":[{"content":{"parts":[{"text":"hello"},{"functionCall":{"name":"get_time","args":{}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"totalTokenCount":7}}`
	chat, err := GeminiToChat([]byte(raw), "gemini-4-argon-contributor")
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Choices[0].Message.ToolCalls) != 1 {
		t.Errorf("tool calls missing: %+v", chat.Choices[0].Message)
	}
}

func TestEffortFromChatAliases(t *testing.T) {
	budget := 32000
	req := openai.ChatRequest{Model: "x", Thinking: &openai.ThinkingParam{BudgetTokens: &budget}}
	if got := EffortFromChat(req); got.String() != "xhigh" {
		t.Errorf("budget 32000 -> %q, want xhigh", got)
	}
	budget = 64000
	req.Thinking.BudgetTokens = &budget
	if got := EffortFromChat(req); got.String() != "max" {
		t.Errorf("budget 64000 -> %q, want max", got)
	}
	req = openai.ChatRequest{Model: "x", Effort: "max"}
	if got := EffortFromChat(req); got.String() != "max" {
		t.Errorf("effort alias -> %q, want max", got)
	}
}

func TestTranslateResponsesStream(t *testing.T) {
	sse := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hel\"}\n\n" +
		"data: {\"type\":\"response.output_text.delta\",\"delta\":\"lo\"}\n\n" +
		"data: [DONE]\n\n"
	var got []string
	err := TranslateResponsesStreamToChat(strings.NewReader(sse), "grok-4.7", func(s string) bool {
		got = append(got, s)
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "hel") || !strings.Contains(joined, "lo") {
		t.Errorf("deltas missing: %s", joined)
	}
	var first map[string]any
	_ = json.Unmarshal([]byte(got[0]), &first)
	_ = first
}
