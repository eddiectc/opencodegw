package translate

import (
	"encoding/json"
	"strings"

	"github.com/eddiectc/opencodegw/internal/effort"
	"github.com/eddiectc/opencodegw/internal/openai"
)

// ResponsesToChatRequest converts a Responses request into a ChatRequest
// so non-responses backends can serve /v1/responses via the chat translators.
func ResponsesToChatRequest(req openai.ResponsesRequest, defaultEffort effort.Level) openai.ChatRequest {
	out := openai.ChatRequest{
		Model: req.Model,
	}
	if req.Temperature != nil {
		out.Temperature = req.Temperature
	}
	if req.TopP != nil {
		out.TopP = req.TopP
	}
	if req.MaxOutputTokens != nil {
		out.MaxCompletionTokens = req.MaxOutputTokens
	}
	if len(req.Tools) > 0 {
		out.Tools = req.Tools
	}
	if req.ToolChoice != nil {
		out.ToolChoice = req.ToolChoice
	}
	if req.ParallelToolCalls != nil {
		out.ParallelToolCalls = req.ParallelToolCalls
	}
	eff := EffortFromResponses(req)
	if !eff.Valid() {
		eff = defaultEffort
	}
	if e := eff.OpenAIEffort(); e != "" {
		out.ReasoningEffort = e
	}
	if req.User != "" {
		out.User = req.User
	}

	msgs := []openai.ChatMessage{}
	if req.Instructions != "" {
		msgs = append(msgs, openai.ChatMessage{Role: "system", Content: req.Instructions})
	}
	for _, m := range responsesInputToChatMessages(req.Input) {
		msgs = append(msgs, m)
	}
	out.Messages = msgs
	return out
}

func responsesInputToChatMessages(input any) []openai.ChatMessage {
	if input == nil {
		return nil
	}
	// String input -> single user message.
	if s, ok := input.(string); ok {
		return []openai.ChatMessage{{Role: "user", Content: s}}
	}
	b, _ := json.Marshal(input)
	// Try [] of items.
	var items []map[string]any
	if err := json.Unmarshal(b, &items); err != nil {
		// Single item?
		var one map[string]any
		if err2 := json.Unmarshal(b, &one); err2 == nil {
			items = []map[string]any{one}
		} else {
			return []openai.ChatMessage{{Role: "user", Content: string(b)}}
		}
	}
	var msgs []openai.ChatMessage
	for _, it := range items {
		t, _ := it["type"].(string)
		switch t {
		case "message":
			role, _ := it["role"].(string)
			if role == "" {
				role = "user"
			}
			msgs = append(msgs, openai.ChatMessage{Role: role, Content: responsesContentToText(it["content"])})
		case "function_call":
			name, _ := it["name"].(string)
			args := ""
			switch a := it["arguments"].(type) {
			case string:
				args = a
			default:
				ab, _ := json.Marshal(a)
				args = string(ab)
			}
			callID, _ := it["call_id"].(string)
			if callID == "" {
				callID, _ = it["id"].(string)
			}
			msgs = append(msgs, openai.ChatMessage{
				Role: "assistant",
				ToolCalls: []openai.ToolCall{{
					ID:   callID,
					Type: "function",
					Function: openai.FunctionCall{
						Name:      name,
						Arguments: args,
					},
				}},
			})
		case "function_call_output":
			callID, _ := it["call_id"].(string)
			var text string
			switch o := it["output"].(type) {
			case string:
				text = o
			default:
				ob, _ := json.Marshal(o)
				text = string(ob)
			}
			msgs = append(msgs, openai.ChatMessage{Role: "tool", Content: text, ToolCallID: callID})
		default:
			// Unknown item: stringify as user text.
			ib, _ := json.Marshal(it)
			msgs = append(msgs, openai.ChatMessage{Role: "user", Content: string(ib)})
		}
	}
	return msgs
}

func responsesContentToText(c any) string {
	switch v := c.(type) {
	case string:
		return v
	case []any:
		var sb strings.Builder
		for _, p := range v {
			m, ok := p.(map[string]any)
			if !ok {
				continue
			}
			switch m["type"] {
			case "input_text", "output_text", "text":
				if t, ok := m["text"].(string); ok {
					sb.WriteString(t)
				}
			}
		}
		return sb.String()
	default:
		b, _ := json.Marshal(c)
		var parts []map[string]any
		if err := json.Unmarshal(b, &parts); err == nil {
			return responsesContentToText(any(parts))
		}
		if s, ok := c.(string); ok {
			return s
		}
		return string(b)
	}
}

// ResponsesUpstreamBody normalizes a gateway Responses request into the
// upstream Responses body for responses-native models.
func ResponsesUpstreamBody(req openai.ResponsesRequest, model string, defaultEffort effort.Level) map[string]any {
	b, _ := json.Marshal(req)
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	m["model"] = model
	eff := EffortFromResponses(req).ClampForResponses()
	if !eff.Valid() {
		eff = defaultEffort.ClampForResponses()
	}
	if e := eff.OpenAIEffort(); e != "" {
		m["reasoning"] = map[string]any{"effort": e}
	} else {
		delete(m, "reasoning")
		delete(m, "reasoning_effort")
	}
	if s, ok := m["stream"].(bool); !ok || !s {
		delete(m, "stream")
	}
	return m
}
