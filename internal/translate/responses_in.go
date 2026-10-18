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
//
// The input list is sanitized so upstream validation passes:
//
//   - `input_text` parts on `assistant` messages are rewritten to
//     `output_text` (upstream returns 400 otherwise).
//   - Empty-text parts and empty messages are dropped.
//   - Unknown item types and non-text assistant parts (images have no
//     assistant equivalent) are dropped rather than forwarded.
//   - Tools are flattened to the Responses shape and empty fields omitted.
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
	}
	// reasoning_effort is gateway-only: upstream Responses accepts the effort
	// only inside the reasoning object and rejects the alias as unknown.
	delete(m, "reasoning_effort")
	if s, ok := m["stream"].(bool); !ok || !s {
		delete(m, "stream")
	}
	if v, ok := m["input"]; ok {
		m["input"] = sanitizeResponsesInput(v)
	}
	if len(req.Tools) > 0 {
		m["tools"] = flattenResponsesTools(req.Tools)
	} else {
		delete(m, "tools")
	}
	return m
}

// flattenResponsesTools converts gateway tools to the flat Responses shape,
// omitting empty optional fields that strict upstreams reject.
func flattenResponsesTools(tools []openai.Tool) []any {
	out := make([]any, 0, len(tools))
	for _, t := range tools {
		tm := map[string]any{
			"type": "function",
			"name": t.Function.Name,
		}
		if t.Function.Description != "" {
			tm["description"] = t.Function.Description
		}
		if t.Function.Parameters != nil {
			tm["parameters"] = t.Function.Parameters
		}
		if t.Function.Strict != nil {
			tm["strict"] = *t.Function.Strict
		}
		out = append(out, tm)
	}
	return out
}

// sanitizeResponsesInput rewrites a Responses `input` value into
// upstream-safe items. String input passes through; item lists are
// normalized per role-sensitive content rules.
func sanitizeResponsesInput(v any) any {
	if v == nil {
		return nil
	}
	if s, ok := v.(string); ok {
		return s
	}
	b, _ := json.Marshal(v)
	var items []map[string]any
	if err := json.Unmarshal(b, &items); err != nil {
		var one map[string]any
		if err2 := json.Unmarshal(b, &one); err2 == nil {
			items = []map[string]any{one}
		} else {
			return v
		}
	}
	out := make([]any, 0, len(items))
	for _, it := range items {
		if fixed, ok := sanitizeResponsesItem(it); ok {
			out = append(out, fixed)
		}
	}
	return matchToolPairs(out)
}

// matchToolPairs enforces the Responses pairing rule upstream validates:
// every function_call needs exactly one matching function_call_output with
// the same call_id.
//
// Live probes (2026-10-18, muse-spark-1.3-contributor) proved:
//   - an unmatched call fails even on a tiny 68KB body (generic 400);
//   - an output without its call fails ("No function call found ...");
//   - duplicate outputs for one call fail ("Duplicate function_call_output
//     ... exactly one matching function_call_output");
//   - order does not matter (output-before-call passes) and duplicate
//     calls with one output pass.
//
// So: drop calls with no output, drop outputs with no call, and keep only
// the first output per call_id. Non-tool items pass through untouched and
// relative order is preserved.
func matchToolPairs(items []any) []any {
	if len(items) == 0 {
		return items
	}
	const (
		other  = 0
		call   = 1
		output = 2
	)
	kinds := make([]uint8, len(items))
	ids := make([]string, len(items))
	calls := map[string]int{}
	outs := map[string]int{}
	for i, it := range items {
		m, ok := it.(map[string]any)
		if !ok {
			continue
		}
		t, _ := m["type"].(string)
		id, _ := m["call_id"].(string)
		switch t {
		case "function_call":
			kinds[i] = call
			if id == "" {
				continue // unmatchable: dropped below
			}
			ids[i] = id
			calls[id]++
		case "function_call_output":
			kinds[i] = output
			if id == "" {
				continue // unmatchable: dropped below
			}
			ids[i] = id
			outs[id]++
		}
	}
	keptOut := map[string]bool{}
	out := make([]any, 0, len(items))
	for i, it := range items {
		switch kinds[i] {
		case call:
			if outs[ids[i]] > 0 {
				out = append(out, it)
			}
		case output:
			if calls[ids[i]] > 0 && !keptOut[ids[i]] {
				keptOut[ids[i]] = true
				out = append(out, it)
			}
		default:
			out = append(out, it)
		}
	}
	return out
}

// sanitizeResponsesItem normalizes one input item, reporting false when the
// item must be dropped (empty message, unknown type).
func sanitizeResponsesItem(it map[string]any) (any, bool) {
	t, _ := it["type"].(string)
	if t == "" {
		// Easy input shape: {"role":..., "content":...} without type.
		t = "message"
	}
	switch t {
	case "message":
		role, _ := it["role"].(string)
		if role == "" {
			role = "user"
		}
		switch role {
		case "user", "system", "developer":
			content := sanitizeInputContent(it["content"])
			if len(content) == 0 {
				return nil, false
			}
			return map[string]any{
				"type":    "message",
				"role":    role,
				"content": content,
			}, true
		case "assistant":
			content := sanitizeAssistantContent(it["content"])
			if len(content) == 0 {
				return nil, false
			}
			return map[string]any{
				"type":    "message",
				"role":    "assistant",
				"content": content,
			}, true
		default:
			return nil, false
		}
	case "function_call":
		name, _ := it["name"].(string)
		if name == "" {
			return nil, false
		}
		fc := map[string]any{
			"type": "function_call",
			"name": name,
		}
		if v, ok := it["call_id"].(string); ok && v != "" {
			fc["call_id"] = v
		} else if v, ok := it["id"].(string); ok && v != "" {
			fc["call_id"] = v
		}
		if v, ok := it["arguments"]; ok {
			switch a := v.(type) {
			case string:
				fc["arguments"] = a
			case nil:
				fc["arguments"] = "{}"
			default:
				ab, _ := json.Marshal(a)
				fc["arguments"] = string(ab)
			}
		} else {
			fc["arguments"] = "{}"
		}
		return fc, true
	case "function_call_output":
		fco := map[string]any{
			"type": "function_call_output",
		}
		if v, ok := it["call_id"].(string); ok && v != "" {
			fco["call_id"] = v
		}
		if v, ok := it["output"]; ok {
			switch o := v.(type) {
			case string:
				fco["output"] = o
			case nil:
				fco["output"] = ""
			default:
				ob, _ := json.Marshal(o)
				fco["output"] = string(ob)
			}
		} else {
			fco["output"] = ""
		}
		return fco, true
	default:
		// Reasoning items, MCP/tool-search wrappers and other output
		// item types cannot be replayed as input; drop them.
		return nil, false
	}
}

// sanitizeInputContent keeps valid user/system/developer parts and drops
// empty ones. A bare string becomes one input_text part.
func sanitizeInputContent(c any) []any {
	if s, ok := c.(string); ok {
		if s == "" {
			return nil
		}
		return []any{map[string]any{"type": "input_text", "text": s}}
	}
	parts := contentParts(c)
	out := make([]any, 0, len(parts))
	for _, p := range parts {
		switch p["type"] {
		case "input_text":
			if t, ok := p["text"].(string); ok && t != "" {
				out = append(out, map[string]any{"type": "input_text", "text": t})
			}
		case "input_image":
			out = append(out, p)
		case "input_file":
			out = append(out, p)
		case "text":
			// Chat-style part: normalize to input_text.
			if t, ok := p["text"].(string); ok && t != "" {
				out = append(out, map[string]any{"type": "input_text", "text": t})
			}
		}
	}
	return out
}

// sanitizeAssistantContent keeps valid assistant history parts. `input_text`
// and chat `text` parts are rewritten to `output_text`; anything else
// without an assistant equivalent is dropped.
func sanitizeAssistantContent(c any) []any {
	if s, ok := c.(string); ok {
		if s == "" {
			return nil
		}
		return []any{map[string]any{"type": "output_text", "text": s}}
	}
	parts := contentParts(c)
	out := make([]any, 0, len(parts))
	for _, p := range parts {
		switch p["type"] {
		case "output_text":
			if t, ok := p["text"].(string); ok && t != "" {
				out = append(out, map[string]any{"type": "output_text", "text": t})
			}
		case "input_text", "text":
			if t, ok := p["text"].(string); ok && t != "" {
				out = append(out, map[string]any{"type": "output_text", "text": t})
			}
		case "refusal":
			if t, ok := p["refusal"].(string); ok && t != "" {
				out = append(out, map[string]any{"type": "refusal", "refusal": t})
			} else if t, ok := p["text"].(string); ok && t != "" {
				out = append(out, map[string]any{"type": "refusal", "refusal": t})
			}
		}
	}
	return out
}

func contentParts(c any) []map[string]any {
	if c == nil {
		return nil
	}
	if arr, ok := c.([]any); ok {
		out := make([]map[string]any, 0, len(arr))
		for _, p := range arr {
			if m, ok := p.(map[string]any); ok {
				out = append(out, m)
			}
		}
		return out
	}
	b, _ := json.Marshal(c)
	var parts []map[string]any
	if err := json.Unmarshal(b, &parts); err == nil {
		return parts
	}
	var one map[string]any
	if err := json.Unmarshal(b, &one); err == nil {
		return []map[string]any{one}
	}
	return nil
}
