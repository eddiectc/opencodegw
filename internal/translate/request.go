// Package translate converts between the gateway's OpenAI-compatible
// surface and each upstream backend dialect.
package translate

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/eddiectc/opencodegw/internal/effort"
	"github.com/eddiectc/opencodegw/internal/openai"
)

// ChatToResponses converts an OpenAI Chat Completions request into an
// OpenAI Responses request for responses-native upstream models.
func ChatToResponses(req openai.ChatRequest, model string) map[string]any {
	eff := EffortFromChat(req).ClampForResponses()
	out := map[string]any{
		"model": model,
	}
	input, instructions := chatMessagesToResponsesInput(req.Messages)
	if input != nil {
		out["input"] = input
	}
	if instructions != "" {
		out["instructions"] = instructions
	}
	if req.Temperature != nil {
		out["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		out["top_p"] = *req.TopP
	}
	if req.MaxCompletionTokens != nil {
		out["max_output_tokens"] = *req.MaxCompletionTokens
	} else if req.MaxTokens != nil {
		out["max_output_tokens"] = *req.MaxTokens
	}
	if len(req.Tools) > 0 {
		tools := make([]any, 0, len(req.Tools))
		for _, t := range req.Tools {
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
			tools = append(tools, tm)
		}
		out["tools"] = tools
	}
	if req.ToolChoice != nil {
		out["tool_choice"] = req.ToolChoice
	}
	if req.ParallelToolCalls != nil {
		out["parallel_tool_calls"] = *req.ParallelToolCalls
	}
	if eff := eff.OpenAIEffort(); eff != "" {
		out["reasoning"] = map[string]any{"effort": eff}
	}
	if req.Stream != nil && *req.Stream {
		out["stream"] = true
	}
	if req.User != "" {
		out["user"] = req.User
	}
	return out
}

// chatMessagesToResponsesInput converts chat messages to Responses input items.
// System/developer messages collapse into `instructions`; the rest become input items.
//
// Responses content types are role-sensitive: `user`/`system`/`developer`
// messages use `input_text`/`input_image` parts, while `assistant` history
// must use `output_text` (upstream rejects `input_text` on assistant
// messages with 400). Assistant text is emitted before its function calls
// so history reads in generation order.
func chatMessagesToResponsesInput(msgs []openai.ChatMessage) (any, string) {
	var instructions []string
	var items []any
	for _, m := range msgs {
		switch m.Role {
		case "system", "developer":
			if s := messageText(m.Content); s != "" {
				instructions = append(instructions, s)
			}
		case "user":
			content := toResponsesContent(m.Content)
			if len(content) == 0 {
				continue
			}
			items = append(items, map[string]any{
				"type":    "message",
				"role":    "user",
				"content": content,
			})
		case "assistant":
			if len(m.ToolCalls) > 0 {
				if s := messageText(m.Content); s != "" {
					items = append(items, map[string]any{
						"type":    "message",
						"role":    "assistant",
						"content": []any{map[string]any{"type": "output_text", "text": s}},
					})
				}
				for _, tc := range m.ToolCalls {
					items = append(items, map[string]any{
						"type":      "function_call",
						"call_id":   tc.ID,
						"name":      tc.Function.Name,
						"arguments": tc.Function.Arguments,
					})
				}
			} else {
				content := toResponsesAssistantContent(m.Content)
				if len(content) == 0 {
					continue
				}
				items = append(items, map[string]any{
					"type":    "message",
					"role":    "assistant",
					"content": content,
				})
			}
		case "tool":
			items = append(items, map[string]any{
				"type":    "function_call_output",
				"call_id": m.ToolCallID,
				"output":  messageText(m.Content),
			})
		}
	}
	return items, strings.Join(instructions, "\n")
}

// toResponsesAssistantContent converts chat content to Responses assistant
// history parts. Assistant messages must use `output_text`, never
// `input_text`. Non-text parts (images) have no assistant equivalent and
// are dropped; empty text is skipped so we never emit empty parts.
func toResponsesAssistantContent(c any) []any {
	switch v := c.(type) {
	case nil:
		return nil
	case string:
		if v == "" {
			return nil
		}
		return []any{map[string]any{"type": "output_text", "text": v}}
	case []any:
		out := make([]any, 0, len(v))
		for _, p := range v {
			m, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if m["type"] == "text" {
				if t, ok := m["text"].(string); ok && t != "" {
					out = append(out, map[string]any{"type": "output_text", "text": t})
				}
			}
		}
		if len(out) == 0 {
			// Fall back to flattened text so refusal/reasoning parts
			// carried as chat content are not silently dropped.
			if s := messageText(c); s != "" {
				return []any{map[string]any{"type": "output_text", "text": s}}
			}
			return nil
		}
		return out
	default:
		b, _ := json.Marshal(c)
		var parts []openai.ContentPart
		if err := json.Unmarshal(b, &parts); err == nil {
			out := make([]any, 0, len(parts))
			for _, p := range parts {
				if p.Type == "text" && p.Text != "" {
					out = append(out, map[string]any{"type": "output_text", "text": p.Text})
				}
			}
			if len(out) > 0 {
				return out
			}
		}
		if s := messageText(c); s != "" {
			return []any{map[string]any{"type": "output_text", "text": s}}
		}
		if s, ok := c.(string); ok && s != "" {
			return []any{map[string]any{"type": "output_text", "text": s}}
		}
		return nil
	}
}

func toResponsesContent(c any) []any {
	switch v := c.(type) {
	case nil:
		return nil
	case string:
		if v == "" {
			return nil
		}
		return []any{map[string]any{"type": "input_text", "text": v}}
	case []any:
		out := make([]any, 0, len(v))
		for _, p := range v {
			if part, ok := chatPartToResponsesPart(p); ok {
				out = append(out, part)
			}
		}
		return out
	default:
		// Try []ContentPart via re-marshal
		b, _ := json.Marshal(c)
		var parts []openai.ContentPart
		if err := json.Unmarshal(b, &parts); err == nil {
			out := make([]any, 0, len(parts))
			for _, p := range parts {
				if part, ok := chatPartToResponsesPart(map[string]any{
					"type":      p.Type,
					"text":      p.Text,
					"image_url": p.ImageURL,
				}); ok {
					out = append(out, part)
				}
			}
			return out
		}
		if s, ok := c.(string); ok && s != "" {
			return []any{map[string]any{"type": "input_text", "text": s}}
		}
		return nil
	}
}

// chatPartToResponsesPart converts one chat content part to a Responses
// input part, reporting false for parts that must be dropped (empty text,
// unknown types, images without a URL).
func chatPartToResponsesPart(p any) (any, bool) {
	m, ok := p.(map[string]any)
	if !ok {
		return nil, false
	}
	switch m["type"] {
	case "text":
		text, _ := m["text"].(string)
		if text == "" {
			return nil, false
		}
		return map[string]any{"type": "input_text", "text": text}, true
	case "image_url":
		var url, detail string
		switch iu := m["image_url"].(type) {
		case map[string]any:
			url, _ = iu["url"].(string)
			detail, _ = iu["detail"].(string)
		}
		if url == "" {
			return nil, false
		}
		part := map[string]any{"type": "input_image", "image_url": url}
		if detail != "" {
			part["detail"] = detail
		}
		return part, true
	default:
		return nil, false
	}
}

// ChatToAnthropic converts a Chat Completions request to an Anthropic Messages request.
func ChatToAnthropic(req openai.ChatRequest, model string) map[string]any {
	eff := EffortFromChat(req)
	budget := eff.AnthropicBudget()
	out := map[string]any{
		"model":      model,
		"max_tokens": effort.EnsureAnthropicMaxTokens(anthropicMaxTokens(req), budget),
	}
	var system []any
	var messages []any
	for _, m := range req.Messages {
		switch m.Role {
		case "system", "developer":
			if s := messageText(m.Content); s != "" {
				system = append(system, map[string]any{"type": "text", "text": s})
			}
		case "user":
			messages = append(messages, map[string]any{
				"role":    "user",
				"content": toAnthropicContent(m.Content),
			})
		case "assistant":
			content := toAnthropicContent(m.Content)
			for _, tc := range m.ToolCalls {
				content = append(content, map[string]any{
					"type":  "tool_use",
					"id":    tc.ID,
					"name":  tc.Function.Name,
					"input": mustJSONMap(tc.Function.Arguments),
				})
			}
			messages = append(messages, map[string]any{"role": "assistant", "content": content})
		case "tool":
			messages = append(messages, map[string]any{
				"role": "user",
				"content": []any{map[string]any{
					"type":        "tool_result",
					"tool_use_id": m.ToolCallID,
					"content":     messageText(m.Content),
				}},
			})
		}
	}
	if len(system) > 0 {
		out["system"] = system
	}
	out["messages"] = messages
	if req.Temperature != nil {
		out["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		out["top_p"] = *req.TopP
	}
	if req.Stop != nil {
		out["stop_sequences"] = toStringSlice(req.Stop)
	}
	if len(req.Tools) > 0 {
		tools := make([]any, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, map[string]any{
				"name":         t.Function.Name,
				"description":  t.Function.Description,
				"input_schema": t.Function.Parameters,
			})
		}
		out["tools"] = tools
		if req.ToolChoice != nil {
			out["tool_choice"] = anthropicToolChoice(req.ToolChoice)
		}
	}
	if budget > 0 {
		out["thinking"] = map[string]any{"type": "enabled", "budget_tokens": budget}
		// Anthropic forbids temperature/top_p alongside thinking.
		delete(out, "temperature")
		delete(out, "top_p")
	}
	if req.Stream != nil && *req.Stream {
		out["stream"] = true
	}
	return out
}

func anthropicMaxTokens(req openai.ChatRequest) int {
	if req.MaxCompletionTokens != nil && *req.MaxCompletionTokens > 0 {
		return *req.MaxCompletionTokens
	}
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		return *req.MaxTokens
	}
	return 8192
}

func toAnthropicContent(c any) []any {
	switch v := c.(type) {
	case string:
		return []any{map[string]any{"type": "text", "text": v}}
	case []any:
		out := make([]any, 0, len(v))
		for _, p := range v {
			m, ok := p.(map[string]any)
			if !ok {
				out = append(out, map[string]any{"type": "text", "text": fmt.Sprintf("%v", p)})
				continue
			}
			switch m["type"] {
			case "text":
				text, _ := m["text"].(string)
				out = append(out, map[string]any{"type": "text", "text": text})
			case "image_url":
				var url string
				switch iu := m["image_url"].(type) {
				case map[string]any:
					url, _ = iu["url"].(string)
				}
				out = append(out, imageURLToAnthropic(url))
			default:
				out = append(out, map[string]any{"type": "text", "text": fmt.Sprintf("%v", p)})
			}
		}
		return out
	default:
		b, _ := json.Marshal(c)
		var parts []openai.ContentPart
		if err := json.Unmarshal(b, &parts); err == nil {
			raw := make([]any, 0, len(parts))
			for _, p := range parts {
				raw = append(raw, map[string]any{"type": p.Type, "text": p.Text, "image_url": p.ImageURL})
			}
			return toAnthropicContent(raw)
		}
		if s, ok := c.(string); ok {
			return []any{map[string]any{"type": "text", "text": s}}
		}
		return []any{map[string]any{"type": "text", "text": fmt.Sprintf("%v", c)}}
	}
}

func anthropicToolChoice(tc any) any {
	switch v := tc.(type) {
	case string:
		switch v {
		case "none":
			return map[string]any{"type": "none"}
		case "required":
			return map[string]any{"type": "any"}
		default:
			return map[string]any{"type": "auto"}
		}
	case map[string]any:
		if fn, ok := v["function"].(map[string]any); ok {
			name, _ := fn["name"].(string)
			return map[string]any{"type": "tool", "name": name}
		}
		if t, ok := v["type"].(string); ok {
			switch t {
			case "none":
				return map[string]any{"type": "none"}
			case "function":
				return map[string]any{"type": "auto"}
			}
		}
	}
	return map[string]any{"type": "auto"}
}

// ChatToGemini converts a Chat Completions request to a Gemini generateContent body.
func ChatToGemini(req openai.ChatRequest, model string) map[string]any {
	eff := EffortFromChat(req)
	var systemParts []any
	var contents []any
	for _, m := range req.Messages {
		switch m.Role {
		case "system", "developer":
			if s := messageText(m.Content); s != "" {
				systemParts = append(systemParts, map[string]any{"text": s})
			}
		case "user":
			contents = append(contents, map[string]any{
				"role":  "user",
				"parts": toGeminiParts(m.Content, nil),
			})
		case "assistant":
			parts := toGeminiParts(m.Content, nil)
			for _, tc := range m.ToolCalls {
				parts = append(parts, map[string]any{
					"functionCall": map[string]any{
						"name": tc.Function.Name,
						"args": mustJSONMap(tc.Function.Arguments),
					},
				})
			}
			contents = append(contents, map[string]any{"role": "model", "parts": parts})
		case "tool":
			contents = append(contents, map[string]any{
				"role": "function",
				"parts": []any{map[string]any{
					"functionResponse": map[string]any{
						"name":     m.Name,
						"response": map[string]any{"content": messageText(m.Content)},
					},
				}},
			})
		}
	}
	out := map[string]any{"contents": contents}
	if len(systemParts) > 0 {
		out["systemInstruction"] = map[string]any{"parts": systemParts}
	}
	gen := map[string]any{}
	if req.Temperature != nil {
		gen["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		gen["topP"] = *req.TopP
	}
	if req.MaxCompletionTokens != nil {
		gen["maxOutputTokens"] = *req.MaxCompletionTokens
	} else if req.MaxTokens != nil {
		gen["maxOutputTokens"] = *req.MaxTokens
	}
	if req.Stop != nil {
		gen["stopSequences"] = toStringSlice(req.Stop)
	}
	if lvl := eff.GeminiLevel(); lvl != "" {
		gen["thinkingConfig"] = map[string]any{"thinkingLevel": lvl}
	}
	if len(gen) > 0 {
		out["generationConfig"] = gen
	}
	if len(req.Tools) > 0 {
		decls := make([]any, 0, len(req.Tools))
		for _, t := range req.Tools {
			decls = append(decls, map[string]any{
				"name":        t.Function.Name,
				"description": t.Function.Description,
				"parameters":  t.Function.Parameters,
			})
		}
		out["tools"] = []any{map[string]any{"functionDeclarations": decls}}
	}
	return out
}

func toGeminiParts(c any, _ []openai.ToolCall) []any {
	switch v := c.(type) {
	case string:
		return []any{map[string]any{"text": v}}
	case []any:
		out := make([]any, 0, len(v))
		for _, p := range v {
			m, ok := p.(map[string]any)
			if !ok {
				out = append(out, map[string]any{"text": fmt.Sprintf("%v", p)})
				continue
			}
			switch m["type"] {
			case "text":
				text, _ := m["text"].(string)
				out = append(out, map[string]any{"text": text})
			case "image_url":
				var url string
				switch iu := m["image_url"].(type) {
				case map[string]any:
					url, _ = iu["url"].(string)
				}
				if data, mime := parseDataURL(url); data != "" {
					out = append(out, map[string]any{
						"inlineData": map[string]any{"mimeType": mime, "data": data},
					})
				} else if url != "" {
					out = append(out, map[string]any{
						"fileData": map[string]any{"mimeType": "image/*", "fileUri": url},
					})
				}
			}
		}
		return out
	default:
		if s, ok := c.(string); ok {
			return []any{map[string]any{"text": s}}
		}
		return []any{map[string]any{"text": fmt.Sprintf("%v", c)}}
	}
}

// EffortFromChat extracts the canonical effort from all accepted chat fields.
func EffortFromChat(req openai.ChatRequest) effort.Level {
	for _, s := range []string{req.ReasoningEffort, req.ReasoningEffort2, req.Effort, req.ThinkingEffort} {
		if l := effort.Normalize(s); l.Valid() {
			return l
		}
	}
	if req.Reasoning != nil {
		if l := effort.Normalize(req.Reasoning.Effort); l.Valid() {
			return l
		}
	}
	if req.Thinking != nil {
		if l := effort.Normalize(req.Thinking.Effort); l.Valid() {
			return l
		}
		if req.Thinking.BudgetTokens != nil {
			return effort.FromBudget(*req.Thinking.BudgetTokens)
		}
		if strings.EqualFold(req.Thinking.Type, "disabled") {
			return effort.None
		}
	}
	return effort.Unspecified
}

// EffortFromResponses extracts effort from a Responses request.
func EffortFromResponses(req openai.ResponsesRequest) effort.Level {
	if l := effort.Normalize(req.ReasoningEffort); l.Valid() {
		return l
	}
	if req.Reasoning != nil {
		if l := effort.Normalize(req.Reasoning.Effort); l.Valid() {
			return l
		}
	}
	return effort.Unspecified
}

// ---- helpers ----

func messageText(c any) string {
	switch v := c.(type) {
	case nil:
		return ""
	case string:
		return v
	case []any:
		var sb strings.Builder
		for _, p := range v {
			m, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if m["type"] == "text" {
				if t, ok := m["text"].(string); ok {
					sb.WriteString(t)
				}
			}
		}
		return sb.String()
	default:
		b, _ := json.Marshal(c)
		var parts []openai.ContentPart
		if err := json.Unmarshal(b, &parts); err == nil {
			var sb strings.Builder
			for _, p := range parts {
				if p.Type == "text" {
					sb.WriteString(p.Text)
				}
			}
			return sb.String()
		}
		return ""
	}
}

func mustJSONMap(s string) map[string]any {
	if s == "" {
		return map[string]any{}
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		return map[string]any{"_raw": s}
	}
	return m
}

func toStringSlice(v any) []string {
	switch t := v.(type) {
	case string:
		return []string{t}
	case []any:
		out := make([]string, 0, len(t))
		for _, e := range t {
			if s, ok := e.(string); ok {
				out = append(out, s)
			}
		}
		return out
	case []string:
		return t
	default:
		return nil
	}
}

func imageURLToAnthropic(url string) map[string]any {
	if data, mime := parseDataURL(url); data != "" {
		return map[string]any{
			"type": "image",
			"source": map[string]any{
				"type": "base64", "media_type": mime, "data": data,
			},
		}
	}
	return map[string]any{
		"type":   "image",
		"source": map[string]any{"type": "url", "url": url},
	}
}

func parseDataURL(url string) (data, mime string) {
	if !strings.HasPrefix(url, "data:") {
		return "", ""
	}
	rest := strings.TrimPrefix(url, "data:")
	semi := strings.Index(rest, ";base64,")
	if semi < 0 {
		return "", ""
	}
	return rest[semi+len(";base64,"):], rest[:semi]
}
