package translate

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/eddiectc/opencodegw/internal/effort"
	"github.com/eddiectc/opencodegw/internal/openai"
)

// Shared ingress helpers used by the Anthropic and Gemini converters.

// intJSON reads a JSON number (float64 from encoding/json, or int forms when
// maps are built in code) as an int. Anything else yields 0.
func intJSON(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case float32:
		return int(n)
	case int:
		return n
	case int64:
		return int(n)
	case json.Number:
		if i, err := n.Int64(); err == nil {
			return int(i)
		}
	}
	return 0
}

// numJSON reads a JSON number as float64.
func numJSON(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case json.Number:
		if f, err := n.Float64(); err == nil {
			return f, true
		}
	}
	return 0, false
}

// stringJSON reads a string or returns the fallback.
func stringJSON(v any, fallback string) string {
	if s, ok := v.(string); ok && s != "" {
		return s
	}
	return fallback
}

// partsArray reads a parts/content array as a slice of maps, skipping
// non-object entries.
func partsArray(v any) []map[string]any {
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]map[string]any, 0, len(arr))
	for _, p := range arr {
		if m, ok := p.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

// chatContentParts builds a chat message content value from text and image URL
// parts: a lone text becomes a plain string, otherwise a parts array.
// Returns nil when there is nothing.
func chatContentParts(texts []string, images []string) any {
	if len(images) == 0 && len(texts) == 1 {
		return texts[0]
	}
	if len(texts)+len(images) == 0 {
		return nil
	}
	parts := make([]any, 0, len(texts)+len(images))
	for _, t := range texts {
		parts = append(parts, map[string]any{"type": "text", "text": t})
	}
	for _, u := range images {
		parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": u}})
	}
	return parts
}

// AnthropicNativeBody builds the upstream body for messages-native models from
// the raw client JSON, preserving native/unknown fields verbatim.
//
// Surgical mutations only:
//   - model normalized for upstream routing
//   - effort: a native thinking block is authoritative and preserved (missing
//     budgets are filled from aliases/default when available); otherwise
//     gateway aliases or the server default are mapped to a thinking block.
//     When thinking is enabled, temperature/top_p/top_k are dropped (Anthropic
//     rejects them alongside thinking) and max_tokens is bumped above budget.
//   - max_tokens defaults to 8192 when absent
//   - gateway-only OpenAI aliases are dropped after being honored
func AnthropicNativeBody(raw []byte, model string, defEff effort.Level) map[string]any {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil || m == nil {
		m = map[string]any{}
	}
	m["model"] = model

	// Lenient string form of thinking is an effort alias, not a native block.
	if s, ok := m["thinking"].(string); ok {
		delete(m, "thinking")
		if l := effort.Normalize(s); l.Valid() && l != effort.None {
			applyAnthropicEffort(m, l)
		}
	} else if lvl, native := anthropicNativeEffort(m); native {
		switch {
		case lvl == effort.None:
			// thinking disabled: leave everything untouched.
		case lvl.Valid():
			enforceAnthropicThinking(m, thinkingBudget(m))
		default:
			// Enabled without a budget: fill from aliases/default when
			// available, else leave for upstream to reject clearly.
			budget := 0
			if l := EffortFromRawMap(m); l.Valid() && l != effort.None {
				budget = l.AnthropicBudget()
			} else if defEff.Valid() && defEff != effort.None {
				budget = defEff.AnthropicBudget()
			}
			if budget > 0 {
				if th, ok := m["thinking"].(map[string]any); ok {
					th["budget_tokens"] = budget
				}
				enforceAnthropicThinking(m, budget)
			}
		}
	} else if th, isMap := m["thinking"].(map[string]any); isMap && len(th) > 0 {
		// Unrecognized non-empty thinking shape: preserve untouched and skip
		// injection rather than second-guess a future schema.
	} else {
		delete(m, "thinking")
		lvl := EffortFromRawMap(m)
		if !lvl.Valid() {
			lvl = defEff
		}
		if lvl.Valid() {
			applyAnthropicEffort(m, lvl)
		}
	}

	delete(m, "reasoning_effort")
	delete(m, "reasoningEffort")
	delete(m, "effort")
	delete(m, "thinking_effort")
	delete(m, "reasoning")
	// Known foreign-dialect keys that strict upstreams reject. These are
	// never valid Anthropic fields, so dropping them can only help lenient
	// clients; anything else unknown passes through for upstream to judge.
	delete(m, "response_format")
	delete(m, "refusal")
	delete(m, "modalities")
	delete(m, "logprobs")
	delete(m, "top_logprobs")
	delete(m, "stream_options")
	delete(m, "streamOptions")
	// Explicit stream:false is dropped for a clean unary body; stream:true
	// is preserved for the streaming path (callers set it explicitly).
	if s, ok := m["stream"].(bool); ok && !s {
		delete(m, "stream")
	}

	if intJSON(m["max_tokens"]) <= 0 {
		if b := thinkingBudget(m); b > 0 {
			m["max_tokens"] = b + 4096
		} else {
			m["max_tokens"] = 8192
		}
	}
	return m
}

// anthropicNativeEffort reads a native thinking block. It reports native=true
// for type enabled/disabled; enabled without a budget yields an unspecified
// level so the caller can fill the budget from aliases/default.
func anthropicNativeEffort(m map[string]any) (effort.Level, bool) {
	th, ok := m["thinking"].(map[string]any)
	if !ok {
		return effort.Unspecified, false
	}
	t, _ := th["type"].(string)
	switch {
	case strings.EqualFold(t, "disabled"):
		return effort.None, true
	case strings.EqualFold(t, "enabled"):
		if b := intJSON(th["budget_tokens"]); b > 0 {
			return effort.FromBudget(b), true
		}
		return effort.Unspecified, true
	default:
		return effort.Unspecified, false
	}
}

// thinkingBudget returns the native thinking budget, or 0 when absent.
func thinkingBudget(m map[string]any) int {
	th, ok := m["thinking"].(map[string]any)
	if !ok {
		return 0
	}
	return intJSON(th["budget_tokens"])
}

// enforceAnthropicThinking applies Anthropic's thinking constraints: fill a
// missing budget, bump max_tokens above it, and drop sampling params that
// Anthropic rejects alongside thinking.
func enforceAnthropicThinking(m map[string]any, budget int) {
	if budget <= 0 {
		return
	}
	if th, ok := m["thinking"].(map[string]any); ok {
		if intJSON(th["budget_tokens"]) <= 0 {
			th["budget_tokens"] = budget
		} else {
			budget = intJSON(th["budget_tokens"])
		}
	}
	m["max_tokens"] = effort.EnsureAnthropicMaxTokens(intJSON(m["max_tokens"]), budget)
	delete(m, "temperature")
	delete(m, "top_p")
	delete(m, "top_k")
}

// applyAnthropicEffort writes a canonical level as a native thinking block.
// Level none is a no-op: the native default is already off.
func applyAnthropicEffort(m map[string]any, lvl effort.Level) {
	if lvl == effort.None {
		return
	}
	if b := lvl.AnthropicBudget(); b > 0 {
		m["thinking"] = map[string]any{"type": "enabled", "budget_tokens": b}
		enforceAnthropicThinking(m, b)
	}
}

// AnthropicToChatRequest converts a raw Anthropic Messages request into a
// canonical ChatRequest for cross-dialect backends. Native-only fields without
// a chat equivalent (cache_control, citations, thinking signatures, metadata,
// service_tier, non-custom tools) are dropped.
func AnthropicToChatRequest(raw []byte, fallbackModel string, defEff effort.Level) openai.ChatRequest {
	var m map[string]any
	_ = json.Unmarshal(raw, &m)
	if m == nil {
		m = map[string]any{}
	}
	out := openai.ChatRequest{Model: stringJSON(m["model"], fallbackModel)}

	if s, ok := m["system"].(string); ok && s != "" {
		out.Messages = append(out.Messages, openai.ChatMessage{Role: "system", Content: s})
	} else if arr, ok := m["system"].([]any); ok {
		if txt := anthropicBlocksText(arr); txt != "" {
			out.Messages = append(out.Messages, openai.ChatMessage{Role: "system", Content: txt})
		}
	}

	for _, im := range partsArray(m["messages"]) {
		role, _ := im["role"].(string)
		switch role {
		case "user":
			texts, images, results := splitAnthropicUserContent(im["content"])
			if c := chatContentParts(texts, images); c != nil {
				out.Messages = append(out.Messages, openai.ChatMessage{Role: "user", Content: c})
			}
			for _, tr := range results {
				out.Messages = append(out.Messages, openai.ChatMessage{
					Role: "tool", Content: tr.text, ToolCallID: tr.id,
				})
			}
		case "assistant":
			texts, images, calls := splitAnthropicAssistantContent(im["content"])
			msg := openai.ChatMessage{Role: "assistant", ToolCalls: calls}
			if c := chatContentParts(texts, images); c != nil {
				msg.Content = c
			}
			if msg.Content == nil && len(calls) == 0 {
				continue
			}
			out.Messages = append(out.Messages, msg)
		default:
			// Unknown role: coerce to user text rather than drop history.
			if txt := anthropicBlocksText(im["content"]); txt != "" {
				out.Messages = append(out.Messages, openai.ChatMessage{Role: "user", Content: txt})
			}
		}
	}

	for _, tm := range partsArray(m["tools"]) {
		typ, _ := tm["type"].(string)
		if typ != "" && typ != "custom" {
			continue // computer/bash/text_editor tools have no chat equivalent
		}
		name, _ := tm["name"].(string)
		if name == "" {
			continue
		}
		fn := openai.ToolFunction{Name: name}
		if d, ok := tm["description"].(string); ok {
			fn.Description = d
		}
		if schema, ok := tm["input_schema"].(map[string]any); ok {
			fn.Parameters = schema
		}
		out.Tools = append(out.Tools, openai.Tool{Type: "function", Function: fn})
	}
	if len(out.Tools) > 0 {
		if tc, ok := m["tool_choice"].(map[string]any); ok {
			switch t, _ := tc["type"].(string); t {
			case "auto":
				out.ToolChoice = "auto"
			case "any":
				out.ToolChoice = "required"
				if dis, _ := tc["disable_parallel_tool_use"].(bool); dis {
					out.ParallelToolCalls = boolPtr(false)
				}
			case "none":
				out.ToolChoice = "none"
			case "tool":
				if name, _ := tc["name"].(string); name != "" {
					out.ToolChoice = map[string]any{
						"type":     "function",
						"function": map[string]any{"name": name},
					}
				}
			}
		}
	}

	if t, ok := numJSON(m["temperature"]); ok {
		out.Temperature = &t
	}
	if t, ok := numJSON(m["top_p"]); ok {
		out.TopP = &t
	}
	if n := intJSON(m["max_tokens"]); n > 0 {
		out.MaxCompletionTokens = &n
	}
	if ss := toStringSlice(m["stop_sequences"]); len(ss) > 0 {
		out.Stop = ss
	}
	if s, ok := m["stream"].(bool); ok {
		out.Stream = &s
	}
	if md, ok := m["metadata"].(map[string]any); ok {
		if u, ok := md["user_id"].(string); ok {
			out.User = u
		}
	}

	lvl := EffortFromRawMap(m)
	// Native thinking block wins over gateway aliases when present.
	if th, ok := m["thinking"].(map[string]any); ok {
		if t, _ := th["type"].(string); strings.EqualFold(t, "disabled") {
			lvl = effort.None
		} else if b := intJSON(th["budget_tokens"]); b > 0 {
			lvl = effort.FromBudget(b)
		}
	}
	if !lvl.Valid() {
		lvl = defEff
	}
	if e := lvl.OpenAIEffort(); e != "" {
		out.ReasoningEffort = e
	}
	return out
}

func boolPtr(b bool) *bool { return &b }

type anthropicToolResult struct {
	id   string
	text string
}

// anthropicContentBlocks normalizes message content (string or block array)
// into a block list.
func anthropicContentBlocks(content any) []map[string]any {
	if s, ok := content.(string); ok {
		if s == "" {
			return nil
		}
		return []map[string]any{{"type": "text", "text": s}}
	}
	arr, ok := content.([]any)
	if !ok {
		return nil
	}
	out := make([]map[string]any, 0, len(arr))
	for _, b := range arr {
		switch t := b.(type) {
		case map[string]any:
			out = append(out, t)
		case string:
			if t != "" {
				out = append(out, map[string]any{"type": "text", "text": t})
			}
		}
	}
	return out
}

// anthropicBlocksText flattens text out of content blocks, descending into
// nested content arrays. Images, documents and tool payloads contribute
// nothing here; callers handle those by type.
func anthropicBlocksText(content any) string {
	var sb strings.Builder
	for _, b := range anthropicContentBlocks(content) {
		if t, ok := b["text"].(string); ok && (b["type"] == "text" || b["type"] == nil) {
			sb.WriteString(t)
			continue
		}
		if inner, ok := b["content"].([]any); ok {
			sb.WriteString(anthropicBlocksText(inner))
			continue
		}
		if t, ok := b["text"].(string); ok && b["type"] != "image" && b["type"] != "document" {
			sb.WriteString(t)
		}
	}
	return sb.String()
}

// splitAnthropicUserContent splits user content into text parts, image URLs,
// and tool results (tool_result, web_search_tool_result, mcp_tool_result).
func splitAnthropicUserContent(content any) (texts []string, images []string, results []anthropicToolResult) {
	for _, b := range anthropicContentBlocks(content) {
		switch b["type"] {
		case "text":
			if t, ok := b["text"].(string); ok && t != "" {
				texts = append(texts, t)
			}
		case "image":
			url, ph := anthropicImageRef(b)
			if url != "" {
				images = append(images, url)
			} else if ph != "" {
				texts = append(texts, ph)
			}
		case "document":
			t, img := anthropicDocumentRef(b)
			if t != "" {
				texts = append(texts, t)
			} else if img != "" {
				images = append(images, img)
			}
		case "tool_result", "web_search_tool_result", "mcp_tool_result":
			id, _ := b["tool_use_id"].(string)
			text := anthropicBlocksText(b["content"])
			if s, ok := b["content"].(string); ok {
				text = s
			}
			if isErr, _ := b["is_error"].(bool); isErr {
				text = "Error: " + text
			}
			results = append(results, anthropicToolResult{id: id, text: text})
		default:
			// Future blocks: keep any text, drop the rest.
			if t := anthropicBlocksText([]any{b}); t != "" {
				texts = append(texts, t)
			}
		}
	}
	return texts, images, results
}

// splitAnthropicAssistantContent splits assistant content into text parts,
// image URLs, and tool calls. Thinking blocks are dropped: signatures cannot
// transfer across dialects.
func splitAnthropicAssistantContent(content any) (texts []string, images []string, calls []openai.ToolCall) {
	for _, b := range anthropicContentBlocks(content) {
		switch b["type"] {
		case "text":
			if t, ok := b["text"].(string); ok && t != "" {
				texts = append(texts, t)
			}
		case "image":
			url, ph := anthropicImageRef(b)
			if url != "" {
				images = append(images, url)
			} else if ph != "" {
				texts = append(texts, ph)
			}
		case "document":
			t, img := anthropicDocumentRef(b)
			if t != "" {
				texts = append(texts, t)
			} else if img != "" {
				images = append(images, img)
			}
		case "tool_use", "server_tool_use", "mcp_tool_use":
			id, _ := b["id"].(string)
			if id == "" {
				id = "call_" + newID()
			}
			name, _ := b["name"].(string)
			args := "{}"
			if in, ok := b["input"]; ok && in != nil {
				if bb, err := json.Marshal(in); err == nil {
					args = string(bb)
				}
			}
			calls = append(calls, openai.ToolCall{
				ID:   id,
				Type: "function",
				Function: openai.FunctionCall{
					Name:      name,
					Arguments: args,
				},
			})
		case "thinking", "redacted_thinking":
			// Dropped: signatures cannot transfer across dialects.
		default:
			if t := anthropicBlocksText([]any{b}); t != "" {
				texts = append(texts, t)
			}
		}
	}
	return texts, images, calls
}

// anthropicImageRef converts an Anthropic image block to a chat image URL, or
// a placeholder note when the source cannot transfer (Files API references).
func anthropicImageRef(b map[string]any) (url, placeholder string) {
	src, _ := b["source"].(map[string]any)
	if src == nil {
		return "", ""
	}
	switch src["type"] {
	case "base64":
		media, _ := src["media_type"].(string)
		data, _ := src["data"].(string)
		if data == "" {
			return "", ""
		}
		if media == "" {
			media = "image/png"
		}
		return "data:" + media + ";base64," + data, ""
	case "url":
		if u, ok := src["url"].(string); ok && u != "" {
			return u, ""
		}
	case "file":
		if id, ok := src["file_id"].(string); ok && id != "" {
			return "", "[file input " + id + " omitted in cross-dialect translation]"
		}
		return "", "[file input omitted in cross-dialect translation]"
	}
	return "", ""
}

// anthropicDocumentRef converts an Anthropic document block to text or an
// image URL. Plain-text documents decode inline; anything else that cannot
// transfer becomes a placeholder note.
func anthropicDocumentRef(b map[string]any) (text, image string) {
	src, _ := b["source"].(map[string]any)
	if src == nil {
		return "", ""
	}
	switch src["type"] {
	case "text":
		if t, ok := src["data"].(string); ok {
			return t, ""
		}
	case "base64":
		media, _ := src["media_type"].(string)
		data, _ := src["data"].(string)
		if data == "" {
			return "", ""
		}
		lm := strings.ToLower(media)
		switch {
		case lm == "text/plain":
			if raw, err := base64.StdEncoding.DecodeString(data); err == nil {
				return string(raw), ""
			}
		case strings.HasPrefix(lm, "image/"):
			return "", "data:" + media + ";base64," + data
		default:
			if media == "" {
				media = "document"
			}
			return "[" + media + " document omitted in cross-dialect translation]", ""
		}
	case "url":
		media, _ := src["media_type"].(string)
		if u, ok := src["url"].(string); ok && u != "" {
			if strings.HasPrefix(strings.ToLower(media), "image/") {
				return "", u
			}
			return "[document " + u + " omitted in cross-dialect translation]", ""
		}
	case "file":
		return "[file document omitted in cross-dialect translation]", ""
	}
	return "", ""
}

// ChatToAnthropicResult wraps a ChatResponse as an Anthropic Messages result.
func ChatToAnthropicResult(chat openai.ChatResponse, reqModel string) map[string]any {
	var text string
	var tools []any
	finish := "end_turn"
	if len(chat.Choices) > 0 {
		msg := chat.Choices[0].Message
		text = messageText(msg.Content)
		for _, tc := range msg.ToolCalls {
			var input any = map[string]any{}
			if strings.TrimSpace(tc.Function.Arguments) != "" {
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
				if input == nil {
					input = map[string]any{}
				}
			}
			tools = append(tools, map[string]any{
				"type":  "tool_use",
				"id":    orDefault(tc.ID, "toolu_"+newID()),
				"name":  tc.Function.Name,
				"input": input,
			})
		}
		finish = chatFinishToAnthropic(chat.Choices[0].FinishReason, len(tools) > 0)
	}
	content := []any{}
	if text != "" || len(tools) == 0 {
		content = append(content, map[string]any{"type": "text", "text": text})
	}
	content = append(content, tools...)
	usage := map[string]any{"input_tokens": 0, "output_tokens": 0}
	if chat.Usage != nil {
		usage["input_tokens"] = chat.Usage.PromptTokens
		usage["output_tokens"] = chat.Usage.CompletionTokens
	}
	id := "msg_" + newID()
	if strings.HasPrefix(chat.ID, "msg_") {
		id = chat.ID
	}
	return map[string]any{
		"id":          id,
		"type":        "message",
		"role":        "assistant",
		"model":       orDefault(chat.Model, reqModel),
		"content":     content,
		"stop_reason": finish,
		"usage":       usage,
	}
}

func chatFinishToAnthropic(finish string, sawTool bool) string {
	if sawTool {
		return "tool_use"
	}
	switch finish {
	case "length":
		return "max_tokens"
	case "content_filter":
		return "refusal"
	case "stop_sequence":
		return "stop_sequence"
	default:
		return "end_turn"
	}
}
