package translate

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/eddiectc/opencodegw/internal/effort"
	"github.com/eddiectc/opencodegw/internal/openai"
)

// GeminiNativeBody builds the upstream body for gemini-native models from the
// raw client JSON, preserving native/unknown fields verbatim.
//
// Surgical mutations only:
//   - effort: a native thinkingConfig is authoritative and preserved; aliases
//     or the server default are mapped to thinkingConfig.thinkingLevel only
//     when no native level is present
//   - model key removed: GenerateContent carries the model in the URL path,
//     not the body, and strict upstreams reject unknown fields
//   - gateway-only aliases (response_format, refusal, reasoning objects,
//     thinking blocks, effort spellings) are dropped after being honored.
//     The gateway-only "stream" hint is also removed: streaming travels in
//     the verb, and strict upstreams reject unknown body fields.
func GeminiNativeBody(raw []byte, model string, defEff effort.Level) map[string]any {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil || m == nil {
		m = map[string]any{}
	}
	_ = model // model travels in the URL path, never the body

	_, nativeLevel := geminiNativeThinkingLevel(m)
	if nativeLevel == "" {
		lvl := EffortFromRawMap(m)
		if !lvl.Valid() {
			lvl = defEff
		}
		if e := lvl.GeminiLevel(); e != "" {
			ensureGeminiLevel(m, e)
		}
	}

	delete(m, "model")
	delete(m, "reasoning_effort")
	delete(m, "reasoningEffort")
	delete(m, "effort")
	delete(m, "thinking_effort")
	delete(m, "thinking")
	delete(m, "reasoning")
	delete(m, "response_format")
	delete(m, "refusal")
	delete(m, "stream")
	delete(m, "modalities")
	delete(m, "logprobs")
	delete(m, "top_logprobs")
	delete(m, "stream_options")
	delete(m, "streamOptions")
	return m
}

// geminiNativeThinkingLevel reads generationConfig.thinkingConfig.thinkingLevel.
// Returns (config map, level or "").
func geminiNativeThinkingLevel(m map[string]any) (map[string]any, string) {
	gc, ok := m["generationConfig"].(map[string]any)
	if !ok {
		return nil, ""
	}
	tc, ok := gc["thinkingConfig"].(map[string]any)
	if !ok {
		return gc, ""
	}
	lvl, _ := tc["thinkingLevel"].(string)
	return gc, lvl
}

// ensureGeminiLevel writes thinkingConfig.thinkingLevel, creating the nested
// objects and preserving sibling keys.
func ensureGeminiLevel(m map[string]any, level string) {
	var gc map[string]any
	if g, ok := m["generationConfig"].(map[string]any); ok && g != nil {
		gc = g
	} else {
		gc = map[string]any{}
		m["generationConfig"] = gc
	}
	var tc map[string]any
	if t, ok := gc["thinkingConfig"].(map[string]any); ok && t != nil {
		tc = t
	} else {
		tc = map[string]any{}
		gc["thinkingConfig"] = tc
	}
	tc["thinkingLevel"] = level
}

// GeminiToChatRequest converts a raw Gemini generateContent body into a
// canonical ChatRequest for cross-dialect backends. Safety settings,
// grounding/code-execution tools, cached content and response MIME/schema are
// native-only and dropped.
func GeminiToChatRequest(raw []byte, modelFromPath string, defEff effort.Level) openai.ChatRequest {
	var m map[string]any
	_ = json.Unmarshal(raw, &m)
	if m == nil {
		m = map[string]any{}
	}
	// URL path is authoritative; body model (non-standard) is a fallback.
	model := modelFromPath
	if model == "" {
		model = stringJSON(m["model"], "")
	}
	out := openai.ChatRequest{Model: model}

	if sys, ok := m["systemInstruction"].(map[string]any); ok {
		if txt := geminiPartsText(partsArray(sys["parts"])); txt != "" {
			out.Messages = append(out.Messages, openai.ChatMessage{Role: "system", Content: txt})
		}
	}

	pendingCalls := map[string]openai.ToolCall{}
	for _, cm := range partsArray(m["contents"]) {
		role, _ := cm["role"].(string)
		parts := partsArray(cm["parts"])
		switch role {
		case "model":
			var texts []string
			var calls []openai.ToolCall
			for _, p := range parts {
				if t, ok := p["text"].(string); ok && t != "" {
					texts = append(texts, t)
				}
				if fc, ok := p["functionCall"].(map[string]any); ok {
					call := geminiFunctionCall(fc)
					calls = append(calls, call)
					pendingCalls[call.Function.Name] = call
				}
				if t := geminiInlineDocText(p); t != "" {
					texts = append(texts, t)
				}
				if t := geminiFileRefText(p); t != "" {
					texts = append(texts, t)
				}
			}
			if len(texts) == 0 && len(calls) == 0 {
				continue
			}
			msg := openai.ChatMessage{Role: "assistant", ToolCalls: calls}
			if len(texts) == 1 {
				msg.Content = texts[0]
			} else if len(texts) > 1 {
				cp := make([]any, 0, len(texts))
				for _, t := range texts {
					cp = append(cp, map[string]any{"type": "text", "text": t})
				}
				msg.Content = cp
			}
			out.Messages = append(out.Messages, msg)
		case "user", "":
			var texts []string
			var images []string
			for _, p := range parts {
				if t, ok := p["text"].(string); ok && t != "" {
					texts = append(texts, t)
				}
				if u, ok := geminiInlineImageURL(p); ok {
					images = append(images, u)
				}
				if t := geminiInlineDocText(p); t != "" {
					texts = append(texts, t)
				}
				if t := geminiFileRefText(p); t != "" {
					texts = append(texts, t)
				}
				if fr, ok := p["functionResponse"].(map[string]any); ok {
					name, _ := fr["name"].(string)
					text := geminiFunctionResponseText(fr)
					id := ""
					if call, ok := pendingCalls[name]; ok && name != "" {
						id = call.ID
					}
					out.Messages = append(out.Messages, openai.ChatMessage{
						Role: "tool", Content: text, ToolCallID: id, Name: name,
					})
				}
			}
			if c := chatContentParts(texts, images); c != nil {
				out.Messages = append(out.Messages, openai.ChatMessage{Role: "user", Content: c})
			}
		case "function":
			for _, p := range parts {
				if fr, ok := p["functionResponse"].(map[string]any); ok {
					name, _ := fr["name"].(string)
					text := geminiFunctionResponseText(fr)
					id := ""
					if call, ok := pendingCalls[name]; ok && name != "" {
						id = call.ID
					}
					out.Messages = append(out.Messages, openai.ChatMessage{
						Role: "tool", Content: text, ToolCallID: id, Name: name,
					})
				}
				if t, ok := p["text"].(string); ok && t != "" {
					out.Messages = append(out.Messages, openai.ChatMessage{Role: "user", Content: t})
				}
			}
		default:
			// Unknown role: coerce text to user rather than drop history.
			if txt := geminiPartsText(parts); txt != "" {
				out.Messages = append(out.Messages, openai.ChatMessage{Role: "user", Content: txt})
			}
		}
	}

	for _, tm := range partsArray(m["tools"]) {
		for _, d := range partsArray(tm["functionDeclarations"]) {
			name, _ := d["name"].(string)
			if name == "" {
				continue
			}
			fn := openai.ToolFunction{Name: name}
			if desc, ok := d["description"].(string); ok {
				fn.Description = desc
			}
			if params, ok := d["parameters"].(map[string]any); ok {
				fn.Parameters = params
			}
			out.Tools = append(out.Tools, openai.Tool{Type: "function", Function: fn})
		}
		// googleSearch, codeExecution, urlContext have no chat equivalent.
	}
	if tc, ok := m["toolConfig"].(map[string]any); ok && len(out.Tools) > 0 {
		if fcc, ok := tc["functionCallingConfig"].(map[string]any); ok {
			switch mode := fccMode(fcc); mode {
			case "NONE":
				out.ToolChoice = "none"
			case "ANY":
				if names := fccNames(fcc); len(names) == 1 {
					out.ToolChoice = map[string]any{
						"type":     "function",
						"function": map[string]any{"name": names[0]},
					}
				} else {
					out.ToolChoice = "required"
				}
			default:
				// AUTO, unset, or unknown: let the model decide.
				out.ToolChoice = "auto"
			}
		}
	}

	if gc, ok := m["generationConfig"].(map[string]any); ok {
		if t, ok := numJSON(gc["temperature"]); ok {
			out.Temperature = &t
		}
		if t, ok := numJSON(gc["topP"]); ok {
			out.TopP = &t
		}
		if n := intJSON(gc["maxOutputTokens"]); n > 0 {
			out.MaxCompletionTokens = &n
		}
		if ss := toStringSlice(gc["stopSequences"]); len(ss) > 0 {
			out.Stop = ss
		}
	}
	// Gateway-only stream hint (top-level, non-standard) travels outside
	// generationConfig so it works with or without that block.
	if s, ok := m["stream"].(bool); ok {
		out.Stream = &s
	}

	lvl := EffortFromRawMap(m)
	// Native thinkingLevel wins over gateway aliases when present.
	if _, native := geminiNativeThinkingLevel(m); native != "" {
		if l := effort.FromGeminiLevel(native); l.Valid() {
			lvl = l
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

// geminiPartsText concatenates text parts.
func geminiPartsText(parts []map[string]any) string {
	var sb strings.Builder
	for _, p := range parts {
		if t, ok := p["text"].(string); ok {
			sb.WriteString(t)
		}
	}
	return sb.String()
}

// geminiFunctionCall maps a functionCall part to a chat tool call. Gemini
// carries no call id, so one is synthesized; function responses match by
// name on the way in.
func geminiFunctionCall(fc map[string]any) openai.ToolCall {
	name, _ := fc["name"].(string)
	args := "{}"
	if a, ok := fc["args"]; ok && a != nil {
		if bb, err := json.Marshal(a); err == nil {
			args = string(bb)
		}
	}
	return openai.ToolCall{
		ID:   "call_" + newID(),
		Type: "function",
		Function: openai.FunctionCall{
			Name:      name,
			Arguments: args,
		},
	}
}

// geminiFunctionResponseText flattens a functionResponse part to text.
func geminiFunctionResponseText(fr map[string]any) string {
	resp, ok := fr["response"].(map[string]any)
	if !ok {
		b, _ := json.Marshal(fr["response"])
		if len(b) > 0 && string(b) != "null" {
			return string(b)
		}
		return ""
	}
	if c, ok := resp["content"].(string); ok {
		return c
	}
	if out, ok := resp["output"]; ok {
		if s, ok := out.(string); ok {
			return s
		}
		if b, err := json.Marshal(out); err == nil {
			return string(b)
		}
	}
	if len(resp) == 1 {
		for _, v := range resp {
			if s, ok := v.(string); ok {
				return s
			}
		}
	}
	b, _ := json.Marshal(resp)
	return string(b)
}

// geminiInlineImageURL extracts an image data URL from an inlineData part.
func geminiInlineImageURL(p map[string]any) (string, bool) {
	inline, ok := p["inlineData"].(map[string]any)
	if !ok {
		return "", false
	}
	mime, _ := inline["mimeType"].(string)
	data, _ := inline["data"].(string)
	if data == "" {
		return "", false
	}
	if !strings.HasPrefix(strings.ToLower(mime), "image/") {
		return "", false
	}
	if mime == "" {
		mime = "image/png"
	}
	return "data:" + mime + ";base64," + data, true
}

// geminiInlineDocText decodes text/plain inlineData parts, including nested
// document structures.
func geminiInlineDocText(p map[string]any) string {
	if t, ok := p["text"].(string); ok && t != "" {
		return ""
	}
	inline, ok := p["inlineData"].(map[string]any)
	if !ok {
		return ""
	}
	mime, _ := inline["mimeType"].(string)
	data, _ := inline["data"].(string)
	if data == "" || strings.HasPrefix(strings.ToLower(mime), "image/") {
		return ""
	}
	if strings.ToLower(mime) == "text/plain" {
		if raw, err := base64.StdEncoding.DecodeString(data); err == nil {
			return string(raw)
		}
		return ""
	}
	if mime == "" {
		mime = "document"
	}
	return "[" + mime + " inline data omitted in cross-dialect translation]"
}

// geminiFileRefText renders fileData parts as placeholder notes: file URIs
// cannot transfer across dialects.
func geminiFileRefText(p map[string]any) string {
	fd, ok := p["fileData"].(map[string]any)
	if !ok {
		return ""
	}
	uri, _ := fd["fileUri"].(string)
	if uri == "" {
		uri, _ = fd["file_uri"].(string)
	}
	if uri == "" {
		return "[file reference omitted in cross-dialect translation]"
	}
	return "[file reference " + uri + " omitted in cross-dialect translation]"
}

func fccMode(fcc map[string]any) string {
	mode, _ := fcc["mode"].(string)
	if mode == "" {
		mode, _ = fcc["Mode"].(string)
	}
	return strings.ToUpper(strings.TrimSpace(mode))
}

func fccNames(fcc map[string]any) []string {
	var names []string
	if arr, ok := fcc["allowedFunctionNames"].([]any); ok {
		for _, n := range arr {
			if s, ok := n.(string); ok && s != "" {
				names = append(names, s)
			}
		}
	}
	return names
}

// ChatToGeminiResult wraps a ChatResponse as a Gemini generateContent result.
// Top-level responseId/model are best-effort echoes; multiple chat choices
// collapse to one candidate.
func ChatToGeminiResult(chat openai.ChatResponse, reqModel string) map[string]any {
	var parts []any
	finish := "STOP"
	if len(chat.Choices) > 0 {
		msg := chat.Choices[0].Message
		if t := messageText(msg.Content); t != "" {
			parts = append(parts, map[string]any{"text": t})
		}
		for _, tc := range msg.ToolCalls {
			var args any = map[string]any{}
			if strings.TrimSpace(tc.Function.Arguments) != "" {
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
				if args == nil {
					args = map[string]any{}
				}
			}
			parts = append(parts, map[string]any{
				"functionCall": map[string]any{
					"name": tc.Function.Name,
					"args": args,
				},
			})
		}
		finish = chatFinishToGemini(chat.Choices[0].FinishReason)
	}
	if len(parts) == 0 {
		parts = []any{map[string]any{"text": ""}}
	}
	model := chat.Model
	if model == "" {
		model = reqModel
	}
	out := map[string]any{
		"candidates": []any{
			map[string]any{
				"content":      map[string]any{"role": "model", "parts": parts},
				"finishReason": finish,
				"index":        0,
			},
		},
	}
	if chat.Usage != nil {
		out["usageMetadata"] = map[string]any{
			"promptTokenCount":     chat.Usage.PromptTokens,
			"candidatesTokenCount": chat.Usage.CompletionTokens,
			"totalTokenCount":      chat.Usage.TotalTokens,
		}
	}
	if model != "" {
		out["modelVersion"] = model
	}
	if chat.ID != "" {
		out["responseId"] = chat.ID
	}
	return out
}

func chatFinishToGemini(finish string) string {
	switch finish {
	case "length":
		return "MAX_TOKENS"
	case "content_filter":
		return "SAFETY"
	default:
		return "STOP"
	}
}
