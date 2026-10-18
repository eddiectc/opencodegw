package translate

import (
	"encoding/json"
	"strings"

	"github.com/eddiectc/opencodegw/internal/effort"
	"github.com/eddiectc/opencodegw/internal/openai"
)

// ChatNativeBody builds the upstream body for chat-native models from the raw
// client JSON, preserving every unknown/native field verbatim.
//
// Only surgical mutations are applied:
//   - model is normalized for upstream routing
//   - canonical effort is written to reasoning_effort (native knob)
//   - gateway-only aliases are dropped so strict upstreams do not 400:
//     reasoningEffort, effort, thinking, thinking_effort, reasoning
//     (the object form is Responses-only; chat uses reasoning_effort)
//   - explicit stream:false also drops stream_options
func ChatNativeBody(raw []byte, req openai.ChatRequest, model string, defEff effort.Level) map[string]any {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil || m == nil {
		// Fall back to the struct rebuild when raw is unusable.
		b, _ := json.Marshal(req)
		_ = json.Unmarshal(b, &m)
		if m == nil {
			m = map[string]any{}
		}
	}
	m["model"] = model

	eff := EffortFromChat(req)
	if !eff.Valid() {
		eff = defEff
	}
	if e := eff.OpenAIEffort(); e != "" {
		m["reasoning_effort"] = e
	} else {
		delete(m, "reasoning_effort")
	}
	delete(m, "reasoningEffort")
	delete(m, "effort")
	delete(m, "thinking")
	delete(m, "thinking_effort")
	delete(m, "reasoning")

	if s, ok := m["stream"].(bool); ok && !s {
		delete(m, "stream")
		delete(m, "stream_options")
	}
	return m
}

// ResponsesNativeBody builds the upstream body for responses-native models
// from the raw client JSON, preserving native/unknown fields.
//
// Surgical mutations only:
//   - model normalized
//   - effort merged into reasoning.effort, preserving other reasoning keys
//     (summary, max_tokens, exclude, future keys); effort=none drops the
//     reasoning block to disable; no effort anywhere leaves reasoning alone
//   - gateway-only aliases dropped: reasoning_effort, reasoningEffort,
//     effort, thinking, thinking_effort
//   - input sanitized for upstream validation (role-sensitive content types,
//     empty drops, tool pairing) while preserving input_audio and future
//     input_* part types
//   - tools normalized: non-function tools preserved verbatim, chat-shaped
//     function tools flattened, empty optionals stripped
func ResponsesNativeBody(raw []byte, req openai.ResponsesRequest, model string, defEff effort.Level) map[string]any {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil || m == nil {
		return ResponsesUpstreamBody(req, model, defEff)
	}
	m["model"] = model

	// Canonical effort: struct fields first, then raw-only aliases, then default.
	rawEff := EffortFromResponses(req)
	if !rawEff.Valid() {
		rawEff = effortFromResponsesRawMap(m)
	}
	if !rawEff.Valid() {
		rawEff = defEff
	}
	switch {
	case rawEff == effort.None:
		delete(m, "reasoning")
	case rawEff.Valid():
		clamped := rawEff.ClampForResponses()
		if clamped.Valid() {
			rm, ok := m["reasoning"].(map[string]any)
			if !ok || rm == nil {
				rm = map[string]any{}
			}
			rm["effort"] = clamped.OpenAIEffort()
			m["reasoning"] = rm
		}
	default:
		// Unspecified: leave any client-supplied reasoning block untouched.
	}
	delete(m, "reasoning_effort")
	delete(m, "reasoningEffort")
	delete(m, "effort")
	delete(m, "thinking")
	delete(m, "thinking_effort")

	if v, ok := m["input"]; ok {
		m["input"] = sanitizeResponsesInput(v)
	}
	if v, ok := m["tools"]; ok {
		m["tools"] = sanitizeResponsesToolsRaw(v)
	} else {
		delete(m, "tools")
	}
	if s, ok := m["stream"].(bool); !ok || !s {
		delete(m, "stream")
	}
	return m
}

// EffortFromResponsesRaw reads effort aliases from a raw Responses request
// body that the ResponsesRequest struct does not decode (clients may send
// any alias spelling to any endpoint). Reports Unspecified when absent.
func EffortFromResponsesRaw(raw []byte) effort.Level {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil || m == nil {
		return effort.Unspecified
	}
	return effortFromResponsesRawMap(m)
}

// EffortFromRawMap reads canonical effort from gateway alias spellings in a
// raw request map (any ingress dialect): top-level reasoning_effort,
// reasoningEffort, effort, thinking_effort strings (or 0-5 numbers), the
// reasoning.effort object form, and thinking blocks (effort string,
// budget_tokens, or type disabled -> none). A native Anthropic thinking
// block with a budget maps back to the nearest level. Reports Unspecified
// when no alias is present.
func EffortFromRawMap(m map[string]any) effort.Level {
	for _, k := range []string{"reasoning_effort", "reasoningEffort", "effort", "thinking_effort"} {
		if s, ok := m[k].(string); ok {
			if l := effort.Normalize(s); l.Valid() {
				return l
			}
		}
		if f, ok := m[k].(float64); ok {
			if l := effort.Normalize(floatToString(f)); l.Valid() {
				return l
			}
		}
	}
	if rm, ok := m["reasoning"].(map[string]any); ok {
		if s, ok := rm["effort"].(string); ok {
			if l := effort.Normalize(s); l.Valid() {
				return l
			}
		}
	}
	switch th := m["thinking"].(type) {
	case string:
		if l := effort.Normalize(th); l.Valid() {
			return l
		}
	case map[string]any:
		if s, ok := th["effort"].(string); ok {
			if l := effort.Normalize(s); l.Valid() {
				return l
			}
		}
		if b := intJSON(th["budget_tokens"]); b > 0 {
			return effort.FromBudget(b)
		}
		if s, ok := th["type"].(string); ok && strings.EqualFold(s, "disabled") {
			return effort.None
		}
	}
	if gc, ok := m["generationConfig"].(map[string]any); ok {
		if tc, ok := gc["thinkingConfig"].(map[string]any); ok {
			if s, ok := tc["thinkingLevel"].(string); ok {
				return effort.FromGeminiLevel(s)
			}
		}
	}
	return effort.Unspecified
}

// effortFromResponsesRawMap reads effort aliases that the ResponsesRequest
// struct does not decode (clients may send any alias to any endpoint).
func effortFromResponsesRawMap(m map[string]any) effort.Level {
	for _, k := range []string{"reasoningEffort", "effort", "thinking_effort"} {
		if s, ok := m[k].(string); ok {
			if l := effort.Normalize(s); l.Valid() {
				return l
			}
		}
		if f, ok := m[k].(float64); ok {
			if l := effort.Normalize(floatToString(f)); l.Valid() {
				return l
			}
		}
	}
	if th, ok := m["thinking"].(map[string]any); ok {
		if s, ok := th["effort"].(string); ok {
			if l := effort.Normalize(s); l.Valid() {
				return l
			}
		}
		if b, ok := th["budget_tokens"].(float64); ok {
			return effort.FromBudget(int(b))
		}
		if s, ok := th["type"].(string); ok && strings.EqualFold(s, "disabled") {
			return effort.None
		}
	}
	return effort.Unspecified
}

func floatToString(f float64) string {
	if f == float64(int(f)) {
		switch int(f) {
		case 0:
			return "0"
		case 1:
			return "1"
		case 2:
			return "2"
		case 3:
			return "3"
		case 4:
			return "4"
		case 5:
			return "5"
		}
	}
	return ""
}

// sanitizeResponsesToolsRaw normalizes a raw tools value for upstream
// Responses: non-function tools pass through verbatim, chat-shaped function
// tools are flattened, and null/empty optionals are stripped.
func sanitizeResponsesToolsRaw(v any) any {
	b, _ := json.Marshal(v)
	var tools []map[string]any
	if err := json.Unmarshal(b, &tools); err != nil {
		return v
	}
	out := make([]any, 0, len(tools))
	for _, t := range tools {
		typ, _ := t["type"].(string)
		if typ == "" {
			typ = "function"
		}
		if typ != "function" {
			out = append(out, t)
			continue
		}
		// Chat shape: {"type":"function","function":{...}} -> flatten.
		if inner, ok := t["function"].(map[string]any); ok {
			flat := map[string]any{"type": "function"}
			if n, ok := inner["name"].(string); ok && n != "" {
				flat["name"] = n
			} else if n, ok := t["name"].(string); ok && n != "" {
				flat["name"] = n
			} else {
				continue // nameless: drop rather than 400
			}
			if d, ok := inner["description"].(string); ok && d != "" {
				flat["description"] = d
			}
			if p, ok := inner["parameters"]; ok && p != nil {
				flat["parameters"] = p
			}
			if s, ok := inner["strict"]; ok && s != nil {
				flat["strict"] = s
			}
			// Preserve any other inner/outer keys verbatim.
			for k, val := range inner {
				if _, seen := flat[k]; !seen && k != "name" {
					flat[k] = val
				}
			}
			for k, val := range t {
				if _, seen := flat[k]; !seen && k != "function" && k != "type" {
					flat[k] = val
				}
			}
			out = append(out, flat)
			continue
		}
		// Flat shape: strip null/empty optionals, keep the rest verbatim.
		flat := map[string]any{}
		for k, val := range t {
			flat[k] = val
		}
		flat["type"] = "function"
		if n, ok := flat["name"].(string); !ok || n == "" {
			continue
		}
		if d, ok := flat["description"]; ok {
			if s, isStr := d.(string); !isStr || s == "" {
				if d == nil || isStr {
					delete(flat, "description")
				}
			}
		}
		if p, ok := flat["parameters"]; ok && p == nil {
			delete(flat, "parameters")
		}
		if s, ok := flat["strict"]; ok && s == nil {
			delete(flat, "strict")
		}
		out = append(out, flat)
	}
	return out
}
