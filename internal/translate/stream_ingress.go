package translate

import (
	"encoding/json"
	"io"
	"strings"

	"github.com/eddiectc/opencodegw/internal/openai"
)

// This file frames canonical OpenAI chat streams as Anthropic Messages and
// Gemini GenerateContent SSE for cross-dialect ingress (Anthropic/Gemini
// clients talking to non-native models). Native paths relay upstream SSE
// verbatim and never touch this code.

// ---- Chat stream -> Anthropic SSE ----

// AnthropicStreamFramer incrementally converts chat chunks into Anthropic SSE
// events. Text and thinking stream live; tool_use blocks stream live under
// the ( translator-guaranteed ) assumption that each tool's name precedes its
// argument deltas and tools do not interleave.
type AnthropicStreamFramer struct {
	model     string
	msgID     string
	started   bool
	nextIndex int
	openType  string // "", "text", "thinking", "tool"
	openIndex int
	toolBlock map[int]int // chat tool index -> anthropic block index
	// pending holds argument deltas that arrived before their tool name.
	pendingArgs map[int]string
	pendingID   map[int]string
	sawTool     bool
	finish      string
	usage       *openai.Usage
}

// NewAnthropicStreamFramer creates a framer for displayModel.
func NewAnthropicStreamFramer(displayModel string) *AnthropicStreamFramer {
	return &AnthropicStreamFramer{
		model:       displayModel,
		msgID:       "msg_" + newID(),
		toolBlock:   map[int]int{},
		pendingArgs: map[int]string{},
		pendingID:   map[int]string{},
	}
}

func anthropicJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func (f *AnthropicStreamFramer) ensureStarted(emit func(SSEEvent) bool, chunkModel string) bool {
	if f.started {
		return true
	}
	f.started = true
	if chunkModel != "" {
		f.model = chunkModel
	}
	return emit(SSEEvent{Event: "message_start", Data: anthropicJSON(map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": f.msgID, "type": "message", "role": "assistant",
			"model": f.model, "content": []any{},
			"stop_reason": nil,
			"usage":       map[string]any{"input_tokens": 0, "output_tokens": 0},
		},
	})})
}

func (f *AnthropicStreamFramer) closeOpen(emit func(SSEEvent) bool) bool {
	if f.openType == "" {
		return true
	}
	idx := f.openIndex
	f.openType = ""
	f.openIndex = -1
	return emit(SSEEvent{Event: "content_block_stop", Data: anthropicJSON(map[string]any{
		"type": "content_block_stop", "index": idx,
	})})
}

func (f *AnthropicStreamFramer) openText(emit func(SSEEvent) bool) bool {
	if !f.closeOpen(emit) {
		return false
	}
	idx := f.nextIndex
	f.nextIndex++
	if !emit(SSEEvent{Event: "content_block_start", Data: anthropicJSON(map[string]any{
		"type": "content_block_start", "index": idx,
		"content_block": map[string]any{"type": "text", "text": ""},
	})}) {
		return false
	}
	f.openType = "text"
	f.openIndex = idx
	return true
}

func (f *AnthropicStreamFramer) openThinking(emit func(SSEEvent) bool) bool {
	if !f.closeOpen(emit) {
		return false
	}
	idx := f.nextIndex
	f.nextIndex++
	if !emit(SSEEvent{Event: "content_block_start", Data: anthropicJSON(map[string]any{
		"type": "content_block_start", "index": idx,
		"content_block": map[string]any{"type": "thinking", "thinking": ""},
	})}) {
		return false
	}
	f.openType = "thinking"
	f.openIndex = idx
	return true
}

// Push converts one chat chunk JSON (or "[DONE]", ignored) into Anthropic
// events. It reports false when emit requests a stop.
func (f *AnthropicStreamFramer) Push(chunkJSON string, emit func(SSEEvent) bool) bool {
	if chunkJSON == "[DONE]" || strings.TrimSpace(chunkJSON) == "" {
		return true
	}
	var ch openai.ChatStreamChunk
	if err := json.Unmarshal([]byte(chunkJSON), &ch); err != nil {
		return true
	}
	if !f.ensureStarted(emit, ch.Model) {
		return false
	}
	if ch.Usage != nil {
		f.usage = ch.Usage
	}
	for _, c := range ch.Choices {
		if c.Delta.Content != "" {
			if f.openType != "text" {
				if !f.openText(emit) {
					return false
				}
			}
			if !emit(SSEEvent{Event: "content_block_delta", Data: anthropicJSON(map[string]any{
				"type": "content_block_delta", "index": f.openIndex,
				"delta": map[string]any{"type": "text_delta", "text": c.Delta.Content},
			})}) {
				return false
			}
		}
		if c.Delta.Reasoning != "" {
			if f.openType != "thinking" {
				if !f.openThinking(emit) {
					return false
				}
			}
			if !emit(SSEEvent{Event: "content_block_delta", Data: anthropicJSON(map[string]any{
				"type": "content_block_delta", "index": f.openIndex,
				"delta": map[string]any{"type": "thinking_delta", "thinking": c.Delta.Reasoning},
			})}) {
				return false
			}
		}
		for _, tc := range c.Delta.ToolCalls {
			var name, args string
			if tc.Function != nil {
				name = tc.Function.Name
				args = tc.Function.Arguments
			}
			blk, known := f.toolBlock[tc.Index]
			if name != "" && !known {
				if !f.closeOpen(emit) {
					return false
				}
				id := tc.ID
				if id == "" {
					id = f.pendingID[tc.Index]
				}
				if id == "" {
					id = "toolu_" + newID()
				}
				idx := f.nextIndex
				f.nextIndex++
				if !emit(SSEEvent{Event: "content_block_start", Data: anthropicJSON(map[string]any{
					"type": "content_block_start", "index": idx,
					"content_block": map[string]any{
						"type": "tool_use", "id": id, "name": name, "input": map[string]any{},
					},
				})}) {
					return false
				}
				f.toolBlock[tc.Index] = idx
				f.openType = "tool"
				f.openIndex = idx
				f.sawTool = true
				known = true
				blk = idx
				// Flush arguments that arrived before the name.
				if p := f.pendingArgs[tc.Index]; p != "" {
					args = p + args
					delete(f.pendingArgs, tc.Index)
					delete(f.pendingID, tc.Index)
				}
			}
			if args == "" {
				continue
			}
			if !known {
				// Arguments before any name: buffer until the name arrives.
				f.pendingArgs[tc.Index] += args
				if tc.ID != "" {
					f.pendingID[tc.Index] = tc.ID
				}
				continue
			}
			if f.openType != "tool" || f.openIndex != blk {
				// Another block is open (tools should not interleave, but be
				// lenient): close it and emit to the mapped index.
				if !f.closeOpen(emit) {
					return false
				}
				f.openType = "tool"
				f.openIndex = blk
			}
			if !emit(SSEEvent{Event: "content_block_delta", Data: anthropicJSON(map[string]any{
				"type": "content_block_delta", "index": blk,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": args},
			})}) {
				return false
			}
		}
		if c.FinishReason != nil && *c.FinishReason != "" && f.finish == "" {
			f.finish = *c.FinishReason
		}
	}
	return true
}

// Finish closes the stream: pending-only argument buffers without a name are
// dropped (no valid tool_use can be formed), the open block is stopped, and
// message_delta + message_stop are emitted.
func (f *AnthropicStreamFramer) Finish(emit func(SSEEvent) bool) {
	if !f.started {
		f.started = true
		if !emit(SSEEvent{Event: "message_start", Data: anthropicJSON(map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id": f.msgID, "type": "message", "role": "assistant",
				"model": f.model, "content": []any{},
				"stop_reason": nil,
				"usage":       map[string]any{"input_tokens": 0, "output_tokens": 0},
			},
		})}) {
			return
		}
	}
	if !f.closeOpen(emit) {
		return
	}
	if f.nextIndex == 0 {
		// Empty stream: emit one empty text block so content is valid.
		if !emit(SSEEvent{Event: "content_block_start", Data: anthropicJSON(map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "text", "text": ""},
		})}) {
			return
		}
		if !emit(SSEEvent{Event: "content_block_stop", Data: anthropicJSON(map[string]any{
			"type": "content_block_stop", "index": 0,
		})}) {
			return
		}
		f.nextIndex = 1
	}
	stop := "end_turn"
	if f.sawTool {
		stop = "tool_use"
	} else {
		switch f.finish {
		case "length":
			stop = "max_tokens"
		case "content_filter":
			stop = "refusal"
		case "stop_sequence":
			stop = "stop_sequence"
		}
	}
	delta := map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason": stop,
		},
	}
	if f.usage != nil {
		delta["usage"] = map[string]any{"output_tokens": f.usage.CompletionTokens}
	}
	if !emit(SSEEvent{Event: "message_delta", Data: anthropicJSON(delta)}) {
		return
	}
	emit(SSEEvent{Event: "message_stop", Data: anthropicJSON(map[string]any{
		"type": "message_stop",
	})})
}

// TranslateChatStreamToAnthropic converts a chat SSE stream into Anthropic SSE
// events. Upstream errors surface as *StreamError.
func TranslateChatStreamToAnthropic(r io.Reader, displayModel string, emit func(SSEEvent) bool) error {
	fr := NewAnthropicStreamFramer(displayModel)
	var streamErr *StreamError
	err := ScanSSE(r, func(ev SSEEvent) bool {
		if se, isErr := DetectSSEError(ev); isErr {
			streamErr = se
			return false
		}
		if ev.Data == "[DONE]" {
			return true
		}
		return fr.Push(ev.Data, emit)
	})
	if streamErr != nil {
		return streamErr
	}
	if err != nil {
		return err
	}
	fr.Finish(emit)
	return nil
}

// ---- Chat stream -> Gemini SSE ----

// geminiFunctionCallJSON keeps name-before-args key order for SDK snapshots.
type geminiFunctionCallJSON struct {
	Name string `json:"name"`
	Args any    `json:"args"`
}

type geminiToolBuf struct {
	name string
	args strings.Builder
}

// GeminiStreamFramer incrementally converts chat chunks into Gemini
// streamGenerateContent SSE payloads (JSON strings without the data prefix).
// Text streams live; function calls buffer until Finish because Gemini
// carries complete args, never deltas.
type GeminiStreamFramer struct {
	tools  map[int]*geminiToolBuf
	order  []int
	finish string
	usage  *openai.Usage
}

// NewGeminiStreamFramer creates a framer.
func NewGeminiStreamFramer() *GeminiStreamFramer {
	return &GeminiStreamFramer{tools: map[int]*geminiToolBuf{}}
}

// Push converts one chat chunk JSON (or "[DONE]", ignored). Reasoning deltas
// have no Gemini stream equivalent and are dropped.
func (f *GeminiStreamFramer) Push(chunkJSON string, emit func(string) bool) bool {
	if chunkJSON == "[DONE]" || strings.TrimSpace(chunkJSON) == "" {
		return true
	}
	var ch openai.ChatStreamChunk
	if err := json.Unmarshal([]byte(chunkJSON), &ch); err != nil {
		return true
	}
	if ch.Usage != nil {
		f.usage = ch.Usage
	}
	for _, c := range ch.Choices {
		if c.Delta.Content != "" {
			b, _ := json.Marshal(map[string]any{
				"candidates": []any{map[string]any{
					"content": map[string]any{
						"role":  "model",
						"parts": []any{map[string]any{"text": c.Delta.Content}},
					},
					"index": 0,
				}},
			})
			if !emit(string(b)) {
				return false
			}
		}
		for _, tc := range c.Delta.ToolCalls {
			buf, ok := f.tools[tc.Index]
			if !ok {
				buf = &geminiToolBuf{}
				f.tools[tc.Index] = buf
				f.order = append(f.order, tc.Index)
			}
			if tc.Function != nil {
				if tc.Function.Name != "" && buf.name == "" {
					buf.name = tc.Function.Name
				}
				buf.args.WriteString(tc.Function.Arguments)
			}
		}
		if c.FinishReason != nil && *c.FinishReason != "" && f.finish == "" {
			f.finish = *c.FinishReason
		}
	}
	return true
}

// Finish emits one buffered functionCall chunk per tool plus the final
// finishReason chunk.
func (f *GeminiStreamFramer) Finish(emit func(string) bool) {
	for _, idx := range f.order {
		buf := f.tools[idx]
		var args any = map[string]any{}
		if s := strings.TrimSpace(buf.args.String()); s != "" {
			_ = json.Unmarshal([]byte(s), &args)
			if args == nil {
				args = map[string]any{}
			}
		}
		b, _ := json.Marshal(map[string]any{
			"candidates": []any{map[string]any{
				"content": map[string]any{
					"role": "model",
					"parts": []any{map[string]any{
						"functionCall": geminiFunctionCallJSON{Name: buf.name, Args: args},
					}},
				},
				"index": 0,
			}},
		})
		if !emit(string(b)) {
			return
		}
	}
	final := map[string]any{
		"candidates": []any{map[string]any{
			"finishReason": chatFinishToGemini(f.finish),
			"index":        0,
		}},
	}
	if f.usage != nil {
		final["usageMetadata"] = map[string]any{
			"promptTokenCount":     f.usage.PromptTokens,
			"candidatesTokenCount": f.usage.CompletionTokens,
			"totalTokenCount":      f.usage.TotalTokens,
		}
	}
	b, _ := json.Marshal(final)
	emit(string(b))
}

// TranslateChatStreamToGemini converts a chat SSE stream into Gemini SSE
// payloads (JSON strings without the data prefix). Upstream errors surface
// as *StreamError.
func TranslateChatStreamToGemini(r io.Reader, emit func(string) bool) error {
	fr := NewGeminiStreamFramer()
	var streamErr *StreamError
	err := ScanSSE(r, func(ev SSEEvent) bool {
		if se, isErr := DetectSSEError(ev); isErr {
			streamErr = se
			return false
		}
		if ev.Data == "[DONE]" {
			return true
		}
		return fr.Push(ev.Data, emit)
	})
	if streamErr != nil {
		return streamErr
	}
	if err != nil {
		return err
	}
	fr.Finish(emit)
	return nil
}
