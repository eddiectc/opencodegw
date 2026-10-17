package translate

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/eddiectc/opencodegw/internal/openai"
)

// ChatChunk is one downstream OpenAI chunk ready to serialize as SSE data.
type ChatChunk = openai.ChatStreamChunk

// SSEEvent is a parsed upstream SSE event.
type SSEEvent struct {
	Event string
	Data  string
}

// ScanSSE reads an SSE stream and calls fn for each data-bearing event.
// Stops on io.EOF or when fn returns false. Ignores ": ping" comments.
func ScanSSE(r io.Reader, fn func(ev SSEEvent) bool) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	var event, data string
	var dataLines []string
	flush := func() bool {
		if len(dataLines) == 0 && event == "" {
			return true
		}
		joined := strings.Join(dataLines, "\n")
		if joined == "" && event == "" {
			event, dataLines = "", nil
			return true
		}
		ev := SSEEvent{Event: event, Data: joined}
		_ = data
		event, dataLines = "", nil
		if strings.TrimSpace(ev.Data) == "" && ev.Event == "" {
			return true
		}
		return fn(ev)
	}
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			if !flush() {
				return nil
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		if strings.HasPrefix(line, "event:") {
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		} else if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimPrefix(line, "data:"))
			// Preserve leading single space stripping per SSE spec
			if len(dataLines) > 0 && strings.HasPrefix(dataLines[len(dataLines)-1], " ") {
				dataLines[len(dataLines)-1] = dataLines[len(dataLines)-1][1:]
			}
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	flush()
	return nil
}

// NewChatChunk builds a chunk with fresh id/created.
func NewChatChunk(model string, delta openai.ChatStreamDelta, finish *string) openai.ChatStreamChunk {
	return openai.ChatStreamChunk{
		ID:      "chatcmpl-" + newID(),
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openai.ChatStreamChoice{
			{Index: 0, Delta: delta, FinishReason: finish},
		},
	}
}

func strPtr(s string) *string { return &s }

// ---- Upstream chat stream passthrough with model fix ----

// TranslateChatStream relays upstream OpenAI chat chunks, fixing the model echo.
// Returns translated chunks serialized as JSON strings (without "data: " prefix);
// the caller frames them as SSE.
func TranslateChatStream(r io.Reader, fallbackModel string, emit func(chunkJSON string) bool) error {
	var firstModel string
	var streamErr *StreamError
	err := ScanSSE(r, func(ev SSEEvent) bool {
		if se, isErr := DetectSSEError(ev); isErr {
			streamErr = se
			return false
		}
		if ev.Data == "[DONE]" {
			return emit("[DONE]")
		}
		var ch openai.ChatStreamChunk
		if err := json.Unmarshal([]byte(ev.Data), &ch); err != nil {
			// Skip non-JSON keepalives
			return true
		}
		if firstModel == "" && ch.Model != "" {
			firstModel = ch.Model
		}
		if ch.Model == "" {
			if firstModel != "" {
				ch.Model = firstModel
			} else {
				ch.Model = fallbackModel
			}
		}
		if ch.Object == "" {
			ch.Object = "chat.completion.chunk"
		}
		b, _ := json.Marshal(ch)
		return emit(string(b))
	})
	if streamErr != nil {
		return streamErr
	}
	return err
}

// ---- Upstream Responses stream -> OpenAI chat chunks ----

type responsesStreamEvent struct {
	Type        string `json:"type"`
	Delta       string `json:"delta,omitempty"`
	ItemID      string `json:"item_id,omitempty"`
	OutputIndex *int   `json:"output_index,omitempty"`
	Response    *struct {
		ID    string        `json:"id"`
		Model string        `json:"model"`
		Usage *openai.Usage `json:"usage"`
	} `json:"response,omitempty"`
	Item *struct {
		Type      string `json:"type"`
		Name      string `json:"name,omitempty"`
		CallID    string `json:"call_id,omitempty"`
		Arguments string `json:"arguments,omitempty"`
	} `json:"item,omitempty"`
}

// TranslateResponsesStreamToChat converts upstream Responses SSE into chat chunks.
func TranslateResponsesStreamToChat(r io.Reader, fallbackModel string, emit func(chunkJSON string) bool) error {
	model := fallbackModel
	// role preamble
	b0, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{Role: "assistant"}, nil))
	if !emit(string(b0)) {
		return nil
	}
	toolStarted := map[int]bool{}
	toolIndex := 0
	var streamErr *StreamError
	err := ScanSSE(r, func(ev SSEEvent) bool {
		if se, isErr := DetectSSEError(ev); isErr {
			streamErr = se
			return false
		}
		if ev.Data == "[DONE]" {
			return true // swallowed; caller appends DONE after finish chunk
		}
		var e responsesStreamEvent
		if err := json.Unmarshal([]byte(ev.Data), &e); err != nil {
			return true
		}
		switch e.Type {
		case "response.created", "response.in_progress", "response.output_item.added":
			if e.Response != nil && e.Response.Model != "" {
				model = e.Response.Model
			}
			if e.Item != nil && e.Item.Type == "function_call" {
				idx := toolIndex
				if e.OutputIndex != nil {
					idx = *e.OutputIndex
				}
				delta := openai.ChatStreamDelta{
					ToolCalls: []openai.ToolCallDelta{{
						Index: idx,
						ID:    e.Item.CallID,
						Type:  "function",
						Function: &openai.FuncDelta{
							Name: e.Item.Name,
						},
					}},
				}
				toolStarted[idx] = true
				b, _ := json.Marshal(NewChatChunk(model, delta, nil))
				return emit(string(b))
			}
			return true
		case "response.output_text.delta":
			b, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{Content: e.Delta}, nil))
			return emit(string(b))
		case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
			b, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{Reasoning: e.Delta}, nil))
			return emit(string(b))
		case "response.function_call_arguments.delta":
			idx := toolIndex
			if e.OutputIndex != nil {
				idx = *e.OutputIndex
			}
			if !toolStarted[idx] {
				toolStarted[idx] = true
			}
			b, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{
				ToolCalls: []openai.ToolCallDelta{{
					Index:    idx,
					Function: &openai.FuncDelta{Arguments: e.Delta},
				}},
			}, nil))
			return emit(string(b))
		case "response.output_item.done":
			if e.Item != nil && e.Item.Type == "function_call" {
				toolIndex++
			}
			return true
		case "response.completed", "response.incomplete":
			return true // finish chunk emitted by caller from accumulated state
		case "response.failed":
			// DetectSSEError already handled failures with error details.
			// A bare response.failed still must surface downstream.
			streamErr = &StreamError{
				Status: 502,
				Err:    UpstreamFailure(nil),
				Raw:    ev.Data,
			}
			streamErr.Err.Error.Message = "upstream response failed"
			streamErr.Err.Error.Type = "response.failed"
			streamErr.Err.Error.Code = "response.failed"
			return false
		default:
			// Generic fallbacks: some servers send delta-only events with other names
			if e.Delta != "" {
				b, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{Content: e.Delta}, nil))
				return emit(string(b))
			}
			return true
		}
	})
	if streamErr != nil {
		return streamErr
	}
	return err
}

// ---- Upstream Anthropic stream -> OpenAI chat chunks ----

// TranslateAnthropicStreamToChat converts Anthropic SSE into chat chunks.
func TranslateAnthropicStreamToChat(r io.Reader, fallbackModel string, emit func(chunkJSON string) bool) error {
	model := fallbackModel
	b0, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{Role: "assistant"}, nil))
	if !emit(string(b0)) {
		return nil
	}
	toolIDs := map[int]string{}
	toolNames := map[int]string{}
	var streamErr *StreamError
	err := ScanSSE(r, func(ev SSEEvent) bool {
		if se, isErr := DetectSSEError(ev); isErr {
			streamErr = se
			return false
		}
		if ev.Data == "[DONE]" {
			return true
		}
		var e map[string]any
		if err := json.Unmarshal([]byte(ev.Data), &e); err != nil {
			return true
		}
		t, _ := e["type"].(string)
		switch t {
		case "message_start":
			if msg, ok := e["message"].(map[string]any); ok {
				if m, ok := msg["model"].(string); ok && m != "" {
					model = m
				}
			}
			return true
		case "content_block_start":
			idx := intIndex(e["index"])
			if cb, ok := e["content_block"].(map[string]any); ok {
				if cb["type"] == "tool_use" {
					id, _ := cb["id"].(string)
					name, _ := cb["name"].(string)
					toolIDs[idx] = id
					toolNames[idx] = name
					b, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{
						ToolCalls: []openai.ToolCallDelta{{
							Index: idx, ID: id, Type: "function",
							Function: &openai.FuncDelta{Name: name},
						}},
					}, nil))
					return emit(string(b))
				}
			}
			return true
		case "content_block_delta":
			idx := intIndex(e["index"])
			d, _ := e["delta"].(map[string]any)
			if d == nil {
				return true
			}
			dt, _ := d["type"].(string)
			switch dt {
			case "text_delta":
				text, _ := d["text"].(string)
				b, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{Content: text}, nil))
				return emit(string(b))
			case "thinking_delta":
				th, _ := d["thinking"].(string)
				b, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{Reasoning: th}, nil))
				return emit(string(b))
			case "input_json_delta":
				pj, _ := d["partial_json"].(string)
				b, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{
					ToolCalls: []openai.ToolCallDelta{{
						Index: idx, Function: &openai.FuncDelta{Arguments: pj},
					}},
				}, nil))
				return emit(string(b))
			default:
				// Some servers use "text" directly
				if text, ok := d["text"].(string); ok && text != "" {
					b, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{Content: text}, nil))
					return emit(string(b))
				}
				return true
			}
		case "message_delta", "message_stop":
			return true
		default:
			return true
		}
	})
	if streamErr != nil {
		return streamErr
	}
	return err
}

// ---- Upstream Gemini stream -> OpenAI chat chunks ----

type geminiStreamChunk struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text         string `json:"text,omitempty"`
				FunctionCall *struct {
					Name string         `json:"name"`
					Args map[string]any `json:"args"`
				} `json:"functionCall,omitempty"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
}

// TranslateGeminiStreamToChat converts Gemini SSE (streamGenerateContent) into chat chunks.
func TranslateGeminiStreamToChat(r io.Reader, fallbackModel string, emit func(chunkJSON string) bool) error {
	model := fallbackModel
	b0, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{Role: "assistant"}, nil))
	if !emit(string(b0)) {
		return nil
	}
	toolIdx := 0
	toolOpen := false
	var streamErr *StreamError
	err := ScanSSE(r, func(ev SSEEvent) bool {
		if se, isErr := DetectSSEError(ev); isErr {
			streamErr = se
			return false
		}
		if ev.Data == "[DONE]" {
			return true
		}
		var g geminiStreamChunk
		if err := json.Unmarshal([]byte(ev.Data), &g); err != nil {
			return true
		}
		for _, cand := range g.Candidates {
			for _, p := range cand.Content.Parts {
				if p.Text != "" {
					b, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{Content: p.Text}, nil))
					if !emit(string(b)) {
						return false
					}
				}
				if p.FunctionCall != nil {
					argBytes, _ := json.Marshal(p.FunctionCall.Args)
					if !toolOpen {
						b, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{
							ToolCalls: []openai.ToolCallDelta{{
								Index: toolIdx, ID: fmt.Sprintf("call_%s", newID()[:8]), Type: "function",
								Function: &openai.FuncDelta{Name: p.FunctionCall.Name},
							}},
						}, nil))
						if !emit(string(b)) {
							return false
						}
						toolOpen = true
					}
					b, _ := json.Marshal(NewChatChunk(model, openai.ChatStreamDelta{
						ToolCalls: []openai.ToolCallDelta{{
							Index: toolIdx, Function: &openai.FuncDelta{Arguments: string(argBytes)},
						}},
					}, nil))
					if !emit(string(b)) {
						return false
					}
					toolIdx++
					toolOpen = false
				}
			}
		}
		return true
	})
	if streamErr != nil {
		return streamErr
	}
	return err
}

func intIndex(v any) int {
	switch t := v.(type) {
	case float64:
		return int(t)
	case int:
		return t
	case int64:
		return int(t)
	default:
		return 0
	}
}
