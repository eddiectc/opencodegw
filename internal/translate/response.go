package translate

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/eddiectc/opencodegw/internal/openai"
)

// ---- Upstream non-streaming responses -> OpenAI ChatResponse ----

// ChatPassthrough normalizes an upstream chat/completions body into a ChatResponse.
// Upstream is already OpenAI-shaped; this mainly fixes the model echo and ensures defaults.
func ChatPassthrough(raw []byte, fallbackModel string) (openai.ChatResponse, error) {
	var resp openai.ChatResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return resp, fmt.Errorf("decode upstream chat response: %w body=%s", err, head(raw))
	}
	if resp.Object == "" {
		resp.Object = "chat.completion"
	}
	if resp.Model == "" {
		resp.Model = fallbackModel
	}
	if resp.ID == "" {
		resp.ID = "chatcmpl-" + newID()
	}
	if resp.Created == 0 {
		resp.Created = time.Now().Unix()
	}
	return resp, nil
}

// ResponsesToChat converts an OpenAI Responses body into a ChatResponse.
func ResponsesToChat(raw []byte, fallbackModel string) (openai.ChatResponse, error) {
	var rr openai.ResponsesResponse
	if err := json.Unmarshal(raw, &rr); err != nil {
		return openai.ChatResponse{}, fmt.Errorf("decode upstream responses body: %w body=%s", err, head(raw))
	}
	msg := openai.ChatMessage{Role: "assistant"}
	var reasoning string
	var toolCalls []openai.ToolCall
	finish := "stop"
	for _, item := range rr.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				switch c.Type {
				case "output_text":
					if s, ok := msg.Content.(string); ok {
						msg.Content = s + c.Text
					} else if msg.Content == nil {
						msg.Content = c.Text
					}
				case "refusal":
					if s, ok := msg.Content.(string); ok {
						msg.Content = s + c.Text
					} else if msg.Content == nil {
						msg.Content = c.Text
					}
				}
			}
		case "reasoning":
			for _, s := range item.Summary {
				reasoning += s.Text
			}
		case "function_call":
			toolCalls = append(toolCalls, openai.ToolCall{
				ID:   item.CallID,
				Type: "function",
				Function: openai.FunctionCall{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			})
			finish = "tool_calls"
		}
	}
	if msg.Content == nil {
		msg.Content = ""
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}
	_ = reasoning // reasoning summaries are dropped in chat view for now; kept in Responses endpoint
	model := rr.Model
	if model == "" {
		model = fallbackModel
	}
	id := rr.ID
	if id == "" {
		id = "chatcmpl-" + newID()
	}
	return openai.ChatResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openai.ChatChoice{
			{Index: 0, Message: msg, FinishReason: finish},
		},
		Usage: rr.Usage,
	}, nil
}

// AnthropicToChat converts an Anthropic Messages body into a ChatResponse.
func AnthropicToChat(raw []byte, fallbackModel string) (openai.ChatResponse, error) {
	var ar anthropicResponse
	if err := json.Unmarshal(raw, &ar); err != nil {
		return openai.ChatResponse{}, fmt.Errorf("decode upstream anthropic body: %w body=%s", err, head(raw))
	}
	msg := openai.ChatMessage{Role: "assistant"}
	var sb strings.Builder
	var thinking strings.Builder
	var toolCalls []openai.ToolCall
	finish := anthropicStopToFinish(ar.StopReason)
	for _, b := range ar.Content {
		switch b.Type {
		case "text":
			sb.WriteString(b.Text)
		case "thinking":
			thinking.WriteString(b.Thinking)
		case "tool_use":
			argBytes, _ := json.Marshal(b.Input)
			toolCalls = append(toolCalls, openai.ToolCall{
				ID:   b.ID,
				Type: "function",
				Function: openai.FunctionCall{
					Name:      b.Name,
					Arguments: string(argBytes),
				},
			})
			finish = "tool_calls"
		}
	}
	msg.Content = sb.String()
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}
	_ = thinking
	return openai.ChatResponse{
		ID:      "chatcmpl-" + orDefault(ar.ID, newID()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   orDefault(ar.Model, fallbackModel),
		Choices: []openai.ChatChoice{
			{Index: 0, Message: msg, FinishReason: finish},
		},
		Usage: &openai.Usage{
			PromptTokens:     ar.Usage.InputTokens,
			CompletionTokens: ar.Usage.OutputTokens,
			TotalTokens:      ar.Usage.InputTokens + ar.Usage.OutputTokens,
		},
	}, nil
}

// GeminiToChat converts a Gemini generateContent body into a ChatResponse.
func GeminiToChat(raw []byte, fallbackModel string) (openai.ChatResponse, error) {
	var gr geminiResponse
	if err := json.Unmarshal(raw, &gr); err != nil {
		return openai.ChatResponse{}, fmt.Errorf("decode upstream gemini body: %w body=%s", err, head(raw))
	}
	msg := openai.ChatMessage{Role: "assistant"}
	var sb strings.Builder
	var toolCalls []openai.ToolCall
	finish := "stop"
	for _, cand := range gr.Candidates {
		if f := geminiFinishToOpenAI(cand.FinishReason); f != "" {
			finish = f
		}
		for _, p := range cand.Content.Parts {
			if p.Text != "" {
				sb.WriteString(p.Text)
			}
			if p.FunctionCall != nil {
				argBytes, _ := json.Marshal(p.FunctionCall.Args)
				toolCalls = append(toolCalls, openai.ToolCall{
					ID:   "call_" + newID(),
					Type: "function",
					Function: openai.FunctionCall{
						Name:      p.FunctionCall.Name,
						Arguments: string(argBytes),
					},
				})
				finish = "tool_calls"
			}
		}
	}
	msg.Content = sb.String()
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}
	usage := &openai.Usage{}
	if gr.UsageMetadata != nil {
		usage.PromptTokens = gr.UsageMetadata.PromptTokenCount
		usage.CompletionTokens = gr.UsageMetadata.CandidatesTokenCount
		usage.TotalTokens = gr.UsageMetadata.TotalTokenCount
	}
	return openai.ChatResponse{
		ID:      "chatcmpl-" + newID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   fallbackModel,
		Choices: []openai.ChatChoice{
			{Index: 0, Message: msg, FinishReason: finish},
		},
		Usage: usage,
	}, nil
}

// ---- ChatResponse -> ResponsesResponse (for gateway /v1/responses on non-responses backends) ----

// ChatToResponsesResult wraps a ChatResponse as a ResponsesResponse.
func ChatToResponsesResult(chat openai.ChatResponse, reqModel string) openai.ResponsesResponse {
	var text string
	var toolItems []openai.RespOutput
	if len(chat.Choices) > 0 {
		m := chat.Choices[0].Message
		text = messageText(m.Content)
		for _, tc := range m.ToolCalls {
			toolItems = append(toolItems, openai.RespOutput{
				Type:      "function_call",
				ID:        tc.ID,
				CallID:    tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
				Status:    "completed",
			})
		}
	}
	output := []openai.RespOutput{
		{
			Type:   "message",
			ID:     "msg_" + newID(),
			Status: "completed",
			Role:   "assistant",
			Content: []openai.RespContent{
				{Type: "output_text", Text: text},
			},
		},
	}
	output = append(output, toolItems...)
	model := chat.Model
	if model == "" {
		model = reqModel
	}
	return openai.ResponsesResponse{
		ID:        orDefault(chat.ID, "resp_"+newID()),
		Object:    "response",
		CreatedAt: chat.Created,
		Model:     model,
		Status:    "completed",
		Output:    output,
		Usage:     chat.Usage,
	}
}

// ResponsesPassthrough normalizes an upstream responses body, fixing echo fields.
func ResponsesPassthrough(raw []byte, fallbackModel string) (openai.ResponsesResponse, error) {
	var rr openai.ResponsesResponse
	if err := json.Unmarshal(raw, &rr); err != nil {
		return rr, fmt.Errorf("decode upstream responses body: %w body=%s", err, head(raw))
	}
	if rr.Object == "" {
		rr.Object = "response"
	}
	if rr.Model == "" {
		rr.Model = fallbackModel
	}
	if rr.ID == "" {
		rr.ID = "resp_" + newID()
	}
	if rr.CreatedAt == 0 {
		rr.CreatedAt = time.Now().Unix()
	}
	return rr, nil
}

// ---- upstream wire shapes ----

type anthropicResponse struct {
	ID         string `json:"id"`
	Model      string `json:"model"`
	StopReason string `json:"stop_reason"`
	Content    []struct {
		Type     string         `json:"type"`
		Text     string         `json:"text,omitempty"`
		Thinking string         `json:"thinking,omitempty"`
		ID       string         `json:"id,omitempty"`
		Name     string         `json:"name,omitempty"`
		Input    map[string]any `json:"input,omitempty"`
	} `json:"content"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type geminiResponse struct {
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
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata *struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

func anthropicStopToFinish(s string) string {
	switch s {
	case "tool_use":
		return "tool_calls"
	case "max_tokens":
		return "length"
	case "stop_sequence", "end_turn":
		return "stop"
	default:
		if s == "" {
			return "stop"
		}
		return "stop"
	}
}

func geminiFinishToOpenAI(s string) string {
	switch strings.ToUpper(s) {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
		return "content_filter"
	default:
		return ""
	}
}

func orDefault(s, d string) string {
	if s != "" {
		return s
	}
	return d
}

func head(b []byte) string {
	if len(b) > 2000 {
		return string(b[:2000]) + "...(truncated)"
	}
	return string(b)
}
