// Package openai defines the OpenAI-compatible API surface the gateway exposes.
package openai

import "encoding/json"

// ---- Shared ----

type ResponseFormat struct {
	Type       string         `json:"type,omitempty"` // json_object, json_schema
	JSONSchema *JSONSchemaRef `json:"json_schema,omitempty"`
}

type JSONSchemaRef struct {
	Name   string         `json:"name,omitempty"`
	Schema map[string]any `json:"schema,omitempty"`
	Strict *bool          `json:"strict,omitempty"`
}

type Tool struct {
	Type     string       `json:"type"` // function
	Function ToolFunction `json:"function,omitempty"`
}

// UnmarshalJSON accepts both the chat shape
// ({"type":"function","function":{...}}) and the flat Responses shape
// ({"type":"function","name":...,"parameters":...}).
func (t *Tool) UnmarshalJSON(data []byte) error {
	type alias Tool
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	if a.Function.Name == "" {
		var flat struct {
			Name        string         `json:"name"`
			Description string         `json:"description"`
			Parameters  map[string]any `json:"parameters"`
			Strict      *bool          `json:"strict"`
		}
		if err := json.Unmarshal(data, &flat); err == nil && flat.Name != "" {
			a.Function = ToolFunction{
				Name:        flat.Name,
				Description: flat.Description,
				Parameters:  flat.Parameters,
				Strict:      flat.Strict,
			}
		}
	}
	*t = Tool(a)
	return nil
}

// MarshalJSON emits the chat shape; the Responses translators flatten
// tools explicitly when building upstream bodies.
func (t Tool) MarshalJSON() ([]byte, error) {
	type alias Tool
	return json.Marshal(alias(t))
}

type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
}

type ToolChoice struct {
	Type     string `json:"type,omitempty"` // none, auto, required, function
	Function *struct {
		Name string `json:"name"`
	} `json:"function,omitempty"`
}

// Unmarshal handles both string ("auto"/"none"/"required") and object forms.
func (t *ToolChoice) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		t.Type = s
		return nil
	}
	type alias ToolChoice
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*t = ToolChoice(a)
	return nil
}

// ---- Chat Completions ----

type ChatRequest struct {
	Model               string          `json:"model"`
	Messages            []ChatMessage   `json:"messages"`
	Stream              *bool           `json:"stream,omitempty"`
	StreamOptions       *StreamOptions  `json:"stream_options,omitempty"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	MaxTokens           *int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Stop                any             `json:"stop,omitempty"`
	Tools               []Tool          `json:"tools,omitempty"`
	ToolChoice          any             `json:"tool_choice,omitempty"`
	ParallelToolCalls   *bool           `json:"parallel_tool_calls,omitempty"`
	ResponseFormat      *ResponseFormat `json:"response_format,omitempty"`
	Seed                *int64          `json:"seed,omitempty"`
	User                string          `json:"user,omitempty"`
	// Thinking / reasoning controls (superset accepted from many clients).
	ReasoningEffort  string          `json:"reasoning_effort,omitempty"`
	ReasoningEffort2 string          `json:"reasoningEffort,omitempty"`
	Reasoning        *ReasoningParam `json:"reasoning,omitempty"`
	Effort           string          `json:"effort,omitempty"`
	Thinking         *ThinkingParam  `json:"thinking,omitempty"`
	ThinkingEffort   string          `json:"thinking_effort,omitempty"`
}

type ReasoningParam struct {
	Effort    string `json:"effort,omitempty"`
	MaxTokens *int   `json:"max_tokens,omitempty"`
	Summary   string `json:"summary,omitempty"`
	Exclude   *bool  `json:"exclude,omitempty"`
}

type ThinkingParam struct {
	Type         string `json:"type,omitempty"` // enabled/disabled
	BudgetTokens *int   `json:"budget_tokens,omitempty"`
	Effort       string `json:"effort,omitempty"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type ChatMessage struct {
	Role       string     `json:"role"` // system|user|assistant|tool|developer
	Content    any        `json:"content,omitempty"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type ContentPart struct {
	Type     string    `json:"type"` // text|image_url|input_audio|file|refusal|reasoning_content
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type ImageURL struct {
	URL    string `json:"url,omitempty"`
	Detail string `json:"detail,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"` // function
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ChatResponse struct {
	ID                string       `json:"id"`
	Object            string       `json:"object"` // chat.completion
	Created           int64        `json:"created"`
	Model             string       `json:"model"`
	SystemFingerprint string       `json:"system_fingerprint,omitempty"`
	Choices           []ChatChoice `json:"choices"`
	Usage             *Usage       `json:"usage,omitempty"`
}

type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason,omitempty"`
	Logprobs     any         `json:"logprobs,omitempty"`
}

type ChatStreamChunk struct {
	ID                string             `json:"id"`
	Object            string             `json:"object"` // chat.completion.chunk
	Created           int64              `json:"created"`
	Model             string             `json:"model"`
	SystemFingerprint string             `json:"system_fingerprint,omitempty"`
	Choices           []ChatStreamChoice `json:"choices"`
	Usage             *Usage             `json:"usage,omitempty"`
}

type ChatStreamChoice struct {
	Index        int             `json:"index"`
	Delta        ChatStreamDelta `json:"delta"`
	FinishReason *string         `json:"finish_reason,omitempty"`
	Logprobs     any             `json:"logprobs,omitempty"`
}

type ChatStreamDelta struct {
	Role      string          `json:"role,omitempty"`
	Content   string          `json:"content,omitempty"`
	Reasoning string          `json:"reasoning_content,omitempty"`
	ToolCalls []ToolCallDelta `json:"tool_calls,omitempty"`
}

type ToolCallDelta struct {
	Index    int        `json:"index"`
	ID       string     `json:"id,omitempty"`
	Type     string     `json:"type,omitempty"`
	Function *FuncDelta `json:"function,omitempty"`
}

type FuncDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type Usage struct {
	PromptTokens            int `json:"prompt_tokens"`
	CompletionTokens        int `json:"completion_tokens"`
	TotalTokens             int `json:"total_tokens"`
	PromptTokensDetails     any `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails any `json:"completion_tokens_details,omitempty"`
}

// ---- Responses API ----

type ResponsesRequest struct {
	Model              string          `json:"model"`
	Input              any             `json:"input,omitempty"`
	Instructions       string          `json:"instructions,omitempty"`
	Stream             *bool           `json:"stream,omitempty"`
	Temperature        *float64        `json:"temperature,omitempty"`
	TopP               *float64        `json:"top_p,omitempty"`
	MaxOutputTokens    *int            `json:"max_output_tokens,omitempty"`
	MaxToolCalls       *int            `json:"max_tool_calls,omitempty"`
	Tools              []Tool          `json:"tools,omitempty"`
	ToolChoice         any             `json:"tool_choice,omitempty"`
	ParallelToolCalls  *bool           `json:"parallel_tool_calls,omitempty"`
	Reasoning          *ReasoningParam `json:"reasoning,omitempty"`
	ReasoningEffort    string          `json:"reasoning_effort,omitempty"`
	Text               any             `json:"text,omitempty"`
	User               string          `json:"user,omitempty"`
	PreviousResponseID string          `json:"previous_response_id,omitempty"`
	Store              *bool           `json:"store,omitempty"`
}

type ResponsesResponse struct {
	ID                string       `json:"id"`
	Object            string       `json:"object"` // response
	CreatedAt         int64        `json:"created_at"`
	Model             string       `json:"model"`
	Status            string       `json:"status,omitempty"`
	Output            []RespOutput `json:"output"`
	OutputText        string       `json:"output_text,omitempty"`
	Usage             *Usage       `json:"usage,omitempty"`
	ParallelToolCalls bool         `json:"parallel_tool_calls,omitempty"`
	ToolChoice        any          `json:"tool_choice,omitempty"`
	Error             any          `json:"error,omitempty"`
}

type RespOutput struct {
	Type      string        `json:"type"` // message|function_call|reasoning|...
	ID        string        `json:"id,omitempty"`
	Status    string        `json:"status,omitempty"`
	Role      string        `json:"role,omitempty"`
	Content   []RespContent `json:"content,omitempty"`
	Name      string        `json:"name,omitempty"`
	CallID    string        `json:"call_id,omitempty"`
	Arguments string        `json:"arguments,omitempty"`
	Summary   []RespSummary `json:"summary,omitempty"`
}

type RespContent struct {
	Type        string `json:"type"` // output_text|refusal|...
	Text        string `json:"text,omitempty"`
	Annotations any    `json:"annotations,omitempty"`
}

type RespSummary struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ---- Models ----

type ModelsResponse struct {
	Object string       `json:"object"` // list
	Data   []ModelEntry `json:"data"`
}

type ModelEntry struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // model
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ---- Errors ----

type ErrorResponse struct {
	Error ErrorBody `json:"error"`
}

type ErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
	Code    any    `json:"code,omitempty"`
}
