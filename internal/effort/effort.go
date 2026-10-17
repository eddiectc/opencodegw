// Package effort normalizes thinking/reasoning effort from many client
// dialects into one canonical level, then maps it onto each upstream
// backend's native knob.
package effort

import "strings"

// Level is the canonical effort level.
type Level string

const (
	Unspecified Level = ""
	None        Level = "none"
	Low         Level = "low"
	Medium      Level = "medium"
	High        Level = "high"
	XHigh       Level = "xhigh"
	Max         Level = "max"
)

// Normalize maps free-form client input to a canonical Level.
// Accepts: none/minimal/low/medium/high/xhigh/max, "effort" numbers 0-5,
// and common aliases (e.g. "minimal" -> low).
func Normalize(s string) Level {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "auto", "default":
		return Unspecified
	case "none", "off", "disabled", "0", "zero":
		return None
	case "minimal", "low", "1":
		return Low
	case "medium", "moderate", "balanced", "2":
		return Medium
	case "high", "3":
		return High
	case "xhigh", "x-high", "extra-high", "ultra", "4":
		return XHigh
	case "max", "maximum", "5":
		return Max
	default:
		return Unspecified
	}
}

// FromBudget maps an Anthropic-style thinking budget back to a Level.
func FromBudget(budget int) Level {
	switch {
	case budget <= 0:
		return None
	case budget <= 2048:
		return Low
	case budget <= 8000:
		return Medium
	case budget <= 16000:
		return High
	case budget <= 32000:
		return XHigh
	default:
		return Max
	}
}

// AnthropicBudget maps a Level to an Anthropic thinking budget_tokens value.
// Returns 0 when thinking should be disabled/omitted.
func (l Level) AnthropicBudget() int {
	switch l {
	case Low:
		return 2048
	case Medium:
		return 8192
	case High:
		return 16000
	case XHigh:
		return 32000
	case Max:
		return 64000
	default:
		return 0
	}
}

// GeminiLevel maps to Gemini thinkingLevel strings.
func (l Level) GeminiLevel() string {
	switch l {
	case None:
		return "MINIMAL"
	case Low:
		return "LOW"
	case Medium:
		return "MEDIUM"
	case High:
		return "HIGH"
	case XHigh, Max:
		return "HIGH"
	default:
		return ""
	}
}

// OpenAIEffort maps to the OpenAI reasoning effort string.
func (l Level) OpenAIEffort() string {
	switch l {
	case None:
		return "none"
	case Low:
		return "low"
	case Medium:
		return "medium"
	case High:
		return "high"
	case XHigh:
		return "xhigh"
	case Max:
		return "max"
	default:
		return ""
	}
}

// String returns the canonical string.
func (l Level) String() string { return string(l) }

// Valid reports whether l is a concrete level (not Unspecified).
func (l Level) Valid() bool {
	switch l {
	case None, Low, Medium, High, XHigh, Max:
		return true
	default:
		return false
	}
}
