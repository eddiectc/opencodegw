package effort

import "testing"

func TestNormalize(t *testing.T) {
	cases := map[string]Level{
		"":         Unspecified,
		"auto":     Unspecified,
		"none":     None,
		"off":      None,
		"minimal":  Low,
		"low":      Low,
		"medium":   Medium,
		"balanced": Medium,
		"high":     High,
		"xhigh":    XHigh,
		"x-high":   XHigh,
		"max":      Max,
		"maximum":  Max,
		"0":        None,
		"1":        Low,
		"2":        Medium,
		"3":        High,
		"4":        XHigh,
		"5":        Max,
		"bogus":    Unspecified,
	}
	for in, want := range cases {
		if got := Normalize(in); got != want {
			t.Errorf("Normalize(%q) = %q, want %q", in, got, want)
		}
	}
	// max must be distinct from xhigh
	if Normalize("max") == Normalize("xhigh") {
		t.Errorf("max and xhigh must be distinct levels")
	}
	for _, l := range []Level{None, Low, Medium, High, XHigh, Max} {
		if !l.Valid() {
			t.Errorf("level %q should be valid", l)
		}
	}
	if Unspecified.Valid() {
		t.Errorf("unspecified should not be valid")
	}
}

func TestBackendMappings(t *testing.T) {
	if Max.OpenAIEffort() != "max" {
		t.Errorf("Max.OpenAIEffort = %q, want max", Max.OpenAIEffort())
	}
	if XHigh.OpenAIEffort() != "xhigh" {
		t.Errorf("XHigh.OpenAIEffort = %q, want xhigh", XHigh.OpenAIEffort())
	}
	if Max.AnthropicBudget() <= XHigh.AnthropicBudget() {
		t.Errorf("Max budget %d should exceed XHigh %d", Max.AnthropicBudget(), XHigh.AnthropicBudget())
	}
	if FromBudget(64000) != Max {
		t.Errorf("FromBudget(64000) should be Max")
	}
	if FromBudget(32000) != XHigh {
		t.Errorf("FromBudget(32000) should be XHigh")
	}
	if Max.GeminiLevel() != "HIGH" {
		t.Errorf("Max.GeminiLevel = %q, want HIGH (Gemini tops out at HIGH)", Max.GeminiLevel())
	}
}

func TestClampForResponses(t *testing.T) {
	// none must be expressed by omitting reasoning, not sending "none"
	if got := None.ClampForResponses(); got.Valid() {
		t.Errorf("None.ClampForResponses = %q, want unspecified (omit block)", got)
	}
	for _, l := range []Level{Low, Medium, High, XHigh, Max} {
		if got := l.ClampForResponses(); got != l {
			t.Errorf("%q.ClampForResponses = %q, want passthrough", l, got)
		}
	}
	if got := Unspecified.ClampForResponses(); got.Valid() {
		t.Errorf("Unspecified.ClampForResponses = %q, want unspecified", got)
	}
}

func TestEnsureAnthropicMaxTokens(t *testing.T) {
	// budget 32000 (xhigh) with default 8192 must bump above budget
	if got := EnsureAnthropicMaxTokens(8192, 32000); got <= 32000 {
		t.Errorf("EnsureAnthropicMaxTokens(8192, 32000) = %d, must exceed budget", got)
	}
	// no thinking: small explicit max passes through
	if got := EnsureAnthropicMaxTokens(100, 0); got != 100 {
		t.Errorf("EnsureAnthropicMaxTokens(100, 0) = %d, want 100", got)
	}
	// already large enough: untouched
	if got := EnsureAnthropicMaxTokens(70000, 64000); got != 70000 {
		t.Errorf("EnsureAnthropicMaxTokens(70000, 64000) = %d, want 70000", got)
	}
}
