package models

import "testing"

func TestLookupKnown(t *testing.T) {
	cases := map[string]Backend{
		"grok-4.7":                   BackendResponses,
		"gpt-6-luna":                 BackendResponses,
		"gpt-5.6-luna":               BackendResponses,
		"muse-spark-1.3-contributor": BackendResponses,
		"claude-haiku-5-5":           BackendMessages,
		"minimax-m3":                 BackendMessages,
		"qwen3.8-max":                BackendMessages,
		"qwen3.7-plus":               BackendMessages,
		"gemini-4-argon-contributor": BackendGemini,
		"glm-5.3":                    BackendChat,
		"kimi-k3":                    BackendChat,
		"deepseek-v4-pro":            BackendChat,
		"beam-free":                  BackendChat,
	}
	for model, want := range cases {
		if got := Lookup(model); got != want {
			t.Errorf("Lookup(%q) = %q, want %q", model, got, want)
		}
	}
}

func TestLookupPrefixes(t *testing.T) {
	if got := Lookup("opencode-go/kimi-k3"); got != BackendChat {
		t.Errorf("prefix strip failed: %q", got)
	}
	if got := Lookup("GPT-6-LUNA"); got != BackendResponses {
		t.Errorf("case-insensitive failed: %q", got)
	}
	if got := Lookup("gemini-future-x"); got != BackendGemini {
		t.Errorf("gemini heuristic failed: %q", got)
	}
	if got := Lookup("claude-future"); got != BackendMessages {
		t.Errorf("claude heuristic failed: %q", got)
	}
}

func TestUpstreamPath(t *testing.T) {
	if p := UpstreamPath(BackendChat, "kimi-k3", false); p != "openai/v1/chat/completions" {
		t.Errorf("chat path = %q", p)
	}
	if p := UpstreamPath(BackendResponses, "grok-4.7", false); p != "openai/v1/responses" {
		t.Errorf("responses path = %q", p)
	}
	if p := UpstreamPath(BackendMessages, "claude-haiku-5-5", false); p != "anthropic/v1/messages" {
		t.Errorf("messages path = %q", p)
	}
	if p := UpstreamPath(BackendGemini, "gemini-4-argon-contributor", false); p != "google/v1beta/models/gemini-4-argon-contributor:generateContent" {
		t.Errorf("gemini path = %q", p)
	}
	if p := UpstreamPath(BackendGemini, "gemini-4-argon-contributor", true); p != "google/v1beta/models/gemini-4-argon-contributor:streamGenerateContent?alt=sse" {
		t.Errorf("gemini stream path = %q", p)
	}
}

func TestContextWindow(t *testing.T) {
	if got := ContextWindow("muse-spark-1.3-contributor"); got != 1000000 {
		t.Errorf("muse-spark-1.3 = %d, want 1000000", got)
	}
	if got := ContextWindow("opencode-go/MUSE-SPARK-1.2-CONTRIBUTOR"); got != 1000000 {
		t.Errorf("normalized 1.2 = %d, want 1000000", got)
	}
	if got := ContextWindow("kimi-k3"); got != 0 {
		t.Errorf("unknown model = %d, want 0", got)
	}
}
