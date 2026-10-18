package translate

import "encoding/json"

// EstimateTokens roughly estimates the input tokens of an upstream request
// body as serialized bytes divided by 4. This is intentionally crude: it
// exists only to detect requests near a model's context window so the
// gateway can surface a clear context_length_exceeded error instead of
// proxying upstream's generic 400 message. Real token density for tool
// outputs is typically 0.2-0.35 tokens/byte, so bytes/4 over-estimates,
// which fails safe (earlier, clearer errors, never a missed real cause).
func EstimateTokens(body any) int {
	if body == nil {
		return 0
	}
	b, err := json.Marshal(body)
	if err != nil {
		return 0
	}
	return len(b) / 4
}
