package translate

import "encoding/json"

// EstimateTokens roughly estimates the input tokens of an upstream request
// body as serialized bytes divided by 3. This is intentionally crude: it
// exists only to detect requests at or above a model's context window so
// the gateway can surface a clear context_length_exceeded error instead of
// proxying upstream's generic 400 message. Observed o200k densities on live
// probes range 0.19 (prose) to 0.35 (dense tool output/hex) tokens per byte,
// so bytes/3 catches dense bodies at the window while sparse bodies only
// over-estimate; over-estimation merely risks mislabeling a generic 400
// that upstream already returned, never a missed real cause.
func EstimateTokens(body any) int {
	if body == nil {
		return 0
	}
	b, err := json.Marshal(body)
	if err != nil {
		return 0
	}
	return len(b) / 3
}
