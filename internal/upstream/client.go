// Package upstream handles HTTP forwarding to OpenCode Go.
package upstream

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/eddiectc/opencodegw/internal/config"
)

// Client forwards requests upstream.
type Client struct {
	cfg config.Config
	hc  *http.Client
}

// New creates a Client.
func New(cfg config.Config) *Client {
	return &Client{
		cfg: cfg,
		hc: &http.Client{
			Timeout: time.Duration(cfg.UpstreamTimeoutSec) * time.Second,
		},
	}
}

// DoJSON sends a JSON body upstream and returns status, raw body, headers.
func (c *Client) DoJSON(ctx context.Context, method, path string, body any, extraHeaders http.Header, stream bool) (int, []byte, http.Header, error) {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return 0, nil, nil, fmt.Errorf("marshal upstream body: %w", err)
		}
		if c.cfg.Debug {
			fmt.Printf("[upstream] %s %s%s body=%s\n", method, c.cfg.UpstreamBase, path, string(b))
		}
		rdr = bytes.NewReader(b)
	}
	url := c.cfg.UpstreamBase + "/" + path
	req, err := http.NewRequestWithContext(ctx, method, url, rdr)
	if err != nil {
		return 0, nil, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", acceptHeader(stream))
	req.Header.Set("Authorization", "Bearer "+c.cfg.UpstreamAPIKey)
	if c.cfg.UserAgent != "" {
		req.Header.Set("User-Agent", c.cfg.UserAgent)
	}
	// Forward session + correlation + native version headers. Auth keys are
	// deliberately absent: client credentials authenticate to the gateway
	// only; upstream always uses Bearer <OPENCODE_API_KEY>.
	forwardExtraHeaders(req, extraHeaders)
	resp, err := c.hc.Do(req)
	if err != nil {
		return 0, nil, nil, fmt.Errorf("upstream request failed: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp.StatusCode, nil, resp.Header, err
	}
	if c.cfg.Debug && !stream {
		fmt.Printf("[upstream] status=%d body=%s\n", resp.StatusCode, string(raw))
	}
	if resp.StatusCode >= 400 {
		return resp.StatusCode, raw, resp.Header, nil
	}
	return resp.StatusCode, raw, resp.Header, nil
}

// DoStream sends a JSON body upstream with streaming and returns the raw *http.Response
// for the caller to relay/translate event-by-event. Caller must close the body.
func (c *Client) DoStream(ctx context.Context, method, path string, body any, extraHeaders http.Header) (*http.Response, error) {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshal upstream stream body: %w", err)
		}
		if c.cfg.Debug {
			fmt.Printf("[upstream stream] %s %s%s body=%s\n", method, c.cfg.UpstreamBase, path, string(b))
		}
		rdr = bytes.NewReader(b)
	}
	url := c.cfg.UpstreamBase + "/" + path
	req, err := http.NewRequestWithContext(ctx, method, url, rdr)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Cache-Control", "no-cache")
	req.Header.Set("Authorization", "Bearer "+c.cfg.UpstreamAPIKey)
	if c.cfg.UserAgent != "" {
		req.Header.Set("User-Agent", c.cfg.UserAgent)
	}
	forwardExtraHeaders(req, extraHeaders)
	// Long-lived client without the short timeout for streams.
	hc := &http.Client{Timeout: 0}
	resp, err := hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("upstream stream failed: %w", err)
	}
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, &UpstreamError{Status: resp.StatusCode, Body: raw, Header: resp.Header}
	}
	// Some upstreams answer stream requests with HTTP 200 + JSON (or plain
	// text) carrying an error object instead of SSE. Detect that before the
	// caller commits downstream SSE headers so the failure can be proxied as
	// JSON with no retry and no silent truncation.
	if ct := resp.Header.Get("Content-Type"); isNonStreamErrorContentType(ct) {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		resp.Body.Close()
		if c.cfg.Debug {
			fmt.Printf("[upstream stream] non-SSE status=%d content-type=%q body=%s\n", resp.StatusCode, ct, string(raw))
		}
		return nil, &UpstreamError{Status: resp.StatusCode, Body: raw, Header: resp.Header}
	}
	return resp, nil
}

// isNonStreamErrorContentType reports whether a stream response arrived with a
// non-SSE content type that indicates a plain (likely error) body.
func isNonStreamErrorContentType(ct string) bool {
	lct := strings.ToLower(ct)
	if lct == "" {
		return false
	}
	if strings.Contains(lct, "text/event-stream") {
		return false
	}
	// Streams are always SSE; JSON/text/HTML on a stream request is a
	// non-stream reply (usually an error object) that must be proxied, not
	// fed to the SSE translators.
	return strings.Contains(lct, "application/json") ||
		strings.Contains(lct, "text/plain") ||
		strings.Contains(lct, "text/html")
}

// DoRawGet performs an upstream GET (used for /v1/models).
func (c *Client) DoRawGet(ctx context.Context, path string) (int, []byte, http.Header, error) {
	url := c.cfg.UpstreamBase + "/" + path
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return 0, nil, nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.cfg.UpstreamAPIKey)
	if c.cfg.UserAgent != "" {
		req.Header.Set("User-Agent", c.cfg.UserAgent)
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return 0, nil, nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	return resp.StatusCode, raw, resp.Header, err
}

// UpstreamError is a non-2xx from streaming setup.
type UpstreamError struct {
	Status int
	Body   []byte
	Header http.Header
}

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("upstream status %d: %s", e.Status, truncate(string(e.Body), 2000))
}

func acceptHeader(stream bool) string {
	if stream {
		return "text/event-stream"
	}
	return "application/json"
}

// forwardExtraHeaders copies the allowlisted session/correlation/version
// headers onto the upstream request. Auth material is never forwarded.
func forwardExtraHeaders(req *http.Request, extraHeaders http.Header) {
	for k, vs := range extraHeaders {
		switch http.CanonicalHeaderKey(k) {
		case "X-Opencode-Session", "X-Session-Id", "X-Request-Id",
			"Anthropic-Beta", "Anthropic-Version", "Openai-Beta":
			for _, v := range vs {
				req.Header.Add(http.CanonicalHeaderKey(k), v)
			}
		}
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "...(truncated)"
}
