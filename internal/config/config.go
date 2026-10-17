package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Config holds gateway runtime configuration.
// All values can be set via environment variables; flags are wired in main.
type Config struct {
	// Listen address, e.g. ":8080"
	Addr string
	// Upstream OpenCode Go base URL, e.g. https://opencode.ai/inference/go
	UpstreamBase string
	// Upstream API key (OpenCode Go / Console key). Forwarded as Authorization.
	UpstreamAPIKey string
	// Optional gateway auth: if set, clients must send Authorization: Bearer <GatewayAPIKey>.
	// Empty means no auth (useful for local dev).
	GatewayAPIKey string
	// User-Agent sent upstream. Docs ask for a real coding-agent UA, not a generic SDK name.
	UserAgent string
	// Default thinking effort when client does not specify one: none|minimal|low|medium|high|xhigh|max
	DefaultEffort string
	// Request timeout / upstream knobs
	UpstreamTimeoutSec int
	// Enable debug logging
	Debug bool
}

func Default() Config {
	return Config{
		Addr:               ":8080",
		UpstreamBase:       "https://opencode.ai/inference/go",
		UserAgent:          "opencodegw/0.1.0",
		DefaultEffort:      "",
		UpstreamTimeoutSec: 300,
		Debug:              false,
	}
}

// FromEnv overlays environment variables onto c.
func (c Config) FromEnv() Config {
	if v := os.Getenv("PORT"); v != "" && c.Addr == ":8080" {
		if !strings.Contains(v, ":") {
			v = ":" + v
		}
		c.Addr = v
	}
	if v := os.Getenv("ADDR"); v != "" {
		c.Addr = v
	}
	if v := os.Getenv("OPENCODE_BASE"); v != "" {
		c.UpstreamBase = strings.TrimSuffix(v, "/")
	}
	if v := os.Getenv("OPENCODE_API_KEY"); v != "" {
		c.UpstreamAPIKey = v
	}
	if v := os.Getenv("GATEWAY_API_KEY"); v != "" {
		c.GatewayAPIKey = v
	}
	if v := os.Getenv("USER_AGENT"); v != "" {
		c.UserAgent = v
	}
	if v := os.Getenv("DEFAULT_EFFORT"); v != "" {
		c.DefaultEffort = v
	}
	if v := os.Getenv("UPSTREAM_TIMEOUT_SEC"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.UpstreamTimeoutSec = n
		}
	}
	if v := os.Getenv("DEBUG"); v != "" {
		c.Debug = v == "1" || strings.EqualFold(v, "true")
	}
	return c
}

func (c Config) Validate() error {
	if c.UpstreamBase == "" {
		return fmt.Errorf("upstream base is empty")
	}
	if c.UpstreamAPIKey == "" {
		return fmt.Errorf("OPENCODE_API_KEY is empty: set it to your OpenCode Go API key")
	}
	return nil
}
