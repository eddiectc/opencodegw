// Command opencodegw is an OpenAI-compatible gateway for OpenCode Go.
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/eddiectc/opencodegw/internal/config"
	"github.com/eddiectc/opencodegw/internal/server"
)

func main() {
	cfg := config.Default().FromEnv()

	flag.StringVar(&cfg.Addr, "addr", cfg.Addr, "listen address (e.g. :8080)")
	flag.StringVar(&cfg.UpstreamBase, "upstream", cfg.UpstreamBase, "upstream OpenCode Go base URL")
	flag.StringVar(&cfg.UpstreamAPIKey, "api-key", cfg.UpstreamAPIKey, "upstream OpenCode Go API key (or OPENCODE_API_KEY)")
	flag.StringVar(&cfg.GatewayAPIKey, "gateway-key", cfg.GatewayAPIKey, "gateway auth key; empty = no auth (or GATEWAY_API_KEY)")
	flag.StringVar(&cfg.UserAgent, "user-agent", cfg.UserAgent, "User-Agent sent upstream")
	flag.StringVar(&cfg.DefaultEffort, "default-effort", cfg.DefaultEffort, "default thinking effort: none|low|medium|high|xhigh|max")
	flag.IntVar(&cfg.UpstreamTimeoutSec, "timeout", cfg.UpstreamTimeoutSec, "upstream timeout seconds")
	flag.BoolVar(&cfg.Debug, "debug", cfg.Debug, "debug logging")
	flag.Parse()

	// Env still wins for secrets if flags were not explicitly set? Keep simple:
	// re-overlay env so OPENCODE_API_KEY always works.
	cfg = cfg.FromEnv()

	if err := cfg.Validate(); err != nil {
		fmt.Fprintln(os.Stderr, "opencodegw: "+err.Error())
		flag.Usage()
		os.Exit(1)
	}

	srv := server.New(cfg)
	httpSrv := &http.Server{
		Addr:         cfg.Addr,
		Handler:      srv.Handler(),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 0, // streams are long-lived
		IdleTimeout:  120 * time.Second,
	}
	log.Printf("opencodegw listening on %s -> %s", cfg.Addr, cfg.UpstreamBase)
	if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
