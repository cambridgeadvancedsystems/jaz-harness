package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/executor/execserver"
)

func main() {
	listenAddr := flag.String("listen", envOr("JAZ_EXEC_LISTEN", ":8080"), "Listen address (host:port or ws://host:port)")
	workspaceDir := flag.String("workspace", envOr("JAZ_EXEC_WORKSPACE", "."), "Default workspace directory")
	restrictWorkspace := flag.Bool("restrict-workspace", envBool("JAZ_EXEC_RESTRICT_WORKSPACE"), "Confine filesystem and cwd operations to workspace directory")
	tokenSHA256 := flag.String("token-sha256", os.Getenv("JAZ_EXEC_TOKEN_SHA256"), "Hex-encoded SHA-256 digest of the required bearer capability token")
	tokenFile := flag.String("token-file", os.Getenv("JAZ_EXEC_TOKEN_FILE"), "Optional file containing the raw capability token")
	token := flag.String("token", os.Getenv("JAZ_EXEC_TOKEN"), "Optional raw capability token (hashed with SHA-256 at startup)")
	allowUnauth := flag.Bool("allow-unauthenticated", false, "Allow unauthenticated WebSocket connections (local dev only)")
	maxReadBytes := flag.Int("max-read-bytes", execserver.DefaultMaxReadFileBytes, "Maximum bytes allowed per fs/readFile")
	maxWriteBytes := flag.Int("max-write-bytes", execserver.DefaultMaxWriteFileBytes, "Maximum bytes allowed per fs/writeFile")
	maxOutputBytes := flag.Int("max-output-bytes", execserver.DefaultMaxOutputBytes, "Maximum captured bytes per stdout/stderr stream")
	maxProcesses := flag.Int("max-processes", execserver.DefaultMaxProcesses, "Maximum concurrent/retained processes")
	killOnDisconnect := flag.Bool("kill-on-disconnect", true, "Terminate session process groups when WebSocket disconnects")
	flag.Parse()

	resolvedToken := strings.TrimSpace(*token)
	if resolvedToken == "" && strings.TrimSpace(*tokenFile) != "" {
		raw, err := os.ReadFile(strings.TrimSpace(*tokenFile))
		if err != nil {
			log.Fatalf("read token file: %v", err)
		}
		resolvedToken = strings.TrimSpace(string(raw))
	}

	killFlag := *killOnDisconnect
	srv, err := execserver.New(execserver.Config{
		WorkspaceDir:         *workspaceDir,
		RestrictToWorkspace:  *restrictWorkspace,
		Token:                resolvedToken,
		TokenSHA256:          strings.TrimSpace(*tokenSHA256),
		AllowUnauthenticated: *allowUnauth,
		MaxReadFileBytes:     *maxReadBytes,
		MaxWriteFileBytes:    *maxWriteBytes,
		MaxOutputBytes:       *maxOutputBytes,
		MaxProcesses:         *maxProcesses,
		KillOnDisconnect:     &killFlag,
	})
	if err != nil {
		log.Fatalf("init jaz-exec-server: %v", err)
	}

	addr := normalizeListenAddr(*listenAddr)
	httpSrv := &http.Server{
		Addr:              addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 15 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	shutdownDone := make(chan struct{})
	go func() {
		defer close(shutdownDone)
		<-ctx.Done()
		_ = srv.Close()
		shutCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(shutCtx)
	}()

	log.Printf("jaz-exec-server listening on %s (workspace=%s restrict=%v)", addr, srv.WorkspaceDir(), *restrictWorkspace)
	if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		stop()
		<-shutdownDone
		log.Fatalf("serve jaz-exec-server: %v", err)
	}
	<-shutdownDone
}

func normalizeListenAddr(raw string) string {
	trimmed := strings.TrimSpace(raw)
	for _, prefix := range []string{"ws://", "wss://", "http://", "https://"} {
		if strings.HasPrefix(trimmed, prefix) {
			trimmed = strings.TrimPrefix(trimmed, prefix)
			break
		}
	}
	if idx := strings.IndexByte(trimmed, '/'); idx >= 0 {
		trimmed = trimmed[:idx]
	}
	if trimmed == "" {
		return ":8080"
	}
	return trimmed
}

func envOr(key, fallback string) string {
	if val := strings.TrimSpace(os.Getenv(key)); val != "" {
		return val
	}
	return fallback
}

func envBool(key string) bool {
	val := strings.ToLower(strings.TrimSpace(os.Getenv(key)))
	return val == "1" || val == "true" || val == "yes"
}
