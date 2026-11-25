package server

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/harness"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/cambridgeadvancedsystems/jaz-harness/store"
)

type Config struct {
	AuthToken string // optional Bearer token required on all /v1/* routes
}

// Server exposes Jaz Harness over HTTP + SSE with monotonic `?after_seq=` event
// replay so Desktop, Web, iOS, CLI, or remote control planes can drive it remotely.
type Server struct {
	cfg    Config
	engine *harness.Engine
	mux    *http.ServeMux
}

func New(cfg Config, engine *harness.Engine) *Server {
	s := &Server{
		cfg:    cfg,
		engine: engine,
		mux:    http.NewServeMux(),
	}
	s.registerRoutes()
	return s
}

func (s *Server) Handler() http.Handler {
	return s.mux
}

func (s *Server) registerRoutes() {
	s.mux.HandleFunc("GET /v1/healthz", s.handleHealthz)
	s.mux.HandleFunc("GET /v1/models", s.authWrap(s.handleListModels))
	s.mux.HandleFunc("GET /v1/tools", s.authWrap(s.handleListTools))
	s.mux.HandleFunc("GET /v1/skills", s.authWrap(s.handleListSkills))
	s.mux.HandleFunc("GET /v1/plugins", s.authWrap(s.handleListPlugins))

	s.mux.HandleFunc("POST /v1/threads", s.authWrap(s.handleCreateThread))
	s.mux.HandleFunc("GET /v1/threads", s.authWrap(s.handleListThreads))
	s.mux.HandleFunc("GET /v1/threads/{id}", s.authWrap(s.handleGetThread))
	s.mux.HandleFunc("PATCH /v1/threads/{id}", s.authWrap(s.handlePatchThread))
	s.mux.HandleFunc("GET /v1/threads/{id}/messages", s.authWrap(s.handleListMessages))
	s.mux.HandleFunc("POST /v1/threads/{id}/turns", s.authWrap(s.handleStartTurn))
	s.mux.HandleFunc("POST /v1/threads/{id}/steer", s.authWrap(s.handleSteer))
	s.mux.HandleFunc("POST /v1/threads/{id}/cancel", s.authWrap(s.handleCancel))
	s.mux.HandleFunc("POST /v1/threads/{id}/compact", s.authWrap(s.handleCompact))
	s.mux.HandleFunc("GET /v1/threads/{id}/events", s.authWrap(s.handleEventsSSE))
}

func (s *Server) authWrap(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.AuthToken != "" {
			authz := r.Header.Get("Authorization")
			got := strings.TrimPrefix(authz, "Bearer ")
			if !strings.HasPrefix(authz, "Bearer ") || subtle.ConstantTimeCompare([]byte(got), []byte(s.cfg.AuthToken)) != 1 {
				writeError(w, http.StatusUnauthorized, "unauthorized")
				return
			}
		}
		next(w, r)
	}
}

func (s *Server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":        true,
		"providers": s.engine.Providers().Names(),
	})
}

func (s *Server) handleListModels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"providers": s.engine.Providers().Names(),
		"models":    schema.BuiltinModels(),
	})
}

func (s *Server) handleListTools(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"tools": s.engine.Tools().AllToolDefs(),
	})
}

func (s *Server) handleListSkills(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"skills": s.engine.Skills().List(),
	})
}

func (s *Server) handleListPlugins(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query().Get("q")
	writeJSON(w, http.StatusOK, map[string]any{
		"plugins": s.engine.Plugins().Search(q),
	})
}

const maxServerRequestBodyBytes = 64 << 20

func decodeJSONBody(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxServerRequestBodyBytes)
	return json.NewDecoder(r.Body).Decode(dst)
}

func (s *Server) handleCreateThread(w http.ResponseWriter, r *http.Request) {
	var body struct {
		ID              string            `json:"id"`
		Title           string            `json:"title"`
		Provider        string            `json:"provider"`
		Model           string            `json:"model"`
		ReasoningEffort string            `json:"reasoning_effort"`
		SystemPrompt    string            `json:"system_prompt"`
		Metadata        map[string]string `json:"metadata"`
	}
	if err := decodeJSONBody(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json")
		return
	}
	th, err := s.engine.CreateThread(r.Context(), schema.Thread{
		ID:              body.ID,
		Title:           body.Title,
		Provider:        body.Provider,
		Model:           body.Model,
		ReasoningEffort: body.ReasoningEffort,
		SystemPrompt:    body.SystemPrompt,
		Metadata:        body.Metadata,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, th)
}

func (s *Server) handleListThreads(w http.ResponseWriter, r *http.Request) {
	list, err := s.engine.ListThreads(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"threads": list})
}

func (s *Server) handleGetThread(w http.ResponseWriter, r *http.Request) {
	th, err := s.engine.GetThread(r.Context(), r.PathValue("id"))
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, "thread not found")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, th)
}

func (s *Server) handlePatchThread(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Provider        string `json:"provider"`
		Model           string `json:"model"`
		ReasoningEffort string `json:"reasoning_effort"`
	}
	if err := decodeJSONBody(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json")
		return
	}
	th, err := s.engine.UpdateThreadModel(r.Context(), r.PathValue("id"), body.Provider, body.Model, body.ReasoningEffort)
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, "thread not found")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, th)
}

func (s *Server) handleListMessages(w http.ResponseWriter, r *http.Request) {
	afterSeq, _ := strconv.ParseInt(r.URL.Query().Get("after_seq"), 10, 64)
	msgs, err := s.engine.Store().ListMessages(r.Context(), r.PathValue("id"), afterSeq)
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, "thread not found")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"messages": msgs})
}

func (s *Server) handleStartTurn(w http.ResponseWriter, r *http.Request) {
	threadID := r.PathValue("id")
	var body struct {
		Prompt          string               `json:"prompt"`
		Images          []schema.ImageSource `json:"images"`
		Provider        string               `json:"provider"`
		Model           string               `json:"model"`
		ReasoningEffort string               `json:"reasoning_effort"`
		EnableWebSearch *bool                `json:"enable_web_search"`
		Async           bool                 `json:"async"`
	}
	if err := decodeJSONBody(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json")
		return
	}
	req := harness.TurnRequest{
		ThreadID:        threadID,
		Prompt:          body.Prompt,
		Images:          body.Images,
		Provider:        body.Provider,
		Model:           body.Model,
		ReasoningEffort: body.ReasoningEffort,
		EnableWebSearch: body.EnableWebSearch,
	}
	if body.Async {
		if _, err := s.engine.GetThread(r.Context(), threadID); err != nil {
			if errors.Is(err, store.ErrNotFound) {
				writeError(w, http.StatusNotFound, "thread not found")
				return
			}
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		go func() {
			_, _ = s.engine.RunTurn(context.Background(), req)
		}()
		writeJSON(w, http.StatusAccepted, map[string]any{"accepted": true, "thread_id": threadID})
		return
	}

	res, err := s.engine.RunTurn(r.Context(), req)
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, "thread not found")
		return
	}
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, res)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (s *Server) handleSteer(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Text string `json:"text"`
	}
	if err := decodeJSONBody(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json")
		return
	}
	if err := s.engine.Steer(r.PathValue("id"), body.Text); err != nil {
		writeError(w, http.StatusConflict, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"steered": true})
}

func (s *Server) handleCancel(w http.ResponseWriter, r *http.Request) {
	ok := s.engine.Cancel(r.PathValue("id"))
	writeJSON(w, http.StatusOK, map[string]any{"cancelled": ok})
}

func (s *Server) handleCompact(w http.ResponseWriter, r *http.Request) {
	cp, did, err := s.engine.CompactThread(r.Context(), r.PathValue("id"))
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, "thread not found")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"compacted": did, "checkpoint": cp})
}

func (s *Server) handleEventsSSE(w http.ResponseWriter, r *http.Request) {
	threadID := r.PathValue("id")
	if _, err := s.engine.GetThread(r.Context(), threadID); err != nil {
		if errors.Is(err, store.ErrNotFound) {
			writeError(w, http.StatusNotFound, "thread not found")
			return
		}
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	afterSeq, _ := strconv.ParseInt(r.URL.Query().Get("after_seq"), 10, 64)
	if afterSeq == 0 {
		if lastID := strings.TrimSpace(r.Header.Get("Last-Event-ID")); lastID != "" {
			afterSeq, _ = strconv.ParseInt(lastID, 10, 64)
		}
	}

	// Subscribe BEFORE reading backlog so zero events fall in the gap.
	liveCh, unsub := s.engine.Subscribe(threadID)
	defer unsub()

	backlog, err := s.engine.Store().ListEventsAfter(r.Context(), threadID, afterSeq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache, no-store")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	flusher, _ := w.(http.Flusher)
	lastSent := afterSeq
	sendEvent := func(ev schema.SessionEvent) {
		if ev.Seq <= lastSent {
			return
		}
		lastSent = ev.Seq
		raw, _ := json.Marshal(ev)
		_, _ = fmt.Fprintf(w, "id: %d\nevent: %s\ndata: %s\n\n", ev.Seq, ev.Type, string(raw))
		if flusher != nil {
			flusher.Flush()
		}
	}

	for _, ev := range backlog {
		sendEvent(ev)
	}
	if flusher != nil {
		flusher.Flush()
	}

	keepalive := time.NewTicker(15 * time.Second)
	defer keepalive.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case <-keepalive.C:
			_, _ = fmt.Fprint(w, ": ping\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		case ev, ok := <-liveCh:
			if !ok {
				return
			}
			sendEvent(ev)
		}
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": msg})
}
