package store

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/google/uuid"
)

type threadSeqs struct {
	msgSeq        int64
	compactionSeq int64
	eventSeq      int64
	loaded        bool
}

// FileStore is a zero-CGO durable disk store using per-thread atomic metadata,
// O(1) cached sequence counters, and append-only JSONL logs.
type FileStore struct {
	root string
	mu   sync.Mutex
	seqs map[string]*threadSeqs
}

func NewFileStore(root string) (*FileStore, error) {
	if err := os.MkdirAll(filepath.Join(root, "threads"), 0o700); err != nil {
		return nil, err
	}
	return &FileStore{
		root: root,
		seqs: make(map[string]*threadSeqs),
	}, nil
}

func (s *FileStore) threadDir(threadID string) (string, error) {
	clean := strings.TrimSpace(threadID)
	if clean == "" || strings.ContainsAny(clean, `/\`) || strings.Contains(clean, "..") {
		return "", fmt.Errorf("invalid thread id %q", threadID)
	}
	return filepath.Join(s.root, "threads", clean), nil
}

func (s *FileStore) seqStateLocked(threadID, dir string) (*threadSeqs, error) {
	st, ok := s.seqs[threadID]
	if ok && st.loaded {
		return st, nil
	}
	msgCount, err := countJSONLLines(filepath.Join(dir, "messages.jsonl"))
	if err != nil {
		return nil, err
	}
	cmpCount, err := countJSONLLines(filepath.Join(dir, "compactions.jsonl"))
	if err != nil {
		return nil, err
	}
	evCount, err := countJSONLLines(filepath.Join(dir, "events.jsonl"))
	if err != nil {
		return nil, err
	}
	st = &threadSeqs{
		msgSeq:        msgCount,
		compactionSeq: cmpCount,
		eventSeq:      evCount,
		loaded:        true,
	}
	s.seqs[threadID] = st
	return st, nil
}

func (s *FileStore) CreateThread(_ context.Context, th schema.Thread) (schema.Thread, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if th.ID == "" {
		th.ID = "thr_" + uuid.NewString()
	}
	dir, err := s.threadDir(th.ID)
	if err != nil {
		return schema.Thread{}, err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return schema.Thread{}, err
	}
	now := time.Now().UTC()
	if th.CreatedAt.IsZero() {
		th.CreatedAt = now
	}
	if th.UpdatedAt.IsZero() {
		th.UpdatedAt = now
	}
	if th.Status == "" {
		th.Status = "idle"
	}
	if err := writeJSONAtomic(filepath.Join(dir, "thread.json"), th); err != nil {
		return schema.Thread{}, err
	}
	s.seqs[th.ID] = &threadSeqs{loaded: true}
	return th, nil
}

func (s *FileStore) GetThread(_ context.Context, threadID string) (schema.Thread, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.getThreadLocked(threadID)
}

func (s *FileStore) getThreadLocked(threadID string) (schema.Thread, error) {
	dir, err := s.threadDir(threadID)
	if err != nil {
		return schema.Thread{}, err
	}
	raw, err := os.ReadFile(filepath.Join(dir, "thread.json"))
	if os.IsNotExist(err) {
		return schema.Thread{}, ErrNotFound
	}
	if err != nil {
		return schema.Thread{}, err
	}
	var th schema.Thread
	if err := json.Unmarshal(raw, &th); err != nil {
		return schema.Thread{}, err
	}
	return th, nil
}

func (s *FileStore) UpdateThread(_ context.Context, th schema.Thread) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.getThreadLocked(th.ID); err != nil {
		return err
	}
	dir, err := s.threadDir(th.ID)
	if err != nil {
		return err
	}
	th.UpdatedAt = time.Now().UTC()
	return writeJSONAtomic(filepath.Join(dir, "thread.json"), th)
}

func (s *FileStore) ListThreads(_ context.Context) ([]schema.Thread, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	entries, err := os.ReadDir(filepath.Join(s.root, "threads"))
	if err != nil {
		return nil, err
	}
	out := make([]schema.Thread, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		th, err := s.getThreadLocked(entry.Name())
		if err == nil {
			out = append(out, th)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].UpdatedAt.After(out[j].UpdatedAt)
	})
	return out, nil
}

func (s *FileStore) AppendMessage(_ context.Context, threadID string, msg schema.Message) (schema.Message, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	th, err := s.getThreadLocked(threadID)
	if err != nil {
		return schema.Message{}, err
	}
	dir, _ := s.threadDir(threadID)
	st, err := s.seqStateLocked(threadID, dir)
	if err != nil {
		return schema.Message{}, err
	}
	if msg.ID == "" {
		msg.ID = "msg_" + uuid.NewString()
	}
	if msg.CreatedAt.IsZero() {
		msg.CreatedAt = time.Now().UTC()
	}
	st.msgSeq++
	msg.Seq = st.msgSeq
	if err := appendJSONL(filepath.Join(dir, "messages.jsonl"), msg); err != nil {
		st.msgSeq--
		return schema.Message{}, err
	}
	th.UpdatedAt = time.Now().UTC()
	_ = writeJSONAtomic(filepath.Join(dir, "thread.json"), th)
	return msg, nil
}

func (s *FileStore) ListMessages(_ context.Context, threadID string, afterSeq int64) ([]schema.Message, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.getThreadLocked(threadID); err != nil {
		return nil, err
	}
	dir, _ := s.threadDir(threadID)
	all, err := readJSONL[schema.Message](filepath.Join(dir, "messages.jsonl"))
	if err != nil {
		return nil, err
	}
	out := make([]schema.Message, 0, len(all))
	for _, m := range all {
		if m.Seq > afterSeq {
			out = append(out, m)
		}
	}
	return out, nil
}

func (s *FileStore) SaveCompaction(_ context.Context, cp schema.CompactionCheckpoint) (schema.CompactionCheckpoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.getThreadLocked(cp.ThreadID); err != nil {
		return schema.CompactionCheckpoint{}, err
	}
	dir, _ := s.threadDir(cp.ThreadID)
	st, err := s.seqStateLocked(cp.ThreadID, dir)
	if err != nil {
		return schema.CompactionCheckpoint{}, err
	}
	if cp.ID == "" {
		cp.ID = "cmp_" + uuid.NewString()
	}
	if cp.CreatedAt.IsZero() {
		cp.CreatedAt = time.Now().UTC()
	}
	st.compactionSeq++
	cp.CompactionSeq = st.compactionSeq
	if err := appendJSONL(filepath.Join(dir, "compactions.jsonl"), cp); err != nil {
		st.compactionSeq--
		return schema.CompactionCheckpoint{}, err
	}
	return cp, nil
}

func (s *FileStore) LatestCompaction(_ context.Context, threadID string) (schema.CompactionCheckpoint, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.getThreadLocked(threadID); err != nil {
		return schema.CompactionCheckpoint{}, false, err
	}
	dir, _ := s.threadDir(threadID)
	all, err := readJSONL[schema.CompactionCheckpoint](filepath.Join(dir, "compactions.jsonl"))
	if err != nil {
		return schema.CompactionCheckpoint{}, false, err
	}
	if len(all) == 0 {
		return schema.CompactionCheckpoint{}, false, nil
	}
	return all[len(all)-1], true, nil
}

func (s *FileStore) AppendEvent(_ context.Context, ev schema.SessionEvent) (schema.SessionEvent, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.getThreadLocked(ev.ThreadID); err != nil {
		return schema.SessionEvent{}, err
	}
	dir, _ := s.threadDir(ev.ThreadID)
	st, err := s.seqStateLocked(ev.ThreadID, dir)
	if err != nil {
		return schema.SessionEvent{}, err
	}
	if ev.CreatedAt.IsZero() {
		ev.CreatedAt = time.Now().UTC()
	}
	st.eventSeq++
	ev.Seq = st.eventSeq
	if err := appendJSONL(filepath.Join(dir, "events.jsonl"), ev); err != nil {
		st.eventSeq--
		return schema.SessionEvent{}, err
	}
	return ev, nil
}

func (s *FileStore) ListEventsAfter(_ context.Context, threadID string, afterSeq int64) ([]schema.SessionEvent, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.getThreadLocked(threadID); err != nil {
		return nil, err
	}
	dir, _ := s.threadDir(threadID)
	all, err := readJSONL[schema.SessionEvent](filepath.Join(dir, "events.jsonl"))
	if err != nil {
		return nil, err
	}
	out := make([]schema.SessionEvent, 0, len(all))
	for _, ev := range all {
		if ev.Seq > afterSeq {
			out = append(out, ev)
		}
	}
	return out, nil
}

func writeJSONAtomic(path string, v any) error {
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func appendJSONL(path string, v any) error {
	raw, err := json.Marshal(v)
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(raw)
	return err
}

func countJSONLLines(path string) (int64, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	defer f.Close()
	var n int64
	br := bufio.NewReaderSize(f, 64*1024)
	for {
		line, rerr := br.ReadString('\n')
		if strings.TrimSpace(line) != "" {
			n++
		}
		if rerr != nil {
			if errors.Is(rerr, io.EOF) {
				return n, nil
			}
			return n, rerr
		}
	}
}

func readJSONL[T any](path string) ([]T, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []T
	br := bufio.NewReaderSize(f, 64*1024)
	for {
		rawLine, rerr := br.ReadString('\n')
		line := strings.TrimSpace(rawLine)
		if line != "" {
			var item T
			if err := json.Unmarshal([]byte(line), &item); err != nil {
				return nil, err
			}
			out = append(out, item)
		}
		if rerr != nil {
			if errors.Is(rerr, io.EOF) {
				return out, nil
			}
			return nil, rerr
		}
	}
}
