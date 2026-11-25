package store

import (
	"context"
	"sort"
	"sync"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
	"github.com/google/uuid"
)

type MemoryStore struct {
	mu          sync.RWMutex
	threads     map[string]schema.Thread
	messages    map[string][]schema.Message
	compactions map[string][]schema.CompactionCheckpoint
	events      map[string][]schema.SessionEvent
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		threads:     make(map[string]schema.Thread),
		messages:    make(map[string][]schema.Message),
		compactions: make(map[string][]schema.CompactionCheckpoint),
		events:      make(map[string][]schema.SessionEvent),
	}
}

func (s *MemoryStore) CreateThread(_ context.Context, th schema.Thread) (schema.Thread, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if th.ID == "" {
		th.ID = "thr_" + uuid.NewString()
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
	s.threads[th.ID] = th
	return th, nil
}

func (s *MemoryStore) GetThread(_ context.Context, threadID string) (schema.Thread, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	th, ok := s.threads[threadID]
	if !ok {
		return schema.Thread{}, ErrNotFound
	}
	return th, nil
}

func (s *MemoryStore) UpdateThread(_ context.Context, th schema.Thread) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.threads[th.ID]; !ok {
		return ErrNotFound
	}
	th.UpdatedAt = time.Now().UTC()
	s.threads[th.ID] = th
	return nil
}

func (s *MemoryStore) ListThreads(_ context.Context) ([]schema.Thread, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]schema.Thread, 0, len(s.threads))
	for _, th := range s.threads {
		out = append(out, th)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].UpdatedAt.After(out[j].UpdatedAt)
	})
	return out, nil
}

func (s *MemoryStore) AppendMessage(_ context.Context, threadID string, msg schema.Message) (schema.Message, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.threads[threadID]; !ok {
		return schema.Message{}, ErrNotFound
	}
	if msg.ID == "" {
		msg.ID = "msg_" + uuid.NewString()
	}
	if msg.CreatedAt.IsZero() {
		msg.CreatedAt = time.Now().UTC()
	}
	list := s.messages[threadID]
	msg.Seq = int64(len(list) + 1)
	s.messages[threadID] = append(list, msg)
	th := s.threads[threadID]
	th.UpdatedAt = time.Now().UTC()
	s.threads[threadID] = th
	return msg, nil
}

func (s *MemoryStore) ListMessages(_ context.Context, threadID string, afterSeq int64) ([]schema.Message, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if _, ok := s.threads[threadID]; !ok {
		return nil, ErrNotFound
	}
	list := s.messages[threadID]
	out := make([]schema.Message, 0, len(list))
	for _, m := range list {
		if m.Seq > afterSeq {
			out = append(out, m)
		}
	}
	return out, nil
}

func (s *MemoryStore) SaveCompaction(_ context.Context, cp schema.CompactionCheckpoint) (schema.CompactionCheckpoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.threads[cp.ThreadID]; !ok {
		return schema.CompactionCheckpoint{}, ErrNotFound
	}
	if cp.ID == "" {
		cp.ID = "cmp_" + uuid.NewString()
	}
	if cp.CreatedAt.IsZero() {
		cp.CreatedAt = time.Now().UTC()
	}
	list := s.compactions[cp.ThreadID]
	cp.CompactionSeq = int64(len(list) + 1)
	s.compactions[cp.ThreadID] = append(list, cp)
	return cp, nil
}

func (s *MemoryStore) LatestCompaction(_ context.Context, threadID string) (schema.CompactionCheckpoint, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if _, ok := s.threads[threadID]; !ok {
		return schema.CompactionCheckpoint{}, false, ErrNotFound
	}
	list := s.compactions[threadID]
	if len(list) == 0 {
		return schema.CompactionCheckpoint{}, false, nil
	}
	return list[len(list)-1], true, nil
}

func (s *MemoryStore) AppendEvent(_ context.Context, ev schema.SessionEvent) (schema.SessionEvent, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.threads[ev.ThreadID]; !ok {
		return schema.SessionEvent{}, ErrNotFound
	}
	if ev.CreatedAt.IsZero() {
		ev.CreatedAt = time.Now().UTC()
	}
	list := s.events[ev.ThreadID]
	ev.Seq = int64(len(list) + 1)
	s.events[ev.ThreadID] = append(list, ev)
	return ev, nil
}

func (s *MemoryStore) ListEventsAfter(_ context.Context, threadID string, afterSeq int64) ([]schema.SessionEvent, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if _, ok := s.threads[threadID]; !ok {
		return nil, ErrNotFound
	}
	list := s.events[threadID]
	out := make([]schema.SessionEvent, 0, len(list))
	for _, ev := range list {
		if ev.Seq > afterSeq {
			out = append(out, ev)
		}
	}
	return out, nil
}
