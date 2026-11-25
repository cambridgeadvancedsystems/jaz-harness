package store

import (
	"context"
	"errors"

	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
)

var ErrNotFound = errors.New("record not found")

// Store persists threads, canonical+provider-native turn history, atomic
// compaction checkpoints, and monotonic SSE replay events.
type Store interface {
	CreateThread(ctx context.Context, th schema.Thread) (schema.Thread, error)
	GetThread(ctx context.Context, threadID string) (schema.Thread, error)
	UpdateThread(ctx context.Context, th schema.Thread) error
	ListThreads(ctx context.Context) ([]schema.Thread, error)

	AppendMessage(ctx context.Context, threadID string, msg schema.Message) (schema.Message, error)
	ListMessages(ctx context.Context, threadID string, afterSeq int64) ([]schema.Message, error)

	SaveCompaction(ctx context.Context, cp schema.CompactionCheckpoint) (schema.CompactionCheckpoint, error)
	LatestCompaction(ctx context.Context, threadID string) (schema.CompactionCheckpoint, bool, error)

	AppendEvent(ctx context.Context, ev schema.SessionEvent) (schema.SessionEvent, error)
	ListEventsAfter(ctx context.Context, threadID string, afterSeq int64) ([]schema.SessionEvent, error)
}
