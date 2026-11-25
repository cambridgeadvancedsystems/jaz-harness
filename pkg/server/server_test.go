package server_test

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/harness"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/server"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/store"
)

type echoProvider struct{}

func (echoProvider) ID() string         { return "echo" }
func (echoProvider) WireFormat() string { return "echo" }
func (echoProvider) Stream(_ context.Context, req provider.Request, emit provider.StreamCallback) (provider.Response, error) {
	last := req.Messages[len(req.Messages)-1].TextContent()
	reply := "echo:" + last
	if emit != nil {
		emit(provider.StreamChunk{Kind: provider.DeltaText, Text: reply})
	}
	return provider.Response{
		ID:     "e1",
		Model:  req.Model,
		Blocks: []schema.ContentBlock{{Type: schema.BlockText, Text: reply}},
		Usage:  schema.Usage{InputTokens: 10, OutputTokens: 5},
	}, nil
}

func TestClientServerHTTPAndSSEReplay(t *testing.T) {
	provReg := provider.NewRegistry()
	provReg.Register("echo", echoProvider{})

	eng := harness.New(harness.Config{}, store.NewMemoryStore(), provReg, nil, nil, nil)
	srv := server.New(server.Config{AuthToken: "secret-123"}, eng)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Unauthenticated client must get 401.
	badCli := server.NewClient(ts.URL, "wrong", ts.Client())
	if _, err := badCli.CreateThread(context.Background(), schema.Thread{Provider: "echo", Model: "m1"}); err == nil {
		t.Fatal("expected unauthorized error")
	}

	cli := server.NewClient(ts.URL, "secret-123", ts.Client())
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	th, err := cli.CreateThread(ctx, schema.Thread{
		Title:    "Client/Server test",
		Provider: "echo",
		Model:    "m1",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	res, err := cli.RunTurn(ctx, th.ID, "hello remote", false)
	if err != nil || res.AssistantText != "echo:hello remote" {
		t.Fatalf("RunTurn: err=%v res=%+v", err, res)
	}

	// Verify SSE ?after_seq=0 replays monotonic events cleanly and stops on turn_completed.
	var gotEvents []schema.SessionEvent
	err = cli.StreamEvents(ctx, th.ID, 0, func(ev schema.SessionEvent) bool {
		gotEvents = append(gotEvents, ev)
		return ev.Type != schema.EventTurnCompleted
	})
	if err != nil {
		t.Fatalf("StreamEvents: %v", err)
	}
	if len(gotEvents) < 3 {
		t.Fatalf("expected at least 3 replayed SSE events, got %d", len(gotEvents))
	}
	var lastSeq int64
	sawDelta := false
	for _, ev := range gotEvents {
		if ev.Seq <= lastSeq {
			t.Fatalf("non-monotonic event seq %d after %d", ev.Seq, lastSeq)
		}
		lastSeq = ev.Seq
		if ev.Type == schema.EventAssistantDelta {
			if d, _ := ev.Payload["delta"].(string); strings.Contains(d, "echo:hello remote") {
				sawDelta = true
			}
		}
	}
	if !sawDelta {
		t.Fatalf("expected assistant_delta in replayed SSE stream: %+v", gotEvents)
	}

	msgs, err := cli.ListMessages(ctx, th.ID)
	if err != nil || len(msgs) != 2 {
		t.Fatalf("ListMessages: err=%v count=%d", err, len(msgs))
	}
}
