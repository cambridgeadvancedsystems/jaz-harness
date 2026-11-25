package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/harness"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
)

// Client is a thin Go SDK for driving a remote `jaz-harness serve` daemon over
// HTTP + SSE with automatic `?after_seq=` event streaming.
type Client struct {
	baseURL    string
	authToken  string
	httpClient *http.Client
}

func NewClient(baseURL, authToken string, hc *http.Client) *Client {
	if hc == nil {
		hc = &http.Client{Timeout: 5 * time.Minute}
	}
	return &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		authToken:  authToken,
		httpClient: hc,
	}
}

func (c *Client) doJSON(ctx context.Context, method, path string, in any, out any) error {
	var body io.Reader
	if in != nil {
		raw, err := json.Marshal(in)
		if err != nil {
			return err
		}
		body = bytes.NewReader(raw)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, body)
	if err != nil {
		return err
	}
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.authToken)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode >= 300 {
		return fmt.Errorf("server %s %s status %d: %s", method, path, resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	if out != nil && len(raw) > 0 {
		return json.Unmarshal(raw, out)
	}
	return nil
}

func (c *Client) CreateThread(ctx context.Context, th schema.Thread) (schema.Thread, error) {
	var out schema.Thread
	err := c.doJSON(ctx, http.MethodPost, "/v1/threads", th, &out)
	return out, err
}

func (c *Client) SwitchModel(ctx context.Context, threadID, providerName, model, effort string) (schema.Thread, error) {
	var out schema.Thread
	err := c.doJSON(ctx, http.MethodPatch, "/v1/threads/"+threadID, map[string]any{
		"provider":         providerName,
		"model":            model,
		"reasoning_effort": effort,
	}, &out)
	return out, err
}

func (c *Client) RunTurn(ctx context.Context, threadID, prompt string, async bool) (harness.TurnResult, error) {
	var out harness.TurnResult
	err := c.doJSON(ctx, http.MethodPost, "/v1/threads/"+threadID+"/turns", map[string]any{
		"prompt": prompt,
		"async":  async,
	}, &out)
	return out, err
}

func (c *Client) Steer(ctx context.Context, threadID, text string) error {
	return c.doJSON(ctx, http.MethodPost, "/v1/threads/"+threadID+"/steer", map[string]any{"text": text}, nil)
}

func (c *Client) Cancel(ctx context.Context, threadID string) error {
	return c.doJSON(ctx, http.MethodPost, "/v1/threads/"+threadID+"/cancel", nil, nil)
}

func (c *Client) Compact(ctx context.Context, threadID string) error {
	return c.doJSON(ctx, http.MethodPost, "/v1/threads/"+threadID+"/compact", nil, nil)
}

func (c *Client) ListMessages(ctx context.Context, threadID string) ([]schema.Message, error) {
	var resp struct {
		Messages []schema.Message `json:"messages"`
	}
	err := c.doJSON(ctx, http.MethodGet, "/v1/threads/"+threadID+"/messages", nil, &resp)
	return resp.Messages, err
}

// StreamEvents subscribes to `/v1/threads/{id}/events?after_seq=` and invokes
// onEvent for each received SessionEvent until ctx is cancelled or onEvent returns false.
func (c *Client) StreamEvents(ctx context.Context, threadID string, afterSeq int64, onEvent func(schema.SessionEvent) bool) error {
	endpoint := fmt.Sprintf("%s/v1/threads/%s/events?after_seq=%d", c.baseURL, threadID, afterSeq)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "text/event-stream")
	if c.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.authToken)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("sse status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		var ev schema.SessionEvent
		if json.Unmarshal([]byte(data), &ev) != nil {
			continue
		}
		if !onEvent(ev) {
			return nil
		}
	}
	return sc.Err()
}
