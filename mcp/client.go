package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Transport interface {
	Call(ctx context.Context, method string, params any, out any) error
	Notify(ctx context.Context, method string, params any) error
	Close() error
}

// Client is a full MCP client supporting Streamable HTTP, SSE responses, stdio
// subprocesses, and SEP-2640 Skills over MCP (io.modelcontextprotocol/skills).
type Client struct {
	serverName   string
	transport    Transport
	initMu       sync.Mutex
	initialized  bool
	serverCaps   Capabilities
	serverInfo   Implementation
	protoVersion string
}

func NewHTTPClient(serverName, endpoint string, headers map[string]string, hc *http.Client) *Client {
	return &Client{
		serverName: serverName,
		transport:  newHTTPTransport(endpoint, headers, hc),
	}
}

func NewStdioClient(serverName, command string, args []string, env []string) (*Client, error) {
	tr, err := startStdioTransport(command, args, env)
	if err != nil {
		return nil, err
	}
	return &Client{
		serverName: serverName,
		transport:  tr,
	}, nil
}

func (c *Client) ServerName() string { return c.serverName }

func (c *Client) Close() error {
	if c.transport == nil {
		return nil
	}
	return c.transport.Close()
}

func (c *Client) Initialize(ctx context.Context) (InitializeResult, error) {
	c.initMu.Lock()
	defer c.initMu.Unlock()
	if c.initialized {
		return InitializeResult{
			ProtocolVersion: c.protoVersion,
			Capabilities:    c.serverCaps,
			ServerInfo:      c.serverInfo,
		}, nil
	}
	params := map[string]any{
		"protocolVersion": ProtocolVersion,
		"capabilities": map[string]any{
			"resources": map[string]any{},
			"extensions": map[string]any{
				SkillsExtensionID: map[string]any{
					"directoryRead": true,
				},
			},
		},
		"clientInfo": Implementation{
			Name:    "jaz-harness",
			Version: "0.1.0",
		},
	}
	var res InitializeResult
	if err := c.transport.Call(ctx, "initialize", params, &res); err != nil {
		return InitializeResult{}, err
	}
	_ = c.transport.Notify(ctx, "notifications/initialized", map[string]any{})
	c.initialized = true
	c.serverCaps = res.Capabilities
	c.serverInfo = res.ServerInfo
	c.protoVersion = res.ProtocolVersion
	return res, nil
}

func (c *Client) ensureInitialized(ctx context.Context) error {
	c.initMu.Lock()
	ok := c.initialized
	c.initMu.Unlock()
	if ok {
		return nil
	}
	_, err := c.Initialize(ctx)
	return err
}

func (c *Client) SupportsSkillsExtension() bool {
	c.initMu.Lock()
	defer c.initMu.Unlock()
	if c.serverCaps.Extensions == nil {
		return false
	}
	_, ok := c.serverCaps.Extensions[SkillsExtensionID]
	return ok
}

func (c *Client) ListTools(ctx context.Context) ([]Tool, error) {
	if err := c.ensureInitialized(ctx); err != nil {
		return nil, err
	}
	var res struct {
		Tools []Tool `json:"tools"`
	}
	if err := c.transport.Call(ctx, "tools/list", map[string]any{}, &res); err != nil {
		return nil, err
	}
	return res.Tools, nil
}

func (c *Client) CallTool(ctx context.Context, name string, args json.RawMessage) (CallToolResult, error) {
	if err := c.ensureInitialized(ctx); err != nil {
		return CallToolResult{}, err
	}
	var parsedArgs any = map[string]any{}
	if len(bytes.TrimSpace(args)) > 0 {
		_ = json.Unmarshal(args, &parsedArgs)
	}
	var res CallToolResult
	err := c.transport.Call(ctx, "tools/call", map[string]any{
		"name":      name,
		"arguments": parsedArgs,
	}, &res)
	return res, err
}

func (c *Client) ListResources(ctx context.Context) ([]Resource, error) {
	if err := c.ensureInitialized(ctx); err != nil {
		return nil, err
	}
	var res struct {
		Resources []Resource `json:"resources"`
	}
	if err := c.transport.Call(ctx, "resources/list", map[string]any{}, &res); err != nil {
		return nil, err
	}
	return res.Resources, nil
}

func (c *Client) ReadResource(ctx context.Context, uri string) ([]ResourceContent, error) {
	if err := c.ensureInitialized(ctx); err != nil {
		return nil, err
	}
	var res struct {
		Contents []ResourceContent `json:"contents"`
	}
	if err := c.transport.Call(ctx, "resources/read", map[string]any{"uri": uri}, &res); err != nil {
		return nil, err
	}
	for _, rc := range res.Contents {
		if rc.SHA256 != "" && rc.Text != "" {
			got := ComputeSHA256Hex([]byte(rc.Text))
			if !strings.EqualFold(got, rc.SHA256) {
				return nil, fmt.Errorf("mcp resource %s sha256 mismatch: got %s want %s", uri, got, rc.SHA256)
			}
		}
	}
	return res.Contents, nil
}

// ReadResourceDirectory implements SEP-2640 resources/directory/read.
func (c *Client) ReadResourceDirectory(ctx context.Context, uri string) (DirectoryReadResult, error) {
	if err := c.ensureInitialized(ctx); err != nil {
		return DirectoryReadResult{}, err
	}
	var res DirectoryReadResult
	if err := c.transport.Call(ctx, "resources/directory/read", map[string]any{"uri": uri}, &res); err != nil {
		return DirectoryReadResult{}, err
	}
	for _, entry := range res.Entries {
		if entry.SHA256 != "" && entry.Text != "" {
			got := ComputeSHA256Hex([]byte(entry.Text))
			if !strings.EqualFold(got, entry.SHA256) {
				return DirectoryReadResult{}, fmt.Errorf("mcp skill entry %s sha256 mismatch", entry.URI)
			}
		}
	}
	return res, nil
}

// ListSkills discovers SEP-2640 skills via `skills/list`, falling back to
// filtering `skill://` entries from `resources/list` for servers that only
// expose skill resources.
func (c *Client) ListSkills(ctx context.Context) ([]SkillEntry, error) {
	if err := c.ensureInitialized(ctx); err != nil {
		return nil, err
	}
	if c.SupportsSkillsExtension() {
		var res struct {
			Skills []SkillEntry `json:"skills"`
		}
		if err := c.transport.Call(ctx, "skills/list", map[string]any{}, &res); err == nil {
			return res.Skills, nil
		}
	}
	resources, err := c.ListResources(ctx)
	if err != nil {
		return nil, err
	}
	var skills []SkillEntry
	for _, r := range resources {
		if strings.HasPrefix(r.URI, SkillURISchemePrefix) && strings.HasSuffix(r.URI, "/SKILL.md") {
			skills = append(skills, SkillEntry{
				URI:          strings.TrimSuffix(r.URI, "/SKILL.md"),
				Name:         r.Name,
				Description:  r.Description,
				MainResource: r.URI,
				SHA256:       r.SHA256,
			})
		}
	}
	return skills, nil
}

// GetSkill fetches a single SEP-2640 skill descriptor via `skills/get`.
func (c *Client) GetSkill(ctx context.Context, uri string) (SkillEntry, error) {
	if err := c.ensureInitialized(ctx); err != nil {
		return SkillEntry{}, err
	}
	var res struct {
		Skill SkillEntry `json:"skill"`
	}
	if err := c.transport.Call(ctx, "skills/get", map[string]any{"uri": uri}, &res); err != nil {
		return SkillEntry{}, err
	}
	return res.Skill, nil
}

func (c *Client) ListPrompts(ctx context.Context) ([]Prompt, error) {
	if err := c.ensureInitialized(ctx); err != nil {
		return nil, err
	}
	var res struct {
		Prompts []Prompt `json:"prompts"`
	}
	if err := c.transport.Call(ctx, "prompts/list", map[string]any{}, &res); err != nil {
		return nil, err
	}
	return res.Prompts, nil
}

const maxMCPLineBytes = 64 << 20

func readBoundedLine(br *bufio.Reader, maxBytes int) ([]byte, error) {
	var acc bytes.Buffer
	for {
		frag, isPrefix, err := br.ReadLine()
		if len(frag) > 0 {
			if acc.Len()+len(frag) > maxBytes {
				return nil, fmt.Errorf("mcp stream line exceeds maximum length (%d bytes)", maxBytes)
			}
			acc.Write(frag)
		}
		if isPrefix {
			if err != nil {
				return nil, err
			}
			continue
		}
		if err != nil {
			if errors.Is(err, io.EOF) && acc.Len() > 0 {
				return acc.Bytes(), io.EOF
			}
			return nil, err
		}
		return acc.Bytes(), nil
	}
}

type httpTransport struct {
	endpoint   string
	headers    map[string]string
	httpClient *http.Client
	nextID     atomic.Int64
	mu         sync.Mutex
	sessionID  string
}

func newHTTPTransport(endpoint string, headers map[string]string, hc *http.Client) *httpTransport {
	if hc == nil {
		hc = &http.Client{Timeout: 60 * time.Second}
	}
	return &httpTransport{
		endpoint:   endpoint,
		headers:    headers,
		httpClient: hc,
	}
}

func (t *httpTransport) Close() error { return nil }

func (t *httpTransport) Notify(ctx context.Context, method string, params any) error {
	paramsRaw, err := json.Marshal(params)
	if err != nil {
		return err
	}
	body, err := json.Marshal(JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  method,
		Params:  paramsRaw,
	})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	t.applyHeaders(req)
	resp, err := t.httpClient.Do(req)
	if err != nil {
		return err
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	return nil
}

func (t *httpTransport) Call(ctx context.Context, method string, params any, out any) error {
	id := t.nextID.Add(1)
	idRaw, _ := json.Marshal(id)
	var paramsRaw json.RawMessage
	if params != nil {
		var err error
		paramsRaw, err = json.Marshal(params)
		if err != nil {
			return err
		}
	}
	body, err := json.Marshal(JSONRPCMessage{
		JSONRPC: "2.0",
		ID:      idRaw,
		Method:  method,
		Params:  paramsRaw,
	})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	t.applyHeaders(req)
	resp, err := t.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if sid := strings.TrimSpace(resp.Header.Get("Mcp-Session-Id")); sid != "" {
		t.mu.Lock()
		t.sessionID = sid
		t.mu.Unlock()
	}
	if resp.StatusCode >= 300 {
		rawErr, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		return fmt.Errorf("mcp http status %d: %s", resp.StatusCode, strings.TrimSpace(string(rawErr)))
	}

	var rpcResp JSONRPCMessage
	ct := strings.ToLower(resp.Header.Get("Content-Type"))
	if strings.Contains(ct, "text/event-stream") {
		var found bool
		br := bufio.NewReaderSize(resp.Body, 64*1024)
		for {
			rawBytes, rerr := readBoundedLine(br, maxMCPLineBytes)
			line := strings.TrimSpace(string(rawBytes))
			if strings.HasPrefix(line, "data:") {
				data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
				var msg JSONRPCMessage
				if json.Unmarshal([]byte(data), &msg) == nil && string(msg.ID) == string(idRaw) {
					rpcResp = msg
					found = true
					break
				}
			}
			if rerr != nil {
				if !errors.Is(rerr, io.EOF) {
					return rerr
				}
				break
			}
		}
		if !found {
			return errors.New("mcp sse response ended without matching json-rpc result")
		}
	} else {
		if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<20)).Decode(&rpcResp); err != nil {
			return err
		}
	}
	if rpcResp.Error != nil {
		return rpcResp.Error
	}
	if out != nil && len(rpcResp.Result) > 0 {
		return json.Unmarshal(rpcResp.Result, out)
	}
	return nil
}

func (t *httpTransport) applyHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("MCP-Protocol-Version", ProtocolVersion)
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}
	t.mu.Lock()
	sid := t.sessionID
	t.mu.Unlock()
	if sid != "" {
		req.Header.Set("Mcp-Session-Id", sid)
	}
}

// Stdio subprocess JSON-RPC transport.
type stdioTransport struct {
	cmd       *exec.Cmd
	stdin     io.WriteCloser
	writeMu   sync.Mutex
	nextID    atomic.Int64
	pendingMu sync.Mutex
	pending   map[string]chan JSONRPCMessage
	done      chan struct{}
}

func startStdioTransport(command string, args []string, env []string) (*stdioTransport, error) {
	cmd := exec.Command(command, args...)
	if len(env) > 0 {
		cmd.Env = env
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	tr := &stdioTransport{
		cmd:     cmd,
		stdin:   stdin,
		pending: make(map[string]chan JSONRPCMessage),
		done:    make(chan struct{}),
	}
	go tr.readLoop(stdout)
	return tr, nil
}

func (t *stdioTransport) readLoop(r io.Reader) {
	defer close(t.done)
	br := bufio.NewReaderSize(r, 256*1024)
	for {
		line, err := readBoundedLine(br, maxMCPLineBytes)
		if len(line) > 0 {
			var msg JSONRPCMessage
			if json.Unmarshal(bytes.TrimSpace(line), &msg) == nil && len(msg.ID) > 0 {
				key := string(msg.ID)
				t.pendingMu.Lock()
				ch, ok := t.pending[key]
				if ok {
					delete(t.pending, key)
				}
				t.pendingMu.Unlock()
				if ok {
					ch <- msg
				}
			}
		}
		if err != nil {
			t.pendingMu.Lock()
			for k, ch := range t.pending {
				close(ch)
				delete(t.pending, k)
			}
			t.pendingMu.Unlock()
			return
		}
	}
}

func (t *stdioTransport) sendRaw(msg JSONRPCMessage) error {
	raw, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	t.writeMu.Lock()
	defer t.writeMu.Unlock()
	_, err = t.stdin.Write(raw)
	return err
}

func (t *stdioTransport) Notify(_ context.Context, method string, params any) error {
	paramsRaw, err := json.Marshal(params)
	if err != nil {
		return err
	}
	return t.sendRaw(JSONRPCMessage{
		JSONRPC: "2.0",
		Method:  method,
		Params:  paramsRaw,
	})
}

func (t *stdioTransport) Call(ctx context.Context, method string, params any, out any) error {
	id := t.nextID.Add(1)
	idRaw, _ := json.Marshal(id)
	key := string(idRaw)
	ch := make(chan JSONRPCMessage, 1)

	t.pendingMu.Lock()
	t.pending[key] = ch
	t.pendingMu.Unlock()

	var paramsRaw json.RawMessage
	if params != nil {
		var err error
		paramsRaw, err = json.Marshal(params)
		if err != nil {
			t.pendingMu.Lock()
			delete(t.pending, key)
			t.pendingMu.Unlock()
			return err
		}
	}
	if err := t.sendRaw(JSONRPCMessage{
		JSONRPC: "2.0",
		ID:      idRaw,
		Method:  method,
		Params:  paramsRaw,
	}); err != nil {
		t.pendingMu.Lock()
		delete(t.pending, key)
		t.pendingMu.Unlock()
		return err
	}

	select {
	case <-ctx.Done():
		t.pendingMu.Lock()
		delete(t.pending, key)
		t.pendingMu.Unlock()
		return ctx.Err()
	case resp, ok := <-ch:
		if !ok {
			return errors.New("mcp stdio server closed before response")
		}
		if resp.Error != nil {
			return resp.Error
		}
		if out != nil && len(resp.Result) > 0 {
			return json.Unmarshal(resp.Result, out)
		}
		return nil
	}
}

func (t *stdioTransport) Close() error {
	_ = t.stdin.Close()
	if t.cmd != nil && t.cmd.Process != nil {
		_ = t.cmd.Process.Kill()
		_ = t.cmd.Wait()
	}
	<-t.done
	return nil
}
