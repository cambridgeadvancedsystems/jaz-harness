package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/cambridgeadvancedsystems/jaz-harness/mcp"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/cambridgeadvancedsystems/jaz-harness/skills"
)

const DefaultMaxMCPToolOutputBytes = 256 * 1024

type Handler func(ctx context.Context, args json.RawMessage) (string, error)

type entry struct {
	def     schema.ToolDef
	handler Handler
}

// Registry manages a small, deterministic, immutable eager core alongside a
// strictly deferred tier for external MCP and plugin tools:
//  1. Eager core: static discovery/gateway tools (`search_tools`, `call_tool`,
//     `plugins_search`, `skills_read`) and core workspace tools. Never mutates
//     mid-thread when deferred MCP/plugin tools are registered, searched, or called.
//  2. Deferred tier: all external MCP and plugin tools (`ServerID != ""` or
//     `PluginID != ""`) are strictly deferred — projected via Anthropic native
//     `defer_loading: true` + `tool_search_tool_bm25_20251119` on Anthropic, and
//     via the static `search_tools` / `call_tool` gateway everywhere else.
type Registry struct {
	mu             sync.RWMutex
	entries        map[string]entry
	skills         *skills.Catalog
	plugins        *skills.PluginRegistry
	maxMCPOutBytes int
}

func NewRegistry(skillCat *skills.Catalog, pluginReg *skills.PluginRegistry) *Registry {
	r := &Registry{
		entries:        make(map[string]entry),
		skills:         skillCat,
		plugins:        pluginReg,
		maxMCPOutBytes: DefaultMaxMCPToolOutputBytes,
	}
	r.registerBuiltins()
	return r
}

func (r *Registry) SetMaxMCPToolOutputBytes(limit int) {
	r.mu.Lock()
	if limit > 0 {
		r.maxMCPOutBytes = limit
	}
	r.mu.Unlock()
}

// Register adds a tool definition and handler. External MCP or plugin tools
// (`ServerID != ""` or `PluginID != ""`) are strictly enforced as deferred so
// the eager core schema stays small and immutable across turns.
func (r *Registry) Register(def schema.ToolDef, h Handler) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(def.InputSchema) == 0 {
		def.InputSchema = json.RawMessage(`{"type":"object","properties":{}}`)
	}
	if def.ServerID != "" || def.PluginID != "" {
		def.DeferLoading = true
		def.AlwaysLoad = false
	}
	r.entries[def.Name] = entry{def: def, handler: h}
}

// RegisterPluginTool registers an external plugin tool in the strictly deferred tier.
func (r *Registry) RegisterPluginTool(pluginID string, def schema.ToolDef, h Handler) {
	def.PluginID = pluginID
	def.DeferLoading = true
	def.AlwaysLoad = false
	r.Register(def, h)
}

// RegisterMCPServer imports all tools from an MCP client into the strictly
// deferred tier (external MCP tools are never loaded eagerly) and syncs any
// SEP-2640 MCP skills exposed by the same server.
func (r *Registry) RegisterMCPServer(ctx context.Context, client *mcp.Client) (toolCount, skillCount int, err error) {
	mcpTools, err := client.ListTools(ctx)
	if err != nil {
		return 0, 0, err
	}
	serverID := client.ServerName()
	if serverID == "" {
		serverID = "mcp"
	}
	for _, mt := range mcpTools {
		toolName := mt.Name
		scopedName := toolName
		if client.ServerName() != "" && !strings.HasPrefix(toolName, client.ServerName()+"_") {
			scopedName = client.ServerName() + "_" + toolName
		}
		mcpClient := client
		remoteName := mt.Name
		r.Register(schema.ToolDef{
			Name:         scopedName,
			Description:  mt.Description,
			InputSchema:  mt.InputSchema,
			DeferLoading: true,
			AlwaysLoad:   false,
			ServerID:     serverID,
		}, func(callCtx context.Context, args json.RawMessage) (string, error) {
			res, cerr := mcpClient.CallTool(callCtx, remoteName, args)
			if cerr != nil {
				return "", cerr
			}
			var parts []string
			for _, item := range res.Content {
				if item.Text != "" {
					parts = append(parts, item.Text)
				}
			}
			out := strings.Join(parts, "\n")
			if out == "" && len(res.StructuredContent) > 0 {
				out = string(res.StructuredContent)
			}
			r.mu.RLock()
			capBytes := r.maxMCPOutBytes
			r.mu.RUnlock()
			out = TruncateUTF8(out, capBytes)
			if res.IsError {
				return "", fmt.Errorf("%s", out)
			}
			return out, nil
		})
		toolCount++
	}

	if r.skills != nil {
		if n, serr := r.skills.SyncFromMCP(ctx, client); serr == nil {
			skillCount = n
		}
	}
	return toolCount, skillCount, nil
}

// ActiveToolDefs returns the deterministic tool definitions for a provider request:
//   - Universal mode (`nativeDeferred == false`): returns ONLY the immutable
//     eager core (including static `search_tools`, `call_tool`, `plugins_search`,
//     `skills_read`, and core workspace tools). Registering, searching, or calling
//     deferred MCP/plugin tools produces zero change to this slice.
//   - Anthropic native mode (`nativeDeferred == true`): omits `search_tools` and
//     `call_tool` and passes eager core plus deferred tools with `DeferLoading: true`
//     so Anthropic attaches `tool_search_tool_bm25_20251119`.
func (r *Registry) ActiveToolDefs(nativeDeferred bool) []schema.ToolDef {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var out []schema.ToolDef
	for _, e := range r.entries {
		isGateway := e.def.Name == "search_tools" || e.def.Name == "call_tool"
		if nativeDeferred {
			if isGateway {
				continue
			}
			out = append(out, e.def)
			continue
		}
		if e.def.DeferLoading && !e.def.AlwaysLoad {
			continue
		}
		out = append(out, e.def)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func (r *Registry) AllToolDefs() []schema.ToolDef {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]schema.ToolDef, 0, len(r.entries))
	for _, e := range r.entries {
		out = append(out, e.def)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func (r *Registry) Execute(ctx context.Context, name string, args json.RawMessage) (string, error) {
	r.mu.RLock()
	e, ok := r.entries[name]
	r.mu.RUnlock()
	if !ok {
		return "", fmt.Errorf("unknown tool %q", name)
	}
	return e.handler(ctx, args)
}

// SearchBM25 ranks deferred tools using Okapi BM25 over tool name, server,
// plugin, and description tokens.
func (r *Registry) SearchBM25(query string, limit int) []schema.ToolDef {
	if limit <= 0 {
		limit = 10
	}
	r.mu.RLock()
	docs := make([]schema.ToolDef, 0, len(r.entries))
	for _, e := range r.entries {
		if e.def.DeferLoading && !e.def.AlwaysLoad {
			docs = append(docs, e.def)
		}
	}
	r.mu.RUnlock()

	qTerms := tokenize(query)
	if len(qTerms) == 0 {
		sort.Slice(docs, func(i, j int) bool { return docs[i].Name < docs[j].Name })
		if len(docs) > limit {
			return docs[:limit]
		}
		return docs
	}

	N := float64(len(docs))
	if N == 0 {
		return nil
	}
	docTokens := make([][]string, len(docs))
	df := make(map[string]int)
	totalLen := 0
	for i, d := range docs {
		toks := tokenize(d.Name + " " + d.Name + " " + d.ServerID + " " + d.PluginID + " " + d.Description)
		docTokens[i] = toks
		totalLen += len(toks)
		seen := map[string]bool{}
		for _, tok := range toks {
			if !seen[tok] {
				seen[tok] = true
				df[tok]++
			}
		}
	}
	avgDL := float64(totalLen) / N
	if avgDL == 0 {
		avgDL = 1
	}

	type scored struct {
		def   schema.ToolDef
		score float64
	}
	const (
		k1 = 1.5
		b  = 0.75
	)
	var hits []scored
	for i, d := range docs {
		tf := map[string]int{}
		for _, tok := range docTokens[i] {
			tf[tok]++
		}
		dl := float64(len(docTokens[i]))
		score := 0.0
		for _, qt := range qTerms {
			freq := float64(tf[qt])
			if freq == 0 {
				for tok, c := range tf {
					if strings.Contains(tok, qt) {
						freq += float64(c) * 0.6
					}
				}
			}
			if freq == 0 {
				continue
			}
			n := float64(df[qt])
			idf := math.Log((N-n+0.5)/(n+0.5) + 1.0)
			score += idf * (freq * (k1 + 1)) / (freq + k1*(1-b+b*(dl/avgDL)))
		}
		if score > 0 {
			hits = append(hits, scored{def: d, score: score})
		}
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].score != hits[j].score {
			return hits[i].score > hits[j].score
		}
		return hits[i].def.Name < hits[j].def.Name
	})
	if len(hits) > limit {
		hits = hits[:limit]
	}
	out := make([]schema.ToolDef, len(hits))
	for i, h := range hits {
		out[i] = h.def
	}
	return out
}

func (r *Registry) registerBuiltins() {
	r.Register(schema.ToolDef{
		Name:        "search_tools",
		Description: "Search deferred tools by keywords and return exact tool names, descriptions, and JSON input schemas.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":20}},"required":["query"]}`),
		AlwaysLoad:  true,
		Category:    "core",
	}, func(_ context.Context, args json.RawMessage) (string, error) {
		var p struct {
			Query string `json:"query"`
			Limit int    `json:"limit"`
		}
		_ = json.Unmarshal(args, &p)
		hits := r.SearchBM25(p.Query, p.Limit)
		raw, _ := json.Marshal(map[string]any{"tools": hits})
		return string(raw), nil
	})

	r.Register(schema.ToolDef{
		Name:        "call_tool",
		Description: "Invoke a deferred tool discovered via search_tools with its exact name and arguments object.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"},"arguments":{"type":"object","additionalProperties":true}},"required":["name","arguments"]}`),
		AlwaysLoad:  true,
		Category:    "core",
	}, func(ctx context.Context, args json.RawMessage) (string, error) {
		var p struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(args, &p); err != nil {
			return "", err
		}
		if p.Name == "" || p.Name == "call_tool" {
			return "", fmt.Errorf("invalid target tool name %q", p.Name)
		}
		return r.Execute(ctx, p.Name, p.Arguments)
	})

	if r.skills != nil {
		r.Register(schema.ToolDef{
			Name:        "skills_read",
			Description: "Read a skill's SKILL.md instructions or reference document by skill name or skill:// URI (supports local, cloud snapshot, and MCP SEP-2640 skills).",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"target":{"type":"string","description":"Skill name or skill:// URI"}},"required":["target"]}`),
			AlwaysLoad:  true,
			Category:    "core",
		}, func(ctx context.Context, args json.RawMessage) (string, error) {
			var p struct {
				Target string `json:"target"`
			}
			if err := json.Unmarshal(args, &p); err != nil {
				return "", err
			}
			return r.skills.Read(ctx, p.Target)
		})
	}

	if r.plugins != nil {
		r.Register(schema.ToolDef{
			Name:        "plugins_search",
			Description: "Search available plugins, their connection status, and connect links.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
			AlwaysLoad:  true,
			Category:    "core",
		}, func(_ context.Context, args json.RawMessage) (string, error) {
			var p struct {
				Query string `json:"query"`
			}
			_ = json.Unmarshal(args, &p)
			hits := r.plugins.Search(p.Query)
			raw, _ := json.Marshal(map[string]any{"plugins": hits})
			return string(raw), nil
		})
	}
}

// TruncateUTF8 trims s to at most limit bytes on valid UTF-8 rune boundaries
// (keeping 2/3 head and 1/3 tail) with an explicit truncation notice.
func TruncateUTF8(s string, limit int) string {
	if limit <= 0 || len(s) <= limit {
		return s
	}
	head := limit * 2 / 3
	tail := limit - head
	for head > 0 && !utf8.RuneStart(s[head]) {
		head--
	}
	tailStart := len(s) - tail
	for tailStart < len(s) && !utf8.RuneStart(s[tailStart]) {
		tailStart++
	}
	omitted := tailStart - head
	if omitted <= 0 {
		return s
	}
	return s[:head] + fmt.Sprintf("\n... [truncated %d of %d bytes — narrow query or range to inspect the rest] ...\n", omitted, len(s)) + s[tailStart:]
}

func tokenize(s string) []string {
	s = strings.ToLower(s)
	replacer := strings.NewReplacer("_", " ", "-", " ", "/", " ", ".", " ", ":", " ", ",", " ", "(", " ", ")", " ")
	return strings.Fields(replacer.Replace(s))
}
