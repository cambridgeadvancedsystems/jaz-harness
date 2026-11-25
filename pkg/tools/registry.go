package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/mcp"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/skills"
)

type Handler func(ctx context.Context, args json.RawMessage) (string, error)

type entry struct {
	def     schema.ToolDef
	handler Handler
}

// Registry manages core (always-loaded) tools and deferred tools, providing:
//  1. Universal static gateway (`search_tools`, `call_tool`, `plugins_search`, `skills_read`)
//     that works across every LLM provider without mutating tool schemas (0 KV cache bust).
//  2. Native Anthropic `defer_loading: true` + `tool_search_tool_bm25` projection when enabled.
type Registry struct {
	mu      sync.RWMutex
	entries map[string]entry
	skills  *skills.Catalog
	plugins *skills.PluginRegistry
}

func NewRegistry(skillCat *skills.Catalog, pluginReg *skills.PluginRegistry) *Registry {
	r := &Registry{
		entries: make(map[string]entry),
		skills:  skillCat,
		plugins: pluginReg,
	}
	r.registerBuiltins()
	return r
}

func (r *Registry) Register(def schema.ToolDef, h Handler) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(def.InputSchema) == 0 {
		def.InputSchema = json.RawMessage(`{"type":"object","properties":{}}`)
	}
	r.entries[def.Name] = entry{def: def, handler: h}
}

// RegisterMCPServer imports all tools from an MCP client (marking them deferred
// by default so 1,000+ org MCP tools never bloat prompt tokens) and also syncs
// any SEP-2640 MCP skills exposed by the same server!
func (r *Registry) RegisterMCPServer(ctx context.Context, client *mcp.Client, deferLoading bool) (toolCount, skillCount int, err error) {
	mcpTools, err := client.ListTools(ctx)
	if err != nil {
		return 0, 0, err
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
			DeferLoading: deferLoading,
			AlwaysLoad:   !deferLoading,
			ServerID:     client.ServerName(),
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

// ActiveToolDefs returns the tool definitions to send on a provider request:
//   - If nativeDeferred is false (universal mode): returns always-loaded tools
//     plus the static gateway (`search_tools`, `call_tool`, `plugins_search`, `skills_read`).
//   - If nativeDeferred is true (Anthropic native tool search): passes all tools
//     with `DeferLoading: true` preserved so Anthropic attaches `tool_search_tool_bm25`.
func (r *Registry) ActiveToolDefs(nativeDeferred bool) []schema.ToolDef {
	r.mu.RLock()
	defer r.mu.RUnlock()

	hasDeferred := false
	for _, e := range r.entries {
		if e.def.DeferLoading && !e.def.AlwaysLoad {
			hasDeferred = true
			break
		}
	}

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
		if isGateway && !hasDeferred {
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
				// Prefix boost for partial identifiers like "mail" -> "gmail_search".
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

func tokenize(s string) []string {
	s = strings.ToLower(s)
	replacer := strings.NewReplacer("_", " ", "-", " ", "/", " ", ".", " ", ":", " ", ",", " ", "(", " ", ")", " ")
	return strings.Fields(replacer.Replace(s))
}
