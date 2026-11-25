package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
)

type ToolHandler func(ctx context.Context, args json.RawMessage) (CallToolResult, error)

type RegisteredSkill struct {
	Entry SkillEntry
	Files map[string]string // URI -> UTF-8 content
}

// Server is an embeddable MCP 2025-11-25 + SEP-2640 Skills over MCP HTTP server.
type Server struct {
	info      Implementation
	mu        sync.RWMutex
	tools     map[string]Tool
	handlers  map[string]ToolHandler
	resources map[string]ResourceContent
	skills    map[string]RegisteredSkill
}

func NewServer(name, version string) *Server {
	return &Server{
		info:      Implementation{Name: name, Version: version},
		tools:     make(map[string]Tool),
		handlers:  make(map[string]ToolHandler),
		resources: make(map[string]ResourceContent),
		skills:    make(map[string]RegisteredSkill),
	}
}

func (s *Server) RegisterTool(tool Tool, handler ToolHandler) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tools[tool.Name] = tool
	s.handlers[tool.Name] = handler
}

// RegisterSkill publishes a SEP-2640 skill directory under skill://<name>/...
// with automatic SHA-256 digests on every resource.
func (s *Server) RegisterSkill(name, description string, files map[string]string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	baseURI := SkillURISchemePrefix + strings.Trim(strings.TrimSpace(name), "/")
	entry := SkillEntry{
		URI:          baseURI,
		Name:         name,
		Description:  description,
		MainResource: baseURI + "/SKILL.md",
	}
	uriFiles := make(map[string]string, len(files))
	keys := make([]string, 0, len(files))
	for k := range files {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, rel := range keys {
		cleanRel := strings.TrimPrefix(rel, "/")
		uri := baseURI + "/" + cleanRel
		content := files[rel]
		digest := ComputeSHA256Hex([]byte(content))
		uriFiles[uri] = content
		s.resources[uri] = ResourceContent{
			URI:      uri,
			MimeType: "text/markdown",
			Text:     content,
			SHA256:   digest,
		}
		entry.Resources = append(entry.Resources, Resource{
			URI:      uri,
			Name:     cleanRel,
			MimeType: "text/markdown",
			Size:     int64(len(content)),
			SHA256:   digest,
		})
		if cleanRel == "SKILL.md" {
			entry.SHA256 = digest
		}
	}
	s.skills[baseURI] = RegisteredSkill{
		Entry: entry,
		Files: uriFiles,
	}
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var msg JSONRPCMessage
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}
	if len(msg.ID) == 0 {
		w.WriteHeader(http.StatusAccepted)
		return
	}

	resp := s.handleMessage(r.Context(), msg)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleMessage(ctx context.Context, msg JSONRPCMessage) JSONRPCMessage {
	reply := func(res any, rpcErr *JSONRPCError) JSONRPCMessage {
		out := JSONRPCMessage{
			JSONRPC: "2.0",
			ID:      msg.ID,
			Error:   rpcErr,
		}
		if res != nil && rpcErr == nil {
			raw, _ := json.Marshal(res)
			out.Result = raw
		}
		return out
	}

	switch msg.Method {
	case "initialize":
		caps := Capabilities{
			Tools:     map[string]any{},
			Resources: map[string]any{},
			Extensions: map[string]map[string]any{
				SkillsExtensionID: {
					"directoryRead": true,
				},
			},
		}
		return reply(InitializeResult{
			ProtocolVersion: ProtocolVersion,
			Capabilities:    caps,
			ServerInfo:      s.info,
		}, nil)

	case "tools/list":
		s.mu.RLock()
		list := make([]Tool, 0, len(s.tools))
		for _, t := range s.tools {
			list = append(list, t)
		}
		s.mu.RUnlock()
		sort.Slice(list, func(i, j int) bool { return list[i].Name < list[j].Name })
		return reply(map[string]any{"tools": list}, nil)

	case "tools/call":
		var params struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			return reply(nil, &JSONRPCError{Code: -32602, Message: err.Error()})
		}
		s.mu.RLock()
		h, ok := s.handlers[params.Name]
		s.mu.RUnlock()
		if !ok {
			return reply(nil, &JSONRPCError{Code: -32601, Message: "unknown tool: " + params.Name})
		}
		res, err := h(ctx, params.Arguments)
		if err != nil {
			return reply(CallToolResult{
				Content: []ToolContent{{Type: "text", Text: err.Error()}},
				IsError: true,
			}, nil)
		}
		return reply(res, nil)

	case "resources/list":
		s.mu.RLock()
		var list []Resource
		for _, sk := range s.skills {
			list = append(list, sk.Entry.Resources...)
		}
		s.mu.RUnlock()
		sort.Slice(list, func(i, j int) bool { return list[i].URI < list[j].URI })
		return reply(map[string]any{"resources": list}, nil)

	case "resources/read":
		var params struct {
			URI string `json:"uri"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			return reply(nil, &JSONRPCError{Code: -32602, Message: err.Error()})
		}
		s.mu.RLock()
		rc, ok := s.resources[params.URI]
		s.mu.RUnlock()
		if !ok {
			return reply(nil, &JSONRPCError{Code: -32002, Message: "resource not found: " + params.URI})
		}
		return reply(map[string]any{"contents": []ResourceContent{rc}}, nil)

	case "resources/directory/read":
		var params struct {
			URI string `json:"uri"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			return reply(nil, &JSONRPCError{Code: -32602, Message: err.Error()})
		}
		prefix := strings.TrimRight(params.URI, "/") + "/"
		s.mu.RLock()
		var entries []ResourceContent
		for uri, rc := range s.resources {
			if strings.HasPrefix(uri, prefix) {
				entries = append(entries, rc)
			}
		}
		s.mu.RUnlock()
		sort.Slice(entries, func(i, j int) bool { return entries[i].URI < entries[j].URI })
		return reply(DirectoryReadResult{URI: params.URI, Entries: entries}, nil)

	case "skills/list":
		s.mu.RLock()
		list := make([]SkillEntry, 0, len(s.skills))
		for _, sk := range s.skills {
			list = append(list, sk.Entry)
		}
		s.mu.RUnlock()
		sort.Slice(list, func(i, j int) bool { return list[i].URI < list[j].URI })
		return reply(map[string]any{"skills": list}, nil)

	case "skills/get":
		var params struct {
			URI string `json:"uri"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			return reply(nil, &JSONRPCError{Code: -32602, Message: err.Error()})
		}
		s.mu.RLock()
		sk, ok := s.skills[strings.TrimRight(params.URI, "/")]
		s.mu.RUnlock()
		if !ok {
			return reply(nil, &JSONRPCError{Code: -32002, Message: fmt.Sprintf("skill not found: %s", params.URI)})
		}
		return reply(map[string]any{"skill": sk.Entry}, nil)

	default:
		return reply(nil, &JSONRPCError{Code: -32601, Message: "method not found: " + msg.Method})
	}
}
