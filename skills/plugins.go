package skills

import (
	"sort"
	"strings"
	"sync"
)

type PluginStatus string

const (
	PluginConnected    PluginStatus = "connected"
	PluginDisconnected PluginStatus = "disconnected"
)

type Plugin struct {
	ID                 string       `json:"id"`
	Name               string       `json:"name"`
	Description        string       `json:"description"`
	RepoURL            string       `json:"repo_url,omitempty"`
	CommitSHA          string       `json:"commit_sha,omitempty"`
	Execution          string       `json:"execution,omitempty"` // cloud | sandbox | device
	Privacy            string       `json:"privacy,omitempty"`   // org | personal
	Enabled            bool         `json:"enabled"`
	Status             PluginStatus `json:"status"`
	ConnectURL         string       `json:"connect_url,omitempty"`
	HasScriptsOrAssets bool         `json:"has_scripts_or_assets,omitempty"`
	SandboxPath        string       `json:"sandbox_path,omitempty"` // e.g. /jaz/plugins/<id>
	Tools              []string     `json:"tools,omitempty"`
	Skills             []string     `json:"skills,omitempty"`
}

type PluginRegistry struct {
	mu      sync.RWMutex
	plugins map[string]Plugin
}

func NewPluginRegistry() *PluginRegistry {
	return &PluginRegistry{
		plugins: make(map[string]Plugin),
	}
}

func (r *PluginRegistry) Register(p Plugin) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if p.Status == "" {
		p.Status = PluginConnected
	}
	if p.ConnectURL == "" && p.Status == PluginDisconnected {
		p.ConnectURL = "jaz://plugin/" + p.ID + "/connect"
	}
	r.plugins[p.ID] = p
}

func (r *PluginRegistry) SetEnabled(id string, enabled bool) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	p, ok := r.plugins[id]
	if !ok {
		return false
	}
	p.Enabled = enabled
	r.plugins[id] = p
	return true
}

func (r *PluginRegistry) Search(query string) []Plugin {
	r.mu.RLock()
	defer r.mu.RUnlock()
	terms := strings.Fields(strings.ToLower(strings.TrimSpace(query)))
	var out []Plugin
	for _, p := range r.plugins {
		if len(terms) == 0 {
			out = append(out, p)
			continue
		}
		hay := strings.ToLower(p.ID + " " + p.Name + " " + p.Description + " " + strings.Join(p.Tools, " ") + " " + strings.Join(p.Skills, " "))
		matched := true
		for _, t := range terms {
			if !strings.Contains(hay, t) {
				matched = false
				break
			}
		}
		if matched {
			out = append(out, p)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}
