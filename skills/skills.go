package skills

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/cambridgeadvancedsystems/jaz-harness/mcp"
)

// CloudSkillsSnapshot matches jaz-cloud's compiled in-memory snapshot schema
// so 0-pod hosted turns load skills directly from RAM without temp files.
type CloudSkillsSnapshot struct {
	Skills []CloudSnapshotSkill `json:"skills"`
}

type CloudSnapshotSkill struct {
	Package       string            `json:"package"`
	Name          string            `json:"name"`
	Description   string            `json:"description"`
	AliasRoot     string            `json:"alias_root"`
	MainResource  string            `json:"main_resource"`
	PromptVisible bool              `json:"prompt_visible"`
	Resources     map[string]string `json:"resources"`
}

type SkillSourceKind string

const (
	SourceLocal    SkillSourceKind = "local"
	SourceSnapshot SkillSourceKind = "snapshot"
	SourceMCP      SkillSourceKind = "mcp"
)

type Skill struct {
	Name          string            `json:"name"`
	Description   string            `json:"description"`
	URI           string            `json:"uri"`
	MainResource  string            `json:"main_resource"`
	Source        SkillSourceKind   `json:"source"`
	ServerName    string            `json:"server_name,omitempty"`
	PluginID      string            `json:"plugin_id,omitempty"`
	PromptVisible bool              `json:"prompt_visible"`
	Resources     map[string]string `json:"-"` // in-memory UTF-8 text resources
	MCPClient     *mcp.Client       `json:"-"`
}

// Catalog unifies local directory skills, Jaz Cloud in-memory snapshots, and
// remote SEP-2640 MCP skills under one progressive-disclosure interface.
type Catalog struct {
	mu     sync.RWMutex
	skills map[string]*Skill // keyed by normalized URI or name
}

func NewCatalog() *Catalog {
	return &Catalog{
		skills: make(map[string]*Skill),
	}
}

func (c *Catalog) Register(sk *Skill) {
	c.mu.Lock()
	defer c.mu.Unlock()
	key := strings.TrimRight(strings.TrimSpace(sk.URI), "/")
	if key == "" {
		key = "skill://local/" + strings.TrimSpace(sk.Name)
		sk.URI = key
	}
	c.skills[key] = sk
}

// LoadSnapshot ingests a jaz-cloud compiled CloudSkillsSnapshot JSON straight
// into memory (0 pods, 0 temp files).
func (c *Catalog) LoadSnapshot(rawJSON []byte) (int, error) {
	var snap CloudSkillsSnapshot
	if err := json.Unmarshal(rawJSON, &snap); err != nil {
		return 0, err
	}
	count := 0
	for _, item := range snap.Skills {
		sk := &Skill{
			Name:          item.Name,
			Description:   item.Description,
			URI:           item.Package,
			MainResource:  item.MainResource,
			Source:        SourceSnapshot,
			PromptVisible: item.PromptVisible,
			Resources:     item.Resources,
		}
		c.Register(sk)
		count++
	}
	return count, nil
}

// LoadDir scans a local directory tree for SKILL.md directories and loads UTF-8
// documentation files for progressive disclosure.
func (c *Catalog) LoadDir(rootDir, pluginID string) (int, error) {
	skillDirs := map[string]bool{}
	err := filepath.WalkDir(rootDir, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !d.IsDir() && d.Name() == "SKILL.md" {
			skillDirs[filepath.Dir(p)] = true
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	dirs := make([]string, 0, len(skillDirs))
	for d := range skillDirs {
		dirs = append(dirs, d)
	}
	sort.Strings(dirs)

	loaded := 0
	for _, dir := range dirs {
		mainBytes, err := os.ReadFile(filepath.Join(dir, "SKILL.md"))
		if err != nil || !utf8.Valid(mainBytes) {
			continue
		}
		name, desc := ParseSkillFrontmatter(string(mainBytes))
		if name == "" {
			name = filepath.Base(dir)
		}
		if desc == "" {
			desc = "Skill " + name
		}
		uri := "skill://local/" + name
		if pluginID != "" {
			uri = "skill://" + pluginID + "/" + name
		}
		resources := map[string]string{
			uri + "/SKILL.md": string(mainBytes),
		}
		_ = filepath.WalkDir(dir, func(p string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil || d.IsDir() {
				return nil
			}
			rel, rerr := filepath.Rel(dir, p)
			if rerr != nil {
				return nil
			}
			slashRel := filepath.ToSlash(rel)
			if slashRel == "SKILL.md" || !isTextDoc(slashRel) {
				return nil
			}
			raw, rerr := os.ReadFile(p)
			if rerr == nil && utf8.Valid(raw) && len(raw) <= 4<<20 {
				resources[uri+"/"+slashRel] = string(raw)
			}
			return nil
		})
		c.Register(&Skill{
			Name:          name,
			Description:   desc,
			URI:           uri,
			MainResource:  uri + "/SKILL.md",
			Source:        SourceLocal,
			PluginID:      pluginID,
			PromptVisible: true,
			Resources:     resources,
		})
		loaded++
	}
	return loaded, nil
}

// SyncFromMCP discovers skills exposed by an MCP server supporting SEP-2640
// (io.modelcontextprotocol/skills) or `skill://` resources.
func (c *Catalog) SyncFromMCP(ctx context.Context, client *mcp.Client) (int, error) {
	entries, err := client.ListSkills(ctx)
	if err != nil {
		return 0, err
	}
	loaded := 0
	for _, e := range entries {
		mainRes := e.MainResource
		if mainRes == "" {
			mainRes = strings.TrimRight(e.URI, "/") + "/SKILL.md"
		}
		c.Register(&Skill{
			Name:          e.Name,
			Description:   e.Description,
			URI:           e.URI,
			MainResource:  mainRes,
			Source:        SourceMCP,
			ServerName:    client.ServerName(),
			PromptVisible: true,
			MCPClient:     client,
		})
		loaded++
	}
	return loaded, nil
}

func (c *Catalog) List() []Skill {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]Skill, 0, len(c.skills))
	for _, sk := range c.skills {
		out = append(out, *sk)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// RenderPromptIndex produces a compact progressive-disclosure skill table for
// the system prompt (~100 tokens instead of loading full SKILL.md bodies).
func (c *Catalog) RenderPromptIndex() string {
	list := c.List()
	var lines []string
	for _, sk := range list {
		if !sk.PromptVisible {
			continue
		}
		origin := string(sk.Source)
		if sk.ServerName != "" {
			origin = "mcp:" + sk.ServerName
		}
		lines = append(lines, fmt.Sprintf("- `%s` (%s, `%s`): %s", sk.Name, origin, sk.MainResource, sk.Description))
	}
	if len(lines) == 0 {
		return ""
	}
	return "## Available Skills\nCall `skills_read` with a skill name or `skill://` URI to load its instructions on demand.\n" + strings.Join(lines, "\n")
}

// Read resolves either a skill name, skill package URI, or specific resource
// URI (e.g. `skill://acme/deploy/references/checklist.md`). Remote MCP skills
// are framed with untrusted origin metadata per SEP-2640 security rules.
func (c *Catalog) Read(ctx context.Context, target string) (string, error) {
	q := strings.TrimSpace(target)
	if q == "" {
		return "", fmt.Errorf("skill name or uri is required")
	}

	c.mu.RLock()
	var (
		matchedSkill *Skill
		matchedURI   string
		bestPrefix   int
	)
	for _, sk := range c.skills {
		if q == sk.URI || q == sk.MainResource {
			matchedSkill = sk
			matchedURI = sk.MainResource
			break
		}
		if q == sk.Name && matchedSkill == nil {
			matchedSkill = sk
			matchedURI = sk.MainResource
		}
		prefix := strings.TrimRight(sk.URI, "/") + "/"
		if strings.HasPrefix(q, prefix) && len(prefix) > bestPrefix {
			bestPrefix = len(prefix)
			matchedSkill = sk
			matchedURI = q
		}
	}
	c.mu.RUnlock()

	if matchedSkill == nil {
		return "", fmt.Errorf("skill %q not found", target)
	}

	if matchedSkill.Source == SourceMCP && matchedSkill.MCPClient != nil {
		contents, err := matchedSkill.MCPClient.ReadResource(ctx, matchedURI)
		if err != nil {
			return "", err
		}
		var parts []string
		for _, rc := range contents {
			if rc.Text != "" {
				parts = append(parts, rc.Text)
			}
		}
		body := strings.Join(parts, "\n")
		return fmt.Sprintf(
			"<untrusted_mcp_skill server=%q uri=%q>\n%s\n</untrusted_mcp_skill>",
			matchedSkill.ServerName,
			matchedURI,
			body,
		), nil
	}

	if content, ok := matchedSkill.Resources[matchedURI]; ok {
		return content, nil
	}
	return "", fmt.Errorf("skill resource %q not found in %s", matchedURI, matchedSkill.Name)
}

// ParseSkillFrontmatter extracts YAML frontmatter `name` and `description`.
func ParseSkillFrontmatter(md string) (name, description string) {
	trimmed := strings.TrimSpace(md)
	if !strings.HasPrefix(trimmed, "---") {
		return "", ""
	}
	rest := strings.TrimPrefix(trimmed, "---")
	rest = strings.TrimLeft(rest, "\r\n")
	end := strings.Index(rest, "\n---")
	if end < 0 {
		return "", ""
	}
	block := rest[:end]
	for _, line := range strings.Split(block, "\n") {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(k))
		val := strings.Trim(strings.TrimSpace(v), `"'`)
		switch key {
		case "name":
			name = val
		case "description":
			description = val
		}
	}
	return name, description
}

func isTextDoc(rel string) bool {
	lower := strings.ToLower(rel)
	if strings.Contains("/"+lower, "/scripts/") || strings.Contains("/"+lower, "/assets/") {
		return false
	}
	switch strings.ToLower(path.Ext(lower)) {
	case ".md", ".markdown", ".txt", ".mdx", ".rst":
		return true
	default:
		return false
	}
}
