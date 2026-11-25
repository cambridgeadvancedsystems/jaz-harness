package tools_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/executor"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/mcp"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/schema"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/skills"
	"github.com/cambridgeadvancedsystems/jaz-harness/pkg/tools"
)

func TestDeferredBM25AndMCPSkillsAndLazyWorkspace(t *testing.T) {
	// 1. Skills catalog: local dir + jaz-cloud snapshot + remote SEP-2640 MCP server!
	skillDir := t.TempDir()
	localSkill := filepath.Join(skillDir, "deploy-prod")
	if err := os.MkdirAll(localSkill, 0o755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(localSkill, "SKILL.md"), []byte("---\nname: deploy-prod\ndescription: Production release checklist\n---\nRun canary first."), 0o644)

	cat := skills.NewCatalog()
	if n, err := cat.LoadDir(skillDir, ""); err != nil || n != 1 {
		t.Fatalf("LoadDir: n=%d err=%v", n, err)
	}
	snapJSON := []byte(`{"skills":[{"package":"skill://cloud/tender-review","name":"tender-review","description":"Diesel generator tender review","alias_root":"skill://cloud","main_resource":"skill://cloud/tender-review/SKILL.md","prompt_visible":true,"resources":{"skill://cloud/tender-review/SKILL.md":"Check prime vs standby rating."}}]}`)
	if n, err := cat.LoadSnapshot(snapJSON); err != nil || n != 1 {
		t.Fatalf("LoadSnapshot: n=%d err=%v", n, err)
	}

	mcpSrv := mcp.NewServer("crm", "1.0")
	mcpSrv.RegisterTool(mcp.Tool{
		Name:        "search_deals",
		Description: "Find active sales deals in CRM",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
	}, func(_ context.Context, args json.RawMessage) (mcp.CallToolResult, error) {
		return mcp.CallToolResult{Content: []mcp.ToolContent{{Type: "text", Text: "Deal #42 matches " + string(args)}}}, nil
	})
	mcpSrv.RegisterSkill("crm-triage", "Inbound email triage workflow", map[string]string{
		"SKILL.md": "---\nname: crm-triage\ndescription: Inbound email triage workflow\n---\nSkip newsletters.",
	})
	ts := httptest.NewServer(mcpSrv)
	defer ts.Close()
	mcpCli := mcp.NewHTTPClient("crm", ts.URL, nil, ts.Client())
	defer mcpCli.Close()

	plugs := skills.NewPluginRegistry()
	plugs.Register(skills.Plugin{
		ID:          "salesforce",
		Name:        "Salesforce",
		Description: "CRM integration",
		Status:      skills.PluginDisconnected,
		Enabled:     true,
	})

	reg := tools.NewRegistry(cat, plugs)
	tc, sc, err := reg.RegisterMCPServer(context.Background(), mcpCli, true)
	if err != nil || tc != 1 || sc != 1 {
		t.Fatalf("RegisterMCPServer: tools=%d skills=%d err=%v", tc, sc, err)
	}
	for i := range 30 {
		name := fmt.Sprintf("erp_tool_%02d", i)
		reg.Register(schema.ToolDef{
			Name:         name,
			Description:  "Unrelated ERP action",
			DeferLoading: true,
		}, func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil })
	}

	// Universal mode keeps active tool list tiny (gateway only, no 31 deferred tools).
	universalDefs := reg.ActiveToolDefs(false)
	for _, d := range universalDefs {
		if d.Name == "crm_search_deals" {
			t.Fatal("deferred tool must not be sent upfront in universal mode")
		}
	}
	// Anthropic native mode sends deferred tools with DeferLoading=true and omits search_tools/call_tool.
	nativeDefs := reg.ActiveToolDefs(true)
	foundDeferredCRM := false
	for _, d := range nativeDefs {
		if d.Name == "search_tools" || d.Name == "call_tool" {
			t.Fatalf("gateway tool %s should be omitted in native tool search mode", d.Name)
		}
		if d.Name == "crm_search_deals" && d.DeferLoading {
			foundDeferredCRM = true
		}
	}
	if !foundDeferredCRM {
		t.Fatal("expected crm_search_deals with DeferLoading=true in native mode")
	}

	// BM25 search finds crm_search_deals first, and call_tool invokes it.
	searchOut, err := reg.Execute(context.Background(), "search_tools", json.RawMessage(`{"query":"sales deals crm","limit":3}`))
	if err != nil || !strings.Contains(searchOut, "crm_search_deals") {
		t.Fatalf("search_tools failed: err=%v out=%s", err, searchOut)
	}
	callOut, err := reg.Execute(context.Background(), "call_tool", json.RawMessage(`{"name":"crm_search_deals","arguments":{"query":"ACME"}}`))
	if err != nil || !strings.Contains(callOut, "Deal #42") {
		t.Fatalf("call_tool failed: err=%v out=%s", err, callOut)
	}

	// skills_read serves all 3 sources and frames MCP skills as untrusted.
	mcpSkillText, err := reg.Execute(context.Background(), "skills_read", json.RawMessage(`{"target":"crm-triage"}`))
	if err != nil || !strings.Contains(mcpSkillText, `<untrusted_mcp_skill server="crm"`) || !strings.Contains(mcpSkillText, "Skip newsletters.") {
		t.Fatalf("MCP skill read failed: err=%v out=%s", err, mcpSkillText)
	}
	snapSkillText, err := reg.Execute(context.Background(), "skills_read", json.RawMessage(`{"target":"tender-review"}`))
	if err != nil || !strings.Contains(snapSkillText, "prime vs standby") {
		t.Fatalf("snapshot skill read failed: err=%v out=%s", err, snapSkillText)
	}

	// Plugins search returns jaz://plugin/salesforce/connect card link.
	plugOut, err := reg.Execute(context.Background(), "plugins_search", json.RawMessage(`{"query":"salesforce"}`))
	if err != nil || !strings.Contains(plugOut, "jaz://plugin/salesforce/connect") {
		t.Fatalf("plugins_search failed: err=%v out=%s", err, plugOut)
	}

	// Lazy workspace: 0 claims until a workspace tool is actually invoked!
	wsRoot := t.TempDir()
	_ = os.WriteFile(filepath.Join(wsRoot, "AGENTS.md"), []byte("Always run go test."), 0o644)
	var claims atomic.Int32
	wsTools := tools.NewWorkspaceToolset(func(_ context.Context) (executor.Executor, error) {
		claims.Add(1)
		return executor.NewLocal(wsRoot)
	})
	wsTools.RegisterInto(reg)
	if claims.Load() != 0 || wsTools.Attached() {
		t.Fatal("sandbox must not be claimed at registration time")
	}

	writeOut, err := reg.Execute(context.Background(), "write_file", json.RawMessage(`{"path":"hello.txt","content":"hello world"}`))
	if err != nil || claims.Load() != 1 || !strings.Contains(writeOut, "Always run go test.") {
		t.Fatalf("first workspace call should claim once and prepend AGENTS.md: err=%v claims=%d out=%s", err, claims.Load(), writeOut)
	}
	if _, err := reg.Execute(context.Background(), "edit_file", json.RawMessage(`{"path":"hello.txt","old_text":"world","new_text":"jaz"}`)); err != nil {
		t.Fatalf("edit_file failed: %v", err)
	}
	execOut, err := reg.Execute(context.Background(), "exec_command", json.RawMessage(`{"command":"cat hello.txt"}`))
	if err != nil || !strings.Contains(execOut, "hello jaz") || claims.Load() != 1 {
		t.Fatalf("exec_command failed: err=%v claims=%d out=%s", err, claims.Load(), execOut)
	}
}
