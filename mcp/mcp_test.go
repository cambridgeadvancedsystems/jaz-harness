package mcp

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMCPClientServerAndSEP2640Skills(t *testing.T) {
	srv := NewServer("acme-mcp", "1.0.0")
	srv.RegisterTool(Tool{
		Name:        "query_invoices",
		Description: "Query unpaid invoices from ERP",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"status":{"type":"string"}}}`),
	}, func(_ context.Context, args json.RawMessage) (CallToolResult, error) {
		return CallToolResult{
			Content: []ToolContent{{Type: "text", Text: "3 unpaid invoices for " + string(args)}},
		}, nil
	})
	srv.RegisterSkill("invoice-audit", "Step-by-step ERP invoice reconciliation", map[string]string{
		"SKILL.md":              "---\nname: invoice-audit\ndescription: Step-by-step ERP invoice reconciliation\n---\nAlways check PO numbers first.",
		"references/clauses.md": "Standard payment terms are net 30.",
	})

	ts := httptest.NewServer(srv)
	defer ts.Close()

	cli := NewHTTPClient("acme", ts.URL, nil, ts.Client())
	defer cli.Close()

	ctx := context.Background()
	initRes, err := cli.Initialize(ctx)
	if err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	if !cli.SupportsSkillsExtension() {
		t.Fatalf("expected SEP-2640 skills extension capability: %+v", initRes.Capabilities)
	}

	tools, err := cli.ListTools(ctx)
	if err != nil || len(tools) != 1 || tools[0].Name != "query_invoices" {
		t.Fatalf("ListTools: err=%v tools=%+v", err, tools)
	}

	callRes, err := cli.CallTool(ctx, "query_invoices", json.RawMessage(`{"status":"open"}`))
	if err != nil || len(callRes.Content) == 0 || !strings.Contains(callRes.Content[0].Text, "3 unpaid invoices") {
		t.Fatalf("CallTool: err=%v res=%+v", err, callRes)
	}

	skills, err := cli.ListSkills(ctx)
	if err != nil || len(skills) != 1 || skills[0].URI != "skill://invoice-audit" {
		t.Fatalf("ListSkills: err=%v skills=%+v", err, skills)
	}

	detail, err := cli.GetSkill(ctx, "skill://invoice-audit")
	if err != nil || detail.SHA256 == "" || len(detail.Resources) != 2 {
		t.Fatalf("GetSkill: err=%v detail=%+v", err, detail)
	}

	contents, err := cli.ReadResource(ctx, "skill://invoice-audit/SKILL.md")
	if err != nil || len(contents) != 1 || !strings.Contains(contents[0].Text, "Always check PO numbers first.") {
		t.Fatalf("ReadResource: err=%v contents=%+v", err, contents)
	}

	dirRes, err := cli.ReadResourceDirectory(ctx, "skill://invoice-audit")
	if err != nil || len(dirRes.Entries) != 2 {
		t.Fatalf("ReadResourceDirectory: err=%v res=%+v", err, dirRes)
	}
}
