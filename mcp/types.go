package mcp

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

const (
	ProtocolVersion      = "2025-11-25"
	SkillsExtensionID    = "io.modelcontextprotocol/skills"
	SkillURISchemePrefix = "skill://"
)

type JSONRPCMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *JSONRPCError   `json:"error,omitempty"`
}

type JSONRPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *JSONRPCError) Error() string {
	return fmt.Sprintf("mcp rpc error %d: %s", e.Code, e.Message)
}

type Implementation struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type Capabilities struct {
	Tools      map[string]any            `json:"tools,omitempty"`
	Resources  map[string]any            `json:"resources,omitempty"`
	Prompts    map[string]any            `json:"prompts,omitempty"`
	Extensions map[string]map[string]any `json:"extensions,omitempty"`
}

type InitializeResult struct {
	ProtocolVersion string         `json:"protocolVersion"`
	Capabilities    Capabilities   `json:"capabilities"`
	ServerInfo      Implementation `json:"serverInfo"`
}

type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

type ToolContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
	Data     string `json:"data,omitempty"`
}

type CallToolResult struct {
	Content           []ToolContent   `json:"content"`
	StructuredContent json.RawMessage `json:"structuredContent,omitempty"`
	IsError           bool            `json:"isError,omitempty"`
}

type Resource struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	MimeType    string `json:"mimeType,omitempty"`
	Size        int64  `json:"size,omitempty"`
	SHA256      string `json:"sha256,omitempty"`
}

type ResourceContent struct {
	URI      string `json:"uri"`
	MimeType string `json:"mimeType,omitempty"`
	Text     string `json:"text,omitempty"`
	Blob     string `json:"blob,omitempty"`
	SHA256   string `json:"sha256,omitempty"`
}

type Prompt struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

// SEP-2640 (io.modelcontextprotocol/skills) types.
type SkillEntry struct {
	URI          string            `json:"uri"`
	Name         string            `json:"name"`
	Description  string            `json:"description"`
	MainResource string            `json:"mainResource,omitempty"`
	Version      string            `json:"version,omitempty"`
	SHA256       string            `json:"sha256,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
	Resources    []Resource        `json:"resources,omitempty"`
}

type DirectoryReadResult struct {
	URI     string            `json:"uri"`
	Entries []ResourceContent `json:"entries"`
}

func ComputeSHA256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
