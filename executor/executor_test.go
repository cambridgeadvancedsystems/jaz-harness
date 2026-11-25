package executor

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/websocket"
)

func TestPathToFileURI(t *testing.T) {
	cases := []struct {
		ws   string
		in   string
		want string
	}{
		{"/workspace", "", "file:///workspace"},
		{"/workspace", ".", "file:///workspace"},
		{"/workspace", "/workspace", "file:///workspace"},
		{"/workspace", "README.md", "file:///workspace/README.md"},
		{"/workspace", "dir with space/hello #1?x%2.txt", "file:///workspace/dir%20with%20space/hello%20%231%3Fx%252.txt"},
		{"/workspace", "file:///workspace/already.txt", "file:///workspace/already.txt"},
		{`C:\Users\Alice\repo`, "src\\main.go", "file:///C:/Users/Alice/repo/src/main.go"},
		{`C:\Users\Alice Smith\repo`, "", "file:///C:/Users/Alice%20Smith/repo"},
		{"/workspace", `d:\data\a b.txt`, "file:///D:/data/a%20b.txt"},
	}
	for _, tc := range cases {
		got := pathToFileURI(tc.ws, tc.in)
		if got != tc.want {
			t.Errorf("pathToFileURI(%q, %q) = %q, want %q", tc.ws, tc.in, got, tc.want)
		}
	}
}

func TestCodexExecClientWebSocketProtocol(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	var (
		sawAuth       string
		sawInit       bool
		sawStartCwd   string
		sawStartTTY   bool
		sawEnvNotNull bool
		sawWriteID    string
	)
	files := map[string][]byte{
		"file:///workspace/README.md": []byte("hello from gvisor"),
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawAuth = r.Header.Get("Authorization")
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()

		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var req rpcFrame
			if json.Unmarshal(data, &req) != nil {
				continue
			}
			if req.Method == "initialized" && len(req.ID) == 0 {
				sawInit = true
				continue
			}
			if len(req.ID) == 0 {
				continue
			}
			writeResp := func(res any) {
				raw, _ := json.Marshal(res)
				out, _ := json.Marshal(rpcFrame{ID: req.ID, Result: raw})
				_ = conn.WriteMessage(websocket.TextMessage, out)
			}
			switch req.Method {
			case "initialize":
				writeResp(map[string]any{
					"sessionId": "sess-1",
					"environmentInfo": map[string]any{
						"platformOs": "linux",
						"cwd":        "file:///workspace",
					},
				})
			case "process/start":
				var rawMap map[string]json.RawMessage
				_ = json.Unmarshal(req.Params, &rawMap)
				if _, ok := rawMap["tty"]; ok {
					sawStartTTY = true
				}
				if envRaw, ok := rawMap["env"]; ok && string(envRaw) != "null" {
					sawEnvNotNull = true
				}
				var p processStartParams
				_ = json.Unmarshal(req.Params, &p)
				sawStartCwd = p.Cwd
				writeResp(map[string]any{"processId": p.ProcessID})
				outNotif, _ := json.Marshal(rpcFrame{
					Method: "process/output",
					Params: json.RawMessage(`{"processId":"` + p.ProcessID + `","seq":1,"stream":"stdout","chunk":"` + base64.StdEncoding.EncodeToString([]byte("sandbox-ok\n")) + `"}`),
				})
				_ = conn.WriteMessage(websocket.TextMessage, outNotif)
				exitNotif, _ := json.Marshal(rpcFrame{
					Method: "process/exited",
					Params: json.RawMessage(`{"processId":"` + p.ProcessID + `","seq":2,"exitCode":0}`),
				})
				_ = conn.WriteMessage(websocket.TextMessage, exitNotif)
				closedNotif, _ := json.Marshal(rpcFrame{
					Method: "process/closed",
					Params: json.RawMessage(`{"processId":"` + p.ProcessID + `","seq":3}`),
				})
				_ = conn.WriteMessage(websocket.TextMessage, closedNotif)
			case "process/write":
				var p processWriteParams
				_ = json.Unmarshal(req.Params, &p)
				sawWriteID = p.WriteID
				writeResp(map[string]any{"status": "accepted"})
			case "fs/readFile":
				var p fsPathParams
				_ = json.Unmarshal(req.Params, &p)
				writeResp(map[string]any{"dataBase64": base64.StdEncoding.EncodeToString(files[p.Path])})
			case "fs/writeFile":
				var p fsWriteFileParams
				_ = json.Unmarshal(req.Params, &p)
				raw, _ := base64.StdEncoding.DecodeString(p.DataBase64)
				files[p.Path] = raw
				writeResp(map[string]any{})
			case "fs/readDirectory":
				var p fsPathParams
				_ = json.Unmarshal(req.Params, &p)
				if p.Path != "file:///workspace" {
					t.Errorf("expected file:///workspace URI, got %q", p.Path)
				}
				writeResp(map[string]any{
					"entries": []map[string]any{
						{"fileName": "README.md", "isDirectory": false, "isFile": true},
					},
				})
			}
		}
	}))
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")
	cli, err := DialCodexExecServer(context.Background(), wsURL, "sandbox-jwt-token", "/workspace")
	if err != nil {
		t.Fatalf("DialCodexExecServer: %v", err)
	}
	defer cli.Close()

	if sawAuth != "Bearer sandbox-jwt-token" {
		t.Fatalf("expected direct Bearer header over ws://, got %q", sawAuth)
	}
	res, err := cli.ExecStart(context.Background(), ExecRequest{Command: "echo sandbox-ok", Shell: true})
	if err != nil || res.ExitCode != 0 || strings.TrimSpace(res.Stdout) != "sandbox-ok" {
		t.Fatalf("ExecStart failed: err=%v res=%+v", err, res)
	}
	if !sawInit || sawStartCwd != "file:///workspace" || !sawStartTTY || !sawEnvNotNull {
		t.Fatalf("process/start wire contract mismatch: init=%v cwd=%q tty=%v envNotNull=%v", sawInit, sawStartCwd, sawStartTTY, sawEnvNotNull)
	}
	if _, err := cli.ExecWriteStdin(context.Background(), res.ProcessID, "hi\n"); err != nil || sawWriteID == "" {
		t.Fatalf("ExecWriteStdin failed: err=%v writeId=%q", err, sawWriteID)
	}
	if err := cli.WriteFile(context.Background(), "out.txt", []byte("written")); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	got, err := cli.ReadFile(context.Background(), "out.txt")
	if err != nil || string(got) != "written" {
		t.Fatalf("ReadFile: err=%v got=%q", err, string(got))
	}
	entries, err := cli.ListDir(context.Background(), "/workspace")
	if err != nil || len(entries) != 1 || entries[0].Name != "README.md" {
		t.Fatalf("ListDir: err=%v entries=%+v", err, entries)
	}
}
