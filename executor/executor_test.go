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

func TestCodexExecClientWebSocketProtocol(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	var sawAuth string
	files := map[string][]byte{
		"/workspace/README.md": []byte("hello from gvisor"),
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
			if len(req.ID) == 0 {
				continue
			}
			writeResp := func(res any) {
				raw, _ := json.Marshal(res)
				out, _ := json.Marshal(rpcFrame{JSONRPC: "2.0", ID: req.ID, Result: raw})
				_ = conn.WriteMessage(websocket.TextMessage, out)
			}
			switch req.Method {
			case "initialize":
				writeResp(map[string]any{"serverName": "codex-exec-server"})
			case "process/start":
				var p struct {
					ProcessID string `json:"processId"`
				}
				_ = json.Unmarshal(req.Params, &p)
				writeResp(map[string]any{"processId": p.ProcessID})
				outNotif, _ := json.Marshal(rpcFrame{
					JSONRPC: "2.0",
					Method:  "process/output",
					Params:  json.RawMessage(`{"processId":"` + p.ProcessID + `","stream":"stdout","chunk":"` + base64.StdEncoding.EncodeToString([]byte("sandbox-ok\n")) + `"}`),
				})
				_ = conn.WriteMessage(websocket.TextMessage, outNotif)
				exitNotif, _ := json.Marshal(rpcFrame{
					JSONRPC: "2.0",
					Method:  "process/exited",
					Params:  json.RawMessage(`{"processId":"` + p.ProcessID + `","exitCode":0}`),
				})
				_ = conn.WriteMessage(websocket.TextMessage, exitNotif)
			case "fs/readFile":
				var p struct {
					Path string `json:"path"`
				}
				_ = json.Unmarshal(req.Params, &p)
				writeResp(map[string]any{"dataBase64": base64.StdEncoding.EncodeToString(files[p.Path])})
			case "fs/writeFile":
				var p struct {
					Path       string `json:"path"`
					DataBase64 string `json:"dataBase64"`
				}
				_ = json.Unmarshal(req.Params, &p)
				raw, _ := base64.StdEncoding.DecodeString(p.DataBase64)
				files[p.Path] = raw
				writeResp(map[string]any{})
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
	if err := cli.WriteFile(context.Background(), "out.txt", []byte("written")); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	got, err := cli.ReadFile(context.Background(), "out.txt")
	if err != nil || string(got) != "written" {
		t.Fatalf("ReadFile: err=%v got=%q", err, string(got))
	}
}
