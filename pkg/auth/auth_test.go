package auth

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func makeTestJWT(t *testing.T, claims map[string]any) string {
	t.Helper()
	hdr := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none","typ":"JWT"}`))
	raw, err := json.Marshal(claims)
	if err != nil {
		t.Fatal(err)
	}
	body := base64.RawURLEncoding.EncodeToString(raw)
	return hdr + "." + body + ".sig"
}

func TestCodexOAuthSerializedRefreshAndPersistence(t *testing.T) {
	now := time.Date(2026, 11, 25, 12, 0, 0, 0, time.UTC)
	expiredJWT := makeTestJWT(t, map[string]any{
		"exp": now.Add(30 * time.Second).Unix(), // inside lead window -> triggers refresh
		"https://api.openai.com/auth": map[string]any{
			"chatgpt_account_id": "acct_initial",
		},
	})
	freshJWT := makeTestJWT(t, map[string]any{
		"exp": now.Add(2 * time.Hour).Unix(),
		"https://api.openai.com/auth": map[string]any{
			"chatgpt_account_id": "acct_refreshed",
		},
	})

	var refreshCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/oauth/token" {
			http.NotFound(w, r)
			return
		}
		refreshCalls.Add(1)
		time.Sleep(30 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token":  freshJWT,
			"refresh_token": "rt_rotated_2",
			"id_token":      freshJWT,
		})
	}))
	defer srv.Close()

	authPath := filepath.Join(t.TempDir(), "auth.json")
	mgr := NewCodexOAuthManager(CodexOAuthConfig{
		AuthPath:   authPath,
		Issuer:     srv.URL,
		HTTPClient: srv.Client(),
		Now:        func() time.Time { return now },
	})
	mgr.LoadFromMemory(CodexAuthFile{
		Tokens: &CodexTokenSet{
			AccessToken:  expiredJWT,
			RefreshToken: "rt_initial_1",
		},
	})

	// Launch 8 concurrent callers; mutex serialization must execute only ONE refresh.
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			creds, err := mgr.Credentials(context.Background())
			if err != nil {
				errs <- err
				return
			}
			if creds.AccessToken != freshJWT {
				errs <- os.ErrInvalid
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("unexpected concurrent credential error: %v", err)
	}
	if got := refreshCalls.Load(); got != 1 {
		t.Fatalf("refresh calls = %d, want 1 (serialized)", got)
	}

	raw, err := os.ReadFile(authPath)
	if err != nil {
		t.Fatalf("expected persisted auth.json: %v", err)
	}
	var saved CodexAuthFile
	if err := json.Unmarshal(raw, &saved); err != nil {
		t.Fatal(err)
	}
	if saved.Tokens == nil || saved.Tokens.RefreshToken != "rt_rotated_2" {
		t.Fatalf("persisted refresh_token = %#v", saved.Tokens)
	}
}

func TestCodexDeviceCodeLoginFlow(t *testing.T) {
	now := time.Date(2026, 11, 25, 12, 0, 0, 0, time.UTC)
	issuedJWT := makeTestJWT(t, map[string]any{
		"exp":                now.Add(1 * time.Hour).Unix(),
		"chatgpt_account_id": "acct_device_99",
	})
	var polls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/accounts/deviceauth/usercode":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"device_auth_id": "da_123",
				"user_code":      "ABCD-EFGH",
				"interval":       "1",
			})
		case "/api/accounts/deviceauth/token":
			if polls.Add(1) == 1 {
				w.WriteHeader(http.StatusForbidden)
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"authorization_code": "code_xyz",
				"code_verifier":      "verifier_123",
				"code_challenge":     "challenge_123",
			})
		case "/oauth/token":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"access_token":  issuedJWT,
				"refresh_token": "rt_device",
				"id_token":      issuedJWT,
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	authPath := filepath.Join(t.TempDir(), "auth.json")
	mgr := NewCodexOAuthManager(CodexOAuthConfig{
		AuthPath:   authPath,
		Issuer:     srv.URL,
		HTTPClient: srv.Client(),
		Now:        func() time.Time { return now },
	})
	dc, err := mgr.RequestDeviceCode(context.Background())
	if err != nil {
		t.Fatalf("RequestDeviceCode: %v", err)
	}
	if dc.UserCode != "ABCD-EFGH" {
		t.Fatalf("user code = %q", dc.UserCode)
	}
	dc.Interval = 10 * time.Millisecond
	file, err := mgr.CompleteDeviceCodeLogin(context.Background(), dc)
	if err != nil {
		t.Fatalf("CompleteDeviceCodeLogin: %v", err)
	}
	if file.Tokens == nil || file.Tokens.AccountID != "acct_device_99" {
		t.Fatalf("unexpected account id: %#v", file.Tokens)
	}
}

func TestSignAWSV4(t *testing.T) {
	body := []byte(`{"anthropic_version":"bedrock-2023-05-31"}`)
	req, err := http.NewRequest(http.MethodPost, "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-5-v1:0/invoke-with-response-stream", strings.NewReader(string(body)))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	SignAWSV4(req, body, AWSCredentials{
		AccessKeyID:     "AKIAEXAMPLE",
		SecretAccessKey: "secret-key",
		SessionToken:    "sess-tok",
		Region:          "us-east-1",
	}, "bedrock", time.Date(2026, 11, 25, 12, 0, 0, 0, time.UTC))

	authz := req.Header.Get("Authorization")
	if !strings.HasPrefix(authz, "AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/20261125/us-east-1/bedrock/aws4_request,") {
		t.Fatalf("unexpected Authorization header: %s", authz)
	}
	if req.Header.Get("X-Amz-Date") != "20261125T120000Z" {
		t.Fatalf("unexpected X-Amz-Date: %s", req.Header.Get("X-Amz-Date"))
	}
	if req.Header.Get("X-Amz-Security-Token") != "sess-tok" {
		t.Fatalf("missing session token header")
	}
}
