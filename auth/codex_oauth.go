package auth

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	CodexDefaultIssuer     = "https://auth.openai.com"
	CodexDefaultClientID   = "app_EMoamEEZ73f0CkXaXp7hrann"
	CodexDefaultBackendURL = "https://chatgpt.com/backend-api/codex"
	codexRefreshLeadWindow = 2 * time.Minute
)

// CodexAuthFile matches ~/.codex/auth.json so credentials interoperate seamlessly
// with the official Codex CLI and Jaz Cloud ACP profiles.
type CodexAuthFile struct {
	OpenAIAPIKey string         `json:"OPENAI_API_KEY,omitempty"`
	Tokens       *CodexTokenSet `json:"tokens,omitempty"`
	LastRefresh  *time.Time     `json:"last_refresh,omitempty"`
}

type CodexTokenSet struct {
	IDToken      string `json:"id_token,omitempty"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token,omitempty"`
	AccountID    string `json:"account_id,omitempty"`
}

type CodexCredentials struct {
	AccessToken string
	AccountID   string
	ExpiresAt   time.Time
}

type CodexDeviceCode struct {
	VerificationURL string        `json:"verification_url"`
	UserCode        string        `json:"user_code"`
	DeviceAuthID    string        `json:"device_auth_id"`
	Interval        time.Duration `json:"interval"`
}

type CodexOAuthConfig struct {
	AuthPath   string
	Issuer     string
	ClientID   string
	HTTPClient *http.Client
	Now        func() time.Time
	OnPersist  func(ctx context.Context, file CodexAuthFile) error
}

// CodexOAuthManager handles ChatGPT/Codex OAuth loading, JWT expiry inspection,
// single-flight serialized refresh-token rotation, and device-code login.
type CodexOAuthManager struct {
	cfg        CodexOAuthConfig
	mu         sync.Mutex
	file       CodexAuthFile
	loaded     bool
	httpClient *http.Client
	nowFn      func() time.Time
}

func NewCodexOAuthManager(cfg CodexOAuthConfig) *CodexOAuthManager {
	if strings.TrimSpace(cfg.Issuer) == "" {
		cfg.Issuer = CodexDefaultIssuer
	}
	cfg.Issuer = strings.TrimRight(cfg.Issuer, "/")
	if strings.TrimSpace(cfg.ClientID) == "" {
		cfg.ClientID = CodexDefaultClientID
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: 30 * time.Second}
	}
	nowFn := cfg.Now
	if nowFn == nil {
		nowFn = time.Now
	}
	return &CodexOAuthManager{
		cfg:        cfg,
		httpClient: hc,
		nowFn:      nowFn,
	}
}

func DefaultCodexAuthPath() string {
	if codexHome := strings.TrimSpace(os.Getenv("CODEX_HOME")); codexHome != "" {
		return filepath.Join(codexHome, "auth.json")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".codex", "auth.json")
}

func (m *CodexOAuthManager) LoadFromMemory(file CodexAuthFile) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.file = file
	m.loaded = true
}

func (m *CodexOAuthManager) LoadFromDisk() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.loadLocked()
}

func (m *CodexOAuthManager) loadLocked() error {
	path := m.cfg.AuthPath
	if path == "" {
		path = DefaultCodexAuthPath()
	}
	if path == "" {
		return errors.New("codex auth path not configured")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read codex auth.json: %w", err)
	}
	var parsed CodexAuthFile
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return fmt.Errorf("parse codex auth.json: %w", err)
	}
	m.file = parsed
	m.loaded = true
	return nil
}

func (m *CodexOAuthManager) persistLocked(ctx context.Context) error {
	if m.cfg.OnPersist != nil {
		if err := m.cfg.OnPersist(ctx, m.file); err != nil {
			return err
		}
	}
	path := m.cfg.AuthPath
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(m.file, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Credentials returns a valid access token and account ID, proactively
// refreshing with a mutex lock so single-use refresh tokens never race.
func (m *CodexOAuthManager) Credentials(ctx context.Context) (CodexCredentials, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.loaded {
		if err := m.loadLocked(); err != nil {
			return CodexCredentials{}, err
		}
	}
	if m.file.Tokens == nil || strings.TrimSpace(m.file.Tokens.AccessToken) == "" {
		return CodexCredentials{}, errors.New("codex oauth tokens not present; run login codex")
	}

	exp := extractJWTExpiry(m.file.Tokens.AccessToken)
	now := m.nowFn().UTC()
	if !exp.IsZero() && now.Add(codexRefreshLeadWindow).After(exp) {
		if strings.TrimSpace(m.file.Tokens.RefreshToken) == "" {
			return CodexCredentials{}, errors.New("codex oauth access token expired and no refresh_token available")
		}
		if err := m.refreshLocked(ctx); err != nil {
			return CodexCredentials{}, err
		}
		exp = extractJWTExpiry(m.file.Tokens.AccessToken)
	}

	accountID := strings.TrimSpace(m.file.Tokens.AccountID)
	if accountID == "" {
		accountID = extractChatGPTAccountID(m.file.Tokens.AccessToken, m.file.Tokens.IDToken)
	}
	return CodexCredentials{
		AccessToken: m.file.Tokens.AccessToken,
		AccountID:   accountID,
		ExpiresAt:   exp,
	}, nil
}

// ForceRefresh rotates the OAuth token pair immediately (e.g. after a 401).
func (m *CodexOAuthManager) ForceRefresh(ctx context.Context) (CodexCredentials, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.loaded {
		if err := m.loadLocked(); err != nil {
			return CodexCredentials{}, err
		}
	}
	if m.file.Tokens == nil || strings.TrimSpace(m.file.Tokens.RefreshToken) == "" {
		return CodexCredentials{}, errors.New("codex refresh_token missing")
	}
	if err := m.refreshLocked(ctx); err != nil {
		return CodexCredentials{}, err
	}
	accountID := strings.TrimSpace(m.file.Tokens.AccountID)
	if accountID == "" {
		accountID = extractChatGPTAccountID(m.file.Tokens.AccessToken, m.file.Tokens.IDToken)
	}
	return CodexCredentials{
		AccessToken: m.file.Tokens.AccessToken,
		AccountID:   accountID,
		ExpiresAt:   extractJWTExpiry(m.file.Tokens.AccessToken),
	}, nil
}

func (m *CodexOAuthManager) refreshLocked(ctx context.Context) error {
	payload, err := json.Marshal(map[string]string{
		"client_id":     m.cfg.ClientID,
		"grant_type":    "refresh_token",
		"refresh_token": m.file.Tokens.RefreshToken,
		"scope":         "openid profile email",
	})
	if err != nil {
		return err
	}
	tokenURL := m.cfg.Issuer + "/oauth/token"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("codex oauth refresh request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return fmt.Errorf("codex oauth refresh read: %w", err)
	}
	if resp.StatusCode >= 300 {
		return fmt.Errorf("codex oauth refresh status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var refreshed struct {
		IDToken      string `json:"id_token"`
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
	}
	if err := json.Unmarshal(body, &refreshed); err != nil {
		return fmt.Errorf("decode codex oauth refresh: %w", err)
	}
	if strings.TrimSpace(refreshed.AccessToken) == "" {
		return errors.New("codex oauth refresh returned empty access_token")
	}
	m.file.Tokens.AccessToken = refreshed.AccessToken
	if refreshed.IDToken != "" {
		m.file.Tokens.IDToken = refreshed.IDToken
	}
	if refreshed.RefreshToken != "" {
		m.file.Tokens.RefreshToken = refreshed.RefreshToken
	}
	if m.file.Tokens.AccountID == "" {
		m.file.Tokens.AccountID = extractChatGPTAccountID(m.file.Tokens.AccessToken, m.file.Tokens.IDToken)
	}
	now := m.nowFn().UTC()
	m.file.LastRefresh = &now
	return m.persistLocked(ctx)
}

// RequestDeviceCode initiates OpenAI Codex device-code authorization.
func (m *CodexOAuthManager) RequestDeviceCode(ctx context.Context) (CodexDeviceCode, error) {
	endpoint := m.cfg.Issuer + "/api/accounts/deviceauth/usercode"
	body, err := json.Marshal(map[string]string{
		"client_id": m.cfg.ClientID,
	})
	if err != nil {
		return CodexDeviceCode{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return CodexDeviceCode{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return CodexDeviceCode{}, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return CodexDeviceCode{}, err
	}
	if resp.StatusCode >= 300 {
		return CodexDeviceCode{}, fmt.Errorf("device code request status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var parsed struct {
		DeviceAuthID string `json:"device_auth_id"`
		UserCode     string `json:"user_code"`
		UserCodeAlt  string `json:"usercode"`
		Interval     any    `json:"interval"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return CodexDeviceCode{}, err
	}
	userCode := parsed.UserCode
	if userCode == "" {
		userCode = parsed.UserCodeAlt
	}
	interval := parseIntervalSeconds(parsed.Interval, 5*time.Second)
	return CodexDeviceCode{
		VerificationURL: m.cfg.Issuer + "/codex/device",
		UserCode:        userCode,
		DeviceAuthID:    parsed.DeviceAuthID,
		Interval:        interval,
	}, nil
}

// CompleteDeviceCodeLogin polls until the user approves the device code,
// exchanges the PKCE authorization code for OAuth tokens, and persists them.
func (m *CodexOAuthManager) CompleteDeviceCodeLogin(ctx context.Context, dc CodexDeviceCode) (CodexAuthFile, error) {
	pollURL := m.cfg.Issuer + "/api/accounts/deviceauth/token"
	interval := dc.Interval
	if interval <= 0 {
		interval = 5 * time.Second
	}

	var codeResp struct {
		AuthorizationCode string `json:"authorization_code"`
		CodeVerifier      string `json:"code_verifier"`
	}

	for {
		payload, _ := json.Marshal(map[string]string{
			"device_auth_id": dc.DeviceAuthID,
			"user_code":      dc.UserCode,
		})
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, pollURL, bytes.NewReader(payload))
		if err != nil {
			return CodexAuthFile{}, err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := m.httpClient.Do(req)
		if err != nil {
			return CodexAuthFile{}, err
		}
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		resp.Body.Close()

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			if err := json.Unmarshal(raw, &codeResp); err != nil {
				return CodexAuthFile{}, err
			}
			break
		}
		if resp.StatusCode != http.StatusForbidden && resp.StatusCode != http.StatusNotFound {
			return CodexAuthFile{}, fmt.Errorf("device auth poll status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
		}
		select {
		case <-ctx.Done():
			return CodexAuthFile{}, ctx.Err()
		case <-time.After(interval):
		}
	}

	form := url.Values{}
	form.Set("grant_type", "authorization_code")
	form.Set("code", codeResp.AuthorizationCode)
	form.Set("redirect_uri", m.cfg.Issuer+"/deviceauth/callback")
	form.Set("client_id", m.cfg.ClientID)
	form.Set("code_verifier", codeResp.CodeVerifier)

	exchReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.cfg.Issuer+"/oauth/token", strings.NewReader(form.Encode()))
	if err != nil {
		return CodexAuthFile{}, err
	}
	exchReq.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	exchResp, err := m.httpClient.Do(exchReq)
	if err != nil {
		return CodexAuthFile{}, err
	}
	defer exchResp.Body.Close()
	exchRaw, _ := io.ReadAll(io.LimitReader(exchResp.Body, 1<<20))
	if exchResp.StatusCode >= 300 {
		return CodexAuthFile{}, fmt.Errorf("device code token exchange status %d: %s", exchResp.StatusCode, strings.TrimSpace(string(exchRaw)))
	}

	var tok struct {
		IDToken      string `json:"id_token"`
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
	}
	if err := json.Unmarshal(exchRaw, &tok); err != nil {
		return CodexAuthFile{}, err
	}
	now := m.nowFn().UTC()
	m.mu.Lock()
	defer m.mu.Unlock()
	m.file = CodexAuthFile{
		Tokens: &CodexTokenSet{
			IDToken:      tok.IDToken,
			AccessToken:  tok.AccessToken,
			RefreshToken: tok.RefreshToken,
			AccountID:    extractChatGPTAccountID(tok.AccessToken, tok.IDToken),
		},
		LastRefresh: &now,
	}
	m.loaded = true
	if err := m.persistLocked(ctx); err != nil {
		return CodexAuthFile{}, err
	}
	return m.file, nil
}

func parseIntervalSeconds(raw any, fallback time.Duration) time.Duration {
	switch v := raw.(type) {
	case float64:
		if v > 0 {
			return time.Duration(v) * time.Second
		}
	case string:
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			return time.Duration(n) * time.Second
		}
	}
	return fallback
}

func decodeJWTPayload(jwt string) map[string]any {
	parts := strings.Split(strings.TrimSpace(jwt), ".")
	if len(parts) < 2 {
		return nil
	}
	raw, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil
	}
	var claims map[string]any
	if json.Unmarshal(raw, &claims) != nil {
		return nil
	}
	return claims
}

func extractJWTExpiry(jwt string) time.Time {
	claims := decodeJWTPayload(jwt)
	if claims == nil {
		return time.Time{}
	}
	exp, ok := claims["exp"].(float64)
	if !ok || exp <= 0 {
		return time.Time{}
	}
	return time.Unix(int64(exp), 0).UTC()
}

func extractChatGPTAccountID(tokens ...string) string {
	for _, tok := range tokens {
		claims := decodeJWTPayload(tok)
		if claims == nil {
			continue
		}
		if authObj, ok := claims["https://api.openai.com/auth"].(map[string]any); ok {
			if acct, ok := authObj["chatgpt_account_id"].(string); ok && strings.TrimSpace(acct) != "" {
				return strings.TrimSpace(acct)
			}
		}
		if acct, ok := claims["chatgpt_account_id"].(string); ok && strings.TrimSpace(acct) != "" {
			return strings.TrimSpace(acct)
		}
	}
	return ""
}
