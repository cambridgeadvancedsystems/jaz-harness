package execserver

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// HashToken returns the lowercase hex-encoded SHA-256 digest of a bearer
// capability token.
func HashToken(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])
}

// ParseTokenSHA256 parses and validates a 64-character hex-encoded SHA-256
// capability-token digest.
func ParseTokenSHA256(raw string) ([32]byte, error) {
	var out [32]byte
	trimmed := strings.TrimSpace(raw)
	if len(trimmed) != 64 {
		return out, fmt.Errorf("invalid token sha256 length %d: want 64 hex chars", len(trimmed))
	}
	decoded, err := hex.DecodeString(trimmed)
	if err != nil {
		return out, fmt.Errorf("invalid token sha256 hex: %w", err)
	}
	copy(out[:], decoded)
	return out, nil
}

type tokenAuthenticator struct {
	allowUnauth bool
	hasExpected bool
	expected    [32]byte
}

func newTokenAuthenticator(token, tokenSHA256 string, allowUnauth bool) (tokenAuthenticator, error) {
	trimmedToken := strings.TrimSpace(token)
	trimmedHash := strings.TrimSpace(tokenSHA256)

	if trimmedToken == "" && trimmedHash == "" {
		if allowUnauth {
			return tokenAuthenticator{allowUnauth: true}, nil
		}
		return tokenAuthenticator{}, errors.New("capability token or token SHA-256 is required unless AllowUnauthenticated is true")
	}

	var auth tokenAuthenticator
	if trimmedHash != "" {
		parsed, err := ParseTokenSHA256(trimmedHash)
		if err != nil {
			return tokenAuthenticator{}, err
		}
		auth.expected = parsed
		auth.hasExpected = true
	}
	if trimmedToken != "" {
		sum := sha256.Sum256([]byte(trimmedToken))
		if auth.hasExpected {
			if subtle.ConstantTimeCompare(sum[:], auth.expected[:]) != 1 {
				return tokenAuthenticator{}, errors.New("configured token does not match configured token SHA-256")
			}
		} else {
			auth.expected = sum
			auth.hasExpected = true
		}
	}
	return auth, nil
}

func (a tokenAuthenticator) authorizeRequest(r *http.Request) bool {
	if !a.hasExpected {
		return a.allowUnauth
	}
	bearer, ok := extractBearerToken(r.Header.Get("Authorization"))
	if !ok {
		return false
	}
	got := sha256.Sum256([]byte(bearer))
	return subtle.ConstantTimeCompare(got[:], a.expected[:]) == 1
}

func extractBearerToken(header string) (string, bool) {
	if len(header) < len("Bearer ")+1 {
		return "", false
	}
	scheme, rest, found := strings.Cut(header, " ")
	if !found || !strings.EqualFold(scheme, "Bearer") {
		return "", false
	}
	if rest == "" || strings.ContainsAny(rest, " \t\r\n") {
		return "", false
	}
	return rest, true
}
