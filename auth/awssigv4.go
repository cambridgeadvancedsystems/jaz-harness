package auth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

type AWSCredentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
	Region          string
}

// SignAWSV4 signs an outbound HTTP request with AWS Signature Version 4
// (used for Bedrock InvokeModelWithResponseStream and Bedrock OpenAI endpoints)
// with zero external SDK dependencies.
func SignAWSV4(req *http.Request, payload []byte, creds AWSCredentials, service string, now time.Time) {
	if now.IsZero() {
		now = time.Now().UTC()
	} else {
		now = now.UTC()
	}
	amzDate := now.Format("20060102T150405Z")
	dateStamp := now.Format("20060102")
	payloadHash := sha256Hex(payload)

	req.Header.Set("X-Amz-Date", amzDate)
	req.Header.Set("X-Amz-Content-Sha256", payloadHash)
	if creds.SessionToken != "" {
		req.Header.Set("X-Amz-Security-Token", creds.SessionToken)
	}

	host := req.Host
	if host == "" {
		host = req.URL.Host
	}

	canonicalURI := awsCanonicalURI(req.URL.Path)
	canonicalQuery := awsCanonicalQuery(req.URL.RawQuery)
	canonicalHeaders, signedHeaders := awsCanonicalHeaders(req.Header, host)

	canonicalReq := strings.Join([]string{
		req.Method,
		canonicalURI,
		canonicalQuery,
		canonicalHeaders,
		signedHeaders,
		payloadHash,
	}, "\n")

	credentialScope := fmt.Sprintf("%s/%s/%s/aws4_request", dateStamp, creds.Region, service)
	stringToSign := strings.Join([]string{
		"AWS4-HMAC-SHA256",
		amzDate,
		credentialScope,
		sha256Hex([]byte(canonicalReq)),
	}, "\n")

	kDate := hmacSHA256([]byte("AWS4"+creds.SecretAccessKey), dateStamp)
	kRegion := hmacSHA256(kDate, creds.Region)
	kService := hmacSHA256(kRegion, service)
	kSigning := hmacSHA256(kService, "aws4_request")
	signature := hex.EncodeToString(hmacSHA256(kSigning, stringToSign))

	authHeader := fmt.Sprintf(
		"AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		creds.AccessKeyID,
		credentialScope,
		signedHeaders,
		signature,
	)
	req.Header.Set("Authorization", authHeader)
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func hmacSHA256(key []byte, data string) []byte {
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(data))
	return mac.Sum(nil)
}

func awsCanonicalURI(rawPath string) string {
	if rawPath == "" {
		return "/"
	}
	segments := strings.Split(rawPath, "/")
	for i, seg := range segments {
		segments[i] = awsURIEncode(seg, false)
	}
	return strings.Join(segments, "/")
}

func awsCanonicalQuery(rawQuery string) string {
	if rawQuery == "" {
		return ""
	}
	pairs := strings.Split(rawQuery, "&")
	encoded := make([]string, 0, len(pairs))
	for _, p := range pairs {
		if p == "" {
			continue
		}
		k, v, hasEq := strings.Cut(p, "=")
		ek := awsURIEncode(k, true)
		ev := ""
		if hasEq {
			ev = awsURIEncode(v, true)
		}
		encoded = append(encoded, ek+"="+ev)
	}
	sort.Strings(encoded)
	return strings.Join(encoded, "&")
}

func awsCanonicalHeaders(h http.Header, host string) (string, string) {
	lowerMap := map[string][]string{
		"host": {strings.TrimSpace(host)},
	}
	for k, vals := range h {
		lk := strings.ToLower(k)
		if lk == "authorization" || lk == "user-agent" {
			continue
		}
		cleaned := make([]string, 0, len(vals))
		for _, v := range vals {
			cleaned = append(cleaned, collapseSpaces(strings.TrimSpace(v)))
		}
		lowerMap[lk] = cleaned
	}
	keys := make([]string, 0, len(lowerMap))
	for k := range lowerMap {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var headerBuf strings.Builder
	for _, k := range keys {
		headerBuf.WriteString(k)
		headerBuf.WriteByte(':')
		headerBuf.WriteString(strings.Join(lowerMap[k], ","))
		headerBuf.WriteByte('\n')
	}
	return headerBuf.String(), strings.Join(keys, ";")
}

func collapseSpaces(s string) string {
	fields := strings.Fields(s)
	return strings.Join(fields, " ")
}

func awsURIEncode(s string, encodeSlash bool) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		isUnreserved := (c >= 'A' && c <= 'Z') ||
			(c >= 'a' && c <= 'z') ||
			(c >= '0' && c <= '9') ||
			c == '_' || c == '-' || c == '~' || c == '.'
		if isUnreserved || (!encodeSlash && c == '/') {
			b.WriteByte(c)
		} else {
			fmt.Fprintf(&b, "%%%02X", c)
		}
	}
	return b.String()
}
