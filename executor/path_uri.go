package executor

import (
	"net/url"
	"path"
	"strings"
)

func pathToFileURI(workspaceDir, raw string) string {
	trimmed := strings.TrimSpace(raw)
	if strings.HasPrefix(trimmed, "file://") {
		return trimmed
	}
	ws := strings.TrimSpace(workspaceDir)
	if ws == "" {
		ws = "/workspace"
	}
	if isWindowsDrivePath(trimmed) || isWindowsDrivePath(ws) {
		var abs string
		if isWindowsDrivePath(trimmed) {
			abs = normalizeWindowsPath(trimmed)
		} else if trimmed == "" || trimmed == "." {
			abs = normalizeWindowsPath(ws)
		} else {
			rel := strings.ReplaceAll(trimmed, `\`, "/")
			abs = normalizeWindowsPath(ws + "/" + strings.TrimLeft(rel, "/"))
		}
		return "file://" + (&url.URL{Path: "/" + abs}).EscapedPath()
	}
	var abs string
	switch {
	case trimmed == "" || trimmed == ".":
		abs = path.Clean(ws)
	case strings.HasPrefix(trimmed, "/"):
		abs = path.Clean(trimmed)
	default:
		abs = path.Join(ws, trimmed)
	}
	if !strings.HasPrefix(abs, "/") {
		abs = "/" + abs
	}
	return "file://" + (&url.URL{Path: abs}).EscapedPath()
}

func parentFileURI(fileURI string) string {
	u, err := url.Parse(fileURI)
	if err != nil || u.Scheme != "file" {
		return ""
	}
	cleaned := strings.TrimRight(u.Path, "/")
	if cleaned == "" {
		return ""
	}
	dir := path.Dir(cleaned)
	if dir == "." || dir == "/" || dir == cleaned {
		return ""
	}
	// Preserve Windows drive root anchor (e.g. "/C:").
	if len(dir) == 3 && dir[0] == '/' && isASCIIAlpha(dir[1]) && dir[2] == ':' {
		return ""
	}
	return "file://" + u.Host + (&url.URL{Path: dir}).EscapedPath()
}

func displayDirPath(dirPath string) string {
	trimmed := strings.TrimSpace(dirPath)
	if strings.HasPrefix(trimmed, "file://") {
		if u, err := url.Parse(trimmed); err == nil && u.Path != "" {
			return u.Path
		}
	}
	return trimmed
}

func isWindowsDrivePath(s string) bool {
	if len(s) < 2 || !isASCIIAlpha(s[0]) || s[1] != ':' {
		return false
	}
	return len(s) == 2 || s[2] == '/' || s[2] == '\\'
}

func normalizeWindowsPath(s string) string {
	slashed := strings.ReplaceAll(strings.TrimSpace(s), `\`, "/")
	drive := strings.ToUpper(slashed[:2])
	rest := path.Clean("/" + strings.TrimPrefix(slashed[2:], "/"))
	if rest == "/" {
		return drive + "/"
	}
	return drive + rest
}

func isASCIIAlpha(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}
