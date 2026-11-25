package execserver

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/executor"
)

const maxSymlinkHops = 40

type fsService struct {
	workspaceDir        string
	resolvedWorkspace   string
	restrictToWorkspace bool
	maxReadBytes        int
	maxWriteBytes       int
}

func newFSService(workspaceDir string, restrict bool, maxReadBytes, maxWriteBytes int) (*fsService, error) {
	if strings.TrimSpace(workspaceDir) == "" {
		workspaceDir = "."
	}
	abs, err := filepath.Abs(workspaceDir)
	if err != nil {
		return nil, fmt.Errorf("resolve workspace dir: %w", err)
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return nil, fmt.Errorf("create workspace dir: %w", err)
	}
	resolvedWS := abs
	if eval, err := filepath.EvalSymlinks(abs); err == nil {
		resolvedWS = eval
	}
	if maxReadBytes <= 0 {
		maxReadBytes = DefaultMaxReadFileBytes
	}
	if maxWriteBytes <= 0 {
		maxWriteBytes = DefaultMaxWriteFileBytes
	}
	return &fsService{
		workspaceDir:        abs,
		resolvedWorkspace:   resolvedWS,
		restrictToWorkspace: restrict,
		maxReadBytes:        maxReadBytes,
		maxWriteBytes:       maxWriteBytes,
	}, nil
}

func (f *fsService) resolvePath(raw string) (string, *rpcErrorBody) {
	if strings.ContainsRune(raw, 0) {
		return "", &rpcErrorBody{
			Code:    ErrCodeInvalidParams,
			Message: "path must not contain NUL bytes",
		}
	}
	trimmed := strings.TrimSpace(raw)
	var candidate string
	switch {
	case trimmed == "" || trimmed == ".":
		candidate = f.workspaceDir
	case filepath.IsAbs(trimmed):
		candidate = filepath.Clean(trimmed)
	default:
		candidate = filepath.Clean(filepath.Join(f.workspaceDir, trimmed))
	}

	if !f.restrictToWorkspace {
		return candidate, nil
	}

	rel, ok := f.relativeToWorkspace(candidate)
	if !ok {
		return "", &rpcErrorBody{
			Code:    ErrCodePathEscapes,
			Message: fmt.Sprintf("path %q escapes workspace root", raw),
		}
	}
	resolved, errBody := f.resolveWithinWorkspace(rel, raw)
	if errBody != nil {
		return "", errBody
	}
	return resolved, nil
}

func (f *fsService) relativeToWorkspace(candidate string) (string, bool) {
	if isWithinRoot(f.workspaceDir, candidate) {
		rel, err := filepath.Rel(f.workspaceDir, candidate)
		if err == nil {
			return rel, true
		}
	}
	if f.resolvedWorkspace != f.workspaceDir && isWithinRoot(f.resolvedWorkspace, candidate) {
		rel, err := filepath.Rel(f.resolvedWorkspace, candidate)
		if err == nil {
			return rel, true
		}
	}
	return "", false
}

// resolveWithinWorkspace walks path components from resolvedWorkspace and
// resolves both existing and dangling symlinks so neither leaf nor intermediate
// symlinks can escape the workspace root.
func (f *fsService) resolveWithinWorkspace(rel, raw string) (string, *rpcErrorBody) {
	if rel == "." || rel == "" {
		return f.resolvedWorkspace, nil
	}
	parts := strings.Split(filepath.ToSlash(rel), "/")
	cur := f.resolvedWorkspace
	hops := 0

	for i := 0; i < len(parts); i++ {
		part := parts[i]
		if part == "" || part == "." {
			continue
		}
		if part == ".." {
			parent := filepath.Dir(cur)
			if !isWithinRoot(f.resolvedWorkspace, parent) {
				return "", &rpcErrorBody{
					Code:    ErrCodePathEscapes,
					Message: fmt.Sprintf("path %q resolves outside workspace root", raw),
				}
			}
			cur = parent
			continue
		}

		next := filepath.Join(cur, part)
		info, err := os.Lstat(next)
		if errors.Is(err, fs.ErrNotExist) {
			remaining := make([]string, 0, len(parts)-i)
			remaining = append(remaining, part)
			remaining = append(remaining, parts[i+1:]...)
			finalPath := filepath.Clean(filepath.Join(append([]string{cur}, remaining...)...))
			if !isWithinRoot(f.resolvedWorkspace, finalPath) {
				return "", &rpcErrorBody{
					Code:    ErrCodePathEscapes,
					Message: fmt.Sprintf("path %q resolves outside workspace root", raw),
				}
			}
			return finalPath, nil
		}
		if err != nil {
			return "", mapFSError(err)
		}

		if info.Mode()&os.ModeSymlink != 0 {
			hops++
			if hops > maxSymlinkHops {
				return "", &rpcErrorBody{
					Code:    ErrCodeInvalidParams,
					Message: fmt.Sprintf("too many symlink hops resolving %q", raw),
				}
			}
			target, err := os.Readlink(next)
			if err != nil {
				return "", mapFSError(err)
			}
			var linked string
			if filepath.IsAbs(target) {
				linked = filepath.Clean(target)
			} else {
				linked = filepath.Clean(filepath.Join(cur, target))
			}
			linkedRel, ok := f.relativeToWorkspace(linked)
			if !ok {
				return "", &rpcErrorBody{
					Code:    ErrCodePathEscapes,
					Message: fmt.Sprintf("path %q resolves outside workspace root", raw),
				}
			}
			rest := parts[i+1:]
			nextParts := make([]string, 0, len(rest)+4)
			if linkedRel != "." && linkedRel != "" {
				nextParts = append(nextParts, strings.Split(filepath.ToSlash(linkedRel), "/")...)
			}
			nextParts = append(nextParts, rest...)
			parts = nextParts
			cur = f.resolvedWorkspace
			i = -1
			continue
		}

		cur = next
	}

	if !isWithinRoot(f.resolvedWorkspace, cur) {
		return "", &rpcErrorBody{
			Code:    ErrCodePathEscapes,
			Message: fmt.Sprintf("path %q resolves outside workspace root", raw),
		}
	}
	return cur, nil
}

func isWithinRoot(root, target string) bool {
	rel, err := filepath.Rel(root, target)
	if err != nil {
		return false
	}
	if rel == "." {
		return true
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return false
	}
	return true
}

func (f *fsService) isWorkspaceRoot(resolved string) bool {
	return resolved == "/" || resolved == f.workspaceDir || resolved == f.resolvedWorkspace
}

func (f *fsService) displayPath(absPath string) string {
	if rel, ok := f.relativeToWorkspace(absPath); ok {
		return filepath.ToSlash(rel)
	}
	return filepath.ToSlash(absPath)
}

func (f *fsService) readFile(rawPath string, reqMaxBytes int) ([]byte, *rpcErrorBody) {
	if strings.TrimSpace(rawPath) == "" {
		return nil, &rpcErrorBody{
			Code:    ErrCodeInvalidParams,
			Message: "path is required",
		}
	}
	resolved, errBody := f.resolvePath(rawPath)
	if errBody != nil {
		return nil, errBody
	}
	limit := f.maxReadBytes
	if reqMaxBytes > 0 && reqMaxBytes < limit {
		limit = reqMaxBytes
	}

	file, err := os.Open(resolved)
	if err != nil {
		return nil, mapFSError(err)
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return nil, mapFSError(err)
	}
	if info.IsDir() {
		return nil, &rpcErrorBody{
			Code:    ErrCodeInvalidParams,
			Message: fmt.Sprintf("%q is a directory", rawPath),
		}
	}
	if info.Size() > int64(limit) {
		return nil, &rpcErrorBody{
			Code:    ErrCodeTooLarge,
			Message: fmt.Sprintf("file size %d exceeds maximum read limit of %d bytes", info.Size(), limit),
		}
	}

	limited := io.LimitReader(file, int64(limit)+1)
	data, err := io.ReadAll(limited)
	if err != nil {
		return nil, mapFSError(err)
	}
	if len(data) > limit {
		return nil, &rpcErrorBody{
			Code:    ErrCodeTooLarge,
			Message: fmt.Sprintf("file exceeds maximum read limit of %d bytes", limit),
		}
	}
	return data, nil
}

func (f *fsService) writeFile(rawPath string, data []byte, mode uint32, createParents bool) *rpcErrorBody {
	if strings.TrimSpace(rawPath) == "" {
		return &rpcErrorBody{
			Code:    ErrCodeInvalidParams,
			Message: "path is required",
		}
	}
	if len(data) > f.maxWriteBytes {
		return &rpcErrorBody{
			Code:    ErrCodeTooLarge,
			Message: fmt.Sprintf("write payload %d exceeds maximum write limit of %d bytes", len(data), f.maxWriteBytes),
		}
	}
	resolved, errBody := f.resolvePath(rawPath)
	if errBody != nil {
		return errBody
	}
	if f.isWorkspaceRoot(resolved) {
		return &rpcErrorBody{
			Code:    ErrCodeInvalidParams,
			Message: fmt.Sprintf("cannot write file over directory root %q", rawPath),
		}
	}
	dir := filepath.Dir(resolved)
	if createParents {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return mapFSError(err)
		}
		if f.restrictToWorkspace {
			if _, errBody := f.resolvePath(rawPath); errBody != nil {
				return errBody
			}
		}
	}
	perm := os.FileMode(0o644)
	if mode != 0 {
		perm = os.FileMode(mode) & os.ModePerm
	}

	if err := writeFileAtomic(resolved, data, perm); err != nil {
		return mapFSError(err)
	}
	return nil
}

func writeFileAtomic(target string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(target)
	tmp, err := os.CreateTemp(dir, ".jaz-exec-write-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	cleanup := true
	defer func() {
		if cleanup {
			_ = os.Remove(tmpName)
		}
	}()

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Chmod(perm); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, target); err != nil {
		return err
	}
	cleanup = false
	return nil
}

func (f *fsService) listDir(rawPath string) ([]executor.FileEntry, *rpcErrorBody) {
	resolved, errBody := f.resolvePath(rawPath)
	if errBody != nil {
		return nil, errBody
	}
	entries, err := os.ReadDir(resolved)
	if err != nil {
		return nil, mapFSError(err)
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Name() < entries[j].Name()
	})
	out := make([]executor.FileEntry, 0, len(entries))
	for _, entry := range entries {
		var size int64
		if info, err := entry.Info(); err == nil && info != nil {
			size = info.Size()
		}
		full := filepath.Join(resolved, entry.Name())
		out = append(out, executor.FileEntry{
			Name:  entry.Name(),
			Path:  f.displayPath(full),
			IsDir: entry.IsDir(),
			Size:  size,
		})
	}
	return out, nil
}

func (f *fsService) stat(rawPath string) (executor.FileMetadata, *rpcErrorBody) {
	if strings.TrimSpace(rawPath) == "" {
		return executor.FileMetadata{}, &rpcErrorBody{
			Code:    ErrCodeInvalidParams,
			Message: "path is required",
		}
	}
	resolved, errBody := f.resolvePath(rawPath)
	if errBody != nil {
		return executor.FileMetadata{}, errBody
	}
	linfo, err := os.Lstat(resolved)
	if err != nil {
		return executor.FileMetadata{}, mapFSError(err)
	}
	isSymlink := linfo.Mode()&os.ModeSymlink != 0
	info := linfo
	if isSymlink {
		if targetInfo, statErr := os.Stat(resolved); statErr == nil {
			info = targetInfo
		}
	}
	modMS := info.ModTime().UnixMilli()
	return executor.FileMetadata{
		IsDirectory:  info.IsDir(),
		IsFile:       info.Mode().IsRegular(),
		IsSymlink:    isSymlink,
		Size:         uint64(max(info.Size(), 0)),
		CreatedAtMS:  modMS,
		ModifiedAtMS: modMS,
	}, nil
}

func (f *fsService) mkdirAll(rawPath string, mode uint32) *rpcErrorBody {
	if strings.TrimSpace(rawPath) == "" {
		return &rpcErrorBody{
			Code:    ErrCodeInvalidParams,
			Message: "path is required",
		}
	}
	resolved, errBody := f.resolvePath(rawPath)
	if errBody != nil {
		return errBody
	}
	perm := os.FileMode(0o755)
	if mode != 0 {
		perm = os.FileMode(mode) & os.ModePerm
	}
	if err := os.MkdirAll(resolved, perm); err != nil {
		return mapFSError(err)
	}
	return nil
}

func (f *fsService) remove(rawPath string, recursive bool) *rpcErrorBody {
	if strings.TrimSpace(rawPath) == "" {
		return &rpcErrorBody{
			Code:    ErrCodeInvalidParams,
			Message: "path is required",
		}
	}
	resolved, errBody := f.resolvePath(rawPath)
	if errBody != nil {
		return errBody
	}
	if f.isWorkspaceRoot(resolved) {
		return &rpcErrorBody{
			Code:    ErrCodePermission,
			Message: fmt.Sprintf("refusing to remove root path %q", rawPath),
		}
	}
	if recursive {
		if _, err := os.Lstat(resolved); err != nil {
			return mapFSError(err)
		}
		if err := os.RemoveAll(resolved); err != nil {
			return mapFSError(err)
		}
		return nil
	}
	if err := os.Remove(resolved); err != nil {
		return mapFSError(err)
	}
	return nil
}

func (f *fsService) rename(oldRaw, newRaw string, createParents bool) *rpcErrorBody {
	if strings.TrimSpace(oldRaw) == "" || strings.TrimSpace(newRaw) == "" {
		return &rpcErrorBody{
			Code:    ErrCodeInvalidParams,
			Message: "oldPath and newPath are required",
		}
	}
	oldResolved, errBody := f.resolvePath(oldRaw)
	if errBody != nil {
		return errBody
	}
	newResolved, errBody := f.resolvePath(newRaw)
	if errBody != nil {
		return errBody
	}
	if f.isWorkspaceRoot(oldResolved) || f.isWorkspaceRoot(newResolved) {
		return &rpcErrorBody{
			Code:    ErrCodePermission,
			Message: "refusing to rename workspace or filesystem root",
		}
	}
	if createParents {
		if err := os.MkdirAll(filepath.Dir(newResolved), 0o755); err != nil {
			return mapFSError(err)
		}
		if f.restrictToWorkspace {
			if _, errBody := f.resolvePath(newRaw); errBody != nil {
				return errBody
			}
		}
	}
	if err := os.Rename(oldResolved, newResolved); err != nil {
		return mapFSError(err)
	}
	return nil
}

func mapFSError(err error) *rpcErrorBody {
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return &rpcErrorBody{Code: ErrCodeNotFound, Message: err.Error()}
	case errors.Is(err, fs.ErrPermission):
		return &rpcErrorBody{Code: ErrCodePermission, Message: err.Error()}
	case errors.Is(err, fs.ErrExist):
		return &rpcErrorBody{Code: ErrCodeAlreadyExists, Message: err.Error()}
	default:
		return &rpcErrorBody{Code: ErrCodeInternal, Message: err.Error()}
	}
}
