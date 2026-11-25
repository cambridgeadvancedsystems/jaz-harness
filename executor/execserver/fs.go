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
	if !isWithinRoot(f.workspaceDir, candidate) {
		return "", &rpcErrorBody{
			Code:    ErrCodePathEscapes,
			Message: fmt.Sprintf("path %q escapes workspace root", raw),
		}
	}
	if errBody := f.checkSymlinkContainment(candidate, raw); errBody != nil {
		return "", errBody
	}
	return candidate, nil
}

func (f *fsService) checkSymlinkContainment(candidate, raw string) *rpcErrorBody {
	cur := candidate
	for {
		eval, err := filepath.EvalSymlinks(cur)
		if err == nil {
			if !isWithinRoot(f.resolvedWorkspace, eval) {
				return &rpcErrorBody{
					Code:    ErrCodePathEscapes,
					Message: fmt.Sprintf("path %q resolves outside workspace root", raw),
				}
			}
			return nil
		}
		if !errors.Is(err, fs.ErrNotExist) {
			return mapFSError(err)
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			return nil
		}
		cur = parent
	}
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

func (f *fsService) displayPath(absPath string) string {
	if isWithinRoot(f.workspaceDir, absPath) {
		rel, err := filepath.Rel(f.workspaceDir, absPath)
		if err == nil {
			return filepath.ToSlash(rel)
		}
	}
	return filepath.ToSlash(absPath)
}

func (f *fsService) readFile(rawPath string, reqMaxBytes int) ([]byte, *rpcErrorBody) {
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
	dir := filepath.Dir(resolved)
	if createParents {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return mapFSError(err)
		}
		if f.restrictToWorkspace {
			if errBody := f.checkSymlinkContainment(dir, rawPath); errBody != nil {
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
		return os.WriteFile(target, data, perm)
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
	if resolved == "/" || resolved == f.workspaceDir {
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
	if createParents {
		if err := os.MkdirAll(filepath.Dir(newResolved), 0o755); err != nil {
			return mapFSError(err)
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
