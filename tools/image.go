package tools

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"path"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
	"github.com/google/uuid"
)

const (
	DefaultMaxImageBytes = 20 << 20 // 20 MiB per loaded image ceiling
	MaxReferenceImages   = 5        // matches Codex MAX_EDIT_IMAGES
)

// FileReaderFunc reads raw bytes from a workspace or local filesystem path.
type FileReaderFunc func(ctx context.Context, filePath string) ([]byte, error)

// FileReaderWithPrefixFunc reads raw bytes and returns an optional leading
// workspace notice (such as <workspace_agents_md>) on first workspace access.
type FileReaderWithPrefixFunc func(ctx context.Context, filePath string) ([]byte, string, error)

// FileWriterFunc writes raw bytes to a workspace or local filesystem path.
type FileWriterFunc func(ctx context.Context, filePath string, data []byte) error

// ViewImageConfig configures the universal view_image tool.
type ViewImageConfig struct {
	ReadFile           FileReaderFunc
	ReadFileWithPrefix FileReaderWithPrefixFunc
	MaxImageBytes      int
	Category           string
}

// ImageGenerationConfig configures the universal generate_image tool.
type ImageGenerationConfig struct {
	ReadFile       FileReaderFunc
	WriteFile      FileWriterFunc
	Fallback       provider.ImageGenerator
	PreferFallback bool
	DefaultModel   string
	OutputDir      string // defaults to "generated_images" when WriteFile is configured
	MaxImageBytes  int
}

// ValidateImageBytes verifies that data is a well-formed PNG, JPEG, GIF, or
// WebP image within maxBytes and returns its normalized MIME type and dimensions.
func ValidateImageBytes(data []byte, maxBytes int) (mediaType string, width, height int, err error) {
	if len(data) == 0 {
		return "", 0, 0, errors.New("unable to process image: empty image file")
	}
	if maxBytes > 0 && len(data) > maxBytes {
		return "", 0, 0, fmt.Errorf("unable to process image: file size %d exceeds limit %d bytes", len(data), maxBytes)
	}
	mediaType = schema.DetectImageMediaType(data)
	if mediaType == "" {
		return "", 0, 0, errors.New("unable to process image: invalid or unsupported image data")
	}
	if mediaType == "image/webp" {
		if len(data) < 16 {
			return "", 0, 0, errors.New("unable to process image: invalid or unsupported image data")
		}
		chunk := string(data[12:16])
		if chunk != "VP8 " && chunk != "VP8L" && chunk != "VP8X" {
			return "", 0, 0, errors.New("unable to process image: invalid or unsupported image data")
		}
		return mediaType, 0, 0, nil
	}
	cfg, _, derr := image.DecodeConfig(bytes.NewReader(data))
	if derr != nil || cfg.Width <= 0 || cfg.Height <= 0 {
		return "", 0, 0, errors.New("unable to process image: invalid or unsupported image data")
	}
	return mediaType, cfg.Width, cfg.Height, nil
}

// RegisterViewImageTool registers the eager `view_image` tool matching Codex's
// reference semantics: reads a workspace/local image file, validates format and
// bounds, and injects the image as multimodal tool-result content across all
// providers without leaking raw base64 into logs.
func RegisterViewImageTool(reg *Registry, cfg ViewImageConfig) {
	maxBytes := cfg.MaxImageBytes
	if maxBytes <= 0 {
		maxBytes = DefaultMaxImageBytes
	}
	category := cfg.Category
	if category == "" {
		category = "image"
	}
	reg.RegisterRich(schema.ToolDef{
		Name:        "view_image",
		Description: "View a local or workspace image file when visual inspection is needed. Injects the image directly into model context.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string","description":"Workspace or local filesystem path to an image file."},"detail":{"type":"string","enum":["high","original"],"description":"Image detail level. Defaults to high; use original to preserve exact resolution when supported."}},"required":["path"]}`),
		AlwaysLoad:  true,
		Category:    category,
	}, func(ctx context.Context, args json.RawMessage) (Result, error) {
		var p struct {
			Path   string `json:"path"`
			Detail string `json:"detail"`
		}
		if err := json.Unmarshal(args, &p); err != nil {
			return Result{}, err
		}
		p.Path = strings.TrimSpace(p.Path)
		if p.Path == "" {
			return Result{}, errors.New("view_image: path is required")
		}
		detail := strings.ToLower(strings.TrimSpace(p.Detail))
		switch detail {
		case "":
			detail = "high"
		case "high", "original":
		default:
			return Result{}, fmt.Errorf("view_image.detail only supports `high` or `original`; omit `detail` for default high behavior, got `%s`", p.Detail)
		}

		var (
			raw    []byte
			prefix string
			err    error
		)
		switch {
		case cfg.ReadFileWithPrefix != nil:
			raw, prefix, err = cfg.ReadFileWithPrefix(ctx, p.Path)
		case cfg.ReadFile != nil:
			raw, err = cfg.ReadFile(ctx, p.Path)
		default:
			return Result{}, errors.New("view_image: filesystem reader is not configured")
		}
		if err != nil {
			return Result{}, fmt.Errorf("unable to read image at `%s`: %w", p.Path, err)
		}

		mediaType, width, height, err := ValidateImageBytes(raw, maxBytes)
		if err != nil {
			return Result{}, err
		}

		b64 := base64.StdEncoding.EncodeToString(raw)
		dimNote := ""
		if width > 0 && height > 0 {
			dimNote = fmt.Sprintf(", %dx%d", width, height)
		}
		text := fmt.Sprintf("%sLoaded image `%s` (%s, %d bytes%s, detail=%s).", prefix, p.Path, mediaType, len(raw), dimNote, detail)
		return Result{
			Text: text,
			Images: []schema.ImageSource{{
				MediaType: mediaType,
				Base64:    b64,
				Detail:    detail,
				Path:      p.Path,
			}},
			Metadata: map[string]any{
				"path":       p.Path,
				"media_type": mediaType,
				"bytes":      len(raw),
				"width":      width,
				"height":     height,
				"detail":     detail,
			},
		}, nil
	})
}

// RegisterImageGenerationTool registers the universal `generate_image` tool
// that works across all providers: uses the active provider's native
// `provider.ImageGenerator` when available (OpenAI/Codex, Gemini/Vertex, Grok)
// and seamlessly falls back to `cfg.Fallback` otherwise, with full support for
// workspace reference images (`referenced_image_paths`) and recent conversation
// images (`num_last_images_to_include`) matching Codex's reference contract.
func RegisterImageGenerationTool(reg *Registry, cfg ImageGenerationConfig) {
	maxBytes := cfg.MaxImageBytes
	if maxBytes <= 0 {
		maxBytes = DefaultMaxImageBytes
	}
	outDir := strings.TrimSpace(cfg.OutputDir)
	if outDir == "" {
		outDir = "generated_images"
	}

	reg.RegisterRich(schema.ToolDef{
		Name:        "generate_image",
		Description: "Generate a new image from a text description or edit existing images using reference inputs. Returns the generated image into model context and saves it to the workspace when available.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"prompt":{"type":"string","description":"Detailed prompt describing the image to generate or the edit to apply."},"referenced_image_paths":{"type":"array","items":{"type":"string"},"maxItems":5,"description":"Up to 5 workspace/local image paths to edit or use as visual references. Mutually exclusive with num_last_images_to_include."},"num_last_images_to_include":{"type":"integer","minimum":1,"maximum":5,"description":"Include the last N images from conversation history (uploads, view_image, or prior generated images) as reference inputs. Mutually exclusive with referenced_image_paths."},"transparent_background":{"type":"boolean","description":"Request a transparent background when supported."},"size":{"type":"string","description":"Requested output dimensions such as auto, 1024x1024, 1024x1536, or 1536x1024."},"quality":{"type":"string","enum":["auto","low","medium","high"],"description":"Image generation quality tier."},"input_fidelity":{"type":"string","enum":["low","high"],"description":"How strictly edits should preserve details from reference images."},"output_path":{"type":"string","description":"Optional workspace path where the generated image should be saved."}},"required":["prompt"]}`),
		AlwaysLoad:  true,
		Category:    "image",
	}, func(ctx context.Context, args json.RawMessage) (Result, error) {
		var p struct {
			Prompt                 string   `json:"prompt"`
			ReferencedImagePaths   []string `json:"referenced_image_paths"`
			NumLastImagesToInclude *int     `json:"num_last_images_to_include"`
			TransparentBackground  bool     `json:"transparent_background"`
			Size                   string   `json:"size"`
			Quality                string   `json:"quality"`
			InputFidelity          string   `json:"input_fidelity"`
			OutputPath             string   `json:"output_path"`
		}
		if err := json.Unmarshal(args, &p); err != nil {
			return Result{}, err
		}
		p.Prompt = strings.TrimSpace(p.Prompt)
		if p.Prompt == "" {
			return Result{}, errors.New("generate_image: prompt is required")
		}
		if len(p.ReferencedImagePaths) > 0 && p.NumLastImagesToInclude != nil {
			return Result{}, errors.New("provide only one of `referenced_image_paths` or `num_last_images_to_include`")
		}
		if len(p.ReferencedImagePaths) > MaxReferenceImages {
			return Result{}, fmt.Errorf("`referenced_image_paths` must contain at most %d paths", MaxReferenceImages)
		}

		var (
			refs       []schema.ImageSource
			inputMeta  []map[string]any
			background string
		)
		if p.TransparentBackground {
			background = "transparent"
		}

		if len(p.ReferencedImagePaths) > 0 {
			if cfg.ReadFile == nil {
				return Result{}, errors.New("referenced image paths are unavailable in this session")
			}
			for _, refPath := range p.ReferencedImagePaths {
				trimmed := strings.TrimSpace(refPath)
				if trimmed == "" {
					continue
				}
				raw, err := cfg.ReadFile(ctx, trimmed)
				if err != nil {
					return Result{}, fmt.Errorf("unable to read referenced image at `%s`: %w", trimmed, err)
				}
				mediaType, _, _, err := ValidateImageBytes(raw, maxBytes)
				if err != nil {
					return Result{}, fmt.Errorf("unable to process referenced image at `%s`: %w", trimmed, err)
				}
				sum := sha256.Sum256(raw)
				refs = append(refs, schema.ImageSource{
					MediaType: mediaType,
					Base64:    base64.StdEncoding.EncodeToString(raw),
					Path:      trimmed,
				})
				inputMeta = append(inputMeta, map[string]any{
					"path":   trimmed,
					"sha256": hex.EncodeToString(sum[:]),
				})
			}
		} else if p.NumLastImagesToInclude != nil {
			count := *p.NumLastImagesToInclude
			if count < 1 || count > MaxReferenceImages {
				return Result{}, fmt.Errorf("`num_last_images_to_include` must be between 1 and %d", MaxReferenceImages)
			}
			historyImgs := ConversationImagesFromContext(ctx)
			if len(historyImgs) < count {
				return Result{}, fmt.Errorf("requested the last %d conversation images, but only %d were available", count, len(historyImgs))
			}
			refs = append(refs, historyImgs[len(historyImgs)-count:]...)
		}

		gen := resolveImageBackend(ctx, cfg)
		if gen == nil {
			return Result{}, errors.New("no image generation backend is configured for this session")
		}

		genResp, err := gen.GenerateImage(ctx, provider.ImageGenerationRequest{
			Prompt:                p.Prompt,
			Model:                 cfg.DefaultModel,
			ReferenceImages:       refs,
			Size:                  strings.TrimSpace(p.Size),
			Quality:               strings.TrimSpace(p.Quality),
			Background:            background,
			InputFidelity:         strings.TrimSpace(p.InputFidelity),
			TransparentBackground: p.TransparentBackground,
		})
		if err != nil {
			return Result{}, fmt.Errorf("image generation failed: %w", err)
		}
		if len(genResp.Images) == 0 {
			return Result{}, errors.New("image generation returned no image data")
		}

		callID := ToolCallIDFromContext(ctx)
		if callID == "" {
			callID = "img_" + uuid.NewString()[:8]
		}
		var savedPaths []string
		for i := range genResp.Images {
			img := &genResp.Images[i]
			if img.Detail == "" {
				img.Detail = "high"
			}
			if cfg.WriteFile == nil || strings.TrimSpace(img.Base64) == "" {
				continue
			}
			decoded, derr := base64.StdEncoding.DecodeString(strings.TrimSpace(img.Base64))
			if derr != nil {
				continue
			}
			if img.MediaType == "" {
				img.MediaType = schema.DetectImageMediaType(decoded)
				if img.MediaType == "" {
					img.MediaType = "image/png"
				}
			}
			targetPath := buildGeneratedImagePath(outDir, strings.TrimSpace(p.OutputPath), callID, i, len(genResp.Images), img.MediaType)
			if werr := cfg.WriteFile(ctx, targetPath, decoded); werr == nil {
				img.Path = targetPath
				savedPaths = append(savedPaths, targetPath)
			}
		}

		summary := map[string]any{
			"status":                 "completed",
			"image_count":            len(genResp.Images),
			"reference_images_count": len(refs),
		}
		if genResp.Model != "" {
			summary["model"] = genResp.Model
		}
		if genResp.ID != "" {
			summary["generation_id"] = genResp.ID
		}
		if genResp.RevisedPrompt != "" {
			summary["revised_prompt"] = genResp.RevisedPrompt
		}
		if len(savedPaths) == 1 {
			summary["saved_path"] = savedPaths[0]
		} else if len(savedPaths) > 1 {
			summary["saved_paths"] = savedPaths
		}
		if len(inputMeta) > 0 {
			summary["input_images"] = inputMeta
		}
		rawSummary, _ := json.Marshal(summary)

		return Result{
			Text:     string(rawSummary),
			Images:   genResp.Images,
			Metadata: summary,
		}, nil
	})
}

func resolveImageBackend(ctx context.Context, cfg ImageGenerationConfig) provider.ImageGenerator {
	active := ActiveImageGeneratorFromContext(ctx)
	if cfg.PreferFallback {
		if cfg.Fallback != nil {
			return cfg.Fallback
		}
		return active
	}
	if active != nil {
		return active
	}
	return cfg.Fallback
}

func buildGeneratedImagePath(outDir, requestedPath, callID string, idx, total int, mediaType string) string {
	ext := ".png"
	switch strings.ToLower(strings.TrimSpace(mediaType)) {
	case "image/jpeg", "image/jpg":
		ext = ".jpg"
	case "image/webp":
		ext = ".webp"
	case "image/gif":
		ext = ".gif"
	}
	if requestedPath != "" {
		clean := path.Clean(strings.ReplaceAll(requestedPath, "\\", "/"))
		if total <= 1 {
			return clean
		}
		baseExt := path.Ext(clean)
		stem := strings.TrimSuffix(clean, baseExt)
		if baseExt == "" {
			baseExt = ext
		}
		return fmt.Sprintf("%s_%d%s", stem, idx+1, baseExt)
	}
	safeID := strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			return r
		}
		return '_'
	}, callID)
	if safeID == "" {
		safeID = "image"
	}
	if total <= 1 {
		return path.Join(outDir, safeID+ext)
	}
	return path.Join(outDir, fmt.Sprintf("%s_%d%s", safeID, idx+1, ext))
}
