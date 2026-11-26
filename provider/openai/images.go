package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/textproto"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
)

const (
	// DefaultImageModel tracks OpenAI's latest official Images API flagship model
	// (GPT Image 2.6 Sunburst: high detail, typography, transparent backgrounds, and multi-reference consistency).
	DefaultImageModel = schema.DefaultOpenAIImageModel
	// DefaultCodexImageModel matches Codex's upstream image-generation extension
	// (`codex-rs/ext/image-generation/src/tool.rs` `IMAGE_MODEL`).
	DefaultCodexImageModel = schema.DefaultCodexImageModel
)

type normalizedImageParams struct {
	model         string
	prompt        string
	background    string
	quality       string
	size          string
	outputFormat  string
	inputFidelity string
}

// DefaultImageModel returns the active client's default image generation model
// (`gpt-image-2` on Codex OAuth to match upstream Codex, `gpt-image-2.6-sunburst`
// on direct OpenAI / Azure OpenAI).
func (c *Client) DefaultImageModel() string {
	if c.isCodexOAuth() {
		return DefaultCodexImageModel
	}
	return DefaultImageModel
}

// GenerateImage implements provider.ImageGenerator using OpenAI / Codex native
// image endpoints (`POST /images/generations` and `POST /images/edits`).
// Reference images use Codex's JSON `{images: [{image_url}]}` contract on
// Codex OAuth and multipart `image[]` (or JSON when URL-only) on direct OpenAI.
func (c *Client) GenerateImage(ctx context.Context, req provider.ImageGenerationRequest) (provider.ImageGenerationResponse, error) {
	params, err := c.prepareImageParams(req)
	if err != nil {
		return provider.ImageGenerationResponse{}, err
	}

	var (
		endpoint    string
		body        []byte
		contentType string
	)
	if len(req.ReferenceImages) == 0 {
		endpoint = c.cfg.BaseURL + "/images/generations"
		contentType = "application/json"
		body, err = buildImageGenerateBody(params)
	} else {
		endpoint = c.cfg.BaseURL + "/images/edits"
		if c.isCodexOAuth() || !allHaveInlineImageBytes(req.ReferenceImages) {
			contentType = "application/json"
			body, err = buildImageEditJSONBody(params, req.ReferenceImages)
		} else {
			body, contentType, err = buildImageEditMultipartBody(params, req.ReferenceImages)
		}
	}
	if err != nil {
		return provider.ImageGenerationResponse{}, err
	}

	resp, err := c.doAuthenticatedWithContentType(ctx, endpoint, body, contentType, "application/json", "")
	if err != nil {
		return provider.ImageGenerationResponse{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		rawErr, _ := io.ReadAll(io.LimitReader(resp.Body, provider.DefaultMaxErrorBodyBytes))
		return provider.ImageGenerationResponse{}, &provider.APIError{
			Provider:   c.ID(),
			StatusCode: resp.StatusCode,
			Body:       strings.TrimSpace(string(rawErr)),
			RetryAfter: provider.ParseRetryAfter(resp.Header),
		}
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, provider.DefaultMaxBodyBytes))
	if err != nil {
		return provider.ImageGenerationResponse{}, err
	}

	var parsed struct {
		Created    int64  `json:"created"`
		Background string `json:"background"`
		Quality    string `json:"quality"`
		Size       string `json:"size"`
		Data       []struct {
			B64JSON       string         `json:"b64_json"`
			URL           string         `json:"url"`
			RevisedPrompt string         `json:"revised_prompt"`
			GenerationID  string         `json:"generation_id"`
			ImageMetadata map[string]any `json:"image_metadata"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return provider.ImageGenerationResponse{}, fmt.Errorf("openai image decode: %w", err)
	}
	if len(parsed.Data) == 0 {
		return provider.ImageGenerationResponse{}, errors.New("openai image generation returned no images")
	}

	out := provider.ImageGenerationResponse{
		Model:    params.model,
		Metadata: map[string]any{},
	}
	if reqID := strings.TrimSpace(resp.Header.Get("x-codex-imagegen-request-id")); reqID != "" {
		out.Metadata["imagegen_request_id"] = reqID
	}
	if parsed.Size != "" {
		out.Metadata["size"] = parsed.Size
	}
	if parsed.Quality != "" {
		out.Metadata["quality"] = parsed.Quality
	}
	if parsed.Background != "" {
		out.Metadata["background"] = parsed.Background
	}

	for _, item := range parsed.Data {
		if out.ID == "" && item.GenerationID != "" {
			out.ID = item.GenerationID
		}
		if out.RevisedPrompt == "" && item.RevisedPrompt != "" {
			out.RevisedPrompt = item.RevisedPrompt
		}
		if strings.TrimSpace(item.B64JSON) != "" {
			b64 := strings.TrimSpace(item.B64JSON)
			out.Images = append(out.Images, schema.ImageSource{
				MediaType: schema.DetectBase64ImageMediaType(b64),
				Base64:    b64,
			})
		} else if strings.TrimSpace(item.URL) != "" {
			out.Images = append(out.Images, schema.ImageSource{
				URL: strings.TrimSpace(item.URL),
			})
		}
	}
	if len(out.Images) == 0 {
		return provider.ImageGenerationResponse{}, errors.New("openai image generation returned empty image payloads")
	}
	return out, nil
}

func (c *Client) prepareImageParams(req provider.ImageGenerationRequest) (normalizedImageParams, error) {
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		return normalizedImageParams{}, errors.New("openai image generation: prompt is required")
	}
	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = c.DefaultImageModel()
	}
	background := resolveImageBackground(req)
	outputFormat, err := resolveImageOutputFormat(background, req.OutputFormat, c.isCodexOAuth())
	if err != nil {
		return normalizedImageParams{}, err
	}
	inputFidelity := ""
	if supportsInputFidelity(model) {
		inputFidelity = strings.ToLower(strings.TrimSpace(req.InputFidelity))
	}
	return normalizedImageParams{
		model:         model,
		prompt:        prompt,
		background:    background,
		quality:       strings.ToLower(strings.TrimSpace(req.Quality)),
		size:          strings.TrimSpace(req.Size),
		outputFormat:  outputFormat,
		inputFidelity: inputFidelity,
	}, nil
}

func resolveImageBackground(req provider.ImageGenerationRequest) string {
	if bg := strings.ToLower(strings.TrimSpace(req.Background)); bg != "" {
		return bg
	}
	if req.TransparentBackground {
		return "transparent"
	}
	return ""
}

// supportsInputFidelity reports whether model accepts the legacy `input_fidelity`
// parameter. OpenAI GPT Image 2 / 2.5 / 2.6 models always preserve reference
// images at high fidelity and deprecate/reject `input_fidelity` (matching Codex).
func supportsInputFidelity(model string) bool {
	norm := schema.NormalizeModelSlug(model)
	return norm != "gpt-image-2" && !strings.HasPrefix(norm, "gpt-image-2.") && !strings.HasPrefix(norm, "gpt-image-2-")
}

// resolveImageOutputFormat validates transparent-background format compatibility
// and defaults unset direct OpenAI/Azure transparent requests to alpha-capable `png`.
// Codex OAuth omits `output_format` unless explicitly requested (matching codex-api).
func resolveImageOutputFormat(background, explicit string, codexOAuth bool) (string, error) {
	fmtVal := strings.ToLower(strings.TrimSpace(explicit))
	if background == "transparent" && (fmtVal == "jpeg" || fmtVal == "jpg") {
		return "", errors.New("openai image generation: transparent background requires an alpha-capable output_format (png or webp)")
	}
	if fmtVal != "" {
		return fmtVal, nil
	}
	if !codexOAuth && background == "transparent" {
		return "png", nil
	}
	return "", nil
}

func applyOptionalImageFields(payload map[string]any, p normalizedImageParams, includeInputFidelity bool) {
	if p.background != "" {
		payload["background"] = p.background
	}
	if p.quality != "" {
		payload["quality"] = p.quality
	}
	if p.size != "" {
		payload["size"] = p.size
	}
	if p.outputFormat != "" {
		payload["output_format"] = p.outputFormat
	}
	if includeInputFidelity && p.inputFidelity != "" {
		payload["input_fidelity"] = p.inputFidelity
	}
}

func buildImageGenerateBody(p normalizedImageParams) ([]byte, error) {
	payload := map[string]any{
		"model":  p.model,
		"prompt": p.prompt,
	}
	applyOptionalImageFields(payload, p, false)
	return json.Marshal(payload)
}

func buildImageEditJSONBody(p normalizedImageParams, refs []schema.ImageSource) ([]byte, error) {
	images := make([]map[string]any, 0, len(refs))
	for _, ref := range refs {
		if u := ref.DataURL(); u != "" {
			images = append(images, map[string]any{"image_url": u})
		}
	}
	if len(images) == 0 {
		return nil, errors.New("openai image edit: no valid reference images")
	}
	payload := map[string]any{
		"model":  p.model,
		"prompt": p.prompt,
		"images": images,
	}
	applyOptionalImageFields(payload, p, true)
	return json.Marshal(payload)
}

func allHaveInlineImageBytes(refs []schema.ImageSource) bool {
	if len(refs) == 0 {
		return false
	}
	for _, r := range refs {
		if strings.TrimSpace(r.Base64) == "" && !strings.HasPrefix(r.URL, "data:") {
			return false
		}
	}
	return true
}

func buildImageEditMultipartBody(p normalizedImageParams, refs []schema.ImageSource) ([]byte, string, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	for _, kv := range [][2]string{
		{"model", p.model},
		{"prompt", p.prompt},
		{"background", p.background},
		{"quality", p.quality},
		{"size", p.size},
		{"output_format", p.outputFormat},
		{"input_fidelity", p.inputFidelity},
	} {
		if kv[1] == "" {
			continue
		}
		if err := mw.WriteField(kv[0], kv[1]); err != nil {
			return nil, "", err
		}
	}
	fieldName := "image[]"
	if len(refs) == 1 {
		fieldName = "image"
	}
	attached := 0
	for i, ref := range refs {
		raw, mt, err := decodeReferenceImageBytes(ref)
		if err != nil {
			return nil, "", fmt.Errorf("openai image edit reference %d: %w", i, err)
		}
		ext := schema.ImageExtension(mt)
		hdr := make(textproto.MIMEHeader)
		hdr.Set("Content-Disposition", fmt.Sprintf(`form-data; name="%s"; filename="reference_%d.%s"`, fieldName, i, ext))
		hdr.Set("Content-Type", mt)
		part, err := mw.CreatePart(hdr)
		if err != nil {
			return nil, "", err
		}
		if _, err := part.Write(raw); err != nil {
			return nil, "", err
		}
		attached++
	}
	if attached == 0 {
		return nil, "", errors.New("openai image edit: no decodable reference images")
	}
	if err := mw.Close(); err != nil {
		return nil, "", err
	}
	return buf.Bytes(), mw.FormDataContentType(), nil
}

func decodeReferenceImageBytes(ref schema.ImageSource) ([]byte, string, error) {
	b64 := strings.TrimSpace(ref.Base64)
	if b64 == "" && strings.HasPrefix(ref.URL, "data:") {
		if idx := strings.Index(ref.URL, ";base64,"); idx > 0 {
			b64 = ref.URL[idx+len(";base64,"):]
		}
	}
	if b64 == "" {
		return nil, "", errors.New("reference image requires base64 bytes for multipart upload")
	}
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, "", err
	}
	mt := strings.TrimSpace(ref.MediaType)
	if mt == "" {
		mt = schema.DetectImageMediaType(raw)
	}
	if mt == "" {
		mt = "image/png"
	}
	return raw, mt, nil
}
