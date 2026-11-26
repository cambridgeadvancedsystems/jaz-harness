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
	DefaultImageModel = "gpt-image-2.6-sunburst"
	// DefaultCodexImageModel matches Codex's upstream image-generation extension
	// (`codex-rs/ext/image-generation/src/tool.rs` `IMAGE_MODEL`).
	DefaultCodexImageModel = "gpt-image-2"
)

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
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		return provider.ImageGenerationResponse{}, errors.New("openai image generation: prompt is required")
	}
	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = c.DefaultImageModel()
	}
	background := resolveImageBackground(req)
	outputFormat := resolveImageOutputFormat(model, background, req.OutputFormat, c.isCodexOAuth())

	var (
		endpoint    string
		body        []byte
		contentType string
		err         error
	)
	if len(req.ReferenceImages) == 0 {
		endpoint = c.cfg.BaseURL + "/images/generations"
		contentType = "application/json"
		body, err = buildImageGenerateBody(model, prompt, background, outputFormat, req)
	} else {
		endpoint = c.cfg.BaseURL + "/images/edits"
		if c.isCodexOAuth() || !allHaveInlineImageBytes(req.ReferenceImages) {
			contentType = "application/json"
			body, err = buildImageEditJSONBody(model, prompt, background, outputFormat, req)
		} else {
			body, contentType, err = buildImageEditMultipartBody(model, prompt, background, outputFormat, req)
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
		Model:    model,
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
	norm := strings.ToLower(strings.TrimSpace(model))
	return norm != "gpt-image-2" && !strings.HasPrefix(norm, "gpt-image-2.") && !strings.HasPrefix(norm, "gpt-image-2-")
}

// resolveImageOutputFormat ensures transparent backgrounds on direct OpenAI
// GPT Image 2.5/2.6 requests specify an alpha-capable format (`png`) when unset.
func resolveImageOutputFormat(model, background, explicit string, codexOAuth bool) string {
	if fmtVal := strings.ToLower(strings.TrimSpace(explicit)); fmtVal != "" {
		return fmtVal
	}
	if !codexOAuth && background == "transparent" {
		norm := strings.ToLower(strings.TrimSpace(model))
		if strings.HasPrefix(norm, "gpt-image-2.") {
			return "png"
		}
	}
	return ""
}

func buildImageGenerateBody(model, prompt, background, outputFormat string, req provider.ImageGenerationRequest) ([]byte, error) {
	payload := map[string]any{
		"model":  model,
		"prompt": prompt,
	}
	if background != "" {
		payload["background"] = background
	}
	if req.Quality != "" {
		payload["quality"] = strings.ToLower(strings.TrimSpace(req.Quality))
	}
	if req.Size != "" {
		payload["size"] = strings.TrimSpace(req.Size)
	}
	if outputFormat != "" {
		payload["output_format"] = outputFormat
	}
	return json.Marshal(payload)
}

func buildImageEditJSONBody(model, prompt, background, outputFormat string, req provider.ImageGenerationRequest) ([]byte, error) {
	images := make([]map[string]any, 0, len(req.ReferenceImages))
	for _, ref := range req.ReferenceImages {
		if u := ref.DataURL(); u != "" {
			images = append(images, map[string]any{"image_url": u})
		}
	}
	if len(images) == 0 {
		return nil, errors.New("openai image edit: no valid reference images")
	}
	payload := map[string]any{
		"model":  model,
		"prompt": prompt,
		"images": images,
	}
	if background != "" {
		payload["background"] = background
	}
	if req.Quality != "" {
		payload["quality"] = strings.ToLower(strings.TrimSpace(req.Quality))
	}
	if req.Size != "" {
		payload["size"] = strings.TrimSpace(req.Size)
	}
	if outputFormat != "" {
		payload["output_format"] = outputFormat
	}
	if req.InputFidelity != "" && supportsInputFidelity(model) {
		payload["input_fidelity"] = strings.ToLower(strings.TrimSpace(req.InputFidelity))
	}
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

func buildImageEditMultipartBody(model, prompt, background, outputFormat string, req provider.ImageGenerationRequest) ([]byte, string, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	writeField := func(name, val string) error {
		if val == "" {
			return nil
		}
		return mw.WriteField(name, val)
	}
	inputFidelity := ""
	if supportsInputFidelity(model) {
		inputFidelity = strings.ToLower(strings.TrimSpace(req.InputFidelity))
	}
	for _, kv := range [][2]string{
		{"model", model},
		{"prompt", prompt},
		{"background", background},
		{"quality", strings.ToLower(strings.TrimSpace(req.Quality))},
		{"size", strings.TrimSpace(req.Size)},
		{"output_format", outputFormat},
		{"input_fidelity", inputFidelity},
	} {
		if err := writeField(kv[0], kv[1]); err != nil {
			return nil, "", err
		}
	}
	fieldName := "image[]"
	if len(req.ReferenceImages) == 1 {
		fieldName = "image"
	}
	attached := 0
	for i, ref := range req.ReferenceImages {
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
