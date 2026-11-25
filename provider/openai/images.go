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

const DefaultImageModel = "gpt-image-1.5"

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
		model = DefaultImageModel
	}
	background := resolveImageBackground(req)

	var (
		endpoint    string
		body        []byte
		contentType string
		err         error
	)
	if len(req.ReferenceImages) == 0 {
		endpoint = c.cfg.BaseURL + "/images/generations"
		contentType = "application/json"
		body, err = buildImageGenerateBody(model, prompt, background, req)
	} else {
		endpoint = c.cfg.BaseURL + "/images/edits"
		if c.isCodexOAuth() || !hasBase64Images(req.ReferenceImages) {
			contentType = "application/json"
			body, err = buildImageEditJSONBody(model, prompt, background, req)
		} else {
			body, contentType, err = buildImageEditMultipartBody(model, prompt, background, req)
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

	raw, err := io.ReadAll(io.LimitReader(resp.Body, provider.DefaultMaxBodyBytes))
	if err != nil {
		return provider.ImageGenerationResponse{}, err
	}
	if resp.StatusCode >= 300 {
		return provider.ImageGenerationResponse{}, &provider.APIError{
			Provider:   c.ID(),
			StatusCode: resp.StatusCode,
			Body:       strings.TrimSpace(string(raw)),
			RetryAfter: provider.ParseRetryAfter(resp.Header),
		}
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

func buildImageGenerateBody(model, prompt, background string, req provider.ImageGenerationRequest) ([]byte, error) {
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
	if req.OutputFormat != "" {
		payload["output_format"] = strings.ToLower(strings.TrimSpace(req.OutputFormat))
	}
	return json.Marshal(payload)
}

func buildImageEditJSONBody(model, prompt, background string, req provider.ImageGenerationRequest) ([]byte, error) {
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
	if req.InputFidelity != "" {
		payload["input_fidelity"] = strings.ToLower(strings.TrimSpace(req.InputFidelity))
	}
	return json.Marshal(payload)
}

func hasBase64Images(refs []schema.ImageSource) bool {
	for _, r := range refs {
		if strings.TrimSpace(r.Base64) != "" {
			return true
		}
	}
	return false
}

func buildImageEditMultipartBody(model, prompt, background string, req provider.ImageGenerationRequest) ([]byte, string, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	_ = mw.WriteField("model", model)
	_ = mw.WriteField("prompt", prompt)
	if background != "" {
		_ = mw.WriteField("background", background)
	}
	if req.Quality != "" {
		_ = mw.WriteField("quality", strings.ToLower(strings.TrimSpace(req.Quality)))
	}
	if req.Size != "" {
		_ = mw.WriteField("size", strings.TrimSpace(req.Size))
	}
	if req.InputFidelity != "" {
		_ = mw.WriteField("input_fidelity", strings.ToLower(strings.TrimSpace(req.InputFidelity)))
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
		ext := extensionForMediaType(mt)
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

func extensionForMediaType(mt string) string {
	switch strings.ToLower(strings.TrimSpace(mt)) {
	case "image/jpeg", "image/jpg":
		return "jpg"
	case "image/webp":
		return "webp"
	case "image/gif":
		return "gif"
	default:
		return "png"
	}
}
