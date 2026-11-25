package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
)

const DefaultImageModel = "gemini-2.5-flash-image"

// GenerateImage implements provider.ImageGenerator using Gemini's native
// multimodal `:generateContent` surface (`responseModalities: ["TEXT", "IMAGE"]`),
// supporting both text-to-image generation and reference-image editing across
// Vertex AI, AI Studio, and Relay modes.
func (c *Client) GenerateImage(ctx context.Context, req provider.ImageGenerationRequest) (provider.ImageGenerationResponse, error) {
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		return provider.ImageGenerationResponse{}, errors.New("gemini image generation: prompt is required")
	}
	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = DefaultImageModel
	}

	parts := make([]map[string]any, 0, len(req.ReferenceImages)+1)
	for _, ref := range req.ReferenceImages {
		if imgPart := buildGeminiImagePart(ref); imgPart != nil {
			parts = append(parts, imgPart)
		}
	}
	parts = append(parts, map[string]any{"text": prompt})

	genCfg := map[string]any{
		"responseModalities": []string{"TEXT", "IMAGE"},
	}
	payload := map[string]any{
		"contents": []map[string]any{
			{
				"role":  "user",
				"parts": parts,
			},
		},
		"generationConfig": genCfg,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return provider.ImageGenerationResponse{}, err
	}
	endpoint, err := c.endpointForModelAction(model, "generateContent")
	if err != nil {
		return provider.ImageGenerationResponse{}, err
	}

	resp, err := c.doHTTPWithAccept(ctx, endpoint, body, "application/json")
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
		ModelVersion string `json:"modelVersion"`
		Candidates   []struct {
			Content *struct {
				Parts []struct {
					Text       string `json:"text"`
					InlineData *struct {
						MimeType string `json:"mimeType"`
						Data     string `json:"data"`
					} `json:"inlineData"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return provider.ImageGenerationResponse{}, fmt.Errorf("gemini image decode: %w", err)
	}

	out := provider.ImageGenerationResponse{Model: model}
	if parsed.ModelVersion != "" {
		out.Model = parsed.ModelVersion
	}
	var notes []string
	for _, cand := range parsed.Candidates {
		if cand.Content == nil {
			continue
		}
		for _, p := range cand.Content.Parts {
			if strings.TrimSpace(p.Text) != "" {
				notes = append(notes, strings.TrimSpace(p.Text))
			}
			if p.InlineData != nil && strings.TrimSpace(p.InlineData.Data) != "" {
				b64 := strings.TrimSpace(p.InlineData.Data)
				mt := p.InlineData.MimeType
				if mt == "" {
					mt = schema.DetectBase64ImageMediaType(b64)
				}
				out.Images = append(out.Images, schema.ImageSource{
					MediaType: mt,
					Base64:    b64,
				})
			}
		}
	}
	out.RevisedPrompt = strings.Join(notes, "\n")
	if len(out.Images) == 0 {
		return provider.ImageGenerationResponse{}, errors.New("gemini image generation returned no images")
	}
	return out, nil
}
