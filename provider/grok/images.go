package grok

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/cambridgeadvancedsystems/jaz-harness/provider"
	"github.com/cambridgeadvancedsystems/jaz-harness/schema"
)

const DefaultImageModel = "grok-imagine-image"

// GenerateImage implements provider.ImageGenerator using xAI's native image
// endpoints (`POST /v1/images/generations` and `POST /v1/images/edits`).
func (c *Client) GenerateImage(ctx context.Context, req provider.ImageGenerationRequest) (provider.ImageGenerationResponse, error) {
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		return provider.ImageGenerationResponse{}, errors.New("grok image generation: prompt is required")
	}
	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = DefaultImageModel
	}

	payload := map[string]any{
		"model":           model,
		"prompt":          prompt,
		"response_format": "b64_json",
	}
	if req.Quality != "" {
		payload["quality"] = strings.ToLower(strings.TrimSpace(req.Quality))
	}

	endpoint := c.cfg.BaseURL + "/images/generations"
	if len(req.ReferenceImages) > 0 {
		endpoint = c.cfg.BaseURL + "/images/edits"
		refs := make([]map[string]any, 0, len(req.ReferenceImages))
		for _, img := range req.ReferenceImages {
			if u := img.DataURL(); u != "" {
				refs = append(refs, map[string]any{"url": u})
			}
		}
		if len(refs) == 0 {
			return provider.ImageGenerationResponse{}, errors.New("grok image edit: no valid reference images")
		}
		if len(refs) == 1 {
			payload["image"] = refs[0]
		} else {
			payload["images"] = refs
		}
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return provider.ImageGenerationResponse{}, err
	}

	resp, err := provider.DoWithRetry(ctx, 3, func(_ int) (*http.Response, error) {
		httpReq, rerr := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if rerr != nil {
			return nil, rerr
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "application/json")
		for k, v := range c.cfg.ExtraHeaders {
			httpReq.Header.Set(k, v)
		}
		if c.cfg.APIKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
		}
		return c.httpClient.Do(httpReq)
	})
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
		Data []struct {
			B64JSON       string `json:"b64_json"`
			URL           string `json:"url"`
			RevisedPrompt string `json:"revised_prompt"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return provider.ImageGenerationResponse{}, fmt.Errorf("grok image decode: %w", err)
	}
	out := provider.ImageGenerationResponse{Model: model}
	for _, item := range parsed.Data {
		if out.RevisedPrompt == "" && item.RevisedPrompt != "" {
			out.RevisedPrompt = item.RevisedPrompt
		}
		if b64 := strings.TrimSpace(item.B64JSON); b64 != "" {
			out.Images = append(out.Images, schema.ImageSource{
				MediaType: schema.DetectBase64ImageMediaType(b64),
				Base64:    b64,
			})
		} else if u := strings.TrimSpace(item.URL); u != "" {
			out.Images = append(out.Images, schema.ImageSource{URL: u})
		}
	}
	if len(out.Images) == 0 {
		return provider.ImageGenerationResponse{}, errors.New("grok image generation returned no images")
	}
	return out, nil
}
