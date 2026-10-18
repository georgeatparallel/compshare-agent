// Package websearch retrieves public technical references independently of the
// CompShare knowledge base and account-scoped platform observations.
package websearch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const Endpoint = "https://search.parallel.ai/mcp"
const UserAgent = "compshare-agent/1.0 (+https://github.com/guli20001221/compshare-agent)"

type Page struct {
	URL      string   `json:"url"`
	Title    string   `json:"title"`
	Excerpts []string `json:"excerpts"`
}

type Result struct {
	Results   []Page `json:"results"`
	Truncated bool   `json:"truncated,omitempty"`
}

// Client contains immutable transport configuration; conversation identity is
// supplied by the session engine rather than shared between tenants.
type Client struct {
	httpClient *http.Client
	timeout    time.Duration
}

func NewClient(httpClient *http.Client, timeout time.Duration) *Client {
	if httpClient == nil {
		httpClient = &http.Client{}
	}
	clone := *httpClient
	transport := clone.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	clone.Transport = projectTransport{base: transport}
	if timeout <= 0 {
		timeout = 20 * time.Second
	}
	return &Client{httpClient: &clone, timeout: timeout}
}

type projectTransport struct{ base http.RoundTripper }

func (t projectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	clone.Header = req.Header.Clone()
	clone.Header.Set("User-Agent", UserAgent)
	return t.base.RoundTrip(clone)
}

func Validate(objective string, queries []string) error {
	if n := utf8.RuneCountInString(strings.TrimSpace(objective)); n == 0 || n > 1000 {
		return errors.New("objective must contain 1 to 1000 characters")
	}
	if len(queries) == 0 || len(queries) > 3 {
		return errors.New("search_queries must contain 1 to 3 queries")
	}
	for _, query := range queries {
		if n := utf8.RuneCountInString(strings.TrimSpace(query)); n == 0 || n > 256 {
			return errors.New("each search query must contain 1 to 256 characters")
		}
	}
	return nil
}

// Search uses the anonymous Fast-mode MCP endpoint. It never reads environment
// credentials or substitutes web results for a failed CompShare knowledge call.
func (c *Client) Search(ctx context.Context, sessionID, objective string, queries []string) (Result, error) {
	if err := Validate(objective, queries); err != nil {
		return Result{}, err
	}
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	client := mcp.NewClient(&mcp.Implementation{Name: "compshare-agent", Version: "1.0.0"}, nil)
	session, err := client.Connect(ctx, &mcp.StreamableClientTransport{
		Endpoint: Endpoint, HTTPClient: c.httpClient, DisableStandaloneSSE: true, MaxRetries: -1,
	}, nil)
	if err != nil {
		return Result{}, fmt.Errorf("connect Parallel MCP: %w", err)
	}
	defer session.Close()
	listed, err := session.ListTools(ctx, nil)
	if err != nil {
		return Result{}, fmt.Errorf("discover Parallel MCP: %w", err)
	}
	found := false
	for _, tool := range listed.Tools {
		found = found || tool.Name == "web_search"
	}
	if !found {
		return Result{}, errors.New("Parallel MCP does not advertise web_search")
	}
	args := map[string]any{"objective": objective, "search_queries": queries}
	if sessionID != "" {
		args["session_id"] = sessionID
	}
	response, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "web_search", Arguments: args})
	if err != nil {
		return Result{}, fmt.Errorf("call Parallel MCP: %w", err)
	}
	if response.IsError {
		var details []string
		for _, content := range response.Content {
			if block, ok := content.(*mcp.TextContent); ok {
				details = append(details, block.Text)
			}
		}
		return Result{}, fmt.Errorf("Parallel MCP search failed: %s", strings.Join(details, "\n"))
	}
	var data []byte
	if response.StructuredContent != nil {
		data, err = json.Marshal(response.StructuredContent)
	} else {
		for _, content := range response.Content {
			if block, ok := content.(*mcp.TextContent); ok {
				data = append(data, block.Text...)
			}
		}
	}
	if err != nil {
		return Result{}, fmt.Errorf("decode Parallel MCP: %w", err)
	}
	var wire struct {
		Results *[]Page `json:"results"`
	}
	if err := json.Unmarshal(data, &wire); err != nil || wire.Results == nil {
		return Result{}, errors.New("Parallel MCP search returned an invalid results payload")
	}
	result := Result{Results: []Page{}}
	budget := 24000
	for i, page := range *wire.Results {
		if i >= 8 || budget <= 0 {
			result.Truncated = true
			break
		}
		u, err := url.Parse(page.URL)
		if err != nil || u.Host == "" || u.User != nil || len(page.URL) > 4096 || (u.Scheme != "https" && u.Scheme != "http") {
			return Result{}, errors.New("Parallel MCP search returned an invalid source URL")
		}
		page.Title = clip(page.Title, 300, &result.Truncated)
		if len(page.Excerpts) > 8 {
			page.Excerpts = page.Excerpts[:8]
			result.Truncated = true
		}
		for j, excerpt := range page.Excerpts {
			page.Excerpts[j] = clip(excerpt, budget, &result.Truncated)
			budget -= utf8.RuneCountInString(page.Excerpts[j])
		}
		result.Results = append(result.Results, page)
	}
	return result, nil
}

func clip(value string, limit int, truncated *bool) string {
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	*truncated = true
	return string(runes[:limit])
}
