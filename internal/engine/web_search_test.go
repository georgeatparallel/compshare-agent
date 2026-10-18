package engine

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/compshare-agent/internal/capability"
	"github.com/compshare-agent/internal/config"
	"github.com/compshare-agent/internal/knowledge"
	"github.com/compshare-agent/internal/llm"
	"github.com/compshare-agent/internal/websearch"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/require"
)

type parallelSearchInput struct {
	Objective     string   `json:"objective"`
	SearchQueries []string `json:"search_queries"`
	SessionID     string   `json:"session_id"`
}

type parallelRequest struct {
	URL, UserAgent, Authorization, Method string
}

type parallelTransport struct {
	base     http.RoundTripper
	target   *url.URL
	mu       sync.Mutex
	requests []parallelRequest
}

func (t *parallelTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	var method string
	if r.Body != nil {
		data, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}
		r.Body = io.NopCloser(strings.NewReader(string(data)))
		var payload struct {
			Method string `json:"method"`
		}
		_ = json.Unmarshal(data, &payload)
		method = payload.Method
	}
	t.mu.Lock()
	t.requests = append(t.requests, parallelRequest{r.URL.String(), r.UserAgent(), r.Header.Get("Authorization"), method})
	t.mu.Unlock()
	clone := r.Clone(r.Context())
	u := *r.URL
	u.Scheme, u.Host = t.target.Scheme, t.target.Host
	clone.URL = &u
	clone.Host = u.Host
	return t.base.RoundTrip(clone)
}

func installParallelFixture(t *testing.T, run func(context.Context, parallelSearchInput) (*mcp.CallToolResult, websearch.Result, error), onDelete ...func(context.Context)) *parallelTransport {
	t.Helper()
	server := mcp.NewServer(&mcp.Implementation{Name: "parallel-fixture", Version: "1"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "web_search"}, func(ctx context.Context, _ *mcp.CallToolRequest, in parallelSearchInput) (*mcp.CallToolResult, websearch.Result, error) {
		return run(ctx, in)
	})
	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, &mcp.StreamableHTTPOptions{Stateless: true, JSONResponse: true})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete && len(onDelete) > 0 {
			onDelete[0](r.Context())
		}
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	u, err := url.Parse(srv.URL)
	require.NoError(t, err)
	old := http.DefaultTransport
	transport := &parallelTransport{base: old, target: u}
	http.DefaultTransport = transport
	t.Cleanup(func() { http.DefaultTransport = old })
	return transport
}

func newWebSearchEngine(t *testing.T, cfg config.WebSearchConfig, model *mockLLM) *Engine {
	t.Helper()
	deps, err := NewSharedDeps(&config.Config{Agent: config.AgentConfig{LLM: config.LLMConfig{Model: "scripted-fixture"}, WebSearch: cfg}})
	require.NoError(t, err)
	deps.LLMClient = model
	deps.ExternalExecutor = &mockExecutor{}
	return NewSession(deps, SessionOptions{})
}

func webSearchModel() *mockLLM {
	return &mockLLM{responses: []llm.ChatResponse{
		{ToolCalls: []openai.ToolCall{toolCall("web", capability.WebSearchTool, `{"objective":"Find current CUDA compatibility documentation","search_queries":["CUDA compatibility driver documentation"]}`)}},
		{Content: "参考 [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)。"},
	}}
}

func TestWebSearchDefaultThroughCentralLoop(t *testing.T) {
	t.Setenv("PARALLEL_API_KEY", "unused-ambient-key")
	var inputs []parallelSearchInput
	transport := installParallelFixture(t, func(_ context.Context, in parallelSearchInput) (*mcp.CallToolResult, websearch.Result, error) {
		inputs = append(inputs, in)
		return nil, websearch.Result{Results: []websearch.Page{{URL: "https://docs.nvidia.com/deploy/cuda-compatibility/", Title: "CUDA compatibility", Excerpts: []string{"Driver compatibility reference from fixture."}}}}, nil
	})
	model := webSearchModel()
	eng := newWebSearchEngine(t, config.WebSearchConfig{}, model)
	require.NotNil(t, eng.webSearch, "an unconfigured installation gets anonymous web search")
	reply, err := eng.Chat(context.Background(), "查当前 CUDA 驱动兼容文档", noopStep)
	require.NoError(t, err)
	require.Contains(t, reply, "https://docs.nvidia.com/")
	require.Len(t, model.calls, 2)
	require.Contains(t, toolNames(model.calls[0].Tools), capability.WebSearchTool)
	observation := renderTestMessages(model.calls[1].Messages)
	require.Contains(t, observation, "Driver compatibility reference from fixture.")
	require.Contains(t, observation, "https://docs.nvidia.com/deploy/cuda-compatibility/")
	require.Len(t, inputs, 1)
	require.Equal(t, "Find current CUDA compatibility documentation", inputs[0].Objective)
	require.Equal(t, []string{"CUDA compatibility driver documentation"}, inputs[0].SearchQueries)
	require.NotEmpty(t, inputs[0].SessionID)
	_, err = eng.webSearch.Search(context.Background(), eng.webSearchSessionID, "CUDA reference", []string{"CUDA compatibility documentation"})
	require.NoError(t, err)
	require.Equal(t, inputs[0].SessionID, inputs[1].SessionID)
	other := newWebSearchEngine(t, config.WebSearchConfig{}, &mockLLM{})
	require.NotEqual(t, eng.webSearchSessionID, other.webSearchSessionID)
	transport.mu.Lock()
	defer transport.mu.Unlock()
	methods := map[string]bool{}
	for _, req := range transport.requests {
		require.Equal(t, websearch.Endpoint, req.URL)
		require.Equal(t, websearch.UserAgent, req.UserAgent)
		require.Empty(t, req.Authorization)
		methods[req.Method] = true
	}
	require.True(t, methods["initialize"])
	require.True(t, methods["tools/list"])
	require.True(t, methods["tools/call"])
}

func TestWebSearchDisabledAndChannelPermissions(t *testing.T) {
	transport := installParallelFixture(t, func(context.Context, parallelSearchInput) (*mcp.CallToolResult, websearch.Result, error) {
		t.Error("disabled or restricted search reached provider")
		return nil, websearch.Result{}, nil
	})
	eng := newWebSearchEngine(t, config.WebSearchConfig{Disabled: true}, webSearchModel())
	require.Nil(t, eng.webSearch)
	require.NotContains(t, toolNames(eng.toolWindowForRound(ChatOptions{})), capability.WebSearchTool)
	_, err := eng.Chat(context.Background(), "查 CUDA 文档", noopStep)
	require.NoError(t, err)
	for _, opts := range []ChatOptions{{KnowledgeOnly: true}, {PublicPlatformReadOnly: true}} {
		eng = newWebSearchEngine(t, config.WebSearchConfig{}, webSearchModel())
		require.NotContains(t, toolNames(eng.toolWindowForRound(opts)), capability.WebSearchTool)
		_, err = eng.ChatWithOptions(context.Background(), "查 CUDA 文档", noopStep, opts)
		require.NoError(t, err)
	}
	require.Empty(t, transport.requests)
}

func TestWebSearchFailureAndDeadlineReachModel(t *testing.T) {
	for _, mode := range []string{"error", "timeout", "bad_url"} {
		t.Run(mode, func(t *testing.T) {
			installParallelFixture(t, func(ctx context.Context, _ parallelSearchInput) (*mcp.CallToolResult, websearch.Result, error) {
				switch mode {
				case "error":
					return &mcp.CallToolResult{IsError: true, Content: []mcp.Content{&mcp.TextContent{Text: "provider unavailable"}}}, websearch.Result{}, nil
				case "timeout":
					select {
					case <-ctx.Done():
					case <-time.After(300 * time.Millisecond):
					}
					return nil, websearch.Result{}, ctx.Err()
				default:
					return nil, websearch.Result{Results: []websearch.Page{{URL: "javascript:alert(1)"}}}, nil
				}
			})
			model := webSearchModel()
			eng := newWebSearchEngine(t, config.WebSearchConfig{TimeoutMS: 100}, model)
			_, err := eng.Chat(context.Background(), "查 CUDA 文档", noopStep)
			require.NoError(t, err)
			require.Len(t, model.calls, 2)
			observation := renderTestMessages(model.calls[1].Messages)
			if mode == "timeout" {
				require.Contains(t, observation, `"code":"UPSTREAM_TIMEOUT"`)
			} else {
				require.Contains(t, observation, `"status":"failed"`)
			}
			require.NotContains(t, observation, "javascript:")
		})
	}
}

func TestWebSearchCancellation(t *testing.T) {
	started := make(chan struct{})
	installParallelFixture(t, func(ctx context.Context, _ parallelSearchInput) (*mcp.CallToolResult, websearch.Result, error) {
		close(started)
		select {
		case <-ctx.Done():
		case <-time.After(300 * time.Millisecond):
		}
		return nil, websearch.Result{}, ctx.Err()
	})
	eng := newWebSearchEngine(t, config.WebSearchConfig{}, &mockLLM{})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := eng.webSearch.Search(ctx, eng.webSearchSessionID, "CUDA docs", []string{"CUDA compatibility documentation"})
		done <- err
	}()
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("search did not reach MCP caller")
	}
	cancel()
	select {
	case err := <-done:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(time.Second):
		t.Fatal("cancellation did not stop search")
	}
}

func TestWebSearchCleanupHonorsLifetime(t *testing.T) {
	for _, mode := range []string{"deadline", "stricter_client", "cancellation"} {
		t.Run(mode, func(t *testing.T) {
			cleanupStarted := make(chan struct{})
			installParallelFixture(t, func(context.Context, parallelSearchInput) (*mcp.CallToolResult, websearch.Result, error) {
				return nil, websearch.Result{Results: []websearch.Page{{URL: "https://docs.nvidia.com/"}}}, nil
			}, func(ctx context.Context) {
				close(cleanupStarted)
				select {
				case <-ctx.Done():
				case <-time.After(300 * time.Millisecond):
				}
			})
			timeout := 70 * time.Millisecond
			httpClient := &http.Client{}
			if mode == "stricter_client" {
				timeout = time.Second
				httpClient.Timeout = 70 * time.Millisecond
			}
			if mode == "cancellation" {
				timeout = time.Second
			}
			client := websearch.NewClient(httpClient, timeout)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan struct{})
			started := time.Now()
			go func() {
				_, _ = client.Search(ctx, "fixture-session", "CUDA docs", []string{"CUDA compatibility documentation"})
				close(done)
			}()
			select {
			case <-cleanupStarted:
			case <-time.After(time.Second):
				t.Fatal("successful search did not reach cleanup")
			}
			if mode == "cancellation" {
				started = time.Now()
				cancel()
			}
			select {
			case <-done:
			case <-time.After(200 * time.Millisecond):
				t.Fatal("cleanup exceeded the search lifetime")
			}
			require.Less(t, time.Since(started), 200*time.Millisecond)
		})
	}
}

func TestWebSearchDoesNotReplaceFailedKnowledgeSearch(t *testing.T) {
	transport := installParallelFixture(t, func(context.Context, parallelSearchInput) (*mcp.CallToolResult, websearch.Result, error) {
		t.Error("knowledge failure silently invoked web search")
		return nil, websearch.Result{}, nil
	})
	model := &mockLLM{responses: []llm.ChatResponse{
		{ToolCalls: []openai.ToolCall{toolCall("kb", "SearchKnowledge", `{"query":"平台计费规则"}`)}},
		{Content: "目前无法核实平台规则。"},
	}}
	eng := newWebSearchEngine(t, config.WebSearchConfig{}, model)
	eng.SetKnowledgeRetriever(&scriptedKnowledgeRetriever{results: []knowledge.RetrievalResult{{Enabled: true, Unavailable: true, Empty: true}}})
	_, err := eng.Chat(context.Background(), "平台计费规则是什么", noopStep)
	require.NoError(t, err)
	require.Empty(t, transport.requests)
}

func TestWebSearchInvalidArgumentsAndBoundedResults(t *testing.T) {
	calls := 0
	installParallelFixture(t, func(context.Context, parallelSearchInput) (*mcp.CallToolResult, websearch.Result, error) {
		calls++
		pages := make([]websearch.Page, 10)
		for i := range pages {
			pages[i] = websearch.Page{URL: "https://docs.nvidia.com/", Excerpts: []string{strings.Repeat("文", 30000)}}
		}
		return nil, websearch.Result{Results: pages}, nil
	})
	for _, args := range []string{
		`{}`,
		`{"objective":"CUDA docs"}`,
		`{"search_queries":["CUDA docs"]}`,
		`{"objective":"","search_queries":["CUDA docs"]}`,
		`{"objective":"CUDA docs","search_queries":[]}`,
		`{"objective":"CUDA docs","search_queries":null}`,
		`{"objective":" ","search_queries":["CUDA docs"]}`,
		`{"objective":"CUDA docs","search_queries":[" "]}`,
		`{"objective":"CUDA docs","search_queries":["a","b","c","d"]}`,
		`{"objective":"CUDA docs","search_queries":12}`,
	} {
		eng := newWebSearchEngine(t, config.WebSearchConfig{}, &mockLLM{})
		out := execToolInTurn(eng, toolCall("invalid", capability.WebSearchTool, args), noopStep)
		require.Contains(t, out.Observation, "INVALID_TOOL_ARGUMENTS")
		require.Contains(t, out.Observation, "correct_tool_call")
	}
	require.Zero(t, calls)
	eng := newWebSearchEngine(t, config.WebSearchConfig{}, &mockLLM{})
	result, err := eng.webSearch.Search(context.Background(), eng.webSearchSessionID, "CUDA docs", []string{"CUDA compatibility documentation"})
	require.NoError(t, err)
	require.True(t, result.Truncated)
	require.Len(t, result.Results, 1)
	require.Len(t, []rune(result.Results[0].Excerpts[0]), 24000)
}

func TestWebSearchTurnBudget(t *testing.T) {
	for _, mode := range []string{"sequential", "one_batch", "three_then_two"} {
		t.Run(mode, func(t *testing.T) { testWebSearchTurnBudget(t, mode) })
	}
}

func testWebSearchTurnBudget(t *testing.T, mode string) {
	calls := 0
	installParallelFixture(t, func(context.Context, parallelSearchInput) (*mcp.CallToolResult, websearch.Result, error) {
		calls++
		return nil, websearch.Result{Results: []websearch.Page{{URL: "https://docs.nvidia.com/", Excerpts: []string{"Public technical reference."}}}}, nil
	})
	model := &mockLLM{}
	var batch []openai.ToolCall
	for i := 0; i < 5; i++ {
		args := fmt.Sprintf(`{"objective":"CUDA reference %d","search_queries":["CUDA compatibility documentation"]}`, i)
		call := toolCall(fmt.Sprintf("web-%d", i), capability.WebSearchTool, args)
		if mode == "sequential" || (mode == "three_then_two" && i < 3) {
			model.responses = append(model.responses, llm.ChatResponse{ToolCalls: []openai.ToolCall{call}})
		} else {
			batch = append(batch, call)
		}
	}
	if len(batch) > 0 {
		model.responses = append(model.responses, llm.ChatResponse{ToolCalls: batch})
	}
	model.responses = append(model.responses, llm.ChatResponse{Content: "参考 NVIDIA 文档。"})
	eng := newWebSearchEngine(t, config.WebSearchConfig{}, model)
	_, err := eng.Chat(context.Background(), "查 CUDA 技术文档", noopStep)
	require.NoError(t, err)
	require.Equal(t, 4, calls)
	last := model.calls[len(model.calls)-1]
	require.NotContains(t, toolNames(last.Tools), capability.WebSearchTool)
	require.Contains(t, toolNames(last.Tools), "SearchKnowledge")
}

type liveParallelTransport struct {
	base http.RoundTripper
	t    *testing.T
}

func (r liveParallelTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r.t.Logf("MCP request url=%s user_agent=%s authorization_present=%t", req.URL, req.UserAgent(), req.Header.Get("Authorization") != "")
	resp, err := r.base.RoundTrip(req)
	if resp != nil {
		r.t.Logf("MCP response status=%d date=%s retry_after=%s", resp.StatusCode, resp.Header.Get("Date"), resp.Header.Get("Retry-After"))
		resp.Body = &liveParallelBody{ReadCloser: resp.Body, t: r.t}
	}
	return resp, err
}

type liveParallelBody struct {
	io.ReadCloser
	t    *testing.T
	data []byte
}

func (b *liveParallelBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if len(b.data) < 65536 {
		b.data = append(b.data, p[:min(n, 65536-len(b.data))]...)
	}
	return n, err
}

func (b *liveParallelBody) Close() error {
	b.t.Logf("MCP response body (up to 64 KiB): %s", b.data)
	return b.ReadCloser.Close()
}

// This is a scripted-model actual-caller probe, not a live-model routing test.
func TestWebSearchLiveActualCaller(t *testing.T) {
	if os.Getenv("COMPSHARE_PARALLEL_LIVE") != "1" {
		t.Skip("set COMPSHARE_PARALLEL_LIVE=1 to call anonymous Parallel MCP")
	}
	t.Setenv("PARALLEL_API_KEY", "")
	old := http.DefaultTransport
	http.DefaultTransport = liveParallelTransport{base: old, t: t}
	t.Cleanup(func() { http.DefaultTransport = old })
	model := webSearchModel()
	eng := newWebSearchEngine(t, config.WebSearchConfig{}, model)
	_, err := eng.Chat(context.Background(), "查当前 CUDA 兼容文档", noopStep)
	require.NoError(t, err)
	require.Len(t, model.calls, 2)
	var observation string
	for _, message := range model.calls[1].Messages {
		if message.Role == openai.ChatMessageRoleTool {
			observation = message.Content
		}
	}
	t.Log(observation)
	if strings.Contains(observation, `"status":"failed"`) {
		t.Fatal("live provider failed; retained observation above")
	}
	require.Contains(t, observation, "https://")
	require.Contains(t, observation, "excerpts")
}
