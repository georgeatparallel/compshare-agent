package agentpool_test

import (
	"context"
	"testing"
	"time"

	"github.com/compshare-agent/internal/agentpool"
	"github.com/compshare-agent/internal/capability"
	"github.com/compshare-agent/internal/engine"
	"github.com/compshare-agent/internal/llm"
	"github.com/compshare-agent/internal/store"
	"github.com/compshare-agent/internal/websearch"
	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/require"
)

type sessionSearchRecorder struct{ ids []string }

func (r *sessionSearchRecorder) Search(_ context.Context, sessionID, _ string, _ []string) (websearch.Result, error) {
	r.ids = append(r.ids, sessionID)
	return websearch.Result{Results: []websearch.Page{{URL: "https://docs.nvidia.com/", Excerpts: []string{"GPU documentation"}}}}, nil
}

type sessionSearchModel struct{ calls int }

func (m *sessionSearchModel) Chat(context.Context, llm.ChatRequest) (*llm.ChatResponse, error) {
	m.calls++
	if m.calls%2 == 0 {
		return &llm.ChatResponse{Content: "已检索公开技术资料。"}, nil
	}
	return &llm.ChatResponse{ToolCalls: []openai.ToolCall{{ID: "web", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: capability.WebSearchTool, Arguments: `{"objective":"CUDA docs","search_queries":["CUDA compatibility documentation"]}`}}}}, nil
}

func TestWebSearchConversationIdentitySurvivesColdRebuild(t *testing.T) {
	search := &sessionSearchRecorder{}
	deps := &engine.SharedDeps{LLMClient: &sessionSearchModel{}, WebSearch: search}
	newPool := func() *agentpool.Pool {
		p := agentpool.NewWithDeps(deps, &mockMessageStore{}, agentpool.Options{Capacity: 1, IdleTTL: time.Hour})
		t.Cleanup(p.Close)
		return p
	}
	query := func(p *agentpool.Pool, owner store.Owner, session string) *engine.Engine {
		e, release, err := p.Lease(context.Background(), owner, session)
		require.NoError(t, err)
		defer release()
		_, err = e.Chat(context.Background(), "查 CUDA 技术资料", func(engine.StepEvent) {})
		require.NoError(t, err)
		return e
	}
	p := newPool()
	first := query(p, owner1, "session-a")
	query(p, owner1, "session-b")
	rebuilt := query(p, owner1, "session-a")
	require.NotSame(t, first, rebuilt)
	query(p, store.Owner{TopOrganizationID: 2, OrganizationID: 1}, "session-a")
	query(newPool(), owner1, "session-a")
	require.Len(t, search.ids, 5)
	require.Regexp(t, `^[a-f0-9]{64}$`, search.ids[0])
	require.Equal(t, search.ids[0], search.ids[2])
	require.Equal(t, search.ids[0], search.ids[4])
	require.NotEqual(t, search.ids[0], search.ids[1])
	require.NotEqual(t, search.ids[0], search.ids[3])
	require.NotContains(t, search.ids[0], "session-a")
}
