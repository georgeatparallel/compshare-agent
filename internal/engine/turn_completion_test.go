package engine

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/compshare-agent/internal/config"
	"github.com/compshare-agent/internal/governance"
	"github.com/compshare-agent/internal/llm"
	"github.com/compshare-agent/internal/observability"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatEmitsExactlyOneCompletionForAgentChosenHandoff(t *testing.T) {
	eng := NewWithDeps(&mockLLM{responses: []llm.ChatResponse{customerSupportToolCall()}}, &mockExecutor{}, nil)
	var completions []observability.TurnCompletionTrace
	eng.SetTurnCompletionObserver(func(trace observability.TurnCompletionTrace) {
		completions = append(completions, trace)
	})

	_, err := eng.Chat(context.Background(), "帮我转接人工", noopStep)
	require.NoError(t, err)
	require.Len(t, completions, 1, "every return path must pass the one top-level completion defer")
	got := completions[0]
	assert.Equal(t, observability.CompletionClassAgent, got.Class)
	assert.Equal(t, observability.CompletionReasonAgentLoop, got.Reason)
	assert.Equal(t, "deterministic_reply", got.RuntimeFinishReason)
	assert.Empty(t, got.ModelAttempts, "the in-process mock does not emit outbound-call telemetry")
}

func TestChatCompletionCountsRealOutboundModelRequests(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"正常回答\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":8,\"total_tokens\":128,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	client := llm.NewClient(config.LLMConfig{BaseURL: srv.URL + "/v1", APIKey: "test-key", Model: "test-model"})
	eng := NewWithDeps(client, &mockExecutor{}, nil)
	var completions []observability.TurnCompletionTrace
	eng.SetTurnCompletionObserver(func(trace observability.TurnCompletionTrace) {
		completions = append(completions, trace)
	})

	reply, err := eng.Chat(context.Background(), "介绍一下平台", noopStep)
	require.NoError(t, err)
	assert.Equal(t, "正常回答", reply)
	require.Len(t, completions, 1)
	assert.Equal(t, observability.CompletionClassAgent, completions[0].Class)
	assert.Equal(t, "final_answer", completions[0].RuntimeFinishReason,
		"the trace must retain the runtime's exact terminal reason, not just a hand-mapped class")
	require.Len(t, completions[0].ModelAttempts, 1)
	attempt := completions[0].ModelAttempts[0]
	assert.Equal(t, "model-1", attempt.ID)
	assert.Equal(t, "openai_compatible", attempt.Provider)
	assert.Equal(t, "test-model", attempt.Model)
	assert.Equal(t, 1, attempt.AttemptInCall)
	assert.Equal(t, "success", attempt.Outcome)
	assert.Equal(t, "stop", attempt.FinishReason)
	require.NotNil(t, attempt.FirstChunkMS)
	require.NotNil(t, attempt.PromptTokens)
	assert.Equal(t, 120, *attempt.PromptTokens)
	require.NotNil(t, attempt.CachedPromptTokens)
	assert.Zero(t, *attempt.CachedPromptTokens)
	assert.Equal(t, len(eng.toolWindowForRound(ChatOptions{})), attempt.ToolCount)
	assert.Positive(t, attempt.ToolWindowRunes)
	assert.Regexp(t, `^sha256:[0-9a-f]{64}$`, attempt.ToolWindowHash)

	// A second logical model call in the same Engine/session starts its own
	// attempt numbering and does not inherit the prior turn's attempt slice.
	_, err = eng.Chat(context.Background(), "再介绍一句", noopStep)
	require.NoError(t, err)
	require.Len(t, completions, 2)
	require.Len(t, completions[1].ModelAttempts, 1)
	secondAttempt := completions[1].ModelAttempts[0]
	assert.Equal(t, "model-1", secondAttempt.ID)
	assert.Equal(t, "openai_compatible", secondAttempt.Provider)
	assert.Equal(t, "test-model", secondAttempt.Model)
	assert.Equal(t, 1, secondAttempt.AttemptInCall)
	assert.Equal(t, "success", secondAttempt.Outcome)
	assert.Equal(t, "stop", secondAttempt.FinishReason)
}

// [DONE] without a choice finish_reason cannot attest that the streamed answer
// is complete. The attempt must remain an error, not success/unspecified.
func TestChatCompletionRejectsUnspecifiedProviderFinishReason(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"正常回答\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	client := llm.NewClient(config.LLMConfig{BaseURL: srv.URL + "/v1", APIKey: "test-key", Model: "test-model"})
	eng := NewWithDeps(client, &mockExecutor{}, nil)
	var completions []observability.TurnCompletionTrace
	eng.SetTurnCompletionObserver(func(trace observability.TurnCompletionTrace) {
		completions = append(completions, trace)
	})

	_, err := eng.Chat(context.Background(), "介绍一下平台", noopStep)
	require.ErrorIs(t, err, io.ErrUnexpectedEOF)
	require.Len(t, completions, 1)
	require.Len(t, completions[0].ModelAttempts, 2)
	for _, attempt := range completions[0].ModelAttempts {
		assert.Equal(t, "openai_compatible", attempt.Provider)
		assert.Equal(t, "test-model", attempt.Model)
		assert.Equal(t, "error", attempt.Outcome)
		assert.Equal(t, llm.OutboundErrorStream, attempt.ErrorClass)
		assert.Empty(t, attempt.FinishReason)
	}
}

func TestChatCompletionMarksTerminalRateLimit(t *testing.T) {
	eng := NewWithDeps(&mockLLM{}, &mockExecutor{}, nil)
	eng.rateLimiter = &scriptedRateLimiter{decisions: []governance.Decision{{
		Allowed: false,
		Reason:  governance.ReasonQPSExceeded,
		Err:     governance.ErrRateLimited,
	}}}
	var completions []observability.TurnCompletionTrace
	eng.SetTurnCompletionObserver(func(trace observability.TurnCompletionTrace) {
		completions = append(completions, trace)
	})

	_, err := eng.Chat(context.Background(), "hello", noopStep)
	require.NoError(t, err)
	require.Len(t, completions, 1)
	assert.Equal(t, observability.CompletionClassSafetyBlock, completions[0].Class)
	assert.Equal(t, observability.CompletionReasonRateLimit, completions[0].Reason)
	assert.Equal(t, "rate_limit", completions[0].RuntimeFinishReason,
		"the limiter runs inside the runtime driver, so its exact loop exit is retained")
	assert.Empty(t, completions[0].ModelAttempts)
}
