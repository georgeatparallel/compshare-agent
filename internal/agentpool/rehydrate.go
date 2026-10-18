package agentpool

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/compshare-agent/internal/engine"
	"github.com/compshare-agent/internal/governance"
	"github.com/compshare-agent/internal/store"
)

// denyConfirm is used as the ConfirmFunc for HTTP-path engines. All L1
// mutating actions are denied — confirmation requires human interaction
// which is not available over the HTTP API.
func denyConfirm(_ string, _ map[string]any) bool { return false }

const recentHistoryPageSize = 128

// historyMessageSourceRunes charges the complete cold-history carrier rather
// than only its display content. Assistant metadata contains the canonical
// tool transcript that RehydrateHistory will parse and retain, while the JSON
// envelope supplies a natural structural cost for very short messages. Without
// both, a one-character conversation can drive thousands of otherwise free DB
// rows and transcript metadata through the fixed source budget.
func historyMessageSourceRunes(msg engine.HistoryMessage) int {
	raw, err := json.Marshal(msg)
	if err == nil {
		return len([]rune(string(raw)))
	}
	// json.RawMessage can make Marshal fail only for malformed stored metadata.
	// Charge that raw payload conservatively instead of turning it into free I/O.
	return len([]rune(msg.Role)) + len([]rune(msg.Content)) + len(msg.Transcript)
}

func historyMessage(msg store.Message) (engine.HistoryMessage, bool) {
	interrupted := msg.Role == "assistant" && (msg.Status == "aborted" || msg.Status == "error")
	if msg.Status != "ok" && !interrupted {
		return engine.HistoryMessage{}, false
	}
	if msg.Role != "user" && msg.Role != "assistant" {
		return engine.HistoryMessage{}, false
	}
	content := msg.Content
	if interrupted {
		content = ""
	}
	return engine.HistoryMessage{
		Role:       msg.Role,
		Content:    content,
		Transcript: msg.Metadata,
	}, true
}

// loadRecentHistory walks persisted rows from newest to oldest until the
// engine's raw-history size budget is full. It commits only whole conversation
// boundaries: a user row and its first following assistant row, or an
// unanswered user row. The result is restored to chronological order before it
// reaches RehydrateHistory.
func loadRecentHistory(ctx context.Context, messages store.MessageStore, sessionID string, budgetRunes int) ([]engine.HistoryMessage, error) {
	var (
		cursor           string
		turnsNewestFirst [][]engine.HistoryMessage
		pendingAssistant *engine.HistoryMessage
		spentRunes       int
		budgetReached    bool
	)

	for {
		page, nextCursor, err := messages.ListRecentBySessionPage(ctx, sessionID, recentHistoryPageSize, cursor)
		if err != nil {
			return nil, err
		}
		for _, row := range page {
			msg, ok := historyMessage(row)
			if !ok {
				continue
			}
			switch msg.Role {
			case "assistant":
				// Descending traversal sees later duplicate assistant rows first.
				// RehydrateHistory accepts the first chronological assistant, so an
				// older one replaces the pending later orphan here.
				copyMsg := msg
				pendingAssistant = &copyMsg
			case "user":
				turn := []engine.HistoryMessage{msg}
				turnRunes := historyMessageSourceRunes(msg)
				if pendingAssistant != nil {
					turn = append(turn, *pendingAssistant)
					turnRunes += historyMessageSourceRunes(*pendingAssistant)
				}
				if budgetRunes > 0 && len(turnsNewestFirst) > 0 && spentRunes+turnRunes > budgetRunes {
					budgetReached = true
					break
				}
				turnsNewestFirst = append(turnsNewestFirst, turn)
				spentRunes += turnRunes
				pendingAssistant = nil
			}
			if budgetReached {
				break
			}
		}
		if budgetReached || nextCursor == "" || len(page) == 0 {
			break
		}
		cursor = nextCursor
	}

	rowCount := 0
	for _, turn := range turnsNewestFirst {
		rowCount += len(turn)
	}
	history := make([]engine.HistoryMessage, 0, rowCount)
	for i := len(turnsNewestFirst) - 1; i >= 0; i-- {
		history = append(history, turnsNewestFirst[i]...)
	}
	return history, nil
}

// buildEngine constructs a fresh *engine.Engine for the given owner+session, then
// rehydrates its history from the MessageStore. engine.Init() is deliberately
// NOT called (HTTP path skips the welcome/suggestion pre-warm — see design §6.3).
func (p *Pool) buildEngine(ctx context.Context, owner store.Owner, sessionID string) (*engine.Engine, error) {
	eng := engine.NewSession(p.deps, engine.SessionOptions{
		ConversationKey:      fmt.Sprintf("%d:%d:%s", owner.TopOrganizationID, owner.OrganizationID, sessionID),
		Subject:              governance.AnonymousSubjectKey,
		ConfirmFn:            denyConfirm,
		MutatingToolsEnabled: p.mutatingToolsEnabled,
	})

	// Read backwards until the engine's existing size budget is satisfied. The
	// database page size is only an I/O batch size, never a memory boundary.
	// owner is threaded through so a future store contract may scope this read
	// without changing the pool key or rebuild path.
	history, err := loadRecentHistory(ctx, p.messageStore, sessionID, engine.HistorySourceRuneBudget())
	if err != nil {
		return nil, fmt.Errorf("agentpool: list messages for session %q: %w", sessionID, err)
	}

	eng.RehydrateHistory(history)
	return eng, nil
}
