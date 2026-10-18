package engine

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"

	"github.com/compshare-agent/internal/capability"
	"github.com/compshare-agent/internal/diagnosis"
	"github.com/compshare-agent/internal/intent"
	"github.com/compshare-agent/internal/knowledge"
	openai "github.com/sashabaranov/go-openai"
)

// repeatableAgentTool identifies reads that may legitimately run more than once
// in a turn as the Agent refines its evidence.
func repeatableAgentTool(action string) bool {
	if action == "SearchKnowledge" || action == "DiagnoseInstanceInternals" || diagnosis.IsDiagnosisTool(action) {
		return true
	}
	_, ok := capability.ReadIntentForTool(action)
	return ok
}

// maxAgentToolCallsPerTurn caps only capabilities whose upstream facts do not
// become more authoritative when the model keeps asking. It counts calls, not
// distinct arguments: a round spent re-asking costs the same whether or not the
// arguments changed. Two attempts leave room for one genuine correction without
// allowing a monitor turn to consume the whole ReAct budget.
func maxAgentToolCallsPerTurn(action string) int {
	if action == capability.WebSearchTool {
		return maxWebSearchCallsPerTurn
	}
	if action == "DiagnoseInstanceInternals" {
		return MaxInstanceOpsRunsPerTurn
	}
	readIntent, ok := capability.ReadIntentForTool(action)
	if !ok {
		return 0
	}
	switch readIntent {
	case intent.IntentMonitorQuery, intent.IntentMonitorHistory:
		return 2
	default:
		return 0
	}
}

// singleShotAgentTool identifies immutable whole-catalog reads. Once one such
// call succeeds, changing arguments cannot reveal fresher facts in the same
// turn; the next round should answer from the existing observation instead of
// spending model rounds re-filtering the same snapshot.
func singleShotAgentTool(action string) bool {
	readIntent, ok := capability.ReadIntentForTool(action)
	return ok && readIntent == intent.IntentZoneCatalog
}

func completedAgentToolCall(results map[string]string, action string) bool {
	for key, raw := range results {
		recordedAction, _, found := strings.Cut(key, ":")
		if !found || recordedAction != action {
			continue
		}
		var observation struct {
			Status string `json:"status"`
		}
		if json.Unmarshal([]byte(raw), &observation) == nil {
			switch observation.Status {
			case "handled", "empty", "conflict", "unavailable":
				return true
			}
		}
	}
	return false
}

// agentToolCallsThisTurn counts the current turn's calls to action that already
// produced a tool result, reading the canonical transcript rather than a side
// counter. Same authority as turnReturnedToolResults: the conversation records
// what was asked and answered, while a cache records only what may be replayed.
//
// Counting settled calls rather than every call in the transcript keeps the
// answer independent of where the caller sits. One assistant message can carry
// several calls to the same capability; charging them all before the first one
// runs would refuse a batch the budget is meant to allow.
func (e *Engine) agentToolCallsThisTurn(action string) int {
	start := currentTurnStart(e.messages)
	if start < 0 {
		return 0
	}
	callIDs := map[string]struct{}{}
	for _, message := range e.messages[start:] {
		for _, call := range message.ToolCalls {
			if call.Function.Name == action {
				callIDs[call.ID] = struct{}{}
			}
		}
	}
	if len(callIDs) == 0 {
		return 0
	}
	settled := 0
	for _, message := range e.messages[start:] {
		if message.Role != openai.ChatMessageRoleTool {
			continue
		}
		if _, ok := callIDs[message.ToolCallID]; ok {
			settled++
		}
	}
	return settled
}

func toolCallBudgetObservation(action string, limit int) string {
	payload, _ := json.Marshal(map[string]any{
		"status":                 "call_budget_exhausted",
		"action":                 action,
		"max_calls_per_turn":     limit,
		"required_next_decision": "answer from the existing observations or ask the user for one specific missing field; do not call this capability again this turn",
	})
	return string(payload)
}

func decodeToolArgsForProgress(raw string) (map[string]any, bool) {
	var args map[string]any
	if json.Unmarshal([]byte(raw), &args) != nil {
		return nil, false
	}
	return args, true
}

func toolProgressCallKey(action string, args map[string]any) string {
	payload, _ := json.Marshal(args)
	return action + ":" + digestToolProgress(payload)
}

func digestToolProgress(payload []byte) string {
	digest := sha256.Sum256(payload)
	return hex.EncodeToString(digest[:])
}

type searchKnowledgeCacheObservation struct {
	EvidenceLedger           *knowledge.EvidenceLedger      `json:"EvidenceLedger"`
	BelowFloorCandidates     []belowFloorKnowledgeCandidate `json:"below_floor_candidates"`
	KnowledgeUnavailable     bool                           `json:"knowledge_unavailable"`
	AutoExpansionUnavailable bool                           `json:"auto_expansion_unavailable"`
	Error                    json.RawMessage                `json:"error"`
}

// An unavailable search is not a reusable observation. A later attempt still
// passes through SearchKnowledge's existing per-turn retrieval budget.
//
// A Guest run is never a reusable observation either, for a stronger reason: the
// observation describes a machine that the run itself may have changed. Replaying
// one would answer a retry after a dropped SSH connection with the previous
// attempt's partial command list, and tell the model not to repeat itself. How
// often a turn may enter a Guest is bounded by maxAgentToolCallsPerTurn instead.
func cacheableAgentToolObservation(action, raw string) bool {
	switch action {
	case "DiagnoseInstanceInternals":
		return false
	case "SearchKnowledge":
		var observation searchKnowledgeCacheObservation
		return json.Unmarshal([]byte(raw), &observation) == nil &&
			observation.EvidenceLedger != nil && !observation.KnowledgeUnavailable &&
			!observation.AutoExpansionUnavailable && len(observation.Error) == 0
	}
	return true
}

// A write or Guest run can change facts a previous read observed. Keep the
// transcript as the account of that earlier state, but let a later read obtain
// fresh evidence. Knowledge and the turn's immutable catalog remain reusable.
func (e *Engine) invalidateLiveToolObservations() {
	for key := range e.toolResultsByCallThisTurn {
		action, _, _ := strings.Cut(key, ":")
		if action != "SearchKnowledge" && !singleShotAgentTool(action) {
			delete(e.toolResultsByCallThisTurn, key)
		}
	}
}

// Expiry removes the capability and only cached searches exposing its IDs.
// Unrelated successful searches remain reusable; the model chooses whether to
// spend another search call to obtain a fresh capability.
func (e *Engine) invalidateSearchKnowledgeCapability(searchID string) {
	expiredIDs := map[string]struct{}{}
	for chunkID, capability := range e.searchKnowledgeCapabilitiesThisTurn {
		if capability == searchID {
			expiredIDs[chunkID] = struct{}{}
			delete(e.searchKnowledgeCapabilitiesThisTurn, chunkID)
			delete(e.automaticKnowledgeBodyIDsThisTurn, chunkID)
		}
	}
	if len(expiredIDs) == 0 {
		return
	}
	for key, raw := range e.toolResultsByCallThisTurn {
		action, _, found := strings.Cut(key, ":")
		if found && action == "SearchKnowledge" && searchKnowledgeObservationUsesIDs(raw, expiredIDs) {
			delete(e.toolResultsByCallThisTurn, key)
		}
	}
}

func searchKnowledgeObservationUsesIDs(raw string, ids map[string]struct{}) bool {
	var observation searchKnowledgeCacheObservation
	if json.Unmarshal([]byte(raw), &observation) != nil {
		return false
	}
	if observation.EvidenceLedger != nil {
		for _, item := range observation.EvidenceLedger.Items {
			if _, ok := ids[item.ChunkID]; ok {
				return true
			}
		}
	}
	for _, candidate := range observation.BelowFloorCandidates {
		if _, ok := ids[candidate.ChunkID]; ok {
			return true
		}
	}
	return false
}

func repeatedToolObservation(action, previous string) string {
	var observation any
	if json.Unmarshal([]byte(previous), &observation) != nil {
		observation = previous
	}
	payload, _ := json.Marshal(map[string]any{
		"status":                 "reused_observation",
		"action":                 action,
		"same_call_blocked":      true,
		"observation":            observation,
		"required_next_decision": "do_not_repeat_the_same_arguments; answer, clarify, or use materially different arguments",
	})
	return string(payload)
}
