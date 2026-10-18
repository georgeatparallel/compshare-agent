package engine

import (
	"sort"
	"strings"

	openai "github.com/sashabaranov/go-openai"

	"github.com/compshare-agent/internal/actionresolver"
	"github.com/compshare-agent/internal/capability"
	"github.com/compshare-agent/internal/tools"
)

// toolWindowForRound decides the window the next request carries and narrows it
// to its final shape. It runs FIRST in the round, because the window is part of
// the request the provider sizes against and the message budget has to be told
// how much of the request it has already spent. It travels as its own field
// (llm.ChatRequest.Tools), so nothing about it is visible in the message list —
// the production window is 40 schemas and 22,806 runes, larger than the system
// prompt by an order of magnitude.
//
// Narrowing is how a spent budget is enforced: the capability is removed rather
// than refused in prose, because injecting another policy prompt would create a
// second and potentially conflicting contract. The window is built before the
// next assistant message, so the transcript it counts holds exactly the calls
// that completed.
func (e *Engine) toolWindowForRound(opts ChatOptions) []openai.Tool {
	toolWindow := centralAgentToolWindow(e.mutatingToolsEnabled, e.instanceOps != nil)
	if opts.KnowledgeOnly {
		toolWindow = centralAgentKnowledgeToolWindow()
	} else if opts.PublicPlatformReadOnly {
		toolWindow = centralAgentPublicPlatformReadOnlyToolWindow()
	}
	if e.webSearch == nil || e.agentToolCallsThisTurn(capability.WebSearchTool) >= 4 {
		toolWindow = toolListWithoutFunction(toolWindow, capability.WebSearchTool)
	}
	if e.agentToolCallsThisTurn("SearchKnowledge") >= maxSearchKnowledgeCallsPerTurn &&
		toolListContainsFunction(toolWindow, "SearchKnowledge") {
		toolWindow = toolListWithoutFunction(toolWindow, "SearchKnowledge")
	}
	// Same rule for full-body reads, on their own budget.
	if e.agentToolCallsThisTurn("ReadChunk") >= maxReadChunkCallsPerTurn &&
		toolListContainsFunction(toolWindow, "ReadChunk") {
		toolWindow = toolListWithoutFunction(toolWindow, "ReadChunk")
	}
	// A whole-catalog read is complete after one successful observation. The
	// model may still reason over that observation, but cannot spend later
	// rounds asking the same immutable snapshot with cosmetic query variants.
	for _, tool := range toolWindow {
		if tool.Function != nil && singleShotAgentTool(tool.Function.Name) &&
			completedAgentToolCall(e.toolResultsByCallThisTurn, tool.Function.Name) {
			toolWindow = toolListWithoutFunction(toolWindow, tool.Function.Name)
		}
	}
	return toolWindow
}

// centralAgentToolWindow is the model-visible capability surface. It intentionally
// does not expose the underlying API tools used by deterministic handlers. Each
// high-level read is a distinct, catalog-generated tool, while every platform
// fact still crosses the same EvidenceEnvelope adapter.
func centralAgentToolWindow(mutatingEnabled, instanceOpsEnabled bool) []openai.Tool {
	registry := tools.DefaultCapabilityRegistry()
	var out []openai.Tool
	if mutatingEnabled {
		if capability, ok := registry.Lookup(tools.ProposeActionName); ok {
			out = append(out, proposalToolsFromCatalog(capability.Tool)...)
		}
	}
	for _, definition := range capability.ReadDefinitions() {
		out = append(out, definition.Tool)
	}
	for _, capability := range registry.All() {
		if !capability.ExposedToAgent || capability.Tool.Function == nil {
			continue
		}
		if capability.Policy.Route == tools.ActionRouteKnowledge ||
			capability.Policy.Route == tools.ActionRouteDiagnosis ||
			capability.Policy.Route == tools.ActionRouteHandoff {
			// DiagnoseInstanceInternals carries an ActionRouteDiagnosis policy (derived
			// from its "Diagnose" prefix) so it would append unconditionally here.
			// Gate it on both the in-instance runner and the deployment's standing
			// write authorization. SSH-ops diagnoses and performs reversible repairs
			// as one autonomous task, so exposing it in a read-only window would make
			// the visible contract disagree with the runtime (INV-10).
			if capability.Name == "DiagnoseInstanceInternals" && (!instanceOpsEnabled || !mutatingEnabled) {
				continue
			}
			out = append(out, capability.Tool)
		}
	}
	return out
}

// centralAgentKnowledgeToolWindow is the fail-closed public-Q&A surface used by
// untrusted chat channels such as Feishu groups. It exposes only knowledge
// retrieval/read capabilities plus the response-only customer-support handoff.
// Platform reads, diagnoses and every action proposal are deliberately absent.
func centralAgentKnowledgeToolWindow() []openai.Tool {
	registry := tools.DefaultCapabilityRegistry()
	var out []openai.Tool
	for _, capability := range registry.All() {
		if !capability.ExposedToAgent || capability.Tool.Function == nil {
			continue
		}
		if capability.Policy.Route == tools.ActionRouteKnowledge ||
			capability.Policy.Route == tools.ActionRouteHandoff {
			out = append(out, capability.Tool)
		}
	}
	return out
}

func proposalToolName(operation string) string {
	return "Request" + strings.TrimSuffix(operation, "Workflow")
}

func proposalOperationForTool(name string) (string, bool) {
	catalog, err := defaultActionCatalog()
	if err != nil {
		return "", false
	}
	for _, operation := range catalog.Operations() {
		if name == proposalToolName(operation) {
			return operation, true
		}
	}
	return "", false
}

// proposalToolsFromCatalog gives every write operation one discoverable,
// generated proposal capability. The model selects the operation by choosing a
// tool; the server injects the canonical operation ID and still sends every
// candidate through Resolver, Gate and confirmation.
func proposalToolsFromCatalog(base openai.Tool) []openai.Tool {
	catalog, err := defaultActionCatalog()
	if err != nil || base.Function == nil {
		return nil
	}
	var out []openai.Tool
	for _, operation := range catalog.Operations() {
		spec, _ := catalog.Lookup(operation)
		// Model-visible proposals have no out-of-band secret channel. Advertising an
		// operation whose required value was removed from its schema creates an
		// impossible loop: the model is told the value exists but the resolver can
		// never receive it. Trusted callers may still use the underlying workflow.
		if operationRequiresSensitiveInput(spec) {
			continue
		}
		tool := proposalToolForOperation(base, spec)
		out = append(out, tool)
	}
	return out
}

func operationRequiresSensitiveInput(spec actionresolver.OperationSpec) bool {
	for _, field := range spec.Fields {
		if field.Required && field.Codec == actionresolver.CodecSensitiveText {
			return true
		}
	}
	return false
}

func proposalToolForOperation(base openai.Tool, spec actionresolver.OperationSpec) openai.Tool {
	tool := base
	function := *base.Function
	capability, ok := tools.DefaultCapabilityRegistry().Lookup(spec.Operation)
	if !ok || capability.Tool.Function == nil {
		return base
	}
	root, ok := cloneSchemaObject(capability.Tool.Function.Parameters)
	if !ok {
		return base
	}
	properties, _ := root["properties"].(map[string]any)
	hasSensitiveField := false
	for name, field := range spec.Fields {
		if field.Codec == actionresolver.CodecSensitiveText {
			delete(properties, name)
			hasSensitiveField = true
		}
	}
	// A proposal may be intentionally incomplete: Resolver returns the exact
	// missing fields and the Agent then asks only for those. Requiring workflow
	// fields in the model schema makes the model ask in prose before it can call.
	root["required"] = []string{}
	function.Name = proposalToolName(spec.Operation)
	// The system prompt owns the action-first / partial-proposal / confirmation
	// rules once. A tool description owns only this operation's semantic boundary;
	// workflow step sequences are runtime details and must never leak in here.
	function.Description = strings.TrimSpace(spec.AgentDescription)
	if hasSensitiveField {
		function.Description += " 敏感字段不能通过模型参数提交；可选敏感字段保持未设置。"
	}
	function.Parameters = root
	tool.Function = &function
	return tool
}

func proposalArgsForOperation(operation string, direct map[string]any) map[string]any {
	names := make([]string, 0, len(direct))
	for name := range direct {
		names = append(names, name)
	}
	sort.Strings(names)
	slots := make([]any, 0, len(names))
	for _, name := range names {
		slots = append(slots, map[string]any{"name": name, "value": direct[name]})
	}
	return map[string]any{"operation": operation, "slots": slots}
}

func cloneSchemaObject(value any) (map[string]any, bool) {
	source, ok := value.(map[string]any)
	if !ok {
		return nil, false
	}
	cloned := make(map[string]any, len(source))
	for key, item := range source {
		switch typed := item.(type) {
		case map[string]any:
			copy, _ := cloneSchemaObject(typed)
			cloned[key] = copy
		case []string:
			cloned[key] = append([]string(nil), typed...)
		case []any:
			cloned[key] = append([]any(nil), typed...)
		default:
			cloned[key] = item
		}
	}
	return cloned, true
}

func centralAgentToolNames(mutatingEnabled, instanceOpsEnabled bool) []string {
	window := centralAgentToolWindow(mutatingEnabled, instanceOpsEnabled)
	names := make([]string, 0, len(window))
	for _, tool := range window {
		if tool.Function != nil && tool.Function.Name != "" {
			names = append(names, tool.Function.Name)
		}
	}
	return names
}

// ModelVisibleToolNames returns every tool name the model can be offered, over
// all runtime authorization/capability combinations, de-duplicated and sorted.
//
// It exists because the window is assembled from three unrelated sources —
// tools.Registry (+ the internal proposal template), capability.ReadDefinitions()
// (the "ReadCapability_" + intent family), and the Request<Operation> proposal
// tools generated from the write catalog. Presentation code covers this list instead
// of maintaining a second catalog, so a new tool without a display label fails tests.
func ModelVisibleToolNames() []string {
	seen := map[string]bool{}
	var names []string
	// Every gate the window is built from gets both values. The in-instance
	// SSH-ops lane added the second one after this function was written: with
	// only mutatingEnabled iterated, DiagnoseInstanceInternals never enters this
	// list, so the label test stops covering it and the lane ships a raw English
	// name into the activity stream — the exact failure this list exists to make
	// impossible. A new gate on centralAgentToolNames must be added here too.
	for _, mutatingEnabled := range []bool{false, true} {
		for _, instanceOpsEnabled := range []bool{false, true} {
			for _, name := range centralAgentToolNames(mutatingEnabled, instanceOpsEnabled) {
				if seen[name] {
					continue
				}
				seen[name] = true
				names = append(names, name)
			}
		}
	}
	sort.Strings(names)
	return names
}
