# Agent Runtime Terms

This document defines the runtime terms used by the Youyun CompShare console
agent. Its purpose is to keep the current primitives — the central Agent loop,
typed read capabilities, the action resolver, sealed workflows, typed
observations, the response gateway, RAG-as-a-tool, guardrails, and MCP —
separate in both code and review language.

## Terms

### Tool

A tool is a typed callable action. It has a name, input schema, policy level, and
execution result. Console API wrappers (instance description, pricing, stock,
image listing) and the mutating API calls are tools; `SearchKnowledge` is the
read-only knowledge-retrieval tool.

Tools are not playbooks. The central Agent or a deterministic workflow may call
them, but the tool definition itself is not a set of instructions to read.

### Central Agent

The central Agent is the single reasoning point (`internal/engine/`). It runs one
ReAct-style loop over a compiled `AgentContext`: each round it selects a read
capability, calls a knowledge tool, or proposes a write, and observes the typed
result in the same loop. There is no separate router and no per-request choice
between three execution forms.

### Read Capability

A read capability is a typed, model-visible read vertical
(`internal/capability/read_*.go`). Each owns its request struct, its **field
contract** (`field_contract.go`'s `schemaNode` — the single source for the tool
schema, runtime validation, and the consistency test), its handler, and its
renderer. `ReadDefinitions()` is the catalog; the engine dispatches through
`executeConcreteReadCapability` → `capability.RegisteredReadForTool(action)` →
`RegisteredRead.Run`. There is no route registry.

### Action Resolver

The action resolver (`internal/actionresolver/`) deterministically resolves the
target instance and spec of a proposed write (e.g. whether an image catalog must
be re-queried, via `SpecNeedsImageCatalog`). A write target is authorized only
when the user's reply deterministically resolves to it — the model does not
"interpret a candidate list" in place of deterministic target selection.

### Sealed Workflow

A sealed workflow (`internal/workflow/`) is a deterministic write process with
confirmation. It has a fixed step sequence, code-enforced safety checks, and no
free-form mutating tool calls by the model. Deployment, disk changes, lifecycle
ops (start/stop/reboot/reset-password/rename) and custom image creation are
sealed workflows. The `SealedActionContract` separates the confirmed action from
runtime metadata, and volatile fields (e.g. image) are re-confirmed before
execution.

Mutation stays behind workflow code and confirmation gates; a read capability or
`SearchKnowledge` may help explain or diagnose, but never mutates.

### Typed Observation

A tool result is not free text. A read capability returns a
`ReadCapabilityObservation` carrying status and a structured evidence envelope.
The Agent reasons over those facts and authors the final Markdown itself. The
ordinary read renderer's text is not a user-visible block and is never appended
to the Agent response a second time.

### Response Gateway

The response gateway (`engine`'s `finalizeResponse`) does not substitute or
append ordinary read results. It enforces the never-0% monitor invariant
(all-no-data historical monitor → whole-answer "cannot confirm", never
0%/healthy) and is the narrow server-side delivery path for credentials that
must not enter model context.

Note: ordinary read fidelity is intentionally delegated to the Agent, grounded
in the envelope facts; it is not a verbatim-insertion contract. The configured-
model monitor acceptance checks the reply against those facts without prescribing
the reply's wording or layout.

### RAG Evidence

RAG is retrieval used **inside** the Agent loop, via the `SearchKnowledge` tool —
not a terminal answer form. The Agent chooses when to search and supplies the
query; each call executes that query once, without an internal planning model
call, against the configured CompShare KB MCP endpoint. The result contains
cited chunks the Agent grounds its answer in. Citation discipline is **fail-open**:
if the Agent cannot cite, the original answer ships with citation markers stripped
— citation formatting never regenerates user-facing prose. The only hard stop is
a raw-evidence leak (security). Citation-marker leakage into the final text is
caught by the output guard.

### Diagnosis Chain

A diagnosis chain (`internal/diagnosis/`) is a read-only diagnostic tool.
`DiagnoseBilling` is the only advertised one; `chainRegistry` equals the
advertised set, so an unadvertised diagnosis name cannot resolve (model-invisible
≠ unreachable). The SSH chain still exists and still runs, but not as a
`DiagnoseSSH` tool — `internal/capability/read_instance_access.go` constructs it
directly, so SSH reaches the model as `ReadCapability_instance_access`. That
precheck is explicitly cloud-side: it verifies the exact instance, lifecycle
state, structured login endpoint, and monitor risk signals, but does not probe a
public port or inspect the guest OS. Symptoms without a dedicated chain
(GPU/init/port/image) are handled by the central Agent gathering evidence via
`SearchKnowledge` + `DescribeCompShareInstance`.

### Conversation History and Execution State

The ordered conversation and canonical tool transcript are the model's semantic
history. Persisted session state holds only execution continuity (selection
provenance, pending selection, form/confirmation state) plus verifier-only
evidence; it does not preserve a second summary of decisions or tool facts.
Current-state verification remains mandatory before a write.

### Guardrail

A guardrail is a code-enforced safety rule: destructive-action blocks, mutating
confirmation, tool policy checks, provenance gates, citation validation and
output redaction. It protects an execution or data boundary; it does not classify
the user's topic before the Agent. Model-generated risk assessments are
observability signals, not the source of truth for write safety.

### MCP Client

MCP is a transport protocol, not another Agent layer. This service is an MCP
client for the configured CompShare knowledge endpoint and the fixed Parallel
Search MCP endpoint. The typed `ReadCapability_web_search` read capability
returns public technical references through the same central Agent loop. It is
independent of `SearchKnowledge`: web results do not establish CompShare rules,
account facts or Guest state. It does not expose a general MCP tool server or
dynamically import arbitrary remote tools.

## Runtime Shape

```text
user request
  -> central Agent loop (engine): each round selects one of
       read capability  |  SearchKnowledge  |  propose write -> resolver -> sealed workflow
  -> response gateway (safety invariants / sensitive credential delivery)
  -> output guard (sanitizer / policy)
```

There is exactly one execution shape — the central Agent loop. Reads, knowledge
retrieval, and writes are choices *within* the loop, not separate top-level
pipelines.

## Naming Rules

- Use `central Agent` for the single engine loop; do not say "router" or "tier".
- Use `read capability` for a typed model-visible read vertical.
- Use `action resolver` for deterministic write-target/spec resolution.
- Use `sealed workflow` for deterministic mutating flows with confirmation.
- Use `typed observation` / `response gateway` for the result → render path.
- Use `RAG evidence` for retrieval consumed inside the loop (`SearchKnowledge`).
- Use `tool` for typed callable actions.
- Use `MCP server` / `MCP client` by direction; do not call MCP an agent layer.
