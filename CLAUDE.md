# Compshare Agent

This repository contains the Go service behind the 优云智算 assistant. The
production entrypoint is the HTTP/WebSocket server in `cmd/`; there is no CLI
chat runtime.

## Build and test

```bash
go build ./...
go vet ./...
go test ./... -count=1

# Run the server with the checked-in baseline configuration.
go run ./cmd server -c deploy/conf/config.local.yaml
```

Before committing, also run `git diff --check`. The pre-commit hook invokes
`scripts/secret_scan.ps1` and needs PowerShell on `PATH`:

```bash
git config core.hooksPath .githooks
```

Live-model tests and PostgreSQL integration tests are opt-in. Their test files
name the required flags and credentials; a normal `go test ./...` must not make
external calls.

## Runtime architecture

The semantic runtime has one path:

```text
HTTP/WS request
  -> session engine from internal/agentpool
  -> engine.ChatWithOptions
  -> internal/agentruntime ReAct loop
  -> model tool call / tool result / final answer
  -> response gateway
  -> assistant row + canonical transcript metadata
```

`internal/agentruntime` owns round progression and runtime events. `Engine`
assembles product tools, context and policy around that loop. Do not add a
second planner, semantic router, hidden answerer or alternate turn executor.

Canonical transcript replay is always enabled. Each completed turn persists a
bounded `agent_transcript_v1` containing the model-visible user, assistant,
tool-call and tool-result messages. A later turn replays that transcript; it
does not reconstruct semantic history from summaries.

The model-visible context card contains only the current instance referent.
Do not reconstruct candidate order or semantic memory from the Agent's prose;
the actual conversation and tool transcript own that history. The citation
ledger is model-invisible and does not decide the final answer. Workflow forms,
confirmations and idempotency records are transaction/authorization state, not
semantic memory, and remain deterministic. Selected-instance provenance describes
the current referent; it does not authorize or overwrite a tool's target.

History is bounded by size, not message or exchange counts:

- `maxReplayedHistoryRunes` budgets prior exchanges.
- `maxAssembledRequestRunes` charges messages plus the serialized tool window.
- assembly removes oldest history before oldest current-turn tool groups;
  system messages, the current question and tool-call/result pairing remain
  intact.

ReAct result projection is always enabled for supported large read results. It
changes only the model-visible copy; trace keeps the full structured result and
records whether projection occurred.

## Tool and workflow boundaries

The model chooses read capabilities and knowledge retrieval in the loop.
Knowledge retrieval always goes to the remote MCP configured at
`agent.retrieval.mcp_url`; the corpus, its build pipeline and its release
publication live in `compshare-kb`, and this repository carries none of them.
Each `SearchKnowledge` executes the main Agent's query once, without an internal
planning model call. The Agent can search again within the four-call turn budget;
weak-candidate inspection, `ReadChunk`, and citation handling remain separate.

There is no keyword topic router or lexical jailbreak/off-topic pre-block in
front of the Agent. Scope belongs in the system prompt. Natural-language support
requests also reach the central Agent, which may call `HandoffToCustomerSupport`;
the active channel renders the actual support entry, so the model never authors
QR codes or adapter markers. A handoff never replaces evidence the turn already
gathered: when other tools ran in the turn, the Agent writes its answer and the
channel appends the entry after it; a handoff that is the turn's only call is
delivered at once. Only an explicit structured transport event may bypass
semantic interpretation.

Model-visible read capabilities live in `internal/capability/`. Each capability
owns its typed request, schema contract, handler and renderer. Do not recreate a
parallel route registry.

Mutating operations live in `internal/workflow/` and are proposed through the
central Agent. They remain subject to all of these controls:

1. the deployment authorization `agent.authorization.mutating_tools`;
2. exact account-scoped existence verification of the Agent-proposed target;
3. permission and policy checks;
4. a user confirmation card;
5. workflow idempotency and write-side revalidation.

Destructive/L2 actions stay refused regardless of the mutating-tools setting.
Do not turn these controls into prompt instructions or let the model attest its
own authorization.

The Agent resolves names, ordinals and conversational references separately for
each operation and supplies its target ID. The server does not derive a single
target from the whole turn, substitute a historical selection, or infer ambiguity
from the number of instances read. Each write still verifies its own target and
receives its own card.

The Agent also supplies image choice, capacities, names, time windows and query
filters. Validate scalar types/units and live platform facts, not whether the
same value appears in the current user message. Do not require source labels or
quotation spans, silently drop optional fields, or replace the Agent's image/GPU
choice. An exact image ID uses the live catalog; an unresolved choice uses the
existing guided form. Pre-creation validation failures return to the same Agent.

Editable confirmation forms and guided creation are stable protocol features.
The server advertises `confirm_form_v1` and `guided_create_v1`; a client must opt
in on the turn before it receives those shapes. They are not server rollout
flags.

The create confirmation is a projection of one sealed draft, including system
disk and price. After the write, readback mismatch, initialization failure or an
incomplete readback is reported deterministically rather than narrated as a
successful delivery; a sole returned instance becomes the existing current
instance referent without bypassing later confirmation gates.

Model-owned read-tool arguments rejected by schema or live-catalog
validation use
`status=needs_input,next_step=correct_tool_call` only with
`INVALID_TOOL_ARGUMENTS`. They must not be converted into a question to the
user. Incomplete provider output (`finish_reason` truncation or partial tool
calls) is never persisted or executed as a normal answer.

When the tool budget or loop ends after obtaining results, one tool-free closing
call retains the ordinary prompt, full task and bounded transcript. Do not build
a separate question/evidence-only answering task or replace a mixed answer with
one tool's no-data status. If the model cannot finish, report committed outcomes
through the existing terminal fallback.

## In-instance diagnosis

`DiagnoseInstanceInternals` is an optional autonomous SSH-ops lane. It requires:

- tenant-scoped STS credentials;
- a configured Python/Agent-SDK harness;
- audit migrations `0011`, `0013` and `0014`;
- the deployment grant `agent.authorization.mutating_tools=true`; and
- an Agent-selected `UHostId`, resolved exactly in the caller's STS account before
  credentials are fetched and bound to the run. The central Agent interprets
  conversation and chooses the target; server-side name/ordinal matching does not
  reinterpret that choice. A run never substitutes another instance or accepts a
  model-provided host/password. Confirmed platform workflows retain their own
  existence, business eligibility and confirmation gates.

Ordinary turn entry does not scan names to overwrite target state. Canonical
history preserves prior IDs and context; an actually reached or successful SSH
run records an `observed` referent, not a new `user_selected` proof. Missing or
inaccessible account targets do not establish verified context.

Once those server-owned conditions hold, the lane performs guest-local reversible
diagnosis, repair and verification without an entry card or per-command prompts.

The lane fails closed when audit storage is unavailable. A missing audit schema
disables only this lane and logs the missing migration; it does not take chat
down. Destructive effects, command substitution and guest-shell reboot remain
refused. Multiline commands, pipes and chains are classified by effect, and
guest changes remain bounded by that task-scope authorization. The definitive command policy
and deployment contract live in `deploy/ssh_ops_harness/README.md` and its
tests—do not duplicate their full history in config comments.

Production address routing is configured in `config.prod.yaml`: UHost internal
mapping first, the translated public-IPv4 candidate second. The advertised public
EIP is diagnostic-only and is never selected as a dial target.

If a browser disconnects during a diagnosis, the next turn may show a bounded
deterministic notice. Ordinary commands are never replayed. When one approved
managed background job emits its opaque handle, SessionState persists only
the instance ID, job ID, lifecycle state, bounded purpose and timestamp. A
later diagnosis on that instance can poll the handle after a browser disconnect,
Engine LRU eviction or process restart; neither the command nor its output enters
conversation/audit storage. Up to 32 unresolved handles are tracked independently;
a matching terminal observation clears only that job. Running services do not
prevent other installation or download jobs.

SessionState also keeps a same-instance opaque Agent SDK session UUID,
a stable opaque workdir UUID, and a content-free SHA-256 high-water mark for the outer conversation already
bridged into it. The SDK transcript stays in its existing local ephemeral store
and never enters PostgreSQL. A fresh inner session receives the canonical bounded
conversation — user and assistant endpoints plus the completed tool observations
of each turn, in order and in their canonical form — and the current user turn; a resume
receives only the new role-labelled suffix. The current turn's unanswered
`DiagnoseInstanceInternals` call and its planner arguments are never part of that
stream. A different instance, changed prompt/tool contract,
changed model, missing local transcript or a compacted-away anchor
starts fresh with the complete currently available snapshot.
Every resume forks the committed transcript into a new attempt UUID. Only a genuine
model-event receipt advances the persisted cursor, so authentication/transport failures
cannot append an uncommitted user prompt to the next retry.
Before the next serialized attempt, the harness retains only that committed source JSONL
inside the manifest-bound workdir and removes unreceipted fork JSONLs.

An SSH report returns to the central Agent as an ordinary tool observation, so
platform workflows and further Guest verification can compose within one turn.
How many runs one turn may start is bounded by the shared per-turn call budget
and read from the turn's own transcript; the lane keeps no side table of its own,
and a Guest report is never replayed from the reuse cache — it describes a
machine the run itself may have changed, so a retry re-enters instead of
receiving the previous attempt's partial command list. Durable replay refusal
lives in audit, keyed by the canonical tool-call ID within the existing turn
identity, with Task hashing for callers that supply no call ID. If parent
synthesis fails, already obtained reports remain deliverable without replaying
Guest commands.

## Configuration

`deploy/conf/config.local.yaml` is the shared deployment baseline.
`config.prod.yaml` extends it with production-network overrides. YAML is the
source of truth; environment variables remain compatibility fallbacks for the
small number of operational controls implemented by `RuntimeGetenv`.

Only genuine operational or authorization choices remain configurable:

| Setting | Purpose |
|---|---|
| `agent.authorization.mutating_tools` | authorize confirmation-gated product writes |
| `agent.llm.fallbacks` | models tried in order, each optionally on its own endpoint/key, when a request fails upstream |
| `agent.retrieval.mcp_*` | remote knowledge MCP endpoint/token/timeout |
| `agent.web_search.*` | disable public technical web queries or set their timeout |
| `agent.trace.*` | completed-turn trace sink and retention inputs |
| `agent.ssh_ops.*` | optional in-instance lane, permissions and network routing |
| `agent.ocr.*` | screenshot interpretation |
| `agent.feishu.*` | Feishu adapter and public-channel scope |
| `agent.rate_limit.*` | tenant request/token budgets |

Do not add a feature flag for code that has only one supported production
behavior. A rollback that requires restoring deleted semantics is a code revert,
not a boolean branch kept forever.

## HTTP and persistence

`cmd/server.go` exposes `POST /`, WebSocket chat, and `GET /healthz`. Gateway
identity comes from `top_organization_id`, `organization_id` and `request_uuid`;
business fields use the existing PascalCase API contract.

Per-session engines live in `internal/agentpool` (bounded LRU with idle expiry).
Persisted user/assistant rows rebuild a cold engine, and assistant metadata
reattaches the canonical tool transcript. Conversation text and tool
observations are persisted and replayed as the model saw them: the account's
own credentials returned by the platform (instance password, Jupyter token,
SSH login line) are ordinary facts on the model-visible path, no text is
scanned for credential-looking values, and the only rewrite of assistant text
is the adapter-marker replacement in `security.PersistedAssistantText`.
Redaction exists on two other paths only: trace copies redact by field name
(`security.RedactForTrace`, `internal/sanitizer`) before hashing, and
arguments the user enters through a confirmation card
(`ToolExecutionPolicy.SensitiveArgs`) stay off the model's proposal schema and
are masked on cards and step frames.

The store is PostgreSQL via `database/sql` and `lib/pq`. Historical names such
as `mysql`, `MYSQL_DSN`, `OpenMySQL` and `MySQLMessageStore` remain API/config
compatibility names; do not infer the database type from them. Apply all SQL
migrations in lexical order before deployment. See
`deploy/migrations/README.md`.

SSE final answers are attempt-atomic and final-answer-atomic, not true upstream
token streaming: failed retries and incomplete output must never leak as a
prefix. Tool/confirmation activity uses separate step frames.

One model call makes one actual request per configured model (at least two,
so a single model keeps its retry). Each `agent.llm.fallbacks` entry is a
further model, optionally on its own endpoint and key; a request that fails
upstream goes to the next one instead of the pool that just failed. The client
keeps no health state between calls: every call starts on the primary again.
A request rejection (4xx, or an in-stream `invalid_request_error`) is never
routed. Each attempt is traced with the model it went to; there is no
turn-level "fallback used" field.

## Observability

`internal/httpapi/trace_recorder.go` writes one content-free trace per completed
turn through `internal/observability`. That record is the root run. Model
attempts, tool calls and retrieval activities are typed child operations;
confirmations and authorizations are terminal events. Keep one representation
of each observed fact: provider/model/finish reason belong to the model attempt,
and tool status/latency belong to the tool call rather than turn-level derived
copies.

Runtime Trace records execution facts: timings, closed-set status/error values,
token counts, result truncation sizes, prompt section IDs, request-size peaks,
selected-instance provenance and deterministic workflow outcomes. Correctness,
faithfulness, completeness and hallucination scores belong to an external eval
joined by trace ID; do not add an online judge or grader fields to the runtime
record. Trace must not become a second conversation database: no raw prompt,
reply, tool payload, credential or canonical transcript.

When adding a tool failure, use the existing closed error-code/error-class
contracts rather than parsing error messages. Preserve three-state metrics where
absence differs from zero.

## Repository conventions

- The central system prompt is assembled from `internal/prompt/segments.go`.
  Follow `docs/dev/prompt-change-checklist.md` for prompt changes.
- Platform facts come from tools or RAG. Do not add static FAQ text or model-name
  mappings to the system prompt.
- `internal/entity` is race-tested in CI; keep registry access synchronized.
- `SecurityToken` participates in API signing parameters before HMAC-SHA1. See
  `internal/tools/README.md`.
- Do not add historical inventories or rollout checklists to production code.
  Current architectural decisions belong in this file; a code comment should
  explain only the invariant adjacent to its implementation, never what the code
  used to do or which change replaced it.
- Preserve unrelated dirty-worktree changes. Stage explicit files, never
  `git add -A` in a shared workspace.
