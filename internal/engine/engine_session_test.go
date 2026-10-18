package engine

import (
	"reflect"
	"testing"

	"github.com/compshare-agent/internal/governance"
	"github.com/compshare-agent/internal/tools"

	openai "github.com/sashabaranov/go-openai"
)

// newTwoSessions constructs two Engines from the same SharedDeps for the
// isolation tests below. mockLLM / mockExecutor live in engine_test.go (same
// package) so we reuse them rather than declaring a parallel stub.
func newTwoSessions(t *testing.T) (engA, engB *Engine, deps *SharedDeps) {
	t.Helper()
	deps = &SharedDeps{
		LLMClient:        &mockLLM{},
		RateLimiter:      governance.NewInMemoryRateLimiter(governance.DefaultLimits()),
		ExternalExecutor: &mockExecutor{results: map[string]map[string]any{}},
	}
	engA = NewSession(deps, SessionOptions{Subject: "subj-A"})
	engB = NewSession(deps, SessionOptions{Subject: "subj-B"})
	return engA, engB, deps
}

// TestSessionIsolation_Messages: messages串了的后果是 user B 看到 user A 原话。This test injects
// a marker into session A's messages slice and asserts session B never sees it,
// even though both sessions share the same SharedDeps. Encodes WHY: cross-user
// data leak is the highest-severity failure mode of single-replica multi-tenant
// deployment.
func TestSessionIsolation_Messages(t *testing.T) {
	engA, engB, _ := newTwoSessions(t)

	const secret = "SECRET-A-PAYLOAD-12345"
	engA.messages = append(engA.messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: secret,
	})

	for _, m := range engB.MessagesSnapshot() {
		if m.Content == secret {
			t.Fatalf("session B leaked session A's message: %q", m.Content)
		}
	}
	// Sanity: A actually has the message — otherwise this test would pass by
	// vacuously empty snapshots.
	gotA := engA.MessagesSnapshot()
	if len(gotA) == 0 || gotA[len(gotA)-1].Content != secret {
		t.Fatalf("session A lost its own message; got %#v", gotA)
	}
}

// TestSessionIsolation_Registry: registry串了的后果是 user B 操作到 user A 的实例（越权）。
// EntityRegistry has no public mutation except SyncFromDescribe — the test
// injects two disjoint UHostSet maps and asserts neither registry sees the
// other's instances. Encodes WHY: entity confusion enables cross-tenant write.
func TestSessionIsolation_Registry(t *testing.T) {
	engA, engB, _ := newTwoSessions(t)

	if err := engA.RegistryPointer().SyncFromDescribe(map[string]any{
		"TotalCount": 1,
		"UHostSet": []any{
			map[string]any{
				"UHostId": "uhost-aaa",
				"Name":    "instance-A",
				"Zone":    "cn-wlcb-01",
				"State":   "Running",
			},
		},
	}, "test-session-A-init"); err != nil {
		t.Fatalf("seed session A registry: %v", err)
	}

	if err := engB.RegistryPointer().SyncFromDescribe(map[string]any{
		"TotalCount": 1,
		"UHostSet": []any{
			map[string]any{
				"UHostId": "uhost-bbb",
				"Name":    "instance-B",
				"Zone":    "cn-wlcb-01",
				"State":   "Running",
			},
		},
	}, "test-session-B-init"); err != nil {
		t.Fatalf("seed session B registry: %v", err)
	}

	snapB := engB.RegistryPointer().Snapshot()
	if _, leaked := snapB.Instances["uhost-aaa"]; leaked {
		t.Fatalf("session B registry leaked session A's instance: %#v", snapB.Instances)
	}
	if _, kept := snapB.Instances["uhost-bbb"]; !kept {
		t.Fatalf("session B lost its own instance; got %#v", snapB.Instances)
	}

	snapA := engA.RegistryPointer().Snapshot()
	if _, leaked := snapA.Instances["uhost-bbb"]; leaked {
		t.Fatalf("session A registry leaked session B's instance: %#v", snapA.Instances)
	}

	// Registries must be DIFFERENT pointer instances.
	if engA.RegistryPointer() == engB.RegistryPointer() {
		t.Fatalf("session A and B share the same registry pointer; per-session isolation broken")
	}
}

// TestSessionIsolation_ConfirmFn: confirmFn串了的后果是 user A 的确认弹窗去问 user B（误操作）。
// Each NewSession receives its own ConfirmFunc. The test wires two functions
// with disjoint side effects and asserts session A invoking confirm never
// triggers session B's callback.
func TestSessionIsolation_ConfirmFn(t *testing.T) {
	deps := &SharedDeps{
		LLMClient:        &mockLLM{},
		RateLimiter:      governance.NewInMemoryRateLimiter(governance.DefaultLimits()),
		ExternalExecutor: &mockExecutor{results: map[string]map[string]any{}},
	}
	var calledA, calledB bool
	confirmA := func(action string, args map[string]any) bool { calledA = true; return true }
	confirmB := func(action string, args map[string]any) bool { calledB = true; return false }

	engA := NewSession(deps, SessionOptions{Subject: "subj-A", ConfirmFn: confirmA})
	engB := NewSession(deps, SessionOptions{Subject: "subj-B", ConfirmFn: confirmB})

	if engA.confirmFn == nil {
		t.Fatalf("session A confirmFn unexpectedly nil")
	}
	engA.confirmFn("StopInstance", nil)
	if !calledA {
		t.Fatalf("session A confirm callback was not invoked")
	}
	if calledB {
		t.Fatalf("session A's confirm call leaked into session B's callback")
	}

	// confirmFn pointers must differ — if NewSession had captured a single
	// function from a process-wide var, both engines would share it.
	if reflect.ValueOf(engA.confirmFn).Pointer() == reflect.ValueOf(engB.confirmFn).Pointer() {
		t.Fatalf("session A and B share confirmFn pointer; per-session wiring broken")
	}
}

// TestSessionIsolation_SharedPointersEqual is the sibling assertion to the per-session checks: shared fields MUST be pointer-
// equal across sessions. If a session refactor accidentally copies an LLM
// client or RateLimiter, this test will catch it. Encodes WHY: shared deps
// hold no per-session state, and copying defeats the purpose of NewSharedDeps.
func TestSessionIsolation_SharedPointersEqual(t *testing.T) {
	engA, engB, deps := newTwoSessions(t)

	if engA.LLMClientPointer() != deps.LLMClient {
		t.Fatalf("session A LLMClient pointer drift")
	}
	if engA.LLMClientPointer() != engB.LLMClientPointer() {
		t.Fatalf("LLMClient must be shared across sessions; got %p vs %p",
			engA.LLMClientPointer(), engB.LLMClientPointer())
	}
	if engA.RateLimiterPointer() != engB.RateLimiterPointer() {
		t.Fatalf("RateLimiter must be shared across sessions; got %p vs %p",
			engA.RateLimiterPointer(), engB.RateLimiterPointer())
	}
}

// TestSessionIsolation_RateLimit: per-user subject keys must isolate quota burn
// across tenants.
// Sets distinct subjects on two sessions, burns session A's LLM bucket, and
// asserts session B's first LLM request still succeeds. If subjects shared
// a bucket (a process-wide subject), this would fail.
func TestSessionIsolation_RateLimit(t *testing.T) {
	engA, engB, _ := newTwoSessions(t)
	engA.SetRateLimitSubject("rl-subj-A")
	engB.SetRateLimitSubject("rl-subj-B")

	// InMemoryRateLimiter default LLMQPS = 5; burn 5 to drain session A's bucket
	// then expect the 6th call to be denied.
	for i := 0; i < governance.DefaultLLMQPS; i++ {
		if dec, _ := engA.allowRateLimited(governance.ClassLLM, "main_react_chat"); !dec.Allowed {
			t.Fatalf("session A: expected first %d LLM calls to succeed, denial at %d", governance.DefaultLLMQPS, i+1)
		}
	}
	denied, _ := engA.allowRateLimited(governance.ClassLLM, "main_react_chat")
	if denied.Allowed {
		t.Fatalf("session A: expected LLM bucket exhaustion after %d calls", governance.DefaultLLMQPS)
	}

	// Session B's first call must succeed — its bucket is independent.
	bDec, _ := engB.allowRateLimited(governance.ClassLLM, "main_react_chat")
	if !bDec.Allowed {
		t.Fatalf("session B was denied LLM call but its bucket was fresh; subjects not isolated")
	}
}

// TestSessionIsolation_AllEngineFieldsClassified — reflection guard.
// Every Engine struct field MUST be classified as shared or per-session. A
// field added without classification fails here, because a silent addition
// defeats the cross-session isolation guarantee.
//
// The embedded turnState needs no per-field entry. Everything inside it belongs
// to one turn of one session, which is strictly narrower than per-session, and
// ChatWithOptions replaces the whole value at entry — so a turn-local field
// cannot reach another session, and cannot reach the next turn of this one
// either. Adding a per-turn field there is the correct move and costs no
// bookkeeping; adding one to Engine still has to be justified below.
func TestSessionIsolation_AllEngineFieldsClassified(t *testing.T) {
	sharedFields := map[string]bool{
		"webSearch":          true, // immutable client shared across sessions; no conversation state
		"llmClient":          true,
		"knowledgeRetriever": true,
		"rateLimiter":        true,
		"maxTokensPerTurn":   true,
		// externalExecutor is the RAW shared tool executor (same instance as the
		// one safeExecutor wraps) — pointer-equal across sessions, used only for
		// read-only L0 catalog calls. Shared like llmClient.
		"externalExecutor": true,
		// zoneCatalog is the process-wide zone-display-name cache (or nil →
		// zones.Default()). A shared read-mostly cache, not per-session state.
		"zoneCatalog": true,
	}
	perSessionFields := map[string]bool{
		"webSearchSessionID":               true, // opaque correlation ID; never shared across sessions
		"safeExecutor":                     true,
		"confirmFn":                        true,
		"confirmEditsFn":                   true,
		"registry":                         true,
		"rateLimitSubject":                 true,
		"mutatingToolsEnabled":             true,
		"messages":                         true,
		"userTurn":                         true,
		"lastConfirmationAcceptedThisCall": true,
		"retrievalTraceObserver":           true,
		"authorizationTraceObserver":       true,
		// Confirmation outcomes are turn-scoped transport facts. Sharing this
		// observer would append one tenant's card result to another's trace.
		"confirmationTraceObserver": true,
		"tokenUsageObserver":        true,
		"rateLimitObserver":         true,
		"hardBlockObserver":         true,
		"turnCompletionObserver":    true,
		// currentCtx and currentTurnID below are the two turn-scoped fields that
		// stay on Engine rather than moving into turnState: they are cleared when
		// the call returns, and their absence between turns is what stops an
		// out-of-turn caller from inheriting a live context or an audit identity.
		"currentCtx": true,
		// guidedCreate is a per-turn HTTP capability gate; sharing it would let
		// one client's opt-in change another client's create workflow shape.
		"guidedCreate": true,
		// M1 SessionState fields — per-session by design. SessionState is
		// the JSON-serializable per-session dialog state envelope; mixing
		// it across sessions would be exactly the cross-user leak this
		// test was created to prevent.
		"sessionState":         true,
		"sessionStateVersion":  true,
		"sessionStateHydrated": true,
		"baseUserContext":      true,
		// In-instance SSH diagnosis lane (INV-9/INV-11). instanceOps is copied from
		// SharedDeps but is per-session-overridable via SetInstanceOps for tests,
		// so a session can hold a different runner than its siblings — classified
		// per-session, not shared like externalExecutor. How many runs one turn may
		// start is read from the turn's own transcript, so the lane keeps no
		// per-turn side table of its own.
		"instanceOps":   true,
		"currentTurnID": true,
		// The notice left by a diagnosis that ended without a verdict, drained by the
		// next turn. Per-session and NOT turn-local — it deliberately outlives the turn
		// that created it, which is the whole point — and emphatically not shared: it
		// names another tenant's instance id and the literal commands run inside their
		// box, so a shared field would show tenant A's half-finished repair to tenant B
		// as if it were their own. Cleared on delivery, never carried further.
		"pendingInstanceOpsInterruption": true,
		// The opaque guest-job handle lives inside the already-classified
		// SessionState field, so it follows the same owner/session hydration boundary
		// instead of adding a second implicit state source here.
		// The canonical transcript of the turn that just finished, held for the
		// metadata write. It carries one tenant's tool arguments and tool results
		// verbatim, so a shared field would persist tenant A's instance ids and
		// diagnosis output onto tenant B's assistant row — precisely the leak
		// this test exists to prevent. Its stats sibling is classified with it so
		// the two cannot drift apart. Both are rewritten at every turn exit.
		"lastTurnTranscript":      true,
		"lastTurnTranscriptStats": true,
		// The cross-turn memory the canonical transcript replaces stripped
		// history with. It holds prior turns' tool arguments and results
		// verbatim and is replayed into the prompt, so a shared field would put
		// one tenant's instance ids and diagnosis output into another tenant's
		// context window — the same leak as its two siblings above, but on the
		// path the model actually reads.
		"recentTurns": true,
	}

	// The fixed counts make an unclassified Engine field fail even when it is
	// accidentally omitted from both maps.
	if want, got := 7, len(sharedFields); want != got {
		t.Fatalf("shared whitelist count drift: expected %d, got %d", want, got)
	}
	if want, got := 29, len(perSessionFields); want != got {
		t.Fatalf("per-session whitelist count drift: expected %d, got %d", want, got)
	}

	typ := reflect.TypeOf(Engine{})
	if want, got := len(sharedFields)+len(perSessionFields)+1, typ.NumField(); want != got {
		t.Fatalf("Engine field count drift: expected %d (whitelists plus the embedded turnState), got %d. "+
			"A new per-turn field belongs in turnState; anything else needs a whitelist entry.", want, got)
	}
	for i := 0; i < typ.NumField(); i++ {
		name := typ.Field(i).Name
		if name == "turnState" {
			if !typ.Field(i).Anonymous {
				t.Errorf("turnState must stay embedded: its fields are read as e.fooThisTurn across the package")
			}
			continue
		}
		if sharedFields[name] || perSessionFields[name] {
			continue
		}
		t.Errorf("Engine field %q is not classified as shared or per-session. "+
			"If it belongs to one turn, move it into turnState instead of adding it here.", name)
	}
}

// TestNewWithDeps_FieldSetMatchesNewSession — §5.3 option (a) guard.
// NewWithDeps keeps its existing signature (used by ~1500 lines of tests) and
// constructs an Engine directly rather than going through NewSession. This
// test asserts NewWithDeps produces a field set equivalent to
// NewSession(deps, SessionOptions{MutatingToolsEnabled: true}) so the two
// construction paths cannot
// silently drift. Encodes WHY: future field additions must land in both
// constructors or this test fails — preventing the cross-session leak that
// motivated §3's classification work.
func TestNewWithDeps_FieldSetMatchesNewSession(t *testing.T) {
	llm := &mockLLM{}
	exec := &mockExecutor{results: map[string]map[string]any{}}
	confirm := func(string, map[string]any) bool { return true }

	withDeps := NewWithDeps(llm, exec, confirm)

	session := NewSession(&SharedDeps{
		LLMClient:        llm,
		ExternalExecutor: exec,
	}, SessionOptions{
		ConfirmFn:            confirm,
		MutatingToolsEnabled: true,
	})

	if withDeps.llmClient != session.llmClient {
		t.Errorf("llmClient pointer differs: NewWithDeps=%p NewSession=%p", withDeps.llmClient, session.llmClient)
	}
	if withDeps.mutatingToolsEnabled != session.mutatingToolsEnabled {
		t.Errorf("mutatingToolsEnabled differs: NewWithDeps=%v NewSession=%v",
			withDeps.mutatingToolsEnabled, session.mutatingToolsEnabled)
	}
	if withDeps.registry == nil {
		t.Errorf("NewWithDeps did not init registry")
	}
	if session.registry == nil {
		t.Errorf("NewSession did not init registry")
	}
	if withDeps.safeExecutor == nil || session.safeExecutor == nil {
		t.Errorf("safeExecutor not initialized: NewWithDeps=%v NewSession=%v",
			withDeps.safeExecutor != nil, session.safeExecutor != nil)
	}
}

// TestNewSession_NilDepsPanics rejects a half-initialized session.
func TestNewSession_NilDepsPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("NewSession(nil, _) should panic but did not")
		}
	}()
	_ = NewSession(nil, SessionOptions{})
}

// ProjectId is constructor state on the process-wide executor. A setter or
// getter would let one session affect another tenant's later requests.
func TestSessionIsolation_NoProjectIdLeak(t *testing.T) {
	typ := reflect.TypeOf(&tools.ExternalExecutor{})
	for _, banned := range []string{"SetProjectId", "ProjectId"} {
		if _, ok := typ.MethodByName(banned); ok {
			t.Fatalf("tools.ExternalExecutor.%s must not expose mutable process-wide ProjectId state", banned)
		}
	}
}

// TestNewSharedDeps_NilCfgErrors asserts the documented error in NewSharedDeps.
func TestNewSharedDeps_NilCfgErrors(t *testing.T) {
	deps, err := NewSharedDeps(nil)
	if err == nil {
		t.Fatalf("NewSharedDeps(nil) returned err=nil, deps=%v", deps)
	}
	if deps != nil {
		t.Fatalf("NewSharedDeps(nil) returned non-nil deps on error: %+v", deps)
	}
}
