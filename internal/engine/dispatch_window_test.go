package engine

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/compshare-agent/internal/capability"
	"github.com/compshare-agent/internal/intent"
	"github.com/compshare-agent/internal/platform"
	"github.com/compshare-agent/internal/prompt"
	"github.com/compshare-agent/internal/tools"
	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/require"
)

// TestProposalToolExposureAndMapping asserts ONLY that the catalog exposes the
// per-operation Request<Operation> tool and maps it back to its operation, and
// that the retired ProposeAction_<Operation> alias is gone. It does NOT — and its
// old name "FirstHopIsRequest" misleadingly implied it did — assert that the real
// model actually calls RequestCreateInstance first on a create turn: under the free
// ReAct loop that is a probabilistic model-behavior property, to be measured from
// real-model traces for acceptance, not something a tool-list test can prove.
func TestProposalToolExposureAndMapping(t *testing.T) {
	names := centralAgentToolNames(true, false)
	require.Contains(t, names, "RequestCreateInstance",
		"the create proposal tool must be advertised as RequestCreateInstance")
	require.NotContains(t, names, "ProposeAction_CreateInstanceWorkflow",
		"the retired ProposeAction_* alias must not be advertised to the model")

	op, ok := proposalOperationForTool("RequestCreateInstance")
	require.True(t, ok)
	require.Equal(t, "CreateInstanceWorkflow", op)

	_, ok = proposalOperationForTool("ProposeAction_CreateInstanceWorkflow")
	require.False(t, ok, "the retired alias must no longer resolve to an operation")
}

// Request tools carry an operation-specific boundary plus the shared interaction
// template. The template is derived at catalog construction from the capability
// registry, never from a workflow's internal execution-step description.
func TestRequestToolDescriptionUsesCapabilityBoundaryAndP2Template(t *testing.T) {
	var desc string
	for _, tool := range centralAgentToolWindow(true, false) {
		if tool.Function != nil && tool.Function.Name == "RequestCreateInstance" {
			desc = tool.Function.Description
			break
		}
	}
	require.NotEmpty(t, desc, "RequestCreateInstance must be advertised when mutating is enabled")
	capability, ok := tools.DefaultCapabilityRegistry().Lookup("CreateInstanceWorkflow")
	require.True(t, ok)
	require.Equal(t, tools.WorkflowAgentDescription(capability.AgentInstruction), desc)
	require.Contains(t, desc, "调用/边界：")
	require.NotContains(t, desc, "接续：")
	require.NotContains(t, desc, "输入示例")
	for _, internalAPI := range []string{"DescribeCompShare", "GetCompShare", "CreateCompShare", "SyncCompShare"} {
		require.NotContains(t, desc, internalAPI)
	}
}

func TestRequestReinstallDescriptionExcludesSoftwarePackageReinstallation(t *testing.T) {
	var description string
	for _, tool := range centralAgentToolWindow(true, false) {
		if tool.Function != nil && tool.Function.Name == "RequestReinstallInstance" {
			description = tool.Function.Description
			break
		}
	}
	require.NotEmpty(t, description)
	require.Contains(t, description, "重装实例操作系统或替换整个运行镜像")
	require.Contains(t, description, "不用于重装软件包或依赖")
	require.Contains(t, description, "具体影响见确认卡")
	require.NotContains(t, centralAgentToolNames(false, false), "RequestReinstallInstance",
		"clarifying the operation boundary must not expand write-tool availability")
}

func TestRequestToolDescriptionsDoNotRepeatSharedPromptOrExecutionChains(t *testing.T) {
	for _, tool := range centralAgentToolWindow(true, false) {
		if tool.Function == nil || !strings.HasPrefix(tool.Function.Name, "Request") {
			continue
		}
		desc := tool.Function.Description
		require.NotEmpty(t, desc, tool.Function.Name)
		require.Contains(t, desc, "调用/边界：", tool.Function.Name)
		require.NotContains(t, desc, "接续：", tool.Function.Name)
		require.NotContains(t, desc, "失败：", tool.Function.Name)
		require.NotContains(t, desc, "输入示例", tool.Function.Name)
		require.NotContains(t, desc, "服务端负责缺失字段", tool.Function.Name)
		require.NotContains(t, desc, "->", tool.Function.Name)
		require.NotContains(t, desc, "自动执行", tool.Function.Name)
		require.NotContains(t, desc, "确认式工作流", tool.Function.Name)
		for _, internalAPI := range []string{"DescribeCompShare", "GetCompShare", "CreateCompShare", "SyncCompShare"} {
			require.NotContains(t, desc, internalAPI, tool.Function.Name)
		}
	}
}

func TestKnowledgeToolExcludesCurrentPlatformFacts(t *testing.T) {
	var description string
	for _, tool := range centralAgentToolWindow(false, false) {
		if tool.Function != nil && tool.Function.Name == "SearchKnowledge" {
			description = tool.Function.Description
			break
		}
	}
	require.Contains(t, description, "平台当前目录")
	require.Contains(t, description, "对应只读能力")
}

// This checks the generated, model-visible boundary; choosing the correct
// object in a real conversation remains a real-model acceptance requirement.
func TestReinstallProposalDescribesWholeInstanceNotGuestPackages(t *testing.T) {
	var description string
	for _, tool := range centralAgentToolWindow(true, true) {
		if tool.Function != nil && tool.Function.Name == "RequestReinstallInstance" {
			description = tool.Function.Description
			break
		}
	}
	require.Contains(t, description, "重装实例操作系统或替换整个运行镜像")
	require.Contains(t, description, "不用于重装软件包或依赖")
}

func TestCentralAgentStaticPromptAndToolWindowStayWithinBudget(t *testing.T) {
	shapes := []struct {
		name        string
		mutating    bool
		instanceOps bool
	}{
		{name: "read_only"},
		{name: "production", mutating: true, instanceOps: true},
	}
	for _, shape := range shapes {
		system := prompt.BuildSystemWithOptions("context", prompt.BuildOptions{MutatingToolsEnabled: shape.mutating})
		toolJSON, err := json.Marshal(centralAgentToolWindow(shape.mutating, shape.instanceOps))
		require.NoError(t, err)
		t.Logf("mutating=%t instance_ops=%t system_bytes=%d system_runes=%d tool_bytes=%d tool_runes=%d total_bytes=%d total_runes=%d",
			shape.mutating, shape.instanceOps, len(system), len([]rune(system)), len(toolJSON), len([]rune(string(toolJSON))),
			len(system)+len(toolJSON), len([]rune(system))+len([]rune(string(toolJSON))))
		// Byte budgets are the measurement plus a small margin, never a round
		// number chosen to be safe: the system prompt measures at most 5512
		// (read-only shape). A raise must name what the bytes buy and whether a
		// probe against the real model showed it; a rule that reads well but
		// does not move the measured behavior is not a reason to spend prompt.
		require.LessOrEqual(t, len(system), 5600, "central system prompt grew past its reviewed byte budget")
		require.NotContains(t, system, "更新任务状态",
			"the retired semantic-memory tool must not remain as a model instruction")
		// The tool window is measured on the production shape, which includes
		// the SSH diagnosis tool; its max is 37860 bytes, including the separate
		// public technical web-search schema. Its caller/consumption probe uses a scripted model,
		// not a live-model routing measurement. Every model-visible
		// parameter states its own fill and omission rule, and identity fields
		// carry values the target tool accepts (both are pinned by their own
		// tests here); recover bytes by neither deleting that guidance nor
		// merging tools without production selection data.
		require.LessOrEqual(t, len(toolJSON), 37950, "model-visible tool window grew past its reviewed byte budget")
		// Measured max is 43202 (production shape).
		require.LessOrEqual(t, len(system)+len(toolJSON), 43300,
			"static prompt plus tool schemas grew past its reviewed byte budget")
	}
}

func TestOperationWithRequiredSensitiveInputIsNotAdvertised(t *testing.T) {
	for _, tool := range centralAgentToolWindow(true, false) {
		if tool.Function != nil && tool.Function.Name == "RequestResetPassword" {
			t.Fatal("an operation with no model-safe path for its required password must not be advertised")
		}
	}
}

func TestKnowledgeOnlyWindowExposesKnowledgeAndSupportHandoffOnly(t *testing.T) {
	names := toolNameSet(centralAgentKnowledgeToolWindow())
	require.Contains(t, names, "SearchKnowledge")
	require.Contains(t, names, "ReadChunk")
	require.Contains(t, names, tools.CustomerSupportHandoffName)

	for name := range names {
		require.NotContains(t, name, "Request", "public Q&A must not expose action proposals")
		require.NotContains(t, name, "DescribeCompShare", "public Q&A must not expose tenant resources")
		require.NotContains(t, name, "Diagnose", "public Q&A must not expose diagnoses")
	}
}

func TestKnowledgeOnlyExecutionAllowlistIsFailClosed(t *testing.T) {
	require.True(t, knowledgeOnlyToolAllowed("SearchKnowledge"))
	require.True(t, knowledgeOnlyToolAllowed("ReadChunk"))
	require.True(t, knowledgeOnlyToolAllowed(tools.CustomerSupportHandoffName))
	require.False(t, knowledgeOnlyToolAllowed("DescribeCompShareInstance"))
	require.False(t, knowledgeOnlyToolAllowed("DiagnoseInstanceInternals"))
	require.False(t, knowledgeOnlyToolAllowed("RequestStopInstance"))
	require.False(t, knowledgeOnlyToolAllowed("invented_tool"))
}

func TestKnowledgeOnlyExecutionBlocksUnadvertisedToolCall(t *testing.T) {
	eng := &Engine{turnState: turnState{knowledgeOnlyThisTurn: true}}
	var step StepEvent
	result := eng.executeTool(context.Background(), openai.ToolCall{
		Function: openai.FunctionCall{
			Name:      "DescribeCompShareInstance",
			Arguments: `{}`,
		},
	}, func(event StepEvent) {
		step = event
	})
	require.Equal(t, StepBlocked, step.Type)
	require.Contains(t, result.Observation, "仅允许查询知识库或转接人工客服")
}

func TestPublicPlatformReadOnlyWindowExposesEveryPublicQueryAndNothingElse(t *testing.T) {
	names := toolNameSet(centralAgentPublicPlatformReadOnlyToolWindow())
	require.Len(t, names, len(feishuPublicPlatformReadTools)+3,
		"the public Feishu window is exactly knowledge, support handoff, and the reviewed public read set")
	require.Contains(t, names, "SearchKnowledge")
	require.Contains(t, names, "ReadChunk")
	require.Contains(t, names, tools.CustomerSupportHandoffName)
	for name := range feishuPublicPlatformReadTools {
		require.Contains(t, names, name)
	}
	for _, definition := range capability.ReadDefinitions() {
		if definition.Tool.Function == nil || feishuPublicPlatformReadTools[definition.Tool.Function.Name] {
			continue
		}
		require.NotContains(t, names, definition.Tool.Function.Name,
			"non-mutating does not make a capability safe for an unauthenticated external group")
	}
	for name := range names {
		require.NotContains(t, name, "Request", "public Feishu must not expose action proposals")
		require.NotContains(t, name, "Diagnose", "public Feishu must not expose diagnoses")
		require.NotContains(t, name, "DescribeCompShare", "public Feishu must not expose tenant resources")
	}

	var imageParams, priceParams map[string]any
	for _, tool := range centralAgentPublicPlatformReadOnlyToolWindow() {
		if tool.Function == nil {
			continue
		}
		switch tool.Function.Name {
		case capability.ReadToolName(intent.IntentImageList):
			imageParams = tool.Function.Parameters.(map[string]any)
		case capability.ReadToolName(intent.IntentPricingQuery):
			priceParams = tool.Function.Parameters.(map[string]any)
		}
	}
	require.NotNil(t, imageParams)
	require.NotNil(t, priceParams)
	imageSource := imageParams["properties"].(map[string]any)["source"].(map[string]any)
	require.Equal(t, []string{string(platform.ImageSourcePlatform), string(platform.ImageSourceCommunity)}, imageSource["enum"])
	priceKind := priceParams["properties"].(map[string]any)["price_kind"].(map[string]any)
	require.Equal(t, []string{string(platform.PriceKindCatalog)}, priceKind["enum"])
}

func TestPublicPlatformReadOnlyExecutionBoundaryIsFailClosed(t *testing.T) {
	for name := range feishuPublicPlatformReadTools {
		require.True(t, publicPlatformReadOnlyToolAllowed(name), name)
	}
	require.True(t, publicPlatformReadOnlyToolAllowed("SearchKnowledge"))
	require.True(t, publicPlatformReadOnlyToolAllowed("ReadChunk"))
	require.True(t, publicPlatformReadOnlyToolAllowed(tools.CustomerSupportHandoffName))
	require.False(t, publicPlatformReadOnlyToolAllowed(capability.ReadToolName(intent.IntentResourceInfo)))
	require.False(t, publicPlatformReadOnlyToolAllowed(capability.ReadToolName(intent.IntentImageTagCatalog)))
	require.False(t, publicPlatformReadOnlyToolAllowed(capability.ReadToolName(intent.IntentNetAcceleratorStatus)))
	require.False(t, publicPlatformReadOnlyToolAllowed("DiagnoseInstanceInternals"))
	require.False(t, publicPlatformReadOnlyToolAllowed("RequestStopInstance"))
	require.False(t, publicPlatformReadOnlyToolAllowed("invented_tool"))

	imageName := capability.ReadToolName(intent.IntentImageList)
	require.True(t, publicPlatformReadOnlyArgsAllowed(imageName, map[string]any{"source": "platform"}))
	require.True(t, publicPlatformReadOnlyArgsAllowed(imageName, map[string]any{"source": "community"}))
	require.False(t, publicPlatformReadOnlyArgsAllowed(imageName, map[string]any{"source": "custom"}))
	require.False(t, publicPlatformReadOnlyArgsAllowed(imageName, map[string]any{"source": "shared"}))

	priceName := capability.ReadToolName(intent.IntentPricingQuery)
	defaultPrice := map[string]any{}
	require.True(t, publicPlatformReadOnlyArgsAllowed(priceName, defaultPrice))
	require.Equal(t, string(platform.PriceKindCatalog), defaultPrice["price_kind"])
	require.True(t, publicPlatformReadOnlyArgsAllowed(priceName, map[string]any{"price_kind": "catalog"}))
	require.False(t, publicPlatformReadOnlyArgsAllowed(priceName, map[string]any{"price_kind": "account"}))

	eng := &Engine{turnState: turnState{publicPlatformReadOnlyThisTurn: true}}
	var step StepEvent
	result := eng.executeTool(context.Background(), openai.ToolCall{Function: openai.FunctionCall{
		Name: capability.ReadToolName(intent.IntentResourceInfo), Arguments: `{}`,
	}}, func(event StepEvent) {
		step = event
	})
	require.Equal(t, StepBlocked, step.Type)
	require.Contains(t, result.Observation, publicPlatformReadOnlyBoundary)

	result = eng.executeTool(context.Background(), openai.ToolCall{Function: openai.FunctionCall{
		Name: priceName, Arguments: `{"price_kind":"account"}`,
	}}, func(event StepEvent) {
		step = event
	})
	require.Equal(t, StepBlocked, step.Type)
	require.Contains(t, result.Observation, "价格仅限目录价")
}

func TestChatWithOptionsUsesPublicPlatformWindowWithKnowledgeOnlyPrecedence(t *testing.T) {
	publicClient := &deltaMockLLM{}
	publicEngine := NewWithDeps(publicClient, &mockExecutor{}, nil)
	publicEngine.InitWithContext("用户当前没有实例。")

	_, err := publicEngine.ChatWithOptions(context.Background(), "A1000 有吗？", noopStep, ChatOptions{
		PublicPlatformReadOnly: true,
	})
	require.NoError(t, err)
	require.Len(t, publicClient.reqs, 1)
	require.Equal(t,
		toolNameSet(centralAgentPublicPlatformReadOnlyToolWindow()),
		toolNameSet(publicClient.reqs[0].Tools),
		"the per-turn public option must reach the actual model request")

	knowledgeClient := &deltaMockLLM{}
	knowledgeEngine := NewWithDeps(knowledgeClient, &mockExecutor{}, nil)
	knowledgeEngine.InitWithContext("用户当前没有实例。")
	_, err = knowledgeEngine.ChatWithOptions(context.Background(), "A1000 有吗？", noopStep, ChatOptions{
		KnowledgeOnly:          true,
		PublicPlatformReadOnly: true,
	})
	require.NoError(t, err)
	require.Len(t, knowledgeClient.reqs, 1)
	require.Equal(t,
		toolNameSet(centralAgentKnowledgeToolWindow()),
		toolNameSet(knowledgeClient.reqs[0].Tools),
		"the legacy strict knowledge-only option must retain precedence")
}
