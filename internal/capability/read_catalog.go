package capability

import (
	"fmt"
	"reflect"
	"sort"

	"github.com/compshare-agent/internal/intent"
	openai "github.com/sashabaranov/go-openai"
)

const ReadToolPrefix = "ReadCapability_"

const (
	readCFSList           = "cfs_list"
	readCFSCreatePrice    = "cfs_create_price"
	readCFSUpgradePrice   = "cfs_upgrade_price"
	readCFSRefundEstimate = "cfs_refund_estimate"
)

type ReadDefinition struct {
	Name        string
	Intent      intent.Intent
	Description string
	Tool        openai.Tool
	// RequestType is the concrete request struct this capability decodes into.
	// The catalog consistency test reflects it against the tool schema so a field
	// added to the struct without the schema (or vice versa) fails loudly.
	RequestType reflect.Type
	decode      func(map[string]any) (ReadRequest, error)
	registered  RegisteredRead
}

func ReadToolName(id intent.Intent) string { return ReadToolPrefix + string(id) }

func ReadDefinitions() []ReadDefinition {
	definitions := []ReadDefinition{
		readDefinition(intent.Intent(WebSearchLabel), NewReadCapability(webSearchReadSpec())),
		readDefinition(intent.IntentResourceInfo, NewReadCapability(resourceReadSpec())),
		readDefinition(intent.IntentMonitorQuery, NewReadCapability(monitorCurrentReadSpec())),
		readDefinition(intent.IntentMonitorHistory, NewReadCapability(monitorHistoryReadSpec())),
		readDefinition(intent.IntentGPUSpecsQuery, NewReadCapability(gpuSpecsReadSpec())),
		readDefinition(intent.IntentStockAvailability, NewReadCapability(stockReadSpec())),
		readDefinition(intent.IntentImageList, NewReadCapability(imageListReadSpec())),
		readDefinition(intent.IntentImageTagCatalog, NewReadCapability(imageTagCatalogReadSpec())),
		readDefinition(intent.IntentZoneCatalog, NewReadCapability(zoneCatalogReadSpec())),
		readDefinition(intent.IntentModelRepositoryBrowse, NewReadCapability(modelRepositoryReadSpec())),
		readDefinition(intent.IntentNetAcceleratorStatus, NewReadCapability(netAcceleratorReadSpec())),
		readDefinition(intent.IntentInstanceAccess, NewReadCapability(instanceAccessReadSpec())),
		readDefinition(intent.IntentPricingQuery, NewReadCapability(pricingReadSpec())),
		readDefinition(intent.IntentRefundEstimate, NewReadCapability(refundReadSpec())),
		readDefinition(intent.IntentInvoiceStatus, NewReadCapability(invoiceStatusReadSpec())),
		readDefinition(intent.IntentCFSInfo, NewReadCapability(cfsListReadSpec())),
		readDefinition(intent.IntentCFSInfo, NewReadCapability(cfsCreatePriceReadSpec())),
		readDefinition(intent.IntentCFSInfo, NewReadCapability(cfsUpgradePriceReadSpec())),
		readDefinition(intent.IntentCFSInfo, NewReadCapability(cfsRefundEstimateReadSpec())),
		// Unavailable capability: the remaining account-level real-time financial
		// data is not queryable; the tool is model-visible so a balance/bill question gets a
		// deterministic non-fabricated answer plus supported alternatives.
		readDefinition(intent.Intent(accountFinanceStatusCapability), NewUnavailableCapability(accountFinanceUnavailableSpec())),
	}
	sort.Slice(definitions, func(i, j int) bool { return definitions[i].Name < definitions[j].Name })
	return definitions
}

func readDefinition(readIntent intent.Intent, reg RegisteredRead) ReadDefinition {
	registered := reg
	return ReadDefinition{
		Name: reg.Label, Intent: readIntent, Description: reg.Description, Tool: reg.Tool,
		RequestType: reg.RequestType(),
		decode:      func(args map[string]any) (ReadRequest, error) { return registered.Decode(args) },
		registered:  registered,
	}
}

func DecodeReadRequest(toolName string, args map[string]any) (intent.Intent, ReadRequest, error) {
	for _, definition := range ReadDefinitions() {
		if definition.Tool.Function != nil && definition.Tool.Function.Name == toolName {
			request, err := definition.decode(args)
			return definition.Intent, request, err
		}
	}
	return "", nil, fmt.Errorf("unknown read capability %q", toolName)
}

func ReadIntentForTool(name string) (intent.Intent, bool) {
	for _, definition := range ReadDefinitions() {
		if definition.Tool.Function != nil && definition.Tool.Function.Name == name {
			return definition.Intent, true
		}
	}
	return "", false
}

// RegisteredReadForTool returns the typed implementation for a read tool.
func RegisteredReadForTool(toolName string) (RegisteredRead, bool) {
	for _, definition := range ReadDefinitions() {
		if definition.Tool.Function != nil && definition.Tool.Function.Name == toolName {
			return definition.registered, true
		}
	}
	return RegisteredRead{}, false
}

// Parameter schemas are not written here: each capability declares a schemaNode
// field contract (field_contract.go) that is the single source for its tool
// schema, runtime validation and consistency-test expectation.
