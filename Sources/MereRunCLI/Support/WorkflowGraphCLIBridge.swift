import Foundation
import MereRunRelayKit

// The graph documents, node registry, and validator live in MereRunRelayKit
// and take plugin nodes as data. Only the CLI can discover plugin providers
// (discovery spawns local executables), so these extensions restore the
// discovery-backed convenience signatures the CLI has always used.

extension WorkflowNodeRegistry {
    static var catalogEntries: [WorkflowNodeCatalogEntry] {
        catalogEntries(pluginNodes: WorkflowGraphProviderRegistry.discoveredCatalog().nodes)
    }

    static func entry(for node: WorkflowNode) -> WorkflowNodeCatalogEntry? {
        entry(for: node, pluginNodes: WorkflowGraphProviderRegistry.discoveredCatalog().nodes)
    }

    static func output(node: WorkflowNode, name: String) -> WorkflowNodeOutput? {
        output(node: node, name: name, pluginNodes: WorkflowGraphProviderRegistry.discoveredCatalog().nodes)
    }

    static func provider(for node: WorkflowNode) -> WorkflowNodeProviderIdentity? {
        if node.resolvedProviderID == WorkflowNodeProviderIdentity.builtInID {
            return builtInProvider
        }
        return WorkflowGraphProviderRegistry.discoveredCatalog().provider(id: node.resolvedProviderID)?.identity
    }
}

extension WorkflowGraphValidator {
    static func validate(
        graph: WorkflowGraphDocument,
        inputs: WorkflowInputsDocument
    ) -> WorkflowGraphValidation {
        validate(
            graph: graph,
            inputs: inputs,
            pluginNodes: WorkflowGraphProviderRegistry.discoveredCatalog().nodes
        )
    }
}
