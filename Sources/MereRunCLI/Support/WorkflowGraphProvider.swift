import ArgumentParser
import Foundation
import MereRunCore
import MereRunRelayKit

struct WorkflowPluginGraphProviderDocument: Codable, Equatable, Sendable {
    static let contractVersion = "mere.run/plugin-graph-provider.v1"

    let contractVersion: String
    let providerID: String
    let providerVersion: String
    let nodes: [WorkflowNodeCatalogEntry]

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case providerID = "provider_id"
        case providerVersion = "provider_version"
        case nodes
    }
}

struct WorkflowDiscoveredGraphProvider: Equatable, Sendable {
    let identity: WorkflowNodeProviderIdentity
    let executable: String
    let nodes: [WorkflowNodeCatalogEntry]

    var requirement: WorkflowGraphProviderRequirement {
        .init(
            id: identity.id,
            version: identity.version,
            catalogSHA256: identity.catalogSHA256,
            nodeKinds: nodes.map(\.kind).sorted()
        )
    }
}

struct WorkflowDiscoveredGraphCatalog: Equatable, Sendable {
    let providers: [WorkflowDiscoveredGraphProvider]

    var nodes: [WorkflowNodeCatalogEntry] {
        providers.flatMap(\.nodes).sorted {
            ($0.provider?.id ?? "", $0.kind) < ($1.provider?.id ?? "", $1.kind)
        }
    }

    func provider(id: String) -> WorkflowDiscoveredGraphProvider? {
        providers.first { $0.identity.id == id }
    }
}

struct WorkflowGraphProviderInstallRegistry: Codable, Equatable, Sendable {
    static let filename = "graph-providers.json"

    var entrypoints: [String]
}

enum WorkflowGraphProviderRegistry {
    static let environmentKey = "MERERUN_GRAPH_PROVIDERS"
    static let knownOfficialEntrypoints = ["mere-dataset-tools"]
    private static let processCatalog = discover(
        environment: ProcessInfo.processInfo.environment,
        fileManager: .default
    )

    static var registryURL: URL {
        MereRunModelPaths.applicationSupportBase.appendingPathComponent(
            WorkflowGraphProviderInstallRegistry.filename,
            isDirectory: false
        )
    }

    static func register(entrypoint: String, fileManager: FileManager = .default) throws {
        var registry = loadInstallRegistry(fileManager: fileManager)
        if !registry.entrypoints.contains(entrypoint) {
            registry.entrypoints.append(entrypoint)
            registry.entrypoints.sort()
        }
        try fileManager.createDirectory(
            at: registryURL.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        try WorkflowBundleCodec.encoder().encode(registry).write(to: registryURL, options: .atomic)
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: registryURL.path)
    }

    static func discoveredCatalog(
        environment: [String: String]? = nil,
        fileManager: FileManager = .default
    ) -> WorkflowDiscoveredGraphCatalog {
        guard let environment else { return processCatalog }
        return discover(environment: environment, fileManager: fileManager)
    }

    private static func discover(
        environment: [String: String],
        fileManager: FileManager
    ) -> WorkflowDiscoveredGraphCatalog {
        let providers = candidateEntrypoints(environment: environment, fileManager: fileManager).compactMap { entrypoint in
            try? loadProvider(entrypoint: entrypoint)
        }
        var byID: [String: WorkflowDiscoveredGraphProvider] = [:]
        for provider in providers.sorted(by: { $0.executable < $1.executable }) {
            if byID[provider.identity.id] == nil {
                byID[provider.identity.id] = provider
            }
        }
        return WorkflowDiscoveredGraphCatalog(providers: byID.values.sorted { $0.identity.id < $1.identity.id })
    }

    static func requireProvider(id: String) throws -> WorkflowDiscoveredGraphProvider {
        guard let provider = discoveredCatalog().provider(id: id) else {
            throw ValidationError(
                "Graph provider '\(id)' is not installed. Install its companion plugin or set \(environmentKey)."
            )
        }
        return provider
    }

    static func loadProvider(entrypoint: String) throws -> WorkflowDiscoveredGraphProvider {
        let manifest = try PluginVerifier.verify(entrypoint: entrypoint)
        guard manifest.graphProvider?.contractVersion == WorkflowPluginGraphProviderDocument.contractVersion else {
            throw ValidationError("Plugin '\(manifest.name)' does not expose a graph node provider.")
        }
        let data = try PluginProcess.captureExecutable(
            entrypoint,
            arguments: ["graph", "catalog", "--json"]
        )
        let document = try WorkflowBundleCodec.decoder().decode(
            WorkflowPluginGraphProviderDocument.self,
            from: data
        )
        guard document.contractVersion == WorkflowPluginGraphProviderDocument.contractVersion else {
            throw ValidationError("Unsupported graph provider contract: \(document.contractVersion)")
        }
        guard document.providerID == manifest.name, document.providerVersion == manifest.version else {
            throw ValidationError("Graph provider catalog identity does not match its plugin manifest.")
        }
        guard document.providerID.range(
            of: "^mere-[a-z0-9-]+$",
            options: .regularExpression
        ) != nil else {
            throw ValidationError("Graph provider id is invalid: \(document.providerID)")
        }
        guard !document.nodes.isEmpty else {
            throw ValidationError("Graph provider '\(document.providerID)' has no nodes.")
        }
        let identity = WorkflowNodeProviderIdentity(
            id: document.providerID,
            version: document.providerVersion,
            catalogSHA256: try WorkflowBundleCodec.hash(document)
        )
        let nodes = document.nodes.map { node in
            WorkflowNodeCatalogEntry(
                kind: node.kind,
                title: node.title,
                description: node.description,
                category: node.category,
                inputs: node.inputs,
                outputs: node.outputs,
                requirements: node.requirements,
                traits: node.traits,
                presentation: node.presentation,
                provider: identity
            )
        }
        let duplicateKinds = Dictionary(grouping: nodes, by: \.kind).filter { $0.value.count > 1 }.keys
        guard duplicateKinds.isEmpty else {
            throw ValidationError(
                "Graph provider '\(document.providerID)' declares duplicate node kinds: \(duplicateKinds.sorted().joined(separator: ", "))."
            )
        }
        for node in nodes {
            guard node.kind.range(
                of: "^[a-z][a-z0-9-]{0,63}(\\.[a-z][a-z0-9-]{0,63})+$",
                options: .regularExpression
            ) != nil else {
                throw ValidationError("Graph provider '\(document.providerID)' declares invalid kind '\(node.kind)'.")
            }
            let duplicateInputs = Dictionary(grouping: node.inputs, by: \.name).filter { $0.value.count > 1 }.keys
            let duplicateOutputs = Dictionary(grouping: node.outputs, by: \.name).filter { $0.value.count > 1 }.keys
            guard duplicateInputs.isEmpty, duplicateOutputs.isEmpty else {
                throw ValidationError("Graph provider '\(document.providerID)' declares duplicate ports for '\(node.kind)'.")
            }
            for input in node.inputs {
                let rangeIsValid = if let minimum = input.minimum, let maximum = input.maximum {
                    minimum <= maximum
                } else {
                    true
                }
                guard input.name.range(of: "^[a-z][a-z0-9_]{0,63}$", options: .regularExpression) != nil,
                      rangeIsValid,
                      input.step.map({ $0 > 0 }) ?? true else {
                    throw ValidationError("Graph provider '\(document.providerID)' has an invalid input on '\(node.kind)'.")
                }
                if input.secret == true, input.type != .string {
                    throw ValidationError(
                        "Graph provider '\(document.providerID)' secret input '\(input.name)' must use type string."
                    )
                }
            }
            for output in node.outputs where output.name.range(
                of: "^[a-z][a-z0-9_]{0,63}$",
                options: .regularExpression
            ) == nil {
                throw ValidationError("Graph provider '\(document.providerID)' has an invalid output on '\(node.kind)'.")
            }
            let requirements = node.requirements
            guard requirements.minimumAcceleratorMemoryBytes.map({ $0 > 0 }) ?? true,
                  requirements.minimumSystemMemoryBytes.map({ $0 > 0 }) ?? true,
                  requirements.minimumDiskBytes.map({ $0 > 0 }) ?? true,
                  requirements.minimumCPUCores.map({ $0 > 0 }) ?? true else {
                throw ValidationError(
                    "Graph provider '\(document.providerID)' has invalid resource requirements on '\(node.kind)'."
                )
            }
        }
        return WorkflowDiscoveredGraphProvider(identity: identity, executable: entrypoint, nodes: nodes)
    }

    private static func candidateEntrypoints(
        environment: [String: String],
        fileManager: FileManager
    ) -> [String] {
        let configured = environment[environmentKey]?
            .split(separator: ":", omittingEmptySubsequences: true)
            .map(String.init) ?? []
        let persisted = loadInstallRegistry(fileManager: fileManager).entrypoints
        let knownInstalled = knownOfficialEntrypoints.filter { PluginProcess.which($0) != nil }
        var seen = Set<String>()
        return (configured + persisted + knownInstalled).filter { seen.insert($0).inserted }
    }

    private static func loadInstallRegistry(fileManager: FileManager) -> WorkflowGraphProviderInstallRegistry {
        guard fileManager.fileExists(atPath: registryURL.path),
              let data = try? Data(contentsOf: registryURL),
              let registry = try? WorkflowBundleCodec.decoder().decode(
                  WorkflowGraphProviderInstallRegistry.self,
                  from: data
              ) else {
            return WorkflowGraphProviderInstallRegistry(entrypoints: [])
        }
        return registry
    }
}
