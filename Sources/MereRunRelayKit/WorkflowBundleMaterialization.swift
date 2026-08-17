import Crypto
import Foundation

public struct WorkflowBundleMaterialization: Equatable {
    public let directory: URL
    public let graph: WorkflowGraphDocument
    public let inputs: WorkflowInputsDocument
    public let assets: WorkflowAssetManifest
    public let job: WorkflowJobManifest

    public init(
        directory: URL,
        graph: WorkflowGraphDocument,
        inputs: WorkflowInputsDocument,
        assets: WorkflowAssetManifest,
        job: WorkflowJobManifest
    ) {
        self.directory = directory
        self.graph = graph
        self.inputs = inputs
        self.assets = assets
        self.job = job
    }
}

/// The environment-specific lookups a bundle materialization needs. The CLI
/// supplies its managed model catalog, per-command default model ids, and
/// discovered plugin providers; clients without a local runtime use
/// `.portable`, which records identity-only provenance and knows no plugin
/// nodes, and therefore always pin models explicitly in the graph.
public struct WorkflowMaterializationEnvironment: Sendable {
    public let mereRunVersion: String
    public let pluginNodes: [WorkflowNodeCatalogEntry]
    public let defaultModelID: @Sendable (_ nodeKind: String) -> String?
    public let modelProvenance: @Sendable (_ modelID: String) -> WorkflowModelProvenance
    public let providerRequirement: @Sendable (_ providerID: String) -> WorkflowGraphProviderRequirement?

    public init(
        mereRunVersion: String,
        pluginNodes: [WorkflowNodeCatalogEntry],
        defaultModelID: @escaping @Sendable (_ nodeKind: String) -> String?,
        modelProvenance: @escaping @Sendable (_ modelID: String) -> WorkflowModelProvenance,
        providerRequirement: @escaping @Sendable (_ providerID: String) -> WorkflowGraphProviderRequirement?
    ) {
        self.mereRunVersion = mereRunVersion
        self.pluginNodes = pluginNodes
        self.defaultModelID = defaultModelID
        self.modelProvenance = modelProvenance
        self.providerRequirement = providerRequirement
    }

    public static var portable: WorkflowMaterializationEnvironment {
        WorkflowMaterializationEnvironment(
            mereRunVersion: MereRunCLIVersion.current,
            pluginNodes: [],
            defaultModelID: { _ in nil },
            modelProvenance: { modelID in
                .fromCatalogIdentity(id: modelID, repository: nil, revision: nil)
            },
            providerRequirement: { _ in nil }
        )
    }
}

public extension WorkflowModelProvenance {
    /// Provenance derived from a catalog identity; matches the CLI's encoding
    /// so identical inputs fingerprint identically everywhere.
    static func fromCatalogIdentity(
        id: String,
        repository: String?,
        revision: String?
    ) -> WorkflowModelProvenance {
        let catalogIdentity = WorkflowManagedModelIdentity(
            id: id,
            repository: repository,
            revision: revision
        )
        return WorkflowModelProvenance(
            id: id,
            repository: repository,
            revision: revision,
            catalogSHA256: (try? WorkflowBundleCodec.hash(catalogIdentity)) ?? "",
            installManifestSHA256: nil
        )
    }
}

private struct WorkflowManagedModelIdentity: Codable {
    let id: String
    let repository: String?
    let revision: String?
}

public struct WorkflowPortableInputFingerprint: Codable {
    public let inputs: WorkflowInputsDocument
    public let assets: WorkflowAssetManifest

    public init(inputs: WorkflowInputsDocument, assets: WorkflowAssetManifest) {
        self.inputs = inputs
        self.assets = assets
    }
}

public func verifiedPortableWorkflowBundle(
    at directory: URL,
    fileManager: FileManager = .default,
    pluginNodes: [WorkflowNodeCatalogEntry]
) throws -> WorkflowBundleMaterialization {
    let directory = directory.standardizedFileURL
    for filename in [
        WorkflowJobManifest.filename,
        "graph.json",
        "inputs.json",
        WorkflowAssetManifest.filename,
    ] {
        let document = directory.appendingPathComponent(filename)
        let values = try document.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard fileManager.fileExists(atPath: document.path),
              values.isRegularFile == true,
              values.isSymbolicLink != true else {
            throw RelayClientError("Workflow bundle document is missing or unsafe: \(filename)")
        }
    }
    let graph = try WorkflowGraphDocument.load(from: directory.appendingPathComponent("graph.json"))
    let inputs = try WorkflowInputsDocument.load(from: directory.appendingPathComponent("inputs.json"))
    let assets = try WorkflowBundleCodec.decoder().decode(
        WorkflowAssetManifest.self,
        from: Data(contentsOf: directory.appendingPathComponent(WorkflowAssetManifest.filename))
    )
    let job = try WorkflowBundleCodec.decoder().decode(
        WorkflowJobManifest.self,
        from: Data(contentsOf: directory.appendingPathComponent(WorkflowJobManifest.filename))
    )
    guard job.contractVersion == WorkflowJobManifest.contractVersion else {
        throw RelayClientError("Unsupported workflow job contract '\(job.contractVersion)'.")
    }
    guard assets.schemaVersion == 1 else {
        throw RelayClientError("Unsupported workflow asset manifest version '\(assets.schemaVersion)'.")
    }
    guard try WorkflowBundleCodec.hash(graph) == job.graphFingerprint else {
        throw RelayClientError("Workflow graph fingerprint does not match job.json.")
    }
    let inputFingerprint = try WorkflowBundleCodec.hash(WorkflowPortableInputFingerprint(inputs: inputs, assets: assets))
    guard inputFingerprint == job.inputFingerprint else {
        throw RelayClientError("Workflow input fingerprint does not match job.json.")
    }
    let validation = WorkflowGraphValidator.validate(graph: graph, inputs: inputs, pluginNodes: pluginNodes)
    guard validation.status != .blocked else {
        throw RelayClientError(validation.diagnostics.map(\.message).joined(separator: " "))
    }
    let assetRoot = directory.appendingPathComponent("assets/sha256", isDirectory: true)
    for entry in assets.groups.flatMap(\.entries) {
        guard isPortableWorkflowPath(entry.path), isWorkflowSHA256(entry.digest) else {
            throw RelayClientError("Workflow asset manifest contains an invalid path or digest.")
        }
        let source = assetRoot.appendingPathComponent(entry.digest)
        let values = try source.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true,
              values.isSymbolicLink != true,
              try ModelArtifactPinDigest.fileByteCount(source) == entry.sizeBytes,
              try ModelArtifactPinDigest.fileSHA256(source) == entry.digest else {
            throw RelayClientError("Workflow asset digest verification failed for '\(entry.path)'.")
        }
    }
    return WorkflowBundleMaterialization(
        directory: directory,
        graph: graph,
        inputs: inputs,
        assets: assets,
        job: job
    )
}

public func copyPortableWorkflowBundle(
    _ bundle: WorkflowBundleMaterialization,
    to destination: URL,
    fileManager: FileManager = .default
) throws {
    let destination = destination.standardizedFileURL
    if fileManager.fileExists(atPath: destination.path),
       !(try fileManager.contentsOfDirectory(atPath: destination.path)).isEmpty {
        throw RelayClientError("Workflow run directory must be empty before bundle submission.")
    }
    try fileManager.createDirectory(at: destination, withIntermediateDirectories: true)
    for filename in [
        WorkflowJobManifest.filename,
        "graph.json",
        "inputs.json",
        WorkflowAssetManifest.filename,
    ] {
        try fileManager.copyItem(
            at: bundle.directory.appendingPathComponent(filename),
            to: destination.appendingPathComponent(filename)
        )
    }
    let destinationAssets = destination.appendingPathComponent("assets/sha256", isDirectory: true)
    try fileManager.createDirectory(at: destinationAssets, withIntermediateDirectories: true)
    for digest in Set(bundle.assets.groups.flatMap(\.entries).map(\.digest)).sorted() {
        try fileManager.copyItem(
            at: bundle.directory.appendingPathComponent("assets/sha256/\(digest)"),
            to: destinationAssets.appendingPathComponent(digest)
        )
    }
}

public func requireSeparateWorkflowDirectories(bundle: URL, run: URL) throws {
    let bundle = bundle.standardizedFileURL.path
    let run = run.standardizedFileURL.path
    guard bundle != run,
          !run.hasPrefix(bundle + "/"),
          !bundle.hasPrefix(run + "/") else {
        throw RelayClientError("Immutable bundle and run directories must be separate and non-nested.")
    }
}

private func isPortableWorkflowPath(_ path: String) -> Bool {
    !path.isEmpty
        && !path.hasPrefix("/")
        && path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy {
            $0 != "." && $0 != ".." && !$0.isEmpty
        }
}

private func isWorkflowSHA256(_ value: String) -> Bool {
    value.count == 64 && value.utf8.allSatisfy { byte in
        (byte >= 48 && byte <= 57) || (byte >= 97 && byte <= 102)
    }
}

public struct WorkflowBundleMaterializer {
    public let graph: WorkflowGraphDocument
    public let suppliedInputs: WorkflowInputsDocument
    public let destination: URL
    public let environment: WorkflowMaterializationEnvironment
    public let fileManager: FileManager
    public let now: () -> Date
    public let jobID: () -> UUID
    public let seed: () -> Int64

    public init(
        graph: WorkflowGraphDocument,
        suppliedInputs: WorkflowInputsDocument,
        destination: URL,
        environment: WorkflowMaterializationEnvironment,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init,
        jobID: @escaping () -> UUID = UUID.init,
        seed: @escaping () -> Int64 = { Int64.random(in: 0...Int64.max) }
    ) {
        self.graph = graph
        self.suppliedInputs = suppliedInputs
        self.destination = destination.standardizedFileURL
        self.environment = environment
        self.fileManager = fileManager
        self.now = now
        self.jobID = jobID
        self.seed = seed
    }

    public func materialize() throws -> WorkflowBundleMaterialization {
        let validation = WorkflowGraphValidator.validate(
            graph: graph,
            inputs: suppliedInputs,
            pluginNodes: environment.pluginNodes
        )
        guard validation.status != .blocked else {
            let messages = validation.diagnostics.filter { $0.severity == .blocker }.map(\.message)
            throw RelayClientError(messages.joined(separator: " "))
        }
        try prepareDestination()

        let sourceGraphFingerprint = try WorkflowBundleCodec.hash(graph)
        let sourceInputFingerprint = try WorkflowBundleCodec.hash(suppliedInputs)
        let resolvedInputs = resolveDefaults()
        let resolvedGraph = resolveSeeds()
        let assetsDirectory = destination
            .appendingPathComponent("assets", isDirectory: true)
            .appendingPathComponent("sha256", isDirectory: true)
        try fileManager.createDirectory(at: assetsDirectory, withIntermediateDirectories: true)

        var portableValues = resolvedInputs.values
        var groups: [WorkflowAssetGroup] = []
        for name in resolvedGraph.inputs.keys.sorted() {
            guard let definition = resolvedGraph.inputs[name],
                  definition.type == .asset || definition.type == .assetDirectory,
                  let sourcePath = resolvedInputs.values[name]?.stringValue else {
                continue
            }
            let sourceURL = URL(fileURLWithPath: sourcePath).standardizedFileURL
            let group = try materializeAsset(name: name, definition: definition, sourceURL: sourceURL, assetsDirectory: assetsDirectory)
            groups.append(group)
            portableValues[name] = .string("asset://\(name)")
        }

        let portableGraph = try materializeNodeAssetConstants(
            graph: resolvedGraph,
            assetsDirectory: assetsDirectory,
            groups: &groups
        )
        let portableInputs = WorkflowInputsDocument(values: portableValues)
        let assets = WorkflowAssetManifest(schemaVersion: 1, groups: groups)
        let graphFingerprint = try WorkflowBundleCodec.hash(portableGraph)
        let inputFingerprint = try WorkflowBundleCodec.hash(
            WorkflowPortableInputFingerprint(inputs: portableInputs, assets: assets)
        )
        let job = WorkflowJobManifest(
            contractVersion: WorkflowJobManifest.contractVersion,
            jobID: jobID().uuidString.lowercased(),
            createdAt: now(),
            graphFingerprint: graphFingerprint,
            inputFingerprint: inputFingerprint,
            sourceGraphFingerprint: sourceGraphFingerprint,
            sourceInputFingerprint: sourceInputFingerprint,
            requirements: requirements(graph: portableGraph, resolvedInputs: resolvedInputs),
            outputs: portableGraph.outputs.keys.sorted().compactMap { name in
                guard case .reference(let reference)? = portableGraph.outputs[name] else { return nil }
                return WorkflowJobOutput(name: name, reference: reference)
            }
        )

        try WorkflowBundleCodec.write(portableGraph, to: destination.appendingPathComponent("graph.json"))
        try WorkflowBundleCodec.write(portableInputs, to: destination.appendingPathComponent("inputs.json"))
        try WorkflowBundleCodec.write(assets, to: destination.appendingPathComponent(WorkflowAssetManifest.filename))
        try WorkflowBundleCodec.write(job, to: destination.appendingPathComponent(WorkflowJobManifest.filename))

        return WorkflowBundleMaterialization(
            directory: destination,
            graph: portableGraph,
            inputs: portableInputs,
            assets: assets,
            job: job
        )
    }

    private func prepareDestination() throws {
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: destination.path, isDirectory: &isDirectory) {
            guard isDirectory.boolValue else {
                throw RelayClientError("Job bundle destination is not a directory: \(destination.path)")
            }
            let existing = try fileManager.contentsOfDirectory(atPath: destination.path)
            guard existing.isEmpty else {
                throw RelayClientError("Job bundle destination is not empty: \(destination.path)")
            }
        } else {
            try fileManager.createDirectory(at: destination, withIntermediateDirectories: true)
        }
    }

    private func resolveDefaults() -> WorkflowInputsDocument {
        var values = suppliedInputs.values
        for (name, definition) in graph.inputs where values[name] == nil {
            values[name] = definition.defaultValue
        }
        return WorkflowInputsDocument(values: values)
    }

    private func resolveSeeds() -> WorkflowGraphDocument {
        let nodes = graph.nodes.map { node -> WorkflowNode in
            guard let catalog = WorkflowNodeRegistry.entry(for: node, pluginNodes: environment.pluginNodes) else { return node }
            var arguments = node.arguments
            for field in catalog.inputs where arguments[field.name] == nil {
                arguments[field.name] = field.defaultValue
            }
            if catalog.inputs.contains(where: { $0.name == "seed" }), arguments["seed"] == nil {
                arguments["seed"] = .integer(seed())
            }
            return WorkflowNode(
                id: node.id,
                kind: node.kind,
                provider: node.provider,
                arguments: arguments,
                dependsOn: node.dependsOn,
                execution: node.execution
            )
        }
        return WorkflowGraphDocument(
            schemaVersion: graph.schemaVersion,
            kind: graph.kind,
            name: graph.name,
            inputs: graph.inputs,
            nodes: nodes,
            outputs: graph.outputs,
            execution: graph.execution,
            metadata: graph.metadata
        )
    }

    private func materializeAsset(
        name: String,
        definition: WorkflowInputDefinition,
        sourceURL: URL,
        assetsDirectory: URL
    ) throws -> WorkflowAssetGroup {
        let resourceValues = try sourceURL.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey])
        if resourceValues.isSymbolicLink == true {
            throw RelayClientError("Workflow asset '\(name)' cannot be a symbolic link: \(sourceURL.path)")
        }
        switch definition.type {
        case .asset:
            guard resourceValues.isRegularFile == true else {
                throw RelayClientError("Workflow asset '\(name)' is not a regular file: \(sourceURL.path)")
            }
            return WorkflowAssetGroup(
                name: name,
                kind: .asset,
                entries: [try materializeFile(sourceURL, relativePath: sourceURL.lastPathComponent, assetsDirectory: assetsDirectory)]
            )
        case .assetDirectory:
            guard resourceValues.isDirectory == true else {
                throw RelayClientError("Workflow asset directory '\(name)' was not found: \(sourceURL.path)")
            }
            let confinedRoot = sourceURL.resolvingSymlinksInPath()
            let entries = try directoryFiles(confinedRoot).map { file in
                try materializeFile(file.url, relativePath: file.relativePath, assetsDirectory: assetsDirectory)
            }
            guard !entries.isEmpty else {
                throw RelayClientError("Workflow asset directory '\(name)' is empty: \(sourceURL.path)")
            }
            return WorkflowAssetGroup(name: name, kind: .assetDirectory, entries: entries)
        default:
            throw RelayClientError("Input '\(name)' is not an asset input.")
        }
    }

    private func materializeNodeAssetConstants(
        graph: WorkflowGraphDocument,
        assetsDirectory: URL,
        groups: inout [WorkflowAssetGroup]
    ) throws -> WorkflowGraphDocument {
        let nodes = try graph.nodes.map { node -> WorkflowNode in
            guard let catalog = WorkflowNodeRegistry.entry(for: node, pluginNodes: environment.pluginNodes) else { return node }
            var arguments = node.arguments
            for field in catalog.inputs {
                guard let value = arguments[field.name] else { continue }
                switch field.type {
                case .asset:
                    arguments[field.name] = try portableNodeAsset(
                        value,
                        kind: .asset,
                        key: "\(node.id).\(field.name).0",
                        assetsDirectory: assetsDirectory,
                        groups: &groups
                    )
                case .assetDirectory:
                    arguments[field.name] = try portableNodeAsset(
                        value,
                        kind: .assetDirectory,
                        key: "\(node.id).\(field.name).0",
                        assetsDirectory: assetsDirectory,
                        groups: &groups
                    )
                case .assetArray, .assetCollection:
                    guard case .array(let values) = value else { continue }
                    arguments[field.name] = .array(try values.enumerated().map { index, nested in
                        try portableNodeAsset(
                            nested,
                            kind: .asset,
                            key: "\(node.id).\(field.name).\(index)",
                            assetsDirectory: assetsDirectory,
                            groups: &groups
                        )
                    })
                default:
                    break
                }
            }
            return WorkflowNode(
                id: node.id,
                kind: node.kind,
                provider: node.provider,
                arguments: arguments,
                dependsOn: node.dependsOn,
                execution: node.execution
            )
        }
        return WorkflowGraphDocument(
            schemaVersion: graph.schemaVersion,
            kind: graph.kind,
            name: graph.name,
            inputs: graph.inputs,
            nodes: nodes,
            outputs: graph.outputs,
            execution: graph.execution,
            metadata: graph.metadata
        )
    }

    private func portableNodeAsset(
        _ value: WorkflowValue,
        kind: WorkflowInputType,
        key: String,
        assetsDirectory: URL,
        groups: inout [WorkflowAssetGroup]
    ) throws -> WorkflowValue {
        guard case .string(let path) = value else { return value }
        guard !path.hasPrefix("asset://") else {
            throw RelayClientError("asset:// is reserved for materialized workflow assets.")
        }
        let digest = SHA256.hash(data: Data(key.utf8)).prefix(16).map { String(format: "%02x", $0) }.joined()
        let name = "asset-\(digest)"
        guard !groups.contains(where: { $0.name == name }) else {
            throw RelayClientError("Workflow asset identifier collision for '\(key)'.")
        }
        let definition = WorkflowInputDefinition(
            type: kind,
            required: true,
            defaultValue: nil,
            values: nil,
            contentTypes: nil
        )
        let group = try materializeAsset(
            name: name,
            definition: definition,
            sourceURL: URL(fileURLWithPath: path).standardizedFileURL,
            assetsDirectory: assetsDirectory
        )
        groups.append(group)
        return .string("asset://\(name)")
    }

    private func directoryFiles(_ root: URL) throws -> [(url: URL, relativePath: String)] {
        var files: [(url: URL, relativePath: String)] = []
        for relativePath in try fileManager.subpathsOfDirectory(atPath: root.path).sorted() {
            if relativePath.split(separator: "/").contains(where: { $0.hasPrefix(".") }) { continue }
            let url = root.appendingPathComponent(relativePath)
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            if values.isSymbolicLink == true {
                throw RelayClientError("Workflow asset directories cannot contain symbolic links: \(url.path)")
            }
            if values.isRegularFile == true {
                files.append((url, relativePath))
            }
        }
        return files
    }

    private func materializeFile(
        _ sourceURL: URL,
        relativePath: String,
        assetsDirectory: URL
    ) throws -> WorkflowAssetEntry {
        guard !relativePath.hasPrefix("/"), !relativePath.split(separator: "/").contains("..") else {
            throw RelayClientError("Workflow asset path escapes its declared root: \(relativePath)")
        }
        let digest = try ModelArtifactPinDigest.fileSHA256(sourceURL)
        let size = try ModelArtifactPinDigest.fileByteCount(sourceURL)
        let destinationURL = assetsDirectory.appendingPathComponent(digest)
        if !fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.copyItem(at: sourceURL, to: destinationURL)
        }
        return WorkflowAssetEntry(
            path: relativePath,
            digest: digest,
            sizeBytes: size,
            contentType: contentType(for: sourceURL)
        )
    }

    private func requirements(
        graph: WorkflowGraphDocument,
        resolvedInputs: WorkflowInputsDocument
    ) -> WorkflowJobRequirements {
        var modelIDs = Set(graph.nodes.compactMap { node -> String? in
            if let value = node.arguments["model"] {
                if let resolved = resolveStatic(value, inputs: resolvedInputs.values)?.stringValue {
                    return resolved
                }
                return nil
            }
            return environment.defaultModelID(node.kind)
        })
        for node in graph.nodes {
            modelIDs.formUnion(
                WorkflowNodeRegistry.entry(for: node, pluginNodes: environment.pluginNodes)?.requirements.modelIDs ?? []
            )
        }
        let modelProvenance = modelIDs.sorted().map { environment.modelProvenance($0) }
        let providers = Array(Set(graph.nodes.compactMap { node -> WorkflowGraphProviderRequirement? in
            guard node.resolvedProviderID != WorkflowNodeProviderIdentity.builtInID else { return nil }
            return environment.providerRequirement(node.resolvedProviderID)
        })).sorted { $0.id < $1.id }
        var acceptedBackends = Set(["cpu", "metal", "cuda", "rocm"])
        var minimumMemory: Int64?
        var minimumSystemMemory: Int64?
        var minimumDisk: Int64?
        var minimumCPUCores: Int?
        var networkAccess = false
        for node in graph.nodes {
            guard let requirements = WorkflowNodeRegistry.entry(
                for: node,
                pluginNodes: environment.pluginNodes
            )?.requirements else { continue }
            if !requirements.acceleratorBackends.isEmpty {
                acceptedBackends.formIntersection(requirements.acceleratorBackends)
            }
            if let nodeMinimum = requirements.minimumAcceleratorMemoryBytes {
                minimumMemory = max(minimumMemory ?? 0, nodeMinimum)
            }
            if let nodeMinimum = requirements.minimumSystemMemoryBytes {
                minimumSystemMemory = max(minimumSystemMemory ?? 0, nodeMinimum)
            }
            if let nodeMinimum = requirements.minimumDiskBytes {
                minimumDisk = max(minimumDisk ?? 0, nodeMinimum)
            }
            if let nodeMinimum = requirements.minimumCPUCores {
                minimumCPUCores = max(minimumCPUCores ?? 0, nodeMinimum)
            }
            networkAccess = networkAccess || requirements.networkAccess == true
        }
        let secretNames = Array(Set(
            graph.nodes.flatMap { $0.arguments.values.flatMap(\.secretNames) }
        )).sorted()
        return WorkflowJobRequirements(
            minimumMereRunVersion: environment.mereRunVersion,
            nodeKinds: Array(Set(graph.nodes.map(\.kind))).sorted(),
            modelIDs: modelIDs.sorted(),
            models: modelProvenance,
            providers: providers,
            secretNames: secretNames,
            acceleratorBackends: acceptedBackends.sorted(),
            minimumAcceleratorMemoryBytes: minimumMemory,
            minimumSystemMemoryBytes: minimumSystemMemory,
            minimumDiskBytes: minimumDisk,
            minimumCPUCores: minimumCPUCores,
            networkAccess: networkAccess
        )
    }

    private func resolveStatic(_ value: WorkflowValue, inputs: [String: WorkflowValue]) -> WorkflowValue? {
        guard case .reference(let rawReference) = value,
              let reference = try? WorkflowReference(rawReference),
              case .input(let input) = reference.source else {
            return value
        }
        return inputs[input]
    }

    private func contentType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "png": "image/png"
        case "jpg", "jpeg": "image/jpeg"
        case "webp": "image/webp"
        case "mp4": "video/mp4"
        case "wav": "audio/wav"
        case "tif", "tiff": "image/tiff"
        case "json": "application/json"
        case "txt": "text/plain"
        case "safetensors": "application/x-safetensors"
        default: "application/octet-stream"
        }
    }
}
