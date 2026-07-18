import ArgumentParser
import Crypto
import Foundation
import MereRunCore

struct WorkflowAssetEntry: Codable, Equatable, Sendable {
    let path: String
    let digest: String
    let sizeBytes: Int64
    let contentType: String

    enum CodingKeys: String, CodingKey {
        case path
        case digest
        case sizeBytes = "size_bytes"
        case contentType = "content_type"
    }
}

struct WorkflowAssetGroup: Codable, Equatable, Sendable {
    let name: String
    let kind: WorkflowInputType
    let entries: [WorkflowAssetEntry]
}

struct WorkflowAssetManifest: Codable, Equatable, Sendable {
    static let filename = "assets.json"

    let schemaVersion: Int
    let groups: [WorkflowAssetGroup]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case groups
    }
}

struct WorkflowJobOutput: Codable, Equatable, Sendable {
    let name: String
    let reference: String
}

struct WorkflowJobRequirements: Codable, Equatable, Sendable {
    let minimumMereRunVersion: String
    let nodeKinds: [String]
    let modelIDs: [String]
    let models: [WorkflowModelProvenance]
    let providers: [WorkflowGraphProviderRequirement]
    let secretNames: [String]
    let acceleratorBackends: [String]
    let minimumAcceleratorMemoryBytes: Int64?
    let minimumSystemMemoryBytes: Int64?
    let minimumDiskBytes: Int64?
    let minimumCPUCores: Int?
    let networkAccess: Bool

    enum CodingKeys: String, CodingKey {
        case minimumMereRunVersion = "minimum_mere_run_version"
        case nodeKinds = "node_kinds"
        case modelIDs = "model_ids"
        case models
        case providers
        case secretNames = "secret_names"
        case acceleratorBackends = "accelerator_backends"
        case minimumAcceleratorMemoryBytes = "minimum_accelerator_memory_bytes"
        case minimumSystemMemoryBytes = "minimum_system_memory_bytes"
        case minimumDiskBytes = "minimum_disk_bytes"
        case minimumCPUCores = "minimum_cpu_cores"
        case networkAccess = "network_access"
    }

    init(
        minimumMereRunVersion: String,
        nodeKinds: [String],
        modelIDs: [String],
        models: [WorkflowModelProvenance] = [],
        providers: [WorkflowGraphProviderRequirement] = [],
        secretNames: [String] = [],
        acceleratorBackends: [String],
        minimumAcceleratorMemoryBytes: Int64?,
        minimumSystemMemoryBytes: Int64? = nil,
        minimumDiskBytes: Int64? = nil,
        minimumCPUCores: Int? = nil,
        networkAccess: Bool = false
    ) {
        self.minimumMereRunVersion = minimumMereRunVersion
        self.nodeKinds = nodeKinds
        self.modelIDs = modelIDs
        self.models = models
        self.providers = providers
        self.secretNames = secretNames
        self.acceleratorBackends = acceleratorBackends
        self.minimumAcceleratorMemoryBytes = minimumAcceleratorMemoryBytes
        self.minimumSystemMemoryBytes = minimumSystemMemoryBytes
        self.minimumDiskBytes = minimumDiskBytes
        self.minimumCPUCores = minimumCPUCores
        self.networkAccess = networkAccess
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        minimumMereRunVersion = try container.decode(String.self, forKey: .minimumMereRunVersion)
        nodeKinds = try container.decode([String].self, forKey: .nodeKinds)
        modelIDs = try container.decode([String].self, forKey: .modelIDs)
        models = try container.decodeIfPresent([WorkflowModelProvenance].self, forKey: .models) ?? []
        providers = try container.decodeIfPresent(
            [WorkflowGraphProviderRequirement].self,
            forKey: .providers
        ) ?? []
        secretNames = try container.decodeIfPresent([String].self, forKey: .secretNames) ?? []
        acceleratorBackends = try container.decode([String].self, forKey: .acceleratorBackends)
        minimumAcceleratorMemoryBytes = try container.decodeIfPresent(
            Int64.self,
            forKey: .minimumAcceleratorMemoryBytes
        )
        minimumSystemMemoryBytes = try container.decodeIfPresent(Int64.self, forKey: .minimumSystemMemoryBytes)
        minimumDiskBytes = try container.decodeIfPresent(Int64.self, forKey: .minimumDiskBytes)
        minimumCPUCores = try container.decodeIfPresent(Int.self, forKey: .minimumCPUCores)
        networkAccess = try container.decodeIfPresent(Bool.self, forKey: .networkAccess) ?? false
    }
}

struct WorkflowModelProvenance: Codable, Equatable, Hashable, Sendable {
    let id: String
    let repository: String?
    let revision: String?
    let catalogSHA256: String
    let installManifestSHA256: String?

    enum CodingKeys: String, CodingKey {
        case id
        case repository
        case revision
        case catalogSHA256 = "catalog_sha256"
        case installManifestSHA256 = "install_manifest_sha256"
    }

    init(
        id: String,
        repository: String?,
        revision: String?,
        catalogSHA256: String,
        installManifestSHA256: String? = nil
    ) {
        self.id = id
        self.repository = repository
        self.revision = revision
        self.catalogSHA256 = catalogSHA256
        self.installManifestSHA256 = installManifestSHA256
    }
}

struct WorkflowJobManifest: Codable, Equatable, Sendable {
    static let contractVersion = "mere.run/job-bundle.v1"
    static let filename = "job.json"

    let contractVersion: String
    let jobID: String
    let createdAt: Date
    let graphFingerprint: String
    let inputFingerprint: String
    let requirements: WorkflowJobRequirements
    let outputs: [WorkflowJobOutput]

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case jobID = "job_id"
        case createdAt = "created_at"
        case graphFingerprint = "graph_fingerprint"
        case inputFingerprint = "input_fingerprint"
        case requirements
        case outputs
    }
}

struct WorkflowBundleMaterialization: Equatable {
    let directory: URL
    let graph: WorkflowGraphDocument
    let inputs: WorkflowInputsDocument
    let assets: WorkflowAssetManifest
    let job: WorkflowJobManifest
}

enum WorkflowBundleCodec {
    static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }

    static func lineEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    static func write<T: Encodable>(_ value: T, to url: URL) throws {
        try encoder().encode(value).write(to: url, options: .atomic)
    }

    static func hash<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return SHA256.hash(data: try encoder.encode(value)).map { String(format: "%02x", $0) }.joined()
    }
}

func verifiedPortableWorkflowBundle(
    at directory: URL,
    fileManager: FileManager = .default
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
            throw ValidationError("Workflow bundle document is missing or unsafe: \(filename)")
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
        throw ValidationError("Unsupported workflow job contract '\(job.contractVersion)'.")
    }
    guard assets.schemaVersion == 1 else {
        throw ValidationError("Unsupported workflow asset manifest version '\(assets.schemaVersion)'.")
    }
    guard try WorkflowBundleCodec.hash(graph) == job.graphFingerprint else {
        throw ValidationError("Workflow graph fingerprint does not match job.json.")
    }
    let inputFingerprint = try WorkflowBundleCodec.hash(WorkflowPortableInputFingerprint(inputs: inputs, assets: assets))
    guard inputFingerprint == job.inputFingerprint else {
        throw ValidationError("Workflow input fingerprint does not match job.json.")
    }
    let validation = WorkflowGraphValidator.validate(graph: graph, inputs: inputs)
    guard validation.status != .blocked else {
        throw ValidationError(validation.diagnostics.map(\.message).joined(separator: " "))
    }
    let assetRoot = directory.appendingPathComponent("assets/sha256", isDirectory: true)
    for entry in assets.groups.flatMap(\.entries) {
        guard isPortableWorkflowPath(entry.path), isWorkflowSHA256(entry.digest) else {
            throw ValidationError("Workflow asset manifest contains an invalid path or digest.")
        }
        let source = assetRoot.appendingPathComponent(entry.digest)
        let values = try source.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true,
              values.isSymbolicLink != true,
              try ModelArtifactPin.fileByteCount(source) == entry.sizeBytes,
              try ModelArtifactPin.fileSHA256(source) == entry.digest else {
            throw ValidationError("Workflow asset digest verification failed for '\(entry.path)'.")
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

func copyPortableWorkflowBundle(
    _ bundle: WorkflowBundleMaterialization,
    to destination: URL,
    fileManager: FileManager = .default
) throws {
    let destination = destination.standardizedFileURL
    if fileManager.fileExists(atPath: destination.path),
       !(try fileManager.contentsOfDirectory(atPath: destination.path)).isEmpty {
        throw ValidationError("Workflow run directory must be empty before bundle submission.")
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

func requireSeparateWorkflowDirectories(bundle: URL, run: URL) throws {
    let bundle = bundle.standardizedFileURL.path
    let run = run.standardizedFileURL.path
    guard bundle != run,
          !run.hasPrefix(bundle + "/"),
          !bundle.hasPrefix(run + "/") else {
        throw ValidationError("Immutable bundle and run directories must be separate and non-nested.")
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

struct WorkflowBundleMaterializer {
    let graph: WorkflowGraphDocument
    let suppliedInputs: WorkflowInputsDocument
    let destination: URL
    let fileManager: FileManager
    let now: () -> Date
    let jobID: () -> UUID
    let seed: () -> Int64

    init(
        graph: WorkflowGraphDocument,
        suppliedInputs: WorkflowInputsDocument,
        destination: URL,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init,
        jobID: @escaping () -> UUID = UUID.init,
        seed: @escaping () -> Int64 = { Int64.random(in: 0...Int64.max) }
    ) {
        self.graph = graph
        self.suppliedInputs = suppliedInputs
        self.destination = destination.standardizedFileURL
        self.fileManager = fileManager
        self.now = now
        self.jobID = jobID
        self.seed = seed
    }

    func materialize() throws -> WorkflowBundleMaterialization {
        let validation = WorkflowGraphValidator.validate(graph: graph, inputs: suppliedInputs)
        guard validation.status != .blocked else {
            let messages = validation.diagnostics.filter { $0.severity == .blocker }.map(\.message)
            throw ValidationError(messages.joined(separator: " "))
        }
        try prepareDestination()

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
                throw ValidationError("Job bundle destination is not a directory: \(destination.path)")
            }
            let existing = try fileManager.contentsOfDirectory(atPath: destination.path)
            guard existing.isEmpty else {
                throw ValidationError("Job bundle destination is not empty: \(destination.path)")
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
            guard let catalog = WorkflowNodeRegistry.entry(for: node) else { return node }
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
            throw ValidationError("Workflow asset '\(name)' cannot be a symbolic link: \(sourceURL.path)")
        }
        switch definition.type {
        case .asset:
            guard resourceValues.isRegularFile == true else {
                throw ValidationError("Workflow asset '\(name)' is not a regular file: \(sourceURL.path)")
            }
            return WorkflowAssetGroup(
                name: name,
                kind: .asset,
                entries: [try materializeFile(sourceURL, relativePath: sourceURL.lastPathComponent, assetsDirectory: assetsDirectory)]
            )
        case .assetDirectory:
            guard resourceValues.isDirectory == true else {
                throw ValidationError("Workflow asset directory '\(name)' was not found: \(sourceURL.path)")
            }
            let confinedRoot = sourceURL.resolvingSymlinksInPath()
            let entries = try directoryFiles(confinedRoot).map { file in
                try materializeFile(file.url, relativePath: file.relativePath, assetsDirectory: assetsDirectory)
            }
            guard !entries.isEmpty else {
                throw ValidationError("Workflow asset directory '\(name)' is empty: \(sourceURL.path)")
            }
            return WorkflowAssetGroup(name: name, kind: .assetDirectory, entries: entries)
        default:
            throw ValidationError("Input '\(name)' is not an asset input.")
        }
    }

    private func materializeNodeAssetConstants(
        graph: WorkflowGraphDocument,
        assetsDirectory: URL,
        groups: inout [WorkflowAssetGroup]
    ) throws -> WorkflowGraphDocument {
        let nodes = try graph.nodes.map { node -> WorkflowNode in
            guard let catalog = WorkflowNodeRegistry.entry(for: node) else { return node }
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
            throw ValidationError("asset:// is reserved for materialized workflow assets.")
        }
        let digest = SHA256.hash(data: Data(key.utf8)).prefix(16).map { String(format: "%02x", $0) }.joined()
        let name = "asset-\(digest)"
        guard !groups.contains(where: { $0.name == name }) else {
            throw ValidationError("Workflow asset identifier collision for '\(key)'.")
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
                throw ValidationError("Workflow asset directories cannot contain symbolic links: \(url.path)")
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
            throw ValidationError("Workflow asset path escapes its declared root: \(relativePath)")
        }
        let digest = try ModelArtifactPin.fileSHA256(sourceURL)
        let size = try ModelArtifactPin.fileByteCount(sourceURL)
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
            switch node.kind {
            case "image.train-lora": return ImageTrainLoRA.defaultManagedModelID.rawValue
            case "image.generate": return ImageGenerate.defaultManagedModelID.rawValue
            case "video.generate": return ModelResolver.ModelID.ltxVideo23AVMLX.rawValue
            default: return nil
            }
        })
        for node in graph.nodes {
            modelIDs.formUnion(WorkflowNodeRegistry.entry(for: node)?.requirements.modelIDs ?? [])
        }
        let modelProvenance = modelIDs.sorted().map { modelID -> WorkflowModelProvenance in
            let spec = ManagedModelCatalog.spec(for: modelID)
            let catalogIdentity = WorkflowManagedModelIdentity(
                id: modelID,
                repository: spec?.upstreamRepoId,
                revision: spec?.upstreamRevision
            )
            return WorkflowModelProvenance(
                id: modelID,
                repository: spec?.upstreamRepoId,
                revision: spec?.upstreamRevision,
                catalogSHA256: (try? WorkflowBundleCodec.hash(catalogIdentity)) ?? "",
                installManifestSHA256: nil
            )
        }
        let providers = Array(Set(graph.nodes.compactMap { node -> WorkflowGraphProviderRequirement? in
            guard node.resolvedProviderID != WorkflowNodeProviderIdentity.builtInID else { return nil }
            return WorkflowGraphProviderRegistry.discoveredCatalog().provider(id: node.resolvedProviderID)?.requirement
        })).sorted { $0.id < $1.id }
        var acceptedBackends = Set(["cpu", "metal", "cuda", "rocm"])
        var minimumMemory: Int64?
        var minimumSystemMemory: Int64?
        var minimumDisk: Int64?
        var minimumCPUCores: Int?
        var networkAccess = false
        for node in graph.nodes {
            guard let requirements = WorkflowNodeRegistry.entry(for: node)?.requirements else { continue }
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
            minimumMereRunVersion: MereRunCLIVersion.current,
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
        case "json": "application/json"
        case "txt": "text/plain"
        case "safetensors": "application/x-safetensors"
        default: "application/octet-stream"
        }
    }
}

private struct WorkflowManagedModelIdentity: Codable {
    let id: String
    let repository: String?
    let revision: String?
}

struct WorkflowPortableInputFingerprint: Codable {
    let inputs: WorkflowInputsDocument
    let assets: WorkflowAssetManifest
}

enum GraphRunState: String, Codable, Equatable, Sendable {
    case planned
    case preflighting
    case queued
    case assigned
    case running
    case finished
    case failed
    case cancelled
}

struct GraphRunArtifact: Codable, Equatable, Sendable {
    let name: String
    let kind: String
    let path: String
    let contentType: String
    let sizeBytes: Int64
    let sha256: String

    enum CodingKeys: String, CodingKey {
        case name
        case kind
        case path
        case contentType = "content_type"
        case sizeBytes = "size_bytes"
        case sha256
    }
}

struct GraphRunNodeOutput: Codable, Equatable, Sendable {
    let name: String
    let type: WorkflowPortType
    let value: WorkflowValue?
    let path: String?
    let contentType: String?
    let sizeBytes: Int64?
    let sha256: String?

    enum CodingKeys: String, CodingKey {
        case name
        case type
        case value
        case path
        case contentType = "content_type"
        case sizeBytes = "size_bytes"
        case sha256
    }
}

struct GraphRunNodeRecord: Codable, Equatable, Sendable {
    let id: String
    let kind: String
    var state: GraphRunState
    var startedAt: Date?
    var completedAt: Date?
    var exitStatus: Int32?
    var attempt: Int
    var maxAttempts: Int
    var fingerprint: String
    var provider: WorkflowNodeProviderIdentity?
    var models: [WorkflowModelProvenance]
    var artifacts: [GraphRunArtifact]
    var outputs: [GraphRunNodeOutput]
    var error: String?

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case state
        case startedAt = "started_at"
        case completedAt = "completed_at"
        case exitStatus = "exit_status"
        case attempt
        case maxAttempts = "max_attempts"
        case fingerprint
        case provider
        case models
        case artifacts
        case outputs
        case error
    }

    init(
        id: String,
        kind: String,
        state: GraphRunState,
        startedAt: Date?,
        completedAt: Date?,
        exitStatus: Int32?,
        attempt: Int = 0,
        maxAttempts: Int = 1,
        fingerprint: String,
        provider: WorkflowNodeProviderIdentity? = nil,
        models: [WorkflowModelProvenance] = [],
        artifacts: [GraphRunArtifact],
        outputs: [GraphRunNodeOutput] = [],
        error: String?
    ) {
        self.id = id
        self.kind = kind
        self.state = state
        self.startedAt = startedAt
        self.completedAt = completedAt
        self.exitStatus = exitStatus
        self.attempt = attempt
        self.maxAttempts = maxAttempts
        self.fingerprint = fingerprint
        self.provider = provider
        self.models = models
        self.artifacts = artifacts
        self.outputs = outputs
        self.error = error
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        kind = try container.decode(String.self, forKey: .kind)
        state = try container.decode(GraphRunState.self, forKey: .state)
        startedAt = try container.decodeIfPresent(Date.self, forKey: .startedAt)
        completedAt = try container.decodeIfPresent(Date.self, forKey: .completedAt)
        exitStatus = try container.decodeIfPresent(Int32.self, forKey: .exitStatus)
        attempt = try container.decodeIfPresent(Int.self, forKey: .attempt) ?? 0
        maxAttempts = try container.decodeIfPresent(Int.self, forKey: .maxAttempts) ?? 1
        fingerprint = try container.decode(String.self, forKey: .fingerprint)
        provider = try container.decodeIfPresent(WorkflowNodeProviderIdentity.self, forKey: .provider)
        models = try container.decodeIfPresent([WorkflowModelProvenance].self, forKey: .models) ?? []
        artifacts = try container.decode([GraphRunArtifact].self, forKey: .artifacts)
        outputs = try container.decodeIfPresent([GraphRunNodeOutput].self, forKey: .outputs) ?? []
        error = try container.decodeIfPresent(String.self, forKey: .error)
    }
}

struct GraphRunExecutorRecord: Codable, Equatable, Sendable {
    let kind: String
    let profile: String?
    let jobReference: String?

    enum CodingKeys: String, CodingKey {
        case kind
        case profile
        case jobReference = "job_reference"
    }
}

struct GraphRunManifest: Codable, Equatable, Sendable {
    static let contractVersion = "mere.run/graph-run.v1"
    static let filename = "run.json"

    let contractVersion: String
    let jobID: String
    let graphName: String
    let graphFingerprint: String
    var state: GraphRunState
    let createdAt: Date
    var updatedAt: Date
    var attempt: Int
    var executor: GraphRunExecutorRecord
    var nodes: [GraphRunNodeRecord]
    var outputs: [GraphRunArtifact]
    var error: String?

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case jobID = "job_id"
        case graphName = "graph_name"
        case graphFingerprint = "graph_fingerprint"
        case state
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case attempt
        case executor
        case nodes
        case outputs
        case error
    }
}

struct GraphRunEvent: Codable, Equatable, Sendable {
    let sequence: Int
    let createdAt: Date
    let type: String
    let state: GraphRunState
    let nodeID: String?
    let message: String?
    let progress: GraphRunProgress?
    let artifact: GraphRunEventArtifact?
    let diagnostic: GraphRunEventDiagnostic?
    let metric: GraphRunMetric?

    enum CodingKeys: String, CodingKey {
        case sequence
        case createdAt = "created_at"
        case type
        case state
        case nodeID = "node_id"
        case message
        case progress
        case artifact
        case diagnostic
        case metric
    }

    init(
        sequence: Int,
        createdAt: Date,
        type: String,
        state: GraphRunState,
        nodeID: String?,
        message: String?,
        progress: GraphRunProgress? = nil,
        artifact: GraphRunEventArtifact? = nil,
        diagnostic: GraphRunEventDiagnostic? = nil,
        metric: GraphRunMetric? = nil
    ) {
        self.sequence = sequence
        self.createdAt = createdAt
        self.type = type
        self.state = state
        self.nodeID = nodeID
        self.message = message
        self.progress = progress
        self.artifact = artifact
        self.diagnostic = diagnostic
        self.metric = metric
    }
}

struct GraphRunProgress: Codable, Equatable, Sendable {
    let phase: String?
    let current: Double?
    let total: Double?
    let fraction: Double?
    let unit: String?
}

struct GraphRunEventArtifact: Codable, Equatable, Sendable {
    let name: String
    let path: String
    let contentType: String

    enum CodingKeys: String, CodingKey {
        case name
        case path
        case contentType = "content_type"
    }
}

struct GraphRunEventDiagnostic: Codable, Equatable, Sendable {
    let id: String
    let severity: String
    let title: String
    let message: String
}

struct GraphRunMetric: Codable, Equatable, Sendable {
    let name: String
    let value: Double
    let unit: String?
}

struct WorkflowInvocationOutput: Codable, Equatable, Sendable {
    let type: WorkflowPortType
    let path: String?
    let optional: Bool
    let contentTypes: [String]

    enum CodingKeys: String, CodingKey {
        case type
        case path
        case optional
        case contentTypes = "content_types"
    }
}

struct WorkflowPluginNodeInvocationDocument: Codable, Equatable, Sendable {
    static let contractVersion = "mere.run/plugin-graph-invocation.v1"

    let contractVersion: String
    let jobID: String
    let nodeID: String
    let kind: String
    let arguments: [String: WorkflowValue]
    let outputs: [String: WorkflowInvocationOutput]

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case jobID = "job_id"
        case nodeID = "node_id"
        case kind
        case arguments
        case outputs
    }
}

struct WorkflowNodeInvocation: Equatable {
    let command: [String]
    let executable: URL
    let preflightArguments: [String]
    let runArguments: [String]
    let outputs: [String: WorkflowInvocationOutput]
    let streamsEvents: Bool
}

enum WorkflowNodeCommandBuilder {
    static func invocation(
        node: WorkflowNode,
        arguments: [String: WorkflowValue],
        nodeDirectory: URL,
        jobID: String = UUID().uuidString.lowercased()
    ) throws -> WorkflowNodeInvocation {
        if node.resolvedProviderID != WorkflowNodeProviderIdentity.builtInID {
            return try pluginInvocation(
                node: node,
                arguments: arguments,
                nodeDirectory: nodeDirectory,
                jobID: jobID
            )
        }
        let artifacts = nodeDirectory.appendingPathComponent("artifacts", isDirectory: true)
        switch node.kind {
        case "image.train-lora":
            let output = artifacts.appendingPathComponent("adapter.safetensors")
            var args = ["image", "train-lora", "--data", try requiredString("data", in: arguments), "--output", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendString("recipe", flag: "--recipe", from: arguments, to: &args)
            appendInteger("training_steps", flag: "--training-steps", from: arguments, to: &args)
            appendInteger("width", flag: "--width", from: arguments, to: &args)
            appendInteger("height", flag: "--height", from: arguments, to: &args)
            appendInteger("max_text_length", flag: "--max-text-length", from: arguments, to: &args)
            appendInteger("seed", flag: "--seed", from: arguments, to: &args)
            appendInteger("rank", flag: "--rank", from: arguments, to: &args)
            appendFlag("lite", flag: "--lite", from: arguments, to: &args)
            appendInteger(
                "base_quantization_bits",
                flag: "--base-quantization-bits",
                from: arguments,
                to: &args
            )
            appendString("sample_prompt", flag: "--sample-prompt", from: arguments, to: &args)
            return .init(
                command: ["image", "train-lora"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: ["adapter": fileOutput(output, contentTypes: ["application/x-safetensors"])],
                streamsEvents: false
            )
        case "image.generate":
            let output = artifacts.appendingPathComponent("image.png")
            var args = ["image", "generate", "--prompt", try requiredString("prompt", in: arguments), "--output", output.path]
            appendString("negative_prompt", flag: "--negative-prompt", from: arguments, to: &args)
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendInteger("width", flag: "--width", from: arguments, to: &args)
            appendInteger("height", flag: "--height", from: arguments, to: &args)
            appendInteger("steps", flag: "--steps", from: arguments, to: &args)
            appendInteger("seed", flag: "--seed", from: arguments, to: &args)
            appendString("input", flag: "--input", from: arguments, to: &args)
            appendString("lora", flag: "--lora", from: arguments, to: &args)
            appendNumber("lora_scale", flag: "--lora-scale", from: arguments, to: &args)
            appendNumber("cfg_scale", flag: "--cfg", from: arguments, to: &args)
            appendNumber("strength", flag: "--strength", from: arguments, to: &args)
            appendInteger(
                "krea_base_quantization_bits",
                flag: "--krea-base-quantization-bits",
                from: arguments,
                to: &args
            )
            if case .array(let references)? = arguments["reference_images"] {
                for reference in references {
                    guard let path = reference.stringValue else {
                        throw ValidationError("image.generate reference_images values must resolve to paths.")
                    }
                    args += ["--ref-image", path]
                }
            }
            return .init(
                command: ["image", "generate"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: ["image": fileOutput(output, contentTypes: ["image/png"])],
                streamsEvents: false
            )
        case "video.generate":
            let output = artifacts.appendingPathComponent("video.mp4")
            var args = ["video", "generate", try requiredString("prompt", in: arguments), "--output", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendInteger("width", flag: "--width", from: arguments, to: &args)
            appendInteger("height", flag: "--height", from: arguments, to: &args)
            appendInteger("num_frames", flag: "--num-frames", from: arguments, to: &args)
            appendNumber("duration", flag: "--duration", from: arguments, to: &args)
            appendInteger("fps", flag: "--fps", from: arguments, to: &args)
            appendInteger("seed", flag: "--seed", from: arguments, to: &args)
            appendInteger("steps", flag: "--steps", from: arguments, to: &args)
            appendString("image", flag: "--image", from: arguments, to: &args)
            appendString("end_image", flag: "--end-image", from: arguments, to: &args)
            return .init(
                command: ["video", "generate"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: ["video": fileOutput(output, contentTypes: ["video/mp4"])],
                streamsEvents: false
            )
        default:
            throw ValidationError("Unsupported workflow node kind '\(node.kind)'.")
        }
    }

    private static func pluginInvocation(
        node: WorkflowNode,
        arguments: [String: WorkflowValue],
        nodeDirectory: URL,
        jobID: String
    ) throws -> WorkflowNodeInvocation {
        let provider = try WorkflowGraphProviderRegistry.requireProvider(id: node.resolvedProviderID)
        guard let catalog = provider.nodes.first(where: { $0.kind == node.kind }),
              let executable = PluginProcess.which(provider.executable) else {
            throw ValidationError(
                "Graph provider '\(node.resolvedProviderID)' does not expose node kind '\(node.kind)'."
            )
        }
        var outputs: [String: WorkflowInvocationOutput] = [:]
        for output in catalog.outputs {
            outputs[output.name] = WorkflowInvocationOutput(
                type: output.type,
                path: outputPath(for: output),
                optional: output.optional,
                contentTypes: output.contentTypes
            )
        }
        let request = WorkflowPluginNodeInvocationDocument(
            contractVersion: WorkflowPluginNodeInvocationDocument.contractVersion,
            jobID: jobID,
            nodeID: node.id,
            kind: node.kind,
            arguments: arguments,
            outputs: outputs
        )
        let requestURL = nodeDirectory.appendingPathComponent("invocation.json")
        try WorkflowBundleCodec.write(request, to: requestURL)
        let shared = ["--request", requestURL.path, "--run-dir", nodeDirectory.path]
        return WorkflowNodeInvocation(
            command: [provider.executable, "graph", "execute"],
            executable: executable,
            preflightArguments: ["graph", "preflight"] + shared + ["--json"],
            runArguments: ["graph", "execute"] + shared + ["--json-stream"],
            outputs: outputs,
            streamsEvents: true
        )
    }

    private static func fileOutput(_ url: URL, contentTypes: [String]) -> WorkflowInvocationOutput {
        WorkflowInvocationOutput(
            type: .asset,
            path: url.path,
            optional: false,
            contentTypes: contentTypes
        )
    }

    private static func outputPath(for output: WorkflowNodeOutput) -> String? {
        switch output.type {
        case .asset:
            let pathExtension = outputExtension(contentTypes: output.contentTypes)
            return "artifacts/\(output.name)\(pathExtension)"
        case .assetDirectory:
            return "artifacts/\(output.name)"
        case .assetCollection, .assetArray:
            return "artifacts/\(output.name).json"
        default:
            return nil
        }
    }

    private static func outputExtension(contentTypes: [String]) -> String {
        switch contentTypes.first {
        case "image/png": ".png"
        case "image/jpeg": ".jpg"
        case "image/webp": ".webp"
        case "video/mp4": ".mp4"
        case "audio/wav": ".wav"
        case "application/json": ".json"
        case "application/x-safetensors": ".safetensors"
        default: ""
        }
    }

    private static func requiredString(_ name: String, in arguments: [String: WorkflowValue]) throws -> String {
        guard let value = arguments[name]?.stringValue, !value.isEmpty else {
            throw ValidationError("Workflow node argument '\(name)' did not resolve to a string.")
        }
        return value
    }

    private static func appendString(
        _ name: String,
        flag: String,
        from arguments: [String: WorkflowValue],
        to output: inout [String]
    ) {
        if let value = arguments[name]?.stringValue { output += [flag, value] }
    }

    private static func appendInteger(
        _ name: String,
        flag: String,
        from arguments: [String: WorkflowValue],
        to output: inout [String]
    ) {
        if let value = arguments[name]?.integerValue { output += [flag, String(value)] }
    }

    private static func appendNumber(
        _ name: String,
        flag: String,
        from arguments: [String: WorkflowValue],
        to output: inout [String]
    ) {
        if let value = arguments[name]?.numberValue { output += [flag, String(value)] }
    }

    private static func appendFlag(
        _ name: String,
        flag: String,
        from arguments: [String: WorkflowValue],
        to output: inout [String]
    ) {
        if arguments[name]?.booleanValue == true { output.append(flag) }
    }
}
