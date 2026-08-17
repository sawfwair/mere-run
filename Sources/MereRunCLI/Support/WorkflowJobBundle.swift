import ArgumentParser
import Crypto
import Foundation
import MereRunCore
import MereRunRelayKit

struct WorkflowBundleMaterialization: Equatable {
    let directory: URL
    let graph: WorkflowGraphDocument
    let inputs: WorkflowInputsDocument
    let assets: WorkflowAssetManifest
    let job: WorkflowJobManifest
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
            case "vision.ground": return VisionGround.defaultManagedModelID.rawValue
            case "vision.segment", "vision.track": return VisionSegment.defaultManagedModelID.rawValue
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
        case "tif", "tiff": "image/tiff"
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
    let intrinsic: WorkflowIntrinsicInvocation?
    let stdoutOutputName: String?

    init(
        command: [String],
        executable: URL,
        preflightArguments: [String],
        runArguments: [String],
        outputs: [String: WorkflowInvocationOutput],
        streamsEvents: Bool,
        intrinsic: WorkflowIntrinsicInvocation? = nil,
        stdoutOutputName: String? = nil
    ) {
        self.command = command
        self.executable = executable
        self.preflightArguments = preflightArguments
        self.runArguments = runArguments
        self.outputs = outputs
        self.streamsEvents = streamsEvents
        self.intrinsic = intrinsic
        self.stdoutOutputName = stdoutOutputName
    }
}

struct WorkflowIntrinsicInvocation: Equatable {
    let kind: String
    let arguments: [String: WorkflowValue]

    func evaluate() throws -> [String: WorkflowValue] {
        switch kind {
        case "text.value":
            return ["text": try required("value")]
        case "integer.value", "number.value", "boolean.value", "json.value":
            return ["value": try required("value")]
        case "seed.value":
            return ["seed": try required("seed")]
        case "choice.value":
            guard case .array(let options) = try required("options"),
                  options.allSatisfy({ $0.stringValue != nil }),
                  case .string(let selected) = try required("selected") else {
                throw ValidationError("choice.value requires string options and a selected string.")
            }
            guard options.contains(.string(selected)) else {
                throw ValidationError("choice.value selected value must appear in options.")
            }
            return ["value": .string(selected)]
        case "text.join":
            guard case .array(let parts) = try required("parts"),
                  parts.allSatisfy({ $0.stringValue != nil }) else {
                throw ValidationError("text.join parts must resolve to an array of strings.")
            }
            let separator = arguments["separator"]?.stringValue ?? "\n"
            return ["text": .string(parts.compactMap(\.stringValue).joined(separator: separator))]
        case "text.template":
            guard case .string(let template) = try required("template"),
                  case .object(let variables) = try required("variables"),
                  variables.values.allSatisfy({ $0.stringValue != nil }) else {
                throw ValidationError("text.template requires a string template and string variables.")
            }
            return ["text": .string(try renderTemplate(template, variables: variables))]
        default:
            throw ValidationError("Unsupported intrinsic workflow node kind '\(kind)'.")
        }
    }

    private func required(_ name: String) throws -> WorkflowValue {
        guard let value = arguments[name], value != .null else {
            throw ValidationError("\(kind) requires argument '\(name)'.")
        }
        return value
    }

    private func renderTemplate(
        _ template: String,
        variables: [String: WorkflowValue]
    ) throws -> String {
        let expression = try NSRegularExpression(pattern: #"(?<!\\)\{\{([^{}]+)\}\}"#)
        let source = template as NSString
        let matches = expression.matches(
            in: template,
            range: NSRange(location: 0, length: source.length)
        )
        var rendered = template
        for match in matches.reversed() {
            let name = source.substring(with: match.range(at: 1))
            guard name.range(
                of: "^[a-z][a-z0-9_]{0,63}$",
                options: .regularExpression
            ) != nil else {
                throw ValidationError("text.template placeholder '{{\(name)}}' is invalid.")
            }
            guard let replacement = variables[name]?.stringValue else {
                throw ValidationError("text.template is missing variable '\(name)'.")
            }
            guard let range = Range(match.range, in: rendered) else {
                throw ValidationError("text.template could not resolve placeholder '\(name)'.")
            }
            rendered.replaceSubrange(range, with: replacement)
        }
        return rendered.replacingOccurrences(of: #"\{{"#, with: "{{")
    }
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
        case "text.value", "integer.value", "number.value", "boolean.value",
             "json.value", "seed.value", "choice.value", "text.join", "text.template":
            guard let entry = WorkflowNodeRegistry.entry(for: node) else {
                throw ValidationError("Unsupported workflow node kind '\(node.kind)'.")
            }
            return .init(
                command: ["intrinsic", node.kind],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: [],
                outputs: Dictionary(uniqueKeysWithValues: entry.outputs.map { output in
                    (
                        output.name,
                        WorkflowInvocationOutput(
                            type: output.type,
                            path: nil,
                            optional: output.optional,
                            contentTypes: output.contentTypes
                        )
                    )
                }),
                streamsEvents: false,
                intrinsic: WorkflowIntrinsicInvocation(kind: node.kind, arguments: arguments)
            )
        case "text.enhance":
            let instruction = try requiredString("instruction", in: arguments)
            let text = try requiredString("text", in: arguments)
            var args = [
                "text", "chat",
                "--prompt", "\(instruction)\n\nText:\n\(text)",
                "--model", try requiredString("model", in: arguments),
            ]
            appendInteger("max_tokens", flag: "--max-tokens", from: arguments, to: &args)
            appendNumber("temperature", flag: "--temperature", from: arguments, to: &args)
            args += ["--no-thinking", "--require-installed", "--quiet"]
            return .init(
                command: ["text", "chat"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args,
                outputs: ["text": valueOutput(.string)],
                streamsEvents: false,
                stdoutOutputName: "text"
            )
        case "image.describe":
            var args = [
                "text", "chat",
                "--prompt", try requiredString("instruction", in: arguments),
                "--image", try requiredString("image", in: arguments),
                "--model", try requiredString("model", in: arguments),
            ]
            appendInteger("max_tokens", flag: "--max-tokens", from: arguments, to: &args)
            appendNumber("temperature", flag: "--temperature", from: arguments, to: &args)
            args += ["--no-thinking", "--require-installed", "--quiet"]
            return .init(
                command: ["text", "chat"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args,
                outputs: ["text": valueOutput(.string)],
                streamsEvents: false,
                stdoutOutputName: "text"
            )
        case "vision.ground":
            let outputImage = artifacts.appendingPathComponent("image.png")
            let detections = artifacts.appendingPathComponent("detections.json")
            let masks = artifacts.appendingPathComponent("masks", isDirectory: true)
            var args = [
                "vision", "ground", try requiredString("image", in: arguments),
                "--query",
            ]
            args.append(contentsOf: try requiredStringArray("queries", in: arguments))
            appendString("model", flag: "--model", from: arguments, to: &args)
            args += [
                "--output", outputImage.path,
                "--json-output", detections.path,
                "--mask-output-dir", masks.path,
            ]
            return .init(
                command: ["vision", "ground"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: [
                    "image": fileOutput(outputImage, contentTypes: ["image/png"]),
                    "detections": fileOutput(detections, contentTypes: ["application/json"]),
                    "masks": directoryOutput(masks, contentTypes: ["image/png"]),
                ],
                streamsEvents: false
            )
        case "vision.segment":
            let outputImage = artifacts.appendingPathComponent("image.png")
            let segments = artifacts.appendingPathComponent("segments.json")
            let masks = artifacts.appendingPathComponent("masks", isDirectory: true)
            var args = [
                "vision", "segment", try requiredString("image", in: arguments),
                "--prompt",
            ]
            args.append(contentsOf: try requiredStringArray("prompts", in: arguments))
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendNumber("threshold", flag: "--threshold", from: arguments, to: &args)
            appendInteger("resolution", flag: "--resolution", from: arguments, to: &args)
            appendFlag("show_boxes", flag: "--show-boxes", from: arguments, to: &args)
            appendFlag("multimask", flag: "--multimask", from: arguments, to: &args)
            args += [
                "--output", outputImage.path,
                "--json-output", segments.path,
                "--mask-output-dir", masks.path,
            ]
            return .init(
                command: ["vision", "segment"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: [
                    "image": fileOutput(outputImage, contentTypes: ["image/png"]),
                    "segments": fileOutput(segments, contentTypes: ["application/json"]),
                    "masks": directoryOutput(masks, contentTypes: ["image/png"]),
                ],
                streamsEvents: false
            )
        case "vision.track":
            let outputVideo = artifacts.appendingPathComponent("video.mp4")
            let tracks = artifacts.appendingPathComponent("tracks.json")
            let masks = artifacts.appendingPathComponent("masks", isDirectory: true)
            var args = [
                "vision", "track", try requiredString("video", in: arguments),
                "--prompt",
            ]
            args.append(contentsOf: try requiredStringArray("prompts", in: arguments))
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendNumber("threshold", flag: "--threshold", from: arguments, to: &args)
            appendInteger("resolution", flag: "--resolution", from: arguments, to: &args)
            appendInteger("init_frame", flag: "--init-frame", from: arguments, to: &args)
            appendInteger("end_frame", flag: "--end-frame", from: arguments, to: &args)
            appendFlag("show_boxes", flag: "--show-boxes", from: arguments, to: &args)
            appendFlag("show_labels", flag: "--show-labels", from: arguments, to: &args)
            args += [
                "--output", outputVideo.path,
                "--json-output", tracks.path,
                "--mask-output-dir", masks.path,
            ]
            return .init(
                command: ["vision", "track"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: [
                    "video": fileOutput(outputVideo, contentTypes: ["video/mp4"]),
                    "tracks": fileOutput(tracks, contentTypes: ["application/json"]),
                    "masks": directoryOutput(masks, contentTypes: ["image/png"]),
                ],
                streamsEvents: false
            )
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

    private static func directoryOutput(_ url: URL, contentTypes: [String]) -> WorkflowInvocationOutput {
        WorkflowInvocationOutput(
            type: .assetDirectory,
            path: url.path,
            optional: false,
            contentTypes: contentTypes
        )
    }

    private static func valueOutput(_ type: WorkflowPortType) -> WorkflowInvocationOutput {
        WorkflowInvocationOutput(
            type: type,
            path: nil,
            optional: false,
            contentTypes: []
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

    static func outputExtension(contentTypes: [String]) -> String {
        switch contentTypes.first {
        case "image/png": ".png"
        case "image/jpeg": ".jpg"
        case "image/webp": ".webp"
        case "video/mp4": ".mp4"
        case "audio/wav": ".wav"
        case "image/tiff": ".tif"
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

    private static func requiredStringArray(
        _ name: String,
        in arguments: [String: WorkflowValue]
    ) throws -> [String] {
        guard case .array(let values)? = arguments[name] else {
            throw ValidationError("Workflow node argument '\(name)' did not resolve to an array.")
        }
        let strings = values.compactMap(\.stringValue)
        guard strings.count == values.count, !strings.isEmpty, strings.allSatisfy({ !$0.isEmpty }) else {
            throw ValidationError("Workflow node argument '\(name)' must contain one or more strings.")
        }
        return strings
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
