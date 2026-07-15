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
    let acceleratorBackends: [String]
    let minimumAcceleratorMemoryBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case minimumMereRunVersion = "minimum_mere_run_version"
        case nodeKinds = "node_kinds"
        case modelIDs = "model_ids"
        case acceleratorBackends = "accelerator_backends"
        case minimumAcceleratorMemoryBytes = "minimum_accelerator_memory_bytes"
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
            guard WorkflowNodeRegistry.entry(for: node.kind)?.inputs.contains(where: { $0.name == "seed" }) == true,
                  node.arguments["seed"] == nil else {
                return node
            }
            var arguments = node.arguments
            arguments["seed"] = .integer(seed())
            return WorkflowNode(id: node.id, kind: node.kind, arguments: arguments, dependsOn: node.dependsOn)
        }
        return WorkflowGraphDocument(
            schemaVersion: graph.schemaVersion,
            kind: graph.kind,
            name: graph.name,
            inputs: graph.inputs,
            nodes: nodes,
            outputs: graph.outputs,
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
            guard let catalog = WorkflowNodeRegistry.entry(for: node.kind) else { return node }
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
                case .assetArray:
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
            return WorkflowNode(id: node.id, kind: node.kind, arguments: arguments, dependsOn: node.dependsOn)
        }
        return WorkflowGraphDocument(
            schemaVersion: graph.schemaVersion,
            kind: graph.kind,
            name: graph.name,
            inputs: graph.inputs,
            nodes: nodes,
            outputs: graph.outputs,
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
        let modelIDs = Set(graph.nodes.compactMap { node -> String? in
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
        return WorkflowJobRequirements(
            minimumMereRunVersion: MereRunCLIVersion.current,
            nodeKinds: Array(Set(graph.nodes.map(\.kind))).sorted(),
            modelIDs: modelIDs.sorted(),
            acceleratorBackends: ["cuda", "metal"],
            minimumAcceleratorMemoryBytes: nil
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

struct GraphRunNodeRecord: Codable, Equatable, Sendable {
    let id: String
    let kind: String
    var state: GraphRunState
    var startedAt: Date?
    var completedAt: Date?
    var exitStatus: Int32?
    var fingerprint: String
    var artifacts: [GraphRunArtifact]
    var error: String?

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case state
        case startedAt = "started_at"
        case completedAt = "completed_at"
        case exitStatus = "exit_status"
        case fingerprint
        case artifacts
        case error
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

    enum CodingKeys: String, CodingKey {
        case sequence
        case createdAt = "created_at"
        case type
        case state
        case nodeID = "node_id"
        case message
    }
}

struct WorkflowNodeInvocation: Equatable {
    let command: [String]
    let preflightArguments: [String]
    let runArguments: [String]
    let outputs: [String: URL]
}

enum WorkflowNodeCommandBuilder {
    static func invocation(
        node: WorkflowNode,
        arguments: [String: WorkflowValue],
        nodeDirectory: URL
    ) throws -> WorkflowNodeInvocation {
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
            appendInteger("seed", flag: "--seed", from: arguments, to: &args)
            appendInteger("rank", flag: "--rank", from: arguments, to: &args)
            appendString("sample_prompt", flag: "--sample-prompt", from: arguments, to: &args)
            return .init(
                command: ["image", "train-lora"],
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: ["adapter": output]
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
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: ["image": output]
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
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: ["video": output]
            )
        default:
            throw ValidationError("Unsupported workflow node kind '\(node.kind)'.")
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
}
