import ArgumentParser
import Foundation
import MereRunCore
import MereRunRelayKit

/// Owns localized inputs, verified outputs, cache entries, and output reuse.
struct WorkflowArtifactStore {
    let bundleDirectory: URL
    let runDirectory: URL
    let cacheDirectory: URL?
    let resume: Bool
    let fileManager: FileManager

    func restoreCachedOutputs(
        fingerprint: String,
        invocation: WorkflowNodeInvocation,
        node: WorkflowNode,
        nodeDirectory: URL
    ) throws -> WorkflowVerifiedNodeOutputs? {
        guard let cacheDirectory else { return nil }
        let entry = cacheDirectory.appendingPathComponent(fingerprint, isDirectory: true)
        let manifestURL = entry.appendingPathComponent(WorkflowNodeCacheManifest.filename)
        guard fileManager.fileExists(atPath: manifestURL.path) else { return nil }
        do {
            let manifest = try WorkflowBundleCodec.decoder().decode(
                WorkflowNodeCacheManifest.self,
                from: Data(contentsOf: manifestURL)
            )
            guard manifest.contractVersion == WorkflowNodeCacheManifest.contractVersion,
                  manifest.fingerprint == fingerprint else {
                throw ValidationError("Node cache manifest does not match its fingerprint.")
            }
            try clearAttemptOutputs(invocation.outputs, nodeDirectory: nodeDirectory)
            var providerValues: [String: WorkflowValue] = [:]
            let filesRoot = entry.appendingPathComponent("files", isDirectory: true)
            for output in manifest.outputs {
                guard let descriptor = invocation.outputs[output.name], descriptor.type == output.type else {
                    throw ValidationError("Node cache output contract has changed for '\(output.name)'.")
                }
                if let relativePath = output.relativePath {
                    guard let descriptorPath = descriptor.path else {
                        throw ValidationError("Node cache output '\(output.name)' no longer has a path.")
                    }
                    let destination = try invocationOutputURL(descriptorPath, nodeDirectory: nodeDirectory)
                    guard try nodeRelativePath(destination, nodeDirectory: nodeDirectory) == relativePath else {
                        throw ValidationError("Node cache output path has changed for '\(output.name)'.")
                    }
                    let source = try confinedCacheURL(relativePath, root: filesRoot)
                    try copyCacheItem(from: source, to: destination)
                } else if let value = output.value {
                    providerValues[output.name] = value
                }
            }
            let verified = try verifyOutputs(
                invocation.outputs,
                providerValues: providerValues,
                node: node,
                nodeDirectory: nodeDirectory
            )
            let restoredRecords = try verified.outputs.map {
                try cacheOutput($0, nodeDirectory: nodeDirectory)
            }.sorted { $0.name < $1.name }
            guard restoredRecords == manifest.outputs.sorted(by: { $0.name < $1.name }) else {
                throw ValidationError("Node cache output hashes do not match its manifest.")
            }
            return verified
        } catch {
            try? clearAttemptOutputs(invocation.outputs, nodeDirectory: nodeDirectory)
            try? fileManager.removeItem(at: entry)
            return nil
        }
    }

    func storeCachedOutputs(
        _ verified: WorkflowVerifiedNodeOutputs,
        fingerprint: String,
        policy: WorkflowNodeCachePolicy,
        nodeDirectory: URL
    ) throws {
        guard let cacheDirectory else { return }
        try fileManager.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        let entry = cacheDirectory.appendingPathComponent(fingerprint, isDirectory: true)
        if fileManager.fileExists(atPath: entry.path), policy == .automatic { return }
        let staging = cacheDirectory.appendingPathComponent(".\(fingerprint).\(UUID().uuidString)", isDirectory: true)
        defer { try? fileManager.removeItem(at: staging) }
        let filesRoot = staging.appendingPathComponent("files", isDirectory: true)
        try fileManager.createDirectory(at: filesRoot, withIntermediateDirectories: true)
        let outputs = try verified.outputs.map { output -> WorkflowNodeCacheOutput in
            let record = try cacheOutput(output, nodeDirectory: nodeDirectory)
            if let relativePath = record.relativePath {
                let source = try invocationOutputURL(relativePath, nodeDirectory: nodeDirectory)
                let destination = try confinedCacheURL(relativePath, root: filesRoot)
                try copyCacheItem(from: source, to: destination)
            }
            return record
        }.sorted { $0.name < $1.name }
        try WorkflowBundleCodec.write(
            WorkflowNodeCacheManifest(
                contractVersion: WorkflowNodeCacheManifest.contractVersion,
                fingerprint: fingerprint,
                outputs: outputs
            ),
            to: staging.appendingPathComponent(WorkflowNodeCacheManifest.filename)
        )
        if fileManager.fileExists(atPath: entry.path) {
            try fileManager.removeItem(at: entry)
        }
        try fileManager.moveItem(at: staging, to: entry)
    }

    func cacheOutput(
        _ output: GraphRunNodeOutput,
        nodeDirectory: URL
    ) throws -> WorkflowNodeCacheOutput {
        guard let sha256 = output.sha256 else {
            throw ValidationError("Workflow output '\(output.name)' has no cache fingerprint.")
        }
        let relativePath: String?
        if let path = output.path {
            relativePath = try nodeRelativePath(artifactURL(for: path), nodeDirectory: nodeDirectory)
        } else {
            relativePath = nil
        }
        return WorkflowNodeCacheOutput(
            name: output.name,
            type: output.type,
            value: output.value,
            relativePath: relativePath,
            contentType: output.contentType,
            sizeBytes: output.sizeBytes,
            sha256: sha256
        )
    }

    func nodeRelativePath(_ url: URL, nodeDirectory: URL) throws -> String {
        let candidate = url.standardizedFileURL.path
        let root = nodeDirectory.standardizedFileURL.path
        guard candidate.hasPrefix(root + "/") else {
            throw ValidationError("Workflow cache output escapes the node directory: \(candidate)")
        }
        return String(candidate.dropFirst(root.count + 1))
    }

    func confinedCacheURL(_ path: String, root: URL) throws -> URL {
        guard isConfinedRelativeWorkflowPath(path) else {
            throw ValidationError("Workflow cache contains an unconfined path: \(path)")
        }
        let candidate = root.appendingPathComponent(path).standardizedFileURL
        guard candidate.path.hasPrefix(root.standardizedFileURL.path + "/") else {
            throw ValidationError("Workflow cache path escapes its entry: \(path)")
        }
        try requireResolvedContainment(candidate, in: root)
        return candidate
    }

    func copyCacheItem(from source: URL, to destination: URL) throws {
        let values = try source.resourceValues(forKeys: [.isSymbolicLinkKey])
        guard values.isSymbolicLink != true else {
            throw ValidationError("Workflow cache items cannot be symbolic links: \(source.path)")
        }
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try fileManager.copyItem(at: source, to: destination)
    }

    func clearAttemptOutputs(
        _ outputs: [String: WorkflowInvocationOutput],
        nodeDirectory: URL
    ) throws {
        for descriptor in outputs.values {
            guard let path = descriptor.path else { continue }
            let outputURL = try invocationOutputURL(path, nodeDirectory: nodeDirectory)
            if fileManager.fileExists(atPath: outputURL.path) {
                try fileManager.removeItem(at: outputURL)
            }
        }
        let artifacts = nodeDirectory.appendingPathComponent("artifacts", isDirectory: true)
        if fileManager.fileExists(atPath: artifacts.path) {
            try fileManager.removeItem(at: artifacts)
        }
        let stdout = nodeDirectory.appendingPathComponent("stdout.txt")
        if fileManager.fileExists(atPath: stdout.path) {
            try fileManager.removeItem(at: stdout)
        }
    }

    func verifyOutputs(
        _ descriptors: [String: WorkflowInvocationOutput],
        providerValues: [String: WorkflowValue]?,
        node: WorkflowNode,
        nodeDirectory: URL
    ) throws -> WorkflowVerifiedNodeOutputs {
        var artifacts: [GraphRunArtifact] = []
        var records: [GraphRunNodeOutput] = []
        var values: [String: WorkflowValue] = [:]
        for name in descriptors.keys.sorted() {
            guard let descriptor = descriptors[name] else { continue }
            let providerValue = providerValues?[name]
            switch descriptor.type {
            case .asset, .assetCollection, .assetArray:
                guard let path = descriptor.path else {
                    throw ValidationError("Node '\(node.id)' output '\(name)' has no declared path.")
                }
                let url = try invocationOutputURL(path, nodeDirectory: nodeDirectory)
                if providerValue == .null || !fileManager.fileExists(atPath: url.path) {
                    if descriptor.optional { continue }
                    throw ValidationError("Node '\(node.id)' did not produce declared output '\(name)'.")
                }
                if let providerPath = providerValue?.stringValue {
                    let reported = try invocationOutputURL(providerPath, nodeDirectory: nodeDirectory)
                    guard reported.standardizedFileURL == url.standardizedFileURL else {
                        throw ValidationError("Node '\(node.id)' reported an unexpected path for output '\(name)'.")
                    }
                }
                let outputArtifact = try artifact(
                    name: name,
                    nodeKind: node.kind,
                    url: url,
                    contentType: descriptor.contentTypes.first
                )
                artifacts.append(outputArtifact)
                records.append(.init(
                    name: name,
                    type: descriptor.type,
                    value: nil,
                    path: outputArtifact.path,
                    contentType: outputArtifact.contentType,
                    sizeBytes: outputArtifact.sizeBytes,
                    sha256: outputArtifact.sha256
                ))
                values[name] = .string(url.path)
            case .assetDirectory:
                guard let path = descriptor.path else {
                    throw ValidationError("Node '\(node.id)' directory output '\(name)' has no declared path.")
                }
                let url = try invocationOutputURL(path, nodeDirectory: nodeDirectory)
                var isDirectory: ObjCBool = false
                if providerValue == .null || !fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) {
                    if descriptor.optional { continue }
                    throw ValidationError("Node '\(node.id)' did not produce declared directory output '\(name)'.")
                }
                guard isDirectory.boolValue else {
                    throw ValidationError("Node '\(node.id)' output '\(name)' is not a directory.")
                }
                if let providerPath = providerValue?.stringValue {
                    let reported = try invocationOutputURL(providerPath, nodeDirectory: nodeDirectory)
                    guard reported.standardizedFileURL == url.standardizedFileURL else {
                        throw ValidationError("Node '\(node.id)' reported an unexpected path for output '\(name)'.")
                    }
                }
                let directory = try directoryIdentity(url)
                let manifestURL = nodeDirectory
                    .appendingPathComponent("artifacts", isDirectory: true)
                    .appendingPathComponent("\(name).manifest.json")
                try WorkflowBundleCodec.write(directory.manifest, to: manifestURL)
                let manifestArtifact = try artifact(
                    name: "\(name)_manifest",
                    nodeKind: node.kind,
                    url: manifestURL,
                    contentType: "application/json"
                )
                artifacts.append(manifestArtifact)
                records.append(.init(
                    name: name,
                    type: .assetDirectory,
                    value: nil,
                    path: try portableArtifactPath(for: url),
                    contentType: descriptor.contentTypes.first,
                    sizeBytes: directory.sizeBytes,
                    sha256: directory.sha256
                ))
                values[name] = .string(url.path)
            case .string, .integer, .number, .boolean, .enumeration, .json:
                guard let providerValue, providerValue != .null else {
                    if descriptor.optional { continue }
                    throw ValidationError("Node '\(node.id)' did not report declared value output '\(name)'.")
                }
                guard workflowValue(providerValue, matches: descriptor.type) else {
                    throw ValidationError("Node '\(node.id)' output '\(name)' has the wrong value type.")
                }
                records.append(.init(
                    name: name,
                    type: descriptor.type,
                    value: providerValue,
                    path: nil,
                    contentType: nil,
                    sizeBytes: nil,
                    sha256: try WorkflowBundleCodec.hash(providerValue)
                ))
                values[name] = providerValue
            }
        }
        if let providerValues {
            let undeclared = Set(providerValues.keys).subtracting(descriptors.keys)
            guard undeclared.isEmpty else {
                throw ValidationError(
                    "Node '\(node.id)' reported undeclared outputs: \(undeclared.sorted().joined(separator: ", "))."
                )
            }
        }
        return WorkflowVerifiedNodeOutputs(artifacts: artifacts, outputs: records, values: values)
    }

    func invocationOutputURL(_ path: String, nodeDirectory: URL) throws -> URL {
        let candidate: URL
        if path.hasPrefix("/") {
            candidate = URL(fileURLWithPath: path).standardizedFileURL
        } else {
            guard isConfinedRelativeWorkflowPath(path) else {
                throw ValidationError("Workflow provider output path is not confined: \(path)")
            }
            candidate = nodeDirectory.appendingPathComponent(path).standardizedFileURL
        }
        let root = nodeDirectory.standardizedFileURL.path
        guard candidate.path.hasPrefix(root + "/") else {
            throw ValidationError("Workflow provider output path escapes the node directory: \(path)")
        }
        try requireResolvedContainment(candidate, in: nodeDirectory)
        try requireResolvedContainment(candidate, in: runDirectory)
        return candidate
    }

    func directoryIdentity(_ directory: URL) throws -> WorkflowDirectoryIdentity {
        let root = directory.resolvingSymlinksInPath()
        var entries: [WorkflowAssetEntry] = []
        for path in try fileManager.subpathsOfDirectory(atPath: root.path).sorted() {
            let url = root.appendingPathComponent(path)
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            if values.isSymbolicLink == true {
                throw ValidationError("Workflow output directories cannot contain symbolic links: \(url.path)")
            }
            guard values.isRegularFile == true else { continue }
            entries.append(WorkflowAssetEntry(
                path: path,
                digest: try ModelArtifactPin.fileSHA256(url),
                sizeBytes: try ModelArtifactPin.fileByteCount(url),
                contentType: contentType(for: url)
            ))
        }
        let manifest = WorkflowOutputDirectoryManifest(contractVersion: "mere.run/output-directory.v1", entries: entries)
        return WorkflowDirectoryIdentity(
            manifest: manifest,
            sizeBytes: entries.reduce(0) { $0 + $1.sizeBytes },
            sha256: try WorkflowBundleCodec.hash(manifest)
        )
    }

    func localizeInputs(
        graph: WorkflowGraphDocument,
        inputs: WorkflowInputsDocument,
        assets: WorkflowAssetManifest
    ) throws -> WorkflowInputsDocument {
        var localized = inputs.values
        let root = try artifactURL(for: "inputs")
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        var groupNames = Set<String>()
        for group in assets.groups {
            guard groupNames.insert(group.name).inserted,
                  group.name.range(of: "^[a-z][a-z0-9-]{0,63}$", options: .regularExpression) != nil else {
                throw ValidationError("Asset manifest contains an invalid or duplicate group '\(group.name)'.")
            }
            let groupRoot = root.appendingPathComponent(group.name, isDirectory: true)
            if group.kind == .assetDirectory {
                try fileManager.createDirectory(at: groupRoot, withIntermediateDirectories: true)
            }
            for entry in group.entries {
                guard isConfinedRelativeWorkflowPath(entry.path),
                      entry.digest.count == 64,
                      entry.digest.allSatisfy({ $0.isHexDigit && !$0.isUppercase }) else {
                    throw ValidationError("Workflow asset manifest contains an invalid path or digest.")
                }
                let source = bundleDirectory
                    .appendingPathComponent("assets", isDirectory: true)
                    .appendingPathComponent("sha256", isDirectory: true)
                    .appendingPathComponent(entry.digest)
                guard fileManager.fileExists(atPath: source.path),
                      try ModelArtifactPin.fileByteCount(source) == entry.sizeBytes,
                      try ModelArtifactPin.fileSHA256(source) == entry.digest else {
                    throw ValidationError("Workflow asset digest verification failed for '\(group.name)/\(entry.path)'.")
                }
                let destination = group.kind == .asset
                    ? root.appendingPathComponent("\(group.name)-\(entry.path)")
                    : groupRoot.appendingPathComponent(entry.path)
                try requireResolvedContainment(destination, in: runDirectory)
                try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                if fileManager.fileExists(atPath: destination.path) {
                    guard try ModelArtifactPin.fileByteCount(destination) == entry.sizeBytes,
                          try ModelArtifactPin.fileSHA256(destination) == entry.digest else {
                        throw ValidationError("Workflow localized input digest mismatch for '\(group.name)/\(entry.path)'.")
                    }
                } else {
                    do {
                        try fileManager.linkItem(at: source, to: destination)
                    } catch {
                        try fileManager.copyItem(at: source, to: destination)
                    }
                }
            }
            if group.kind == .asset, let entry = group.entries.first {
                guard group.entries.count == 1 else {
                    throw ValidationError("Asset group '\(group.name)' must contain exactly one file.")
                }
                localized[group.name] = .string(root.appendingPathComponent("\(group.name)-\(entry.path)").path)
            } else {
                localized[group.name] = .string(groupRoot.path)
            }
        }
        return WorkflowInputsDocument(values: localized)
    }

    func shouldResume(
        _ node: GraphRunNodeRecord,
        expectedFingerprint: String,
        nodeOutputs: inout [String: [String: WorkflowValue]]
    ) throws -> Bool {
        guard resume,
              node.state == .finished,
              node.fingerprint == expectedFingerprint,
              !node.outputs.isEmpty || !node.artifacts.isEmpty else { return false }
        var outputs: [String: WorkflowValue] = [:]
        if !node.outputs.isEmpty {
            for output in node.outputs {
                if let value = output.value {
                    guard try WorkflowBundleCodec.hash(value) == output.sha256 else { return false }
                    outputs[output.name] = value
                    continue
                }
                guard let path = output.path else { return false }
                let url = try artifactURL(for: path)
                if output.type == .assetDirectory {
                    var isDirectory: ObjCBool = false
                    guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory),
                          isDirectory.boolValue,
                          try directoryIdentity(url).sha256 == output.sha256 else { return false }
                } else {
                    guard fileManager.fileExists(atPath: url.path),
                          try ModelArtifactPin.fileByteCount(url) == output.sizeBytes,
                          try ModelArtifactPin.fileSHA256(url) == output.sha256 else { return false }
                }
                outputs[output.name] = .string(url.path)
            }
        } else {
            for artifact in node.artifacts {
                let url = try artifactURL(for: artifact.path)
                guard fileManager.fileExists(atPath: url.path),
                      try ModelArtifactPin.fileByteCount(url) == artifact.sizeBytes,
                      try ModelArtifactPin.fileSHA256(url) == artifact.sha256 else {
                    return false
                }
                outputs[artifact.name] = .string(url.path)
            }
        }
        nodeOutputs[node.id] = outputs
        return true
    }

    func materializeGraphOutputs(
        graph: WorkflowGraphDocument,
        nodeOutputs: [String: [String: WorkflowValue]]
    ) throws -> [GraphRunArtifact] {
        var artifacts: [GraphRunArtifact] = []
        let outputsRoot = runDirectory.appendingPathComponent("outputs", isDirectory: true)
        for name in graph.outputs.keys.sorted() {
            guard case .reference(let rawReference)? = graph.outputs[name] else { continue }
            let reference = try WorkflowReference(rawReference)
            guard case .nodeOutput(let nodeID, let output) = reference.source,
                  let sourceValue = nodeOutputs[nodeID]?[output],
                  let node = graph.nodes.first(where: { $0.id == nodeID }),
                  let outputContract = WorkflowNodeRegistry.output(node: node, name: output) else {
                throw ValidationError("Workflow output '\(name)' was not produced.")
            }
            if outputContract.type != .asset && outputContract.type != .assetCollection && outputContract.type != .assetArray {
                let destination = try artifactURL(for: outputsRoot.appendingPathComponent("\(name).json").path)
                try WorkflowBundleCodec.encoder().encode(sourceValue).write(to: destination, options: .atomic)
                artifacts.append(try artifact(
                    name: name,
                    nodeKind: "graph.output",
                    url: destination,
                    contentType: "application/json"
                ))
                continue
            }
            guard let sourcePath = sourceValue.stringValue else {
                throw ValidationError("Workflow output '\(name)' did not resolve to an artifact path.")
            }
            let source = try artifactURL(for: sourcePath)
            let suffix = source.pathExtension.isEmpty ? "" : ".\(source.pathExtension)"
            let destination = try artifactURL(for: outputsRoot.appendingPathComponent("\(name)\(suffix)").path)
            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            do {
                try fileManager.linkItem(at: source, to: destination)
            } catch {
                try fileManager.copyItem(at: source, to: destination)
            }
            artifacts.append(try artifact(name: name, nodeKind: "graph.output", url: destination))
        }
        return artifacts
    }

    func artifact(
        name: String,
        nodeKind: String,
        url: URL,
        contentType explicitContentType: String? = nil
    ) throws -> GraphRunArtifact {
        GraphRunArtifact(
            name: name,
            kind: nodeKind,
            path: try portableArtifactPath(for: url),
            contentType: explicitContentType ?? contentType(for: url),
            sizeBytes: try ModelArtifactPin.fileByteCount(url),
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }

    func artifactURL(for path: String) throws -> URL {
        let candidate = path.hasPrefix("/")
            ? URL(fileURLWithPath: path).standardizedFileURL
            : runDirectory.appendingPathComponent(path).standardizedFileURL
        let root = runDirectory.standardizedFileURL.path
        guard candidate.path == root || candidate.path.hasPrefix(root + "/") else {
            throw ValidationError("Workflow artifact path escapes the run directory: \(path)")
        }
        try requireResolvedContainment(candidate, in: runDirectory)
        return candidate
    }

    func portableArtifactPath(for url: URL) throws -> String {
        let candidate = url.standardizedFileURL.path
        let root = runDirectory.standardizedFileURL.path
        guard candidate.hasPrefix(root + "/") else {
            throw ValidationError("Workflow artifact path escapes the run directory: \(candidate)")
        }
        try requireResolvedContainment(url, in: runDirectory)
        return String(candidate.dropFirst(root.count + 1))
    }

    private func requireResolvedContainment(_ url: URL, in directory: URL) throws {
        let root = directory.resolvingSymlinksInPath().standardizedFileURL.path
        let resolved = url.resolvingSymlinksInPath().standardizedFileURL.path
        guard resolved.hasPrefix(root + "/") else {
            throw ValidationError("Workflow artifact resolves outside its owning directory: \(url.path)")
        }
    }

    func contentType(for url: URL) -> String {
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

struct WorkflowVerifiedNodeOutputs {
    let artifacts: [GraphRunArtifact]
    let outputs: [GraphRunNodeOutput]
    let values: [String: WorkflowValue]
}

struct WorkflowNodeCacheManifest: Codable {
    static let contractVersion = "mere.run/node-cache.v1"
    static let filename = "cache.json"

    let contractVersion: String
    let fingerprint: String
    let outputs: [WorkflowNodeCacheOutput]

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case fingerprint
        case outputs
    }
}

struct WorkflowNodeCacheOutput: Codable, Equatable {
    let name: String
    let type: WorkflowPortType
    let value: WorkflowValue?
    let relativePath: String?
    let contentType: String?
    let sizeBytes: Int64?
    let sha256: String

    enum CodingKeys: String, CodingKey {
        case name
        case type
        case value
        case relativePath = "relative_path"
        case contentType = "content_type"
        case sizeBytes = "size_bytes"
        case sha256
    }
}

struct WorkflowOutputDirectoryManifest: Codable {
    let contractVersion: String
    let entries: [WorkflowAssetEntry]

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case entries
    }
}

struct WorkflowDirectoryIdentity {
    let manifest: WorkflowOutputDirectoryManifest
    let sizeBytes: Int64
    let sha256: String
}


private func workflowValue(_ value: WorkflowValue, matches type: WorkflowPortType) -> Bool {
    switch type {
    case .string, .enumeration, .asset, .assetDirectory:
        value.stringValue != nil
    case .integer:
        value.integerValue != nil
    case .number:
        value.numberValue != nil
    case .boolean:
        value.booleanValue != nil
    case .json:
        true
    case .assetCollection, .assetArray:
        if case .array = value { true } else { false }
    }
}

private func isConfinedRelativeWorkflowPath(_ path: String) -> Bool {
    !path.isEmpty
        && !path.hasPrefix("/")
        && path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { component in
            !component.isEmpty && component != "." && component != ".."
        }
}

