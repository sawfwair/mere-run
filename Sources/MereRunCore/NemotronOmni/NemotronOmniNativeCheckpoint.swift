import Foundation

public struct NemotronOmniNativeCheckpointManifest: Codable, Sendable, Hashable {
    public let schemaVersion: Int
    public let format: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let totalWeightBytes: Int64
    public let expertWeightBytes: Int64
    public let nonExpertWeightBytes: Int64
    public let nonExpertShardCount: Int
}

public struct NemotronOmniNativeCheckpointResult: Sendable, Hashable {
    public let outputRootURL: URL
    public let expertPackURL: URL
    public let indexURL: URL
    public let nonExpertShardURLs: [URL]
    public let totalWeightBytes: Int64
}

public enum NemotronOmniNativeCheckpointError: LocalizedError {
    case invalidSource([String])
    case outputExists(URL)
    case insufficientDisk(required: Int64, available: Int64)
    case missingSupportFile(URL)
    case unexpectedNonExpertBytes(expected: Int64, actual: Int64)
    case invalidOutput([String])

    public var errorDescription: String? {
        switch self {
        case .invalidSource(let messages):
            "Nemotron Omni source checkpoint is invalid: \(messages.joined(separator: "; "))"
        case .outputExists(let url):
            "Nemotron Omni native checkpoint already exists: \(url.path)"
        case .insufficientDisk(let required, let available):
            "Nemotron Omni native export requires \(required) bytes but only \(available) bytes are available."
        case .missingSupportFile(let url):
            "Nemotron Omni native export is missing support file: \(url.path)"
        case .unexpectedNonExpertBytes(let expected, let actual):
            "Nemotron Omni native export expected \(expected) non-expert bytes but wrote \(actual)."
        case .invalidOutput(let messages):
            "Nemotron Omni native checkpoint failed validation: \(messages.joined(separator: "; "))"
        }
    }
}

/// Produces a standalone native checkpoint with each parameter stored once.
/// Non-expert tensors retain their upstream keys and payload bytes, while the
/// routed experts are stacked into the physical layout used by the MLX model.
public enum NemotronOmniNativeCheckpoint {
    public static let format = "mere-run-nemotron-omni-native-v1"
    public static let manifestFilename = "mere-run-native-checkpoint.json"

    public static let supportFilenames = [
        "bias.md",
        "chat_template.jinja",
        "config.json",
        "explainability.md",
        "generation_config.json",
        "preprocessor_config.json",
        "privacy.md",
        "safety.md",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    public static func manifestURL(rootURL: URL) -> URL {
        rootURL.appendingPathComponent(manifestFilename)
    }

    public static func validationMessages(
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> [String] {
        var messages: [String] = []
        for filename in supportFilenames {
            let url = rootURL.appendingPathComponent(filename)
            if !fileManager.fileExists(atPath: url.path) {
                messages.append("Missing required file: \(url.path)")
            }
        }
        let manifest: NemotronOmniNativeCheckpointManifest?
        do {
            manifest = try JSONDecoder().decode(
                NemotronOmniNativeCheckpointManifest.self,
                from: Data(contentsOf: manifestURL(rootURL: rootURL))
            )
        } catch {
            messages.append("Invalid \(manifestFilename): \(error.localizedDescription)")
            manifest = nil
        }
        if let manifest,
           manifest.schemaVersion != 1
            || manifest.format != format
            || manifest.sourceRepository != NemotronOmniResources.upstreamRepoID
            || manifest.sourceRevision != NemotronOmniResources.upstreamRevision
            || manifest.totalWeightBytes != NemotronOmniResources.checkpointWeightBytes
            || manifest.expertWeightBytes != NemotronOmniResources.packedExpertWeightBytes
            || manifest.nonExpertWeightBytes != NemotronOmniResources.nativeNonExpertWeightBytes
            || manifest.nonExpertShardCount != NemotronOmniResources.checkpointShardCount {
            messages.append("Native checkpoint manifest does not match the pinned format.")
        }

        let expertURL = rootURL.appendingPathComponent(
            NemotronOmniExpertPack.publishedFilename
        )
        if !NemotronOmniExpertPack.isValid(url: expertURL, fileManager: fileManager) {
            messages.append("Published native expert shard is missing or invalid.")
        }

        let indexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        do {
            let index = try JSONDecoder().decode(
                HFSafetensorsIndex.self,
                from: Data(contentsOf: indexURL)
            )
            let expectedShards = (1...NemotronOmniResources.checkpointShardCount).map {
                String(
                    format: "model-%05d-of-%05d.safetensors",
                    $0,
                    NemotronOmniResources.checkpointShardCount
                )
            }
            if index.metadata?.totalSize != NemotronOmniResources.nativeNonExpertWeightBytes {
                messages.append("Native non-expert index reports the wrong payload size.")
            }
            if index.shardFilenames != expectedShards {
                messages.append("Native non-expert index reports the wrong shard inventory.")
            }
            if index.weightMap.keys.contains(where: {
                NemotronOmniExpertWeightKey(checkpointKey: $0) != nil
            }) {
                messages.append("Native non-expert index still contains routed expert tensors.")
            }
            for filename in expectedShards {
                let url = rootURL.appendingPathComponent(filename)
                if !fileManager.fileExists(atPath: url.path) {
                    messages.append("Missing required file: \(url.path)")
                }
            }
        } catch {
            messages.append("Invalid native safetensors index: \(error.localizedDescription)")
        }
        return messages
    }

    public static func isValid(
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> Bool {
        validationMessages(rootURL: rootURL, fileManager: fileManager).isEmpty
    }

    public static func export(
        sourceRootURL: URL,
        destinationRootURL: URL,
        replacing: Bool = false,
        progressHandler: ((String, Int, Int) -> Void)? = nil,
        fileManager: FileManager = .default
    ) throws -> NemotronOmniNativeCheckpointResult {
        let sourceRootURL = sourceRootURL.standardizedFileURL
        let destinationRootURL = destinationRootURL.standardizedFileURL
        guard !fileManager.fileExists(
            atPath: manifestURL(rootURL: sourceRootURL).path
        ) else {
            throw NemotronOmniNativeCheckpointError.invalidSource([
                "Source is already a standalone native checkpoint.",
            ])
        }
        let sourceMessages = NemotronOmniResources.validationMessages(
            rootURL: sourceRootURL,
            fileManager: fileManager
        )
        guard sourceMessages.isEmpty else {
            throw NemotronOmniNativeCheckpointError.invalidSource(sourceMessages)
        }
        if fileManager.fileExists(atPath: destinationRootURL.path), !replacing {
            throw NemotronOmniNativeCheckpointError.outputExists(destinationRootURL)
        }

        let parentURL = destinationRootURL.deletingLastPathComponent()
        try fileManager.createDirectory(at: parentURL, withIntermediateDirectories: true)
        let available = try NemotronOmniExpertPack.availableCapacity(
            at: parentURL,
            fileManager: fileManager
        )
        let reserve: Int64 = 8 * 1_073_741_824
        let required = NemotronOmniResources.checkpointWeightBytes + reserve
        guard available <= 0 || available >= required else {
            throw NemotronOmniNativeCheckpointError.insufficientDisk(
                required: required,
                available: available
            )
        }

        let stagingRootURL = parentURL.appendingPathComponent(
            ".\(destinationRootURL.lastPathComponent).\(UUID().uuidString).tmp",
            isDirectory: true
        )
        try fileManager.createDirectory(at: stagingRootURL, withIntermediateDirectories: true)
        do {
            try copySupportFiles(
                sourceRootURL: sourceRootURL,
                destinationRootURL: stagingRootURL,
                fileManager: fileManager
            )
            let sourceIndex = try JSONDecoder().decode(
                HFSafetensorsIndex.self,
                from: Data(
                    contentsOf: sourceRootURL.appendingPathComponent(
                        "model.safetensors.index.json"
                    )
                )
            )
            let expertPackURL = stagingRootURL.appendingPathComponent(
                NemotronOmniExpertPack.publishedFilename
            )
            try NemotronOmniExpertPack.optimize(
                rootURL: sourceRootURL,
                destinationURL: expertPackURL,
                progressHandler: { completed, total in
                    progressHandler?("experts", completed, total)
                },
                fileManager: fileManager
            )

            let nativeWeightMap = sourceIndex.weightMap.filter {
                NemotronOmniExpertWeightKey(checkpointKey: $0.key) == nil
            }
            var nonExpertBytes: Int64 = 0
            var nonExpertShardURLs: [URL] = []
            let shardFilenames = sourceIndex.shardFilenames
            for (shardIndex, filename) in shardFilenames.enumerated() {
                let sourceURL = sourceRootURL.appendingPathComponent(filename)
                let destinationURL = stagingRootURL.appendingPathComponent(filename)
                let result = try SafetensorsSubsetWriter.write(
                    sourceURL: sourceURL,
                    destinationURL: destinationURL,
                    fileMetadata: [
                        "format": format,
                        "source_filename": filename,
                        "source_repository": NemotronOmniResources.upstreamRepoID,
                        "source_revision": NemotronOmniResources.upstreamRevision,
                    ],
                    include: { nativeWeightMap[$0] == filename },
                    fileManager: fileManager
                )
                nonExpertBytes += result.payloadBytes
                nonExpertShardURLs.append(destinationURL)
                progressHandler?("non-expert-shards", shardIndex + 1, shardFilenames.count)
            }
            guard nonExpertBytes == NemotronOmniResources.nativeNonExpertWeightBytes else {
                throw NemotronOmniNativeCheckpointError.unexpectedNonExpertBytes(
                    expected: NemotronOmniResources.nativeNonExpertWeightBytes,
                    actual: nonExpertBytes
                )
            }

            let nativeIndex = HFSafetensorsIndex(
                metadata: HFSafetensorsIndex.Metadata(totalSize: nonExpertBytes),
                weightMap: nativeWeightMap
            )
            let indexURL = stagingRootURL.appendingPathComponent(
                "model.safetensors.index.json"
            )
            try writeJSON(nativeIndex, to: indexURL)
            let manifest = NemotronOmniNativeCheckpointManifest(
                schemaVersion: 1,
                format: format,
                sourceRepository: NemotronOmniResources.upstreamRepoID,
                sourceRevision: NemotronOmniResources.upstreamRevision,
                totalWeightBytes: NemotronOmniResources.checkpointWeightBytes,
                expertWeightBytes: NemotronOmniResources.packedExpertWeightBytes,
                nonExpertWeightBytes: nonExpertBytes,
                nonExpertShardCount: nonExpertShardURLs.count
            )
            try writeJSON(manifest, to: manifestURL(rootURL: stagingRootURL))

            let messages = validationMessages(
                rootURL: stagingRootURL,
                fileManager: fileManager
            )
            guard messages.isEmpty else {
                throw NemotronOmniNativeCheckpointError.invalidOutput(messages)
            }
            try installStagingRoot(
                stagingRootURL,
                at: destinationRootURL,
                replacing: replacing,
                fileManager: fileManager
            )
        } catch {
            try? fileManager.removeItem(at: stagingRootURL)
            throw error
        }

        let indexURL = destinationRootURL.appendingPathComponent(
            "model.safetensors.index.json"
        )
        let index = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: indexURL)
        )
        return NemotronOmniNativeCheckpointResult(
            outputRootURL: destinationRootURL,
            expertPackURL: destinationRootURL.appendingPathComponent(
                NemotronOmniExpertPack.publishedFilename
            ),
            indexURL: indexURL,
            nonExpertShardURLs: index.shardFilenames.map {
                destinationRootURL.appendingPathComponent($0)
            },
            totalWeightBytes: NemotronOmniResources.checkpointWeightBytes
        )
    }

    private static func copySupportFiles(
        sourceRootURL: URL,
        destinationRootURL: URL,
        fileManager: FileManager
    ) throws {
        for filename in supportFilenames {
            let sourceURL = sourceRootURL.appendingPathComponent(filename)
            guard fileManager.fileExists(atPath: sourceURL.path) else {
                throw NemotronOmniNativeCheckpointError.missingSupportFile(sourceURL)
            }
            try fileManager.copyItem(
                at: sourceURL,
                to: destinationRootURL.appendingPathComponent(filename)
            )
        }
    }

    private static func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        var data = try encoder.encode(value)
        data.append(0x0A)
        try data.write(to: url, options: .atomic)
    }

    private static func installStagingRoot(
        _ stagingRootURL: URL,
        at destinationRootURL: URL,
        replacing: Bool,
        fileManager: FileManager
    ) throws {
        guard fileManager.fileExists(atPath: destinationRootURL.path) else {
            try fileManager.moveItem(at: stagingRootURL, to: destinationRootURL)
            return
        }
        guard replacing else {
            throw NemotronOmniNativeCheckpointError.outputExists(destinationRootURL)
        }
        let backupURL = destinationRootURL.deletingLastPathComponent().appendingPathComponent(
            ".\(destinationRootURL.lastPathComponent).\(UUID().uuidString).backup",
            isDirectory: true
        )
        try fileManager.moveItem(at: destinationRootURL, to: backupURL)
        do {
            try fileManager.moveItem(at: stagingRootURL, to: destinationRootURL)
            try fileManager.removeItem(at: backupURL)
        } catch {
            if !fileManager.fileExists(atPath: destinationRootURL.path) {
                try? fileManager.moveItem(at: backupURL, to: destinationRootURL)
            }
            throw error
        }
    }
}
