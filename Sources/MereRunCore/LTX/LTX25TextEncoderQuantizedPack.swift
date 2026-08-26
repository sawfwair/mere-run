import Foundation
import MLX

public struct LTX25TextEncoderQuantizedPackResult: Codable, Hashable, Sendable {
    public let sourceURL: URL
    public let outputDirectoryURL: URL
    public let indexURL: URL
    public let manifestURL: URL
    public let sourceTensorCount: Int
    public let quantizedTensorCount: Int
    public let shardCount: Int
    public let sourceBytes: Int64
    public let packedBytes: Int64
}

public struct LTX25TextEncoderQuantizedPackManifest: Codable, Hashable, Sendable {
    public let format: String
    public let sourceRevision: String
    public let sourceFilename: String
    public let sourceBytes: Int64
    public let bits: Int
    public let groupSize: Int
    public let sourceTensorCount: Int
    public let quantizedTensorCount: Int
    public let packedBytes: Int64
    public let runtimeAssetsFilename: String
    public let shards: [String]

    private enum CodingKeys: String, CodingKey {
        case format
        case sourceRevision = "source_revision"
        case sourceFilename = "source_filename"
        case sourceBytes = "source_bytes"
        case bits
        case groupSize = "group_size"
        case sourceTensorCount = "source_tensor_count"
        case quantizedTensorCount = "quantized_tensor_count"
        case packedBytes = "packed_bytes"
        case runtimeAssetsFilename = "runtime_assets_filename"
        case shards
    }
}

public enum LTX25TextEncoderQuantizedPackError: LocalizedError {
    case sourceMissing(URL)
    case outputExists(URL)
    case noLanguageTensors(URL)
    case missingTensor(String)
    case invalidOutput(URL)

    public var errorDescription: String? {
        switch self {
        case .sourceMissing(let url):
            "Missing LTX 2.5 packed text encoder: \(url.path)"
        case .outputExists(let url):
            "Quantized LTX 2.5 text-encoder pack already exists: \(url.path)"
        case .noLanguageTensors(let url):
            "No Gemma language tensors were found in \(url.path)"
        case .missingTensor(let key):
            "The LTX 2.5 text-encoder source did not load tensor \(key)"
        case .invalidOutput(let url):
            "The generated LTX 2.5 text-encoder pack failed validation: \(url.path)"
        }
    }
}

/// Builds a local, source-bound MLX affine Q4 pack for the custom Gemma 4
/// language tower embedded in the official LTX 2.5 text encoder. The pack also
/// carries the BF16 LTX projection and embedded tokenizer assets required at runtime.
public enum LTX25TextEncoderQuantizedPack {
    public static let format = "mere-run-ltx25-text-q4-v1"
    public static let relativeDirectory = ".mere-run/ltx25-text-q4-v1"
    public static let runtimeAssetsFilename = "text-encoder-assets-bf16.safetensors"
    public static let bits = 4
    public static let groupSize = 64

    public static func outputDirectoryURL(resources: LTX25Resources) -> URL {
        resources.rootURL.appendingPathComponent(relativeDirectory, isDirectory: true)
    }

    public static func indexURL(resources: LTX25Resources) -> URL {
        outputDirectoryURL(resources: resources)
            .appendingPathComponent("model.safetensors.index.json", isDirectory: false)
    }

    public static func manifestURL(resources: LTX25Resources) -> URL {
        outputDirectoryURL(resources: resources)
            .appendingPathComponent("pack.json", isDirectory: false)
    }

    public static func runtimeAssetsURL(resources: LTX25Resources) -> URL {
        outputDirectoryURL(resources: resources)
            .appendingPathComponent(runtimeAssetsFilename, isDirectory: false)
    }

    public static func runtimeAssetsURL(indexURL: URL) -> URL {
        indexURL.deletingLastPathComponent()
            .appendingPathComponent(runtimeAssetsFilename, isDirectory: false)
    }

    public static func optimizedIndexURLIfValid(
        resources: LTX25Resources,
        fileManager: FileManager = .default
    ) -> URL? {
        let sourceURL = resources.textEncoderURL
        return validatedIndexURL(
            outputDirectory: outputDirectoryURL(resources: resources),
            sourceURL: fileManager.fileExists(atPath: sourceURL.path) ? sourceURL : nil,
            fileManager: fileManager
        )
    }

    private static func validatedIndexURL(
        outputDirectory: URL,
        sourceURL: URL?,
        fileManager: FileManager
    ) -> URL? {
        let manifestURL = outputDirectory.appendingPathComponent(
            "pack.json",
            isDirectory: false
        )
        let indexURL = outputDirectory.appendingPathComponent(
            "model.safetensors.index.json",
            isDirectory: false
        )
        let expectedSourceFilename = sourceURL?.lastPathComponent
            ?? URL(fileURLWithPath: LTX25Resources.textEncoderRelativePath).lastPathComponent
        let expectedSourceBytes = sourceURL.map(fileSize)
            ?? LTX25Resources.textEncoderSourceBytes
        let sourceExists = sourceURL.map { fileManager.fileExists(atPath: $0.path) } ?? true
        guard sourceExists,
              fileManager.fileExists(atPath: manifestURL.path),
              fileManager.fileExists(atPath: indexURL.path),
              let manifestData = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder().decode(
                  LTX25TextEncoderQuantizedPackManifest.self,
                  from: manifestData
              ),
              manifest.format == format,
              manifest.sourceRevision == LTX25Resources.sourceRevision,
              manifest.sourceFilename == expectedSourceFilename,
              manifest.bits == bits,
              manifest.groupSize == groupSize,
              manifest.sourceBytes == expectedSourceBytes,
              manifest.runtimeAssetsFilename == runtimeAssetsFilename,
              !manifest.shards.isEmpty,
              let indexData = try? Data(contentsOf: indexURL),
              let index = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: indexData),
              index.weightMap.keys.contains(where: { $0.hasSuffix(".scales") }),
              index.shardFilenames == manifest.shards.sorted() else {
            return nil
        }
        let runtimeAssetsURL = outputDirectory.appendingPathComponent(
            manifest.runtimeAssetsFilename,
            isDirectory: false
        )
        guard fileManager.fileExists(atPath: runtimeAssetsURL.path),
              let runtimeMetadata = try? SafetensorsStreamingLoader.metadata(
                  url: runtimeAssetsURL
              ),
              requiredRuntimeAssetKeys.allSatisfy({ runtimeMetadata[$0] != nil }),
              let runtimeFileMetadata = try? SafetensorsStreamingLoader.fileMetadata(
                  url: runtimeAssetsURL
              ),
              runtimeFileMetadata["format"] == format,
              runtimeFileMetadata["source_revision"] == LTX25Resources.sourceRevision,
              runtimeFileMetadata["gemma_config"] != nil else {
            return nil
        }
        for shard in manifest.shards {
            let url = outputDirectory.appendingPathComponent(shard, isDirectory: false)
            guard fileManager.fileExists(atPath: url.path),
                  let metadata = try? SafetensorsStreamingLoader.fileMetadata(url: url),
                  metadata["format"] == format,
                  metadata["source_revision"] == LTX25Resources.sourceRevision else {
                return nil
            }
        }
        return indexURL
    }

    public static func artifactURLs(
        resources: LTX25Resources,
        fileManager: FileManager = .default
    ) -> [URL] {
        guard optimizedIndexURLIfValid(
            resources: resources,
            fileManager: fileManager
        ) != nil,
        let data = try? Data(contentsOf: manifestURL(resources: resources)),
        let manifest = try? JSONDecoder().decode(
            LTX25TextEncoderQuantizedPackManifest.self,
            from: data
        ) else {
            return []
        }
        let outputDirectory = outputDirectoryURL(resources: resources)
        let shards = manifest.shards.map {
            outputDirectory.appendingPathComponent($0, isDirectory: false)
        }
        return [
            manifestURL(resources: resources),
            indexURL(resources: resources),
            runtimeAssetsURL(resources: resources),
        ] + shards
    }

    public static func optimize(
        resources: LTX25Resources,
        replacing: Bool = false,
        progressHandler: ((Int, Int) -> Void)? = nil,
        fileManager: FileManager = .default
    ) throws -> LTX25TextEncoderQuantizedPackResult {
        let sourceURL = resources.textEncoderURL
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            throw LTX25TextEncoderQuantizedPackError.sourceMissing(sourceURL)
        }
        let outputDirectory = outputDirectoryURL(resources: resources)
        if fileManager.fileExists(atPath: outputDirectory.path), !replacing {
            throw LTX25TextEncoderQuantizedPackError.outputExists(outputDirectory)
        }

        let sourceMetadata = try SafetensorsStreamingLoader.metadata(url: sourceURL)
        let sourceFileMetadata = try SafetensorsStreamingLoader.fileMetadata(url: sourceURL)
        guard let gemmaConfig = sourceFileMetadata["gemma_config"] else {
            throw LTX25TextEncoderQuantizedPackError.invalidOutput(sourceURL)
        }
        let languageKeys = sourceMetadata.keys.filter(isLanguageTensor).sorted()
        guard !languageKeys.isEmpty else {
            throw LTX25TextEncoderQuantizedPackError.noLanguageTensors(sourceURL)
        }
        let groups = makeGroups(keys: languageKeys)
        let sourceBytes = fileSize(sourceURL)
        let temporaryDirectory = outputDirectory.deletingLastPathComponent()
            .appendingPathComponent(
                ".ltx25-text-q4-\(UUID().uuidString).tmp",
                isDirectory: true
            )
        try fileManager.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )

        var weightMap: [String: String] = [:]
        var shardNames: [String] = []
        var packedBytes: Int64 = 0
        var quantizedTensorCount = 0
        do {
            for (index, group) in groups.enumerated() {
                let shardName = String(
                    format: "model-%05d-of-%05d.safetensors",
                    index + 1,
                    groups.count
                )
                let shardURL = temporaryDirectory.appendingPathComponent(
                    shardName,
                    isDirectory: false
                )
                let keySet = Set(group.keys)
                let sourceArrays = try SafetensorsStreamingLoader.loadArrays(
                    url: sourceURL,
                    where: { keySet.contains($0) }
                )
                var outputArrays: [String: MLXArray] = [:]
                for sourceKey in group.keys {
                    guard let sourceArray = sourceArrays[sourceKey],
                          let metadata = sourceMetadata[sourceKey] else {
                        throw LTX25TextEncoderQuantizedPackError.missingTensor(sourceKey)
                    }
                    let outputKey = nativeLanguageKey(sourceKey)
                    if shouldQuantize(key: sourceKey, metadata: metadata) {
                        let quantized = MLX.quantized(
                            sourceArray,
                            groupSize: groupSize,
                            bits: bits,
                            mode: .affine
                        )
                        let baseKey = String(outputKey.dropLast(".weight".count))
                        outputArrays[outputKey] = quantized.wq
                        outputArrays[baseKey + ".scales"] = quantized.scales
                        if let biases = quantized.biases {
                            outputArrays[baseKey + ".biases"] = biases
                        }
                        quantizedTensorCount += 1
                    } else {
                        outputArrays[outputKey] = sourceArray
                    }
                }
                MLX.eval(Array(outputArrays.values))
                try MLX.save(
                    arrays: outputArrays,
                    metadata: [
                        "format": format,
                        "source_revision": LTX25Resources.sourceRevision,
                        "bits": String(bits),
                        "group_size": String(groupSize),
                    ],
                    url: shardURL
                )
                for key in outputArrays.keys {
                    weightMap[key] = shardName
                }
                shardNames.append(shardName)
                packedBytes += fileSize(shardURL)
                Memory.clearCache()
                progressHandler?(index + 1, groups.count)
            }

            let runtimeAssets = try SafetensorsStreamingLoader.loadArrays(
                url: sourceURL,
                where: { requiredRuntimeAssetKeys.contains($0) }
            )
            guard requiredRuntimeAssetKeys.allSatisfy({ runtimeAssets[$0] != nil }) else {
                throw LTX25TextEncoderQuantizedPackError.invalidOutput(sourceURL)
            }
            MLX.eval(Array(runtimeAssets.values))
            let runtimeAssetsURL = temporaryDirectory.appendingPathComponent(
                runtimeAssetsFilename,
                isDirectory: false
            )
            try MLX.save(
                arrays: runtimeAssets,
                metadata: [
                    "format": format,
                    "source_revision": LTX25Resources.sourceRevision,
                    "gemma_config": gemmaConfig,
                ],
                url: runtimeAssetsURL
            )
            packedBytes += fileSize(runtimeAssetsURL)
            Memory.clearCache()

            let index = LTX25TextEncoderPackIndex(
                metadata: .init(totalSize: packedBytes),
                weightMap: weightMap
            )
            try writeJSON(
                index,
                to: temporaryDirectory.appendingPathComponent(
                    "model.safetensors.index.json",
                    isDirectory: false
                )
            )
            let manifest = LTX25TextEncoderQuantizedPackManifest(
                format: format,
                sourceRevision: LTX25Resources.sourceRevision,
                sourceFilename: sourceURL.lastPathComponent,
                sourceBytes: sourceBytes,
                bits: bits,
                groupSize: groupSize,
                sourceTensorCount: languageKeys.count,
                quantizedTensorCount: quantizedTensorCount,
                packedBytes: packedBytes,
                runtimeAssetsFilename: runtimeAssetsFilename,
                shards: shardNames
            )
            try writeJSON(
                manifest,
                to: temporaryDirectory.appendingPathComponent("pack.json", isDirectory: false)
            )
        } catch {
            try? fileManager.removeItem(at: temporaryDirectory)
            throw error
        }

        if fileManager.fileExists(atPath: outputDirectory.path) {
            try fileManager.removeItem(at: outputDirectory)
        }
        try fileManager.moveItem(at: temporaryDirectory, to: outputDirectory)
        guard let validIndexURL = optimizedIndexURLIfValid(
            resources: resources,
            fileManager: fileManager
        ) else {
            throw LTX25TextEncoderQuantizedPackError.invalidOutput(outputDirectory)
        }
        return LTX25TextEncoderQuantizedPackResult(
            sourceURL: sourceURL,
            outputDirectoryURL: outputDirectory,
            indexURL: validIndexURL,
            manifestURL: manifestURL(resources: resources),
            sourceTensorCount: languageKeys.count,
            quantizedTensorCount: quantizedTensorCount,
            shardCount: shardNames.count,
            sourceBytes: sourceBytes,
            packedBytes: packedBytes
        )
    }

    private static func isLanguageTensor(_ key: String) -> Bool {
        key == "model.embed_tokens.weight"
            || key == "model.norm.weight"
            || key.hasPrefix("model.layers.")
    }

    private static let requiredRuntimeAssetKeys: Set<String> = [
        "hf_asset__tokenizer_config.json",
        "text_embedding_projection.audio_aggregate_embed.bias",
        "text_embedding_projection.audio_aggregate_embed.weight",
        "text_embedding_projection.video_aggregate_embed.bias",
        "text_embedding_projection.video_aggregate_embed.weight",
        "tokenizer_json",
    ]

    private static func shouldQuantize(
        key: String,
        metadata: SafetensorsStreamingLoader.TensorMetadata
    ) -> Bool {
        let belongsToQuantizedModule = key == "model.embed_tokens.weight"
            || key.hasPrefix("model.layers.")
        return belongsToQuantizedModule
            && key.hasSuffix(".weight")
            && metadata.shape.count == 2
            && metadata.shape[0].isMultiple(of: 32)
            && metadata.shape[1].isMultiple(of: groupSize)
    }

    private static func nativeLanguageKey(_ sourceKey: String) -> String {
        String(sourceKey.dropFirst("model.".count))
    }

    private static func makeGroups(keys: [String]) -> [LTX25TextEncoderTensorGroup] {
        Dictionary(grouping: keys, by: groupName)
            .map { name, keys in
                LTX25TextEncoderTensorGroup(name: name, keys: keys.sorted())
            }
            .sorted { groupOrder($0.name) < groupOrder($1.name) }
    }

    private static func groupName(_ key: String) -> String {
        if key == "model.embed_tokens.weight" {
            return "embed_tokens"
        }
        let components = key.split(separator: ".")
        if components.count > 2, components[0] == "model", components[1] == "layers" {
            return "layer-\(components[2])"
        }
        return "final"
    }

    private static func groupOrder(_ name: String) -> Int {
        if name == "embed_tokens" { return 0 }
        if name.hasPrefix("layer-"), let layer = Int(name.dropFirst("layer-".count)) {
            return layer + 1
        }
        return Int.max
    }

    private static func fileSize(_ url: URL) -> Int64 {
        Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
    }

    private static func writeJSON<Value: Encodable>(_ value: Value, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(value).write(to: url, options: .atomic)
    }
}

private struct LTX25TextEncoderTensorGroup {
    let name: String
    let keys: [String]
}

private struct LTX25TextEncoderPackIndex: Encodable {
    struct Metadata: Encodable {
        let totalSize: Int64

        private enum CodingKeys: String, CodingKey {
            case totalSize = "total_size"
        }
    }

    let metadata: Metadata
    let weightMap: [String: String]

    private enum CodingKeys: String, CodingKey {
        case metadata
        case weightMap = "weight_map"
    }
}
