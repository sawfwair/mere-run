import Foundation
import MLX

public enum NemotronOmniExpertPackError: LocalizedError {
    case outputExists(URL)
    case insufficientDisk(required: Int64, available: Int64)
    case invalidExpertInventory(String)
    case unsupportedDType(DType)
    case truncatedTensor(String)
    case invalidOutput(URL)

    public var errorDescription: String? {
        switch self {
        case .outputExists(let url):
            "Nemotron Omni native expert pack already exists: \(url.path)"
        case .insufficientDisk(let required, let available):
            "Nemotron Omni expert packing requires \(required) bytes but only \(available) bytes are available."
        case .invalidExpertInventory(let message):
            "Nemotron Omni expert inventory is invalid: \(message)"
        case .unsupportedDType(let dtype):
            "Nemotron Omni expert packing cannot write dtype \(dtype)."
        case .truncatedTensor(let key):
            "Nemotron Omni source ended while copying expert tensor \(key)."
        case .invalidOutput(let url):
            "Nemotron Omni native expert pack failed validation: \(url.path)"
        }
    }
}

/// Reorders the official per-expert BF16 tensors into the stacked layout used
/// by MLX gather matmul. Payload bytes are copied directly and atomically; no
/// model tensor is converted, rounded, or materialized in unified memory.
public enum NemotronOmniExpertPack {
    public static let format = "mere-run-nemotron-omni-experts-v1"
    public static let relativePath = ".mere-run/nemotron-omni-experts-v1/experts-bf16.safetensors"

    public static func outputURL(
        rootURL: URL,
        fileManager: FileManager = .default
    ) throws -> URL {
        let standardized = rootURL.standardizedFileURL
        let configURL = standardized.appendingPathComponent("config.json")
        let base: URL
        if let destination = try? fileManager.destinationOfSymbolicLink(atPath: configURL.path) {
            let destinationURL = URL(fileURLWithPath: destination)
            let absolute = destinationURL.path.hasPrefix("/")
                ? destinationURL
                : configURL.deletingLastPathComponent().appendingPathComponent(destination)
            base = absolute.deletingLastPathComponent()
        } else {
            base = standardized
        }
        return base.appendingPathComponent(relativePath)
    }

    public static func optimizedURLIfValid(
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> URL? {
        guard let url = try? outputURL(rootURL: rootURL, fileManager: fileManager),
              fileManager.fileExists(atPath: url.path),
              let metadata = try? SafetensorsStreamingLoader.fileMetadata(url: url),
              metadata["format"] == format,
              metadata["source_revision"] == NemotronOmniResources.upstreamRevision,
              metadata["payload_bytes"] == String(NemotronOmniResources.packedExpertWeightBytes),
              let tensors = try? SafetensorsStreamingLoader.metadata(url: url),
              tensors.count == 46 else {
            return nil
        }
        return url
    }

    @discardableResult
    public static func optimize(
        rootURL: URL,
        replacing: Bool = false,
        progressHandler: ((Int, Int) -> Void)? = nil,
        fileManager: FileManager = .default
    ) throws -> URL {
        let rootURL = rootURL.standardizedFileURL
        let outputURL = try outputURL(rootURL: rootURL, fileManager: fileManager)
        if fileManager.fileExists(atPath: outputURL.path), !replacing {
            throw NemotronOmniExpertPackError.outputExists(outputURL)
        }
        var capacityURL = outputURL.deletingLastPathComponent()
        while !fileManager.fileExists(atPath: capacityURL.path),
              capacityURL.path != "/" {
            capacityURL.deleteLastPathComponent()
        }
        let capacity = try capacityURL.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        ).volumeAvailableCapacityForImportantUsage ?? 0
        let reserve: Int64 = 8 * 1_073_741_824
        let required = NemotronOmniResources.packedExpertWeightBytes + reserve
        guard capacity <= 0 || capacity >= required else {
            throw NemotronOmniExpertPackError.insufficientDisk(
                required: required,
                available: capacity
            )
        }

        let indexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        let index = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: indexURL)
        )
        let shardMetadata = try Dictionary(uniqueKeysWithValues: index.shardFilenames.map { filename in
            let url = rootURL.appendingPathComponent(filename)
            return (filename, (url, try SafetensorsStreamingLoader.metadata(url: url)))
        })
        var groups: [NemotronOmniPackedExpertGroupKey: [Int: NemotronOmniPackedExpertSource]] = [:]
        for (key, filename) in index.weightMap {
            guard let parsed = NemotronOmniExpertWeightKey(checkpointKey: key) else { continue }
            guard let shard = shardMetadata[filename], let metadata = shard.1[key] else {
                throw NemotronOmniExpertPackError.invalidExpertInventory(
                    "missing metadata for \(key)"
                )
            }
            let group = NemotronOmniPackedExpertGroupKey(
                layer: parsed.layer,
                projection: parsed.projection
            )
            groups[group, default: [:]][parsed.expert] = NemotronOmniPackedExpertSource(
                key: key,
                url: shard.0,
                metadata: metadata
            )
        }

        let expectedLayers = NemotronOmniPackedExpertGroupKey.expectedLayers
        var outputGroups: [NemotronOmniPackedExpertGroup] = []
        outputGroups.reserveCapacity(expectedLayers.count * 2)
        for layer in expectedLayers {
            for projection in [
                NemotronOmniExpertWeightKey.Projection.up,
                NemotronOmniExpertWeightKey.Projection.down,
            ] {
                let groupKey = NemotronOmniPackedExpertGroupKey(
                    layer: layer,
                    projection: projection
                )
                guard let values = groups[groupKey], values.count == 128 else {
                    throw NemotronOmniExpertPackError.invalidExpertInventory(
                        "layer \(layer) \(projection.rawValue) has \(groups[groupKey]?.count ?? 0)/128 experts"
                    )
                }
                let sources = try (0..<128).map { expert in
                    guard let source = values[expert] else {
                        throw NemotronOmniExpertPackError.invalidExpertInventory(
                            "layer \(layer) \(projection.rawValue) is missing expert \(expert)"
                        )
                    }
                    return source
                }
                guard let first = sources.first,
                      sources.allSatisfy({
                          $0.metadata.dtype == first.metadata.dtype
                              && $0.metadata.shape == first.metadata.shape
                      }) else {
                    throw NemotronOmniExpertPackError.invalidExpertInventory(
                        "layer \(layer) \(projection.rawValue) tensors disagree on shape or dtype"
                    )
                }
                outputGroups.append(NemotronOmniPackedExpertGroup(
                    outputKey:
                        "backbone.layers.\(layer).mixer.experts.\(projection.rawValue).weight",
                    sources: sources,
                    dtype: try safetensorsDType(first.metadata.dtype),
                    shape: [128] + first.metadata.shape
                ))
            }
        }
        outputGroups.sort { $0.outputKey < $1.outputKey }

        var runningOffset = 0
        var headers: [String: NemotronOmniPackedExpertHeader] = [:]
        headers.reserveCapacity(outputGroups.count)
        for group in outputGroups {
            let byteCount = group.sources.reduce(0) {
                $0 + ($1.metadata.endOffset - $1.metadata.startOffset)
            }
            headers[group.outputKey] = NemotronOmniPackedExpertHeader(
                dtype: group.dtype,
                shape: group.shape,
                dataOffsets: [runningOffset, runningOffset + byteCount]
            )
            runningOffset += byteCount
        }
        guard Int64(runningOffset) == NemotronOmniResources.packedExpertWeightBytes else {
            throw NemotronOmniExpertPackError.invalidExpertInventory(
                "expected \(NemotronOmniResources.packedExpertWeightBytes) payload bytes; found \(runningOffset)"
            )
        }

        let header = NemotronOmniPackedExpertFileHeader(
            tensors: headers,
            metadata: [
                "format": format,
                "source_repo": NemotronOmniResources.upstreamRepoID,
                "source_revision": NemotronOmniResources.upstreamRevision,
                "payload_bytes": String(runningOffset),
            ]
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var headerData = try encoder.encode(header)
        let padding = (8 - (headerData.count % 8)) % 8
        headerData.append(Data(repeating: 0x20, count: padding))

        try fileManager.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let temporaryURL = outputURL.deletingLastPathComponent().appendingPathComponent(
            ".\(outputURL.lastPathComponent).\(UUID().uuidString).tmp"
        )
        _ = fileManager.createFile(atPath: temporaryURL.path, contents: nil)
        do {
            let output = try FileHandle(forWritingTo: temporaryURL)
            var handles: [URL: FileHandle] = [:]
            defer {
                try? output.close()
                for handle in handles.values { try? handle.close() }
            }
            var headerLength = UInt64(headerData.count).littleEndian
            try withUnsafeBytes(of: &headerLength) { try output.write(contentsOf: $0) }
            try output.write(contentsOf: headerData)
            let sourceCount = outputGroups.reduce(0) { $0 + $1.sources.count }
            var copiedSources = 0
            for group in outputGroups {
                for source in group.sources {
                    let handle: FileHandle
                    if let existing = handles[source.url] {
                        handle = existing
                    } else {
                        handle = try FileHandle(forReadingFrom: source.url)
                        handles[source.url] = handle
                    }
                    try handle.seek(toOffset: UInt64(source.metadata.startOffset))
                    var remaining = source.metadata.endOffset - source.metadata.startOffset
                    while remaining > 0 {
                        let requested = min(remaining, 8 * 1_024 * 1_024)
                        #if canImport(ObjectiveC)
                        let data = try autoreleasepool {
                            try handle.read(upToCount: requested)
                        }
                        #else
                        let data = try handle.read(upToCount: requested)
                        #endif
                        guard let data, data.count == requested else {
                            throw NemotronOmniExpertPackError.truncatedTensor(source.key)
                        }
                        try output.write(contentsOf: data)
                        remaining -= data.count
                    }
                    copiedSources += 1
                    if copiedSources.isMultiple(of: 32) || copiedSources == sourceCount {
                        progressHandler?(copiedSources, sourceCount)
                    }
                }
            }
            try output.synchronize()
        } catch {
            try? fileManager.removeItem(at: temporaryURL)
            throw error
        }
        if fileManager.fileExists(atPath: outputURL.path) {
            try fileManager.removeItem(at: outputURL)
        }
        try fileManager.moveItem(at: temporaryURL, to: outputURL)
        guard optimizedURLIfValid(rootURL: rootURL, fileManager: fileManager) == outputURL else {
            throw NemotronOmniExpertPackError.invalidOutput(outputURL)
        }
        return outputURL
    }

    private static func safetensorsDType(_ dtype: DType) throws -> String {
        switch dtype {
        case .bfloat16: "BF16"
        case .float32: "F32"
        default: throw NemotronOmniExpertPackError.unsupportedDType(dtype)
        }
    }
}

private struct NemotronOmniPackedExpertGroupKey: Hashable {
    let layer: Int
    let projection: NemotronOmniExpertWeightKey.Projection

    static let expectedLayers = [
        1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27,
        29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51,
    ]
}

private struct NemotronOmniPackedExpertSource {
    let key: String
    let url: URL
    let metadata: SafetensorsStreamingLoader.TensorMetadata
}

private struct NemotronOmniPackedExpertGroup {
    let outputKey: String
    let sources: [NemotronOmniPackedExpertSource]
    let dtype: String
    let shape: [Int]
}

private struct NemotronOmniPackedExpertHeader: Encodable {
    let dtype: String
    let shape: [Int]
    let dataOffsets: [Int]

    private enum CodingKeys: String, CodingKey {
        case dtype
        case shape
        case dataOffsets = "data_offsets"
    }
}

private struct NemotronOmniPackedExpertFileHeader: Encodable {
    let tensors: [String: NemotronOmniPackedExpertHeader]
    let metadata: [String: String]

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: NemotronOmniPackedExpertCodingKey.self)
        try container.encode(metadata, forKey: NemotronOmniPackedExpertCodingKey("__metadata__"))
        for (key, tensor) in tensors {
            try container.encode(tensor, forKey: NemotronOmniPackedExpertCodingKey(key))
        }
    }
}

private struct NemotronOmniPackedExpertCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil

    init(_ stringValue: String) { self.stringValue = stringValue }
    init?(stringValue: String) { self.init(stringValue) }
    init?(intValue: Int) { return nil }
}
