import Foundation
import MLX

public enum LTX25NativeModelPackKind: String, Codable, CaseIterable, Sendable {
    case connector
    case distilled
    case dev
}

public struct LTX25NativeModelPackResult: Codable, Hashable, Sendable {
    public let kind: LTX25NativeModelPackKind
    public let sourceURL: URL
    public let outputURL: URL
    public let tensorCount: Int
    public let sourceBytes: Int64
    public let packedBytes: Int64
}

public enum LTX25NativeModelPackError: LocalizedError {
    case sourceMissing(URL)
    case outputExists(URL)
    case unsupportedDType(DType)
    case truncatedTensor(String)
    case invalidOutput(URL)

    public var errorDescription: String? {
        switch self {
        case .sourceMissing(let url):
            "Missing LTX 2.5 transformer source: \(url.path)"
        case .outputExists(let url):
            "Native LTX 2.5 model pack already exists: \(url.path)"
        case .unsupportedDType(let dtype):
            "Cannot write \(dtype) to an LTX 2.5 native safetensors pack."
        case .truncatedTensor(let key):
            "The LTX 2.5 source ended while copying tensor \(key)."
        case .invalidOutput(let url):
            "The generated LTX 2.5 native model pack failed validation: \(url.path)"
        }
    }
}

/// Rewrites the official packed transformer into the exact native module-key
/// namespace and physical tensor order used by mere.run. Tensor payload bytes
/// are copied without materializing the 22B checkpoint or changing precision.
public enum LTX25NativeModelPack {
    public static let format = "mere-run-ltx25-native-v1"
    public static let relativeDirectory = ".mere-run/ltx25-native-v1"

    public static func outputURL(
        resources: LTX25Resources,
        kind: LTX25NativeModelPackKind
    ) -> URL {
        let filename = switch kind {
        case .connector: "connector-bf16.safetensors"
        case .distilled, .dev: "\(kind.rawValue)-transformer-bf16.safetensors"
        }
        return resources.rootURL
            .appendingPathComponent(relativeDirectory, isDirectory: true)
            .appendingPathComponent(filename)
    }

    public static func optimizedURLIfValid(
        resources: LTX25Resources,
        kind: LTX25NativeModelPackKind,
        fileManager: FileManager = .default
    ) -> URL? {
        let url = outputURL(resources: resources, kind: kind)
        guard fileManager.fileExists(atPath: url.path),
              let metadata = try? SafetensorsStreamingLoader.fileMetadata(url: url),
              metadata["format"] == format,
              metadata["kind"] == kind.rawValue,
              metadata["source_revision"] == LTX25Resources.sourceRevision else {
            return nil
        }
        return url
    }

    public static func isNativePack(_ url: URL) -> Bool {
        guard let metadata = try? SafetensorsStreamingLoader.fileMetadata(url: url) else {
            return false
        }
        return metadata["format"] == format
            && metadata["source_revision"] == LTX25Resources.sourceRevision
    }

    public static func optimize(
        resources: LTX25Resources,
        kind: LTX25NativeModelPackKind,
        replacing: Bool = false,
        progressHandler: ((Int, Int) -> Void)? = nil,
        fileManager: FileManager = .default
    ) throws -> LTX25NativeModelPackResult {
        let sourceURL = switch kind {
        case .connector, .distilled: resources.distilledTransformerURL
        case .dev: resources.devTransformerURL
        }
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            throw LTX25NativeModelPackError.sourceMissing(sourceURL)
        }
        let outputURL = outputURL(resources: resources, kind: kind)
        if fileManager.fileExists(atPath: outputURL.path), !replacing {
            throw LTX25NativeModelPackError.outputExists(outputURL)
        }

        let sourceMetadata = try SafetensorsStreamingLoader.metadata(url: sourceURL)
        let tensors = try sourceMetadata.compactMap { key, tensor -> LTX25NativePackTensor? in
            let mapped: String?
            switch kind {
            case .connector:
                mapped = isLTX25ConnectorTensorKey(key) ? key : nil
            case .distilled, .dev:
                mapped = mapUnifiedTransformerKey(key)
            }
            guard let mapped else { return nil }
            return LTX25NativePackTensor(
                sourceKey: key,
                outputKey: mapped,
                metadata: tensor,
                dtype: try safetensorsDType(tensor.dtype)
            )
        }.sorted { lhs, rhs in
            lhs.metadata.startOffset < rhs.metadata.startOffset
        }
        var runningOffset = 0
        var headers: [String: LTX25NativePackTensorHeader] = [:]
        headers.reserveCapacity(tensors.count)
        for tensor in tensors {
            let byteCount = tensor.metadata.endOffset - tensor.metadata.startOffset
            headers[tensor.outputKey] = LTX25NativePackTensorHeader(
                dtype: tensor.dtype,
                shape: tensor.metadata.shape,
                dataOffsets: [runningOffset, runningOffset + byteCount]
            )
            runningOffset += byteCount
        }
        let sourceBytes = Int64(
            try sourceURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        )
        let fileMetadata = [
            "format": format,
            "kind": kind.rawValue,
            "source_revision": LTX25Resources.sourceRevision,
            "source_filename": sourceURL.lastPathComponent,
            "source_bytes": String(sourceBytes),
        ]
        let header = LTX25NativePackHeader(tensors: headers, metadata: fileMetadata)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var headerData = try encoder.encode(header)
        let padding = (8 - (headerData.count % 8)) % 8
        if padding > 0 {
            headerData.append(Data(repeating: 0x20, count: padding))
        }

        try fileManager.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let temporaryURL = outputURL.deletingLastPathComponent().appendingPathComponent(
            ".\(outputURL.lastPathComponent).\(UUID().uuidString).tmp"
        )
        _ = fileManager.createFile(atPath: temporaryURL.path, contents: nil)
        do {
            let source = try FileHandle(forReadingFrom: sourceURL)
            let output = try FileHandle(forWritingTo: temporaryURL)
            defer {
                try? source.close()
                try? output.close()
            }
            var headerLength = UInt64(headerData.count).littleEndian
            try withUnsafeBytes(of: &headerLength) { bytes in
                try output.write(contentsOf: bytes)
            }
            try output.write(contentsOf: headerData)
            let copyChunkBytes = 8 * 1_024 * 1_024
            let copyChunk: (Int, String) throws -> Int = { requested, sourceKey in
                guard let data = try source.read(upToCount: requested),
                      data.count == requested else {
                    throw LTX25NativeModelPackError.truncatedTensor(sourceKey)
                }
                try output.write(contentsOf: data)
                return data.count
            }
            for (index, tensor) in tensors.enumerated() {
                try source.seek(toOffset: UInt64(tensor.metadata.startOffset))
                var remaining = tensor.metadata.endOffset - tensor.metadata.startOffset
                while remaining > 0 {
                    let requested = min(remaining, copyChunkBytes)
                    #if canImport(ObjectiveC)
                    let copied = try autoreleasepool {
                        try copyChunk(requested, tensor.sourceKey)
                    }
                    #else
                    let copied = try copyChunk(requested, tensor.sourceKey)
                    #endif
                    remaining -= copied
                }
                progressHandler?(index + 1, tensors.count)
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
        guard optimizedURLIfValid(
            resources: resources,
            kind: kind,
            fileManager: fileManager
        ) == outputURL else {
            throw LTX25NativeModelPackError.invalidOutput(outputURL)
        }
        let packedBytes = Int64(
            try outputURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        )
        return LTX25NativeModelPackResult(
            kind: kind,
            sourceURL: sourceURL,
            outputURL: outputURL,
            tensorCount: tensors.count,
            sourceBytes: sourceBytes,
            packedBytes: packedBytes
        )
    }

    private static func safetensorsDType(_ dtype: DType) throws -> String {
        switch dtype {
        case .float32: "F32"
        case .float16: "F16"
        case .bfloat16: "BF16"
        case .int64: "I64"
        case .int32: "I32"
        case .int16: "I16"
        case .int8: "I8"
        case .uint8: "U8"
        case .uint32: "U32"
        case .bool: "BOOL"
        default: throw LTX25NativeModelPackError.unsupportedDType(dtype)
        }
    }
}

func isLTX25ConnectorTensorKey(_ key: String) -> Bool {
    key.hasPrefix("model.diffusion_model.video_embeddings_connector.")
        || key.hasPrefix("model.diffusion_model.audio_embeddings_connector.")
}

private struct LTX25NativePackTensor {
    let sourceKey: String
    let outputKey: String
    let metadata: SafetensorsStreamingLoader.TensorMetadata
    let dtype: String
}

private struct LTX25NativePackTensorHeader: Encodable {
    let dtype: String
    let shape: [Int]
    let dataOffsets: [Int]

    private enum CodingKeys: String, CodingKey {
        case dtype
        case shape
        case dataOffsets = "data_offsets"
    }
}

private struct LTX25NativePackHeader: Encodable {
    let tensors: [String: LTX25NativePackTensorHeader]
    let metadata: [String: String]

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: LTX25NativePackCodingKey.self)
        try container.encode(metadata, forKey: LTX25NativePackCodingKey("__metadata__"))
        for (key, tensor) in tensors {
            try container.encode(tensor, forKey: LTX25NativePackCodingKey(key))
        }
    }
}

private struct LTX25NativePackCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil

    init(_ stringValue: String) {
        self.stringValue = stringValue
    }

    init?(stringValue: String) {
        self.init(stringValue)
    }

    init?(intValue: Int) {
        return nil
    }
}
