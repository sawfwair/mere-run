import Foundation
import MLX

struct SafetensorsSubsetWriteResult: Sendable, Hashable {
    let outputURL: URL
    let tensorCount: Int
    let payloadBytes: Int64
}

enum SafetensorsSubsetWriterError: LocalizedError {
    case outputExists(URL)
    case emptySelection(URL)
    case unsupportedDType(DType)
    case truncatedTensor(String)
    case invalidOutput(URL)

    var errorDescription: String? {
        switch self {
        case .outputExists(let url):
            "Safetensors output already exists: \(url.path)"
        case .emptySelection(let url):
            "Safetensors selection is empty: \(url.path)"
        case .unsupportedDType(let dtype):
            "Safetensors subset writer cannot preserve dtype \(dtype)."
        case .truncatedTensor(let key):
            "Safetensors source ended while copying tensor \(key)."
        case .invalidOutput(let url):
            "Safetensors subset output failed validation: \(url.path)"
        }
    }
}

/// Rewrites a filtered safetensors file without materializing tensor values.
/// Payload bytes, shapes, dtypes, and source ordering are preserved exactly.
enum SafetensorsSubsetWriter {
    static func write(
        sourceURL: URL,
        destinationURL: URL,
        replacing: Bool = false,
        fileMetadata: [String: String] = [:],
        include: (String) -> Bool,
        progressHandler: ((Int, Int) -> Void)? = nil,
        fileManager: FileManager = .default
    ) throws -> SafetensorsSubsetWriteResult {
        if fileManager.fileExists(atPath: destinationURL.path), !replacing {
            throw SafetensorsSubsetWriterError.outputExists(destinationURL)
        }
        let sourceMetadata = try SafetensorsStreamingLoader.metadata(url: sourceURL)
        let tensors = try sourceMetadata
            .filter { include($0.key) }
            .map { key, metadata in
                SafetensorsSubsetTensor(
                    key: key,
                    metadata: metadata,
                    dtype: try safetensorsDType(metadata.dtype)
                )
            }
            .sorted { $0.metadata.startOffset < $1.metadata.startOffset }
        guard !tensors.isEmpty else {
            throw SafetensorsSubsetWriterError.emptySelection(sourceURL)
        }

        var runningOffset = 0
        var headers: [String: SafetensorsSubsetTensorHeader] = [:]
        headers.reserveCapacity(tensors.count)
        for tensor in tensors {
            let byteCount = tensor.metadata.endOffset - tensor.metadata.startOffset
            headers[tensor.key] = SafetensorsSubsetTensorHeader(
                dtype: tensor.dtype,
                shape: tensor.metadata.shape,
                dataOffsets: [runningOffset, runningOffset + byteCount]
            )
            runningOffset += byteCount
        }

        let header = SafetensorsSubsetHeader(
            tensors: headers,
            metadata: fileMetadata
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var headerData = try encoder.encode(header)
        let padding = (8 - (headerData.count % 8)) % 8
        if padding > 0 {
            headerData.append(Data(repeating: 0x20, count: padding))
        }

        try fileManager.createDirectory(
            at: destinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let temporaryURL = destinationURL.deletingLastPathComponent().appendingPathComponent(
            ".\(destinationURL.lastPathComponent).\(UUID().uuidString).tmp"
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
            for (index, tensor) in tensors.enumerated() {
                try source.seek(toOffset: UInt64(tensor.metadata.startOffset))
                var remaining = tensor.metadata.endOffset - tensor.metadata.startOffset
                while remaining > 0 {
                    let requested = min(remaining, 8 * 1_024 * 1_024)
                    #if canImport(ObjectiveC)
                    let data = try autoreleasepool {
                        try source.read(upToCount: requested)
                    }
                    #else
                    let data = try source.read(upToCount: requested)
                    #endif
                    guard let data, data.count == requested else {
                        throw SafetensorsSubsetWriterError.truncatedTensor(tensor.key)
                    }
                    try output.write(contentsOf: data)
                    remaining -= data.count
                }
                progressHandler?(index + 1, tensors.count)
            }
            try output.synchronize()
        } catch {
            try? fileManager.removeItem(at: temporaryURL)
            throw error
        }

        if fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        }
        try fileManager.moveItem(at: temporaryURL, to: destinationURL)
        guard let outputMetadata = try? SafetensorsStreamingLoader.metadata(url: destinationURL),
              outputMetadata.count == tensors.count,
              outputMetadata.keys.allSatisfy({ headers[$0] != nil }) else {
            throw SafetensorsSubsetWriterError.invalidOutput(destinationURL)
        }
        return SafetensorsSubsetWriteResult(
            outputURL: destinationURL,
            tensorCount: tensors.count,
            payloadBytes: Int64(runningOffset)
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
        default: throw SafetensorsSubsetWriterError.unsupportedDType(dtype)
        }
    }
}

private struct SafetensorsSubsetTensor {
    let key: String
    let metadata: SafetensorsStreamingLoader.TensorMetadata
    let dtype: String
}

private struct SafetensorsSubsetTensorHeader: Encodable {
    let dtype: String
    let shape: [Int]
    let dataOffsets: [Int]

    private enum CodingKeys: String, CodingKey {
        case dtype
        case shape
        case dataOffsets = "data_offsets"
    }
}

private struct SafetensorsSubsetHeader: Encodable {
    let tensors: [String: SafetensorsSubsetTensorHeader]
    let metadata: [String: String]

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: SafetensorsSubsetCodingKey.self)
        if !metadata.isEmpty {
            try container.encode(metadata, forKey: SafetensorsSubsetCodingKey("__metadata__"))
        }
        for (key, tensor) in tensors {
            try container.encode(tensor, forKey: SafetensorsSubsetCodingKey(key))
        }
    }
}

private struct SafetensorsSubsetCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil

    init(_ stringValue: String) { self.stringValue = stringValue }
    init?(stringValue: String) { self.init(stringValue) }
    init?(intValue: Int) { return nil }
}
