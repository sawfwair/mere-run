import Foundation
import MLX
import MLXNN

public enum SafetensorsStreamingLoader {
    public struct TensorMetadata: Sendable, Hashable {
        public let shape: [Int]
        public let dtype: DType
        public let startOffset: Int
        public let endOffset: Int

        public init(shape: [Int], dtype: DType, startOffset: Int, endOffset: Int) {
            self.shape = shape
            self.dtype = dtype
            self.startOffset = startOffset
            self.endOffset = endOffset
        }
    }

    public enum LoaderError: LocalizedError {
        case fileTooSmall(URL)
        case truncatedHeader(URL)
        case invalidHeader(URL)
        case malformedTensorMetadata(URL, String)
        case invalidTensorDataRange(URL, String)
        case missingTensorPair(URL, String)

        public var errorDescription: String? {
            switch self {
            case .fileTooSmall(let url):
                return "Invalid safetensors file (too small): \(url.path)"
            case .truncatedHeader(let url):
                return "Invalid safetensors file (header truncated): \(url.path)"
            case .invalidHeader(let url):
                return "Invalid safetensors header JSON: \(url.path)"
            case .malformedTensorMetadata(let url, let key):
                return "Malformed tensor metadata for key '\(key)' in \(url.path)"
            case .invalidTensorDataRange(let url, let key):
                return "Invalid tensor data range for key '\(key)' in \(url.path)"
            case .missingTensorPair(let url, let key):
                return "Missing safetensors pair for key '\(key)' in \(url.path)"
            }
        }
    }

    public static func metadata(url: URL) throws -> [String: TensorMetadata] {
        let parsed = try parseHeader(fileURL: url)
        return try parseTensorMetadata(
            header: parsed.header,
            dataOffset: parsed.dataOffset,
            fileDataCount: parsed.fileSize,
            fileURL: url
        )
    }

    public static func fileMetadata(url: URL) throws -> [String: String] {
        try parseHeader(fileURL: url).header.fileMetadata
    }

    static func metadata(fileData data: Data, fileURL url: URL) throws -> [String: TensorMetadata] {
        let parsed = try parseHeader(fileData: data, fileURL: url)
        return try parseTensorMetadata(
            header: parsed.header,
            dataOffset: parsed.dataOffset,
            fileDataCount: data.count,
            fileURL: url
        )
    }

    public static func loadArrays(
        url: URL,
        where shouldInclude: (String) -> Bool = { _ in true },
        dtype: DType? = nil
    ) throws -> [String: MLXArray] {
        let parsed = try parseHeader(fileURL: url)
        let tensorMetadata = try parseTensorMetadata(
            header: parsed.header,
            dataOffset: parsed.dataOffset,
            fileDataCount: parsed.fileSize,
            fileURL: url
        )
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var arrays: [String: MLXArray] = [:]
        arrays.reserveCapacity(tensorMetadata.count)

        let orderedMetadata = tensorMetadata.sorted { lhs, rhs in
            lhs.value.startOffset < rhs.value.startOffset
        }
        for (key, metadata) in orderedMetadata where shouldInclude(key) {
            arrays[key] = try makeArray(
                metadata: metadata,
                fileHandle: handle,
                fileURL: url,
                dtype: dtype
            )
        }

        return arrays
    }

    public static func applyWeightsStreaming(
        url: URL,
        to model: Module,
        dtype: DType? = nil,
        verify: Module.VerifyUpdate = .none,
        include: (String) -> Bool = { _ in true },
        mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in [(key, value)] },
        batchSize: Int = 32
    ) throws {
        let parsed = try parseHeader(fileURL: url)
        let tensorMetadata = try parseTensorMetadata(
            header: parsed.header,
            dataOffset: parsed.dataOffset,
            fileDataCount: parsed.fileSize,
            fileURL: url
        )
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var updates: [(String, MLXArray)] = []
        updates.reserveCapacity(max(1, batchSize))

        let orderedMetadata = tensorMetadata.sorted { lhs, rhs in
            lhs.value.startOffset < rhs.value.startOffset
        }
        for (key, metadata) in orderedMetadata where include(key) {
            let casted = try makeArray(
                metadata: metadata,
                fileHandle: handle,
                fileURL: url,
                dtype: dtype
            )
            let mapped = mapper(key, casted)
            if mapped.isEmpty {
                continue
            }
            updates.append(contentsOf: mapped)
            if updates.count >= batchSize {
                try model.update(parameters: ModuleParameters.unflattened(updates), verify: verify)
                updates.removeAll(keepingCapacity: true)
                Memory.clearCache()
            }
        }

        if !updates.isEmpty {
            try model.update(parameters: ModuleParameters.unflattened(updates), verify: verify)
            Memory.clearCache()
        }
    }

    /// Installs file-backed MLX load nodes and materializes them in bounded
    /// batches. Native packs use model parameter names directly, so this path
    /// avoids the extra Foundation `Data` copy required by mapped checkpoints.
    public static func applyWeightsLazyMaterialized(
        url: URL,
        to model: Module,
        verify: Module.VerifyUpdate = .none,
        include: (String) -> Bool = { _ in true },
        mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in [(key, value)] },
        batchSize: Int = 32
    ) throws {
        let metadata = try metadata(url: url)
        let arrays = try MLX.loadArrays(url: url)
        let orderedKeys = metadata
            .sorted { lhs, rhs in lhs.value.startOffset < rhs.value.startOffset }
            .map(\.key)

        var updates: [(String, MLXArray)] = []
        updates.reserveCapacity(max(1, batchSize))

        func applyPendingUpdates() throws {
            guard !updates.isEmpty else { return }
            let values = updates.map(\.1)
            try model.update(parameters: ModuleParameters.unflattened(updates), verify: verify)
            Memory.clearCache()
            MLX.eval(values)
            updates.removeAll(keepingCapacity: true)
            Memory.clearCache()
        }

        for key in orderedKeys where include(key) {
            guard let value = arrays[key] else {
                throw LoaderError.missingTensorPair(url, key)
            }
            updates.append(contentsOf: mapper(key, value))
            if updates.count >= batchSize {
                try applyPendingUpdates()
            }
        }
        try applyPendingUpdates()
    }

    /// Reads and yields one named tensor pair at a time. This is intended for
    /// very large LoRA checkpoints where loading all adapters at once would
    /// double resident memory before fusion even begins.
    @discardableResult
    public static func forEachTensorPair(
        url: URL,
        firstSuffix: String,
        secondSuffix: String,
        dtype: DType? = nil,
        body: (_ baseKey: String, _ first: MLXArray, _ second: MLXArray) throws -> Void
    ) throws -> Int {
        let parsed = try parseHeader(fileURL: url)
        let tensorMetadata = try parseTensorMetadata(
            header: parsed.header,
            dataOffset: parsed.dataOffset,
            fileDataCount: parsed.fileSize,
            fileURL: url
        )
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let firstKeys = tensorMetadata
            .filter { $0.key.hasSuffix(firstSuffix) }
            .sorted { lhs, rhs in lhs.value.startOffset < rhs.value.startOffset }
            .map(\.key)
        var pairCount = 0

        for firstKey in firstKeys {
            let baseKey = String(firstKey.dropLast(firstSuffix.count))
            let secondKey = baseKey + secondSuffix
            guard let firstMetadata = tensorMetadata[firstKey],
                  let secondMetadata = tensorMetadata[secondKey] else {
                throw LoaderError.missingTensorPair(url, firstKey)
            }
            let first = try makeArray(
                metadata: firstMetadata,
                fileHandle: handle,
                fileURL: url,
                dtype: dtype
            )
            let second = try makeArray(
                metadata: secondMetadata,
                fileHandle: handle,
                fileURL: url,
                dtype: dtype
            )
            try body(baseKey, first, second)
            pairCount += 1
        }
        return pairCount
    }

    /// Reads selected tensors in file order without retaining the complete checkpoint.
    @discardableResult
    public static func forEachTensor(
        url: URL,
        where shouldInclude: (String) -> Bool,
        dtype: DType? = nil,
        body: (_ key: String, _ value: MLXArray) throws -> Void
    ) throws -> Int {
        let parsed = try parseHeader(fileURL: url)
        let tensorMetadata = try parseTensorMetadata(
            header: parsed.header,
            dataOffset: parsed.dataOffset,
            fileDataCount: parsed.fileSize,
            fileURL: url
        )
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var count = 0
        for (key, metadata) in tensorMetadata.sorted(by: {
            $0.value.startOffset < $1.value.startOffset
        }) where shouldInclude(key) {
            try body(
                key,
                makeArray(
                    metadata: metadata,
                    fileHandle: handle,
                    fileURL: url,
                    dtype: dtype
                )
            )
            count += 1
        }
        return count
    }

    private static func makeArray(
        metadata: TensorMetadata,
        fileHandle: FileHandle,
        fileURL: URL,
        dtype: DType?
    ) throws -> MLXArray {
        let byteCount = metadata.endOffset - metadata.startOffset
        try fileHandle.seek(toOffset: UInt64(metadata.startOffset))
        guard let tensorData = try fileHandle.read(upToCount: byteCount),
              tensorData.count == byteCount else {
            throw LoaderError.invalidTensorDataRange(fileURL, "offset-\(metadata.startOffset)")
        }
        let rawArray = MLXArray(tensorData, metadata.shape, dtype: metadata.dtype)
        return HFSafetensorsWeightsLoader.castIfNeeded(rawArray, dtype: dtype)
    }

    private static func parseHeader(
        fileData: Data,
        fileURL: URL
    ) throws -> (header: SafetensorsHeader, dataOffset: Int) {
        guard fileData.count >= 8 else {
            throw LoaderError.fileTooSmall(fileURL)
        }

        var headerSizeLE: UInt64 = 0
        _ = withUnsafeMutableBytes(of: &headerSizeLE) { destination in
            fileData.copyBytes(to: destination, from: 0..<8)
        }
        let headerSize = Int(UInt64(littleEndian: headerSizeLE))

        guard fileData.count >= 8 + headerSize else {
            throw LoaderError.truncatedHeader(fileURL)
        }

        let headerData = fileData.subdata(in: 8..<(8 + headerSize))
        return (try decodeHeader(headerData, fileURL: fileURL), 8 + headerSize)
    }

    private static func parseHeader(
        fileURL: URL
    ) throws -> (header: SafetensorsHeader, dataOffset: Int, fileSize: Int) {
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }
        let fileSize = Int(try handle.seekToEnd())
        guard fileSize >= 8 else {
            throw LoaderError.fileTooSmall(fileURL)
        }
        try handle.seek(toOffset: 0)
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
            throw LoaderError.fileTooSmall(fileURL)
        }
        var headerSizeLE: UInt64 = 0
        _ = withUnsafeMutableBytes(of: &headerSizeLE) { destination in
            prefix.copyBytes(to: destination)
        }
        let headerSize = Int(UInt64(littleEndian: headerSizeLE))
        guard fileSize >= 8 + headerSize,
              let headerData = try handle.read(upToCount: headerSize),
              headerData.count == headerSize else {
            throw LoaderError.truncatedHeader(fileURL)
        }
        return (
            try decodeHeader(headerData, fileURL: fileURL),
            8 + headerSize,
            fileSize
        )
    }

    private static func decodeHeader(
        _ headerData: Data,
        fileURL: URL
    ) throws -> SafetensorsHeader {
        do {
            return try JSONDecoder().decode(SafetensorsHeader.self, from: headerData)
        } catch let error as SafetensorsHeader.DecodingFailure {
            throw LoaderError.malformedTensorMetadata(fileURL, error.key)
        } catch {
            throw LoaderError.invalidHeader(fileURL)
        }
    }

    private static func parseTensorMetadata(
        header: SafetensorsHeader,
        dataOffset: Int,
        fileDataCount: Int,
        fileURL: URL
    ) throws -> [String: TensorMetadata] {
        var result: [String: TensorMetadata] = [:]
        result.reserveCapacity(header.tensors.count)

        for (key, metadata) in header.tensors {
            let localStart = metadata.dataOffsets[0]
            let localEnd = metadata.dataOffsets[1]
            let absoluteStart = dataOffset + localStart
            let absoluteEnd = dataOffset + localEnd

            guard absoluteStart >= dataOffset,
                  absoluteEnd >= absoluteStart,
                  absoluteEnd <= fileDataCount else {
                throw LoaderError.invalidTensorDataRange(fileURL, key)
            }

            let dtype = safetensorsDTypeToDType(metadata.dtype)
            result[key] = TensorMetadata(
                shape: metadata.shape,
                dtype: dtype,
                startOffset: absoluteStart,
                endOffset: absoluteEnd
            )
        }
        return result
    }

    private static func safetensorsDTypeToDType(_ dtype: String) -> DType {
        switch dtype {
        case "F32": return .float32
        case "F16": return .float16
        case "BF16": return .bfloat16
        case "I64": return .int64
        case "I32": return .int32
        case "I16": return .int16
        case "I8": return .int8
        case "U8": return .uint8
        case "U32": return .uint32
        case "BOOL": return .bool
        default: return .float32
        }
    }

    private struct SafetensorsHeader: Decodable {
        let tensors: [String: TensorHeader]
        let fileMetadata: [String: String]

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: DynamicCodingKey.self)
            var tensors: [String: TensorHeader] = [:]
            tensors.reserveCapacity(container.allKeys.count)

            let metadataKey = DynamicCodingKey("__metadata__")
            self.fileMetadata = try container.decodeIfPresent(
                [String: String].self,
                forKey: metadataKey
            ) ?? [:]

            for key in container.allKeys where key.stringValue != "__metadata__" {
                do {
                    let header = try container.decode(TensorHeader.self, forKey: key)
                    guard header.dataOffsets.count == 2 else {
                        throw DecodingFailure(key: key.stringValue)
                    }
                    tensors[key.stringValue] = header
                } catch let error as DecodingFailure {
                    throw error
                } catch {
                    throw DecodingFailure(key: key.stringValue)
                }
            }
            self.tensors = tensors
        }

        struct DecodingFailure: Error {
            let key: String
        }
    }

    private struct TensorHeader: Decodable {
        let shape: [Int]
        let dtype: String
        let dataOffsets: [Int]

        private enum CodingKeys: String, CodingKey {
            case shape
            case dtype
            case dataOffsets = "data_offsets"
        }
    }
}
