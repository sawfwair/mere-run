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
            }
        }
    }

    public static func metadata(url: URL) throws -> [String: TensorMetadata] {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let parsed = try parseHeader(fileData: data, fileURL: url)
        return try parseTensorMetadata(
            headerJSON: parsed.headerJSON,
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
        let fileData = try Data(contentsOf: url, options: .mappedIfSafe)
        let parsed = try parseHeader(fileData: fileData, fileURL: url)
        let tensorMetadata = try parseTensorMetadata(
            headerJSON: parsed.headerJSON,
            dataOffset: parsed.dataOffset,
            fileDataCount: fileData.count,
            fileURL: url
        )

        var arrays: [String: MLXArray] = [:]
        arrays.reserveCapacity(tensorMetadata.count)

        for (key, metadata) in tensorMetadata where shouldInclude(key) {
            let tensorData = fileData.subdata(in: metadata.startOffset..<metadata.endOffset)
            let rawArray = MLXArray(tensorData, metadata.shape, dtype: metadata.dtype)
            arrays[key] = HFSafetensorsWeightsLoader.castIfNeeded(rawArray, dtype: dtype)
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
        let fileData = try Data(contentsOf: url, options: .mappedIfSafe)
        let parsed = try parseHeader(fileData: fileData, fileURL: url)
        let tensorMetadata = try parseTensorMetadata(
            headerJSON: parsed.headerJSON,
            dataOffset: parsed.dataOffset,
            fileDataCount: fileData.count,
            fileURL: url
        )

        var updates: [(String, MLXArray)] = []
        updates.reserveCapacity(max(1, batchSize))

        for (key, metadata) in tensorMetadata where include(key) {
            let tensorData = fileData.subdata(in: metadata.startOffset..<metadata.endOffset)
            let rawArray = MLXArray(tensorData, metadata.shape, dtype: metadata.dtype)
            let casted = HFSafetensorsWeightsLoader.castIfNeeded(rawArray, dtype: dtype)
            let mapped = mapper(key, casted)
            if mapped.isEmpty {
                continue
            }
            updates.append(contentsOf: mapped)
            if updates.count >= batchSize {
                try model.update(parameters: ModuleParameters.unflattened(updates), verify: verify)
                updates.removeAll(keepingCapacity: true)
            }
        }

        if !updates.isEmpty {
            try model.update(parameters: ModuleParameters.unflattened(updates), verify: verify)
        }
    }

    private static func parseHeader(
        fileData: Data,
        fileURL: URL
    ) throws -> (headerJSON: [String: Any], dataOffset: Int) {
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
        guard let headerJSON = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw LoaderError.invalidHeader(fileURL)
        }
        return (headerJSON, 8 + headerSize)
    }

    private static func parseTensorMetadata(
        headerJSON: [String: Any],
        dataOffset: Int,
        fileDataCount: Int,
        fileURL: URL
    ) throws -> [String: TensorMetadata] {
        var result: [String: TensorMetadata] = [:]
        result.reserveCapacity(headerJSON.count)

        for (key, value) in headerJSON {
            if key == "__metadata__" {
                continue
            }
            guard let metadataDict = value as? [String: Any],
                  let shapeRaw = metadataDict["shape"] as? [NSNumber],
                  let dtypeRaw = metadataDict["dtype"] as? String,
                  let offsetsRaw = metadataDict["data_offsets"] as? [NSNumber],
                  offsetsRaw.count == 2
            else {
                throw LoaderError.malformedTensorMetadata(fileURL, key)
            }

            let shape = shapeRaw.map { $0.intValue }
            let localStart = offsetsRaw[0].intValue
            let localEnd = offsetsRaw[1].intValue
            let absoluteStart = dataOffset + localStart
            let absoluteEnd = dataOffset + localEnd

            guard absoluteStart >= dataOffset,
                  absoluteEnd >= absoluteStart,
                  absoluteEnd <= fileDataCount else {
                throw LoaderError.invalidTensorDataRange(fileURL, key)
            }

            let dtype = safetensorsDTypeToDType(dtypeRaw)
            result[key] = TensorMetadata(
                shape: shape,
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
}
