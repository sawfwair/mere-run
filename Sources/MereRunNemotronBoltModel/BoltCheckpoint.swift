import Foundation
import MLX
import MereRunTensor

/// A typed safetensors boundary that retains F8_E4M3 scale bytes verbatim.
/// Ordinary loaders must not reinterpret those bytes as FP32 tensors.
public enum BoltCheckpoint {
    private struct Tensor: Decodable {
        let dtype: String
        let shape: [Int]
        let dataOffsets: [Int]
        enum CodingKeys: String, CodingKey {
            case dtype, shape
            case dataOffsets = "data_offsets"
        }
    }
    private struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    private struct Header: Decodable {
        let tensors: [String: Tensor]
        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: Key.self)
            var decoded: [String: Tensor] = [:]
            for key in container.allKeys where key.stringValue != "__metadata__" {
                decoded[key.stringValue] = try container.decode(Tensor.self, forKey: key)
            }
            tensors = decoded
        }
    }

    public static func loadShard(at url: URL) throws -> [String: MLXArray] {
        // FileHandle bridges through autoreleased Foundation buffers. Drain
        // them at the shard boundary after MLX has copied the tensor payloads;
        // otherwise a full async load can retain a second checkpoint in RAM.
        #if canImport(ObjectiveC)
        return try autoreleasepool { try readShard(at: url) }
        #else
        return try readShard(at: url)
        #endif
    }

    private static func readShard(at url: URL) throws -> [String: MLXArray] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let fileSize = try handle.seekToEnd()
        try handle.seek(toOffset: 0)
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
            throw NemotronBoltError.invalidCheckpoint("Truncated safetensors header")
        }
        let rawSize = prefix.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
        guard rawSize > 0, rawSize <= 16 * 1_024 * 1_024 else {
            throw NemotronBoltError.invalidCheckpoint("Invalid safetensors header size")
        }
        let size = Int(rawSize)
        guard
              let data = try handle.read(upToCount: size), data.count == size else {
            throw NemotronBoltError.invalidCheckpoint("Invalid safetensors header size")
        }
        let header = try JSONDecoder().decode(Header.self, from: data)
        var arrays: [String: MLXArray] = [:]
        var previousEnd = 0
        for (key, tensor) in header.tensors.sorted(by: { $0.value.dataOffsets.first ?? -1 < $1.value.dataOffsets.first ?? -1 }) {
            let dtype: DType
            switch tensor.dtype {
            case "U8", "F8_E4M3": dtype = .uint8
            case "BF16": dtype = .bfloat16
            case "F16": dtype = .float16
            case "F32": dtype = .float32
            case "I64": dtype = .int64
            default: throw NemotronBoltError.invalidCheckpoint("Unsupported tensor dtype: \(tensor.dtype)")
            }
            guard tensor.dataOffsets.count == 2, tensor.shape.allSatisfy({ $0 > 0 }),
                  tensor.dataOffsets[0] == previousEnd,
                  tensor.dataOffsets[1] >= tensor.dataOffsets[0],
                  UInt64(size + 8) <= fileSize,
                  tensor.dataOffsets[1] <= Int(fileSize - UInt64(size + 8)) else {
                throw NemotronBoltError.invalidCheckpoint("Invalid tensor range for \(key)")
            }
            let count = tensor.dataOffsets[1] - tensor.dataOffsets[0]
            var elements = 1
            for dimension in tensor.shape {
                let product = elements.multipliedReportingOverflow(by: dimension)
                guard !product.overflow else { throw NemotronBoltError.invalidCheckpoint("Tensor shape overflow") }
                elements = product.partialValue
            }
            guard elements <= Int.max / dtype.size, elements * dtype.size == count else {
                throw NemotronBoltError.invalidCheckpoint("Tensor byte count differs from shape for \(key)")
            }
            try handle.seek(toOffset: UInt64(size + 8 + tensor.dataOffsets[0]))
            guard let payload = try handle.read(upToCount: count), payload.count == count else {
                throw NemotronBoltError.invalidCheckpoint("Truncated tensor \(key)")
            }
            arrays[key] = MLXArray(payload, tensor.shape, dtype: dtype)
            previousEnd = tensor.dataOffsets[1]
        }
        guard UInt64(previousEnd + size + 8) == fileSize else {
            throw NemotronBoltError.invalidCheckpoint("Unexpected safetensors trailing bytes")
        }
        return arrays
    }
}

struct BoltWeightStore {
    var arrays: [String: MLXArray]
    mutating func take(_ key: String, shape: [Int]? = nil, dtype: DType? = nil) throws -> MLXArray {
        guard let value = arrays.removeValue(forKey: key), shape == nil || value.shape == shape,
              dtype == nil || value.dtype == dtype else {
            throw NemotronBoltError.invalidCheckpoint("Missing or incompatible tensor \(key)")
        }
        return value
    }
}
