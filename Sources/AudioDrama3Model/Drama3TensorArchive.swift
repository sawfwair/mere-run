#if !os(iOS)
import Foundation
import MLX
import MereRunTensor

/// Typed, non-executing safetensors reader with bounded file I/O. FP8 bytes are
/// decoded explicitly; MLX has no native E4M3FN storage dtype. Each read
/// materializes only one tensor, including on volumes that disallow safe mapping.
public final class Drama3TensorArchive {
    public struct Tensor: Decodable {
        public let dtype: String
        public let shape: [Int]
        public let offsets: [Int]
        enum CodingKeys: String, CodingKey { case dtype, shape, offsets = "data_offsets" }
    }
    private struct Header: Decodable {
        let tensors: [String: Tensor]
        struct Key: CodingKey {
            var stringValue: String
            var intValue: Int? { nil }
            init?(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { return nil }
        }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: Key.self)
            var result: [String: Tensor] = [:]
            for key in c.allKeys where key.stringValue != "__metadata__" {
                result[key.stringValue] = try c.decode(Tensor.self, forKey: key)
            }
            tensors = result
        }
    }
    private let handle: FileHandle
    private let dataStart: Int
    public let tensors: [String: Tensor]

    public init(url: URL) throws {
        handle = try FileHandle(forReadingFrom: url)
        let fileSize = try handle.seekToEnd()
        try handle.seek(toOffset: 0)
        guard fileSize >= 8, fileSize <= Int.max,
              let prefix = try Self.readBytes(8, from: handle), prefix.count == 8 else {
            throw Drama3Error.invalid("truncated safetensors header")
        }
        let length = prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        guard length > 0, length <= 64 * 1_024 * 1_024, length <= fileSize - 8 else {
            throw Drama3Error.invalid("invalid safetensors header length")
        }
        dataStart = 8 + Int(length)
        guard let header = try Self.readBytes(Int(length), from: handle), header.count == Int(length) else {
            throw Drama3Error.invalid("truncated safetensors header")
        }
        tensors = try JSONDecoder().decode(Header.self, from: header).tensors
        for (name, tensor) in tensors {
            let widths = ["F8_E4M3": 1, "F8_E4M3FN": 1, "BF16": 2, "F16": 2, "F32": 4, "U32": 4]
            guard let width = widths[tensor.dtype], !tensor.shape.isEmpty,
                  tensor.shape.allSatisfy({ $0 > 0 }), tensor.offsets.count == 2 else {
                throw Drama3Error.invalid("unsupported tensor metadata: \(name)")
            }
            var bytes = width
            for dim in tensor.shape {
                let product = bytes.multipliedReportingOverflow(by: dim)
                guard !product.overflow else { throw Drama3Error.invalid("tensor size overflow: \(name)") }
                bytes = product.partialValue
            }
            guard tensor.offsets[0] >= 0, tensor.offsets[1] >= tensor.offsets[0],
                  tensor.offsets[1] <= Int(fileSize) - dataStart,
                  tensor.offsets[1] - tensor.offsets[0] == bytes else {
                throw Drama3Error.invalid("tensor data range mismatch: \(name)")
            }
        }
    }

    deinit { try? handle.close() }

    private static func readBytes(_ count: Int, from handle: FileHandle) throws -> Data? {
        #if canImport(ObjectiveC)
        return try autoreleasepool { try handle.read(upToCount: count) }
        #else
        return try handle.read(upToCount: count)
        #endif
    }

    public static func decodeE4M3(_ byte: UInt8) -> Float {
        let exponent = Int((byte >> 3) & 15)
        let mantissa = Int(byte & 7)
        let sign: Float = byte & 128 == 0 ? 1 : -1
        if exponent == 15 && mantissa == 7 { return .nan }
        let value = exponent == 0 ? Float(mantissa) * pow(2, -9)
            : (1 + Float(mantissa) / 8) * pow(2, Float(exponent - 7))
        return sign * value
    }

    public func read(_ name: String) throws -> MLXArray {
        guard let tensor = tensors[name] else { throw Drama3Error.invalid("missing tensor \(name)") }
        let count = tensor.offsets[1] - tensor.offsets[0]
        try handle.seek(toOffset: UInt64(dataStart + tensor.offsets[0]))
        guard let bytes = try Self.readBytes(count, from: handle), bytes.count == count else {
            throw Drama3Error.invalid("truncated tensor data: \(name)")
        }
        switch tensor.dtype {
        case "BF16": return MLXArray(bytes, tensor.shape, dtype: .bfloat16)
        case "F16": return MLXArray(bytes, tensor.shape, dtype: .float16)
        case "F32": return MLXArray(bytes, tensor.shape, dtype: .float32)
        case "U32": return MLXArray(bytes, tensor.shape, dtype: .uint32)
        case "F8_E4M3", "F8_E4M3FN":
            let values = MLXArray(bytes, tensor.shape, dtype: .uint8).asType(.int32)
            let lookup = MLXArray((0..<256).map { Self.decodeE4M3(UInt8($0)) })
            return lookup[values]
        default: throw Drama3Error.invalid("unsupported dtype \(tensor.dtype)")
        }
    }

    public func decodedWeight(_ name: String, scaleArchive: Drama3TensorArchive? = nil) throws -> MLXArray {
        let value = try read(name)
        guard tensors[name]?.dtype.hasPrefix("F8_") == true else { return value }
        let scale = try (scaleArchive ?? self).read(name + "_scale_inv")
        guard value.ndim == 2, value.dim(0).isMultiple(of: 128), value.dim(1).isMultiple(of: 128),
              scale.shape == [value.dim(0) / 128, value.dim(1) / 128] else {
            throw Drama3Error.invalid("invalid FP8 block scales: \(name)")
        }
        let blocked = value.reshaped(value.dim(0) / 128, 128, value.dim(1) / 128, 128)
        return (blocked * scale.asType(.float32).reshaped(scale.dim(0), 1, scale.dim(1), 1))
            .reshaped(value.shape).asType(.bfloat16)
    }
}

public final class Drama3Weights {
    public let index: HFSafetensorsIndex
    private let root: URL
    private var archives: [String: Drama3TensorArchive] = [:]

    public init(root: URL) throws {
        self.root = root
        index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: Data(contentsOf:
            root.appendingPathComponent("model.safetensors.index.json")))
        guard index.shardFilenames.allSatisfy({ !$0.contains("/") && !$0.contains("..") && $0.hasSuffix(".safetensors") }) else {
            throw Drama3Error.invalid("unsafe shard filename")
        }
    }

    private func archive(_ name: String) throws -> Drama3TensorArchive {
        guard let file = index.weightMap[name] else { throw Drama3Error.invalid("missing weight \(name)") }
        if let existing = archives[file] { return existing }
        let loaded = try Drama3TensorArchive(url: root.appendingPathComponent(file))
        archives[file] = loaded
        return loaded
    }

    public func metadata(_ name: String) throws -> Drama3TensorArchive.Tensor {
        guard let metadata = try archive(name).tensors[name] else { throw Drama3Error.invalid("missing tensor metadata: \(name)") }
        return metadata
    }

    public func read(_ name: String) throws -> MLXArray { try archive(name).read(name) }
    public func decodedWeight(_ name: String) throws -> MLXArray {
        let source = try archive(name)
        let scale = index.weightMap[name + "_scale_inv"] == nil ? nil : try archive(name + "_scale_inv")
        return try source.decodedWeight(name, scaleArchive: scale)
    }
}

#endif
