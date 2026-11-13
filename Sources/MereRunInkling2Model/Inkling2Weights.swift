import Foundation
import MLX

/// Original safetensors names and byte encodings. Expert slices stay on disk;
/// only a selected expert is expanded to floating point for reference inference.
public final class Inkling2Weights {
    public struct Tensor: Decodable {
        public let dtype: String
        public let shape: [Int]
        let dataOffsets: [Int]
        enum CodingKeys: String, CodingKey { case dtype, shape, dataOffsets = "data_offsets" }
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
    private struct Index: Decodable {
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
    }
    private struct Location { let url: URL; let start: Int; let tensor: Tensor }
    private var locations: [String: Location] = [:]
    private var resident: [String: MLXArray] = [:]

    public init(root: URL) throws {
        let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json")))
        for name in Set(index.weightMap.values).sorted() {
            try Task.checkCancellation()
            guard !name.isEmpty, name == URL(fileURLWithPath: name).lastPathComponent,
                  name.hasSuffix(".safetensors") else { throw Inkling2Error.invalid("invalid shard filename") }
            let url = root.appendingPathComponent(name)
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            let size = try handle.seekToEnd()
            try handle.seek(toOffset: 0)
            let prefix = try Self.readBytes(8, handle: handle)
            let length = prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
            guard length > 0, length <= 64 * 1_024 * 1_024, size >= 8, length <= size - 8 else {
                throw Inkling2Error.invalid("invalid safetensors header")
            }
            let header = try JSONDecoder().decode(Header.self, from: Self.readBytes(Int(length), handle: handle))
            guard Set(header.tensors.keys) == Set(index.weightMap.filter { $0.value == name }.keys) else {
                throw Inkling2Error.invalid("shard ownership differs from index: \(name)")
            }
            for (key, tensor) in header.tensors {
                let widths = ["BF16": 2, "F16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1, "F8_E4M3FN": 1]
                guard let width = widths[tensor.dtype], tensor.shape.allSatisfy({ $0 > 0 }), tensor.dataOffsets.count == 2 else {
                    throw Inkling2Error.invalid("unsupported tensor metadata: \(key)")
                }
                var bytes = width
                for dim in tensor.shape {
                    let product = bytes.multipliedReportingOverflow(by: dim)
                    guard !product.overflow else { throw Inkling2Error.invalid("tensor size overflow: \(key)") }
                    bytes = product.partialValue
                }
                let a = tensor.dataOffsets[0], b = tensor.dataOffsets[1]
                guard a >= 0, b >= a, UInt64(b) <= size - 8 - length, b - a == bytes else {
                    throw Inkling2Error.invalid("invalid tensor range: \(key)")
                }
                locations[key] = Location(url: url, start: Int(length) + 8 + a, tensor: tensor)
            }
        }
    }

    /// Fixture constructor uses the same original checkpoint layout.
    init(arrays: [String: MLXArray]) { resident = arrays }

    /// Header-only schema validation without allocating checkpoint-sized arrays.
    init(metadata: [String: Tensor]) {
        locations = metadata.mapValues { Location(url: URL(fileURLWithPath: "/dev/null"), start: 0, tensor: $0) }
    }

    public var names: Set<String> { Set(locations.keys).union(resident.keys) }

    func shape(_ key: String) -> [Int]? { resident[key]?.shape ?? locations[key]?.tensor.shape }
    func dtype(_ key: String) -> String? {
        if let value = resident[key] {
            switch value.dtype { case .uint8: return "U8"; case .bfloat16: return "BF16"; default: return "F32" }
        }
        return locations[key]?.tensor.dtype
    }

    func require(_ key: String, shape: [Int], bytes: Bool = false) throws {
        guard self.shape(key) == shape,
              bytes || ["BF16", "F16", "F32"].contains(dtype(key) ?? "") else {
            throw Inkling2Error.invalid("missing tensor or shape/dtype mismatch: \(key), expected \(shape)")
        }
    }

    private static func readBytes(_ count: Int, handle: FileHandle) throws -> Data {
        let bytes = try handle.read(upToCount: count) ?? Data()
        guard bytes.count == count else { throw Inkling2Error.invalid("truncated safetensors data") }
        return bytes
    }

    func read(_ key: String, expert: Int? = nil) throws -> MLXArray {
        if let value = resident[key] { return expert.map { value[$0] } ?? value }
        guard let loc = locations[key] else { throw Inkling2Error.invalid("missing tensor \(key)") }
        var shape = loc.tensor.shape
        var offset = loc.start
        var count = loc.tensor.dataOffsets[1] - loc.tensor.dataOffsets[0]
        if let expert {
            guard !shape.isEmpty, (0..<shape[0]).contains(expert) else { throw Inkling2Error.invalid("invalid expert index") }
            count /= shape.removeFirst()
            offset += expert * count
        }
        let handle = try FileHandle(forReadingFrom: loc.url)
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(offset))
        let data = try Self.readBytes(count, handle: handle)
        let type: DType
        switch loc.tensor.dtype {
        case "BF16": type = .bfloat16
        case "F16": type = .float16
        case "F32": type = .float32
        default: type = .uint8
        }
        let array = MLXArray(data, shape, dtype: type)
        if expert == nil, !key.contains(".experts.") {
            eval(array)
            resident[key] = array
        }
        return array
    }

    func expert(_ key: String, index: Int) throws -> MLXArray {
        let raw = try read(key, expert: index)
        switch dtype(key) {
        case "U8":
            let scales = try read(key + "_scale", expert: index)
            let amax = try read(key + "_amax")[index]
            let denominator = try read(key + "_outer_scale_denominator")
            return Inkling2Math.decodeNVFP4(raw, scales: scales, global: amax / denominator)
        case "F8_E4M3", "F8_E4M3FN":
            return Inkling2Math.decodeMXFP8(raw, scales: try read(key + "_scale", expert: index))
        default: return raw
        }
    }
}
