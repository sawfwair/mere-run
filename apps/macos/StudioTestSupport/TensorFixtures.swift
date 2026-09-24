import Foundation

/// Tensor files the Sound and Earth tests write: an `.npy` with the header the reader parses
/// over a zeroed payload, and safetensors files laid out the way `MLX.save(arrays:)` lays one
/// out — an 8-byte little-endian header length, the JSON header, then each tensor's bytes in
/// header order — so tests can hand the Earth commands and Studio's header reader a real
/// bundle without committing one.
public enum TensorFixtures {
    /// One tensor of a safetensors file: its name, safetensors dtype, shape, and bytes.
    public struct Tensor: Sendable {
        public let name: String
        public let dtype: String
        public let shape: [Int]
        public let data: Data

        /// A float32 tensor filled with `value`.
        public static func float32(_ name: String, shape: [Int], value: Float = 0) -> Tensor {
            let count = shape.reduce(1, *)
            var bytes = Data(capacity: count * 4)
            var little = value.bitPattern.littleEndian
            let pattern = withUnsafeBytes(of: &little) { Data($0) }
            for _ in 0..<count { bytes.append(pattern) }
            return Tensor(name: name, dtype: "F32", shape: shape, data: bytes)
        }

        /// An int32 tensor holding `values` in row-major order.
        public static func int32(_ name: String, shape: [Int], values: [Int32]) -> Tensor {
            precondition(values.count == shape.reduce(1, *), "\(name) needs \(shape.reduce(1, *)) values")
            var bytes = Data(capacity: values.count * 4)
            for value in values {
                var little = value.littleEndian
                bytes.append(withUnsafeBytes(of: &little) { Data($0) })
            }
            return Tensor(name: name, dtype: "I32", shape: shape, data: bytes)
        }
    }

    /// A NumPy 1.0 file whose header declares `descriptor` and `shape`, over sixteen zero bytes.
    public static func npy(descriptor: String, shape: String) -> Data {
        var header = "{'descr': '\(descriptor)', 'fortran_order': False, 'shape': \(shape), }"
        let remainder = (16 - ((10 + header.utf8.count + 1) % 16)) % 16
        header += String(repeating: " ", count: remainder) + "\n"
        var data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
        data.append(UInt8(header.utf8.count & 0xff))
        data.append(UInt8((header.utf8.count >> 8) & 0xff))
        data.append(Data(header.utf8))
        data.append(Data(repeating: 0, count: 16))
        return data
    }

    /// A safetensors file of `tensors` in the order given, with the writer's `__metadata__`
    /// strings first in the header.
    public static func safetensors(tensors: [Tensor], metadata: [String: String] = [:]) -> Data {
        var offset = 0
        var entries: [String] = []
        if !metadata.isEmpty {
            let pairs = metadata.keys.sorted().map { "\"\($0)\":\"\(metadata[$0] ?? "")\"" }.joined(separator: ",")
            entries.append("\"__metadata__\":{\(pairs)}")
        }
        for tensor in tensors {
            let end = offset + tensor.data.count
            entries.append(
                "\"\(tensor.name)\":{\"dtype\":\"\(tensor.dtype)\",\"shape\":[\(tensor.shape.map(String.init).joined(separator: ","))],"
                    + "\"data_offsets\":[\(offset),\(end)]}"
            )
            offset = end
        }
        var header = Data("{\(entries.joined(separator: ","))}".utf8)
        // MLX pads the header to eight bytes with spaces; readers tolerate either.
        while header.count % 8 != 0 { header.append(0x20) }
        var length = UInt64(header.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(header)
        for tensor in tensors { file.append(tensor.data) }
        return file
    }

    /// A safetensors file of zeroed float32 tensors, for a test that cares about names and
    /// shapes alone.
    public static func safetensors(_ tensors: [(name: String, shape: [Int])], metadata: [String: String] = [:]) -> Data {
        safetensors(tensors: tensors.map { Tensor.float32($0.name, shape: $0.shape) }, metadata: metadata)
    }

    /// Writes `tensors` as a safetensors file at `url`, making its folder first.
    @discardableResult
    public static func write(to url: URL, tensors: [Tensor], metadata: [String: String] = [:]) throws -> URL {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try safetensors(tensors: tensors, metadata: metadata).write(to: url, options: .atomic)
        return url
    }
}
