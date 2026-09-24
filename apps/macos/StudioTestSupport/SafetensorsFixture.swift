import Foundation

/// Writes a small safetensors file the way `MLX.save(arrays:)` lays one out — an 8-byte
/// little-endian header length, the JSON header, then each tensor's bytes in header order — so
/// tests can hand the Earth commands and Studio's header reader a real bundle without committing
/// one.
package enum SafetensorsFixture {
    package struct Tensor {
        package let name: String
        package let dtype: String
        package let shape: [Int]
        package let data: Data

        /// A float32 tensor filled with `value`.
        package static func float32(_ name: String, shape: [Int], value: Float = 0) -> Tensor {
            let count = shape.reduce(1, *)
            var bytes = Data(capacity: count * 4)
            var little = value.bitPattern.littleEndian
            let pattern = withUnsafeBytes(of: &little) { Data($0) }
            for _ in 0..<count { bytes.append(pattern) }
            return Tensor(name: name, dtype: "F32", shape: shape, data: bytes)
        }

        /// An int32 tensor holding `values` in row-major order.
        package static func int32(_ name: String, shape: [Int], values: [Int32]) -> Tensor {
            precondition(values.count == shape.reduce(1, *), "\(name) needs \(shape.reduce(1, *)) values")
            var bytes = Data(capacity: values.count * 4)
            for value in values {
                var little = value.littleEndian
                bytes.append(withUnsafeBytes(of: &little) { Data($0) })
            }
            return Tensor(name: name, dtype: "I32", shape: shape, data: bytes)
        }
    }

    package static func data(tensors: [Tensor], metadata: [String: String] = [:]) -> Data {
        var offset = 0
        var entries: [String] = []
        if !metadata.isEmpty {
            let pairs = metadata.keys.sorted().map { "\"\($0)\":\"\(metadata[$0]!)\"" }.joined(separator: ",")
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

    @discardableResult
    package static func write(to url: URL, tensors: [Tensor], metadata: [String: String] = [:]) throws -> URL {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data(tensors: tensors, metadata: metadata).write(to: url, options: .atomic)
        return url
    }
}
