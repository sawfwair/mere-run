import Foundation

/// Tensor files the Sound and Earth tests write: an `.npy` and a safetensors file with the
/// headers the readers parse and zeroed payloads of the declared size.
public enum TensorFixtures {
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

    /// A safetensors file of float32 tensors in the order given, with zeroed payloads and the
    /// writer's `__metadata__` strings.
    public static func safetensors(_ tensors: [(name: String, shape: [Int])], metadata: [String: String] = [:]) -> Data {
        var offset = 0
        var entries: [String] = []
        for tensor in tensors {
            let bytes = tensor.shape.reduce(1, *) * 4
            entries.append(
                "\"\(tensor.name)\":{\"dtype\":\"F32\",\"shape\":[\(tensor.shape.map(String.init).joined(separator: ","))]," +
                "\"data_offsets\":[\(offset),\(offset + bytes)]}"
            )
            offset += bytes
        }
        if !metadata.isEmpty {
            let strings = metadata.keys.sorted().map { "\"\($0)\":\"\(metadata[$0] ?? "")\"" }.joined(separator: ",")
            entries.append("\"__metadata__\":{\(strings)}")
        }
        let header = "{\(entries.joined(separator: ","))}"
        var data = Data()
        withUnsafeBytes(of: UInt64(header.utf8.count).littleEndian) { data.append(contentsOf: $0) }
        data.append(Data(header.utf8))
        data.append(Data(repeating: 0, count: offset))
        return data
    }
}
