import Foundation

// The headers of the tensor files Studio runs write — `.npy` from `sfx ae encode`, safetensors
// from the Earth commands and `sfx condition text` — read without loading the tensor, so the
// tensor inspector can say what a file holds. Both formats put a self-describing header first.

/// The header of a NumPy `.npy` file: format version, dtype descriptor, shape, and byte order.
package struct StudioNPYMetadata: Equatable {
    package let version: String
    package let descriptor: String
    package let shape: String
    package let fortranOrder: Bool
    package let byteCount: Int

    package init(version: String, descriptor: String, shape: String, fortranOrder: Bool, byteCount: Int) {
        self.version = version
        self.descriptor = descriptor
        self.shape = shape
        self.fortranOrder = fortranOrder
        self.byteCount = byteCount
    }

    /// The header alone, off the front of the file (`StudioTensorHeader.load(from:)`), with
    /// `byteCount` from the file's size rather than a read of it.
    package static func load(from url: URL) -> StudioNPYMetadata? {
        guard case .npy(let metadata)? = StudioTensorHeader.load(from: url) else { return nil }
        return metadata
    }

    package static func decode(_ data: Data) -> StudioNPYMetadata? {
        guard data.count >= 10,
              Array(data.prefix(6)) == [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59] else {
            return nil
        }
        let major = Int(data[6])
        let minor = Int(data[7])
        let headerLength: Int
        let headerStart: Int
        if major <= 1 {
            headerLength = Int(data[8]) | (Int(data[9]) << 8)
            headerStart = 10
        } else {
            guard data.count >= 12 else { return nil }
            headerLength = Int(data[8])
                | (Int(data[9]) << 8)
                | (Int(data[10]) << 16)
                | (Int(data[11]) << 24)
            headerStart = 12
        }
        guard headerLength >= 0, data.count >= headerStart + headerLength,
              let header = String(
                  data: data.subdata(in: headerStart..<(headerStart + headerLength)),
                  encoding: .ascii
              ) else {
            return nil
        }
        return StudioNPYMetadata(
            version: "\(major).\(minor)",
            descriptor: dictionaryValue("descr", in: header) ?? "unknown",
            shape: tupleValue("shape", in: header) ?? "unknown",
            fortranOrder: header.contains("'fortran_order': True")
                || header.contains("\"fortran_order\": true"),
            byteCount: data.count
        )
    }

    private static func dictionaryValue(_ key: String, in header: String) -> String? {
        for quote in ["'", "\""] {
            let marker = "\(quote)\(key)\(quote)"
            guard let keyRange = header.range(of: marker),
                  let colon = header[keyRange.upperBound...].firstIndex(of: ":") else { continue }
            let tail = header[header.index(after: colon)...]
                .trimmingCharacters(in: .whitespaces)
            guard let first = tail.first, first == "'" || first == "\"",
                  let end = tail.dropFirst().firstIndex(of: first) else { continue }
            return String(tail[tail.index(after: tail.startIndex)..<end])
        }
        return nil
    }

    private static func tupleValue(_ key: String, in header: String) -> String? {
        for quote in ["'", "\""] {
            let marker = "\(quote)\(key)\(quote)"
            guard let keyRange = header.range(of: marker),
                  let open = header[keyRange.upperBound...].firstIndex(of: "("),
                  let close = header[open...].firstIndex(of: ")") else { continue }
            return String(header[open...close])
        }
        return nil
    }
}

/// The header of a safetensors file: one entry per tensor with its dtype, shape, and byte range,
/// plus the free-form `__metadata__` strings a writer may add. The format is an 8-byte
/// little-endian header length followed by that many bytes of JSON.
package struct StudioSafetensorsHeader: Equatable {
    package struct Tensor: Equatable, Identifiable {
        package let name: String
        package let dtype: String
        package let shape: [Int]
        package let byteCount: Int

        package var id: String { name }

        /// "float32 [1, 256, 256]"
        package var summary: String {
            "\(dtype) [\(shape.map(String.init).joined(separator: ", "))]"
        }

        package var elementCount: Int {
            shape.reduce(1, *)
        }
    }

    /// The tensors in the order the header names them.
    package let tensors: [Tensor]
    package let metadata: [String: String]
    package let byteCount: Int

    package init(tensors: [Tensor], metadata: [String: String], byteCount: Int) {
        self.tensors = tensors
        self.metadata = metadata
        self.byteCount = byteCount
    }

    /// The header alone, without reading the tensors behind it (`StudioTensorHeader.load(from:)`):
    /// an Earth tile bundle can be hundreds of megabytes, and the checklist needs only the
    /// names and shapes in front.
    package static func load(from url: URL) -> StudioSafetensorsHeader? {
        guard case .safetensors(let header)? = StudioTensorHeader.load(from: url) else { return nil }
        return header
    }

    package static func decode(_ data: Data) -> StudioSafetensorsHeader? {
        guard data.count >= 8 else { return nil }
        let length = data.prefix(8).enumerated().reduce(UInt64(0)) { total, byte in
            total | UInt64(byte.element) << (8 * UInt64(byte.offset))
        }
        guard length > 0, length <= UInt64(data.count - 8),
              let document = try? JSONDecoder().decode([String: Entry].self, from: data.subdata(in: 8..<(8 + Int(length))))
        else { return nil }
        var tensors: [Tensor] = []
        var metadata: [String: String] = [:]
        for (name, entry) in document {
            switch entry {
            case .tensor(let tensor):
                tensors.append(Tensor(
                    name: name, dtype: tensor.dtype, shape: tensor.shape,
                    byteCount: max(0, tensor.dataOffsets.count == 2 ? tensor.dataOffsets[1] - tensor.dataOffsets[0] : 0)
                ))
            case .metadata(let strings):
                metadata = strings
            }
        }
        // The JSON object has no order; the tensors' byte offsets are the order they were written.
        tensors.sort { lhs, rhs in
            let left = document[lhs.name]?.offset ?? 0
            let right = document[rhs.name]?.offset ?? 0
            return left == right ? lhs.name < rhs.name : left < right
        }
        return StudioSafetensorsHeader(tensors: tensors, metadata: metadata, byteCount: data.count)
    }

    /// One header value: a tensor record, or the `__metadata__` strings.
    private enum Entry: Decodable {
        struct TensorRecord: Decodable {
            let dtype: String
            let shape: [Int]
            let dataOffsets: [Int]

            enum CodingKeys: String, CodingKey {
                case dtype
                case shape
                case dataOffsets = "data_offsets"
            }
        }

        case tensor(TensorRecord)
        case metadata([String: String])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let strings = try? container.decode([String: String].self) {
                self = .metadata(strings)
            } else {
                self = .tensor(try container.decode(TensorRecord.self))
            }
        }

        var offset: Int {
            if case .tensor(let record) = self { return record.dataOffsets.first ?? 0 }
            return 0
        }
    }
}

/// Whichever tensor header a file carries, for the Analyze `.tensor` view.
package enum StudioTensorHeader: Equatable {
    case npy(StudioNPYMetadata)
    case safetensors(StudioSafetensorsHeader)

    /// The extensions of the tensor files Studio runs write; their header is read on its own.
    package static let fileExtensions: Set<String> = ["safetensors", "npy"]

    /// The most header either format can declare that is read from disk; a longer one is not
    /// a tensor file Studio wrote.
    private static let headerReadLimit = 16 * 1_024 * 1_024

    /// Reads the header alone — the magic, the declared header length, then that many bytes —
    /// and reports the whole file's size, so a card or panel never loads the tensor payload.
    package static func load(from url: URL) -> StudioTensorHeader? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let fileSize = try? handle.seekToEnd(), (try? handle.seek(toOffset: 0)) != nil,
              let prefix = try? handle.read(upToCount: 12), prefix.count >= 8 else { return nil }
        let headerLength: Int
        let headerStart: Int
        if Array(prefix.prefix(6)) == [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59] {
            if prefix[6] <= 1 {
                headerLength = Int(prefix[8]) | (Int(prefix[9]) << 8)
                headerStart = 10
            } else {
                guard prefix.count >= 12 else { return nil }
                headerLength = Int(prefix[8]) | (Int(prefix[9]) << 8) | (Int(prefix[10]) << 16) | (Int(prefix[11]) << 24)
                headerStart = 12
            }
        } else {
            let length = prefix.prefix(8).enumerated().reduce(UInt64(0)) { total, byte in
                total | UInt64(byte.element) << (8 * UInt64(byte.offset))
            }
            guard length <= UInt64(headerReadLimit) else { return nil }
            headerLength = Int(length)
            headerStart = 8
        }
        guard headerLength <= headerReadLimit, (try? handle.seek(toOffset: 0)) != nil,
              let data = try? handle.read(upToCount: headerStart + headerLength) else { return nil }
        switch decode(data) {
        case .npy(let npy):
            return .npy(StudioNPYMetadata(
                version: npy.version, descriptor: npy.descriptor, shape: npy.shape,
                fortranOrder: npy.fortranOrder, byteCount: Int(fileSize)
            ))
        case .safetensors(let header):
            return .safetensors(StudioSafetensorsHeader(tensors: header.tensors, metadata: header.metadata, byteCount: Int(fileSize)))
        case nil:
            return nil
        }
    }

    /// `.npy` announces itself with a magic string; anything else is tried as safetensors.
    package static func decode(_ data: Data) -> StudioTensorHeader? {
        if let npy = StudioNPYMetadata.decode(data) { return .npy(npy) }
        if let header = StudioSafetensorsHeader.decode(data) { return .safetensors(header) }
        return nil
    }

    /// "float32 (1, 64, 1875) · 480 KB" or "3 tensors · 12 MB".
    package var summary: String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        switch self {
        case .npy(let npy):
            return "\(npy.descriptor) \(npy.shape) · \(formatter.string(fromByteCount: Int64(npy.byteCount)))"
        case .safetensors(let header):
            let count = header.tensors.count == 1 ? "1 tensor" : "\(header.tensors.count) tensors"
            return "\(count) · \(formatter.string(fromByteCount: Int64(header.byteCount)))"
        }
    }
}
