import Foundation

// A safetensors file read for its values, not just its header (`StudioSafetensorsHeader` names
// what a file holds without reading it). The format is an 8-byte little-endian header length,
// that many bytes of JSON naming each tensor's dtype, shape, and byte range, then the raw
// little-endian buffers back to back. The Earth map reads the logits and embeddings the `geo`
// commands write (always F32) and the bands of the bundle they read (whatever its writer chose),
// so every element type a writer is likely to use decodes to `Float`.

/// The element types a safetensors header declares, spelled as the header spells them.
package enum StudioTensorDType: String, Equatable, Sendable, CaseIterable {
    case float64 = "F64"
    case float32 = "F32"
    case float16 = "F16"
    case bfloat16 = "BF16"
    case int64 = "I64"
    case int32 = "I32"
    case int16 = "I16"
    case int8 = "I8"
    case uint16 = "U16"
    case uint8 = "U8"

    package var byteWidth: Int {
        switch self {
        case .float64, .int64: return 8
        case .float32, .int32: return 4
        case .float16, .bfloat16, .int16, .uint16: return 2
        case .int8, .uint8: return 1
        }
    }
}

/// Why a safetensors file couldn't be read. `reason` is one sentence for the UI to put after
/// its own "Couldn't …" sentence.
package enum StudioSafetensorsError: Error, Equatable, Sendable {
    /// The file couldn't be opened.
    case unreadable(String)
    /// The bytes don't start with a safetensors header.
    case notSafetensors(String)
    /// The header names no tensor by this name.
    case missingTensor(String)
    /// The tensor's element type isn't one `StudioTensorDType` decodes.
    case unsupportedDType(name: String, dtype: String)
    /// The tensor's byte range disagrees with its shape and dtype, or runs past the file.
    case malformedTensor(String)
    /// A slice asked for elements the tensor doesn't have.
    case outOfRange(String)

    package var reason: String {
        switch self {
        case .unreadable(let file): return "\(file) can't be opened."
        case .notSafetensors(let file): return "\(file) isn't a safetensors file."
        case .missingTensor(let name): return "The file has no \(name) tensor."
        case .unsupportedDType(let name, let dtype): return "\(name) is \(dtype), which Studio can't decode."
        case .malformedTensor(let name): return "\(name)'s data doesn't match its shape."
        case .outOfRange(let name): return "\(name) has fewer elements than the slice asked for."
        }
    }
}

/// Decoded values and the shape they came with, row-major.
package struct StudioTensorValues: Equatable, Sendable {
    package let shape: [Int]
    package let values: [Float]

    package init(shape: [Int], values: [Float]) {
        self.shape = shape
        self.values = values
    }
}

/// A safetensors file opened for reading. The file is memory-mapped, so opening a bundle of
/// hundreds of megabytes costs only its header; values are decoded on request, whole or as a
/// contiguous run of elements (one tile, one band).
package struct StudioSafetensorsFile: @unchecked Sendable {
    /// One tensor the header declares.
    package struct Entry: Equatable, Sendable {
        package let name: String
        /// The dtype as the header spells it, whether or not it decodes.
        package let dtypeName: String
        package let shape: [Int]
        /// Where the tensor's bytes are, relative to the end of the header.
        package let byteRange: Range<Int>

        package var dtype: StudioTensorDType? { StudioTensorDType(rawValue: dtypeName) }
        package var elementCount: Int { shape.reduce(1, *) }
    }

    /// The tensors, in the order their bytes are written.
    package let entries: [Entry]
    package let metadata: [String: String]
    // `Data` is a value type whose mapped storage is never written; sharing it across tasks is safe.
    private let data: Data
    private let dataStart: Int

    package init(url: URL) throws {
        let data: Data
        do {
            data = try Data(contentsOf: url, options: .alwaysMapped)
        } catch {
            throw StudioSafetensorsError.unreadable(url.lastPathComponent)
        }
        try self.init(data: data, fileName: url.lastPathComponent)
    }

    package init(data: Data, fileName: String = "The file") throws {
        guard data.count >= 8 else { throw StudioSafetensorsError.notSafetensors(fileName) }
        let start = data.startIndex
        let length = (0..<8).reduce(UInt64(0)) { total, offset in
            total | UInt64(data[start + offset]) << (8 * UInt64(offset))
        }
        guard length > 0, length <= UInt64(data.count - 8),
              let header = try? JSONDecoder().decode(
                  [String: HeaderValue].self, from: data.subdata(in: (start + 8)..<(start + 8 + Int(length)))
              ) else {
            throw StudioSafetensorsError.notSafetensors(fileName)
        }
        var entries: [Entry] = []
        var metadata: [String: String] = [:]
        for (name, value) in header {
            switch value {
            case .metadata(let strings):
                metadata = strings
            case .tensor(let record):
                guard record.dataOffsets.count == 2, record.dataOffsets[0] >= 0,
                      record.dataOffsets[0] <= record.dataOffsets[1] else {
                    throw StudioSafetensorsError.malformedTensor(name)
                }
                entries.append(Entry(
                    name: name, dtypeName: record.dtype, shape: record.shape,
                    byteRange: record.dataOffsets[0]..<record.dataOffsets[1]
                ))
            }
        }
        entries.sort { $0.byteRange.lowerBound == $1.byteRange.lowerBound ? $0.name < $1.name : $0.byteRange.lowerBound < $1.byteRange.lowerBound }
        self.entries = entries
        self.metadata = metadata
        self.data = data
        self.dataStart = 8 + Int(length)
    }

    package func entry(_ name: String) throws -> Entry {
        guard let entry = entries.first(where: { $0.name == name }) else {
            throw StudioSafetensorsError.missingTensor(name)
        }
        return entry
    }

    package func contains(_ name: String) -> Bool {
        entries.contains { $0.name == name }
    }

    /// Every element of `name`, decoded to `Float`.
    package func values(_ name: String) throws -> StudioTensorValues {
        let entry = try entry(name)
        return StudioTensorValues(shape: entry.shape, values: try values(name, elements: 0..<entry.elementCount))
    }

    /// A contiguous run of `name`'s elements in row-major order, decoded to `Float`.
    package func values(_ name: String, elements: Range<Int>) throws -> [Float] {
        let entry = try entry(name)
        guard let dtype = entry.dtype else {
            throw StudioSafetensorsError.unsupportedDType(name: name, dtype: entry.dtypeName)
        }
        guard entry.shape.allSatisfy({ $0 >= 0 }),
              entry.byteRange.count == entry.elementCount * dtype.byteWidth,
              dataStart + entry.byteRange.upperBound <= data.count else {
            throw StudioSafetensorsError.malformedTensor(name)
        }
        guard elements.lowerBound >= 0, elements.upperBound <= entry.elementCount else {
            throw StudioSafetensorsError.outOfRange(name)
        }
        let first = data.startIndex + dataStart + entry.byteRange.lowerBound + elements.lowerBound * dtype.byteWidth
        let bytes = first..<(first + elements.count * dtype.byteWidth)
        return data[bytes].withUnsafeBytes { Self.decode($0, dtype: dtype, count: elements.count) }
    }

    /// Little-endian elements to `Float`. Apple silicon is little-endian, so F32 is a copy.
    static func decode(_ bytes: UnsafeRawBufferPointer, dtype: StudioTensorDType, count: Int) -> [Float] {
        [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
            switch dtype {
            case .float32:
                if count > 0, let base = buffer.baseAddress, let source = bytes.baseAddress {
                    memcpy(base, source, count * 4)
                }
            case .float64:
                for index in 0..<count {
                    buffer[index] = Float(Double(bitPattern: UInt64(littleEndian: bytes.loadUnaligned(fromByteOffset: index * 8, as: UInt64.self))))
                }
            case .float16:
                for index in 0..<count {
                    buffer[index] = halfToFloat(UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)))
                }
            case .bfloat16:
                for index in 0..<count {
                    let bits = UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self))
                    buffer[index] = Float(bitPattern: UInt32(bits) << 16)
                }
            case .int64:
                for index in 0..<count {
                    buffer[index] = Float(Int64(littleEndian: bytes.loadUnaligned(fromByteOffset: index * 8, as: Int64.self)))
                }
            case .int32:
                for index in 0..<count {
                    buffer[index] = Float(Int32(littleEndian: bytes.loadUnaligned(fromByteOffset: index * 4, as: Int32.self)))
                }
            case .int16:
                for index in 0..<count {
                    buffer[index] = Float(Int16(littleEndian: bytes.loadUnaligned(fromByteOffset: index * 2, as: Int16.self)))
                }
            case .uint16:
                for index in 0..<count {
                    buffer[index] = Float(UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)))
                }
            case .int8:
                for index in 0..<count {
                    buffer[index] = Float(Int8(bitPattern: bytes[index]))
                }
            case .uint8:
                for index in 0..<count {
                    buffer[index] = Float(bytes[index])
                }
            }
            initialized = count
        }
    }

    /// IEEE 754 half precision to single, by its bits, so the reader doesn't depend on `Float16`.
    static func halfToFloat(_ half: UInt16) -> Float {
        let sign = UInt32(half & 0x8000) << 16
        let exponent = UInt32((half >> 10) & 0x1F)
        let mantissa = UInt32(half & 0x03FF)
        switch exponent {
        case 0:
            // Zero or subnormal: mantissa × 2^-24.
            let magnitude = Float(mantissa) * 0x1p-24
            return sign == 0 ? magnitude : -magnitude
        case 0x1F:
            return Float(bitPattern: sign | 0x7F80_0000 | (mantissa << 13))
        default:
            return Float(bitPattern: sign | ((exponent + 112) << 23) | (mantissa << 13))
        }
    }

    /// One value of the header: a tensor record, or the `__metadata__` strings.
    private enum HeaderValue: Decodable {
        struct Record: Decodable {
            let dtype: String
            let shape: [Int]
            let dataOffsets: [Int]

            enum CodingKeys: String, CodingKey {
                case dtype
                case shape
                case dataOffsets = "data_offsets"
            }
        }

        case tensor(Record)
        case metadata([String: String])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let strings = try? container.decode([String: String].self) {
                self = .metadata(strings)
            } else {
                self = .tensor(try container.decode(Record.self))
            }
        }
    }
}
