import Foundation

public enum ONNXTensorDataType: Int, Codable, Equatable, Sendable {
    case float32 = 1
    case uint8 = 2
    case int8 = 3
    case uint16 = 4
    case int16 = 5
    case int32 = 6
    case int64 = 7
    case bool = 9
    case float16 = 10
    case float64 = 11
    case uint32 = 12
    case uint64 = 13
    case bfloat16 = 16

    public var byteCount: Int {
        switch self {
        case .uint8, .int8, .bool: 1
        case .uint16, .int16, .float16, .bfloat16: 2
        case .float32, .int32, .uint32: 4
        case .int64, .float64, .uint64: 8
        }
    }
}

public struct ONNXTensorDescriptor: Equatable, Sendable {
    public let name: String
    public let shape: [Int]
    public let dataType: ONNXTensorDataType
    fileprivate let rawDataRange: Range<Int>

    public var elementCount: Int {
        shape.reduce(1, *)
    }

    public var byteCount: Int {
        elementCount * dataType.byteCount
    }
}

public enum ONNXInitializerError: Error, Equatable, LocalizedError, Sendable {
    case malformed(String)
    case missingGraph
    case missingTensorName
    case unsupportedTensorType(Int)
    case missingRawTensorData(String)
    case duplicateTensorName(String)
    case tensorByteCountMismatch(name: String, expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .malformed(let detail): "Malformed ONNX protobuf: \(detail)"
        case .missingGraph: "The ONNX model does not contain a graph."
        case .missingTensorName: "An ONNX graph initializer has no name."
        case .unsupportedTensorType(let value): "Unsupported ONNX tensor data type \(value)."
        case .missingRawTensorData(let name): "ONNX initializer '\(name)' does not use embedded raw_data."
        case .duplicateTensorName(let name): "ONNX initializer name '\(name)' occurs more than once."
        case .tensorByteCountMismatch(let name, let expected, let actual):
            "ONNX initializer '\(name)' expected \(expected) raw bytes but contains \(actual)."
        }
    }
}

/// Memory-mapped reader for embedded ONNX graph initializers.
///
/// This is deliberately not an ONNX inference runtime. It reads authoritative
/// tensor payloads into native MLX modules while the forward graph stays owned
/// and tested by mere.run.
public final class ONNXInitializerArchive: @unchecked Sendable {
    private let data: Data
    public let tensors: [ONNXTensorDescriptor]
    private let tensorByName: [String: ONNXTensorDescriptor]

    public init(url: URL) throws {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        let tensors = try Self.parseModel(data)
        var lookup: [String: ONNXTensorDescriptor] = [:]
        for tensor in tensors {
            guard lookup[tensor.name] == nil else {
                throw ONNXInitializerError.duplicateTensorName(tensor.name)
            }
            lookup[tensor.name] = tensor
        }
        self.data = data
        self.tensors = tensors
        self.tensorByName = lookup
    }

    public func descriptor(named name: String) -> ONNXTensorDescriptor? {
        tensorByName[name]
    }

    public func rawData(for tensor: ONNXTensorDescriptor) -> Data {
        data.subdata(in: tensor.rawDataRange)
    }

    public func rawData(named name: String) -> Data? {
        tensorByName[name].map(rawData(for:))
    }

    private static func parseModel(_ data: Data) throws -> [ONNXTensorDescriptor] {
        var cursor = ProtobufCursor(data: data, range: 0..<data.count)
        var graphRange: Range<Int>?
        while !cursor.isAtEnd {
            let key = try cursor.readVarint()
            let field = Int(key >> 3)
            let wire = Int(key & 7)
            if field == 7, wire == 2 {
                graphRange = try cursor.readLengthDelimitedRange()
            } else {
                try cursor.skip(wireType: wire)
            }
        }
        guard let graphRange else { throw ONNXInitializerError.missingGraph }
        return try parseGraph(data, range: graphRange)
    }

    private static func parseGraph(_ data: Data, range: Range<Int>) throws -> [ONNXTensorDescriptor] {
        var cursor = ProtobufCursor(data: data, range: range)
        var tensors: [ONNXTensorDescriptor] = []
        while !cursor.isAtEnd {
            let key = try cursor.readVarint()
            let field = Int(key >> 3)
            let wire = Int(key & 7)
            if field == 5, wire == 2 {
                tensors.append(try parseTensor(data, range: cursor.readLengthDelimitedRange()))
            } else {
                try cursor.skip(wireType: wire)
            }
        }
        return tensors
    }

    private static func parseTensor(_ data: Data, range: Range<Int>) throws -> ONNXTensorDescriptor {
        var cursor = ProtobufCursor(data: data, range: range)
        var shape: [Int] = []
        var typeValue: Int?
        var name: String?
        var rawDataRange: Range<Int>?
        while !cursor.isAtEnd {
            let key = try cursor.readVarint()
            let field = Int(key >> 3)
            let wire = Int(key & 7)
            switch (field, wire) {
            case (1, 0):
                shape.append(Int(try cursor.readVarint()))
            case (1, 2):
                let packedRange = try cursor.readLengthDelimitedRange()
                var packed = ProtobufCursor(data: data, range: packedRange)
                while !packed.isAtEnd { shape.append(Int(try packed.readVarint())) }
            case (2, 0):
                typeValue = Int(try cursor.readVarint())
            case (8, 2):
                let valueRange = try cursor.readLengthDelimitedRange()
                name = String(data: data.subdata(in: valueRange), encoding: .utf8)
            case (9, 2):
                rawDataRange = try cursor.readLengthDelimitedRange()
            default:
                try cursor.skip(wireType: wire)
            }
        }
        guard let name, !name.isEmpty else { throw ONNXInitializerError.missingTensorName }
        guard let typeValue, let dataType = ONNXTensorDataType(rawValue: typeValue) else {
            throw ONNXInitializerError.unsupportedTensorType(typeValue ?? -1)
        }
        guard let rawDataRange else { throw ONNXInitializerError.missingRawTensorData(name) }
        let tensor = ONNXTensorDescriptor(name: name, shape: shape, dataType: dataType, rawDataRange: rawDataRange)
        guard rawDataRange.count == tensor.byteCount else {
            throw ONNXInitializerError.tensorByteCountMismatch(
                name: name,
                expected: tensor.byteCount,
                actual: rawDataRange.count
            )
        }
        return tensor
    }
}

private struct ProtobufCursor {
    let data: Data
    let range: Range<Int>
    var index: Int

    init(data: Data, range: Range<Int>) {
        self.data = data
        self.range = range
        self.index = range.lowerBound
    }

    var isAtEnd: Bool { index >= range.upperBound }

    mutating func readVarint() throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        for _ in 0..<10 {
            guard index < range.upperBound else { throw ONNXInitializerError.malformed("truncated varint") }
            let byte = data[index]
            index += 1
            result |= UInt64(byte & 0x7F) << shift
            if byte & 0x80 == 0 { return result }
            shift += 7
        }
        throw ONNXInitializerError.malformed("varint exceeds 10 bytes")
    }

    mutating func readLengthDelimitedRange() throws -> Range<Int> {
        let length = Int(try readVarint())
        guard length >= 0, index <= range.upperBound - length else {
            throw ONNXInitializerError.malformed("length-delimited field exceeds its parent message")
        }
        let valueRange = index..<(index + length)
        index += length
        return valueRange
    }

    mutating func skip(wireType: Int) throws {
        switch wireType {
        case 0:
            _ = try readVarint()
        case 1:
            try advance(8)
        case 2:
            _ = try readLengthDelimitedRange()
        case 5:
            try advance(4)
        default:
            throw ONNXInitializerError.malformed("unsupported protobuf wire type \(wireType)")
        }
    }

    private mutating func advance(_ count: Int) throws {
        guard count >= 0, index <= range.upperBound - count else {
            throw ONNXInitializerError.malformed("fixed-width field exceeds its parent message")
        }
        index += count
    }
}
