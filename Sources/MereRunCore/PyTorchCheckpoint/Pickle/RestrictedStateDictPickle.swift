import Foundation

struct ParsedStorage: Equatable {
    let key: String
    let dataType: PyTorchTensorDataType
    let elementCount: Int
}

struct ParsedTensor {
    let name: String
    let storage: ParsedStorage
    let storageOffset: Int
    let shape: [Int]
    let stride: [Int]
    let requiresGradient: Bool
}

private indirect enum PickleValue {
    case boolean(Bool)
    case dictionary([(String, PickleValue)])
    case global(PickleGlobal)
    case integer(Int)
    case marker
    case storage(ParsedStorage)
    case string(String)
    case tensor(ParsedTensorValue)
    case tuple([PickleValue])
}

private enum PickleGlobal {
    case orderedDictionary
    case rebuildTensorV2
    case storage(PyTorchTensorDataType)
}

private struct ParsedTensorValue {
    let storage: ParsedStorage
    let storageOffset: Int
    let shape: [Int]
    let stride: [Int]
    let requiresGradient: Bool
}

struct RestrictedStateDictPickle {
    let data: Data
    private var index = 0
    private var stack: [PickleValue] = []
    private var memo: [Int: PickleValue] = [:]

    init(data: Data) { self.data = data }

    mutating func parse() throws -> [ParsedTensor] {
        guard try readByte() == 0x80 else { throw PyTorchStateDictError.unsupportedPickleProtocol(-1) }
        let protocolVersion = Int(try readByte())
        guard protocolVersion == 2 else {
            throw PyTorchStateDictError.unsupportedPickleProtocol(protocolVersion)
        }
        while index < data.count {
            let opcodeOffset = index
            let opcode = try readByte()
            switch opcode {
            case 0x7D: try push(.dictionary([])) // EMPTY_DICT
            case 0x29: try push(.tuple([])) // EMPTY_TUPLE
            case 0x28: try push(.marker) // MARK
            case 0x58: try push(.string(readUnicode())) // BINUNICODE
            case 0x4A: try push(.integer(Int(Int32(bitPattern: readUInt32())))) // BININT
            case 0x4B: try push(.integer(Int(readByte()))) // BININT1
            case 0x4D: try push(.integer(Int(readUInt16()))) // BININT2
            case 0x89: try push(.boolean(false)) // NEWFALSE
            case 0x63: try push(.global(readGlobal())) // GLOBAL
            case 0x71: try memoize(Int(readByte())) // BINPUT
            case 0x72: try memoize(Int(readUInt32())) // LONG_BINPUT
            case 0x68: try push(memoized(Int(readByte()))) // BINGET
            case 0x74: try buildTuple() // TUPLE
            case 0x85: try buildFixedTuple(1) // TUPLE1
            case 0x86: try buildFixedTuple(2) // TUPLE2
            case 0x87: try buildFixedTuple(3) // TUPLE3
            case 0x51: try persistentStorage() // BINPERSID
            case 0x52: try reduce() // REDUCE
            case 0x75: try setItems() // SETITEMS
            case 0x2E: return try finish() // STOP
            default: throw PyTorchStateDictError.unsupportedPickleOpcode(opcode, offset: opcodeOffset)
            }
        }
        throw PyTorchStateDictError.malformedPickle("missing STOP opcode")
    }

    private mutating func readGlobal() throws -> PickleGlobal {
        let module = try readLine()
        let name = try readLine()
        switch (module, name) {
        case ("torch._utils", "_rebuild_tensor_v2"): return .rebuildTensorV2
        case ("collections", "OrderedDict"): return .orderedDictionary
        case ("torch", let storageName):
            let types: [String: PyTorchTensorDataType] = [
                "BFloat16Storage": .bfloat16, "BoolStorage": .bool, "ByteStorage": .uint8,
                "CharStorage": .int8, "FloatStorage": .float32, "HalfStorage": .float16,
                "IntStorage": .int32, "LongStorage": .int64, "ShortStorage": .int16,
            ]
            guard let type = types[storageName] else {
                throw PyTorchStateDictError.unsupportedStorageType(storageName)
            }
            return .storage(type)
        default: throw PyTorchStateDictError.unsupportedPickleGlobal(module: module, name: name)
        }
    }

    private mutating func persistentStorage() throws {
        guard case .tuple(let fields) = try pop(), fields.count == 5,
              case .string("storage") = fields[0], case .global(.storage(let type)) = fields[1],
              case .string(let key) = fields[2], case .string("cpu") = fields[3],
              case .integer(let count) = fields[4], count >= 0,
              isASCIIDecimal(key) else {
            throw PyTorchStateDictError.malformedPickle("invalid persistent storage identifier")
        }
        try push(.storage(ParsedStorage(key: key, dataType: type, elementCount: count)))
    }

    private mutating func reduce() throws {
        guard case .tuple(let arguments) = try pop(), case .global(let callable) = try pop() else {
            throw PyTorchStateDictError.malformedPickle("REDUCE requires a whitelisted global and tuple")
        }
        switch callable {
        case .orderedDictionary:
            guard arguments.isEmpty else {
                throw PyTorchStateDictError.malformedPickle("OrderedDict constructor received arguments")
            }
            try push(.dictionary([]))
        case .rebuildTensorV2:
            guard arguments.count == 6, case .storage(let storage) = arguments[0],
                  case .integer(let offset) = arguments[1],
                  let shape = integerTuple(arguments[2]), let stride = integerTuple(arguments[3]),
                  case .boolean(let requiresGradient) = arguments[4],
                  case .dictionary(let hooks) = arguments[5], hooks.isEmpty else {
                throw PyTorchStateDictError.malformedPickle("invalid _rebuild_tensor_v2 arguments")
            }
            try push(.tensor(ParsedTensorValue(
                storage: storage,
                storageOffset: offset,
                shape: shape,
                stride: stride,
                requiresGradient: requiresGradient
            )))
        case .storage:
            throw PyTorchStateDictError.malformedPickle("storage globals cannot be reduced")
        }
    }

    private mutating func setItems() throws {
        guard let marker = stack.lastIndex(where: { if case .marker = $0 { true } else { false } }), marker > 0,
              case .dictionary(var dictionary) = stack[marker - 1] else {
            throw PyTorchStateDictError.malformedPickle("SETITEMS is missing its dictionary marker")
        }
        let items = Array(stack[(marker + 1)...])
        guard items.count.isMultiple(of: 2) else {
            throw PyTorchStateDictError.malformedPickle("SETITEMS contains an odd number of values")
        }
        var names = Set(dictionary.map(\.0))
        for pair in stride(from: 0, to: items.count, by: 2) {
            guard case .string(let key) = items[pair], names.insert(key).inserted else {
                throw PyTorchStateDictError.malformedPickle("state dictionary contains an invalid or duplicate key")
            }
            dictionary.append((key, items[pair + 1]))
        }
        stack.removeSubrange((marker - 1)...)
        try push(.dictionary(dictionary))
    }

    private mutating func finish() throws -> [ParsedTensor] {
        guard index == data.count, stack.count == 1, case .dictionary(let dictionary) = stack[0] else {
            throw PyTorchStateDictError.malformedPickle("STOP did not terminate one state dictionary")
        }
        return try dictionary.map { name, value in
            guard case .tensor(let tensor) = value else {
                throw PyTorchStateDictError.malformedPickle("state-dict value '\(name)' is not a tensor")
            }
            return ParsedTensor(
                name: name,
                storage: tensor.storage,
                storageOffset: tensor.storageOffset,
                shape: tensor.shape,
                stride: tensor.stride,
                requiresGradient: tensor.requiresGradient
            )
        }
    }

    private mutating func buildTuple() throws {
        guard let marker = stack.lastIndex(where: { if case .marker = $0 { true } else { false } }) else {
            throw PyTorchStateDictError.malformedPickle("TUPLE has no marker")
        }
        let values = Array(stack[(marker + 1)...])
        stack.removeSubrange(marker...)
        try push(.tuple(values))
    }

    private mutating func buildFixedTuple(_ count: Int) throws {
        guard stack.count >= count else { throw PyTorchStateDictError.malformedPickle("truncated fixed tuple") }
        let values = Array(stack.suffix(count))
        stack.removeLast(count)
        try push(.tuple(values))
    }

    private func integerTuple(_ value: PickleValue) -> [Int]? {
        guard case .tuple(let values) = value else { return nil }
        return values.compactMap { if case .integer(let integer) = $0 { integer } else { nil } }.count == values.count
            ? values.compactMap { if case .integer(let integer) = $0 { integer } else { nil } }
            : nil
    }

    private mutating func memoize(_ memoIndex: Int) throws {
        guard memoIndex <= 1_000_000, memo[memoIndex] == nil, let value = stack.last else {
            throw PyTorchStateDictError.malformedPickle("invalid memo write")
        }
        memo[memoIndex] = value
    }

    private func memoized(_ memoIndex: Int) throws -> PickleValue {
        guard let value = memo[memoIndex] else {
            throw PyTorchStateDictError.malformedPickle("unknown memo index \(memoIndex)")
        }
        return value
    }

    private mutating func push(_ value: PickleValue) throws {
        guard stack.count < 100_000 else { throw PyTorchStateDictError.malformedPickle("stack limit exceeded") }
        stack.append(value)
    }

    private mutating func pop() throws -> PickleValue {
        guard let value = stack.popLast() else { throw PyTorchStateDictError.malformedPickle("stack underflow") }
        return value
    }

    private mutating func readUnicode() throws -> String {
        let length = Int(try readUInt32())
        guard length <= 1_048_576, index <= data.count - length else {
            throw PyTorchStateDictError.malformedPickle("invalid BINUNICODE length")
        }
        let bytes = data.subdata(in: index..<(index + length))
        index += length
        guard let value = String(data: bytes, encoding: .utf8), !value.contains("\0") else {
            throw PyTorchStateDictError.malformedPickle("invalid BINUNICODE text")
        }
        return value
    }

    private mutating func readLine() throws -> String {
        let start = index
        while index < data.count, data[index] != 0x0A, index - start <= 256 { index += 1 }
        guard index < data.count, data[index] == 0x0A else {
            throw PyTorchStateDictError.malformedPickle("unterminated GLOBAL name")
        }
        let bytes = data.subdata(in: start..<index)
        index += 1
        guard let value = String(data: bytes, encoding: .ascii), !value.isEmpty else {
            throw PyTorchStateDictError.malformedPickle("invalid GLOBAL name")
        }
        return value
    }

    private mutating func readByte() throws -> UInt8 {
        guard index < data.count else { throw PyTorchStateDictError.malformedPickle("unexpected end of data") }
        defer { index += 1 }
        return data[index]
    }

    private mutating func readUInt16() throws -> UInt16 {
        let low = UInt16(try readByte())
        return low | (UInt16(try readByte()) << 8)
    }

    private mutating func readUInt32() throws -> UInt32 {
        let low = UInt32(try readUInt16())
        return low | (UInt32(try readUInt16()) << 16)
    }

    private func isASCIIDecimal(_ value: String) -> Bool {
        !value.isEmpty && value.utf8.allSatisfy { (0x30...0x39).contains($0) }
    }
}
