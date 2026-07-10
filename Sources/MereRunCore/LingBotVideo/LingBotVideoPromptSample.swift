import Foundation

public struct LingBotVideoPromptSample: Sendable, Equatable {
    public enum SampleError: LocalizedError {
        case emptyList
        case invalidRoot
        case invalidCaption
        case invalidDuration

        public var errorDescription: String? {
            switch self {
            case .emptyList:
                return "LingBot prompt JSON contains an empty list."
            case .invalidRoot:
                return "LingBot prompt JSON must contain an object or a non-empty list of objects."
            case .invalidCaption:
                return "LingBot prompt JSON does not contain a usable caption."
            case .invalidDuration:
                return "LingBot prompt JSON duration must be a positive number."
            }
        }
    }

    public let caption: String
    public let duration: Double?

    public init(caption: String, duration: Double? = nil) {
        self.caption = caption
        self.duration = duration
    }

    public static func load(from url: URL) throws -> LingBotVideoPromptSample {
        try decode(Data(contentsOf: url))
    }

    public static func compactJSONDocument(_ data: Data) throws -> String {
        _ = try JSONDecoder().decode(LingBotJSONValue.self, from: data)
        return String(decoding: OrderedJSONObject.minify(data), as: UTF8.self)
    }

    public static func decode(_ data: Data) throws -> LingBotVideoPromptSample {
        let root = try JSONDecoder().decode(LingBotJSONValue.self, from: data)
        let sample: [String: LingBotJSONValue]
        switch root {
        case .object(let object):
            sample = object
        case .array(let values):
            guard let first = values.first else { throw SampleError.emptyList }
            guard case .object(let object) = first else { throw SampleError.invalidRoot }
            sample = object
        default:
            throw SampleError.invalidRoot
        }

        let duration = try decodeDuration(sample["duration"])
        let orderedObject = try OrderedJSONObject(data: data)
        let caption: String
        if let captionMember = orderedObject.members.first(where: { $0.key == "caption" }) {
            caption = try captionString(captionMember.value)
        } else {
            let runtimeKeys: Set<String> = [
                "duration", "fps", "height", "width", "num_frames", "resolution", "ratio",
            ]
            caption = orderedObject.compactObject(excluding: runtimeKeys)
        }
        guard !caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw SampleError.invalidCaption
        }
        return LingBotVideoPromptSample(caption: caption, duration: duration)
    }

    private static func decodeDuration(_ value: LingBotJSONValue?) throws -> Double? {
        guard let value else { return nil }
        let duration: Double?
        switch value {
        case .number(let number):
            duration = number
        case .string(let string):
            duration = Double(string)
        default:
            duration = nil
        }
        guard let duration, duration.isFinite, duration > 0 else {
            throw SampleError.invalidDuration
        }
        return duration
    }

    private static func captionString(_ rawValue: Data) throws -> String {
        if rawValue.first == UInt8(ascii: "\"") {
            return try JSONDecoder().decode(String.self, from: rawValue)
        }
        guard let value = String(data: OrderedJSONObject.minify(rawValue), encoding: .utf8) else {
            throw SampleError.invalidCaption
        }
        return value
    }
}

private indirect enum LingBotJSONValue: Decodable, Sendable {
    case object([String: LingBotJSONValue])
    case array([LingBotJSONValue])
    case string(String)
    case number(Double)
    case boolean(Bool)
    case null

    init(from decoder: Decoder) throws {
        if let keyed = try? decoder.container(keyedBy: LingBotJSONKey.self) {
            var object: [String: LingBotJSONValue] = [:]
            object.reserveCapacity(keyed.allKeys.count)
            for key in keyed.allKeys {
                object[key.stringValue] = try keyed.decode(LingBotJSONValue.self, forKey: key)
            }
            self = .object(object)
            return
        }
        if var unkeyed = try? decoder.unkeyedContainer() {
            var values: [LingBotJSONValue] = []
            while !unkeyed.isAtEnd {
                values.append(try unkeyed.decode(LingBotJSONValue.self))
            }
            self = .array(values)
            return
        }

        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let string = try? container.decode(String.self) {
            self = .string(string)
        } else if let boolean = try? container.decode(Bool.self) {
            self = .boolean(boolean)
        } else if let number = try? container.decode(Double.self) {
            self = .number(number)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported LingBot JSON value."
            )
        }
    }
}

private struct LingBotJSONKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init?(stringValue: String) {
        self.stringValue = stringValue
        self.intValue = nil
    }

    init?(intValue: Int) {
        self.stringValue = String(intValue)
        self.intValue = intValue
    }
}

private struct OrderedJSONObject {
    struct Member {
        let key: String
        let rawKey: Data
        let value: Data
    }

    let members: [Member]

    init(data: Data) throws {
        let bytes = [UInt8](data)
        var index = Self.skipWhitespace(in: bytes, from: 0)
        if index < bytes.count, bytes[index] == UInt8(ascii: "[") {
            index = Self.skipWhitespace(in: bytes, from: index + 1)
        }
        let rootRange = try Self.valueRange(in: bytes, from: index)
        guard bytes[rootRange.lowerBound] == UInt8(ascii: "{") else {
            throw LingBotVideoPromptSample.SampleError.invalidRoot
        }
        self.members = try Self.parseMembers(in: bytes, range: rootRange)
    }

    func compactObject(excluding excludedKeys: Set<String>) -> String {
        let included = members.filter { !excludedKeys.contains($0.key) }
        var data = Data([UInt8(ascii: "{")])
        for (index, member) in included.enumerated() {
            if index > 0 { data.append(UInt8(ascii: ",")) }
            data.append(Self.minify(member.rawKey))
            data.append(UInt8(ascii: ":"))
            data.append(Self.minify(member.value))
        }
        data.append(UInt8(ascii: "}"))
        return String(decoding: data, as: UTF8.self)
    }

    static func minify(_ data: Data) -> Data {
        var output = Data()
        output.reserveCapacity(data.count)
        var inString = false
        var escaped = false
        for byte in data {
            if inString {
                output.append(byte)
                if escaped {
                    escaped = false
                } else if byte == UInt8(ascii: "\\") {
                    escaped = true
                } else if byte == UInt8(ascii: "\"") {
                    inString = false
                }
            } else if byte == UInt8(ascii: "\"") {
                inString = true
                output.append(byte)
            } else if !isWhitespace(byte) {
                output.append(byte)
            }
        }
        return output
    }

    private static func parseMembers(in bytes: [UInt8], range: Range<Int>) throws -> [Member] {
        var members: [Member] = []
        var index = skipWhitespace(in: bytes, from: range.lowerBound + 1)
        while index < range.upperBound, bytes[index] != UInt8(ascii: "}") {
            let keyRange = try stringRange(in: bytes, from: index)
            let rawKey = Data(bytes[keyRange])
            let key = try JSONDecoder().decode(String.self, from: rawKey)
            index = skipWhitespace(in: bytes, from: keyRange.upperBound)
            guard index < range.upperBound, bytes[index] == UInt8(ascii: ":") else {
                throw LingBotVideoPromptSample.SampleError.invalidRoot
            }
            index = skipWhitespace(in: bytes, from: index + 1)
            let memberValueRange = try valueRange(in: bytes, from: index)
            members.append(Member(
                key: key,
                rawKey: rawKey,
                value: Data(bytes[memberValueRange])
            ))
            index = skipWhitespace(in: bytes, from: memberValueRange.upperBound)
            if index < range.upperBound, bytes[index] == UInt8(ascii: ",") {
                index = skipWhitespace(in: bytes, from: index + 1)
            }
        }
        return members
    }

    private static func valueRange(in bytes: [UInt8], from start: Int) throws -> Range<Int> {
        guard start < bytes.count else {
            throw LingBotVideoPromptSample.SampleError.invalidRoot
        }
        if bytes[start] == UInt8(ascii: "\"") {
            return try stringRange(in: bytes, from: start)
        }
        if bytes[start] == UInt8(ascii: "{") || bytes[start] == UInt8(ascii: "[") {
            var stack: [UInt8] = [bytes[start]]
            var inString = false
            var escaped = false
            var index = start + 1
            while index < bytes.count {
                let byte = bytes[index]
                if inString {
                    if escaped {
                        escaped = false
                    } else if byte == UInt8(ascii: "\\") {
                        escaped = true
                    } else if byte == UInt8(ascii: "\"") {
                        inString = false
                    }
                } else if byte == UInt8(ascii: "\"") {
                    inString = true
                } else if byte == UInt8(ascii: "{") || byte == UInt8(ascii: "[") {
                    stack.append(byte)
                } else if byte == UInt8(ascii: "}") || byte == UInt8(ascii: "]") {
                    guard let opening = stack.popLast(),
                          (opening == UInt8(ascii: "{") && byte == UInt8(ascii: "}"))
                            || (opening == UInt8(ascii: "[") && byte == UInt8(ascii: "]"))
                    else {
                        throw LingBotVideoPromptSample.SampleError.invalidRoot
                    }
                    if stack.isEmpty { return start..<(index + 1) }
                }
                index += 1
            }
            throw LingBotVideoPromptSample.SampleError.invalidRoot
        }

        var end = start
        while end < bytes.count,
              bytes[end] != UInt8(ascii: ","),
              bytes[end] != UInt8(ascii: "}"),
              bytes[end] != UInt8(ascii: "]") {
            end += 1
        }
        while end > start, isWhitespace(bytes[end - 1]) { end -= 1 }
        return start..<end
    }

    private static func stringRange(in bytes: [UInt8], from start: Int) throws -> Range<Int> {
        guard start < bytes.count, bytes[start] == UInt8(ascii: "\"") else {
            throw LingBotVideoPromptSample.SampleError.invalidRoot
        }
        var escaped = false
        var index = start + 1
        while index < bytes.count {
            let byte = bytes[index]
            if escaped {
                escaped = false
            } else if byte == UInt8(ascii: "\\") {
                escaped = true
            } else if byte == UInt8(ascii: "\"") {
                return start..<(index + 1)
            }
            index += 1
        }
        throw LingBotVideoPromptSample.SampleError.invalidRoot
    }

    private static func skipWhitespace(in bytes: [UInt8], from start: Int) -> Int {
        var index = start
        while index < bytes.count, isWhitespace(bytes[index]) { index += 1 }
        return index
    }

    private static func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D
    }
}
