import Foundation

public struct GLM47ToolCall: Sendable, Equatable {
    public let name: String
    public let arguments: [String: GLM47ToolValue]

    public init(name: String, arguments: [String: GLM47ToolValue]) {
        self.name = name
        self.arguments = arguments
    }
}

public indirect enum GLM47ToolValue: Codable, Equatable, Sendable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case array([GLM47ToolValue])
    case object([String: GLM47ToolValue])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let bool = try? container.decode(Bool.self) {
            self = .bool(bool)
        } else if let int = try? container.decode(Int.self) {
            self = .int(int)
        } else if let double = try? container.decode(Double.self) {
            self = .double(double)
        } else if let array = try? container.decode([GLM47ToolValue].self) {
            self = .array(array)
        } else if let object = try? container.decode([String: GLM47ToolValue].self) {
            self = .object(object)
        } else if let string = try? container.decode(String.self) {
            self = .string(string)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported tool argument value."
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .int(let value):
            try container.encode(value)
        case .double(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }
}

public enum GLM47ToolParser {
    private static let toolCallStart = "<tool_call>"
    private static let toolCallEnd = "</tool_call>"

    public static func parseToolCalls(_ text: String) -> [GLM47ToolCall] {
        var results: [GLM47ToolCall] = []
        var remainder = text

        while let start = remainder.range(of: toolCallStart),
              let end = remainder.range(of: toolCallEnd, range: start.upperBound..<remainder.endIndex) {
            let payload = String(remainder[start.upperBound..<end.lowerBound])
            if let call = parseSingleToolCall(payload) {
                results.append(call)
            }
            remainder = String(remainder[end.upperBound...])
        }

        return results
    }

    private static func parseSingleToolCall(_ payload: String) -> GLM47ToolCall? {
        let nameSplit = payload.components(separatedBy: "<arg_key>")
        guard let name = nameSplit.first?.trimmingCharacters(in: .whitespacesAndNewlines),
              !name.isEmpty else {
            return nil
        }

        var args: [String: GLM47ToolValue] = [:]
        let pattern = "<arg_key>(.*?)</arg_key>\\s*<arg_value>(.*?)</arg_value>"
        let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators])
        let range = NSRange(payload.startIndex..<payload.endIndex, in: payload)
        let matches = regex?.matches(in: payload, options: [], range: range) ?? []

        for match in matches {
            guard match.numberOfRanges == 3,
                  let keyRange = Range(match.range(at: 1), in: payload),
                  let valueRange = Range(match.range(at: 2), in: payload) else {
                continue
            }
            let key = payload[keyRange].trimmingCharacters(in: .whitespacesAndNewlines)
            let value = payload[valueRange].trimmingCharacters(in: .whitespacesAndNewlines)
            args[key] = parseValue(String(value))
        }

        return GLM47ToolCall(name: name, arguments: args)
    }

    private static func parseValue(_ value: String) -> GLM47ToolValue {
        if let data = value.data(using: .utf8) {
            if let json = try? JSONDecoder().decode(GLM47ToolValue.self, from: data) {
                return json
            }
        }

        if let intValue = Int(value) { return .int(intValue) }
        if let doubleValue = Double(value) { return .double(doubleValue) }
        if value == "true" { return .bool(true) }
        if value == "false" { return .bool(false) }
        return .string(value)
    }
}
