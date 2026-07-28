import Foundation

enum LagunaToolParser {
    static func parseToolCalls(_ text: String) -> [ToolCall] {
        GLM47ToolParser.parseToolCalls(text).map { call in
            ToolCall(
                name: call.name,
                arguments: call.arguments.mapValues(argumentString)
            )
        }
    }

    private static func argumentString(_ value: GLM47ToolValue) -> String {
        switch value {
        case .string(let value):
            return value
        case .int(let value):
            return String(value)
        case .double(let value):
            return String(value)
        case .bool(let value):
            return String(value)
        case .null:
            return "null"
        case .array, .object:
            guard let data = try? JSONEncoder().encode(value),
                  let result = String(data: data, encoding: .utf8) else {
                return ""
            }
            return result
        }
    }
}
