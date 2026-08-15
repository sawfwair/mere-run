import Foundation

/// Parses the native Qwen3.5/Qwen3.8 function-call protocol emitted by the
/// checkpoint chat template.
///
/// Qwen-family format:
/// ```
/// <tool_call>
/// <function=tool_name>
/// <parameter=argument_name>
/// argument value
/// </parameter>
/// </function>
/// </tool_call>
/// ```
public enum Q35ToolParser {
    public static let closingTag = "</tool_call>"

    private static let callPattern = #"(?s)<tool_call>\s*(.*?)\s*</tool_call>"#
    private static let functionPattern = #"(?s)<function=([^>\r\n]+)>\s*(.*?)\s*</function>"#
    private static let parameterPattern = #"(?s)<parameter=([^>\r\n]+)>\s*(.*?)\s*</parameter>"#

    public static func parseToolCalls(_ text: String) -> [ToolCall] {
        matches(pattern: callPattern, in: text).compactMap { call in
            guard call.count == 2,
                  let function = matches(pattern: functionPattern, in: call[1]).first,
                  function.count == 3 else {
                return nil
            }

            let name = function[1].trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { return nil }

            var arguments: [String: String] = [:]
            for parameter in matches(pattern: parameterPattern, in: function[2]) where parameter.count == 3 {
                let key = parameter[1].trimmingCharacters(in: .whitespacesAndNewlines)
                guard !key.isEmpty else { continue }
                arguments[key] = parameter[2].trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return ToolCall(name: name, arguments: arguments)
        }
    }

    public static func containsCompletedToolCall(_ text: String) -> Bool {
        text.contains(closingTag)
    }

    private static func matches(pattern: String, in text: String) -> [[String]] {
        guard let expression = try? NSRegularExpression(pattern: pattern) else { return [] }
        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
        return expression.matches(in: text, range: fullRange).map { match in
            (0..<match.numberOfRanges).map { index in
                guard let range = Range(match.range(at: index), in: text) else { return "" }
                return String(text[range])
            }
        }
    }
}
