import Foundation

/// Parses Muse Glimmer's ATEM tool-call protocol without treating it as XML.
public enum MuseGlimmerToolParser {
    private static let invokePattern = #"(?s)<atem:invoke\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</atem:invoke>"#
    private static let parameterPattern = #"(?s)<atem:parameter\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</atem:parameter>"#

    public static func parseToolCalls(_ text: String) -> [ToolCall] {
        matches(pattern: invokePattern, in: executableSource(text)).compactMap { match in
            guard match.count == 3 else { return nil }
            let name = unescape(match[1]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { return nil }
            var arguments: [String: String] = [:]
            for parameter in matches(pattern: parameterPattern, in: match[2]) where parameter.count == 3 {
                let key = unescape(parameter[1]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !key.isEmpty {
                    arguments[key] = unescape(parameter[2])
                }
            }
            return ToolCall(name: name, arguments: arguments)
        }
    }

    static func validatedToolCalls(_ text: String, tools: [ToolDefinition], parallel: Bool) throws -> [ToolCall] {
        let text = executableSource(text)
        let invocations = matches(pattern: invokePattern, in: text)
        let openings = text.components(separatedBy: "<atem:invoke").count - 1
        guard openings == invocations.count,
              text.components(separatedBy: "<atem:function_calls>").count == text.components(separatedBy: "</atem:function_calls>").count else {
            throw ChatRequestIssue("tool_use", "model generated an incomplete tool call")
        }
        let packets = matches(pattern: #"(?s)<atem:function_calls>(.*?)</atem:function_calls>"#, in: text)
        if !packets.isEmpty, invocations.isEmpty {
            throw ChatRequestIssue("tool_use", "model generated a tool packet without a complete invocation")
        }
        let recipients = matches(pattern: #"to=([^<\s]+)<\|message\|>"#, in: text)
        if invocations.isEmpty, recipients.contains(where: { $0.count > 1 && !["self", "user"].contains($0[1]) }) {
            throw ChatRequestIssue("tool_use", "model generated a tool recipient without a complete invocation")
        }
        guard parallel || invocations.count <= 1 else { throw ChatRequestIssue("tool_use", "model generated parallel calls when disabled") }
        return try invocations.map { match in
            let name = unescape(match[1]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let tool = tools.first(where: { $0.name == name }) else { throw ChatRequestIssue("tool_use.name", "model generated an undeclared tool '\(name)'") }
            var arguments: [String: ChatJSONValue] = [:]
            let parameters = matches(pattern: parameterPattern, in: match[2])
            let residue = match[2].replacingOccurrences(of: parameterPattern, with: "", options: .regularExpression)
            guard residue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ChatRequestIssue("tool_use.input", "model generated unparseable parameter content")
            }
            guard parameters.count == match[2].components(separatedBy: "<atem:parameter").count - 1 else {
                throw ChatRequestIssue("tool_use.input", "model generated incomplete parameters")
            }
            for parameter in parameters {
                let key = unescape(parameter[1])
                guard arguments[key] == nil else { throw ChatRequestIssue("tool_use.input", "model repeated parameter '\(key)'") }
                let raw = unescape(parameter[2])
                let property = tool.parametersJSONSchema["properties"]?.objectValue?[key]?.objectValue
                if try ChatToolValidation.resolvedType(property ?? [:], root: tool.parametersJSONSchema) == "string" {
                    arguments[key] = .string(raw)
                } else {
                    do { arguments[key] = try JSONDecoder().decode(ChatJSONValue.self, from: Data(raw.utf8)) }
                    catch { throw ChatRequestIssue("tool_use.input", "parameter '\(key)' is not valid JSON for its schema") }
                }
            }
            try ChatToolValidation.validate(.object(arguments), schema: tool.parametersJSONSchema)
            return ToolCall(name: name, jsonArguments: arguments)
        }
    }

    public static func visibleText(_ text: String) -> String {
        text.replacingOccurrences(
            of: #"(?s)<atem:function_calls>.*?</atem:function_calls>"#,
            with: "",
            options: .regularExpression
        ).trimmingCharacters(in: .whitespacesAndNewlines)
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

    private static func executableSource(_ text: String) -> String {
        var channels = MuseGlimmerContentStream()
        _ = channels.append(text, final: true)
        return channels.toolSource
    }

    private static func unescape(_ text: String) -> String {
        text
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&amp;", with: "&")
    }
}
