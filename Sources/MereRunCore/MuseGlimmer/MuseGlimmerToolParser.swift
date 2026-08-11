import Foundation

/// Parses Muse Glimmer's ATEM tool-call protocol without treating it as XML.
public enum MuseGlimmerToolParser {
    private static let invokePattern = #"(?s)<atem:invoke\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</atem:invoke>"#
    private static let parameterPattern = #"(?s)<atem:parameter\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</atem:parameter>"#

    public static func parseToolCalls(_ text: String) -> [ToolCall] {
        matches(pattern: invokePattern, in: text).compactMap { match in
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

    private static func unescape(_ text: String) -> String {
        text
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&amp;", with: "&")
    }
}
