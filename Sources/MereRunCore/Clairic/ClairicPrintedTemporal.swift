#if !os(iOS)
import Foundation

enum ClairicPrintedTemporal {
    static let instruction = " Copy only a date or time explicitly printed for this exact requested role. "
        + "Return null if this role has no printed date or time. Do not use invoice or due dates for service or billing period boundaries. "
        + "Do not infer a period from its duration. Copy the requested date or time exactly as printed, including AM or PM when present. "
        + "Preserve every digit. Do not normalize or convert the value. Return null only if the requested value is absent."

    static func normalize(_ value: ChatJSONValue, as kind: ClairicField.Kind?) -> ChatJSONValue? {
        guard let kind, value != .null else { return value }
        guard case .string(let printed) = value else { return nil }
        let text = printed.trimmingCharacters(in: .whitespacesAndNewlines)
        switch kind {
        case .date: return day(text).map(ChatJSONValue.string)
        case .time: return clock(text).map(ChatJSONValue.string)
        case .datetime:
            guard let parts = match(#"^(.+?)(?:T|\s+(?:at\s+)?)(\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AP]\.?M\.?)?)$"#, in: text),
                  let date = day(parts[0]), let time = clock(parts[1]) else { return nil }
            return .string(date + "T" + time)
        default: return nil
        }
    }

    private static func clock(_ text: String) -> String? {
        guard let parts = match(#"^(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AP]\.?M\.?)?$"#, in: text),
              var hour = Int(parts[0]), let minute = Int(parts[1]) else { return nil }
        let second = Int(parts[2]) ?? 0
        if !parts[3].isEmpty {
            guard (1...12).contains(hour) else { return nil }
            hour = hour % 12 + (parts[3].uppercased().hasPrefix("P") ? 12 : 0)
        }
        guard hour < 24, minute < 60, second < 60 else { return nil }
        return String(format: "%02d:%02d:%02d", hour, minute, second)
    }

    private static func day(_ text: String) -> String? {
        if let parts = match(#"^(\d{4})([-/.])(\d{1,2})\2(\d{1,2})$"#, in: text),
           let year = Int(parts[0]), let month = Int(parts[2]), let day = Int(parts[3]) {
            return date(year: year, month: month, day: day)
        }
        if let parts = match(#"^(\d{1,2})([-/.])(\d{1,2})\2(\d{4})$"#, in: text),
           let first = Int(parts[0]), let second = Int(parts[2]), let year = Int(parts[3]) {
            let readings = Set([date(year: year, month: first, day: second),
                                date(year: year, month: second, day: first)].compactMap { $0 })
            return readings.count == 1 ? readings.first : nil
        }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.isLenient = false
        for format in ["MMMM d, yyyy", "MMM d, yyyy", "MMMM d yyyy", "MMM d yyyy", "d MMMM yyyy", "d MMM yyyy"] {
            formatter.dateFormat = format
            guard let parsed = formatter.date(from: text) else { continue }
            formatter.dateFormat = "yyyy-MM-dd"
            return formatter.string(from: parsed)
        }
        return nil
    }

    private static func date(year: Int, month: Int, day: Int) -> String? {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = .gmt
        let components = DateComponents(year: year, month: month, day: day)
        guard year > 0, let value = calendar.date(from: components),
              calendar.dateComponents([.year, .month, .day], from: value) == components else { return nil }
        return String(format: "%04d-%02d-%02d", year, month, day)
    }

    private static func match(_ pattern: String, in text: String) -> [String]? {
        guard let expression = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
              let result = expression.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) else { return nil }
        return (1..<result.numberOfRanges).map { index in
            Range(result.range(at: index), in: text).map { String(text[$0]) } ?? ""
        }
    }
}
#endif
