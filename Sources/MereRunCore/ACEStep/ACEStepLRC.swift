import Foundation

public struct ACEStepLRCLine: Codable, Hashable, Sendable {
    public var timestampSeconds: Double
    public var text: String

    public init(timestampSeconds: Double, text: String) {
        self.timestampSeconds = timestampSeconds
        self.text = text
    }
}

public struct ACEStepLRCDocument: Codable, Hashable, Sendable {
    public var metadata: [String: String]
    public var lines: [ACEStepLRCLine]
    public var timingIsApproximate: Bool

    public init(
        metadata: [String: String] = [:],
        lines: [ACEStepLRCLine],
        timingIsApproximate: Bool = false
    ) {
        self.metadata = metadata
        self.lines = lines.sorted {
            if $0.timestampSeconds == $1.timestampSeconds {
                return $0.text < $1.text
            }
            return $0.timestampSeconds < $1.timestampSeconds
        }
        self.timingIsApproximate = timingIsApproximate
    }

    public var lyrics: String {
        lines.map(\.text).joined(separator: "\n")
    }

    public func rendered() -> String {
        var output = metadata.keys.sorted().map {
            "[\($0):\(metadata[$0] ?? "")]"
        }
        if timingIsApproximate {
            output.append("[re:mere.run approximate line timing]")
        }
        output.append(contentsOf: lines.map { line in
            "[\(Self.timestamp(line.timestampSeconds))]\(line.text)"
        })
        return output.joined(separator: "\n") + "\n"
    }

    public static func parse(_ text: String) throws -> ACEStepLRCDocument {
        var metadata: [String: String] = [:]
        var parsedLines: [ACEStepLRCLine] = []
        let timestampExpression = try NSRegularExpression(
            pattern: #"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]"#
        )
        let metadataExpression = try NSRegularExpression(
            pattern: #"^\[([A-Za-z][A-Za-z0-9_-]*):(.*)\]$"#
        )

        for rawLine in text.components(separatedBy: .newlines) {
            let range = NSRange(rawLine.startIndex..., in: rawLine)
            let matches = timestampExpression.matches(
                in: rawLine,
                range: range
            )
            if matches.isEmpty {
                if let match = metadataExpression.firstMatch(
                    in: rawLine,
                    range: range
                ),
                   let keyRange = Range(match.range(at: 1), in: rawLine),
                   let valueRange = Range(match.range(at: 2), in: rawLine)
                {
                    metadata[String(rawLine[keyRange]).lowercased()] =
                        String(rawLine[valueRange])
                }
                continue
            }
            let textStart = matches.last!.range.location
                + matches.last!.range.length
            let lyricRange = rawLine.index(
                rawLine.startIndex,
                offsetBy: textStart
            )..<rawLine.endIndex
            let lyric = String(rawLine[lyricRange])
                .trimmingCharacters(in: .whitespaces)
            for match in matches {
                guard let minuteRange = Range(match.range(at: 1), in: rawLine),
                      let secondRange = Range(match.range(at: 2), in: rawLine),
                      let minutes = Double(rawLine[minuteRange]),
                      let seconds = Double(rawLine[secondRange])
                else {
                    continue
                }
                var fraction = 0.0
                if match.range(at: 3).location != NSNotFound,
                   let fractionRange = Range(match.range(at: 3), in: rawLine)
                {
                    let digits = String(rawLine[fractionRange])
                    fraction = (Double(digits) ?? 0)
                        / pow(10, Double(digits.count))
                }
                parsedLines.append(
                    ACEStepLRCLine(
                        timestampSeconds: minutes * 60 + seconds + fraction,
                        text: lyric
                    )
                )
            }
        }
        guard !parsedLines.isEmpty else {
            throw ACEStepLRCError.noTimedLines
        }
        return ACEStepLRCDocument(metadata: metadata, lines: parsedLines)
    }

    public static func approximate(
        lyrics: String,
        durationSeconds: Double
    ) -> ACEStepLRCDocument {
        let lyricLines = lyrics.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        let count = max(lyricLines.count, 1)
        let usableDuration = max(0, durationSeconds)
        return ACEStepLRCDocument(
            lines: lyricLines.enumerated().map { index, text in
                ACEStepLRCLine(
                    timestampSeconds: usableDuration
                        * Double(index) / Double(count),
                    text: text
                )
            },
            timingIsApproximate: true
        )
    }

    private static func timestamp(_ seconds: Double) -> String {
        let centiseconds = max(0, Int((seconds * 100).rounded()))
        let minutes = centiseconds / 6_000
        let remaining = centiseconds % 6_000
        return String(
            format: "%02d:%02d.%02d",
            minutes,
            remaining / 100,
            remaining % 100
        )
    }
}

public enum ACEStepLRCError: LocalizedError {
    case noTimedLines

    public var errorDescription: String? {
        "LRC input contains no [mm:ss.xx] timed lyric lines."
    }
}
