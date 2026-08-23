import Foundation

public enum MiniMaxMusic3SongMeter: String, CaseIterable, Codable, Sendable {
    case fourFour = "4/4"
    case threeFour = "3/4"
    case sixEight = "6/8"

    var beatsPerBar: Int {
        switch self {
        case .fourFour:
            4
        case .threeFour:
            3
        case .sixEight:
            2
        }
    }
}

public enum MiniMaxMusic3SectionTag: String, CaseIterable, Codable, Sendable {
    case intro
    case verse
    case preChorus = "pre-chorus"
    case chorus
    case postChorus = "post-chorus"
    case bridge
    case instrumental
    case solo
    case outro

    var normallyInstrumental: Bool {
        self == .instrumental || self == .solo
    }
}

public struct MiniMaxMusic3SongSection: Codable, Equatable, Sendable {
    public var tag: MiniMaxMusic3SectionTag
    public var approximateBars: Int
    public var targetLyricLines: Int
    public var vocalPlan: String
    public var productionEvents: String

    public init(
        tag: MiniMaxMusic3SectionTag,
        approximateBars: Int,
        targetLyricLines: Int,
        vocalPlan: String,
        productionEvents: String
    ) {
        self.tag = tag
        self.approximateBars = approximateBars
        self.targetLyricLines = targetLyricLines
        self.vocalPlan = vocalPlan
        self.productionEvents = productionEvents
    }

    enum CodingKeys: String, CodingKey {
        case tag
        case approximateBars = "approximate_bars"
        case targetLyricLines = "target_lyric_lines"
        case vocalPlan = "vocal_plan"
        case productionEvents = "production_events"
    }
}

public struct MiniMaxMusic3SongBlueprint: Codable, Equatable, Sendable {
    public var bpm: Int
    public var meter: MiniMaxMusic3SongMeter
    public var durationUse: String
    public var sections: [MiniMaxMusic3SongSection]

    public init(
        bpm: Int,
        meter: MiniMaxMusic3SongMeter,
        durationUse: String,
        sections: [MiniMaxMusic3SongSection]
    ) {
        self.bpm = bpm
        self.meter = meter
        self.durationUse = durationUse
        self.sections = sections
    }

    enum CodingKeys: String, CodingKey {
        case bpm
        case meter
        case durationUse = "duration_use"
        case sections
    }
}

public struct MiniMaxMusic3CompositionRequest: Codable, Equatable, Sendable {
    public var brief: String
    public var durationSeconds: Float
    public var instrumental: Bool
    public var authoritativeLyrics: String?

    public init(
        brief: String,
        durationSeconds: Float,
        instrumental: Bool,
        authoritativeLyrics: String? = nil
    ) {
        self.brief = brief
        self.durationSeconds = durationSeconds
        self.instrumental = instrumental
        self.authoritativeLyrics = authoritativeLyrics
    }

    enum CodingKeys: String, CodingKey {
        case brief
        case durationSeconds = "duration_seconds"
        case instrumental
        case authoritativeLyrics = "authoritative_lyrics"
    }
}

public struct MiniMaxMusic3ComposedSong: Codable, Equatable, Sendable {
    public var title: String
    public var tags: [String]
    public var bpm: Int
    public var language: String
    public var lyrics: String
    public var globalMetadata: String
    public var vocalDetails: String
    public var arrangement: String

    public init(
        title: String,
        tags: [String],
        bpm: Int,
        language: String,
        lyrics: String,
        globalMetadata: String,
        vocalDetails: String,
        arrangement: String
    ) {
        self.title = title
        self.tags = tags
        self.bpm = bpm
        self.language = language
        self.lyrics = lyrics
        self.globalMetadata = globalMetadata
        self.vocalDetails = vocalDetails
        self.arrangement = arrangement
    }

    public var caption: String {
        """
        ### Global Metadata
        \(globalMetadata)

        ### Vocal Details
        \(vocalDetails)

        ### Arrangement
        \(arrangement)
        """
    }

    enum CodingKeys: String, CodingKey {
        case title
        case tags
        case bpm
        case language
        case lyrics
        case globalMetadata = "global_metadata"
        case vocalDetails = "vocal_details"
        case arrangement
    }
}

public enum MiniMaxMusic3CompositionError: LocalizedError, Equatable {
    case invalid(String)

    public var errorDescription: String? {
        switch self {
        case .invalid(let message):
            "Invalid MiniMax Music 3 composition: \(message)"
        }
    }
}

public enum MiniMaxMusic3LyricIssueSeverity: String, Codable, Equatable, Sendable {
    case warning
    case blocker
}

public enum MiniMaxMusic3LyricPreflightPolicy: String, CaseIterable, Codable, Sendable {
    case off
    case warn
    case strict
}

public struct MiniMaxMusic3LyricIssue: Codable, Equatable, Sendable {
    public var id: String
    public var severity: MiniMaxMusic3LyricIssueSeverity
    public var message: String

    public init(id: String, severity: MiniMaxMusic3LyricIssueSeverity, message: String) {
        self.id = id
        self.severity = severity
        self.message = message
    }
}

public struct MiniMaxMusic3LyricPreflightReport: Codable, Equatable, Sendable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var durationSeconds: Float
    public var instrumental: Bool
    public var sectionTags: [MiniMaxMusic3SectionTag]
    public var sungLineCount: Int
    public var sungWordCount: Int
    public var issues: [MiniMaxMusic3LyricIssue]

    public init(
        durationSeconds: Float,
        instrumental: Bool,
        sectionTags: [MiniMaxMusic3SectionTag],
        sungLineCount: Int,
        sungWordCount: Int,
        issues: [MiniMaxMusic3LyricIssue]
    ) {
        self.schemaVersion = Self.currentSchemaVersion
        self.durationSeconds = durationSeconds
        self.instrumental = instrumental
        self.sectionTags = sectionTags
        self.sungLineCount = sungLineCount
        self.sungWordCount = sungWordCount
        self.issues = issues
    }

    public var hasBlockers: Bool {
        issues.contains { $0.severity == .blocker }
    }

    public var isClean: Bool {
        issues.isEmpty
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case durationSeconds = "duration_seconds"
        case instrumental
        case sectionTags = "section_tags"
        case sungLineCount = "sung_line_count"
        case sungWordCount = "sung_word_count"
        case issues
    }
}

public enum MiniMaxMusic3ComposerContract {
    public static let systemPrompt = """
    You create typed inputs for MiniMax Music 3. Return exactly one JSON object and no Markdown.
    Keep musical direction separate from lyrical subject. Use only the supported section tags intro,
    verse, pre-chorus, chorus, post-chorus, bridge, instrumental, solo, and outro. Never put stage
    directions or production notes in lyrics. Preserve explicit constraints and any authoritative
    lyrics. Captions must describe an energy arc and instrument lifecycles rather than a static tag
    list. Do not quote, paraphrase, or summarize lyric lines inside the caption fields. Do not invent
    an exact key, BPM, singer, or production technique when the brief does not support it.
    """

    public static func blueprintPrompt(_ request: MiniMaxMusic3CompositionRequest) throws -> String {
        try validateRequest(request)
        let requestJSON = try encodedJSONString(request)
        let range = recommendedSectionRange(durationSeconds: request.durationSeconds)
        return """
        Plan the complete song timeline described by this request:
        \(requestJSON)

        Return this exact JSON shape:
        {"bpm":96,"meter":"4/4","duration_use":"how the full requested timeline is used","sections":[{"tag":"intro","approximate_bars":4,"target_lyric_lines":0,"vocal_plan":"wordless opening","production_events":"what enters, exits, or changes"}]}

        Requirements:
        - Use \(range.lowerBound)-\(range.upperBound) ordered sections and end with outro.
        - BPM must be between 60 and 180. Meter must be 4/4, 3/4, or 6/8.
        - Fill the requested duration using duration_seconds * bpm / 240 bars for 4/4,
          / 180 for 3/4, or / 120 for 6/8. Every section needs 1-64 bars; reserve
          one-bar sections for short-form transitions.
        - Every vocal section needs a realistic target_lyric_lines budget; instrumental, solo,
          wordless, and no-lead-vocal sections use zero.
        - If authoritative_lyrics is present, preserve its exact supported section order instead
          of the recommended section count, do not add an outro, and copy the actual sung-line
          count into each target_lyric_lines value.
        - If instrumental is true, include at least one instrumental section and make every
          target_lyric_lines value zero.
        """
    }

    public static func songPrompt(
        _ request: MiniMaxMusic3CompositionRequest,
        blueprint: MiniMaxMusic3SongBlueprint
    ) throws -> String {
        try validateRequest(request)
        let requestJSON = try encodedJSONString(request)
        let blueprintJSON = try encodedJSONString(blueprint)
        return """
        Write the finished MiniMax Music 3 inputs for this request and normalized timeline.

        Request:
        \(requestJSON)

        Timeline:
        \(blueprintJSON)

        Return this exact JSON shape:
        {"title":"short title","tags":["genre","mood","vocal cue"],"bpm":96,"language":"en","lyrics":"[intro]\\n[verse]\\nfirst sung line","global_metadata":"Basic Attributes: ... Global Emotional Progression: ... Application Scenarios & Imagery: ... Sonics & Production Profile: ...","vocal_details":"Vocal Gender & Timbre: ... Vocal Style: ... Harmony/Backing Vocals: ... Vocal FX: ...","arrangement":"Instrument Lifecycle Description (Primary/Secondary Layering): ... Groove & Foundation Progression: ... Embellishments, Textures & Spatial FX: ..."}

        Requirements:
        - Return 3-6 concise tags and use the timeline BPM exactly.
        - Put every section tag alone on its own line and follow the exact timeline order.
        - Meet each vocal section's target_lyric_lines budget with complete sung lines.
        - If authoritative_lyrics is present, return it unchanged. If instrumental is true, return
          only supported instrumental section markers and set language to instrumental.
        - The three caption fields together should be about 250-450 words, concrete, and consistent.
        - Keep all lyric text out of the three caption fields.
        """
    }

    public static func normalize(
        blueprint: MiniMaxMusic3SongBlueprint,
        request: MiniMaxMusic3CompositionRequest
    ) throws -> MiniMaxMusic3SongBlueprint {
        try validateRequest(request)
        guard (60...180).contains(blueprint.bpm) else {
            throw MiniMaxMusic3CompositionError.invalid("blueprint BPM must be between 60 and 180")
        }
        let authoritative = request.authoritativeLyrics.map(MiniMaxMusic3LyricPreflight.parse)
        if let authoritative {
            guard !authoritative.tags.isEmpty else {
                throw MiniMaxMusic3CompositionError.invalid(
                    "authoritative lyrics need supported standalone section tags"
                )
            }
            guard authoritative.tags == blueprint.sections.map(\.tag) else {
                throw MiniMaxMusic3CompositionError.invalid(
                    "blueprint section order does not match the authoritative lyrics"
                )
            }
        } else {
            let range = recommendedSectionRange(durationSeconds: request.durationSeconds)
            guard range.contains(blueprint.sections.count) else {
                throw MiniMaxMusic3CompositionError.invalid(
                    "blueprint needs \(range.lowerBound)-\(range.upperBound) sections for this duration"
                )
            }
            guard blueprint.sections.last?.tag == .outro else {
                throw MiniMaxMusic3CompositionError.invalid("blueprint must end with outro")
            }
            if request.instrumental,
               !blueprint.sections.contains(where: { $0.tag == .instrumental })
            {
                throw MiniMaxMusic3CompositionError.invalid(
                    "instrumental blueprint needs an instrumental section"
                )
            }
        }
        guard blueprint.sections.allSatisfy({ (1...64).contains($0.approximateBars) }) else {
            throw MiniMaxMusic3CompositionError.invalid("every section must contain 1-64 bars")
        }
        guard blueprint.sections.allSatisfy({ (0...64).contains($0.targetLyricLines) }) else {
            throw MiniMaxMusic3CompositionError.invalid("lyric-line targets must be between 0 and 64")
        }

        let targetBars = Int(round(
            Double(request.durationSeconds) * Double(blueprint.bpm)
                / Double(60 * blueprint.meter.beatsPerBar)
        ))
        guard targetBars >= blueprint.sections.count,
              targetBars <= blueprint.sections.count * 64
        else {
            throw MiniMaxMusic3CompositionError.invalid(
                "the selected BPM, meter, and section count cannot fill the requested duration"
            )
        }

        var normalized = blueprint
        normalized.durationUse = normalized.durationUse.trimmingCharacters(in: .whitespacesAndNewlines)
        normalized.sections = try fitBars(normalized.sections, target: targetBars)
        for index in normalized.sections.indices {
            normalized.sections[index].vocalPlan = normalized.sections[index].vocalPlan
                .trimmingCharacters(in: .whitespacesAndNewlines)
            normalized.sections[index].productionEvents = normalized.sections[index].productionEvents
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let authoritative {
                normalized.sections[index].targetLyricLines = authoritative.linesBySection[index].count
            } else if request.instrumental || normalized.sections[index].tag.normallyInstrumental {
                normalized.sections[index].targetLyricLines = 0
            } else if normalized.sections[index].targetLyricLines > 0 {
                normalized.sections[index].targetLyricLines = max(
                    (normalized.sections[index].approximateBars + 1) / 2,
                    normalized.sections[index].targetLyricLines
                )
            }
        }

        if !request.instrumental,
           !normalized.sections.contains(where: { $0.targetLyricLines > 0 })
        {
            throw MiniMaxMusic3CompositionError.invalid("a vocal composition needs at least one vocal section")
        }
        return normalized
    }

    public static func normalize(
        song: MiniMaxMusic3ComposedSong,
        request: MiniMaxMusic3CompositionRequest,
        blueprint: MiniMaxMusic3SongBlueprint
    ) throws -> MiniMaxMusic3ComposedSong {
        var normalized = song
        normalized.title = normalized.title.trimmingCharacters(in: .whitespacesAndNewlines)
        normalized.tags = normalized.tags.map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        }.filter { !$0.isEmpty }
        normalized.language = normalized.language.trimmingCharacters(in: .whitespacesAndNewlines)
        normalized.globalMetadata = normalized.globalMetadata.trimmingCharacters(in: .whitespacesAndNewlines)
        normalized.vocalDetails = normalized.vocalDetails.trimmingCharacters(in: .whitespacesAndNewlines)
        normalized.arrangement = normalized.arrangement.trimmingCharacters(in: .whitespacesAndNewlines)
        normalized.lyrics = if request.instrumental {
            blueprint.sections.map { "[\($0.tag.rawValue)]" }.joined(separator: "\n")
        } else if let authoritativeLyrics = request.authoritativeLyrics {
            authoritativeLyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            MiniMaxMusic3LyricPreflight.normalizedTags(in: normalized.lyrics)
        }

        guard !normalized.title.isEmpty else {
            throw MiniMaxMusic3CompositionError.invalid("song title is empty")
        }
        guard (3...6).contains(normalized.tags.count) else {
            throw MiniMaxMusic3CompositionError.invalid("song needs 3-6 non-empty tags")
        }
        guard normalized.bpm == blueprint.bpm else {
            throw MiniMaxMusic3CompositionError.invalid("song BPM does not match the blueprint")
        }
        guard !normalized.globalMetadata.isEmpty,
              !normalized.vocalDetails.isEmpty,
              !normalized.arrangement.isEmpty
        else {
            throw MiniMaxMusic3CompositionError.invalid("all three structured caption fields are required")
        }
        if request.instrumental {
            normalized.language = "instrumental"
            guard normalized.vocalDetails.localizedCaseInsensitiveContains("instrumental")
                    || normalized.vocalDetails.localizedCaseInsensitiveContains("no vocal")
            else {
                throw MiniMaxMusic3CompositionError.invalid(
                    "instrumental vocal details must state that there are no vocals"
                )
            }
        }

        let report = MiniMaxMusic3LyricPreflight.inspect(
            lyrics: normalized.lyrics,
            durationSeconds: request.durationSeconds,
            instrumental: request.instrumental,
            blueprint: blueprint
        )
        if let blocker = report.issues.first(where: { $0.severity == .blocker }) {
            throw MiniMaxMusic3CompositionError.invalid(blocker.message)
        }
        let caption = normalized.caption.lowercased()
        for lyricLine in MiniMaxMusic3LyricPreflight.sungLines(in: normalized.lyrics) {
            let lowered = lyricLine.lowercased()
            if lowered.count >= 12, caption.contains(lowered) {
                throw MiniMaxMusic3CompositionError.invalid(
                    "structured caption reproduces a lyric line"
                )
            }
        }
        return normalized
    }

    static func recommendedSectionRange(durationSeconds: Float) -> ClosedRange<Int> {
        switch durationSeconds {
        case ...45:
            2...4
        case ...90:
            4...7
        case ...150:
            6...10
        case ...210:
            8...14
        default:
            10...18
        }
    }

    private static func validateRequest(_ request: MiniMaxMusic3CompositionRequest) throws {
        guard !request.brief.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MiniMaxMusic3CompositionError.invalid("brief is empty")
        }
        guard request.durationSeconds >= 10, request.durationSeconds <= 360 else {
            throw MiniMaxMusic3CompositionError.invalid("composer duration must be 10-360 seconds")
        }
        if request.instrumental, request.authoritativeLyrics != nil {
            throw MiniMaxMusic3CompositionError.invalid(
                "instrumental composition cannot include authoritative lyrics"
            )
        }
    }

    private static func fitBars(
        _ sections: [MiniMaxMusic3SongSection],
        target: Int
    ) throws -> [MiniMaxMusic3SongSection] {
        let originalTotal = sections.reduce(0) { $0 + $1.approximateBars }
        guard originalTotal > 0 else {
            throw MiniMaxMusic3CompositionError.invalid("blueprint bar total is zero")
        }
        let scaled = sections.map {
            Double(target * $0.approximateBars) / Double(originalTotal)
        }
        var bars = scaled.map { min(64, max(1, Int($0.rounded(.down)))) }
        while bars.reduce(0, +) < target {
            guard let index = bars.indices.filter({ bars[$0] < 64 }).max(by: {
                scaled[$0] - Double(bars[$0]) < scaled[$1] - Double(bars[$1])
            }) else {
                throw MiniMaxMusic3CompositionError.invalid("could not fit section bars")
            }
            bars[index] += 1
        }
        while bars.reduce(0, +) > target {
            guard let index = bars.indices.filter({ bars[$0] > 1 }).max(by: {
                Double(bars[$0]) - scaled[$0] < Double(bars[$1]) - scaled[$1]
            }) else {
                throw MiniMaxMusic3CompositionError.invalid("could not fit section bars")
            }
            bars[index] -= 1
        }
        return zip(sections, bars).map { section, fittedBars in
            var fitted = section
            let ratio = Double(fittedBars) / Double(section.approximateBars)
            fitted.approximateBars = fittedBars
            if section.targetLyricLines > 0 {
                fitted.targetLyricLines = min(64, max(1, Int(round(Double(section.targetLyricLines) * ratio))))
            }
            return fitted
        }
    }

    private static func encodedJSONString<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}

public enum MiniMaxMusic3LyricPreflight {
    struct ParsedLyrics {
        var tags: [MiniMaxMusic3SectionTag]
        var linesBySection: [[String]]
        var untaggedLines: [String]
        var issues: [MiniMaxMusic3LyricIssue]
    }

    public static func inspect(
        lyrics: String,
        durationSeconds: Float,
        instrumental: Bool,
        blueprint: MiniMaxMusic3SongBlueprint? = nil
    ) -> MiniMaxMusic3LyricPreflightReport {
        let parsed = parse(lyrics)
        let lines = sungLines(in: lyrics)
        let hasNonLatin = lines.joined().unicodeScalars.contains {
            CharacterSet.letters.contains($0) && !$0.isASCII
        }
        let words = lines.reduce(0) { $0 + wordCount(in: $1) }
        var issues = parsed.issues

        if instrumental {
            if !lines.isEmpty {
                issues.append(.init(
                    id: "instrumental_has_lyrics",
                    severity: .blocker,
                    message: "instrumental input contains sung lyric lines"
                ))
            }
            if !parsed.tags.contains(.instrumental) {
                issues.append(.init(
                    id: "instrumental_tag_missing",
                    severity: .warning,
                    message: "instrumental input should include an [instrumental] section"
                ))
            }
        } else {
            let minimums = minimumCoverage(for: durationSeconds)
            if lines.count < minimums.lines {
                issues.append(.init(
                    id: "lyrics_underfilled_lines",
                    severity: .warning,
                    message: "lyrics have \(lines.count) sung lines; about \(minimums.lines) are recommended for a \(Int(durationSeconds))-second upper bound"
                ))
            }
            if !hasNonLatin, words < minimums.words {
                issues.append(.init(
                    id: "lyrics_underfilled_words",
                    severity: .warning,
                    message: "lyrics have \(words) sung words; about \(minimums.words) are recommended for this duration"
                ))
            }
            if parsed.tags.isEmpty {
                issues.append(.init(
                    id: "lyrics_sections_missing",
                    severity: .warning,
                    message: "lyrics have no supported section tags"
                ))
            }
            if durationSeconds >= 120, parsed.tags.last != .outro {
                issues.append(.init(
                    id: "lyrics_outro_missing",
                    severity: .warning,
                    message: "long-song lyrics should end with an [outro] section"
                ))
            }
        }

        if !parsed.untaggedLines.isEmpty, !parsed.tags.isEmpty {
            issues.append(.init(
                id: "lyrics_untagged_content",
                severity: .warning,
                message: "lyrics contain sung lines before the first section tag"
            ))
        }

        if let blueprint {
            let expected = blueprint.sections.map(\.tag)
            if parsed.tags != expected {
                issues.append(.init(
                    id: "lyrics_section_sequence_mismatch",
                    severity: .blocker,
                    message: "lyric section order does not match the composed timeline"
                ))
            } else {
                for (index, pair) in zip(blueprint.sections, parsed.linesBySection).enumerated() {
                    let (section, sectionLines) = pair
                    if sectionLines.count < section.targetLyricLines {
                        issues.append(.init(
                            id: "lyrics_section_\(index + 1)_underfilled",
                            severity: .blocker,
                            message: "section \(index + 1) [\(section.tag.rawValue)] has \(sectionLines.count) sung lines but the timeline requires \(section.targetLyricLines)"
                        ))
                    }
                    let maximum = section.targetLyricLines == 0
                        ? 0
                        : section.targetLyricLines + max(2, section.targetLyricLines / 2)
                    if sectionLines.count > maximum {
                        issues.append(.init(
                            id: "lyrics_section_\(index + 1)_overfilled",
                            severity: .blocker,
                            message: "section \(index + 1) [\(section.tag.rawValue)] has \(sectionLines.count) sung lines but the timeline allows at most \(maximum)"
                        ))
                    }
                }
            }
        }

        return MiniMaxMusic3LyricPreflightReport(
            durationSeconds: durationSeconds,
            instrumental: instrumental,
            sectionTags: parsed.tags,
            sungLineCount: lines.count,
            sungWordCount: words,
            issues: uniqueIssues(issues)
        )
    }

    public static func normalizedTags(in lyrics: String) -> String {
        lyrics.split(whereSeparator: \.isNewline).map { rawLine in
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let tag = tag(forStandaloneLine: line) else { return line }
            return "[\(tag.rawValue)]"
        }.joined(separator: "\n")
    }

    static func parse(_ lyrics: String) -> ParsedLyrics {
        var tags: [MiniMaxMusic3SectionTag] = []
        var linesBySection: [[String]] = []
        var untaggedLines: [String] = []
        var issues: [MiniMaxMusic3LyricIssue] = []
        var currentSection: Int?

        for rawLine in lyrics.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline) {
            let line = String(rawLine).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty else { continue }
            if let tag = tag(forStandaloneLine: line) {
                tags.append(tag)
                linesBySection.append([])
                currentSection = linesBySection.count - 1
                continue
            }
            if line.hasPrefix("[") || line.hasSuffix("]") {
                issues.append(.init(
                    id: "lyrics_unsupported_or_inline_tag",
                    severity: .blocker,
                    message: "unsupported or inline lyric tag: \(line)"
                ))
                continue
            }
            if let currentSection {
                linesBySection[currentSection].append(line)
            } else {
                untaggedLines.append(line)
            }
        }

        let lowered = lyrics.lowercased()
        for placeholder in ["[end]", "[lyrics]", "[lyritic]", "[end song]"] where lowered.contains(placeholder) {
            issues.append(.init(
                id: "lyrics_placeholder_tag",
                severity: .blocker,
                message: "lyrics contain unsupported placeholder \(placeholder)"
            ))
        }
        return ParsedLyrics(
            tags: tags,
            linesBySection: linesBySection,
            untaggedLines: untaggedLines,
            issues: issues
        )
    }

    static func sungLines(in lyrics: String) -> [String] {
        let parsed = parse(lyrics)
        return parsed.untaggedLines + parsed.linesBySection.flatMap { $0 }
    }

    private static func tag(forStandaloneLine line: String) -> MiniMaxMusic3SectionTag? {
        guard line.first == "[", line.last == "]", line.count >= 3 else { return nil }
        let value = line.dropFirst().dropLast()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return MiniMaxMusic3SectionTag(rawValue: value)
    }

    private static func wordCount(in line: String) -> Int {
        line.split { character in
            character.isWhitespace || character.isPunctuation || character.isSymbol
        }.count
    }

    private static func minimumCoverage(for durationSeconds: Float) -> (lines: Int, words: Int) {
        switch durationSeconds {
        case ...30:
            (2, 12)
        case ...60:
            (6, 24)
        case ...120:
            (10, 52)
        default:
            (12, 60)
        }
    }

    private static func uniqueIssues(_ issues: [MiniMaxMusic3LyricIssue]) -> [MiniMaxMusic3LyricIssue] {
        var seen: Set<String> = []
        return issues.filter { seen.insert($0.id).inserted }
    }
}
