import Foundation

/// The two-voice ABC dialect shared by YuE2 and SheetSage2. Times use 1/1024 whole-note ticks.
public struct YuE2Score: Sendable {
    public struct Note: Equatable, Codable, Sendable {
        public var onset: Int
        public var pitch: Int
        public var duration: Int
    }

    public struct Bar: Equatable, Sendable {
        public let onset: Int
        public let numerator: Int
        public let denominator: Int
        public let key: String
        public var duration: Int { numerator * 1024 / denominator }
    }

    public struct Chord: Equatable, Sendable {
        public let onset: Int
        public let symbol: String
    }

    public let bpm: Int
    public let bars: [Bar]
    public let vocal: [Note]
    public let instrumental: [Note]
    public let chords: [Chord]
    public let sections: [Int: String]

    public init(abc: String) throws {
        self = try YuE2ScoreParser.parse(abc)
    }

    init(bpm: Int, bars: [Bar], vocal: [Note], instrumental: [Note], chords: [Chord], sections: [Int: String]) {
        self.bpm = bpm
        self.bars = bars
        self.vocal = vocal
        self.instrumental = instrumental
        self.chords = chords
        self.sections = sections
    }

    public enum OverlapPolicy: String, Codable, Sendable { case vocal, error }

    public struct Transfer: Codable, Sendable {
        public let abc: String
        public let vocalNotes: Int
        public let affectedInstrumentalNotes: [Int]
        public let overlapPolicy: OverlapPolicy
    }

    /// Preserve every vocal pitch, onset and duration; retain accompaniment only outside vocal intervals.
    public func instrumentalized(overlap: OverlapPolicy = .vocal, keepChords: Bool = true) throws -> Transfer {
        guard !vocal.isEmpty || !instrumental.isEmpty else {
            throw YuE2Error.invalidRequest("The score contains no sounding notes.")
        }
        var notes = vocal
        var affected: [Int] = []
        for (index, note) in instrumental.enumerated() {
            var pieces = [note.onset..<(note.onset + note.duration)]
            for melody in vocal {
                let occupied = melody.onset..<(melody.onset + melody.duration)
                pieces = pieces.flatMap { piece -> [Range<Int>] in
                    if !piece.overlaps(occupied) { return [piece] }
                    return [piece.lowerBound..<max(piece.lowerBound, occupied.lowerBound),
                            min(piece.upperBound, occupied.upperBound)..<piece.upperBound].filter { !$0.isEmpty }
                }
            }
            if pieces != [note.onset..<(note.onset + note.duration)] {
                guard overlap == .vocal else {
                    throw YuE2Error.invalidRequest("Vocal and Ins overlap; select vocal priority or edit the score.")
                }
                affected.append(index)
            }
            notes += pieces.map { Note(onset: $0.lowerBound, pitch: note.pitch, duration: $0.count) }
        }
        notes.sort { $0.onset < $1.onset }
        let output = try serialized(vocal: [], instrumental: notes, chords: keepChords ? chords : [])
        let checked = try Self(abc: output)
        guard checked.vocal.isEmpty, checked.instrumental == notes, checked.bars == bars,
              checked.bpm == bpm, vocal.allSatisfy(checked.instrumental.contains) else {
            throw YuE2Error.invalidRequest("Instrumental conversion changed the intended musical events.")
        }
        return Transfer(abc: output, vocalNotes: vocal.count, affectedInstrumentalNotes: affected, overlapPolicy: overlap)
    }

    public func melodyOnly() throws -> String {
        try serialized(vocal: vocal, instrumental: instrumental, chords: [])
    }

    public func serialized() throws -> String {
        try serialized(vocal: vocal, instrumental: instrumental, chords: chords)
    }

    private func serialized(vocal: [Note], instrumental: [Note], chords: [Chord]) throws -> String {
        guard let first = bars.first else { throw YuE2Error.invalidRequest("The score has no measures.") }
        var lines = ["X:1", "T:", "M:\(first.numerator)/\(first.denominator)", "L:1/32", "Q:1/4=\(bpm)",
                     "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"",
                     "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"", "K:\(first.key)"]
        for bar in bars {
            if let section = sections[bar.onset] { lines.append("% \(section)") }
            for (name, notes) in [("Vocal", vocal), ("Ins", instrumental)] {
                lines += ["V: \(name)", "M:\(bar.numerator)/\(bar.denominator)", "K:\(bar.key)"]
                lines.append(try Self.render(bar, notes: notes, chords: name == "Vocal" ? chords : []) + " |")
            }
        }
        return lines.joined(separator: "\n") + "\n"
    }

    private static func render(_ bar: Bar, notes: [Note], chords: [Chord]) throws -> String {
        let end = bar.onset + bar.duration
        var cursor = bar.onset
        var tokens: [String] = []
        while cursor < end {
            let note = notes.first { $0.onset <= cursor && cursor < $0.onset + $0.duration }
            let nextNote = notes.first { $0.onset > cursor }?.onset ?? end
            let nextChord = chords.first { $0.onset > cursor }?.onset ?? end
            var boundary = min(end, nextNote, nextChord, note.map { $0.onset + $0.duration } ?? end)
            if let chord = chords.first(where: { $0.onset == cursor }) { tokens.append("\"\(chord.symbol)\"") }
            guard (boundary - cursor) % 32 == 0 else {
                throw YuE2Error.invalidRequest("The instrumental workflow requires durations on the native 1/32 grid.")
            }
            // Split at supported duration values and tie continuations, including across measures.
            let units = (boundary - cursor) / 32
            guard let duration = [48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1].first(where: { $0 <= units }) else {
                throw YuE2Error.invalidRequest("Score events must advance musical time.")
            }
            boundary = cursor + duration * 32
            let symbol = note.map { pitchName($0.pitch) } ?? "z"
            let tie = note.map { boundary < $0.onset + $0.duration } ?? false
            tokens.append(symbol + (duration == 1 ? "" : String(duration)) + (tie ? "-" : ""))
            cursor = boundary
        }
        return tokens.joined(separator: " ")
    }

    private static func pitchName(_ pitch: Int) -> String {
        let names = ["=C", "^C", "=D", "^D", "=E", "=F", "^F", "=G", "^G", "=A", "^A", "=B"]
        let octave = pitch / 12 - 5
        let base = names[pitch % 12]
        if octave >= 1 { return base.lowercased() + String(repeating: "'", count: octave - 1) }
        return base + String(repeating: ",", count: -octave)
    }
}
