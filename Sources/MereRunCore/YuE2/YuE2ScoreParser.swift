import Foundation

enum YuE2ScoreParser {
    private struct Voice {
        var numerator: Int
        var denominator: Int
        var key: String
        var time = 0
        var notes: [YuE2Score.Note] = []
        var bars: [YuE2Score.Bar] = []
        var chords: [YuE2Score.Chord] = []
        var pending: (pitch: Int, written: Int)?
    }

    private static let tokenPattern = #""([^"\n]*)"|\[K:([^\]\n]+)\]|(\^\^|__|\^|_|=)?([A-Ga-gz])([,']*)([0-9]*)(-?)"#
    private static let majorKeys = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"]
    private static let minorKeys = ["Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m"]

    static func parse(_ abc: String) throws -> YuE2Score {
        var lines = abc.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n")
        while lines.last == "" { lines.removeLast() }
        guard lines.count >= 12, lines[0] == "X:1", lines[1] == "T:", lines[2].hasPrefix("M:"),
              lines[3].hasPrefix("L:1/"), let denominator = Int(lines[3].dropFirst(4)),
              denominator > 0, denominator <= 1024, denominator.nonzeroBitCount == 1,
              lines[4].hasPrefix("Q:1/4="), let bpm = Int(lines[4].dropFirst(6)), bpm > 0,
              lines[5] == "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"",
              lines[6] == "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"", lines[7].hasPrefix("K:") else {
            throw YuE2Error.invalidRequest("Expected a native two-voice YuE2/SheetSage2 ABC score.")
        }
        let meter = try parseMeter(String(lines[2].dropFirst(2)))
        let key = String(lines[7].dropFirst(2))
        _ = try accidentals(key)
        var voices = [Voice(numerator: meter.0, denominator: meter.1, key: key),
                      Voice(numerator: meter.0, denominator: meter.1, key: key)]
        var sections: [Int: String] = [:]
        var cursor = 8
        while cursor < lines.count {
            while cursor < lines.count, lines[cursor].hasPrefix("% ") {
                sections[voices[0].time] = String(lines[cursor].dropFirst(2))
                cursor += 1
            }
            for (index, name) in ["Vocal", "Ins"].enumerated() {
                guard cursor < lines.count, lines[cursor] == "V: \(name)" else {
                    throw YuE2Error.invalidRequest("Expected alternating Vocal and Ins music groups.")
                }
                cursor += 1
                var fields = Set<String>()
                while cursor < lines.count, lines[cursor].hasPrefix("M:") || lines[cursor].hasPrefix("K:") {
                    let field = String(lines[cursor].prefix(2))
                    guard fields.insert(field).inserted else { throw YuE2Error.invalidRequest("Duplicate score field.") }
                    let value = String(lines[cursor].dropFirst(2))
                    if field == "M:" {
                        let meter = try parseMeter(value)
                        voices[index].numerator = meter.0
                        voices[index].denominator = meter.1
                    } else {
                        _ = try accidentals(value)
                        voices[index].key = value
                    }
                    cursor += 1
                }
                guard cursor < lines.count, lines[cursor].hasSuffix("|") else {
                    throw YuE2Error.invalidRequest("Music lines must end with a plain barline.")
                }
                let measures = lines[cursor].dropLast().components(separatedBy: "|")
                cursor += 1
                for measure in measures {
                    let body = measure.trimmingCharacters(in: .whitespaces)
                    if ["Z", "Z2", "Z3", "Z4"].contains(body) {
                        for _ in 0..<(Int(body.dropFirst()) ?? 1) {
                            try parseBar("Z", voice: &voices[index], unit: 1024 / denominator)
                        }
                    } else { try parseBar(body, voice: &voices[index], unit: 1024 / denominator) }
                }
            }
            guard voices[0].bars == voices[1].bars else {
                throw YuE2Error.invalidRequest("Voice meter, key and timing grids differ.")
            }
        }
        guard voices.allSatisfy({ $0.pending == nil }), voices[1].chords.isEmpty else {
            throw YuE2Error.invalidRequest("Unresolved tie or chord outside the Vocal voice.")
        }
        return YuE2Score(bpm: bpm, bars: voices[0].bars, vocal: voices[0].notes,
                         instrumental: voices[1].notes, chords: voices[0].chords, sections: sections)
    }

    private static func parseMeter(_ value: String) throws -> (Int, Int) {
        let parts = value.split(separator: "/")
        guard parts.count == 2, let numerator = Int(parts[0]), let denominator = Int(parts[1]),
              (1...1024).contains(numerator), (1...1024).contains(denominator), denominator.nonzeroBitCount == 1 else {
            throw YuE2Error.invalidRequest("Unsupported score meter \(value).")
        }
        return (numerator, denominator)
    }

    private static func accidentals(_ key: String) throws -> [Character: Int] {
        guard let index = majorKeys.firstIndex(of: key) ?? minorKeys.firstIndex(of: key) else {
            throw YuE2Error.invalidRequest("Unsupported score key \(key).")
        }
        let count = index - 7
        return Dictionary(uniqueKeysWithValues: (count > 0 ? "FCGDAEB" : "BEADGCF").prefix(abs(count)).map {
            ($0, count > 0 ? 1 : -1)
        })
    }

    private static func parseBar(_ body: String, voice: inout Voice, unit: Int) throws {
        let bar = YuE2Score.Bar(onset: voice.time, numerator: voice.numerator, denominator: voice.denominator, key: voice.key)
        let end = voice.time + bar.duration
        if body == "Z" {
            guard voice.pending == nil else { throw YuE2Error.invalidRequest("A tie enters a full-measure rest.") }
            voice.time = end
        } else {
            let regex = try NSRegularExpression(pattern: tokenPattern)
            let text = body as NSString
            var position = 0
            var local: [Character: Int] = [:]
            while position < text.length {
                if let scalar = UnicodeScalar(text.character(at: position)), CharacterSet.whitespaces.contains(scalar) {
                    position += 1
                    continue
                }
                guard voice.time < end, let match = regex.firstMatch(in: body, options: .anchored,
                    range: NSRange(location: position, length: text.length - position)) else {
                    throw YuE2Error.invalidRequest("Unsupported ABC token or event past measure end.")
                }
                func group(_ index: Int) -> String {
                    let range = match.range(at: index)
                    return range.location == NSNotFound ? "" : text.substring(with: range)
                }
                position = match.range.upperBound
                if match.range(at: 1).location != NSNotFound {
                    if let last = voice.chords.last, last.onset == voice.time {
                        guard last.symbol == group(1) else { throw YuE2Error.invalidRequest("Conflicting chords at the same score position.") }
                    } else { voice.chords.append(.init(onset: voice.time, symbol: group(1))) }
                    continue
                }
                if match.range(at: 2).location != NSNotFound {
                    guard voice.time == bar.onset else {
                        throw YuE2Error.invalidRequest("Normalize inline key changes inside a bar before converting.")
                    }
                    _ = try accidentals(group(2))
                    voice.key = group(2)
                    local = [:]
                    continue
                }
                guard let note = group(4).first else { throw YuE2Error.invalidRequest("Missing ABC note.") }
                let units = group(6).isEmpty ? 1 : Int(group(6)) ?? 0
                guard [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48].contains(units) else {
                    throw YuE2Error.invalidRequest("Unsupported score duration.")
                }
                let duration = units * unit
                guard voice.time + duration <= end else {
                    throw YuE2Error.invalidRequest("Unsupported duration or note beyond measure boundary.")
                }
                if note == "z" {
                    guard group(3).isEmpty, group(5).isEmpty, group(7).isEmpty, voice.pending == nil else {
                        throw YuE2Error.invalidRequest("Invalid rest or tie entering a rest.")
                    }
                } else {
                    let letter = Character(String(note).uppercased())
                    let natural: [Character: Int] = ["C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11]
                    let octave = group(5)
                    guard !(octave.contains(",") && octave.contains("'")), octave.count <= 10,
                          let semitone = natural[letter] else { throw YuE2Error.invalidRequest("Invalid note spelling.") }
                    let written = 60 + semitone + (note.isLowercase ? 12 : 0)
                        + 12 * (octave.filter { $0 == "'" }.count - octave.filter { $0 == "," }.count)
                    let key = try accidentals(voice.key)
                    let explicit: [String: Int] = ["=": 0, "_": -1, "__": -2, "^": 1, "^^": 2]
                    if let alteration = explicit[group(3)] { local[letter] = alteration }
                    var pitch = written + (local[letter] ?? key[letter] ?? 0)
                    if let pending = voice.pending {
                        if group(3).isEmpty, written == pending.written { pitch = pending.pitch }
                        guard pitch == pending.pitch else { throw YuE2Error.invalidRequest("A tie changes pitch.") }
                        voice.notes[voice.notes.count - 1].duration += duration
                    } else {
                        guard (0...127).contains(pitch) else { throw YuE2Error.invalidRequest("Pitch outside MIDI range.") }
                        voice.notes.append(.init(onset: voice.time, pitch: pitch, duration: duration))
                    }
                    voice.pending = group(7).isEmpty ? nil : (pitch, written)
                }
                voice.time += duration
            }
        }
        guard voice.time == end else { throw YuE2Error.invalidRequest("Measure duration does not match its meter.") }
        voice.bars.append(.init(onset: bar.onset, numerator: bar.numerator, denominator: bar.denominator, key: voice.key))
    }
}
