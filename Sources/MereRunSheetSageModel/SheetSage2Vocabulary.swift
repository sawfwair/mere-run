import Foundation
import MLX

public enum SheetSage2Vocabulary {
    public static let prefix = [1, 4, 5, 6, 7, 9, 11, 3]
    public static let shift = 260..<517
    public static let time = 517..<30517
    public static let meter = 30517..<30709
    public static let eighth = 30709..<30965
    public static let structure = 30965..<30988
    public static let key = 30988..<31012
    public static let majminChord = 31012..<31037
    public static let chord = 31037..<31398
    public static let pitch = 31398..<31654
    public static let duration = 31654..<31678
    public static let durations = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    public static let structures = ["silence", "intro", "outro", "verse", "chorus", "bridge", "pre-chorus", "post-chorus",
        "interlude", "fade-out", "loop", "rap", "preshot", "irregular", "instrumental", "intro and verse",
        "pre-chorus and chorus", "verse and pre-chorus", "solo", "theme", "development", "variation", "pre-outro"]
    public static let roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    public static var chords: [String] {
        var output = ["N"]
        for quality in ["maj", "min", "dim", "aug", "maj7", "min7", "7", "hdim7", "dim7", "minmaj7", "sus2", "sus4", "sus4(b7)", "maj6", "min6"] {
            let inversions: [String]
            switch quality {
            case "maj": inversions = ["/2", "/3", "/5"]
            case "min": inversions = ["/2", "/b3", "/5"]
            case "maj7": inversions = ["/3", "/5", "/7"]
            case "min7": inversions = ["/b3", "/5", "/b7"]
            case "7": inversions = ["/3", "/5", "/b7"]
            default: inversions = []
            }
            for root in roots { for inversion in inversions + [""] { output.append("\(root):\(quality)\(inversion)") } }
        }
        return output
    }

    public struct Event: Codable, Equatable, Sendable {
        public var subbeat: Int
        public var tokens: [Int]
        public init(subbeat: Int, tokens: [Int]) { self.subbeat = subbeat; self.tokens = tokens }
        public var timestamp: Double? { tokens.first(where: time.contains).map { Double($0 - time.lowerBound) / 100 } }
        public var meter: (numerator: Int, denominator: Int)? {
            tokens.first(where: SheetSage2Vocabulary.meter.contains).map {
                let index = $0 - SheetSage2Vocabulary.meter.lowerBound
                return (index / 6 + 1, [1, 2, 4, 8, 16, 32][index % 6])
            }
        }
    }

    public static func events(_ tokens: [Int]) throws -> [Event] {
        guard tokens.starts(with: prefix), tokens.last == 2 else { throw SheetSage2Error.invalid("Invalid transcription prompt or missing EOS.") }
        var events: [Event] = []
        var step = 0
        var payload: [Int] = []
        var shifting = true
        for token in tokens.dropFirst(prefix.count).dropLast() {
            if shift.contains(token) {
                if !shifting { events.append(Event(subbeat: step, tokens: payload)); payload = [] }
                shifting = true
                step += token - shift.lowerBound
            } else {
                guard (time.lowerBound..<duration.upperBound).contains(token), !majminChord.contains(token) else {
                    throw SheetSage2Error.invalid("Unexpected event token \(token).")
                }
                shifting = false
                payload.append(token)
            }
        }
        if !payload.isEmpty { events.append(Event(subbeat: step, tokens: payload)) }
        return events
    }

    public struct Grammar {
        var shifting = true
        var shiftRun = 0
        var payload = 0
        var field = -1
        var incomplete: Int?
        public init() {}

        public func mask() -> MLXArray {
            var allowed = Array(repeating: false, count: 31678)
            func include(_ range: Range<Int>) { for token in range { allowed[token] = true } }
            if payload > 0 { allowed[2] = true }
            if (payload > 0 || shifting) && shiftRun < 4 { include(shift) }
            if incomplete == 1 { include(eighth); return MLXArray(allowed) }
            if incomplete == 5 { include(duration); include(pitch); return MLXArray(allowed) }
            if field < 0 { include(time) }
            if field < 1 { include(meter); include(eighth) }
            if field < 2 { include(structure) }
            if field < 3 { include(key) }
            if field < 4 { include(chord) }
            if field <= 5 { include(pitch) }
            return MLXArray(allowed)
        }

        public mutating func update(_ token: Int) throws {
            guard (0..<31678).contains(token), mask()[token].item(Bool.self) else { throw SheetSage2Error.invalid("Token violates transcription grammar.") }
            if token == 2 { return }
            if shift.contains(token) {
                if !shifting && payload > 0 { payload = 0; field = -1; incomplete = nil }
                shifting = true
                shiftRun += 1
                return
            }
            shifting = false
            shiftRun = 0
            payload += 1
            incomplete = nil
            if time.contains(token) { field = 0 }
            else if meter.contains(token) { field = 1; incomplete = 1 }
            else if eighth.contains(token) { field = 1 }
            else if structure.contains(token) { field = 2 }
            else if key.contains(token) { field = 3 }
            else if chord.contains(token) { field = 4 }
            else if pitch.contains(token) { field = 5; incomplete = 5 }
            else if duration.contains(token) { field = 5 }
        }
    }
}
