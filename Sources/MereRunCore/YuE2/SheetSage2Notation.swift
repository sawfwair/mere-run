import Foundation
import MereRunSheetSageModel

enum SheetSage2Notation {
    struct Result { let abc: String; let diagnostics: [String] }

    static func render(events: [SheetSage2TimedEvent], duration: Double, melodyOnly: Bool) throws -> Result {
        let vocabulary = SheetSage2Vocabulary.self
        guard !events.isEmpty else { throw SheetSage2Error.invalid("Transcription produced no musical events.") }
        var meter = (numerator: 4, denominator: 4)
        var ticks = 0
        var step = 0
        var positions: [(step: Int, ticks: Int, denominator: Int)] = [(0, 0, 4)]
        var meters: [(ticks: Int, numerator: Int, denominator: Int)] = [(0, 4, 4)]
        var keys: [(ticks: Int, name: String)] = [(0, "C")]
        var sections: [Int: String] = [:]
        let ordered = events.sorted { $0.globalSubbeat < $1.globalSubbeat }
        for event in ordered {
            ticks += max(0, event.globalSubbeat - step) * 1024 / (meter.denominator * 4)
            step = event.globalSubbeat
            if let value = SheetSage2Vocabulary.Event(subbeat: step, tokens: event.tokens).meter {
                meter = value
                meters.append((ticks, value.numerator, value.denominator))
            }
            positions.append((step, ticks, meter.denominator))
            if let token = event.tokens.first(where: vocabulary.key.contains) {
                let index = token - vocabulary.key.lowerBound
                let names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
                let minor = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
                keys.append((ticks, (index >= 12 ? minor[index % 12] : names[index % 12]) + (index >= 12 ? "m" : "")))
            }
            if let token = event.tokens.first(where: vocabulary.structure.contains) {
                sections[ticks] = vocabulary.structures[token - vocabulary.structure.lowerBound]
            }
        }
        func position(_ subbeat: Int) -> Int {
            let base = positions.last(where: { $0.step <= subbeat }) ?? positions[0]
            return base.ticks + (subbeat - base.step) * 1024 / (base.denominator * 4)
        }
        var voices: [[YuE2Score.Note]] = [[], []]
        var chords: [YuE2Score.Chord] = []
        let chordNames = vocabulary.chords
        for event in ordered {
            let onset = position(event.globalSubbeat)
            let tokens = event.tokens
            for (index, token) in tokens.enumerated() where vocabulary.pitch.contains(token) {
                let id = token - vocabulary.pitch.lowerBound
                let durationToken = index + 1 < tokens.count && vocabulary.duration.contains(tokens[index + 1])
                    ? tokens[index + 1] - vocabulary.duration.lowerBound : 0
                let end = position(event.globalSubbeat + vocabulary.durations[durationToken])
                voices[id / 128].append(.init(onset: onset, pitch: id % 128, duration: end - onset))
            }
            if !melodyOnly, let token = tokens.first(where: vocabulary.chord.contains), token > vocabulary.chord.lowerBound {
                let symbol = try chordSymbol(chordNames[token - vocabulary.chord.lowerBound])
                if chords.last?.symbol != symbol { chords.append(.init(onset: onset, symbol: symbol)) }
            }
        }
        var diagnostics: [String] = []
        for track in voices.indices {
            voices[track].sort { $0.onset == $1.onset ? $0.pitch < $1.pitch : $0.onset < $1.onset }
            for index in voices[track].indices.dropLast() {
                let next = voices[track][index + 1].onset
                if voices[track][index].onset + voices[track][index].duration > next {
                    voices[track][index].duration = next - voices[track][index].onset
                    diagnostics.append("Clipped overlapping notation in voice \(track) at tick \(next); raw events are retained.")
                }
            }
            voices[track].removeAll { $0.duration <= 0 }
        }
        let anchors = events.filter { $0.tokens.contains(where: vocabulary.eighth.contains) }
        let tempos = zip(anchors, anchors.dropFirst()).compactMap { a, b -> Double? in
            let delta = b.time - a.time
            let quarters = Double(position(b.globalSubbeat) - position(a.globalSubbeat)) / 256
            return delta > 0 && quarters > 0 ? quarters * 60 / delta : nil
        }.sorted()
        guard !tempos.isEmpty else { throw SheetSage2Error.invalid("At least two decoded beat positions are required for ABC export.") }
        let bpm = max(1, Int(tempos[tempos.count / 2].rounded()))
        let lastNote = voices.flatMap { $0 }.map { $0.onset + $0.duration }.max() ?? 0
        let limit = max(lastNote, Int((duration * Double(bpm) / 60 * 256).rounded()))
        var bars: [YuE2Score.Bar] = []
        var cursor = 0
        while cursor < limit {
            let meter = meters.last(where: { $0.ticks <= cursor }) ?? meters[0]
            let key = keys.last(where: { $0.ticks <= cursor })?.name ?? "C"
            let bar = YuE2Score.Bar(onset: cursor, numerator: meter.numerator, denominator: meter.denominator, key: key)
            guard !meters.contains(where: { cursor < $0.ticks && $0.ticks < cursor + bar.duration
                && ($0.numerator != meter.numerator || $0.denominator != meter.denominator) }) else {
                throw SheetSage2Error.invalid("A meter change falls inside a measure; inspect the raw transcription before rendering.")
            }
            bars.append(bar)
            cursor += bar.duration
        }
        // Section labels are score annotations; quantize them to the containing measure.
        var barSections: [Int: String] = [:]
        for (time, section) in sections.sorted(by: { $0.key < $1.key }) {
            if let bar = bars.last(where: { $0.onset <= time }) { barSections[bar.onset] = section }
        }
        let score = YuE2Score(bpm: bpm, bars: bars, vocal: voices[0], instrumental: voices[1], chords: chords, sections: barSections)
        let abc = try score.serialized()
        _ = try YuE2Score(abc: abc)
        return Result(abc: abc, diagnostics: diagnostics)
    }

    private static func chordSymbol(_ label: String) throws -> String {
        let parts = label.split(separator: ":", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { throw SheetSage2Error.invalid("Malformed chord label.") }
        let quality = parts[1].split(separator: "/", maxSplits: 1).map(String.init)
        let names = ["maj": "", "min": "m", "dim": "dim", "aug": "aug", "maj7": "maj7", "min7": "m7",
            "7": "7", "hdim7": "m7b5", "dim7": "dim7", "minmaj7": "m(maj7)", "sus2": "sus2",
            "sus4": "sus4", "sus4(b7)": "7sus4", "maj6": "6", "min6": "m6"]
        guard let suffix = names[quality[0]], let root = SheetSage2Vocabulary.roots.firstIndex(of: parts[0]) else {
            throw SheetSage2Error.invalid("Unsupported chord label \(label).")
        }
        var result = parts[0] + suffix
        if quality.count == 2 {
            let intervals = ["2": 2, "3": 4, "b3": 3, "5": 7, "7": 11, "b7": 10]
            guard let interval = intervals[quality[1]] else { throw SheetSage2Error.invalid("Unsupported chord inversion.") }
            result += "/" + SheetSage2Vocabulary.roots[(root + interval) % 12]
        }
        return result
    }
}
