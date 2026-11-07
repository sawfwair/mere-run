import Foundation
import MereRunSheetSageModel

extension SheetSage2Transcription {
    /// Raw timed notes retain model overlaps; the ABC notation view records any clipping separately.
    public var notes: [MuScriptorNote] {
        let vocabulary = SheetSage2Vocabulary.self
        var anchors: [Int: Double] = [:]
        for event in events { anchors[event.globalSubbeat] = event.time }
        let points = anchors.sorted { $0.key < $1.key }
        let slopes = zip(points, points.dropFirst()).map { ($1.value - $0.value) / Double(max(1, $1.key - $0.key)) }.filter { $0 > 0 }.sorted()
        let slope = slopes.isEmpty ? 0.125 : slopes[slopes.count / 2]
        func time(_ step: Int) -> Double {
            guard let first = points.first, let last = points.last else { return 0 }
            if step <= first.key { return max(0, first.value + Double(step - first.key) * slope) }
            if step >= last.key { return min(duration, last.value + Double(step - last.key) * slope) }
            for (left, right) in zip(points, points.dropFirst()) where left.key <= step && step <= right.key {
                return left.value + Double(step - left.key) / Double(right.key - left.key) * (right.value - left.value)
            }
            return duration
        }
        return events.flatMap { event in
            event.tokens.enumerated().compactMap { index, token -> MuScriptorNote? in
                guard vocabulary.pitch.contains(token) else { return nil }
                let id = token - vocabulary.pitch.lowerBound
                let bin = index + 1 < event.tokens.count && vocabulary.duration.contains(event.tokens[index + 1])
                    ? event.tokens[index + 1] - vocabulary.duration.lowerBound : 0
                let end = min(duration, max(event.time + 0.04, time(event.globalSubbeat + vocabulary.durations[bin])))
                guard end > event.time else { return nil }
                return MuScriptorNote(instrument: id < 128 ? "Vocal" : "Ins", program: 0, pitch: id % 128,
                    onset: event.time, offset: end, isDrum: false)
            }
        }
    }

    public func symbolicArtifacts() throws -> [String: Data] {
        let notes = notes
        var output: [String: Data] = ["melody.mid": try MuScriptorMIDI.encode(notes: notes)]
        func row(_ note: MuScriptorNote, track: Bool) -> String {
            "\(note.onset)\t\(note.offset)\t\(note.pitch)" + (track ? "\t\(note.instrument == "Vocal" ? 0 : 1)" : "") + "\n"
        }
        output["melody_full.lab"] = Data(notes.map { row($0, track: true) }.joined().utf8)
        for (name, voice) in [("vocal", "Vocal"), ("instrumental", "Ins")] {
            let selected = notes.filter { $0.instrument == voice }
            output["melody_\(name).mid"] = try MuScriptorMIDI.encode(notes: selected)
            output["melody_\(name).lab"] = Data(selected.map { row($0, track: false) }.joined().utf8)
        }
        let vocabulary = SheetSage2Vocabulary.self
        let chordNames = vocabulary.chords
        for (name, range) in [("key", vocabulary.key), ("chord", vocabulary.chord), ("structure", vocabulary.structure)] {
            let labels: [(Double, String)] = events.compactMap { event in
                guard let token = event.tokens.first(where: range.contains) else { return nil }
                let index = token - range.lowerBound
                let label: String
                switch name {
                case "key": label = vocabulary.roots[index % 12] + (index >= 12 ? ":minor" : ":major")
                case "chord": label = chordNames[index]
                default: label = vocabulary.structures[index]
                }
                return (event.time, label)
            }
            let text = labels.enumerated().map { index, value in
                "\(value.0)\t\(index + 1 < labels.count ? labels[index + 1].0 : duration)\t\(value.1)\n"
            }.joined()
            output[name + ".lab"] = Data(text.utf8)
        }
        return output
    }
}
