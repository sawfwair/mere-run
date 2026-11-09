import Foundation
import MLX
import MereRunSheetSageModel

public struct SheetSage2Window: Codable, Equatable, Sendable {
    public let start: Double
    public let end: Double
    public let acceptStart: Double
    public let acceptEnd: Double

    public static func plan(duration: Double, overlap: Double = 200, lookahead: Double = 100) throws -> [Self] {
        guard duration.isFinite, duration > 0, overlap.isFinite, lookahead.isFinite,
              0 <= lookahead, lookahead <= overlap, overlap < 300 else {
            throw SheetSage2Error.invalid("Require positive audio duration and 0 ≤ lookahead ≤ overlap < 300 seconds.")
        }
        var result: [Self] = []
        var start = 0.0
        var accepted = 0.0
        while true {
            let last = start + 300 >= duration - 1e-6
            let end = last ? duration : start + 300 - lookahead
            result.append(Self(start: start, end: min(duration, start + 300), acceptStart: accepted, acceptEnd: end))
            if last { return result }
            accepted = end
            start = min(start + 300 - overlap, duration - 300)
        }
    }
}

public struct SheetSage2TimedEvent: Codable, Sendable {
    public let time: Double
    public let globalSubbeat: Int
    public let tokens: [Int]
}

public struct SheetSage2Transcription: Codable, Sendable {
    public let modelRevision: String
    public let parentRevision: String
    public let duration: Double
    public let windows: [SheetSage2Window]
    public let events: [SheetSage2TimedEvent]
    public let windowTokens: [[Int]]
    public let truncatedWindows: [Int]
    public let abc: String?
    public let notationError: String?
    public let notationDiagnostics: [String]
}

public actor SheetSage2Transcriber {
    private let resources: SheetSage2Resources
    private let streams = Stream.Context()
    private var transcribing = false
    public init(resources: SheetSage2Resources) { self.resources = resources }

    public func transcribe(samples: [Float], melodyOnly: Bool = false, overlap: Double = 200,
                           lookahead: Double = 100, progress: (@Sendable (String, Int, Int) -> Void)? = nil) async throws -> SheetSage2Transcription {
        guard samples.count >= 1025, samples.allSatisfy(\.isFinite), !transcribing else {
            throw SheetSage2Error.invalid("Expected finite mono 24 kHz audio and an idle transcriber.")
        }
        let duration = Double(samples.count) / 24000
        let windows = try SheetSage2Window.plan(duration: duration, overlap: overlap, lookahead: lookahead)
        transcribing = true
        defer { transcribing = false }
        return try await Stream.withDefaultStream(streams) {
            defer { streams.synchronize() }
            let model = try resources.load()
            var stitched: [SheetSage2TimedEvent] = []
            var windowTokens: [[Int]] = []
            var truncated: [Int] = []
            for (index, window) in windows.enumerated() {
                try Task.checkCancellation()
                let start = Int((window.start * 24000).rounded())
                let end = min(samples.count, Int((window.end * 24000).rounded()))
                let memory = try model.encode(MLXArray(Array(samples[start..<end]))) {
                    progress?("encoding", $0, $1)
                }
                let (prefix, base) = Self.overlapPrefix(stitched, window: window)
                guard prefix.count < model.configuration.maxOutputSequence - 128 else {
                    throw SheetSage2Error.invalid("Overlap prefix fills the decoder context; reduce overlap.")
                }
                var tokens = prefix
                var grammar = SheetSage2Vocabulary.Grammar()
                for token in prefix.dropFirst(SheetSage2Vocabulary.prefix.count) { try grammar.update(token) }
                let cache = SheetSage2DecoderCache()
                var input = prefix
                while tokens.count < model.configuration.maxOutputSequence {
                    try Task.checkCancellation()
                    let logits = try model.logits(input, memory: memory, cache: cache).asType(.float32)
                    let masked = which(grammar.mask(), logits, MLXArray(-Float.infinity))
                    guard max(masked).item(Float.self).isFinite else { throw SheetSage2Error.invalid("No finite selectable transcription logits.") }
                    let next = argMax(masked).item(Int.self)
                    tokens.append(next)
                    try grammar.update(next)
                    if MLX.Memory.cacheMemory > 512 * 1024 * 1024 { MLX.Memory.clearCache() }
                    if next == 2 { break }
                    if SheetSage2Vocabulary.time.contains(next), Double(next - SheetSage2Vocabulary.time.lowerBound) / 100
                        >= window.acceptEnd - window.start { tokens.append(2); break }
                    input = [next]
                    if tokens.count % 64 == 0 { progress?("decoding", tokens.count, model.configuration.maxOutputSequence) }
                }
                if tokens.last != 2 { truncated.append(index); tokens.append(2) }
                let events = try SheetSage2Vocabulary.events(tokens)
                for event in events {
                    let time = window.start + Self.eventTime(event.subbeat, events: events, duration: window.end - window.start)
                    if time >= window.acceptStart - 1e-4 && time < min(duration, window.acceptEnd) - 1e-4 {
                        stitched.append(.init(time: time, globalSubbeat: base + event.subbeat, tokens: event.tokens))
                    }
                }
                windowTokens.append(tokens)
                progress?("windows", index + 1, windows.count)
            }
            stitched.sort { $0.time == $1.time ? $0.globalSubbeat < $1.globalSubbeat : $0.time < $1.time }
            let notation: SheetSage2Notation.Result?
            let notationError: String?
            do { notation = try SheetSage2Notation.render(events: stitched, duration: duration, melodyOnly: melodyOnly); notationError = nil }
            catch { notation = nil; notationError = error.localizedDescription }
            return SheetSage2Transcription(modelRevision: SheetSage2Resources.revision, parentRevision: SheetSage2Resources.parentRevision,
                duration: duration, windows: windows, events: stitched, windowTokens: windowTokens, truncatedWindows: truncated,
                abc: notation?.abc, notationError: notationError, notationDiagnostics: notation?.diagnostics ?? [])
        }
    }

    static func eventTime(_ subbeat: Int, events: [SheetSage2Vocabulary.Event], duration: Double) -> Double {
        var anchors: [Int: Double] = [:]
        for event in events { if let time = event.timestamp { anchors[event.subbeat] = time } }
        let points = anchors.sorted { $0.key < $1.key }
        guard let first = points.first, let last = points.last else { return min(duration, max(0, Double(subbeat) * 0.125)) }
        let differences = zip(points, points.dropFirst()).map { ($1.value - $0.value) / Double(max(1, $1.key - $0.key)) }.sorted()
        let step = differences.isEmpty ? 0.125 : differences[differences.count / 2]
        let slope = step.isFinite && step > 0 ? step : 0.125
        if subbeat <= first.key { return min(duration, max(0, first.value + Double(subbeat - first.key) * slope)) }
        if subbeat >= last.key { return min(duration, max(0, last.value + Double(subbeat - last.key) * slope)) }
        for (a, b) in zip(points, points.dropFirst()) where a.key <= subbeat && subbeat <= b.key {
            return a.value + Double(subbeat - a.key) / Double(b.key - a.key) * (b.value - a.value)
        }
        return last.value
    }

    static func overlapPrefix(_ events: [SheetSage2TimedEvent], window: SheetSage2Window) -> ([Int], Int) {
        let vocabulary = SheetSage2Vocabulary.self
        var selected = events.filter { $0.time >= window.start - 1e-4 && $0.time < window.acceptStart - 1e-4 }
            .sorted { $0.globalSubbeat < $1.globalSubbeat }
        guard let first = selected.firstIndex(where: { event in event.tokens.contains {
            vocabulary.time.contains($0) || vocabulary.meter.contains($0) || vocabulary.eighth.contains($0)
        } }) else { return (vocabulary.prefix, 0) }
        selected = Array(selected[first...])
        guard let firstEvent = selected.first else { return (vocabulary.prefix, 0) }
        let base = firstEvent.globalSubbeat
        var output = vocabulary.prefix
        var previous = 0
        for (index, event) in selected.enumerated() {
            let step = max(0, event.globalSubbeat - base)
            var shift = step - previous
            while shift > 256 { output.append(vocabulary.shift.lowerBound + 256); shift -= 256 }
            output.append(vocabulary.shift.lowerBound + shift)
            previous = step
            var payload = event.tokens.filter { !vocabulary.time.contains($0) }
            if index == 0 {
                var contextFields = [vocabulary.structure, vocabulary.key, vocabulary.chord]
                if payload.contains(where: vocabulary.eighth.contains) { contextFields.insert(vocabulary.meter, at: 0) }
                for range in contextFields {
                    if !payload.contains(where: range.contains), let context = events.last(where: {
                        $0.time <= firstEvent.time + 1e-6 && $0.tokens.contains(where: range.contains)
                    })?.tokens.first(where: range.contains) { payload.append(context) }
                }
                let melody = payload.filter { $0 >= vocabulary.pitch.lowerBound }
                payload = payload.filter { $0 < vocabulary.pitch.lowerBound }.sorted() + melody
            }
            if event.tokens.contains(where: vocabulary.time.contains) {
                output.append(vocabulary.time.lowerBound + min(29999, max(0, Int(((event.time - window.start) * 100).rounded()))))
            }
            output += payload
        }
        return (output, base)
    }
}
