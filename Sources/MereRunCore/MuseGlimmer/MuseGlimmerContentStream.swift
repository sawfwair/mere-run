import Foundation

/// Incremental ATEM channel decoder. Withholds incomplete delimiters and tool
/// payloads; executable calls are validated separately after generation.
struct MuseGlimmerContentStream {
    private var pending = ""
    private var reasoning = false
    private var inTools = false
    private var toolRecipient = false
    private var atMessageBoundary = true
    private var stopped = false
    var didStop: Bool { stopped }
    private let stops: [String]
    private(set) var visible = ""
    private(set) var thinking = ""
    private(set) var toolSource = ""

    init(stops: [String] = []) { self.stops = stops.filter { !$0.isEmpty } }

    mutating func append(_ piece: String, final: Bool = false) -> [ChatGenerationEvent] {
        guard !stopped else { return [] }
        pending += piece
        var events: [ChatGenerationEvent] = []
        let headerPrefixes = ["<|start|>assistant to=", " to=", "to="]
        let endings = ["<|eom|>", "<|eot|>"]
        let open = "<atem:function_calls>"
        let close = "</atem:function_calls>"
        let markers = endings + [open, close] + stops
        while !pending.isEmpty {
            let eligibleHeaders: [String]
            if inTools { eligibleHeaders = [] }
            else if atMessageBoundary { eligibleHeaders = headerPrefixes }
            else { eligibleHeaders = [headerPrefixes[0]] }
            if let prefix = eligibleHeaders.first(where: { pending.hasPrefix($0) }) {
                guard let end = pending.range(of: "<|message|>") else {
                    if final { pending = "" }
                    break
                }
                let recipient = String(pending[pending.index(pending.startIndex, offsetBy: prefix.count)..<end.lowerBound])
                let header = String(pending[..<end.upperBound])
                pending.removeSubrange(..<end.upperBound)
                reasoning = recipient == "self"
                toolRecipient = recipient != "self" && recipient != "user"
                atMessageBoundary = false
                if !reasoning { toolSource += header }
                continue
            }
            if !final, eligibleHeaders.contains(where: { $0.hasPrefix(pending) }) { break }
            if let marker = markers.first(where: { pending.hasPrefix($0) }) {
                pending.removeFirst(marker.count)
                if stops.contains(marker) { stopped = true; break }
                if !reasoning { toolSource += marker }
                if endings.contains(marker) {
                    atMessageBoundary = true
                } else if reasoning {
                    // A tool example in private reasoning is data, not a call.
                    appendContent(marker, events: &events)
                } else if marker == open { inTools = true }
                else if marker == close { inTools = false }
                continue
            }
            if !final, markers.contains(where: { $0.hasPrefix(pending) }) { break }
            let character = String(pending.removeFirst())
            if !reasoning { toolSource += character }
            atMessageBoundary = atMessageBoundary && character.allSatisfy(\.isWhitespace)
            if inTools || toolRecipient {
                continue
            }
            // Ordinary output may start directly in the user channel.
            appendContent(character, events: &events)
        }
        return events
    }

    private mutating func appendContent(_ text: String, events: inout [ChatGenerationEvent]) {
        if reasoning { thinking += text } else { visible += text }
        if let last = events.last {
            switch (last, reasoning) {
            case (.text(let value), false): events[events.count - 1] = .text(value + text)
            case (.reasoning(let value), true): events[events.count - 1] = .reasoning(value + text)
            default: events.append(reasoning ? .reasoning(text) : .text(text))
            }
        } else { events.append(reasoning ? .reasoning(text) : .text(text)) }
    }
}
