import Foundation

/// Incremental ATEM channel decoder. Withholds incomplete delimiters and tool
/// payloads; executable calls are validated separately after generation.
struct MuseGlimmerContentStream {
    private var pending = ""
    private var reasoning = false
    private var inTools = false
    private var toolRecipient = false
    private var stopped = false
    var didStop: Bool { stopped }
    private let stops: [String]
    private(set) var visible = ""
    private(set) var thinking = ""

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
            if let prefix = headerPrefixes.first(where: { pending.hasPrefix($0) }) {
                guard let end = pending.range(of: "<|message|>") else {
                    if final { pending = "" }
                    break
                }
                let recipient = String(pending[pending.index(pending.startIndex, offsetBy: prefix.count)..<end.lowerBound])
                pending.removeSubrange(..<end.upperBound)
                reasoning = recipient == "self"
                toolRecipient = recipient != "self" && recipient != "user"
                continue
            }
            if !final, headerPrefixes.contains(where: { $0.hasPrefix(pending) }) { break }
            if let marker = markers.first(where: { pending.hasPrefix($0) }) {
                pending.removeFirst(marker.count)
                if stops.contains(marker) { stopped = true; break }
                if marker == open { inTools = true }
                else if marker == close { inTools = false }
                continue
            }
            if !final, markers.contains(where: { $0.hasPrefix(pending) }) { break }
            if inTools || toolRecipient {
                pending.removeFirst()
                continue
            }
            // Ordinary output may start directly in the user channel.
            let character = String(pending.removeFirst())
            if reasoning { thinking += character }
            else { visible += character }
            if let last = events.last {
                switch (last, reasoning) {
                case (.text(let value), false): events[events.count - 1] = .text(value + character)
                case (.reasoning(let value), true): events[events.count - 1] = .reasoning(value + character)
                default: events.append(reasoning ? .reasoning(character) : .text(character))
                }
            } else { events.append(reasoning ? .reasoning(character) : .text(character)) }
        }
        return events
    }
}
