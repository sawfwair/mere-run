import Foundation

/// Channel parsing for the Qwen and Gemma native output grammars. Tool bodies
/// remain private until their complete structured parse has passed validation.
struct NativeChatContentStream {
    private var pending = ""
    private var reasoning: Bool
    private var inTools = false
    private let parseTools: Bool
    private(set) var events: [ChatGenerationEvent] = []

    init(initialReasoning: Bool, parseTools: Bool = true) { reasoning = initialReasoning; self.parseTools = parseTools }

    mutating func append(_ piece: String, final: Bool = false) -> [ChatGenerationEvent] {
        pending += piece
        var emitted: [ChatGenerationEvent] = []
        let starts = parseTools ? ["<tool_call>", "<|tool_call>"] : []
        let markers = ["<think>", "</think>", "<|channel>", "<channel|>", "<|im_end|>", "<turn|>"] + starts
        while !pending.isEmpty {
            // These native grammars end assistant prose when the tool envelope
            // starts. The strict final parser rejects stray trailing content.
            if inTools { pending = ""; break }
            if pending.hasPrefix("<|channel>") {
                guard let newline = pending.firstIndex(of: "\n") else {
                    if final { pending = "" }
                    break
                }
                let label = pending.dropFirst("<|channel>".count).prefix(upTo: newline)
                reasoning = label.trimmingCharacters(in: .whitespacesAndNewlines) == "thought"
                pending.removeSubrange(...newline)
                continue
            }
            if let marker = markers.first(where: { pending.hasPrefix($0) }) {
                pending.removeFirst(marker.count)
                if starts.contains(marker) { inTools = true }
                else if marker == "<think>" { reasoning = true }
                else if marker == "</think>" || marker == "<channel|>" { reasoning = false }
                continue
            }
            if !final, markers.contains(where: { $0.hasPrefix(pending) }) { break }
            let character = String(pending.removeFirst())
            let event: ChatGenerationEvent = reasoning ? .reasoning(character) : .text(character)
            Self.coalesce(event, into: &emitted)
        }
        for event in emitted { Self.coalesce(event, into: &events) }
        return emitted
    }

    private static func coalesce(_ event: ChatGenerationEvent, into events: inout [ChatGenerationEvent]) {
        switch (events.last, event) {
        case (.text(let previous), .text(let next)): events[events.count - 1] = .text(previous + next)
        case (.reasoning(let previous), .reasoning(let next)): events[events.count - 1] = .reasoning(previous + next)
        default: events.append(event)
        }
    }
}

/// A request-local observer; batched decoders may deliver callbacks from a
/// different task. Cumulative decoding preserves split Unicode scalars.
final class NativeChatOutputObserver: @unchecked Sendable {
    private let lock = NSLock()
    private var decoded = ""
    private var parser: NativeChatContentStream

    init(initialReasoning: Bool, parseTools: Bool = true) { parser = .init(initialReasoning: initialReasoning, parseTools: parseTools) }

    func observe(_ progress: ChatProgress, handler: (@Sendable (ChatProgress) -> Void)?) {
        handler?(progress)
        guard let text = progress.decodedText else { return }
        let events = lock.withLock { append(text, final: false) }
        for event in events { handler?(.init(stage: .generating, event: event)) }
    }

    func finish(_ text: String, handler: (@Sendable (ChatProgress) -> Void)?) -> [ChatGenerationEvent] {
        let result = lock.withLock { () -> ([ChatGenerationEvent], [ChatGenerationEvent]) in
            let emitted = append(text, final: true)
            return (emitted, parser.events)
        }
        for event in result.0 { handler?(.init(stage: .generating, event: event)) }
        return result.1
    }

    private func append(_ text: String, final: Bool) -> [ChatGenerationEvent] {
        var stable = text
        if !final { while stable.last == "\u{FFFD}" { stable.removeLast() } }
        guard stable.utf8.starts(with: decoded.utf8) else { return [] }
        let delta = String(decoding: stable.utf8.dropFirst(decoded.utf8.count), as: UTF8.self)
        decoded = stable
        return parser.append(delta, final: final)
    }
}
