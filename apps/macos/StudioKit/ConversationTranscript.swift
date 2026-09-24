import Foundation

/// Renders a chat/code conversation into the single `--prompt` string the CLI accepts.
///
/// The CLI is stateless per invocation, so the app serializes history: prior turns become a
/// labeled dialogue and the latest user message ends the prompt with no trailing `Assistant:`
/// cue (the model's chat template adds that). When the history exceeds the character budget the
/// OLDEST messages are dropped from what is *sent* — they stay persisted in the thread; the drop
/// is reported (`droppedCount`/`includedMessageIDs`) so the UI can surface it rather than
/// silently truncating.
package enum ConversationTranscript {
    package struct Rendered: Equatable {
        package let prompt: String
        package let includedMessageIDs: Set<UUID>
        package let droppedCount: Int
        package let approxChars: Int
    }

    /// The character budget when nothing reports the model's context size (~4 chars/token, so
    /// roughly 12k tokens of history). The system prompt rides in `--system` and is reserved
    /// out of the history room.
    package static let defaultBudgetChars = 48_000
    /// Characters of history assumed per context token when sizing from a real context window.
    /// Deliberately below the English average so code-heavy threads still fit.
    package static let charsPerContextToken = 3
    /// The smallest budget a derived context can shrink to; the latest turn is always sent
    /// regardless, so this only bounds how much history rides along.
    package static let minimumBudgetChars = 4_000

    /// The history budget for a model with `contextTokens` of context, keeping `maxOutputTokens`
    /// free for the reply. nil or a non-positive context keeps `defaultBudgetChars`, so a model
    /// the inventory says nothing about behaves exactly as before.
    package static func budgetChars(contextTokens: Int?, maxOutputTokens: Int) -> Int {
        guard let contextTokens, contextTokens > 0 else { return defaultBudgetChars }
        let historyTokens = contextTokens - max(0, maxOutputTokens)
        return max(minimumBudgetChars, historyTokens * charsPerContextToken)
    }

    /// The context size the next turn will run with: an explicit `--context-size` in the draft
    /// wins, else what the model inventory reports for the model, else nothing (fixed budget).
    package static func contextTokens(
        requestedContextSize: Int,
        model: String,
        inventory: [StudioModelInventoryRow]
    ) -> Int? {
        if requestedContextSize > 0 { return requestedContextSize }
        let identity = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !identity.isEmpty,
              let window = inventory.first(where: { $0.id == identity })?.contextWindow,
              window > 0 else { return nil }
        return window
    }

    package static func render(
        messages: [StudioMessage],
        systemPrompt: String? = nil,
        budgetChars: Int = defaultBudgetChars
    ) -> Rendered {
        let reserve = systemPrompt?.count ?? 0
        let historyBudget = max(0, budgetChars - reserve)

        // A failed assistant turn produced no valid reply — it stays visible in the thread but is
        // never replayed into the prompt (it would otherwise inject an error/reasoning as context).
        let usable = messages.filter { !$0.failed }

        // Always keep the latest message; walk backward, including older ones until the next one
        // would exceed the budget.
        var included: [StudioMessage] = []
        var used = 0
        for message in usable.reversed() {
            let cost = renderedCost(message)
            if included.isEmpty || used + cost <= historyBudget {
                included.append(message)
                used += cost
            } else {
                break
            }
        }
        included.reverse()

        let prompt = format(included)
        return Rendered(
            prompt: prompt,
            includedMessageIDs: Set(included.map(\.id)),
            droppedCount: usable.count - included.count,
            approxChars: prompt.count + reserve
        )
    }

    /// A single user message renders verbatim, so the first turn is byte-identical to a
    /// single-shot run; multi-turn windows render as an oldest→newest labeled dialogue.
    package static func format(_ messages: [StudioMessage]) -> String {
        if messages.count == 1, let only = messages.first, only.role == .user {
            return only.content
        }
        return messages.map { message in
            switch message.role {
            case .user: return "User: \(message.content)"
            case .assistant: return "Assistant: \(message.content)"
            }
        }.joined(separator: "\n\n")
    }

    private static func renderedCost(_ message: StudioMessage) -> Int {
        // Content plus an approximation of the "User: "/"Assistant: " label and separator.
        message.content.count + 12
    }

    /// An assistant reply split into what the model said and what it thought first. The answer is
    /// what the thread stores in `StudioMessage.content` and replays; the reasoning is kept beside
    /// it for display only.
    package struct Reply: Equatable {
        /// The reply with every reasoning block removed, trimmed.
        package let answer: String
        /// The reasoning blocks' text, a blank line between blocks; nil when the reply had none.
        package let reasoning: String?
        /// True while a streaming reply is still inside an unclosed reasoning block.
        package let isThinking: Bool

        package init(answer: String, reasoning: String?, isThinking: Bool) {
            self.answer = answer
            self.reasoning = reasoning
            self.isThinking = isThinking
        }

        /// The same reply with its reasoning dropped, for a turn that runs with thinking hidden.
        package var hidingReasoning: Reply {
            Reply(answer: answer, reasoning: nil, isThinking: false)
        }
    }

    /// Splits model reasoning out of an assistant reply. With `--stream` the CLI emits the model's
    /// reasoning markup inline (it only strips it on the non-stream path), so the app must separate
    /// it before storing/replaying — otherwise reasoning leaks into the next turn's prompt.
    ///
    /// The markers are the pairs the CLI's own hide paths strip (`ReasoningMarkers.all`). Complete
    /// blocks always move to `reasoning`, as does the text before a leading orphan `</think>` (some
    /// models pre-fill the opening tag and emit only the close). A trailing UNCLOSED `<think>` block
    /// is only split off while `streaming` — that is reasoning still in progress, and `isThinking`
    /// says so. At finalize it stays in the answer: a completed reply's leftover `<think>` is almost
    /// certainly literal text (e.g. a code reply that discusses the tag), and truncating it would
    /// lose real content. A channel token (`<|channel>thought`, `<|content_thinking|>`) never occurs
    /// in literal text, so an unclosed channel is reasoning even at finalize — a reply cut off by
    /// its token budget mid-thought has no answer, as the CLI's own hide path reads it.
    package static func splitThinking(_ text: String, streaming: Bool = false) -> Reply {
        // One forward pass: this runs over the whole accumulated reply every few chunks.
        let text = streaming ? withoutTrailingPartialMarker(text) : text
        var answer = ""
        var blocks: [String] = []
        var isThinking = false
        var cursor = text.startIndex
        while cursor < text.endIndex {
            guard let (markers, range, isClose) = nextMarker(in: text, from: cursor) else {
                answer += text[cursor...]
                break
            }
            if isClose {
                // A close with no open before it: for `</think>` the model pre-filled the opening
                // tag, so everything up to here was reasoning; a channel's stray close is markup.
                if markers.prefilled {
                    blocks.append(answer + String(text[cursor..<range.lowerBound]))
                    answer = ""
                } else {
                    answer += text[cursor..<range.lowerBound]
                }
                cursor = range.upperBound
            } else {
                answer += text[cursor..<range.lowerBound]
                let body = range.upperBound..<text.endIndex
                if let close = text.range(of: markers.close, range: body) {
                    blocks.append(String(text[range.upperBound..<close.lowerBound]))
                    cursor = close.upperBound
                } else if streaming || markers.isChannel {
                    blocks.append(String(text[body]))
                    isThinking = streaming
                    cursor = text.endIndex
                } else {
                    answer += text[range.lowerBound...]
                    cursor = text.endIndex
                }
            }
        }
        let reasoning = blocks
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
        return Reply(
            answer: ReasoningMarkers.withoutChannelTokens(answer).trimmingCharacters(in: .whitespacesAndNewlines),
            reasoning: reasoning.isEmpty ? nil : reasoning,
            isThinking: isThinking
        )
    }

    /// The reply without its reasoning: `splitThinking(_:streaming:)` keeping only the answer.
    package static func stripThinkTags(_ text: String, streaming: Bool = false) -> String {
        splitThinking(text, streaming: streaming).answer
    }

    /// One reasoning block's markers, as a model family emits them and the CLI strips them.
    package struct ReasoningMarkers: Equatable, Sendable {
        package let open: String
        package let close: String
        /// Some models pre-fill `open` in the prompt and emit only `close`, so a leading orphan
        /// close means everything before it was reasoning.
        package let prefilled: Bool
        /// A control token the tokenizer reserves, which never occurs in literal text.
        package let isChannel: Bool

        /// Every pair the CLI strips on its own hide path, each cited to the code that strips it.
        package static let all: [ReasoningMarkers] = [
            // `ChatReasoningMarkup.splitThinkBlocks` in Sources/MereRunCore/Generation.swift, the
            // shared split for Qwen-style models; `Gemma4Generator+Policies.cleanedResponse` and
            // the benchmark commands strip the same pair.
            ReasoningMarkers(open: "<think>", close: "</think>", prefilled: true, isChannel: false),
            // Gemma 4's thought channel: `Gemma4Generator+Policies.cleanedResponse` in
            // Sources/MereRunCore/Gemma4/ strips `<|channel>thought … <channel|>` and, before the
            // answer, a bare `<|channel>final`.
            ReasoningMarkers(open: "<|channel>thought", close: "<channel|>", prefilled: false, isChannel: true),
            // Inkling's thinking channel: `InklingOutputParser.parse` in
            // Sources/MereRunCore/Inkling/InklingTokenizerAndTemplate.swift reads
            // `<|content_thinking|> … <|end_message|>` as reasoning and `<|content_text|> …
            // <|end_message|>` as the answer, dropping the other message tokens.
            ReasoningMarkers(open: "<|content_thinking|>", close: "<|end_message|>", prefilled: false, isChannel: true),
        ]

        /// The control tokens that frame an answer without being part of it: Gemma 4's
        /// `<|channel>final` and its like, and every Inkling message token — the ones
        /// `InklingOutputParser.stripControlTokens` removes, including the tool-call frame
        /// (`<|content_invoke_tool_json|>`) whose JSON the CLI turns into a tool call.
        static let channelTokens = [
            "<|message_model|>", "<|message_user|>", "<|message_system|>", "<|message_tool|>",
            "<|content_text|>", "<|content_xml|>", "<|content_invoke_tool_json|>", "<|content_model_end_sampling|>",
        ]

        /// A stream can end in the first bytes of a marker; these are what `withoutTrailingPartialMarker`
        /// holds back.
        static let heldBack = all.flatMap { [$0.open, $0.close] } + channelTokens + ["<|channel>final"]

        /// The text with every channel token removed, and with Inkling's tool-call frames
        /// (`<|content_invoke_tool_json|>{…}<|end_message|>`) removed whole: their JSON is the
        /// CLI's tool call, never prose.
        static func withoutChannelTokens(_ text: String) -> String {
            var text = text.replacingOccurrences(
                of: #"<\|content_invoke_tool_json\|>.*?(<\|end_message\|>|\z)"#, with: "", options: [.regularExpression]
            )
            text = text.replacingOccurrences(of: #"<\|channel>[a-z_]+\s*"#, with: "", options: .regularExpression)
            return text.replacingOccurrences(of: #"<\|[a-z_]+\|>"#, with: "", options: .regularExpression)
        }
    }

    /// The earliest marker at or after `cursor`: its pair, where it is, and whether it is the close.
    /// An open marker wins a tie with a close that starts at the same place.
    private static func nextMarker(in text: String, from cursor: String.Index) -> (ReasoningMarkers, Range<String.Index>, Bool)? {
        let rest = cursor..<text.endIndex
        var best: (ReasoningMarkers, Range<String.Index>, Bool)?
        for markers in ReasoningMarkers.all {
            for (marker, isClose) in [(markers.open, false), (markers.close, true)] {
                guard let range = text.range(of: marker, range: rest) else { continue }
                if let current = best, current.1.lowerBound < range.lowerBound
                    || (current.1.lowerBound == range.lowerBound && !current.2) { continue }
                best = (markers, range, isClose)
            }
        }
        return best
    }

    /// A marker can arrive split across chunks. While streaming, a trailing fragment of one ("<",
    /// "</thin", "<|chan") is held back rather than shown literally until the rest lands.
    private static func withoutTrailingPartialMarker(_ text: String) -> String {
        guard let start = text.lastIndex(of: "<") else { return text }
        let fragment = text[start...]
        let isFragment = ReasoningMarkers.heldBack.contains { fragment.count < $0.count && $0.hasPrefix(fragment) }
        return isFragment ? String(text[..<start]) : text
    }

    /// The decode throughput from the CLI's `--stats` line
    /// (`time=… tokens=… decode_tps=41.20 e2e_tps=…`), scanning the run's log from the end so
    /// the turn's own line wins over anything echoed earlier. nil when no line reports it.
    package static func decodeTokensPerSecond(in logLines: [String]) -> Double? {
        for line in logLines.reversed() {
            guard let range = line.range(of: "decode_tps=") else { continue }
            let value = line[range.upperBound...].prefix { $0.isNumber || $0 == "." }
            if let parsed = Double(value), parsed > 0 { return parsed }
        }
        return nil
    }
}
