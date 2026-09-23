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

    /// Splits model reasoning out of an assistant reply. With `--stream` the CLI emits
    /// `<think>…</think>` reasoning inline (it only strips it on the non-stream path), so the app
    /// must separate it before storing/replaying — otherwise reasoning leaks into the next turn's
    /// prompt.
    ///
    /// Complete blocks always move to `reasoning`, as does the text before a leading orphan
    /// `</think>` (some models pre-fill the opening tag and emit only the close). A trailing
    /// UNCLOSED block is only split off while `streaming` — that is reasoning still in progress,
    /// and `isThinking` says so. At finalize it stays in the answer: a completed reply's leftover
    /// `<think>` is almost certainly literal text (e.g. a code reply that discusses the tag), and
    /// truncating it would lose real content.
    package static func splitThinking(_ text: String, streaming: Bool = false) -> Reply {
        var answer = text
        var blocks: [String] = []
        while let block = answer.range(of: "<think>[\\s\\S]*?</think>", options: .regularExpression) {
            let inner = answer[block].dropFirst("<think>".count).dropLast("</think>".count)
            blocks.append(String(inner))
            answer.removeSubrange(block)
        }
        if !answer.contains("<think>"), let close = answer.range(of: "</think>") {
            blocks.insert(String(answer[..<close.lowerBound]), at: 0)
            answer = String(answer[close.upperBound...])
        }
        var isThinking = false
        if streaming, let open = answer.range(of: "<think>") {
            blocks.append(String(answer[open.upperBound...]))
            answer = String(answer[..<open.lowerBound])
            isThinking = true
        }
        let reasoning = blocks
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
        return Reply(
            answer: answer.trimmingCharacters(in: .whitespacesAndNewlines),
            reasoning: reasoning.isEmpty ? nil : reasoning,
            isThinking: isThinking
        )
    }

    /// The reply without its reasoning: `splitThinking(_:streaming:)` keeping only the answer.
    package static func stripThinkTags(_ text: String, streaming: Bool = false) -> String {
        splitThinking(text, streaming: streaming).answer
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
