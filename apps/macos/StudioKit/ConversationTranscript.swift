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

    /// Removes model reasoning blocks from an assistant reply. With `--stream` the CLI emits
    /// `<think>…</think>` reasoning inline (it only strips it on the non-stream path), so the app
    /// must strip it before storing/replaying — otherwise reasoning leaks into the next turn's
    /// prompt.
    ///
    /// Complete blocks are always removed, as is a leading orphan `</think>` (some models pre-fill
    /// the opening tag and emit only the close). A trailing UNCLOSED block is only stripped while
    /// `streaming` — that is reasoning still in progress. At finalize it is kept: a completed
    /// reply's leftover `<think>` is almost certainly literal text (e.g. a code reply that
    /// discusses the tag), and truncating it would lose real content.
    package static func stripThinkTags(_ text: String, streaming: Bool = false) -> String {
        var result = text.replacingOccurrences(
            of: "<think>[\\s\\S]*?</think>",
            with: "",
            options: .regularExpression
        )
        if !result.contains("<think>"), let close = result.range(of: "</think>") {
            result = String(result[close.upperBound...])
        }
        if streaming {
            result = result.replacingOccurrences(
                of: "<think>[\\s\\S]*$",
                with: "",
                options: .regularExpression
            )
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
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
