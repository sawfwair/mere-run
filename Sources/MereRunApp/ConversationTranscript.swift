import Foundation

/// Renders a chat/code conversation into the single `--prompt` string the CLI accepts.
///
/// The CLI is stateless per invocation, so the app serializes history: prior turns become a
/// labeled dialogue and the latest user message ends the prompt with no trailing `Assistant:`
/// cue (the model's chat template adds that). When the history exceeds the character budget the
/// OLDEST messages are dropped from what is *sent* — they stay persisted in the thread; the drop
/// is reported (`droppedCount`/`includedMessageIDs`) so the UI can surface it rather than
/// silently truncating.
enum ConversationTranscript {
    struct Rendered: Equatable {
        let prompt: String
        let includedMessageIDs: Set<UUID>
        let droppedCount: Int
        let approxChars: Int
    }

    /// A conservative character budget (~4 chars/token). Tunable; a per-model budget is a later
    /// refinement. The system prompt rides in `--system` and is reserved out of the history room.
    static let defaultBudgetChars = 48_000

    static func render(
        messages: [StudioMessage],
        systemPrompt: String? = nil,
        budgetChars: Int = defaultBudgetChars
    ) -> Rendered {
        let reserve = systemPrompt?.count ?? 0
        let historyBudget = max(0, budgetChars - reserve)

        // Always keep the latest message; walk backward, including older ones until the next one
        // would exceed the budget.
        var included: [StudioMessage] = []
        var used = 0
        for message in messages.reversed() {
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
            droppedCount: messages.count - included.count,
            approxChars: prompt.count + reserve
        )
    }

    /// A single user message renders verbatim, so the first turn is byte-identical to a
    /// single-shot run; multi-turn windows render as an oldest→newest labeled dialogue.
    static func format(_ messages: [StudioMessage]) -> String {
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
}
