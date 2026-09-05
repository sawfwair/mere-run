import Foundation

extension StudioTaskSessions {
    /// Chat and Code keep separate next-turn settings, including for a not-yet-sent thread.
    package func rememberConversationDraft(_ draft: StudioDraft, conversationID: UUID?, mode: StudioMode) {
        set(Optional(draft), for: conversationDraftKey(conversationID, mode: mode))
    }

    package func conversationDraft(conversationID: UUID?, mode: StudioMode) -> StudioDraft? {
        value(for: conversationDraftKey(conversationID, mode: mode), default: Optional<StudioDraft>.none)
    }

    package func forgetConversationDrafts(_ ids: Set<UUID>) {
        for id in ids {
            for mode in [StudioMode.chat, .code] {
                set(Optional<StudioDraft>.none, for: conversationDraftKey(id, mode: mode))
            }
        }
    }

    private func conversationDraftKey(_ id: UUID?, mode: StudioMode) -> String {
        mode.task.rawValue + ".conversation." + (id?.uuidString ?? "new")
    }
}
