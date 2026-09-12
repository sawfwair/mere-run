import Foundation

extension StudioPromptTaskController {
    package func conversationBudgetChars(inventory: [StudioModelInventoryRow]) -> Int {
        Self.conversationBudgetChars(mode: activatedMode ?? .chat, draft: draft, inventory: inventory)
    }

    private static func conversationBudgetChars(mode: StudioMode, draft: StudioDraft, inventory: [StudioModelInventoryRow]) -> Int {
        ConversationTranscript.budgetChars(
            contextTokens: ConversationTranscript.contextTokens(
                requestedContextSize: draft.contextSize,
                model: StudioModelNaming.resolvedModelID(for: mode, model: draft.model), inventory: inventory),
            maxOutputTokens: draft.maxTokens)
    }

    private func conversationRequest(
        mode: StudioMode, draft: StudioDraft, conversationID: UUID,
        messages: [StudioMessage], systemPrompt: String?, inventory: [StudioModelInventoryRow]
    ) throws -> StudioRunRequest {
        let budget = Self.conversationBudgetChars(mode: mode, draft: draft, inventory: inventory)
        var runDraft = draft
        runDraft.prompt = ConversationTranscript.render(messages: messages, systemPrompt: systemPrompt, budgetChars: budget).prompt
        runDraft.stats = true
        return try preparedRequest(mode: mode, draft: runDraft, conversationID: conversationID)
    }

    func sendConversationTurn(inventory: [StudioModelInventoryRow]) throws -> StudioRunRequest? {
        guard let mode = activatedMode else { return nil }
        let content = draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty else { return nil }
        let conversationID = activeConversationID ?? UUID()
        guard !controller.runningConversationIDs.contains(conversationID) else { return nil }
        let system = draft.secondaryText.trimmingCharacters(in: .whitespacesAndNewlines)
        let systemPrompt = system.isEmpty ? nil : system
        let model = draft.model.isBlank ? nil : draft.model
        let turnImage = mode == .chat && !draft.inputPath.isBlank ? draft.inputPath : nil
        let messages = (activeConversationItem?.messages ?? [])
            + [StudioMessage(role: .user, content: content, imagePath: turnImage)]
        let request = try conversationRequest(mode: mode, draft: draft, conversationID: conversationID,
            messages: messages, systemPrompt: systemPrompt, inventory: inventory)

        library.appendUser(conversationID: conversationID, mode: mode, model: model,
            systemPrompt: systemPrompt, content: content, imagePath: turnImage)
        controller.run(studio: request)
        var sent = draft
        sent.prompt = ""
        sent.inputPath = ""
        if activeConversationID == nil {
            sessions.rememberConversationDraft(sent, conversationID: nil, mode: mode)
        }
        setConversation(conversationID, draft: sent)
        sessions.rememberSelection(conversationID, for: mode)
        return request
    }

    /// Prepare the replacement first. Invalid Command edits must leave the existing reply and
    /// unsent composer text intact, with no new job or transcript mutation.
    package func retryLastTurn(inventory: [StudioModelInventoryRow]) throws -> StudioRunRequest? {
        guard let conversationID = activeConversationID,
              !controller.runningConversationIDs.contains(conversationID),
              let item = activeConversationItem else { return nil }
        var messages = item.messages ?? []
        if messages.last?.role == .assistant { messages.removeLast() }
        guard messages.last?.role == .user else { return nil }
        var runDraft = draft
        runDraft.secondaryText = item.systemPrompt ?? ""
        runDraft.inputPath = messages.last?.imagePath ?? ""
        if let model = item.model, !model.isBlank { runDraft.model = model }
        let request = try conversationRequest(mode: item.mode, draft: runDraft, conversationID: conversationID,
            messages: messages, systemPrompt: item.systemPrompt, inventory: inventory)
        library.dropLastAssistant(conversationID: conversationID)
        controller.run(studio: request)
        return request
    }

    /// Returns whether the composer should receive focus after loading a prior user turn.
    package func editMessage(_ messageID: UUID) -> Bool {
        guard let mode = activatedMode, let conversationID = activeConversationID,
              !controller.runningConversationIDs.contains(conversationID),
              let removed = library.truncate(conversationID: conversationID, removingFrom: messageID) else { return false }
        var edited = draft
        edited.prompt = removed.content
        if mode == .chat { edited.inputPath = removed.imagePath ?? "" }
        if let item = activeConversationItem, item.messages?.isEmpty ?? true {
            library.delete(id: conversationID)
            sessions.forgetConversationDrafts([conversationID])
            setConversation(nil, draft: edited)
            sessions.rememberSelection(nil, for: mode)
        } else {
            draft = edited
        }
        return true
    }

    /// Stages a branch under the preset recorded at the selected turn. Navigation follows the
    /// returned activation; the original thread keeps its transcript and unsent draft.
    package func branchFromMessage(_ messageID: UUID) -> Activation? {
        guard let conversationID = activeConversationID,
              !controller.runningConversationIDs.contains(conversationID),
              let source = activeConversationItem,
              let message = source.messages?.first(where: { $0.id == messageID }),
              let branch = library.branch(conversationID: conversationID, at: messageID,
                                          inclusive: message.role == .assistant) else { return nil }
        var branchDraft = freshDraft(for: branch.mode)
        applyConversationSettings(from: branch, to: &branchDraft)
        if message.role == .user {
            branchDraft.prompt = message.content
            if branch.mode == .chat { branchDraft.inputPath = message.imagePath ?? "" }
        }
        let branchID: UUID?
        if branch.messages?.isEmpty ?? true {
            library.delete(id: branch.id)
            branchID = nil
        } else {
            branchID = branch.id
        }
        sessions.rememberConversationDraft(branchDraft, conversationID: branchID, mode: branch.mode)
        sessions.rememberSelection(branchID, for: branch.mode)
        if branch.mode == activatedMode { setConversation(branchID, draft: branchDraft) }
        return Activation(mode: branch.mode, selectedLibraryID: branchID)
    }
}
