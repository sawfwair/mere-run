import Foundation
import Observation

/// Owns the prompt workspace's active draft and transitions. Inactive drafts live only in
/// task sessions; the shell owns navigation, layout, focus, and platform dialogs.
@MainActor
@Observable
package final class StudioPromptTaskController {
    private struct ActiveTask {
        var mode: StudioMode?
        var draft = StudioDraft()
        var conversationID: UUID?
    }

    private var active = ActiveTask()
    @ObservationIgnored let controller: MereRunController
    @ObservationIgnored let library: StudioLibraryStore
    @ObservationIgnored private let seededDrafts: [StudioMode: StudioDraft]
    @ObservationIgnored private var pendingAnalyzeHandoff: StudioAnalyzeHandoff?

    package init(controller: MereRunController, library: StudioLibraryStore,
                 seededDrafts: [StudioMode: StudioDraft] = [:]) {
        self.controller = controller
        self.library = library
        self.seededDrafts = seededDrafts
    }

    package var activatedMode: StudioMode? { active.mode }
    package var activeConversationID: UUID? { active.conversationID }
    package var draft: StudioDraft {
        get { active.draft }
        set {
            active.draft = newValue
            persistDraft()
        }
    }

    var sessions: StudioTaskSessions { controller.taskSessions }

    package var activeConversationItem: StudioLibraryItem? {
        library.items.first { $0.id == active.conversationID && $0.isConversation }
    }

    /// Imports every legacy task once, including tasks not visited during this launch. Existing
    /// full drafts take precedence. The scene value remains untouched as a migration source.
    package func importLegacyDrafts(_ encoded: String) {
        for (task, entry) in StudioDraftMemory.decode(encoded) {
            guard let mode = task.mode, !sessions.contains(task.rawValue + ".draft") else { continue }
            var restored = freshDraft(for: mode)
            StudioDraftMemory.apply(entry, to: &restored)
            sessions.set(restored, for: task.rawValue + ".draft")
        }
    }

    package func freshDraft(for mode: StudioMode) -> StudioDraft {
        var draft = StudioDraft()
        draft.reset(for: mode)
        controller.applyRecommendedDefaults(to: &draft, for: mode)
        if mode.isConversational { draft.prompt = "" }
        return draft
    }

    package struct Activation: Equatable {
        package let mode: StudioMode
        package let selectedLibraryID: UUID?
    }

    /// Restores a task, honoring explicit Library selection and each thread's unsent draft.
    /// Chat and Code can change the preset of the open thread without changing its identity.
    package func activate(_ newMode: StudioMode, preferredID: UUID?) -> Activation {
        let leavingMode = activatedMode
        let parked: StudioDraft? = seededDrafts[newMode] == nil
            ? sessions.value(for: newMode.task.rawValue + ".draft", default: Optional<StudioDraft>.none) : nil
        var nextDraft = seededDrafts[newMode] ?? parked ?? freshDraft(for: newMode)
        let selection = sessions.selection(for: newMode, items: library.items, preferredID: preferredID)
        let keepsOpenThread = leavingMode?.isConversational == true && newMode.isConversational
            && activeConversationItem != nil && preferredID == activeConversationID && !selection.isExplicit
        let preferred = keepsOpenThread ? nil : selection.item
        var conversationID = activeConversationID
        var selectedID: UUID?

        if newMode.isConversational {
            if let preferred, preferred.isConversation {
                conversationID = preferred.id
                selectedID = preferred.id
                if parked == nil || selection.isExplicit {
                    applyConversationSettings(from: preferred, to: &nextDraft)
                }
            } else if keepsOpenThread, let current = activeConversationItem {
                selectedID = current.id
            } else if !selection.hasMemory, let recent = StudioThreadListPresenter.threads(in: library.items).first {
                // Resolve the preset before replacing active state, so an intermediate navigation
                // update cannot save the departing task's draft under the arriving task's key.
                if recent.mode != newMode { return activate(recent.mode, preferredID: recent.id) }
                conversationID = recent.id
                selectedID = recent.id
                applyConversationSettings(from: recent, to: &nextDraft)
            } else {
                conversationID = nil
            }
            if let saved = sessions.conversationDraft(conversationID: conversationID, mode: newMode) {
                nextDraft = saved
            } else if parked == nil || selection.isExplicit || (keepsOpenThread && leavingMode != newMode) {
                nextDraft = seededDrafts[newMode] ?? freshDraft(for: newMode)
                if !keepsOpenThread, let item = library.items.first(where: { $0.id == conversationID && $0.isConversation }) {
                    applyConversationSettings(from: item, to: &nextDraft)
                }
                nextDraft.prompt = ""
            }
        } else {
            conversationID = nil
            let selected = preferred ?? library.items.first { $0.mode == newMode }
            selectedID = selected?.id
            if newMode.task.isAnalyzeTask, nextDraft.inputPath.isBlank {
                Self.applyAnalyzeInput(from: selected, to: &nextDraft)
            }
        }
        if let handoff = pendingAnalyzeHandoff, handoff.task.mode == newMode {
            handoff.apply(to: &nextDraft)
            selectedID = nil
        }
        pendingAnalyzeHandoff = nil
        active = ActiveTask(mode: newMode, draft: nextDraft, conversationID: conversationID)
        persistDraft()
        sessions.rememberSelection(selectedID, for: newMode)
        return Activation(mode: newMode, selectedLibraryID: selectedID)
    }

    private func persistDraft() {
        guard let mode = activatedMode else { return }
        sessions.set(draft, for: mode.task.rawValue + ".draft")
        if mode.isConversational {
            sessions.rememberConversationDraft(draft, conversationID: activeConversationID, mode: mode)
        }
    }

    func setConversation(_ id: UUID?, draft: StudioDraft) {
        active = ActiveTask(mode: activatedMode, draft: draft, conversationID: id)
        persistDraft()
    }

    func applyConversationSettings(from item: StudioLibraryItem, to draft: inout StudioDraft) {
        draft.secondaryText = item.systemPrompt ?? ""
        draft.model = item.model ?? ""
    }

    package func restoreConversation(_ thread: StudioLibraryItem) {
        guard let mode = activatedMode else { return }
        var restored = freshDraft(for: mode)
        applyConversationSettings(from: thread, to: &restored)
        setConversation(thread.id, draft: sessions.conversationDraft(conversationID: thread.id, mode: mode) ?? restored)
        sessions.rememberSelection(thread.id, for: mode)
    }

    package func startNewConversation() {
        guard let mode = activatedMode, mode.isConversational, activeConversationID != nil else { return }
        setConversation(nil, draft: sessions.conversationDraft(conversationID: nil, mode: mode) ?? freshDraft(for: mode))
        sessions.rememberSelection(nil, for: mode)
    }

    package func forgetConversations(_ ids: Set<UUID>) {
        sessions.forgetConversationDrafts(ids)
        guard let mode = activatedMode, let id = activeConversationID, ids.contains(id) else { return }
        setConversation(nil, draft: sessions.conversationDraft(conversationID: nil, mode: mode) ?? freshDraft(for: mode))
    }

    package func continueResult(_ action: StudioResultContinuation, item: StudioLibraryItem, url: URL) -> Bool {
        guard let mode = action.task.mode,
              let next = action.draft(from: item, url: url, baseline: freshDraft(for: mode)) else { return false }
        sessions.set(next, for: action.task.rawValue + ".draft")
        sessions.set(Optional<StudioTaskCommandState>.none, for: action.task.rawValue + ".commandOverride")
        sessions.set(Optional<StudioResultSelection>.none, for: action.task.rawValue + ".focus")
        if mode == activatedMode { draft = next }
        return true
    }

    package func prepareAnalyzeHandoff(to task: StudioTask) {
        pendingAnalyzeHandoff = StudioAnalyzeHandoff.make(to: task, inputPath: draft.inputPath, prompt: draft.prompt)
    }

    package func selectAnalyzeInput(from item: StudioLibraryItem) {
        var next = draft
        Self.applyAnalyzeInput(from: item, to: &next)
        draft = next
    }

    private static func applyAnalyzeInput(from item: StudioLibraryItem?, to draft: inout StudioDraft) {
        guard let item, let inputURL = item.inputURL else { return }
        draft.inputPath = inputURL.path
        if !item.prompt.isBlank { draft.prompt = item.prompt }
    }
}
