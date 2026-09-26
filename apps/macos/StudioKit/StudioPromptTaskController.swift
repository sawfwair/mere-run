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
    /// The one submission path for every task that is not a conversation turn; the shared task
    /// workspace reaches it through the environment.
    @ObservationIgnored package let runner: StudioTaskRunner
    @ObservationIgnored private let seededDrafts: [StudioMode: StudioDraft]
    @ObservationIgnored private var pendingAnalyzeHandoff: StudioAnalyzeHandoff?

    package init(controller: MereRunController, library: StudioLibraryStore,
                 seededDrafts: [StudioMode: StudioDraft] = [:]) {
        self.controller = controller
        self.library = library
        self.runner = StudioTaskRunner(controller: controller, library: library)
        self.seededDrafts = seededDrafts
        controller.taskSessions.observeRestorations { [weak self] keys in self?.reloadDraft(after: keys) }
    }

    package var activatedMode: StudioMode? { active.mode }
    package var activeConversationID: UUID? { active.conversationID }
    /// Every edit of the open prompt task's draft lands here, so this is where its undo step is
    /// registered. Switching task or thread replaces `active` whole and registers nothing.
    package var draft: StudioDraft {
        get { active.draft }
        set {
            let previous = active.draft
            active.draft = newValue
            persistDraft()
            guard let mode = activatedMode else { return }
            sessions.registerDraftChange(keys: draftKeys(for: mode), from: previous, to: newValue) {
                newValue.undoName(from: previous, mode: mode, source: controller.scopeSource)
            }
        }
    }

    /// Where the open draft is stored: the task's key, and the thread's for Chat and Code.
    private func draftKeys(for mode: StudioMode) -> [String] {
        let taskKey = mode.task.rawValue + ".draft"
        guard mode.isConversational else { return [taskKey] }
        return [taskKey, sessions.conversationDraftKey(activeConversationID, mode: mode)]
    }

    /// An undo or redo wrote stored drafts back: the open draft follows when its key was one.
    private func reloadDraft(after keys: Set<String>) {
        guard let mode = activatedMode, let key = draftKeys(for: mode).last, keys.contains(key),
              let stored = sessions.value(for: key, default: Optional<StudioDraft>.none) else { return }
        active.draft = stored
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
            handoff.apply(to: &nextDraft, source: controller.scopeSource)
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
        replaceDraft(of: mode, with: next, undoName: action.title)
        return true
    }

    /// Makes `next` the task's draft in place of any Command view edits and focus, as one undo
    /// step that brings all three back.
    private func replaceDraft(of mode: StudioMode, with next: StudioDraft, undoName: String) {
        let task = mode.task
        let keys = [task.rawValue + ".draft", task.rawValue + ".commandOverride", task.rawValue + ".focus"]
            + (mode == activatedMode ? draftKeys(for: mode) : [])
        sessions.undoably(undoName, keys: Array(Set(keys))) {
            sessions.set(next, for: task.rawValue + ".draft")
            sessions.set(Optional<StudioTaskCommandState>.none, for: task.rawValue + ".commandOverride")
            sessions.setFocus(nil, for: task)
            if mode == activatedMode { draft = next }
        }
    }

    /// Library ▸ "Use these settings": makes the run's recorded command the task's draft, in
    /// place of any Command view edits, so the composer shows exactly what ran. false when the
    /// row has nothing to restore.
    package func useSettings(from item: StudioLibraryItem) -> Bool {
        if let task = item.templateID?.studioTask, task.usesTaskDraft {
            return useTaskSettings(from: item, task: task)
        }
        let mode = item.mode
        guard let next = StudioLibraryDraftRestoration.draft(
            from: item, baseline: freshDraft(for: mode), source: controller.scopeSource
        ) else { return false }
        replaceDraft(of: mode, with: next, undoName: Self.useSettingsUndoName)
        return true
    }

    /// The Edit menu's name for Library ▸ "Use these settings".
    package static let useSettingsUndoName = "Use These Settings"

    /// "Use these settings" for a row of a task on the shared task workspace: the recorded
    /// command becomes the task's draft (`"<task>.taskDraft"`), in place of any Command edits,
    /// so its composer and inspector show exactly what ran. The workspace reads the parked draft
    /// when it appears, so nothing here depends on which task is open.
    package func useTaskSettings(from item: StudioLibraryItem, task: StudioTask) -> Bool {
        guard let restored = StudioLibraryDraftRestoration.taskDraft(from: item, source: controller.scopeSource),
              var next = sessions.taskDraft(for: task) else { return false }
        next.adopt(restored)
        let keys = [StudioTaskSessions.taskDraftKey(task), task.rawValue + ".commandOverride", task.rawValue + ".focus"]
        sessions.undoably(Self.useSettingsUndoName, keys: keys) {
            sessions.setTaskDraft(next, for: task)
            sessions.set(Optional<StudioTaskCommandState>.none, for: task.rawValue + ".commandOverride")
            sessions.setFocus(nil, for: task)
        }
        return true
    }

    /// Models ▸ "Use for … by default": records the choice and moves the task onto it now — its
    /// parked draft, and the open composer when this is the active task — so the next run uses
    /// it without a restart. A draft that was following the previous default follows the new
    /// one; a model the user picked by hand stays, as does an open thread's. nil restores the
    /// built-in default.
    package func setPreferredModel(_ modelID: String?, for mode: StudioMode) {
        let keys = [mode.task.rawValue + ".preferredModel", mode.task.rawValue + ".draft"]
            + (mode == activatedMode ? draftKeys(for: mode) : [])
        sessions.undoably("Change Default Model", keys: Array(Set(keys))) {
            followPreferredModel(modelID, for: mode)
        }
    }

    private func followPreferredModel(_ modelID: String?, for mode: StudioMode) {
        let previous = freshDraft(for: mode).model
        sessions.setPreferredModel(modelID, for: mode)
        let next = freshDraft(for: mode).model
        func followsDefault(_ model: String) -> Bool { model.isBlank || model == previous }
        let key = mode.task.rawValue + ".draft"
        if var parked = sessions.value(for: key, default: Optional<StudioDraft>.none), followsDefault(parked.model) {
            parked.model = next
            sessions.set(parked, for: key)
        }
        guard mode == activatedMode, !(mode.isConversational && activeConversationID != nil),
              followsDefault(draft.model) else { return }
        var current = draft
        current.model = next
        draft = current
    }

    /// - Parameter detections: what the current result found on the input, carried into the
    ///   target as drawn box prompts when it takes them.
    package func prepareAnalyzeHandoff(to task: StudioTask, detections: [StudioAnalyzeDetection] = []) {
        pendingAnalyzeHandoff = StudioAnalyzeHandoff.make(
            to: task, inputPath: draft.inputPath, prompt: draft.prompt, detections: detections
        )
    }

    package func selectAnalyzeInput(from item: StudioLibraryItem) {
        var next = draft
        Self.applyAnalyzeInput(from: item, to: &next)
        draft = next
    }

    private static func applyAnalyzeInput(from item: StudioLibraryItem?, to draft: inout StudioDraft) {
        guard let item, let inputURL = item.inputURL else { return }
        draft.replaceInput(inputURL.path)
        if !item.prompt.isBlank { draft.prompt = item.prompt }
    }
}
