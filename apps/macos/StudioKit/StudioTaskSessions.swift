import Foundation
import Observation

/// Versioned, typed task state. Views may disappear without taking their drafts or selection with them.
@MainActor
@Observable
package final class StudioTaskSessions {
    private var entries: [String: Data] = [:]
    @ObservationIgnored private var persistedEntries: [String: Data] = [:]
    package private(set) var lastPersistenceError: String?
    @ObservationIgnored private let url: URL?
    @ObservationIgnored private var saveTask: Task<Void, Never>?
    @ObservationIgnored private var canSave = true

    package static var defaultURL: URL {
        StudioLibraryStore.defaultLibraryURL().deletingLastPathComponent()
            .appendingPathComponent("task-sessions-v1.json")
    }

    package init(url: URL? = nil) {
        self.url = url
        guard let url, FileManager.default.fileExists(atPath: url.path) else { return }
        do {
            entries = try JSONDecoder.mereRunApp.decode([String: Data].self, from: Data(contentsOf: url))
            persistedEntries = entries
        } catch {
            canSave = false
            lastPersistenceError = "Saved task settings could not be read. The original file has been preserved."
        }
    }

    package func value<Value: Codable>(for key: String, default initial: Value) -> Value {
        guard let data = entries[key], let value = try? JSONDecoder.mereRunApp.decode(Value.self, from: data) else {
            return initial
        }
        return value
    }

    /// The task drafts as last read, keyed by session key and remembered with the bytes they
    /// were decoded from (nil for a fresh draft nothing has written yet). Not observed: `entries`
    /// is, so a write still re-renders every reader; this only saves decoding the same bytes
    /// again on the next read, and lets a fresh draft be handed to every reader without writing
    /// the store from inside a view update.
    @ObservationIgnored private var taskDraftCache: [String: (data: Data?, draft: StudioTaskDraft)] = [:]

    /// The task draft under `key`: the parked one decoded once per stored value, or the fresh
    /// one `remember` handed out while nothing is parked. A parked draft is read without the
    /// destinations the app named into it (`withoutAppDestinations`), so the next run is named
    /// afresh rather than written over an earlier one's file.
    func cachedTaskDraft(for key: String) -> StudioTaskDraft? {
        let data = entries[key]
        if let cached = taskDraftCache[key], cached.data == data { return cached.draft }
        guard let data, let parked = try? JSONDecoder.mereRunApp.decode(StudioTaskDraft.self, from: data) else { return nil }
        let draft = parked.withoutAppDestinations()
        taskDraftCache[key] = (data, draft)
        return draft
    }

    /// Keeps a fresh draft as the answer for `key` until something is parked under it.
    func rememberFreshTaskDraft(_ draft: StudioTaskDraft, for key: String) {
        taskDraftCache[key] = (nil, draft)
    }

    package func contains(_ key: String) -> Bool { entries[key] != nil }

    package func containsKey(withPrefix prefix: String) -> Bool {
        entries.keys.contains { $0.hasPrefix(prefix) }
    }

    /// The task each run launched in this process was submitted from. A job lives no longer
    /// than the process, so this is never persisted; Stop reads it to tell apart two tasks that
    /// run the same command (Audio ▸ Separate and Music ▸ Separate).
    @ObservationIgnored private var submittingTasks: [UUID: StudioTask] = [:]

    package func noteSubmission(_ requestID: UUID, from task: StudioTask) {
        submittingTasks[requestID] = task
    }

    package func submittingTask(of requestID: UUID) -> StudioTask? {
        submittingTasks[requestID]
    }

    package func set<Value: Codable>(_ value: Value, for key: String) {
        do {
            let data = try JSONEncoder.mereRunApp.encode(value)
            let persisted: Data
            if let state = value as? any StudioSessionPersistable {
                persisted = try state.persistedData()
            } else {
                persisted = data
            }
            guard entries[key] != data else { return }
            entries[key] = data
            persistedEntries[key] = persisted
            saveTask?.cancel()
            saveTask = Task { [weak self] in
                do { try await Task.sleep(for: .milliseconds(250)) } catch { return }
                self?.flush()
            }
        } catch {
            lastPersistenceError = "Task settings could not be saved: \(error.localizedDescription)"
        }
    }

    package struct Selection {
        package let item: StudioLibraryItem?
        package let hasMemory: Bool
        package let isExplicit: Bool
    }

    package func rememberSelection(_ id: UUID?, for mode: StudioMode) {
        set(id, for: mode.task.rawValue + ".selection")
        let focus = value(for: mode.task.rawValue + ".focus", default: Optional<StudioResultSelection>.none)
        if let focus, focus.itemID != id { setFocus(nil, for: mode.task) }
    }

    package func focusedResult(for task: StudioTask, items: [StudioLibraryItem]) -> StudioResultSelection? {
        let focus = value(for: task.rawValue + ".focus", default: Optional<StudioResultSelection>.none)
        return focus.flatMap { selection in
            items.contains { $0.id == selection.itemID && $0.allArtifactURLs.contains(selection.url) } ? selection : nil
        }
    }

    package func setFocus(_ selection: StudioResultSelection?, for task: StudioTask) {
        set(selection, for: task.rawValue + ".focus")
    }

    /// The model the user made this mode's default on the Models page, or nil for the built-in
    /// default. Kept with the task's other state, because it is what the task starts from.
    package func preferredModel(for mode: StudioMode) -> String? {
        value(for: mode.task.rawValue + ".preferredModel", default: Optional<String>.none)
    }

    package func setPreferredModel(_ modelID: String?, for mode: StudioMode) {
        set(modelID, for: mode.task.rawValue + ".preferredModel")
    }

    package func forgetLibraryItems(_ ids: Set<UUID>) {
        for mode in StudioMode.allCases {
            let selection = value(for: mode.task.rawValue + ".selection", default: Optional<UUID>.none)
            if let selection, ids.contains(selection) { rememberSelection(nil, for: mode) }
            let focus = value(for: mode.task.rawValue + ".focus", default: Optional<StudioResultSelection>.none)
            if let focus, ids.contains(focus.itemID) { setFocus(nil, for: mode.task) }
        }
    }

    package func selection(for mode: StudioMode, items: [StudioLibraryItem], preferredID: UUID?) -> Selection {
        let key = mode.task.rawValue + ".selection"
        let rememberedID = value(for: key, default: Optional<UUID>.none)
        let preferred = items.first { $0.id == preferredID && $0.mode == mode }
        let remembered = items.first {
            $0.id == rememberedID && ($0.mode == mode || (mode.isConversational && $0.isConversation))
        }
        return Selection(item: preferred ?? remembered, hasMemory: contains(key),
                         isExplicit: preferred != nil && preferredID != rememberedID)
    }

    private func commandState(for templateID: CommandTemplateID) -> StudioTaskCommandState? {
        let task = templateID.studioTask
        let state = value(for: task.rawValue + ".commandOverride", default: Optional<StudioTaskCommandState>.none)
        return state?.templateID == templateID ? state : nil
    }

    /// The Command view's form for a request. A task on the shared task workspace edits its
    /// task draft's form directly — the composer, the inspector, and the Command view are one
    /// value — so there is never a separate override to merge for it.
    package func commandForm(for request: StudioRunRequest, source: StudioScopeSource) -> StudioConsoleDraft {
        let task = request.templateID.studioTask
        if task.usesTaskDraft, let draft = taskDraft(for: task), draft.templateID == request.templateID {
            return draft.form
        }
        return commandState(for: request.templateID)?.resolved(source: request.template.arguments(from: request.draft, source: source))
            ?? StudioConsoleCommand.seed(template: request.template, draft: request.draft, source: source)
    }

    package func resolving(_ base: StudioRunRequest, source: StudioScopeSource) -> StudioRunRequest {
        // A task draft's request already carries its form as its execution.
        guard !base.templateID.studioTask.usesTaskDraft,
              commandState(for: base.templateID) != nil,
              let launch = StudioConsoleRun(template: base.template,
                  draft: commandForm(for: base, source: source), seed: base.draft, source: source) else { return base }
        return StudioRunRequest(id: base.id, mode: base.mode, templateID: base.templateID,
            template: base.template, draft: launch.commandDraft, createdAt: base.createdAt,
            conversationID: base.conversationID,
            execution: StudioExecution(templateID: base.templateID, arguments: launch.arguments), parentID: base.parentID)
    }

    package func flush() {
        saveTask?.cancel()
        saveTask = nil
        guard let url, canSave else { return }
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder.mereRunApp.encode(persistedEntries).write(to: url, options: .atomic)
            lastPersistenceError = nil
        } catch {
            lastPersistenceError = "Task settings could not be saved: \(error.localizedDescription)"
        }
    }
}

/// Typed sanitization composes through optional values and specialist draft dictionaries.
package protocol StudioSessionPersistable: Codable {
    var withoutSessionSecrets: Self { get }
}

extension StudioSessionPersistable {
    fileprivate func persistedData() throws -> Data {
        try JSONEncoder.mereRunApp.encode(withoutSessionSecrets)
    }
}

extension CommandDraft: StudioSessionPersistable {
    package var withoutSessionSecrets: Self { withoutSecrets }
}

extension StudioTaskCommandState: StudioSessionPersistable {
    package var withoutSessionSecrets: Self { withoutSecrets }
}

extension Optional: StudioSessionPersistable where Wrapped: StudioSessionPersistable {
    package var withoutSessionSecrets: Self { map(\.withoutSessionSecrets) }
}

extension Dictionary: StudioSessionPersistable where Key: Codable, Value: StudioSessionPersistable {
    package var withoutSessionSecrets: Self { mapValues(\.withoutSessionSecrets) }
}
