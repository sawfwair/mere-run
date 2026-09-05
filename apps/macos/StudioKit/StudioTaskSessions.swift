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

    package func contains(_ key: String) -> Bool { entries[key] != nil }

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

    package func resolving(_ base: StudioRunRequest) -> StudioRunRequest {
        let task = base.templateID.studioTask
        let state = value(for: task.rawValue + ".commandOverride", default: Optional<StudioTaskCommandState>.none)
        guard let state, state.templateID == base.templateID,
              let launch = StudioConsoleRun(template: base.template,
                  draft: state.resolved(source: base.template.arguments(from: base.draft)), seed: base.draft) else { return base }
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
