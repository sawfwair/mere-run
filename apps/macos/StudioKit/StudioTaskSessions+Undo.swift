import Foundation

/// The bytes stored under a set of session keys, as `StudioTaskSessions.restore` puts them back.
struct StudioSessionSnapshot: Equatable {
    struct Stored: Equatable {
        let entry: Data?
        /// The secret-free copy the file holds.
        let persisted: Data?
    }

    let values: [String: Stored]
}

extension StudioTaskSessions {
    /// Runs `body` as one undo step named `name`: every value it writes under `keys` comes back
    /// together on Undo. Draft changes `body` makes are part of this step, not steps of their own.
    /// "Use these settings" replaces a task's draft this way, so Undo brings the previous draft,
    /// Command edits, and focus back whole.
    package func undoably(_ name: String, keys: [String], _ body: () -> Void) {
        let before = snapshot(keys)
        undo.suppressing(body)
        registerRestoration(name, of: before)
    }

    private func registerRestoration(_ name: String, of snapshot: StudioSessionSnapshot) {
        let keys = Array(snapshot.values.keys)
        guard snapshot != self.snapshot(keys) else { return }
        undo.register(name) { [weak self] in
            guard let self else { return }
            let current = self.snapshot(keys)
            self.restore(snapshot)
            self.registerRestoration(name, of: current)
        }
    }

    /// Registers the change from `previous` to `next`, already written under every one of
    /// `keys`, as an undo step. Changes to the same fields coalesce (a slider drag is one step);
    /// a change that only typed text is not registered at all (`name` answers nil), and neither
    /// is one arriving in an event a text field already registered its typing in, because the
    /// text field's own undo has it.
    func registerDraftChange<Draft: StudioUndoableDraft>(
        keys: [String],
        from previous: Draft,
        to next: Draft,
        name: () -> String?
    ) {
        guard undo.manager != nil, !undo.isSuppressed, !undo.anotherRegistrarOwnsThisEvent,
              previous != next else { return }
        var retyped = next
        retyped.prompt = previous.prompt
        guard retyped != previous, let title = name() else { return }
        let coalescing = (keys + StudioDraftUndo.changedPaths(from: previous, to: next)).joined(separator: "|")
        let changes = keys.map { StudioDraftChange(key: $0, previous: previous, next: next) }
        undo.register(title, coalescing: coalescing) { [weak self] in
            self?.revert(changes, name: title)
        }
    }

    /// Undoes draft changes as patches, then registers the patches that redo them: each put back
    /// as the key held it just now, which after a coalesced drag is where the drag ended, not
    /// where its first step went.
    private func revert<Draft: StudioUndoableDraft>(_ changes: [StudioDraftChange<Draft>], name: String) {
        let inverse = changes.map { change in
            let current = value(for: change.key, default: change.next)
            let reverted = StudioDraftUndo.reverting(current, from: change.previous, to: change.next)
            set(reverted, for: change.key)
            return StudioDraftChange(key: change.key, previous: current, next: reverted)
        }
        didRestore(Set(changes.map(\.key)))
        undo.register(name) { [weak self] in self?.revert(inverse, name: name) }
    }
}

/// One stored draft's change, as its undo patch reads it.
private struct StudioDraftChange<Draft> {
    let key: String
    let previous: Draft
    let next: Draft
}
