import Foundation

/// The window's undo stack as the stores that write the user's work see it: the task sessions
/// (every draft) and the Library (deletions, renames, favorites) each register here once, at the
/// place they write, rather than each control registering its own step. The window hands its
/// `UndoManager` in while it is key; with none, nothing is registered and every write still
/// happens.
///
/// Each step re-registers its own inverse when it runs, so Redo is the same code path as the
/// original change and carries the original name ("Redo Change Steps", not a name derived from
/// whatever the inverse wrote).
@MainActor
package final class StudioUndo {
    /// How long a run of changes to the same field stays one step: a slider drag, a stepper held
    /// down. Each change restarts the window.
    package static let coalescingInterval: TimeInterval = 1

    package weak var manager: UndoManager? {
        didSet {
            guard manager !== oldValue else { return }
            coalescing = nil
            observeUndoAndRedo()
        }
    }

    private let now: () -> Date
    private var coalescing: (key: String, name: String, at: Date)?
    /// The name every registration takes while it is set: the step being undone or redone (so
    /// its inverse keeps the name), or a caller's own (`naming`).
    private var forcedName: String?
    private var suppression = 0
    private var observers: [NSObjectProtocol] = []
    /// Set while a Studio store is registering, so the group that registration opens is known to
    /// be ours. Shared: the sessions and the Library each hold a registrar on one manager.
    private static var isRegistering = false
    /// Whether the undo group open now was opened by something other than these stores.
    private var foreignGroupIsOpen = false

    package init(now: @escaping () -> Date = Date.init) {
        self.now = now
    }

    /// Whether registrations are being gathered into a caller's step (`suppressing`).
    package var isSuppressed: Bool { suppression > 0 }

    /// Whether something else — a text field's typing — already registered a step in the event
    /// being handled. A draft change arriving in that same event is the text field's own edit
    /// reaching the draft, and the field's step already undoes it; a second step for it would
    /// rewrite the text under the field's own undo.
    package var anotherRegistrarOwnsThisEvent: Bool {
        guard let manager else { return false }
        return foreignGroupIsOpen && manager.groupingLevel > 0 && !manager.isUndoing && !manager.isRedoing
    }

    /// Registers `undo` as one step named `name`. With a `coalescing` key, a change that follows
    /// one with the same key and name within `coalescingInterval` joins that step instead: its
    /// undo already restores the value from before the first change. Nothing coalesces across
    /// an undo or redo, a named step, or a step another registrar (a text field's typing) put on
    /// top.
    package func register(
        _ name: @autoclosure () -> String,
        coalescing key: String? = nil,
        undo: @escaping @MainActor () -> Void
    ) {
        guard let manager, suppression == 0 else { return }
        let time = now()
        let key = forcedName == nil && !manager.isUndoing && !manager.isRedoing ? key : nil
        if let key, let last = coalescing, last.key == key, time.timeIntervalSince(last.at) < Self.coalescingInterval,
           manager.undoActionName == last.name {
            coalescing?.at = time
            return
        }
        let title = forcedName ?? name()
        coalescing = key.map { ($0, title, time) }
        let action = StudioUndoAction(name: title, perform: undo)
        Self.isRegistering = true
        manager.registerUndo(withTarget: self) { registrar in
            MainActor.assumeIsolated { registrar.replay(action) }
        }
        Self.isRegistering = false
        manager.setActionName(title)
    }

    /// Runs `body` with every registration it makes named `name` and none coalescing: a Reset
    /// that writes several fields is "Reset Settings", not "Change Settings".
    package func naming(_ name: String, _ body: () -> Void) {
        let previous = forcedName
        forcedName = name
        coalescing = nil
        body()
        forcedName = previous
    }

    /// Runs `body` without registering anything: the caller registers the whole change as one
    /// step of its own.
    package func suppressing(_ body: () -> Void) {
        suppression += 1
        body()
        suppression -= 1
    }

    private func replay(_ action: StudioUndoAction) {
        let previous = forcedName
        forcedName = action.name
        action.perform()
        forcedName = previous
    }

    private func observeUndoAndRedo() {
        for observer in observers { NotificationCenter.default.removeObserver(observer) }
        observers = []
        guard let manager else { return }
        let center = NotificationCenter.default
        observers = [Notification.Name.NSUndoManagerDidUndoChange, .NSUndoManagerDidRedoChange].map { name in
            center.addObserver(forName: name, object: manager, queue: nil) { [weak self] _ in
                MainActor.assumeIsolated { self?.coalescing = nil }
            }
        }
        observers.append(center.addObserver(forName: .NSUndoManagerDidOpenUndoGroup, object: manager, queue: nil) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, self.manager?.groupingLevel == 1 else { return }
                self.foreignGroupIsOpen = !Self.isRegistering
            }
        })
    }
}

/// One registered step: the name it shows in the Edit menu and the work that undoes it.
private final class StudioUndoAction: @unchecked Sendable {
    let name: String
    let perform: @MainActor () -> Void

    init(name: String, perform: @escaping @MainActor () -> Void) {
        self.name = name
        self.perform = perform
    }
}
