import StudioKit
import SwiftUI

/// Hands the key Studio window's undo manager to the stores that register undo steps (the task
/// sessions and the Library), so ⌘Z and the Edit menu's "Undo Delete Run" reach them. The stores
/// are shared by every window; whichever window is in front owns the stack they write to.
struct StudioUndoBinding: ViewModifier {
    let registrars: [StudioUndo]
    @Environment(\.undoManager) private var undoManager
    @Environment(\.controlActiveState) private var activeState

    func body(content: Content) -> some View {
        content
            .onChange(of: activeState, initial: true) { bind() }
            .onChange(of: undoManager, initial: true) { bind() }
    }

    private func bind() {
        guard activeState != .inactive, let undoManager else { return }
        for registrar in registrars { registrar.manager = undoManager }
    }
}

/// The Edit menu names of the inspector's Reset buttons. The reset itself writes the draft
/// through its usual binding, which registers the step; this only names it.
enum StudioUndoNaming {
    @MainActor
    static func reset(_ title: String, in sessions: StudioTaskSessions?, _ body: () -> Void) {
        guard let sessions else { return body() }
        sessions.undo.naming("Reset \(title)", body)
    }
}
