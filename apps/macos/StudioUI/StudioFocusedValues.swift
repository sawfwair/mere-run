import StudioKit
import SwiftUI

/// What the menu bar can do to the key Studio window. `StudioRootView` publishes one of these via
/// `.focusedSceneValue`; `MereRunCommands` reads it with `@FocusedValue`, so File, View, Go, Run,
/// and Help act on whichever window is key and stay disabled when none is.
package struct StudioSceneActions {
    package let destination: StudioDestination
    package let showLibrary: Binding<Bool>
    /// False on tasks without a prompt workspace, which have no Library column.
    package let canShowLibrary: Bool
    /// The inspector column of the current prompt task; false elsewhere.
    package let showInspector: Binding<Bool>
    package let canShowInspector: Bool
    /// The Command view column on a prompt task; false elsewhere.
    package let showCommand: Binding<Bool>
    package let canShowCommand: Bool
    package let open: (StudioDestination) -> Void
    /// Opens a domain at the task last shown there.
    package let openDomain: (StudioDomain) -> Void
    package let newChat: () -> Void
    package let canNewChat: Bool
    package let runComposer: () -> Void
    package let canRun: Bool
    package let stop: () -> Void
    package let canStop: Bool
    /// Opens the Command Console window (the raw surface for every template).
    package let openConsole: () -> Void
    package let showGuide: () -> Void
    package let importReceipt: () -> Void
}

/// A page's own Run for Run ▸ Run (⌘↩), published while the page's form is what runs rather
/// than the task's Command view: Training's Start training, Voices' Create voice. The menu
/// prefers it over `StudioSceneActions.runComposer`, so the key and the page's button do one thing.
package struct StudioPageRun {
    package let isEnabled: Bool
    package let run: () -> Void

    package init(isEnabled: Bool, run: @escaping () -> Void) {
        self.isEnabled = isEnabled
        self.run = run
    }
}

private struct StudioSceneActionsKey: FocusedValueKey { typealias Value = StudioSceneActions }
private struct StudioPageRunKey: FocusedValueKey { typealias Value = StudioPageRun }

extension FocusedValues {
    package var studioActions: StudioSceneActions? {
        get { self[StudioSceneActionsKey.self] }
        set { self[StudioSceneActionsKey.self] = newValue }
    }

    package var studioPageRun: StudioPageRun? {
        get { self[StudioPageRunKey.self] }
        set { self[StudioPageRunKey.self] = newValue }
    }
}
