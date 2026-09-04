import SwiftUI

/// What the menu bar can do to the key Studio window. `StudioRootView` publishes one of these via
/// `.focusedSceneValue`; `MereRunCommands` reads it with `@FocusedValue`, so File, View, Go, Run,
/// and Help act on whichever window is key and stay disabled when none is.
struct StudioSceneActions {
    let destination: StudioDestination
    let showLibrary: Binding<Bool>
    /// False on tasks without a prompt workspace, which have no Library column.
    let canShowLibrary: Bool
    /// The inspector column of the current prompt task; false elsewhere.
    let showInspector: Binding<Bool>
    let canShowInspector: Bool
    /// The Command view column on a prompt task; false elsewhere.
    let showCommand: Binding<Bool>
    let canShowCommand: Bool
    let open: (StudioDestination) -> Void
    /// Opens a domain at the task last shown there.
    let openDomain: (StudioDomain) -> Void
    let newChat: () -> Void
    let canNewChat: Bool
    let runComposer: () -> Void
    let canRun: Bool
    let stop: () -> Void
    let canStop: Bool
    /// Opens the Command Console window (the raw surface for every template).
    let openConsole: () -> Void
    let showGuide: () -> Void
    let importReceipt: () -> Void
}

private struct StudioSceneActionsKey: FocusedValueKey { typealias Value = StudioSceneActions }

extension FocusedValues {
    var studioActions: StudioSceneActions? {
        get { self[StudioSceneActionsKey.self] }
        set { self[StudioSceneActionsKey.self] = newValue }
    }
}
