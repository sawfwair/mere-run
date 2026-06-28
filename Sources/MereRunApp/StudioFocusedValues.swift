import SwiftUI

/// Focused scene values that let the menu bar (MereRunCommands) drive the active window's view
/// toggles. StudioRootView publishes its bindings via `.focusedSceneValue`; the commands read them
/// with `@FocusedValue` so the View menu acts on whichever Studio window is key.
private struct ShowLibraryKey: FocusedValueKey { typealias Value = Binding<Bool> }
private struct ShowAdvancedKey: FocusedValueKey { typealias Value = Binding<Bool> }
private struct ShowModelsKey: FocusedValueKey { typealias Value = Binding<Bool> }

extension FocusedValues {
    var showLibrary: Binding<Bool>? {
        get { self[ShowLibraryKey.self] }
        set { self[ShowLibraryKey.self] = newValue }
    }

    var showAdvanced: Binding<Bool>? {
        get { self[ShowAdvancedKey.self] }
        set { self[ShowAdvancedKey.self] = newValue }
    }

    var showModels: Binding<Bool>? {
        get { self[ShowModelsKey.self] }
        set { self[ShowModelsKey.self] = newValue }
    }
}
