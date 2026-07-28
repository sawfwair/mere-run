import Sparkle
import SwiftUI

/// Top-level menu bar commands. View-toggle commands (Library/Advanced/Models) are
/// installed by `StudioRootView` via focused scene values once the UI state is shared.
struct MereRunCommands: Commands {
    @ObservedObject var controller: MereRunController
    let updater: SPUUpdater

    @FocusedValue(\.showLibrary) private var showLibrary: Binding<Bool>?
    @FocusedValue(\.showAdvanced) private var showAdvanced: Binding<Bool>?
    @FocusedValue(\.showModels) private var showModels: Binding<Bool>?

    var body: some Commands {
        // Single-window studio: remove the default "New" item rather than spawn windows.
        CommandGroup(replacing: .newItem) {}

        CommandGroup(after: .appInfo) {
            MereRunCheckForUpdatesView(updater: updater)
        }

        // View toggles act on whichever Studio window is key (via focused scene values). When no
        // Studio window holds focus all three are nil, so the group (and its divider) stay empty.
        CommandGroup(after: .sidebar) {
            if showLibrary != nil || showAdvanced != nil || showModels != nil {
                if let showLibrary {
                    Toggle("Show Library", isOn: showLibrary)
                        .keyboardShortcut("l", modifiers: [.command, .control])
                }
                if let showAdvanced {
                    Toggle("Show Advanced", isOn: showAdvanced)
                        .keyboardShortcut("e", modifiers: [.command, .control])
                }
                if let showModels {
                    Button("Browse Models…") { showModels.wrappedValue = true }
                        .keyboardShortcut("m", modifiers: [.command, .shift])
                }
                Divider()
            }
        }

        CommandMenu("Run") {
            Button("Run") {
                controller.run()
            }
            .disabled(controller.isRunning)

            Button("Stop") {
                controller.cancel()
            }
            .keyboardShortcut(".", modifiers: .command)
            .disabled(!controller.isRunning)

            Divider()

            Button("Open Last Output") {
                controller.openLastOutput()
            }
            .keyboardShortcut("o", modifiers: [.command, .shift])
            .disabled(controller.lastOutputURL == nil)

            Button("Reveal Last Output in Finder") {
                controller.revealLastOutput()
            }
            .keyboardShortcut("r", modifiers: [.command, .shift])
            .disabled(controller.lastOutputURL == nil)
        }

        CommandGroup(replacing: .help) {
            Link("mere.run", destination: URL(string: "https://mere.run")!)
        }
    }
}
