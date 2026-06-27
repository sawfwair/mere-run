import SwiftUI

/// Top-level menu bar commands. View-toggle commands (Library/Advanced/Models) are
/// installed by `StudioRootView` via focused scene values once the UI state is shared.
struct MereRunCommands: Commands {
    @ObservedObject var controller: MereRunController

    var body: some Commands {
        // Single-window studio: remove the default "New" item rather than spawn windows.
        CommandGroup(replacing: .newItem) {}

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
