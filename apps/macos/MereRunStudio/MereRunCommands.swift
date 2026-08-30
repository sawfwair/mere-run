import AppKit
import Sparkle
import SwiftUI
import UniformTypeIdentifiers

/// Top-level menu bar commands. View-toggle commands (Library/Advanced/Models) are
/// installed by `StudioRootView` via focused scene values once the UI state is shared.
struct MereRunCommands: Commands {
    @ObservedObject var controller: MereRunController
    @ObservedObject var library: StudioLibraryStore
    let updater: SPUUpdater

    @FocusedValue(\.showLibrary) private var showLibrary: Binding<Bool>?
    @FocusedValue(\.showAdvanced) private var showAdvanced: Binding<Bool>?
    @FocusedValue(\.showModels) private var showModels: Binding<Bool>?
    @FocusedValue(\.showOperations) private var showOperations: Binding<Bool>?
    @FocusedValue(\.showPlugins) private var showPlugins: Binding<Bool>?

    /// Writes a secret-free support report the user can attach to an issue.
    private func exportDiagnostics() {
        let report = controller.diagnosticsReport(libraryItems: library.items)
        guard let url = StudioSpecialistFiles.saveFile(
            title: "Export diagnostics",
            suggestedName: StudioDiagnostics.suggestedFilename(),
            allowedContentTypes: [.plainText]
        ) else { return }
        do {
            try report.write(to: url, atomically: true, encoding: .utf8)
            NSWorkspace.shared.activateFileViewerSelecting([url])
        } catch {
            NSApp.presentError(error)
        }
    }

    var body: some Commands {
        // Single-window studio: remove the default "New" item rather than spawn windows.
        CommandGroup(replacing: .newItem) {}

        CommandGroup(after: .appInfo) {
            MereRunCheckForUpdatesView(updater: updater)
        }

        // View toggles act on whichever Studio window is key (via focused scene values). When no
        // Studio window holds focus all three are nil, so the group (and its divider) stay empty.
        CommandGroup(after: .sidebar) {
            if showLibrary != nil || showAdvanced != nil || showModels != nil
                || showOperations != nil || showPlugins != nil {
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
                if let showOperations {
                    Button("Runs & Operations…") { showOperations.wrappedValue = true }
                        .keyboardShortcut("r", modifiers: [.command, .control])
                }
                if let showPlugins {
                    Button("Plugins…") { showPlugins.wrappedValue = true }
                        .keyboardShortcut("p", modifiers: [.command, .shift])
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
            Divider()
            Button("Export Diagnostics…") { exportDiagnostics() }
        }
    }
}
