import AppKit
import Sparkle
import StudioKit
import StudioUI
import SwiftUI
import UniformTypeIdentifiers

/// Top-level menu bar commands. Everything that acts on a window goes through the
/// `StudioSceneActions` the key window publishes as a focused scene value: the Studio window
/// publishes its composer, Library, and navigation; the Command Console publishes its own
/// Run/Stop and forwards navigation to the Studio window. Items stay disabled when no window
/// of ours is key.
struct MereRunCommands: Commands {
    @ObservedObject var controller: MereRunController
    @ObservedObject var library: StudioLibraryStore
    let updater: SPUUpdater

    @FocusedValue(\.studioActions) private var actions: StudioSceneActions?

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
        // Single-window studio: "New" starts a chat thread instead of spawning windows.
        CommandGroup(replacing: .newItem) {
            Button("New Chat") {
                actions?.newChat()
            }
            .keyboardShortcut("n", modifiers: .command)
            .disabled(actions?.canNewChat != true)

            Divider()

            Button("Import Receipt…") {
                actions?.importReceipt()
            }
            .disabled(actions == nil)
        }

        CommandGroup(after: .appInfo) {
            MereRunCheckForUpdatesView(updater: updater)
        }

        // The system Show/Hide Sidebar item already lives in this group (NavigationSplitView owns it).
        CommandGroup(after: .sidebar) {
            Toggle("Show Library", isOn: actions?.showLibrary ?? .constant(false))
                .keyboardShortcut("l", modifiers: [.command, .option])
                .disabled(actions?.canShowLibrary != true)

            Toggle("Show Inspector", isOn: actions?.showInspector ?? .constant(false))
                .keyboardShortcut("i", modifiers: [.command, .option])
                .disabled(actions?.canShowInspector != true)

            // On a prompt task ⌥⌘C flips the Command view column; elsewhere the raw surface is
            // still the Command Console window until every task renders its own Command view.
            if actions?.canShowCommand == true {
                Toggle("Show Command View", isOn: actions?.showCommand ?? .constant(false))
                    .keyboardShortcut("c", modifiers: [.command, .option])
            } else {
                Button("Command Console") {
                    actions?.openConsole()
                }
                .keyboardShortcut("c", modifiers: [.command, .option])
                .disabled(actions == nil)
            }

            Divider()
        }

        CommandMenu("Go") {
            ForEach(StudioDomainGroup.allCases) { group in
                ForEach(group.domains) { domain in
                    domainItem(domain)
                }
                if group != StudioDomainGroup.allCases.last {
                    Divider()
                }
            }

            if let actions, actions.destination.domain.tasks.count > 1 {
                Divider()
                ForEach(actions.destination.domain.tasks) { task in
                    Toggle(
                        task.title,
                        isOn: Binding(
                            get: { task == actions.destination.task },
                            set: { isOn in
                                if isOn { actions.open(task.destination) }
                            }
                        )
                    )
                }
            }
        }

        CommandMenu("Run") {
            Button("Run") {
                actions?.runComposer()
            }
            .keyboardShortcut(.return, modifiers: .command)
            .disabled(actions?.canRun != true)

            Button("Stop") {
                actions?.stop()
            }
            .keyboardShortcut(".", modifiers: .command)
            .disabled(actions?.canStop != true)

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
            Button("mere.run Guide") {
                actions?.showGuide()
            }
            .keyboardShortcut("?", modifiers: .command)
            .disabled(actions == nil)

            Link("mere.run", destination: URL(string: "https://mere.run")!)
            Divider()
            Button("Command Console") {
                actions?.openConsole()
            }
            .disabled(actions == nil)
            Button("Export Diagnostics…") { exportDiagnostics() }
        }
    }

    @ViewBuilder
    private func domainItem(_ domain: StudioDomain) -> some View {
        let button = Button {
            actions?.openDomain(domain)
        } label: {
            Label(domain.title, systemImage: domain.systemImage)
        }
        .disabled(actions == nil)

        if let shortcut = domain.keyboardShortcut {
            button.keyboardShortcut(shortcut)
        } else {
            button
        }
    }
}
