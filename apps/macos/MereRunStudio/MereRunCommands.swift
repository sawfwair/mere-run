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
/// of ours is key. Every key comes from `StudioKeyboardShortcuts`, which Help ▸ Keyboard
/// Shortcuts lists.
struct MereRunCommands: Commands {
    @ObservedObject var controller: MereRunController
    @ObservedObject var library: StudioLibraryStore
    @ObservedObject var navigation: NavigationModel
    let updater: SPUUpdater
    let isStudioOpen: Bool

    @Environment(\.openWindow) private var openWindow
    @FocusedValue(\.studioActions) private var actions: StudioSceneActions?
    @FocusedValue(\.studioLibraryDeletion) private var libraryDeletion: StudioLibraryDeletion?

    /// The output of the run the Library has selected, for File ▸ Quick Look.
    private var selectedOutput: URL? {
        library.items.first { $0.id == navigation.selectedLibraryID }?.outputURL
    }

    /// The list Edit ▸ Find in List searches on the current page.
    private var findField: StudioSearchField? {
        actions.flatMap { StudioKeyboardShortcuts.findField(for: $0.destination.task) }
    }

    /// Focuses a list's search, showing the Library column first when that is the list.
    private func focusSearch(_ field: StudioSearchField) {
        if field == .library, let actions, !actions.showLibrary.wrappedValue {
            actions.showLibrary.wrappedValue = true
        }
        navigation.requestSearchFocus(field)
    }

    /// Writes a secret-free support report the user can attach to an issue.
    private func exportDiagnostics() {
        let report = controller.diagnosticsReport(libraryItems: library.items)
        guard let url = StudioFilePanels.saveFile(
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
            .studioShortcut(.newChat)
            .disabled(actions?.canNewChat != true)

            Divider()

            Button("Quick Look") {
                if let selectedOutput { QuickLookCoordinator.shared.preview(selectedOutput) }
            }
            .studioShortcut(.quickLook)
            .disabled(actions == nil || selectedOutput == nil)

            // Enabled only while the Library list has focus, so ⌘⌫ in a text field still
            // deletes to the start of the line.
            Button(libraryDeletion?.title ?? "Delete from Library…") {
                libraryDeletion?.confirm()
            }
            .studioShortcut(.deleteFromLibrary)
            .disabled(libraryDeletion == nil)

            Divider()

            Button("Import Receipt…") {
                actions?.importReceipt()
            }
            .disabled(actions == nil)
        }

        CommandGroup(after: .pasteboard) {
            Divider()
            Button("Find in List") {
                if let findField { focusSearch(findField) }
            }
            .studioShortcut(.find)
            .disabled(findField == nil)

            Button("Search Library") {
                focusSearch(.library)
            }
            .studioShortcut(.searchLibrary)
            .disabled(actions?.canShowLibrary != true)
        }

        CommandGroup(after: .appInfo) {
            MereRunCheckForUpdatesView(updater: updater)
        }


        CommandGroup(before: .windowList) {
            Button("Open Studio") { openWindow(id: "studio") }
                .disabled(isStudioOpen)
            // The Console carries the current task's command when a Studio window can hand it one;
            // from anywhere else it opens as it was left.
            Button("Command Console") {
                if let actions { actions.openConsole() } else { openWindow(id: StudioConsoleWindow.id) }
            }
            .studioShortcut(.commandConsole)
            Divider()
        }

        // The system Show/Hide Sidebar item already lives in this group (NavigationSplitView owns it).
        CommandGroup(after: .sidebar) {
            Toggle("Show Library", isOn: actions?.showLibrary ?? .constant(false))
                .studioShortcut(.showLibrary)
                .disabled(actions?.canShowLibrary != true)

            Toggle("Show Inspector", isOn: actions?.showInspector ?? .constant(false))
                .studioShortcut(.showInspector)
                .disabled(actions?.canShowInspector != true)

            // ⌥⌘C is always the task's Command view; the Console window is Window ▸ Command Console.
            Toggle("Show Command View", isOn: actions?.showCommand ?? .constant(false))
                .studioShortcut(.showCommandView)
                .disabled(actions?.canShowCommand != true)

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
            .studioShortcut(.run)
            .disabled(actions?.canRun != true)

            Button("Stop") {
                actions?.stop()
            }
            .studioShortcut(.stop)
            .disabled(actions?.canStop != true)

            Divider()

            Button("Open Last Output") {
                controller.openLastOutput()
            }
            .studioShortcut(.openLastOutput)
            .disabled(controller.lastOutputURL == nil)

            Button("Reveal Last Output in Finder") {
                controller.revealLastOutput()
            }
            .studioShortcut(.revealLastOutput)
            .disabled(controller.lastOutputURL == nil)
        }

        CommandGroup(replacing: .help) {
            Button("mere.run Guide") {
                actions?.showGuide()
            }
            .studioShortcut(.guide)
            .disabled(actions == nil)

            Button("Keyboard Shortcuts") {
                openWindow(id: StudioKeyboardShortcutsWindow.id)
            }

            Link("mere.run", destination: URL(string: "https://mere.run")!)
            Divider()
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
