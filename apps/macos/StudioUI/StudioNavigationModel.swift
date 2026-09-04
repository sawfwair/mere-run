import AppKit
import Combine
import StudioKit
import SwiftUI

extension StudioDomain {
    /// ⌘1…⌘9 for the first nine domains, ⌥⌘1… for the rest, in sidebar order.
    package var keyboardShortcut: KeyboardShortcut? {
        guard let index = Self.allCases.firstIndex(of: self) else { return nil }
        if index < 9, let key = "\(index + 1)".first {
            return KeyboardShortcut(KeyEquivalent(key), modifiers: .command)
        }
        let offset = index - 9
        guard offset < 9, let key = "\(offset + 1)".first else { return nil }
        return KeyboardShortcut(KeyEquivalent(key), modifiers: [.command, .option])
    }
}

/// The Studio's navigation state: the destination, the selected Library row, the Library column's
/// visibility, and whether the Command Console window is open, plus the entry points that used
/// to be scattered across the root view, the App's deep-link handler, and
/// `StudioNavigationCoordinator`. It is app-level so the Console scene and the menu bar can act
/// on the Studio window; the root view mirrors what must persist into `@SceneStorage`.
@MainActor
package final class NavigationModel: ObservableObject {
    @Published package var destination: StudioDestination
    @Published package var selectedLibraryID: UUID?
    @Published package var showLibrary = true
    @Published package var deepLinkError: String?
    /// The task last shown in each domain, so returning to a domain lands where you left it.
    @Published package private(set) var rememberedTasks: [StudioDomain: StudioTask] = [:]
    /// The Vision Lab variant its rail shows (Faces covers detect/embed/compare/batch, Geometry
    /// covers single and multi-view); kept in step with the Vision toolbar task.
    @Published private(set) var visionLabTask: StudioVisionTask = .faceDetect
    /// Set by the Command Console scene while its window exists. Opening the console syncs the
    /// composer draft only when this is false, so raising an open console never clobbers edits.
    @Published package var isConsoleOpen = false
    /// The prompt tasks whose inspector column is shown. Remembered per task, so Image ▸ Generate
    /// can keep its inspector open while Chat stays a plain thread.
    @Published package var inspectorTasks: Set<StudioTask> = []
    /// Whether the Command view column is shown for the current prompt task. It takes the
    /// inspector's place while open (the two are never side by side) and is not remembered.
    @Published package var showCommandColumn = false
    /// Help ▸ mere.run Guide presents the Guide sheet on the Studio window from any key window.
    @Published package var showGuide = false
    /// Whether the sidebar footer's Activity popover is open. The shell draws it over the window,
    /// so the state lives here rather than inside the sidebar column.
    @Published package var showActivity = false

    package init(destination: StudioDestination = .default) {
        self.destination = destination
        rememberedTasks[destination.domain] = destination.task
        if let variant = destination.task.visionLabTask { visionLabTask = variant }
    }

    func open(destination: StudioDestination) {
        rememberedTasks[destination.domain] = destination.task
        if let variant = destination.task.visionLabTask, visionLabTask.studioTask != destination.task {
            visionLabTask = variant
        }
        guard destination != self.destination else { return }
        self.destination = destination
        // The Command view belongs to the task it was opened on; a new task starts closed.
        showCommandColumn = false
    }

    /// Whether opening the Console should carry the composer's draft into it: only when the
    /// caller asks for it and no console window already holds the user's edits.
    package func shouldSyncComposerToConsole(requested: Bool) -> Bool {
        requested && !isConsoleOpen
    }

    /// The Vision Lab rail picked a variant: show it and move the toolbar task to its group.
    func selectVisionLabVariant(_ variant: StudioVisionTask) {
        visionLabTask = variant
        open(task: variant.studioTask)
    }

    /// Scene restore: applies the persisted destination through `open` so remembered tasks and
    /// the Vision Lab variant learn it, and returns the prompt mode the composer should hold —
    /// the destination's own mode when it has one, otherwise the persisted last prompt mode.
    @discardableResult
    package func restore(destination: StudioDestination, lastPromptMode: StudioMode) -> StudioMode {
        open(destination: destination)
        return destination.task.mode ?? lastPromptMode
    }

    /// Whether the inspector column is shown for `task`: remembered on, and not displaced by the
    /// Command view column.
    package func showsInspector(for task: StudioTask) -> Bool {
        task.isPromptTask && inspectorTasks.contains(task) && !showCommandColumn
    }

    /// Whether the Command view column is shown for `task`.
    package func showsCommandColumn(for task: StudioTask) -> Bool {
        task.isPromptTask && showCommandColumn
    }

    /// Shows or hides the inspector for `task`. Showing it closes the Command view column.
    package func toggleInspector(for task: StudioTask) {
        guard task.isPromptTask else { return }
        if showsInspector(for: task) {
            inspectorTasks.remove(task)
        } else {
            inspectorTasks.insert(task)
            showCommandColumn = false
        }
    }

    /// Shows or hides the Command view column; the inspector's memory for the task survives.
    package func toggleCommandColumn(for task: StudioTask) {
        guard task.isPromptTask else { return }
        showCommandColumn.toggle()
    }

    func open(domain: StudioDomain) {
        if destination.domain == domain { return }
        open(destination: StudioDestination(domain: domain, task: rememberedTasks[domain] ?? domain.defaultTask))
    }

    func open(task: StudioTask) {
        open(destination: task.destination)
    }

    /// Shows a Library row in its own domain with the Library column visible.
    func open(libraryItem id: UUID, mode: StudioMode) {
        selectedLibraryID = id
        showLibrary = true
        open(destination: mode.destination)
    }

    /// Handles a `mererun://` link: preview shows Quick Look; library import validates the
    /// receipt through the store, then navigates to the imported row.
    func open(deepLink url: URL, library: StudioLibraryStore) {
        do {
            switch try MereRunDeepLink.parse(url) {
            case .preview(let artifactURL):
                NSApplication.shared.activate(ignoringOtherApps: true)
                guard QuickLookCoordinator.shared.preview(artifactURL) else {
                    deepLinkError = "Quick Look is unavailable."
                    return
                }
                deepLinkError = nil
            case .libraryImport(let receiptURL):
                let item = try library.importReceipt(at: receiptURL)
                NSApplication.shared.activate(ignoringOtherApps: true)
                open(libraryItem: item.id, mode: item.mode)
                deepLinkError = nil
            }
        } catch {
            deepLinkError = error.localizedDescription
        }
    }
}
