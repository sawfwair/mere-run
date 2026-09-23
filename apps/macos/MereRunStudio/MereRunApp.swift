import AppKit
import Quartz
import Sparkle
import StudioKit
import StudioUI
import SwiftUI

final class MereRunAppDelegate: NSObject, NSApplicationDelegate {
    /// The app's shared services. The delegate owns them, rather than a window or a scene, so Quit
    /// can ask about and stop the child processes whether or not a Studio window ever appeared.
    @MainActor lazy var session = StudioAppSession()

    /// Every child process ends with the app, so Quit asks first while a server Studio started or
    /// the user's own work is still running.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let warning = StudioQuitWarning.message(for: session.controller) else { return .terminateNow }
        let alert = NSAlert()
        alert.messageText = "Quit mere.run?"
        alert.informativeText = warning
        alert.addButton(withTitle: "Quit")
        alert.addButton(withTitle: "Cancel")
        return alert.runModal() == .alertFirstButtonReturn ? .terminateNow : .terminateCancel
    }

    func applicationWillTerminate(_ notification: Notification) {
        let controller = session.controller
        controller.taskSessions.flush()
        controller.servingMonitor.stop()
        controller.terminateAllProcesses()
    }

    // QLPreviewPanelController: the app delegate is the end of the responder chain, so it answers
    // the panel's control handshake and supplies QuickLookCoordinator as the data source.
    override func acceptsPreviewPanelControl(_ panel: QLPreviewPanel!) -> Bool { true }

    override func beginPreviewPanelControl(_ panel: QLPreviewPanel!) {
        MainActor.assumeIsolated { QuickLookCoordinator.shared.install(on: panel) }
    }

    override func endPreviewPanelControl(_ panel: QLPreviewPanel!) {
        MainActor.assumeIsolated { panel.dataSource = nil }
    }
}

/// The menu bar extra's panel, with the two ways back into the Studio window.
struct MereRunMenuBarContent: View {
    let controller: MereRunController
    let navigation: NavigationModel
    let isStudioOpen: Bool

    @Environment(\.openWindow) private var openWindow

    var body: some View {
        StudioMenuBarPanel(
            controller: controller,
            onOpenStudio: showStudio,
            onOpenServer: {
                navigation.open(task: .serverServing, windowIsOpen: isStudioOpen)
                showStudio()
            }
        )
    }

    /// Brings the Studio window forward, or opens it when it is closed. `openWindow` always adds a
    /// window to a `WindowGroup`, so an open one is raised rather than opened again.
    private func showStudio() {
        NSApp.activate()
        guard isStudioOpen else {
            openWindow(id: "studio")
            return
        }
        let window = NSApp.windows.first { $0.identifier?.rawValue.hasPrefix("studio") == true }
        if window?.isMiniaturized == true { window?.deminiaturize(nil) }
        window?.makeKeyAndOrderFront(nil)
    }
}

/// The Studio window's content: the shell, on the theme, with the CLI resolved once the window
/// is up so the status cluster and the composer know what they are talking to.
struct MereRunRootView: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        StudioRootView()
            .background(MereRunTheme.background.ignoresSafeArea())
            .foregroundStyle(MereRunTheme.textPrimary)
            .onAppear {
                controller.refreshResolvedCLI()
                controller.refreshCLIVersion()
            }
    }
}

@main
struct MereRunApp: App {
    @NSApplicationDelegateAdaptor(MereRunAppDelegate.self) private var appDelegate
    private var session: StudioAppSession { appDelegate.session }
    @State private var isStudioOpen = false
    private var controller: MereRunController { session.controller }
    private var library: StudioLibraryStore { session.library }
    @StateObject private var navigation = NavigationModel()
    @StateObject private var crashReporter = StudioCrashReporter()
    @AppStorage(StudioMenuBar.visibilityDefaultsKey) private var showsMenuBarExtra = true
    private let updaterController = SPUStandardUpdaterController(
        startingUpdater: true,
        updaterDelegate: nil,
        userDriverDelegate: nil
    )

    init() {
        // The wordmark's face must be registered before the first window draws; a missing bundle
        // degrades to the system serif rather than failing launch.
        MereRunTheme.Brand.register()
    }

    var body: some Scene {
        WindowGroup(id: "studio") {
            MereRunRootView()
                .environmentObject(controller)
                .environmentObject(library)
                .environmentObject(navigation)
                .frame(
                    minWidth: StudioLayoutPolicy.minimumWindowWidth,
                    minHeight: StudioLayoutPolicy.minimumWindowHeight
                )
                .onAppear {
                    isStudioOpen = true
                    crashReporter.applyStoredPreference()
                }
                .onDisappear { isStudioOpen = false }
                .task {
                    await controller.synchronizeCLIInstallationAfterLaunch()
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentMinSize)
        .defaultSize(
            width: StudioLayoutPolicy.defaultWindowWidth,
            height: StudioLayoutPolicy.defaultWindowHeight
        )
        .commands {
            MereRunCommands(
                controller: controller,
                library: library,
                updater: updaterController.updater,
                isStudioOpen: isStudioOpen
            )
        }

        Window(StudioConsoleWindow.title, id: StudioConsoleWindow.id) {
            StudioConsoleView()
                .environmentObject(controller)
                .environmentObject(library)
                .environmentObject(navigation)
                .frame(minWidth: 960, minHeight: 560)
        }
        .defaultSize(width: 1_260, height: 780)
        .windowResizability(.contentMinSize)

        // The server outlives the Studio window, so its control does too.
        MenuBarExtra(isInserted: $showsMenuBarExtra) {
            MereRunMenuBarContent(controller: controller, navigation: navigation, isStudioOpen: isStudioOpen)
        } label: {
            StudioMenuBarLabel(server: controller.localServer)
        }
        .menuBarExtraStyle(.window)

        Settings {
            MereRunSettingsView()
                .environmentObject(controller)
                .environmentObject(crashReporter)
                .frame(width: 560)
        }
    }
}
