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

    func applicationDidFinishLaunching(_ notification: Notification) {
        // A window closing, a window coming forward, or a menu bar setting changing can each move
        // the app in or out of the Dock; check once the change has landed.
        for name in [NSWindow.willCloseNotification, NSWindow.didBecomeKeyNotification, UserDefaults.didChangeNotification] {
            NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) { _ in
                DispatchQueue.main.async { MainActor.assumeIsolated { Self.updateDockPresence() } }
            }
        }
        if UserDefaults.standard.bool(forKey: StudioMenuBar.serveAtLaunchDefaultsKey) {
            let controller = session.controller
            Task { @MainActor in
                // Something may already be serving on the endpoint; start only when nothing answers.
                await controller.servingMonitor.refreshRuntimeNow(controller: controller)
                guard !controller.localServer.phase.isServing,
                      let refusal = controller.localServer.start() else { return }
                controller.servingMonitor.note("API server did not start at launch", detail: refusal, level: .warning)
            }
        }
    }

    /// Set once a Studio, Console, or Settings window has been on screen. Until then a defaults
    /// write at launch — Sparkle's, the crash reporter's — would find no window yet and send the
    /// app out of the Dock just before its first window appears.
    @MainActor private static var hasShownWindow = false

    /// With the menu bar extra in place to reopen a window, mere.run leaves the Dock while no window
    /// is open and comes back when one opens.
    @MainActor
    private static func updateDockPresence() {
        // A minimized window is still open: it is in the Dock, so the app must be too.
        let hasOpenWindow = NSApp.windows.contains { ($0.isVisible || $0.isMiniaturized) && $0.canBecomeMain }
        if hasOpenWindow { hasShownWindow = true }
        let hides = hasShownWindow && StudioMenuBar.hidesDockIcon(
            showsMenuBarExtra: StudioMenuBar.isOn(StudioMenuBar.visibilityDefaultsKey),
            hidesWithoutWindows: StudioMenuBar.isOn(StudioMenuBar.hidesDockIconDefaultsKey),
            hasOpenWindow: hasOpenWindow
        )
        let policy: NSApplication.ActivationPolicy = hides ? .accessory : .regular
        guard NSApp.activationPolicy() != policy else { return }
        NSApp.setActivationPolicy(policy)
        // An accessory app that stays frontmost keeps its main menu in the menu bar with nothing
        // to act on; hand the menu bar to the app beneath.
        if policy == .accessory, NSApp.isActive { NSApp.hide(nil) }
    }

    /// Every child process ends with the app, so Quit asks first while a server Studio started or
    /// the user's own work is still running.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let warning = StudioQuitWarning.message(for: session.controller) else { return .terminateNow }
        let alert = NSAlert()
        alert.messageText = "Quit mere.run?"
        alert.informativeText = warning
        alert.addButton(withTitle: "Quit")
        alert.addButton(withTitle: "Cancel")
        // Quit from the menu bar arrives while another app is in front; bring the question forward.
        NSApp.activate()
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
        // Back into the Dock first, so the window opens as a regular app's window, in front.
        NSApp.setActivationPolicy(.regular)
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
