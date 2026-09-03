import AppKit
import Quartz
import Sparkle
import SwiftUI

final class MereRunAppDelegate: NSObject, NSApplicationDelegate {
    var onTerminate: (() -> Void)?

    func applicationWillTerminate(_ notification: Notification) {
        onTerminate?()
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

/// The Command Console: the raw contract surface in its own resizable window, opened from the
/// toolbar, View ▸ Command Console, readiness "Details", Library "Edit command", and the adapter
/// fallbacks. There is no docked or detached variant any more.
enum StudioConsoleWindow {
    static let id = "console"
    static let title = "Command Console"
}

@main
struct MereRunApp: App {
    @NSApplicationDelegateAdaptor(MereRunAppDelegate.self) private var appDelegate
    @StateObject private var controller = MereRunController()
    @StateObject private var library = StudioLibraryStore()
    @StateObject private var navigation = NavigationModel()
    @StateObject private var crashReporter = StudioCrashReporter()
    private let updaterController = SPUStandardUpdaterController(
        startingUpdater: true,
        updaterDelegate: nil,
        userDriverDelegate: nil
    )

    var body: some Scene {
        WindowGroup {
            MereRunRootView()
                .environmentObject(controller)
                .environmentObject(library)
                .environmentObject(navigation)
                .frame(
                    minWidth: StudioLayoutPolicy.minimumWindowWidth,
                    minHeight: StudioLayoutPolicy.minimumWindowHeight
                )
                .onAppear {
                    crashReporter.applyStoredPreference()
                    appDelegate.onTerminate = { [weak controller] in
                        MainActor.assumeIsolated {
                            controller?.terminateAllProcesses()
                        }
                    }
                }
                .task {
                    await controller.synchronizeCLIInstallationAfterLaunch()
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unified(showsTitle: false))
        .windowResizability(.contentMinSize)
        .defaultSize(
            width: StudioLayoutPolicy.defaultWindowWidth,
            height: StudioLayoutPolicy.defaultWindowHeight
        )
        .commands {
            MereRunCommands(
                controller: controller,
                library: library,
                updater: updaterController.updater
            )
        }

        Window(StudioConsoleWindow.title, id: StudioConsoleWindow.id) {
            AdvancedControlSurface()
                .environmentObject(controller)
                .environmentObject(library)
                .environmentObject(navigation)
                .frame(minWidth: 960, minHeight: 560)
        }
        .defaultSize(width: 1_260, height: 780)
        .windowResizability(.contentMinSize)

        Settings {
            MereRunSettingsView()
                .environmentObject(controller)
                .environmentObject(crashReporter)
                .frame(width: 560)
        }
    }
}
