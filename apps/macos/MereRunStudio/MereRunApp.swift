import AppKit
import Quartz
import Sparkle
import StudioKit
import StudioUI
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
    @StateObject private var session = StudioAppSession()
    private var controller: MereRunController { session.controller }
    private var library: StudioLibraryStore { session.library }
    @StateObject private var navigation = NavigationModel()
    @StateObject private var crashReporter = StudioCrashReporter()
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
                            controller?.taskSessions.flush()
                            controller?.servingMonitor.stop()
                            controller?.terminateAllProcesses()
                        }
                    }
                }
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
                updater: updaterController.updater
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

        Settings {
            MereRunSettingsView()
                .environmentObject(controller)
                .environmentObject(crashReporter)
                .frame(width: 560)
        }
    }
}
