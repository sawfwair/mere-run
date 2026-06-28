import AppKit
import Quartz
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

@main
struct MereRunApp: App {
    @NSApplicationDelegateAdaptor(MereRunAppDelegate.self) private var appDelegate
    @StateObject private var controller = MereRunController()

    var body: some Scene {
        WindowGroup {
            MereRunRootView()
                .environmentObject(controller)
                .frame(minWidth: 880, minHeight: 600)
                .onAppear {
                    appDelegate.onTerminate = { [weak controller] in
                        MainActor.assumeIsolated {
                            controller?.terminateAllProcesses()
                        }
                    }
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unified(showsTitle: false))
        .windowResizability(.contentMinSize)
        .defaultSize(width: 1280, height: 820)
        .commands {
            MereRunCommands(controller: controller)
        }

        Settings {
            MereRunSettingsView()
                .environmentObject(controller)
                .frame(width: 560)
        }
    }
}
