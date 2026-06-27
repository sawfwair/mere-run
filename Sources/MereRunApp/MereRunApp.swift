import AppKit
import SwiftUI

final class MereRunAppDelegate: NSObject, NSApplicationDelegate {
    var onTerminate: (() -> Void)?

    func applicationWillTerminate(_ notification: Notification) {
        onTerminate?()
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
