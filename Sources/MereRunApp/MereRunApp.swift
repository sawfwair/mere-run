import SwiftUI

@main
struct MereRunApp: App {
    @StateObject private var controller = MereRunController()

    var body: some Scene {
        WindowGroup {
            MereRunRootView()
                .environmentObject(controller)
        }
        .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unified(showsTitle: false))
        .defaultSize(width: 1280, height: 820)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Stop") {
                    controller.cancel()
                }
                .keyboardShortcut(".", modifiers: .command)
                .disabled(!controller.isRunning)
            }
        }

        Settings {
            MereRunSettingsView()
                .environmentObject(controller)
                .frame(width: 560)
        }
    }
}
