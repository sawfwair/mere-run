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

@main
struct MereRunApp: App {
    @NSApplicationDelegateAdaptor(MereRunAppDelegate.self) private var appDelegate
    @StateObject private var controller = MereRunController()
    @StateObject private var library = StudioLibraryStore()
    @StateObject private var navigation = StudioNavigationCoordinator()
    @State private var deepLinkError: String?
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
                    appDelegate.onTerminate = { [weak controller] in
                        MainActor.assumeIsolated {
                            controller?.terminateAllProcesses()
                        }
                    }
                }
                .onOpenURL(perform: openDeepLink)
                .alert(
                    "Couldn’t open MereRun link",
                    isPresented: Binding(
                        get: { deepLinkError != nil },
                        set: { if !$0 { deepLinkError = nil } }
                    )
                ) {
                    Button("OK") { deepLinkError = nil }
                } message: {
                    Text(deepLinkError ?? "The MereRun link is invalid.")
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unified(showsTitle: false))
        .windowResizability(.contentMinSize)
        .defaultSize(width: 1280, height: 820)
        .commands {
            MereRunCommands(
                controller: controller,
                updater: updaterController.updater
            )
        }

        Settings {
            MereRunSettingsView()
                .environmentObject(controller)
                .frame(width: 560)
        }
    }

    private func openDeepLink(_ url: URL) {
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
                navigation.openLibraryItem(id: item.id, mode: item.mode)
                deepLinkError = nil
            }
        } catch {
            deepLinkError = error.localizedDescription
        }
    }
}
