import MereRunCore
import SwiftUI
import UIKit

final class StudioAppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        guard HubBackgroundTransferSession.handleEvents(
            for: identifier,
            completionHandler: completionHandler
        ) else {
            completionHandler()
            return
        }
        Task { @MainActor in
            LocalEngine.shared.resumePendingDownloads()
        }
    }
}

@main
struct MereRunStudioApp: App {
    @UIApplicationDelegateAdaptor(StudioAppDelegate.self) private var appDelegate
    @StateObject private var relay = RelayStore()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(relay)
                .tint(MereTheme.accent)
        }
    }
}

struct RootView: View {
    @EnvironmentObject private var relay: RelayStore
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        Group {
            if relay.pairing == .paired {
                TabView {
                    CreateView()
                        .tabItem { Label("Create", systemImage: "sparkles") }
                    ChatView(relay: relay)
                        .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }
                    RunsView()
                        .tabItem { Label("Runs", systemImage: "tray.full") }
                    FleetView()
                        .tabItem { Label("Fleet", systemImage: "server.rack") }
                    SettingsView()
                        .tabItem { Label("Settings", systemImage: "gearshape") }
                }
                .background(MereTheme.background)
            } else {
                PairingView()
            }
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .active:
                LocalEngine.shared.resumePendingDownloads()
            case .background:
                LocalEngine.shared.releaseRuntime()
            case .inactive:
                break
            @unknown default:
                break
            }
        }
        .onReceive(NotificationCenter.default.publisher(
            for: UIApplication.didReceiveMemoryWarningNotification
        )) { _ in
            LocalEngine.shared.releaseRuntime()
        }
    }
}

struct SettingsView: View {
    @EnvironmentObject private var relay: RelayStore

    var body: some View {
        NavigationStack {
            List {
                Section("Relay") {
                    LabeledContent("Account", value: relay.accountEmail ?? "—")
                    LabeledContent("Profile", value: relay.profile?.name ?? "—")
                    LabeledContent("URL", value: relay.profile?.url ?? "—")
                    LabeledContent(
                        "Signed in",
                        value: relay.authStatus?.authenticated == true ? "Yes" : "No"
                    )
                }
                if LocalEngine.showsOnDeviceUI {
                    Section("On this iPhone") {
                        NavigationLink("On-device models") {
                            OnDeviceModelsList()
                        }
                    }
                }
                Section {
                    Button("Sign out and unpair", role: .destructive) {
                        relay.unpair()
                    }
                } footer: {
                    Text(ExecutionPrivacyCopy.create(for: relay.executionPrivacyLane))
                }
            }
            .scrollContentBackground(.hidden)
            .background(MereTheme.background.ignoresSafeArea())
            .navigationTitle("Settings")
            .onAppear { relay.refreshAuthStatus() }
        }
    }
}
