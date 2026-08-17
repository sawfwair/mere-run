import SwiftUI

@main
struct MereRunStudioApp: App {
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

    var body: some View {
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
                Section {
                    Button("Sign out and unpair", role: .destructive) {
                        relay.unpair()
                    }
                } footer: {
                    Text("Work runs on your paired nodes. Prompts and outputs move only between this phone and your relay.")
                }
            }
            .scrollContentBackground(.hidden)
            .background(MereTheme.background.ignoresSafeArea())
            .navigationTitle("Settings")
            .onAppear { relay.refreshAuthStatus() }
        }
    }
}
