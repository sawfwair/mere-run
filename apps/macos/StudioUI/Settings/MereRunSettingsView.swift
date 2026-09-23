import AppKit
import ServiceManagement
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// The Settings window: General, Models, Server, and Advanced. Everything here is machine-wide
/// configuration — where the CLI is, where models and generations live, what the runtime server
/// answers on, and the local diagnostics — rather than anything about one run.
package struct MereRunSettingsView: View {
    package enum Tab: Hashable {
        case general, models, server, advanced
    }

    /// `tab` is where the window opens: General, unless a snapshot board asks for another.
    package init(tab: Tab = .general) {
        _tab = State(initialValue: tab)
    }

    @State private var tab: Tab

    @EnvironmentObject private var crashReporter: StudioCrashReporter
    @EnvironmentObject private var controller: MereRunController
    @State private var hfToken = ""
    @State private var hfStatus: String?
    @State private var hfEndpoint = ""
    @State private var hfEndpointStatus: String?
    @State private var configurationSummary = ""
    @State private var configurationPath = ""
    /// Empty means the per-media defaults in `StudioOutputLocation`.
    @AppStorage(StudioOutputLocation.rootDefaultsKey) private var outputRoot = ""
    @AppStorage(StudioMenuBar.visibilityDefaultsKey) private var showsMenuBarExtra = true
    @AppStorage(StudioMenuBar.hidesDockIconDefaultsKey) private var hidesDockIcon = true
    @AppStorage(StudioMenuBar.serveAtLaunchDefaultsKey) private var servesAtLaunch = false
    /// The login item's registration as macOS reports it; read on appear and after each change.
    @State private var loginItemStatus = SMAppService.Status.notRegistered
    @State private var loginItemError: String?
    /// The runtime endpoint and key as typed. They reach the controller — which retargets the
    /// server monitor and writes the key to the Keychain — on Apply or Return, not per keystroke.
    @State private var runtimeHost = ""
    @State private var runtimePort = 8_080
    @State private var runtimeAPIKey = ""

    package var body: some View {
        TabView(selection: $tab) {
            settingsTab { generalTab }
                .tabItem { Label("General", systemImage: "gearshape") }
                .tag(Tab.general)
            settingsTab { modelsTab }
                .tabItem { Label("Models", systemImage: "shippingbox") }
                .tag(Tab.models)
            settingsTab { serverTab }
                .tabItem { Label("Server", systemImage: "network") }
                .tag(Tab.server)
            settingsTab { advancedTab }
                .tabItem { Label("Advanced", systemImage: "wrench.and.screwdriver") }
                .tag(Tab.advanced)
        }
        .padding(22)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
    }

    private func settingsTab<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            content()
            Spacer(minLength: 0)
        }
        .padding(.top, 12)
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    @ViewBuilder
    private var generalTab: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("mere.run")
                .font(MereRunTheme.titleFont)
            Spacer()
            Text("App \(controller.appVersion) · CLI \(controller.cliVersion ?? "—")")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
        EditorSection("Command line") {
            PathField(label: "command line", path: $controller.cliPath, placeholder: "Auto-detect executable", mode: .openFile([.unixExecutable, .item]))
            Text("The app uses a bundled `mere.run` first, then nearby SwiftPM build products, common install locations, and the current package checkout.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .fixedSize(horizontal: false, vertical: true)
        }
        EditorSection("Working directory") {
            PathField(label: "working directory", path: $controller.workingDirectory, placeholder: "Same as the app", mode: .openDirectory)
        }
        EditorSection("Models root") {
            PathField(label: "models root", path: $controller.modelsRoot, placeholder: "Default: the managed model store", mode: .openDirectory)
            Text("A folder of model links or local model files to search before the managed store.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .fixedSize(horizontal: false, vertical: true)
        }
        EditorSection("Where generations are saved") {
            PathField(label: "output folder", path: $outputRoot, placeholder: outputRootPlaceholder, mode: .openDirectory)
            Text("Leave this empty to file work by what it is: pictures and clips in `~/Pictures/mere.run`, audio in `~/Music/mere.run`, everything else in `~/Documents/mere.run`, each under a folder named for the domain. Set a folder to keep every domain together there instead. Runs already in the Library keep the paths they recorded.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var outputRootPlaceholder: String {
        "~/Pictures/mere.run, ~/Music/mere.run, ~/Documents/mere.run"
    }

    @ViewBuilder
    private var modelsTab: some View {
        EditorSection("Hugging Face token") {
            HStack(spacing: 10) {
                SecureField("hf_… (for gated/private model pulls)", text: $hfToken)
                    .textFieldStyle(.plain)
                    .font(MereRunTheme.bodyFont)
                    .padding(10)
                    .merePanel()
                Button("Save") {
                    Task {
                        let ok = await controller.saveHuggingFaceToken(hfToken)
                        hfStatus = ok ? "Saved" : "Could not save token"
                        if ok { hfToken = "" }
                    }
                }
                .buttonStyle(.merePrimary)
            }
            if let hfStatus {
                Text(hfStatus)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
        EditorSection("Hugging Face endpoint") {
            HStack(spacing: 10) {
                TextField("https://huggingface.co (override mirror)", text: $hfEndpoint)
                    .textFieldStyle(.plain)
                    .font(MereRunTheme.bodyFont)
                    .padding(10)
                    .merePanel()
                Button("Save") {
                    Task {
                        let ok = await controller.saveHuggingFaceEndpoint(hfEndpoint)
                        hfEndpointStatus = ok ? "Saved" : "Could not save endpoint"
                    }
                }
                .buttonStyle(.merePrimary)
            }
            if let hfEndpointStatus {
                Text(hfEndpointStatus)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
        .task {
            hfEndpoint = await controller.loadHuggingFaceEndpoint()
        }
        EditorSection("Model payload storage") {
            PathField(label: "model payload storage", path: $controller.hubCache, placeholder: "Default: Application Support", mode: .openDirectory)
            Text("Downloads land here. Browsing, cleanup, and locations live in the Models domain.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
    }

    @ViewBuilder
    private var serverTab: some View {
        EditorSection("Runtime server") {
            HStack(spacing: 10) {
                TextField("Host", text: $runtimeHost)
                    .textFieldStyle(.plain)
                    .font(MereRunTheme.bodyFont)
                    .padding(10)
                    .merePanel()
                TextField("Port", value: $runtimePort, format: .number.grouping(.never))
                    .textFieldStyle(.plain)
                    .font(MereRunTheme.bodyFont)
                    .frame(width: 90)
                    .padding(10)
                    .merePanel()
                SecureField("API key (optional)", text: $runtimeAPIKey)
                    .textFieldStyle(.plain)
                    .font(MereRunTheme.bodyFont)
                    .padding(10)
                    .merePanel()
            }
            .onSubmit(applyRuntimeServer)
            // Only an edit has anything to apply or revert.
            if runtimeServerEdited {
                HStack(spacing: 10) {
                    Spacer(minLength: 0)
                    Button("Revert", action: loadRuntimeServer)
                        .buttonStyle(.mereSecondary)
                    Button("Apply", action: applyRuntimeServer)
                        .buttonStyle(.merePrimary)
                }
            }
            Text("Where the API server listens, and where Models, Server, and the menu bar reach it (`mere.run api serve`). The key is kept in your login Keychain.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            if let storageNotice = controller.runtimeAPIKeyStorageNotice {
                Text(storageNotice)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.yellow)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .onAppear(perform: loadRuntimeServer)
        // A change applied on the Server page replaces only the field it changed.
        .onChange(of: controller.runtimeHost) { _, host in runtimeHost = host }
        .onChange(of: controller.runtimePort) { _, port in runtimePort = port }
        .onChange(of: controller.runtimeAPIKey) { _, key in runtimeAPIKey = key }
        EditorSection("Menu bar and startup") {
            Toggle("Show mere.run in the menu bar", isOn: $showsMenuBarExtra)
            Text("Start, stop, and watch the servers, their models, and this Mac's load from the menu bar, with or without a Studio window open.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .fixedSize(horizontal: false, vertical: true)
            Toggle("Hide the Dock icon while no window is open", isOn: $hidesDockIcon)
                .disabled(!showsMenuBarExtra)
                .help("mere.run keeps running in the menu bar; opening the Studio brings the Dock icon back")
            Toggle("Start the API server when mere.run opens", isOn: $servesAtLaunch)
            Toggle("Open mere.run at login", isOn: Binding(
                get: { loginItemStatus == .enabled || loginItemStatus == .requiresApproval },
                set: { setOpensAtLogin($0) }
            ))
            if loginItemStatus == .requiresApproval {
                HStack(spacing: 10) {
                    Text("Allow mere.run in System Settings ▸ General ▸ Login Items.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.yellow)
                    Button("Open Login Items") { SMAppService.openSystemSettingsLoginItems() }
                        .buttonStyle(.mereSecondary)
                }
            }
            if let loginItemError {
                Text(loginItemError)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .onAppear { loginItemStatus = SMAppService.mainApp.status }
    }

    /// Registers or removes mere.run as a login item. Only the packaged app can register; a build
    /// run from the command line reports why not.
    private func setOpensAtLogin(_ opens: Bool) {
        do {
            if opens {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
            loginItemError = nil
        } catch {
            loginItemError = "mere.run could not change its login item: \(error.localizedDescription)"
        }
        loginItemStatus = SMAppService.mainApp.status
    }

    private var runtimeServerEdited: Bool {
        runtimeHost.trimmingCharacters(in: .whitespacesAndNewlines) != controller.runtimeHost
            || runtimePort != controller.runtimePort
            || runtimeAPIKey != controller.runtimeAPIKey
    }

    private func loadRuntimeServer() {
        runtimeHost = controller.runtimeHost
        runtimePort = controller.runtimePort
        runtimeAPIKey = controller.runtimeAPIKey
    }

    private func applyRuntimeServer() {
        let host = runtimeHost.trimmingCharacters(in: .whitespacesAndNewlines)
        let port = min(65_535, max(1, runtimePort))
        if host != controller.runtimeHost { controller.runtimeHost = host }
        if port != controller.runtimePort { controller.runtimePort = port }
        if runtimeAPIKey != controller.runtimeAPIKey { controller.runtimeAPIKey = runtimeAPIKey }
        loadRuntimeServer()
    }

    @ViewBuilder
    private var advancedTab: some View {
        EditorSection("Install") {
            VStack(alignment: .leading, spacing: 8) {
                Label(
                    controller.cliInstallationStatus.title,
                    systemImage: controller.cliInstallationStatus.phase == .upToDate
                        ? "checkmark.circle.fill"
                        : "terminal"
                )
                .font(MereRunTheme.bodyFont.weight(.semibold))

                Text(controller.cliInstallationStatus.detail)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)

                if let path = controller.cliInstallationStatus.resolvedPath {
                    Text("Path: \(path)")
                        .font(MereRunTheme.monoFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .textSelection(.enabled)
                }

                HStack(spacing: 14) {
                    Text("Installed: \(controller.cliInstallationStatus.installedVersion ?? "—")")
                    Text("Bundled: \(controller.cliInstallationStatus.bundledVersion ?? "—")")
                }
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)

                if let error = controller.cliInstallationStatus.lastSynchronizationError {
                    Text(error)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.yellow)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            HStack(spacing: 10) {
                if let actionTitle = controller.cliInstallationStatus.actionTitle {
                    Button {
                        controller.installTerminalCLI()
                    } label: {
                        Label(actionTitle, systemImage: "terminal")
                    }
                    .buttonStyle(.merePrimary)
                }

                Button {
                    controller.installCodexSkills()
                } label: {
                    Label("Install Skill", systemImage: "sparkles")
                }
                .buttonStyle(.mereSecondary)
            }
            Text("Studio-managed CLI payloads keep every runtime asset in Application Support and activate the command with an atomic symlink. Skill install copies the bundled `use-mere-run` Codex skill to `~/.codex/skills`.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .fixedSize(horizontal: false, vertical: true)
        }
        EditorSection("Stored configuration") {
            if configurationSummary.isEmpty {
                Text("No configuration values are stored yet.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            } else {
                Text(configurationSummary)
                    .font(MereRunTheme.monoFont)
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
                    .merePanel()
            }
            if !configurationPath.isEmpty {
                HStack(spacing: 10) {
                    Text(configurationPath)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .textSelection(.enabled)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: 0)
                    Button("Reveal in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting(
                            [URL(fileURLWithPath: configurationPath)]
                        )
                    }
                    .buttonStyle(.mereSecondary)
                }
            }
        }
        .task {
            configurationSummary = await controller.loadConfigurationSummary()
            configurationPath = await controller.loadConfigurationPath()
        }
        EditorSection("Diagnostics") {
            Toggle(
                "Capture crash and hang reports locally",
                isOn: Binding(
                    get: { crashReporter.isCapturing },
                    set: { crashReporter.setCapturing($0) }
                )
            )
            Text(
                "Uses MetricKit to record crashes, hangs, and CPU exceptions that already "
                    + "happened. Reports are written to your Application Support folder and "
                    + "are never transmitted."
            )
            .font(MereRunTheme.captionFont)
            .foregroundStyle(MereRunTheme.textMuted)
            .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 10) {
                Text(
                    crashReporter.storedPayloadCount == 1
                        ? "1 stored report"
                        : "\(crashReporter.storedPayloadCount) stored reports"
                )
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                Spacer(minLength: 0)
                Button("Reveal") {
                    let directory = StudioCrashReporter.payloadDirectory()
                    try? FileManager.default.createDirectory(
                        at: directory,
                        withIntermediateDirectories: true
                    )
                    NSWorkspace.shared.activateFileViewerSelecting([directory])
                }
                .buttonStyle(.mereSecondary)
                Button("Delete reports") { crashReporter.deleteStoredPayloads() }
                    .buttonStyle(.mereSecondary)
                    .disabled(crashReporter.storedPayloadCount == 0)
            }
        }
        .task { crashReporter.refreshStoredPayloadCount() }
    }
}

private struct EditorSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(MereRunTheme.sectionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            content
        }
    }
}

private enum PathFieldMode: Equatable {
    case openFile([UTType])
    case openDirectory
    case saveFile
}

/// One path setting: the field (a typed or pasted path still works), Choose… for the panel,
/// Reveal for what is there now, and Reset back to the default the placeholder describes — the
/// same shape as a specialist page's path row.
private struct PathField: View {
    /// What the path is, in lower case, for the buttons' accessibility labels ("Choose models root").
    let label: String
    @Binding var path: String
    let placeholder: String
    let mode: PathFieldMode

    private var trimmed: String { path.trimmingCharacters(in: .whitespacesAndNewlines) }

    /// The path as a file URL, when something is at it.
    private var existingURL: URL? {
        guard !trimmed.isEmpty else { return nil }
        let url = URL(fileURLWithPath: NSString(string: trimmed).expandingTildeInPath)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    var body: some View {
        HStack(spacing: 8) {
            TextField(placeholder, text: $path)
                .textFieldStyle(.plain)
                .font(MereRunTheme.bodyFont)
                .padding(10)
                .merePanel()
                .accessibilityLabel(label.capitalized)
            Button("Choose…") { choosePath() }
                .buttonStyle(.mereSecondary)
                .help(mode == .openDirectory ? "Choose a folder" : "Choose a file")
                .accessibilityLabel("Choose \(label)")
            Button("Reveal") {
                if let existingURL { NSWorkspace.shared.activateFileViewerSelecting([existingURL]) }
            }
            .buttonStyle(.mereSecondary)
            .disabled(existingURL == nil)
            .help("Show in Finder")
            .accessibilityLabel("Reveal \(label) in Finder")
            Button("Reset") { path = "" }
                .buttonStyle(.mereSecondary)
                .disabled(trimmed.isEmpty)
                .help("Use the default")
                .accessibilityLabel("Reset \(label) to the default")
        }
    }

    private func choosePath() {
        switch mode {
        case .openFile(let types):
            let panel = NSOpenPanel()
            panel.canChooseFiles = true
            panel.canChooseDirectories = false
            panel.allowsMultipleSelection = false
            panel.allowedContentTypes = types
            if panel.runModal() == .OK, let url = panel.url {
                path = url.path
            }
        case .openDirectory:
            let panel = NSOpenPanel()
            panel.canChooseFiles = false
            panel.canChooseDirectories = true
            panel.allowsMultipleSelection = false
            if panel.runModal() == .OK, let url = panel.url {
                path = url.path
            }
        case .saveFile:
            let panel = NSSavePanel()
            panel.nameFieldStringValue = URL(fileURLWithPath: path).lastPathComponent
            if panel.runModal() == .OK, let url = panel.url {
                path = url.path
            }
        }
    }
}
