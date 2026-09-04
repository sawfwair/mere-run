import AppKit
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// The Settings window: General, Models, Server, and Advanced. Everything here is machine-wide
/// configuration — where the CLI is, where models and generations live, what the runtime server
/// answers on, and the local diagnostics — rather than anything about one run.
package struct MereRunSettingsView: View {
    package init() {}

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

    package var body: some View {
        TabView {
            settingsTab { generalTab }
                .tabItem { Label("General", systemImage: "gearshape") }
            settingsTab { modelsTab }
                .tabItem { Label("Models", systemImage: "shippingbox") }
            settingsTab { serverTab }
                .tabItem { Label("Server", systemImage: "network") }
            settingsTab { advancedTab }
                .tabItem { Label("Advanced", systemImage: "wrench.and.screwdriver") }
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
            PathField(path: $controller.cliPath, placeholder: "Auto-detect executable", mode: .openFile([.unixExecutable, .item]))
            Text("The app uses a bundled `mere.run` first, then nearby SwiftPM build products, common install locations, and the current package checkout.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
        EditorSection("Working directory") {
            PathField(path: $controller.workingDirectory, placeholder: "Working directory", mode: .openDirectory)
        }
        EditorSection("Models root") {
            PathField(path: $controller.modelsRoot, placeholder: "Optional model links/local-files root", mode: .openDirectory)
        }
        EditorSection("Where generations are saved") {
            PathField(path: $outputRoot, placeholder: outputRootPlaceholder, mode: .openDirectory)
            Text("Leave this empty to file work by what it is: pictures and clips in `~/Pictures/mere.run`, audio in `~/Music/mere.run`, everything else in `~/Documents/mere.run`, each under a folder named for the domain. Set a folder to keep every domain together there instead. Runs already in the Library keep the paths they recorded.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
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
            PathField(path: $controller.hubCache, placeholder: "Optional model payload storage", mode: .openDirectory)
            Text("Downloads land here. Browsing, cleanup, and locations live in the Models domain.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
    }

    @ViewBuilder
    private var serverTab: some View {
        EditorSection("Runtime server") {
            HStack(spacing: 10) {
                TextField("Host", text: $controller.runtimeHost)
                    .textFieldStyle(.plain)
                    .font(MereRunTheme.bodyFont)
                    .padding(10)
                    .merePanel()
                TextField("Port", value: $controller.runtimePort, format: .number.grouping(.never))
                    .textFieldStyle(.plain)
                    .font(MereRunTheme.bodyFont)
                    .frame(width: 90)
                    .padding(10)
                    .merePanel()
                SecureField("API key (optional)", text: $controller.runtimeAPIKey)
                    .textFieldStyle(.plain)
                    .font(MereRunTheme.bodyFont)
                    .padding(10)
                    .merePanel()
            }
            Text("Where Models and Server send load/unload requests for the running runtime (`mere.run api serve`).")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
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
                    .buttonStyle(.borderedProminent)
                }

                Button {
                    controller.installCodexSkills()
                } label: {
                    Label("Install Skill", systemImage: "sparkles")
                }
                .buttonStyle(.bordered)
            }
            Text("Studio-managed CLI payloads keep every runtime asset in Application Support and activate the command with an atomic symlink. Skill install copies the bundled `use-mere-run` Codex skill to `~/.codex/skills`.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
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

private struct PathField: View {
    @Binding var path: String
    let placeholder: String
    let mode: PathFieldMode

    var body: some View {
        HStack(spacing: 8) {
            TextField(placeholder, text: $path)
                .textFieldStyle(.plain)
                .font(MereRunTheme.bodyFont)
            Button {
                choosePath()
            } label: {
                Image(systemName: mode == .saveFile ? "square.and.arrow.down" : "folder")
            }
            .buttonStyle(.borderless)
            .help(mode == .saveFile ? "Choose output path" : "Choose path")
            .accessibilityLabel(mode == .saveFile ? "Choose output path" : "Choose path")
        }
        .padding(10)
        .merePanel()
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
