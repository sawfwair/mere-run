import AppKit
import StudioKit
import SwiftUI

/// Server ▸ World: the resident `world serve` endpoint that keeps one DreamX or Cosmos3
/// conditioned-video world warm between requests. It drives `controller.worldServer`, the same
/// process the menu bar can stop. The API key travels in the environment
/// (`CommandLaunchEnvironment`), never in the command line.
struct StudioWorldServerView: View {
    @ObservedObject var server: StudioServiceProcess

    @StudioStoredValue("WorldServer.draft") private var draft = CommandCatalog.template(id: .worldServe)?.defaultDraft()
        ?? CommandDraft()

    private var isRunning: Bool { server.state.isRunning }

    private var isDreamX: Bool { draft.operationsWorldBackend == WorldBackend.dreamX.rawValue }

    private var endpoint: String {
        let host = draft.host.trimmingCharacters(in: .whitespacesAndNewlines)
        return "http://\(host.isEmpty ? "127.0.0.1" : host):\(draft.port)"
    }

    /// The CLI refuses to bind beyond loopback without a key; say so before Start fails.
    private var exposedWithoutKey: Bool {
        StudioServingSafety.evaluate(host: draft.host.isBlank ? "127.0.0.1" : draft.host, apiKey: draft.apiKey)
            == .exposedWithoutAuthentication
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                settings
                if let job = server.job {
                    StudioServingCard {
                        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
                            Text("Server log")
                                .font(MereRunTheme.sectionFont)
                            StudioServiceLogView(job: job)
                                .frame(minHeight: 220)
                        }
                    }
                }
            }
            .frame(maxWidth: 760, alignment: .leading)
            .padding(MereRunTheme.Spacing.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(MereRunTheme.background)
        .studioTaskCommand(.worldServe, draft: draft)
    }

    private var settings: some View {
        StudioServingCard {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
                HStack {
                    Text("World session server")
                        .font(MereRunTheme.sectionFont)
                    Spacer()
                    Text(StudioResidentServerCopy.title(server.state))
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(StudioResidentServerCopy.color(server.state))
                }
                Text(
                    "Keeps one conditioned-video world warm so a client can step it with actions "
                        + "without loading the model on every request."
                )
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .fixedSize(horizontal: false, vertical: true)

                Picker("Backend", selection: backend) {
                    ForEach(WorldBackend.allCases) { backend in
                        Text(backend.title).tag(backend)
                    }
                }
                .pickerStyle(.segmented)
                .disabled(isRunning)

                HStack(spacing: 8) {
                    TextField("Host", text: $draft.host)
                        .mereField(cornerRadius: MereRunTheme.Radius.sm)
                    TextField("Port", value: $draft.port, format: .number.grouping(.never))
                        .frame(width: 90)
                        .mereField(cornerRadius: MereRunTheme.Radius.sm)
                }
                TextField("World model ID or path", text: $draft.model)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
                if isDreamX {
                    TextField("Base model ID or path", text: $draft.operationsBaseModel)
                        .mereField(cornerRadius: MereRunTheme.Radius.sm)
                }
                ContractFormPathRow(
                    label: "State folder",
                    path: $draft.operationsStateDirectory,
                    isDirectory: true,
                    placeholder: "Optional. Keeps the world between restarts."
                )
                SecureField("API key (optional)", text: $draft.apiKey)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
                Toggle("Load models before the first request", isOn: $draft.operationsPrepare)
                    .help("Warms the models at start so the first request doesn't wait for loading")

                Text("Endpoint: \(endpoint)")
                    .font(MereRunTheme.monoFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .textSelection(.enabled)

                if exposedWithoutKey {
                    Label(
                        "This binds beyond loopback with no API key. Add a key or bind to 127.0.0.1.",
                        systemImage: "exclamationmark.triangle.fill"
                    )
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.yellow)
                }

                controls

                if let failure = StudioResidentServerCopy.failure(server.state) {
                    Text(failure)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.red)
                        .textSelection(.enabled)
                }
            }
        }
    }

    private var controls: some View {
        HStack(spacing: 8) {
            if isRunning {
                Button("Stop") { server.stop() }
                    .buttonStyle(.mereSecondary)
                Button("Restart") {
                    Task { _ = await server.restart(draft: launchDraft) }
                }
                .buttonStyle(.mereSecondary)
                .help("Restart with these settings")
            } else {
                Button("Start") { server.start(draft: launchDraft) }
                    .buttonStyle(.merePrimary)
                    .disabled(exposedWithoutKey)
            }
            Spacer()
            Button("Copy endpoint") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(endpoint, forType: .string)
            }
            .buttonStyle(.mereSecondary)
        }
    }

    /// Switching backends moves the world model to that backend's default when the field still
    /// holds the other backend's default, so a typed path is never thrown away.
    private var backend: Binding<WorldBackend> {
        Binding(
            get: { WorldBackend(rawValue: draft.operationsWorldBackend) ?? .dreamX },
            set: { next in
                let previous = WorldBackend(rawValue: draft.operationsWorldBackend) ?? .dreamX
                draft.operationsWorldBackend = next.rawValue
                if draft.model.isBlank || draft.model == previous.defaultModel {
                    draft.model = next.defaultModel
                }
            }
        )
    }

    private var launchDraft: CommandDraft {
        var launch = draft
        launch.port = min(65_535, max(1, launch.port))
        return launch
    }
}

/// `world serve --backend`, with the model each one starts on (the contract's routing defaults).
private enum WorldBackend: String, CaseIterable, Identifiable {
    case dreamX = "dreamx"
    case cosmos3

    var id: String { rawValue }

    var title: String {
        switch self {
        case .dreamX: return "DreamX World"
        case .cosmos3: return "Cosmos3 Edge"
        }
    }

    var defaultModel: String {
        switch self {
        case .dreamX: return "video-dreamx-world-5b-ar-mlx"
        case .cosmos3: return "video-cosmos3-edge-mlx"
        }
    }
}
