import AppKit
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// Server ▸ Music server: the resident `music serve` endpoint that keeps ACE-Step, its language
/// model, and an adapter stack warm behind a local API. It drives `controller.musicServer`, the
/// same process the menu bar can stop, and keeps its settings under the key the Music Tools page
/// used, so a server configured before the page moved starts with the same command.
struct StudioMusicServerView: View {
    @ObservedObject var server: StudioServiceProcess

    @StudioStoredValue("MusicTools.serveDraft") private var draft = CommandCatalog.template(id: .musicServe)?.defaultDraft()
        ?? CommandDraft()

    private static let symbol = "bolt.horizontal.circle"

    var body: some View {
        StudioAnalysisLayout { configuration } result: { status }
        .studioTaskCommand(.musicServe, draft: draft)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
    }

    private var configuration: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
                Text("Keep ACE-Step, its language model, and adapter stack warm behind a local API.")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
                HStack {
                    labeledTextField("Host", placeholder: "127.0.0.1", text: $draft.host)
                    Stepper("Port \(String(draft.port))", value: $draft.port, in: 1...65_535)
                }
                labeledTextField("ACE-Step model", placeholder: "music-acestep", text: $draft.model)
                checkpointControls
                StudioPathField(
                    label: "Adapters",
                    placeholder: "One adapter path per line",
                    path: $draft.musicAdapterPaths,
                    allowsMultipleSelection: true,
                    allowedContentTypes: [.data]
                )
                if !draft.musicAdapterPaths.isBlank {
                    Picker("Adapter format", selection: $draft.musicAdapterKind) {
                        Text("Automatic").tag("auto")
                        Text("LoRA").tag("lora")
                        Text("LoKr").tag("lokr")
                    }
                    labeledTextField(
                        "Adapter scales",
                        placeholder: "One scale per line",
                        text: $draft.musicAdapterScales
                    )
                }
                SecureField("Optional bearer token", text: $draft.apiKey)
                    .mereField()
                Text("The token is injected through MERERUN_API_KEY and is never placed in argv.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                StudioMusicServerControl(server: server, draft: draft, symbol: Self.symbol)
            }
            .padding(18)
        }
    }

    private var checkpointControls: some View {
        DisclosureGroup("Checkpoint layout") {
            VStack(alignment: .leading, spacing: 10) {
                StudioPathField(
                    label: "Checkpoint root",
                    placeholder: "Auto-discover",
                    path: $draft.musicCheckpointsRoot,
                    picksDirectory: true
                )
                HStack {
                    labeledTextField(
                        "Decoder",
                        placeholder: "acestep-v15-turbo",
                        text: $draft.musicDecoderSubdirectory
                    )
                    labeledTextField("VAE", placeholder: "vae", text: $draft.musicVAESubdirectory)
                }
                labeledTextField(
                    "LM model",
                    placeholder: "music-acestep-lm-1.7b",
                    text: $draft.musicLMModel
                )
                labeledTextField(
                    "LM subdirectory",
                    placeholder: "Auto-discover",
                    text: $draft.musicLMSubdirectory
                )
                labeledTextField(
                    "Text encoder",
                    placeholder: "Auto-discover",
                    text: $draft.musicTextSubdirectory
                )
            }
            .padding(.top, 9)
        }
    }

    /// The server is a process, not a Library run; its state is the whole result pane.
    private var status: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Resident server")
                    .font(MereRunTheme.sectionFont)
                Spacer()
            }
            StudioMusicServerStatus(server: server, host: draft.host, port: draft.port)
        }
        .padding(18)
    }

    private func labeledTextField(
        _ label: String,
        placeholder: String,
        text: Binding<String>
    ) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            TextField(placeholder, text: text)
                .mereField()
        }
    }
}

/// Start or Stop for the resident music server. It observes `controller.musicServer`, the process
/// the menu bar can stop too, rather than whichever run happens to hold the console.
private struct StudioMusicServerControl: View {
    @ObservedObject var server: StudioServiceProcess
    let draft: CommandDraft
    let symbol: String

    var body: some View {
        if server.state.isRunning {
            HStack {
                Button { Task { _ = await server.restart(draft: draft) } } label: {
                    Label("Restart", systemImage: "arrow.clockwise")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.mereSecondary)
                .help("Restart the server with these settings")
                Button { server.stop() } label: {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.mereSecondary)
            }
            .disabled(server.state.isStopping)
        } else {
            Button { server.start(draft: draft) } label: {
                Label("Start resident server", systemImage: symbol)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.merePrimary)
        }
    }
}

/// The resident music server's state, endpoint, and live log.
private struct StudioMusicServerStatus: View {
    @ObservedObject var server: StudioServiceProcess
    let host: String
    let port: Int

    private var isRunning: Bool { server.state.isRunning && !server.state.isStopping }

    var body: some View {
        VStack(spacing: 18) {
            ZStack {
                Circle()
                    .fill(isRunning ? MereRunTheme.green.opacity(0.14) : MereRunTheme.surface)
                Image(systemName: isRunning ? "bolt.horizontal.circle.fill" : "bolt.slash.circle")
                    .font(.system(size: 58, weight: .semibold))
                    .foregroundStyle(isRunning ? MereRunTheme.green : MereRunTheme.textMuted)
            }
            .frame(width: 130, height: 130)
            Text("Resident music is \(StudioResidentServerCopy.title(server.state).lowercased())")
                .font(MereRunTheme.titleFont)
            if let failure = StudioResidentServerCopy.failure(server.state) {
                Text(failure)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.red)
                    .multilineTextAlignment(.center)
                    .textSelection(.enabled)
            }
            Text("http://\(host):\(String(port))")
                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                .textSelection(.enabled)
            Button("Open health endpoint") {
                guard let url = URL(string: "http://\(host):\(port)/health") else { return }
                NSWorkspace.shared.open(url)
            }
            .buttonStyle(.mereSecondary)
            .disabled(!isRunning)
            if let job = server.job {
                StudioServiceLogView(job: job)
                    .frame(minHeight: 200)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
