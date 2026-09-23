import AppKit
import StudioKit
import SwiftUI

/// What a resident server's page and the menu bar say about its process.
enum StudioResidentServerCopy {
    /// "Running", "Stopping…", "Stopped", "Stopped unexpectedly".
    static func title(_ state: StudioServiceProcess.State) -> String {
        switch state {
        case .running(stopRequested: false, _): return "Running"
        case .running(stopRequested: true, _): return "Stopping…"
        case .none: return "Stopped"
        case .failed: return "Stopped unexpectedly"
        }
    }

    static func color(_ state: StudioServiceProcess.State) -> Color {
        switch state {
        case .running(stopRequested: false, _): return MereRunTheme.green
        case .running(stopRequested: true, _): return MereRunTheme.yellow
        case .none: return MereRunTheme.textMuted
        case .failed: return MereRunTheme.red
        }
    }

    /// Why the server stopped, when it stopped on its own.
    static func failure(_ state: StudioServiceProcess.State) -> String? {
        if case .failed(let message, _) = state { return message }
        return nil
    }
}

/// A server's live log: the last lines its process wrote, following the tail.
struct StudioServiceLogView: View {
    @ObservedObject var job: Job

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(job.log.lines.suffix(400)) { line in
                        Text(line.text)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(color(for: line.stream))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(line.id)
                    }
                }
                .padding(10)
            }
            .onChange(of: job.log.lines.last?.id) { _, id in
                if let id { proxy.scrollTo(id, anchor: .bottom) }
            }
        }
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                .fill(MereRunTheme.surfaceRaised.opacity(0.5))
        }
        .accessibilityLabel("Server log")
    }

    private func color(for stream: LogStream) -> Color {
        switch stream {
        case .stdout: return MereRunTheme.textPrimary
        case .stderr: return MereRunTheme.textSecondary
        case .system: return MereRunTheme.textMuted
        }
    }
}

/// Server ▸ Vision server: the resident `vision serve` endpoint, which streams binary-frame
/// grounding over HTTP without paying model load on every request. It drives
/// `controller.visionServer`, the same process the menu bar can stop.
struct StudioVisionServerView: View {
    @ObservedObject var server: StudioServiceProcess

    @StudioStoredValue("VisionServer.draft") private var draft = CommandCatalog.template(id: .visionServe)?.defaultDraft()
        ?? CommandDraft()
    @State private var message: String?
    @State private var isPreflighting = false

    private var isRunning: Bool { server.state.isRunning }

    private var endpoint: String {
        let host = draft.host.trimmingCharacters(in: .whitespacesAndNewlines)
        return "http://\(host.isEmpty ? "127.0.0.1" : host):\(draft.port)"
    }

    /// Binding beyond loopback without a key would publish the endpoint to the LAN.
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
        .studioTaskCommand(.visionServe, draft: draft)
        .onChange(of: server.state) { _, _ in message = nil }
    }

    private var settings: some View {
        StudioServingCard {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
                HStack {
                    Text("Resident grounding server")
                        .font(MereRunTheme.sectionFont)
                    Spacer()
                    Text(StudioResidentServerCopy.title(server.state))
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(StudioResidentServerCopy.color(server.state))
                }
                Text(
                    "Serves binary-frame vision grounding over HTTP so a client can stream frames "
                        + "without paying model load on every request."
                )
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 8) {
                    TextField("Host", text: $draft.host)
                        .mereField(cornerRadius: MereRunTheme.Radius.sm)
                    TextField("Port", value: $draft.port, format: .number.grouping(.never))
                        .frame(width: 90)
                        .mereField(cornerRadius: MereRunTheme.Radius.sm)
                }
                TextField("Model id or path (optional)", text: $draft.model)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
                SecureField("API key (optional)", text: $draft.apiKey)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
                Stepper(
                    "Maximum batch size: \(draft.visionServeMaxBatchSize)",
                    value: $draft.visionServeMaxBatchSize,
                    in: 1...32
                )
                .help("Maximum image-query pairs accepted by one batch request")

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
                if let message {
                    Text(message)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
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
                Button("Preflight") { preflight() }
                    .buttonStyle(.mereSecondary)
                    .disabled(isPreflighting)
                    .help("Validate the model and port without holding the server open")
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

    private var launchDraft: CommandDraft {
        var launch = draft
        launch.port = min(65_535, max(1, launch.port))
        return launch
    }

    private func preflight() {
        isPreflighting = true
        message = "Preflighting the grounding server…"
        Task { @MainActor in
            message = await server.preflight(draft: launchDraft) ?? "Preflight passed. The server can start with these settings."
            isPreflighting = false
        }
    }
}
