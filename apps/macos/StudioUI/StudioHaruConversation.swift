import AppKit
import MereRunContract
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// Chat ▸ Conversation. The draft is shared with the Command view and the job owns the live
/// transcript, so navigating away from the workspace leaves the conversation running.
struct StudioHaruConversation: View {
    @ObservedObject var models: StudioModelStore
    @ObservedObject var jobMonitor: StudioJobMonitor
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @EnvironmentObject private var navigation: NavigationModel
    @Environment(\.studioScopeSource) private var scopeSource
    @Environment(\.studioTaskRunner) private var runner
    @Environment(\.studioTaskSessions) private var sessions
    @State private var showsSetup = false
    @State private var selectedResultID: UUID?
    @State private var error: String?

    private static let task = StudioTask.chatConversation

    private var draft: StudioTaskDraft {
        get { sessions?.taskDraft(for: Self.task) ?? StudioTaskDraft(templateID: .speechConverse) }
        nonmutating set { sessions?.setTaskDraft(newValue, for: Self.task) }
    }

    private var readiness: ModelReadinessState { controller.readiness(for: Self.task) }

    private var modelID: String { StudioTaskSchema.modelID(for: draft, source: scopeSource) }

    private var history: [StudioLibraryItem] {
        library.items.filter { $0.templateID == .speechConverse && $0.status == .completed }
            .sorted { $0.createdAt > $1.createdAt }
    }

    var body: some View {
        let job = runner?.currentJob(for: Self.task)
        let canStart = !readiness.blocksRun && job == nil && runner != nil
        let pullJob = jobMonitor.pullJob(for: modelID)
        let results = history
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let job {
                    StudioHaruRunningConversation(job: job) { runner?.stop(task: Self.task) }
                } else {
                    idlePanel(canStart: canStart)
                }
                if !readiness.isChecking, readiness.blocksRun || pullJob != nil, job == nil {
                    readinessPanel(pullJob: pullJob)
                }
                if let error {
                    MereBanner(severity: .error, text: error, onDismiss: { self.error = nil })
                }
                setup.disabled(job != nil)
                if job == nil,
                   let item = results.first(where: { $0.id == selectedResultID }) ?? results.first {
                    StudioHaruSavedConversation(item: item, results: results) { selectedResultID = $0 }
                        .id(item.id)
                }
            }
            .frame(maxWidth: 760)
            .padding(24)
            .frame(maxWidth: .infinity)
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .task(id: modelID) { refreshReadiness() }
        .onChange(of: job?.id) { _, id in
            if id != nil { selectedResultID = nil }
        }
        .onReceive(controller.jobs.events) { event in
            guard case .finished(let job, let result) = event,
                  job.request.templateID == .speechConverse, result.exitCode != 0 else { return }
            if job.stopRequest != nil || result.exitCode == JobResult.cancelledBeforeStartExitCode {
                // Ending during model load or while queued need not produce an audio artifact.
                error = nil
            } else {
                error = StudioFailureSummary.summary(outputText: result.outputText, exitCode: result.exitCode)
            }
        }
        .focusedSceneValue(\.studioPageRun, StudioPageRun(isEnabled: canStart, run: start))
    }

    private func idlePanel(canStart: Bool) -> some View {
        let presentation = Self.task.presentation
        let usesFile = !draft.text("--input").isBlank
        return VStack(spacing: 15) {
            Image(systemName: "waveform")
                .font(.system(size: 33, weight: .medium))
                .foregroundStyle(MereRunTheme.accent)
                .frame(width: 78, height: 78)
                .background(MereRunTheme.accentSoft, in: Circle())
                .accessibilityHidden(true)
            VStack(spacing: 7) {
                Text(presentation.emptyTitle)
                    .font(MereRunTheme.displayFont)
                    .accessibilityAddTraits(.isHeader)
                Text(usesFile ? "Haru will respond to the audio file you selected." : presentation.emptyMessage)
                    .font(.callout)
                    .foregroundStyle(MereRunTheme.textSecondary)
            }
            Button(usesFile ? "Respond to audio file" : "Start conversation", systemImage: usesFile ? "play.fill" : "mic.fill", action: start)
                .buttonStyle(.merePrimary)
                .disabled(!canStart)
                .accessibilityIdentifier("haru.start")
            if readiness.isChecking {
                Text("Checking Haru-4B…")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 29)
        .multilineTextAlignment(.center)
        .frame(maxWidth: .infinity)
        .merePanel()
    }

    private var setup: some View {
        DisclosureGroup(isExpanded: $showsSetup) {
            VStack(alignment: .leading, spacing: 13) {
                fileRow("Input", flag: "--input", types: [.audio], placeholder: "System microphone")
                fileRow("Character card", flag: "--system-prompt-file", types: [.plainText], placeholder: "Optional UTF-8 text file")
                fileRow("Voice reference", flag: "--reference-audio", types: [.audio], placeholder: "Optional audio file")
                Toggle("Haru speaks first", isOn: Binding(
                    get: { draft.form["--speak-first"].flag == true },
                    set: { draft.form["--speak-first"] = .flag($0) }
                ))
                .toggleStyle(.checkbox)
            }
            .padding(.top, 14)
        } label: {
            Label("Conversation setup", systemImage: "slider.horizontal.3")
                .font(MereRunTheme.sectionFont)
        }
        .padding(16)
        .merePanel()
    }

    private func fileRow(_ title: String, flag: String, types: [UTType], placeholder: String) -> some View {
        let path = draft.text(flag)
        return LabeledContent(title) {
            HStack(spacing: 8) {
                Text(path.isEmpty ? placeholder : URL(fileURLWithPath: path).lastPathComponent)
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(path.isEmpty ? placeholder : path)
                if !path.isEmpty {
                    Button("Clear \(title.lowercased())", systemImage: "xmark.circle", role: nil) {
                        draft.form[flag] = .unset
                    }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.plain)
                }
                Button(path.isEmpty ? "Choose…" : "Replace…") {
                    guard let url = StudioFilePanels.chooseFile(title: title, allowedContentTypes: types).first else { return }
                    var next = draft
                    next.form[flag] = .text(url.path)
                    if flag == "--input" { next.form["--live"] = .unset }
                    draft = next
                }
                .accessibilityLabel("Choose \(title.lowercased())")
            }
        }
        .font(.callout)
    }

    private func readinessPanel(pullJob: Job?) -> some View {
        StudioReadinessCard(
            readiness: readiness, pullJob: pullJob,
            actions: StudioReadinessActions(
                scope: StudioTaskSchema.modelScope(for: draft, source: scopeSource),
                model: Binding(get: { draft.model }, set: { draft.model = $0 }),
                modelInventory: models.rows, pullModel: pullModel,
                openModels: { navigation.open(task: .modelsInstalled) }, recheck: refreshReadiness,
                showsModelPicker: false
            ),
            onCancelPull: { _ = jobMonitor.cancel($0) }
        )
    }

    private func refreshReadiness() {
        controller.checkReadiness(for: Self.task, requirement: StudioTaskSchema.requirement(for: draft, source: scopeSource))
    }

    private func start() {
        error = nil
        do {
            _ = try runner?.run(draft, task: Self.task)
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func pullModel() {
        guard !modelID.isBlank, let template = CommandCatalog.template(id: .modelPull) else { return }
        var options = template.defaultDraft()
        options.model = modelID
        let request = StudioRunRequest(mode: template.libraryMode, templateID: .modelPull, template: template, draft: options)
        if !models.startPull(request) {
            error = controller.status
            refreshReadiness()
        }
    }
}

private struct StudioHaruRunningConversation: View {
    @ObservedObject var job: Job
    let stop: () -> Void

    private var isLive: Bool { job.request.configuration.arguments.contains("--live") }

    private var microphoneActive: Bool {
        job.stopRequest == nil && (job.haruLiveOutput?.isCapturing == true || legacyCaptureStarted)
    }

    private var legacyCaptureStarted: Bool {
        job.haruLiveOutput == nil && job.haruLegacyCaptureStarted
    }

    private var haruSpeaking: Bool { microphoneActive && job.haruLiveOutput?.isSpeaking == true }

    private var playbackStatus: String {
        if haruSpeaking { return "Haru speaking" }
        if legacyCaptureStarted { return "Playback status unavailable" }
        return microphoneActive ? "Haru quiet" : "Playback pending"
    }

    private var title: String {
        if job.stopRequest != nil { return isLive ? "Ending conversation…" : "Canceling response…" }
        if job.isAwaitingMachineAdmission || job.state == .queued { return "Waiting to start…" }
        if microphoneActive { return "Conversation in progress" }
        return isLive ? "Starting Haru…" : "Responding to audio…"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(spacing: 15) {
                timeline
                if job.stopRequest == nil, isLive {
                    HStack(spacing: 10) {
                        indicator(microphoneActive ? "Microphone on" : "Microphone off", icon: "mic", active: microphoneActive)
                        indicator(
                            playbackStatus,
                            icon: "speaker.wave.2", active: haruSpeaking
                        )
                    }
                }
                Button(isLive ? "End conversation" : "Cancel response", systemImage: "stop.fill", action: stop)
                    .buttonStyle(.merePrimary)
                    .disabled(job.stopRequest != nil)
                    .accessibilityIdentifier("haru.stop")
            }
            .padding(24)
            .frame(maxWidth: .infinity)
            .merePanel()

            if isLive {
                StudioHaruTranscript(text: job.haruLiveOutput?.transcript ?? "", isLive: true)
            }
        }
    }

    private var timeline: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            VStack(spacing: 6) {
                Text(title)
                    .font(MereRunTheme.displaySmallFont)
                    .accessibilityAddTraits(.isHeader)
                Text(StudioTimeFormat.string(max(0, job.startedAt.map { context.date.timeIntervalSince($0) } ?? 0)))
                    .font(.system(.title, design: .monospaced, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .monospacedDigit()
                    .accessibilityLabel("Elapsed time")
            }
        }
    }

    private func indicator(_ text: String, icon: String, active: Bool) -> some View {
        Label(text, systemImage: icon)
            .font(MereRunTheme.captionFont)
            .foregroundStyle(active ? MereRunTheme.green : MereRunTheme.textMuted)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(active ? MereRunTheme.green.opacity(0.11) : MereRunTheme.surfaceRaised, in: Capsule())
    }
}

private struct StudioHaruSavedConversation: View {
    let item: StudioLibraryItem
    let results: [StudioLibraryItem]
    let select: (UUID) -> Void
    @State private var exportError: String?

    private var summary: HaruConversationSummary? {
        guard let outputText = item.outputText else { return nil }
        return try? JSONDecoder().decode(HaruConversationSummary.self, from: Data(outputText.utf8))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Saved conversations")
                    .font(MereRunTheme.sectionFont)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Menu {
                    ForEach(results) { result in
                        Button {
                            select(result.id)
                        } label: {
                            Label(result.createdAt.formatted(date: .abbreviated, time: .standard),
                                  systemImage: result.id == item.id ? "checkmark" : "clock")
                        }
                    }
                } label: {
                    Text(item.createdAt, format: .dateTime.month().day().hour().minute())
                }
                .fixedSize()
                .accessibilityLabel("Saved conversations")
            }
            StudioHaruTranscript(text: summary?.transcript ?? "", isLive: false)
            if let audioURL = item.outputURL, FileManager.default.fileExists(atPath: audioURL.path) {
                VStack(alignment: .leading, spacing: 0) {
                    HStack {
                        Label("Haru’s audio", systemImage: "waveform")
                            .font(MereRunTheme.sectionFont)
                        Spacer()
                        StudioSessionSecondaryButton("Save audio…") { export(audioURL) }
                    }
                    StudioAudioPlayerView(url: audioURL)
                }
                .padding(16)
                .merePanel()
            } else if item.outputURL != nil {
                Label("The saved audio file is missing.", systemImage: "waveform.slash")
                    .font(.callout)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            if let exportError {
                MereBanner(severity: .error, text: exportError, onDismiss: { self.exportError = nil })
            }
        }
    }

    private func export(_ audio: URL) {
        guard let destination = StudioFilePanels.saveFile(
            title: "Save Haru’s audio", suggestedName: audio.lastPathComponent, allowedContentTypes: [.wav]
        ) else { return }
        do {
            try StudioFileExport.copy(audio, to: destination)
        } catch {
            exportError = error.localizedDescription
        }
    }
}

private struct StudioHaruTranscript: View {
    let text: String
    let isLive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Haru’s transcript", systemImage: "text.alignleft")
                    .font(MereRunTheme.sectionFont)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                if !text.isEmpty {
                    StudioSessionSecondaryButton("Copy") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(text, forType: .string)
                    }
                    .accessibilityLabel("Copy Haru’s transcript")
                }
            }
            if text.isEmpty {
                Text(isLive ? "Haru’s words appear here as it responds." : "Haru produced no transcript.")
                    .foregroundStyle(MereRunTheme.textMuted)
            } else {
                Text(text)
                    .textSelection(.enabled)
            }
        }
        .font(.callout)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .merePanel()
    }
}
