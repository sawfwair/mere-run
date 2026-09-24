import AppKit
import MereRunContract
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// Audio ▸ Live: the Session surface over `speech listen` and `speech diarize-live`. Start
/// submits the task draft through the task runner, so the session is an inference job with a
/// Library row, Stop, and the Activity panel like any run; the page reads the job's stdout events
/// into the live transcript or the speaker activity as they arrive. Its settings — which of the
/// two commands, the microphone, the language and windows, the model — are the task draft the
/// Command view edits too. The session itself (`StudioLiveListenModel`) belongs to the
/// controller, so leaving and coming back loses nothing.
struct StudioLiveListenSession: View {
    @EnvironmentObject private var controller: MereRunController
    @ObservedObject private var models: StudioModelStore

    init(models: StudioModelStore) {
        _models = ObservedObject(wrappedValue: models)
    }

    var body: some View {
        StudioLiveListenSessionContent(session: controller.liveListen, models: models)
    }
}

private struct StudioLiveListenSessionContent: View {
    @ObservedObject var session: StudioLiveListenModel
    @ObservedObject var models: StudioModelStore

    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var navigation: NavigationModel
    @Environment(\.studioTaskRunner) private var runner
    @Environment(\.studioTaskSessions) private var sessions
    @Environment(\.studioModelTitles) private var titles
    @State private var error: String?
    @State private var showsOptions = false
    private let ticker = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    private static let task = StudioTask.audioLive

    // MARK: Draft

    private var draft: StudioTaskDraft {
        get { sessions?.taskDraft(for: Self.task) ?? StudioTaskDraft(templateID: .speechListen) }
        nonmutating set { sessions?.setTaskDraft(newValue, for: Self.task) }
    }

    private var draftBinding: Binding<StudioTaskDraft> {
        Binding(get: { draft }, set: { draft = $0 })
    }

    private var readiness: ModelReadinessState {
        controller.readiness(for: Self.task)
    }

    /// The command whose output the panel shows: the running (or last) session's, else the one
    /// the draft would start.
    private var shownTemplateID: CommandTemplateID {
        session.templateID ?? draft.templateID
    }

    private var showsSpeakers: Bool {
        shownTemplateID == .speechDiarizeLive
    }

    private var hasOutput: Bool {
        showsSpeakers ? !session.activity.segments.isEmpty : !session.transcript.displayText.isEmpty
    }

    private var phase: StudioSessionPhase {
        guard let job = session.job else { return .idle }
        switch job.state {
        case .queued:
            return .queued
        case .running:
            return .live
        case .finished(let exit, _), .cancelled(let exit, _):
            return .ended(exitCode: exit)
        case .preflightFailed(let failure):
            return .ended(exitCode: failure.exitCode)
        }
    }

    private var logLines: [String] {
        guard let startedAt = session.startedAt else { return [] }
        return StudioRealtimeSessionLog.lines(session.logLines, startedAt: startedAt)
    }

    /// The template's options the page does not draw as chips: everything but the variant, the
    /// model, the device, and the switches the launch owns.
    private var optionFields: [StudioContractField<StudioTaskDraft>] {
        StudioTaskSchema.fields(for: Self.task, draft: draft).filter { field in
            field.overrideID != .variant && field.overrideID != .model
                && !StudioTaskDraft.liveListenOwnedFlags.contains(field.flag)
        }
    }

    private var changedOptionCount: Int {
        let baseline = StudioTaskDraft(templateID: draft.templateID)
        return optionFields.reduce(0) { $0 + $1.changedCount(draft: draft, baseline: baseline) }
    }

    // MARK: Body

    var body: some View {
        VStack(spacing: 0) {
            StudioSessionSurface(
                phase: phase,
                clock: StudioTimeFormat.string(session.elapsed),
                logLines: logLines,
                onToggle: toggle
            ) {
                chips
            } live: {
                livePanel
            }
            if let error {
                MereBanner(severity: .error, text: error, onDismiss: { self.error = nil })
                    .padding(.horizontal, 24)
                    .padding(.bottom, 16)
            }
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onAppear {
            if let runner { session.adoptCurrentSession(runner: runner) }
            refreshReadiness()
            Task {
                if session.devices.isEmpty { await session.refreshDevices() }
            }
        }
        .onReceive(ticker) { _ in
            if session.isActive { session.tick() }
        }
        .onChange(of: draft.templateID) { _, _ in
            error = nil
            refreshReadiness()
        }
        .onChange(of: draft.model) { _, _ in
            error = nil
            refreshReadiness()
        }
        .onChange(of: session.errorMessage) { _, message in
            if let message { error = message }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Live listening")
    }

    // MARK: Chips

    private var chips: some View {
        HStack(spacing: 6) {
            operationChip
            deviceChip
            optionsChip
            StudioModelChip(
                scope: StudioTaskSchema.modelScope(for: draft),
                model: draftBinding.model,
                modelInventory: models.rows,
                readiness: readiness,
                onShowModels: { navigation.open(task: .modelsInstalled) }
            )
        }
        .disabled(session.isActive)
    }

    private var operationChip: some View {
        Menu {
            ForEach(Self.task.variantTemplates) { template in
                Toggle(isOn: Binding(
                    get: { draft.templateID == template.id },
                    set: { _ in
                        var next = draft
                        next.switchTemplate(to: template.id)
                        draft = next
                    }
                )) {
                    Text(template.title)
                }
            }
        } label: {
            StudioComposerChipLabel(title: draft.template?.title ?? "Operation")
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Transcribe what is said, or tell the speakers apart")
        .accessibilityLabel("Operation")
        .accessibilityValue(draft.template?.title ?? "")
    }

    private var selectedDeviceUID: String {
        draft.text("--device")
    }

    private var deviceTitle: String {
        let uid = selectedDeviceUID
        if uid.isEmpty { return "System microphone" }
        return session.devices.first { $0.uid == uid }?.name ?? uid
    }

    private func deviceSelection(_ uid: String) -> Binding<Bool> {
        Binding(
            get: { selectedDeviceUID == uid },
            set: { _ in
                var next = draft
                next.form["--device"] = uid.isEmpty ? .unset : .text(uid)
                draft = next
            }
        )
    }

    private var deviceChip: some View {
        Menu {
            Toggle("System default", isOn: deviceSelection(""))
            ForEach(session.devices) { device in
                Toggle(device.isDefault ? "\(device.name) — default" : device.name, isOn: deviceSelection(device.uid))
            }
            Divider()
            Button("Refresh microphones") {
                Task { await session.refreshDevices() }
            }
        } label: {
            StudioComposerChipLabel(title: deviceTitle, leadingSystemImage: "mic")
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help(session.devicesUnavailable ? "The CLI could not list microphones; the system input is used." : "The microphone the session captures")
        .accessibilityLabel("Microphone")
        .accessibilityValue(deviceTitle)
    }

    private var optionsChip: some View {
        Button {
            showsOptions = true
        } label: {
            StudioComposerChipLabel(
                title: changedOptionCount == 0 ? "Options" : "Options · \(changedOptionCount) changed",
                leadingSystemImage: "slider.horizontal.3",
                menu: false
            )
        }
        .buttonStyle(.plain)
        .help("Language, windows, and thresholds for this session")
        .accessibilityLabel("Session options")
        .popover(isPresented: $showsOptions, arrowEdge: .bottom) {
            ContractForm(
                fields: optionFields,
                dependencies: StudioTaskSchema.dependencies(for: draft),
                draft: draftBinding
            ) { _ in
                EmptyView()
            }
            .frame(width: 300)
            .padding(MereRunTheme.Spacing.md)
            .background(MereRunTheme.background)
            .foregroundStyle(MereRunTheme.textPrimary)
        }
    }

    // MARK: Live panel

    private var livePanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Text(showsSpeakers ? "Speaker activity" : "Live transcript")
                    .font(MereRunTheme.sectionFont)
                if showsSpeakers, session.activity.audioSeconds > 0 {
                    Text("\(session.activity.speakerCount) speakers · \(Int(session.activity.audioSeconds)) s")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .monospacedDigit()
                } else if let url = session.transcriptURL, !session.isActive {
                    Text("Saved as \(url.lastPathComponent)")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(url.path)
                }
                Spacer(minLength: 8)
                StudioSessionSecondaryButton("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(session.text, forType: .string)
                }
                .disabled(session.text.isEmpty)
                .accessibilityLabel(showsSpeakers ? "Copy the speaker activity" : "Copy the live transcript")
                StudioSessionSecondaryButton("Save…", action: save)
                    .disabled(session.text.isEmpty)
                StudioSessionSecondaryButton("Clear", action: session.clear)
                    .disabled(session.isActive || !hasOutput)
            }
            if !session.isActive, readiness.blocksRun {
                Text(readiness.message(titles: titles))
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.yellow)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityLabel("Not ready: \(readiness.message(titles: titles))")
            }
            if hasOutput {
                output
            } else {
                waiting
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
    }

    @ViewBuilder
    private var waiting: some View {
        if session.isActive {
            VStack(spacing: MereRunTheme.Spacing.sm) {
                ProgressView()
                    .controlSize(.small)
                    .accessibilityHidden(true)
                Text(showsSpeakers ? "Waiting for speakers…" : "Waiting for speech…")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .accessibilityLabel(showsSpeakers ? "Listening for speakers" : "Listening for speech")
        } else {
            StudioEmptyState(presentation: Self.task.presentation, isCompact: true)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    @ViewBuilder
    private var output: some View {
        ScrollView {
            if showsSpeakers {
                LazyVStack(alignment: .leading, spacing: 5) {
                    ForEach(Array(session.activity.segments.suffix(200).enumerated()), id: \.offset) { _, segment in
                        HStack {
                            Text(segment.speaker)
                                .fontWeight(.semibold)
                                .foregroundStyle(MereRunTheme.accent)
                            Spacer()
                            Text(String(format: "%.2f–%.2f s", segment.startSeconds, segment.endSeconds))
                                .monospacedDigit()
                                .foregroundStyle(MereRunTheme.textSecondary)
                        }
                        .font(MereRunTheme.bodyFont)
                        .accessibilityElement(children: .combine)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    if !session.transcript.committedText.isEmpty {
                        Text(session.transcript.committedText)
                    }
                    if !session.transcript.partialText.isEmpty {
                        Text(session.transcript.partialText)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .italic()
                            .accessibilityLabel("Partial transcript: \(session.transcript.partialText)")
                    }
                }
                .font(MereRunTheme.bodyFont)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)
            }
        }
        .merePanel()
    }

    // MARK: Actions

    private func toggle() {
        if session.isActive {
            session.stop()
        } else {
            start()
        }
    }

    private func start() {
        error = nil
        guard let runner else { return }
        do {
            let request = try runner.run(draft.liveListenLaunch(), task: Self.task)
            session.begin(request, library: runner.library)
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func refreshReadiness() {
        controller.checkReadiness(for: Self.task, modelID: StudioTaskSchema.modelID(for: draft))
    }

    private func save() {
        let stem = showsSpeakers ? "live-speakers" : "live-transcript"
        let suggested = StudioOutputLocation.specialistFile(
            domain: .audio, name: stem, fileExtension: "txt", now: StudioDisplayClock.now
        ).lastPathComponent
        guard let url = StudioFilePanels.saveFile(
            title: showsSpeakers ? "Save live speaker activity" : "Save live transcript",
            suggestedName: suggested,
            allowedContentTypes: [.plainText]
        ) else { return }
        do {
            try session.text.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            self.error = "Could not save \(url.lastPathComponent): \(error.localizedDescription)"
        }
    }
}
