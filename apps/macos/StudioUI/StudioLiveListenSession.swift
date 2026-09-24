import AppKit
import Combine
import MereRunContract
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// Audio ▸ Live: the Session surface over `speech listen` and `speech diarize-live`. Start
/// submits the task draft through the task runner, so the session is an inference job with a
/// Library row, Stop, and the Activity panel like any run; the page reads the job's stdout events
/// into the live transcript or the speaker activity as they arrive. Its settings — which of the
/// two commands, the microphone, the language and windows, the model — are the task draft the
/// Command view edits too.
struct StudioLiveListenSession: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var navigation: NavigationModel
    @Environment(\.studioTaskRunner) private var runner
    @Environment(\.studioTaskSessions) private var sessions
    @Environment(\.studioModelTitles) private var titles
    @ObservedObject private var models: StudioModelStore
    @StateObject private var session = StudioLiveListenModel()
    @State private var error: String?
    @State private var showsOptions = false
    private let ticker = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    private static let task = StudioTask.audioLive

    init(models: StudioModelStore) {
        _models = ObservedObject(wrappedValue: models)
    }

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

    private var copyText: String {
        showsSpeakers ? session.activity.displayText : session.transcript.committedText
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
                phase: session.phase,
                clock: StudioTimeFormat.string(session.elapsed),
                logLines: session.logLines,
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
            session.attach(controller, runner: runner)
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
                }
                Spacer(minLength: 8)
                StudioSessionSecondaryButton("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(copyText, forType: .string)
                }
                .disabled(copyText.isEmpty)
                .accessibilityLabel(showsSpeakers ? "Copy the speaker activity" : "Copy the live transcript")
                StudioSessionSecondaryButton("Save…", action: save)
                    .disabled(copyText.isEmpty)
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
            session.begin(request)
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
        guard let url = StudioSpecialistFiles.saveFile(
            title: showsSpeakers ? "Save live speaker activity" : "Save live transcript",
            suggestedName: suggested,
            allowedContentTypes: [.plainText]
        ) else { return }
        do {
            try copyText.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            self.error = "Could not save \(url.lastPathComponent): \(error.localizedDescription)"
        }
    }
}

/// The page's session state: which job is the session, the events it has streamed, and the
/// microphones the CLI lists. Outlives view replacement within the page; a session left running
/// while the user is elsewhere is adopted again from the runner's current job, with the stdout
/// the job has kept.
@MainActor
final class StudioLiveListenModel: ObservableObject {
    @Published private(set) var transcript = StudioLiveTranscriptAccumulator()
    @Published private(set) var activity = StudioLiveDiarizationAccumulator()
    @Published private(set) var requestID: UUID?
    @Published private(set) var templateID: CommandTemplateID?
    @Published private(set) var devices: [StudioListenDevice] = []
    @Published private(set) var devicesUnavailable = false
    @Published private(set) var now = Date()

    private var controller: MereRunController?
    private var subscription: AnyCancellable?
    private var stopEscalation: Task<Void, Never>?

    /// How long Stop waits for `speech listen` to finish on SIGINT before the job is terminated.
    static let stopGrace: Duration = .seconds(4)

    func attach(_ controller: MereRunController, runner: StudioTaskRunner?) {
        guard self.controller == nil else { return }
        self.controller = controller
        subscription = controller.jobs.events.sink { [weak self] event in
            self?.handle(event)
        }
        if let job = runner?.currentJob(for: .audioLive) {
            adopt(job)
        }
    }

    var job: Job? {
        guard let requestID else { return nil }
        return controller?.jobs.job(requestID: requestID)
    }

    var isActive: Bool {
        job?.state.isActive ?? false
    }

    var phase: StudioSessionPhase {
        guard let job else { return .idle }
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

    /// Seconds the session has run, frozen at its end.
    var elapsed: TimeInterval {
        guard let job, let started = job.startedAt else { return 0 }
        switch job.state {
        case .finished(_, let ended), .cancelled(_, let ended):
            return max(0, ended.timeIntervalSince(started))
        case .queued, .running, .preflightFailed:
            return max(0, now.timeIntervalSince(started))
        }
    }

    var logLines: [String] {
        guard let job else { return [] }
        return StudioRealtimeSessionLog.lines(job.log.lines, startedAt: job.startedAt ?? job.submittedAt)
    }

    var errorMessage: String? {
        templateID == .speechDiarizeLive ? activity.errorMessage : transcript.errorMessage
    }

    /// A session this page just started.
    func begin(_ request: StudioRunRequest) {
        stopEscalation?.cancel()
        requestID = request.id
        templateID = request.templateID
        transcript.beginSession()
        activity.beginSession()
        now = Date()
    }

    /// A session already running when the page appeared: what the job has printed so far
    /// becomes the transcript, so leaving and coming back loses nothing the job still holds.
    func adopt(_ job: Job) {
        requestID = job.request.requestID
        templateID = job.request.templateID
        transcript = StudioLiveTranscriptAccumulator()
        activity = StudioLiveDiarizationAccumulator()
        receive(job.liveText)
        now = Date()
    }

    /// Stops the session the way Ctrl-C does — SIGINT, which `speech listen` traps to flush its
    /// last events and exit cleanly — and terminates it if it has not ended after `stopGrace`.
    func stop() {
        guard let controller, let job, job.state.isActive else { return }
        if job.state.isQueued {
            controller.jobs.cancel(job.id)
            return
        }
        controller.jobs.interrupt(job.id)
        stopEscalation?.cancel()
        stopEscalation = Task { [weak self] in
            try? await Task.sleep(for: Self.stopGrace)
            guard !Task.isCancelled, let self, let job = self.job, job.state.isActive else { return }
            controller.jobs.cancel(job.id)
        }
    }

    func clear() {
        transcript.clear()
        activity.clear()
    }

    func tick() {
        now = Date()
    }

    func refreshDevices() async {
        guard let controller else { return }
        let result = await controller.utilityCommandResult(args: ["speech", "listen", "--list-devices"])
        devicesUnavailable = result.exitCode != 0
        guard result.exitCode == 0 else { return }
        devices = StudioListenDevice.parseList(result.stdout)
    }

    private func handle(_ event: JobStore.Event) {
        switch event {
        case .output(let job, let stream, let text):
            guard stream == .stdout, let requestID, job.request.requestID == requestID else { return }
            receive(text)
        case .started(let job), .changed(let job):
            if let requestID, job.request.requestID == requestID { objectWillChange.send() }
        case .finished(let job, _):
            guard let requestID, job.request.requestID == requestID else { return }
            stopEscalation?.cancel()
            objectWillChange.send()
        }
    }

    private func receive(_ text: String) {
        guard !text.isEmpty else { return }
        if templateID == .speechDiarizeLive {
            activity.receive(text)
        } else {
            transcript.receive(text)
        }
    }
}
