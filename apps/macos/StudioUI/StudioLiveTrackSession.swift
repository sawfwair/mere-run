import AVFoundation
import AppKit
import StudioKit
import SwiftUI

/// A camera `vision track-live --camera <index>` can open. The CLI indexes
/// `AVCaptureDevice.DiscoverySession` over the built-in, Continuity, and external cameras, so
/// Studio lists the same session in the same order and shows names for its numbers.
struct StudioCamera: Identifiable, Equatable {
    let index: Int
    let name: String

    var id: Int { index }

    static func connected() -> [StudioCamera] {
        AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .continuityCamera, .external],
            mediaType: .video,
            position: .unspecified
        )
        .devices
        .enumerated()
        .map { StudioCamera(index: $0.offset, name: $0.element.localizedName) }
    }
}

/// Vision ▸ Live on the Session archetype: Start/Stop over `vision track-live`, the camera and
/// the model as chips in the transport row, the capture's progress and log while it runs, and
/// the annotated clip once it lands in the Library. The draft is the task's `StudioTaskDraft`,
/// so the Command view edits the same form; the run goes through the task runner, which keeps
/// the camera-access gate (`MereRunController.run(studio:)`) in front of the CLI.
struct StudioLiveTrackSession: View {
    private static let task = StudioTask.visionLive

    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @EnvironmentObject private var navigation: NavigationModel
    @Environment(\.studioTaskRunner) private var runner
    @Environment(\.studioTaskSessions) private var sessions
    @ObservedObject private var models: StudioModelStore
    @StateObject private var jobMonitor = StudioJobMonitor()
    /// The cameras on this Mac, in the order the CLI numbers them.
    @State private var cameras: [StudioCamera] = []
    @State private var error: String?
    @State private var showsSettings = true
    /// The tracking document of the clip on screen, once it decodes.
    @State private var tracking: StudioVisionTrackDocument?

    init(models: StudioModelStore) {
        _models = ObservedObject(wrappedValue: models)
    }

    // MARK: Draft and run

    /// The task's draft, read and written through the session store so the Command view column
    /// edits the same value.
    private var draft: StudioTaskDraft {
        get { sessions?.taskDraft(for: Self.task) ?? StudioTaskDraft(templateID: .visionTrackLive) }
        nonmutating set { sessions?.setTaskDraft(newValue, for: Self.task) }
    }

    private var draftBinding: Binding<StudioTaskDraft> {
        Binding(get: { draft }, set: { draft = $0 })
    }

    /// The run this task last submitted, as the runner remembers it.
    private var requestID: UUID? {
        sessions?.value(for: Self.task.rawValue + ".requestID", default: Optional<UUID>.none)
    }

    private var item: StudioLibraryItem? {
        guard let requestID else { return nil }
        return library.items.first { $0.id == requestID }
    }

    private var job: Job? {
        _ = jobMonitor.generation
        return requestID.flatMap(jobMonitor.job(requestID:))
    }

    private var readiness: ModelReadinessState {
        controller.readiness(for: Self.task)
    }

    private var activePullJob: Job? {
        _ = jobMonitor.generation
        return jobMonitor.pullJob(for: StudioTaskSchema.modelID(for: draft))
    }

    private var phase: StudioSessionPhase {
        guard let item else { return .idle }
        if let job, job.state.isActive { return job.state.isRunning ? .live : .queued }
        switch item.status {
        case .queued: return .queued
        case .running:
            // Submitted but never launched: the system is still asking for the camera (the
            // controller retries once it answers), or it said no.
            return isWaitingForCameraAccess ? .queued : .ended(exitCode: item.exitCode)
        case .completed, .failed, .cancelled, .interrupted: return .ended(exitCode: item.exitCode)
        }
    }

    /// The row was submitted, no job exists, and macOS has not answered the camera prompt yet.
    private var isWaitingForCameraAccess: Bool {
        guard let item, item.status == .running, job == nil else { return false }
        return AVCaptureDevice.authorizationStatus(for: .video) == .notDetermined
    }

    /// The Mac will not give mere.run the camera; the page says so above the empty state.
    private var cameraAccessHint: String? {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .denied, .restricted: return StudioTaskRunner.cameraDeniedMessage
        case .authorized, .notDetermined: return nil
        @unknown default: return nil
        }
    }

    private var isPending: Bool {
        phase == .live || phase == .queued
    }

    private var selectedCamera: Int {
        Int(draft.text("--camera")) ?? 0
    }

    private var cameraTitle: String {
        cameras.first { $0.index == selectedCamera }?.name ?? "Camera \(selectedCamera)"
    }

    private var durationSeconds: Double {
        Double(draft.text("--duration-seconds")) ?? 10
    }

    // MARK: Body

    var body: some View {
        HStack(spacing: 0) {
            Group {
                if let job {
                    StudioJobObserver(job: job) { surface(job: $0) }
                } else {
                    surface(job: nil)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            if showsSettings {
                Rectangle()
                    .fill(MereRunTheme.border.opacity(0.53))
                    .frame(width: 1)
                settingsColumn
            }
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onAppear {
            jobMonitor.attach(controller.jobs)
            refreshCameras()
            refreshReadiness()
        }
        .onChange(of: draft.model) { _, _ in
            error = nil
            refreshReadiness()
        }
        .onReceive(NotificationCenter.default.publisher(for: AVCaptureDevice.wasConnectedNotification)) { _ in refreshCameras() }
        .onReceive(NotificationCenter.default.publisher(for: AVCaptureDevice.wasDisconnectedNotification)) { _ in refreshCameras() }
        .task(id: item?.id) { await loadTracking() }
    }

    /// The Session shell, ticking once a second while the capture runs so the clock moves. The
    /// schedule starts at the row's own start, so a re-render never restarts it.
    private func surface(job: Job?) -> some View {
        let start = job?.startedAt ?? item?.createdAt ?? .distantPast
        return TimelineView(.periodic(from: start, by: phase == .live ? 1 : 3_600)) { context in
            StudioSessionSurface(
                phase: phase,
                clock: clock(at: context.date, job: job),
                logLines: job?.log.lines.map(\.text) ?? [],
                onToggle: toggle
            ) {
                cameraChip
                modelChip
                StudioSessionSecondaryButton(showsSettings ? "Hide settings" : "Settings") {
                    withAnimation(MereRunTheme.Motion.quick) { showsSettings.toggle() }
                }
                .help("Duration, seed frame, threshold, resolution, and the overlay")
            } live: {
                livePanel(job: job)
            }
        }
    }

    private func clock(at now: Date, job: Job?) -> String {
        switch phase {
        case .idle, .queued:
            return StudioRealtimeTransport.timestamp(0)
        case .live:
            let startedAt = job?.startedAt ?? item?.createdAt ?? now
            return StudioRealtimeTransport.timestamp(now.timeIntervalSince(startedAt))
        case .ended:
            guard let item else { return StudioRealtimeTransport.timestamp(0) }
            return StudioRealtimeTransport.timestamp(item.updatedAt.timeIntervalSince(item.createdAt))
        }
    }

    // MARK: Transport chips

    /// Which camera the CLI opens, by the name AVFoundation gives it; a number when the list is
    /// empty, so the argv is still reachable.
    private var cameraChip: some View {
        Menu {
            if cameras.isEmpty {
                Text("No camera found")
            }
            ForEach(cameras) { camera in
                Toggle(isOn: Binding(
                    get: { selectedCamera == camera.index },
                    set: { _ in draft.form["--camera"] = .integer(camera.index) }
                )) {
                    Text(camera.name)
                }
            }
        } label: {
            StudioComposerChipLabel(title: cameraTitle, leadingSystemImage: "web.camera")
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(isPending)
        .help("Camera (--camera)")
        .accessibilityLabel("Camera")
        .accessibilityValue(cameraTitle)
    }

    private var modelChip: some View {
        StudioModelChip(
            scope: StudioTaskSchema.modelScope(for: draft),
            model: draftBinding.model,
            modelInventory: models.rows,
            readiness: readiness,
            onShowModels: { navigation.open(task: .modelsInstalled) }
        )
        .disabled(isPending)
    }

    // MARK: Live panel

    private func livePanel(job: Job?) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                promptField
                if readiness.blocksRun || activePullJob != nil {
                    StudioReadinessCard(
                        readiness: readiness,
                        pullJob: activePullJob,
                        actions: readinessActions,
                        onCancelPull: { _ = jobMonitor.cancel($0) }
                    )
                }
                if let error {
                    MereBanner(severity: .error, text: error, onDismiss: { self.error = nil })
                }
                if let cameraAccessHint {
                    MereBanner(severity: .warning, text: cameraAccessHint)
                }
                stateView(job: job)
            }
            .padding(24)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
    }

    private var promptField: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Track")
                .font(.system(size: 11, weight: .semibold))
                .kerning(0.55)
                .textCase(.uppercase)
                .foregroundStyle(MereRunTheme.textMuted)
                .accessibilityAddTraits(.isHeader)
            StudioInspectorTextField(placeholder: Self.task.presentation.promptPlaceholder, text: draftBinding.prompt, lines: 1...4)
                .disabled(isPending)
                .accessibilityLabel(Self.task.presentation.promptPlaceholder)
            Text("One thing per line, in view when the capture starts.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
    }

    @ViewBuilder
    private func stateView(job: Job?) -> some View {
        switch phase {
        case .idle:
            StudioEmptyState(presentation: Self.task.presentation, isCompact: true, onUseExample: { example in
                var next = draft
                next.prompt = example
                draft = next
            })
            .frame(maxWidth: .infinity)
            .padding(.top, 24)
        case .queued, .live:
            if let job {
                StudioLiveTrackProgress(job: job, cameraName: cameraTitle, durationSeconds: durationSeconds)
            } else if isWaitingForCameraAccess {
                HStack(spacing: 10) {
                    ProgressView().controlSize(.small)
                    Text("Waiting for camera access. Allow mere.run to use the camera in the prompt macOS is showing.")
                        .font(.system(size: 12.5))
                        .foregroundStyle(MereRunTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .accessibilityElement(children: .combine)
            }
        case .ended:
            if let item { endedView(item: item, job: job) }
        }
    }

    @ViewBuilder
    private func endedView(item: StudioLibraryItem, job: Job?) -> some View {
        // Only a finished capture's clip plays: a Stop mid-tracking can leave a truncated file.
        let clip = item.status == .completed ? item.allArtifactURLs.first {
            StudioOutputFileKind.classify($0) == .video && FileManager.default.fileExists(atPath: $0.path)
        } : nil
        VStack(alignment: .leading, spacing: 10) {
            if let clip {
                StudioVideoPlayerView(url: clip)
                    .aspectRatio(videoAspect, contentMode: .fit)
                    .frame(maxHeight: 440)
                    .mereMediaFrame()
                if let tracking {
                    StudioAnalyzeTrackScrubber(document: tracking)
                }
                HStack(spacing: 8) {
                    StudioSessionSecondaryButton("Quick Look") { QuickLookCoordinator.shared.preview(clip) }
                    StudioSessionSecondaryButton("Reveal in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([clip])
                    }
                    StudioSessionSecondaryButton("Save clip…") { saveOutput(clip) }
                    Spacer()
                    Text(StudioFeedTime.label(for: item.updatedAt))
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            } else if item.status == .cancelled {
                Text("Capture stopped before the clip was written. The session ends on its own after \(Self.seconds(durationSeconds)).")
                    .font(.system(size: 13))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else if item.status == .failed || item.status == .interrupted {
                MereBanner(
                    severity: .error,
                    text: StudioFailureSummary.summary(
                        outputText: item.outputText, logLines: job?.log.lines.map(\.text) ?? [], exitCode: item.exitCode
                    )
                )
            } else if item.status == .running {
                Text("The capture never started. Check the log below, then start again.")
                    .font(.system(size: 13))
                    .foregroundStyle(MereRunTheme.textSecondary)
            } else {
                Text("The capture finished but no clip was found. Check the log below.")
                    .font(.system(size: 13))
                    .foregroundStyle(MereRunTheme.textSecondary)
            }
        }
    }

    private var videoAspect: CGFloat {
        guard let tracking, tracking.frameHeight > 0 else { return 16.0 / 9 }
        return CGFloat(tracking.frameWidth) / CGFloat(tracking.frameHeight)
    }

    // MARK: Settings column

    /// The contract's sections without the fields the transport row's chips own: the model, and
    /// the camera while there is a list to pick from (with no camera found, the index stays a
    /// number field here, as the page's stepper was).
    private var sections: [(section: StudioTaskSection, fields: [StudioContractField<StudioTaskDraft>])] {
        StudioTaskSchema.sections(for: Self.task, draft: draft).compactMap { section in
            let fields = section.fields.filter { field in
                field.overrideID != .model && (field.flag != "--camera" || cameras.isEmpty)
            }
            return fields.isEmpty ? nil : (section, fields)
        }
    }

    private var advancedFields: [StudioContractField<StudioTaskDraft>] {
        StudioTaskSchema.advanced(for: Self.task, draft: draft)
    }

    private var settingsColumn: some View {
        let baseline = StudioTaskDraft(templateID: draft.templateID)
        let dependencies = StudioTaskSchema.dependencies(for: draft)
        return VStack(spacing: 0) {
            HStack {
                Text("Session settings")
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textPrimary)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 16)
            .frame(height: 44)
            .overlay(alignment: .bottom) {
                Rectangle().fill(MereRunTheme.border.opacity(0.4)).frame(height: 1)
            }
            ScrollView {
                VStack(spacing: 0) {
                    ForEach(sections, id: \.section.id) { entry in
                        StudioInspectorSectionView(
                            title: entry.section.title,
                            canReset: entry.fields.contains { $0.changedCount(draft: draft, baseline: baseline) > 0 },
                            onReset: {
                                var next = draft
                                for field in entry.fields { field.reset(&next, to: baseline) }
                                draft = next
                            }
                        ) {
                            ContractForm(fields: entry.fields, dependencies: dependencies, draft: draftBinding) { _ in
                                EmptyView()
                            }
                        }
                    }
                    if !advancedFields.isEmpty {
                        StudioInspectorSectionView(title: "Advanced", canReset: false, onReset: {}) {
                            ContractForm(fields: advancedFields, dependencies: dependencies, draft: draftBinding) { _ in
                                EmptyView()
                            }
                        }
                    }
                    Text("Stop ends the capture without a clip; the session ends on its own after \(Self.seconds(durationSeconds)). The Command view shows the exact arguments.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(16)
                }
            }
        }
        .frame(width: StudioLayoutPolicy.inspectorWidth)
        .disabled(isPending)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Session settings")
    }

    // MARK: Actions

    private var readinessActions: StudioReadinessActions {
        StudioReadinessActions(
            scope: StudioTaskSchema.modelScope(for: draft),
            model: draftBinding.model,
            modelInventory: models.rows,
            pullModel: { pull(modelID: StudioTaskSchema.modelID(for: draft)) },
            openModels: { navigation.open(task: .modelsInstalled) },
            recheck: refreshReadiness
        )
    }

    private func toggle() {
        error = nil
        if isPending {
            runner?.stop(task: Self.task)
            return
        }
        guard let runner else { return }
        guard !draft.prompt.isBlank else {
            error = "Name something to track first."
            return
        }
        do {
            let request = try runner.run(draft, task: Self.task)
            tracking = nil
            navigation.selectedLibraryID = request.id
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func refreshReadiness() {
        controller.checkReadiness(for: Self.task, requirement: StudioTaskSchema.requirement(for: draft))
    }

    /// Lists the cameras now attached; an index remembered for a camera that is gone falls back
    /// to the first one, so the chip never names a camera that is not there.
    private func refreshCameras() {
        cameras = StudioCamera.connected()
        if !cameras.isEmpty, !cameras.contains(where: { $0.index == selectedCamera }) {
            draft.form["--camera"] = .integer(0)
        }
    }

    /// Gets a managed model through the same `model pull` job the readiness card reports.
    private func pull(modelID: String) {
        guard !modelID.isBlank, let template = CommandCatalog.template(id: .modelPull) else { return }
        var commandDraft = template.defaultDraft()
        commandDraft.model = modelID
        let request = StudioRunRequest(mode: template.libraryMode, templateID: .modelPull, template: template, draft: commandDraft)
        if !models.startPull(request) {
            error = controller.status
            refreshReadiness()
        }
    }

    private func saveOutput(_ url: URL) {
        guard let destination = StudioFilePanels.saveFile(title: "Save clip", suggestedName: url.lastPathComponent) else {
            return
        }
        do {
            try StudioFileExport.copy(url, to: destination)
        } catch {
            self.error = "Could not save \(url.lastPathComponent): \(error.localizedDescription)"
        }
    }

    private func loadTracking() async {
        guard let item, item.status == .completed, let url = StudioAnalyzeDocumentSource.url(for: item) else {
            tracking = nil
            return
        }
        let loaded = await Task.detached(priority: .userInitiated) {
            (try? Data(contentsOf: url)).flatMap(StudioAnalyzeDocument.decode)
        }.value
        guard !Task.isCancelled else { return }
        if case .tracking(let document) = loaded {
            tracking = document
        } else {
            tracking = nil
        }
    }

    /// "10 s", "1 min 30 s"
    private static func seconds(_ value: Double) -> String {
        let total = Int(value.rounded())
        guard total >= 60 else { return "\(total) s" }
        return total % 60 == 0 ? "\(total / 60) min" : "\(total / 60) min \(total % 60) s"
    }
}

/// Re-renders its content when the job it holds publishes, so the transport clock, the log,
/// and the progress follow the process without the page observing every job in the store.
private struct StudioJobObserver<Content: View>: View {
    @ObservedObject var job: Job
    @ViewBuilder let content: (Job) -> Content

    var body: some View {
        content(job)
    }
}

/// The capture in flight: the status line and progress the CLI reports, the last lines of the
/// log, and which camera it is recording for how long.
private struct StudioLiveTrackProgress: View {
    @ObservedObject var job: Job
    let cameraName: String
    let durationSeconds: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                if let fraction = job.progress?.fractionCompleted {
                    ProgressView(value: fraction)
                        .frame(maxWidth: 220)
                } else {
                    ProgressView()
                        .controlSize(.small)
                }
                Text(job.progress?.label ?? job.status)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                if let detail = job.progress?.detail {
                    Text(detail)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
            }
            Text(job.state.isRunning
                 ? "Recording from \(cameraName) for \(Int(durationSeconds.rounded())) s, then tracking what it saw."
                 : "Waiting for the inference lane.")
                .font(.system(size: 12.5))
                .foregroundStyle(MereRunTheme.textSecondary)
            let recent = job.log.lines.suffix(6)
            if !recent.isEmpty {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(recent) { line in
                        Text(line.text)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(MereRunTheme.textMuted)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                        .fill(MereRunTheme.surface)
                        .overlay {
                            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                                .strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                        }
                }
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Capture in progress: \(job.status)")
    }
}
