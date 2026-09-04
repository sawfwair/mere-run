import AppKit
import AVFoundation
import SwiftUI
import UniformTypeIdentifiers

// F1 token: the text/glyph color on an accent fill. Moves to MereRunTheme with the other v2 tokens.
private let onAccent = MereRunTheme.dynamic(light: "FFFFFF", dark: "1B160A")
// F1 token: the raised tile of a segmented control's selected segment.
private let segmentedSelection = MereRunTheme.dynamic(light: "FFFFFF", dark: "3A362E")

/// Video ▸ Subjects: the Project archetype for the SCAIL flow.
///
/// A stage rail (Plan → Track → Animate) beside the current stage's board, over a job bar. The
/// plan is written to `plan.json` and handed to `video prepare-masks`; the manifest, tracking, and
/// quality reports it writes back drive the preview, the subject rows, and the stats. Animation
/// hands the tracked masks to `video animate`. Every job is a durable Library row through
/// `StudioSpecialistRunner`, and a fingerprint of the mask inputs invalidates prepared masks the
/// moment any of them changes.
struct StudioSCAILView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @Environment(\.studioSubjectsProjectSeed) private var seed

    // Plan
    @State private var mode = "animation"
    @State private var drivingVideo = ""
    @State private var subjects = [
        StudioSCAILSubject(
            name: "subject-1",
            color: "blue",
            referencePrompt: "person",
            drivingPrompt: "person"
        )
    ]
    @State private var width = 832
    @State private var height = 480
    @State private var fps = 24
    @State private var useTrimRange = false
    @State private var inSeconds = 0.0
    @State private var outSeconds = 10.0
    @State private var maskThreshold = 0.05
    @State private var maskResolution = 1008
    @State private var seedSearchFrames = 48
    @State private var maskModel = "vision-segment-sam31"

    // Track
    @State private var previewFrame = 0
    @State private var corrections: [StudioSCAILCorrection] = []
    @State private var preparedManifest: StudioSCAILManifest?
    @State private var preparedDirectory: URL?
    @State private var preparedFingerprint = ""
    @State private var trackingReport: StudioSCAILTrackingReport?
    @State private var qualityReport: StudioSCAILQualityReport?
    @State private var showsMasks = true

    // Animate
    @State private var prompt = "a cinematic full-body performance with natural motion"
    @State private var negativePrompt = ""
    @State private var profile = "fast"
    @State private var steps = 40
    @State private var guidance = 5.0
    @State private var shift = 3.0
    @State private var sampler = "unipc"
    @State private var seedValue = "42"
    @State private var segmentLength = 81
    @State private var segmentOverlap = 5
    @State private var tailPolicy = "drop"
    @State private var carryDrivingAudio = true
    @State private var model = "video-scail2-14b-mlx"
    @State private var modelRoot = ""
    @State private var adapterPath = ""
    @State private var adapterStrength = 1.0
    @State private var outputPath = StudioSpecialistFiles.timestampedDirectory(component: "SCAIL")
        .deletingLastPathComponent()
        .appendingPathComponent("scail-\(UUID().uuidString.prefix(8)).mp4")
        .path

    // Jobs and chrome
    @State private var selectedStage: StudioSubjectsStage?
    @State private var planSavedAt: Date?
    @State private var maskRequestID: UUID?
    @State private var maskJob: StudioSubjectsJobBar.Job?
    @State private var animateRequestID: UUID?
    @State private var animateJob: StudioSubjectsJobBar.Job?
    @State private var notice: Notice?
    @State private var editingSubjectID: UUID?
    @State private var showsPlanMore = false
    @State private var showsTrackMore = false
    @State private var showsAnimateMore = false
    @State private var showsLog = false
    @State private var didSeed = false

    private struct Notice: Equatable {
        let severity: MereBanner.Severity
        let text: String

        static func == (lhs: Notice, rhs: Notice) -> Bool {
            lhs.text == rhs.text
        }
    }

    private static let subjectsPanelWidth: CGFloat = 300
    /// Below this column width the preview cannot sit beside the 300pt side panel without the
    /// transport bar being squeezed, so the panel drops under it.
    private static let sideBySideMinimumWidth: CGFloat = 640

    static func isWide(columnWidth: CGFloat) -> Bool {
        columnWidth - 48 >= sideBySideMinimumWidth
    }

    // MARK: - Derived state

    private var maskConfigurationFingerprint: String {
        let subjectValues = subjects.map {
            [
                $0.name, $0.color, $0.referenceImage, $0.referencePrompt, $0.drivingPrompt,
                $0.referenceBox, $0.drivingBox, $0.referencePositivePoints,
                $0.drivingPositivePoints, $0.referenceNegativePoints, $0.drivingNegativePoints,
            ].joined(separator: "|")
        }.joined(separator: "||")
        let correctionValues = corrections.map {
            [
                $0.subjectID, String($0.frameIndex), $0.box, $0.positivePoints,
                $0.negativePoints, $0.paintedMaskPath,
            ].joined(separator: "|")
        }.joined(separator: "||")
        return [
            mode, drivingVideo, String(width), String(height), String(fps),
            String(useTrimRange), String(inSeconds), String(outSeconds),
            String(maskThreshold), String(maskResolution), String(seedSearchFrames),
            maskModel, subjectValues, correctionValues,
        ].joined(separator: "§")
    }

    private var maskItem: StudioLibraryItem? {
        guard let maskRequestID else { return nil }
        return library.items.first { $0.id == maskRequestID }
    }

    private var animateItem: StudioLibraryItem? {
        guard let animateRequestID else { return nil }
        return library.items.first { $0.id == animateRequestID }
    }

    private var jobPhase: StudioSubjectsJobBar.Phase {
        let jobs: [(StudioLibraryItem, StudioSubjectsJobBar.Job)] = [
            (maskItem, maskJob), (animateItem, animateJob),
        ].compactMap { item, job in
            guard let item, let job else { return nil }
            return (item, job)
        }
        if let pending = jobs.first(where: { $0.0.status == .running || $0.0.status == .queued }) {
            return pending.0.status == .queued ? .queued(pending.1) : .running(pending.1)
        }
        guard let latest = jobs.max(by: { $0.0.updatedAt < $1.0.updatedAt }) else { return .idle }
        return .ended(latest.1, exitCode: latest.0.exitCode)
    }

    private var activeJobRequestID: UUID? {
        switch jobPhase {
        case .running(let job), .queued(let job):
            return job.stage == .track ? maskRequestID : animateRequestID
        case .idle, .ended:
            return nil
        }
    }

    private var progress: StudioSubjectsProgress {
        var progress = StudioSubjectsProgress()
        progress.planReady = planValidationError(includeRender: false) == nil
        progress.previewed = preparedManifest != nil
        progress.tracked = preparedManifest?.isTracked == true
        progress.animated = resultVideoURL != nil
        switch jobPhase {
        case .running(let job), .queued(let job):
            progress.runningStage = job.stage
        case .idle, .ended:
            progress.runningStage = nil
        }
        return progress
    }

    private var stage: StudioSubjectsStage {
        selectedStage ?? progress.defaultStage
    }

    private var frameCount: Int? {
        preparedManifest?.frameCount ?? trackingReport?.frameCount
    }

    private var manifestFPS: Double {
        preparedManifest?.fps ?? trackingReport?.fps ?? Double(fps)
    }

    private var stats: StudioSubjectsStats {
        StudioSubjectsStats.stats(
            manifest: preparedManifest,
            tracking: trackingReport,
            quality: qualityReport,
            correctionCount: corrections.count
        )
    }

    private var overlayURL: URL? {
        preparedManifest.map { resolve($0.overlayPreviewPath) }
    }

    private var beforeURL: URL? {
        if let manifest = preparedManifest {
            return resolve(manifest.drivingProxyPath ?? manifest.drivingSourcePath)
        }
        return pathURL(drivingVideo)
    }

    private var resultVideoURL: URL? {
        guard let animateItem else { return nil }
        return animateItem.allArtifactURLs.first { StudioOutputFileKind.classify($0) == .video }
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                stageRail
                stageContent
            }
            jobBar
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onAppear(perform: applySeed)
        .onReceive(controller.runCompletions) { result in
            guard result.templateID == .videoPrepareMasks, result.requestID == maskRequestID else {
                return
            }
            guard result.exitCode == 0 else {
                notice = Notice(
                    severity: .error,
                    text: "Mask preparation failed. Open Log, or the Library row, for diagnostics."
                )
                return
            }
            loadPreparedManifest()
        }
        .onChange(of: maskConfigurationFingerprint) { _, fingerprint in
            guard preparedManifest != nil, fingerprint != preparedFingerprint else { return }
            preparedManifest = nil
            preparedDirectory = nil
            trackingReport = nil
            qualityReport = nil
            notice = Notice(severity: .info, text: "Mask inputs changed. Preview or track masks again.")
        }
    }

    // MARK: - Stage rail

    private var stageRail: some View {
        VStack(alignment: .leading, spacing: 4) {
            ProjectEyebrow("Stages")
                .padding(EdgeInsets(top: 0, leading: 10, bottom: 6, trailing: 10))
            ForEach(StudioSubjectsStage.allCases) { candidate in
                stageRow(candidate, status: progress.status(of: candidate, selected: stage))
            }
            VStack(alignment: .leading, spacing: 0) {
                ProjectEyebrow("Project")
                    .padding(.bottom, 6)
                VStack(alignment: .leading, spacing: 4) {
                    Text(StudioSubjectsProjectCopy.name(drivingVideo: drivingVideo))
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(1)
                    Text(StudioSubjectsProjectCopy.summary(subjectCount: subjects.count, frameCount: frameCount))
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                    Text(StudioSubjectsProjectCopy.saved(planSavedAt))
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
            .padding(EdgeInsets(top: 18, leading: 10, bottom: 0, trailing: 10))
            Spacer(minLength: 0)
        }
        .padding(EdgeInsets(top: 22, leading: 12, bottom: 22, trailing: 12))
        .frame(width: 200, alignment: .topLeading)
        .frame(maxHeight: .infinity, alignment: .top)
        .overlay(alignment: .trailing) {
            Rectangle().fill(MereRunTheme.border.opacity(0.4)).frame(width: 1)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Stages")
    }

    private func stageRow(_ candidate: StudioSubjectsStage, status: StudioSubjectsStageStatus) -> some View {
        Button {
            selectedStage = candidate
        } label: {
            HStack(spacing: 10) {
                ZStack {
                    Circle().fill(stageCircleFill(status))
                    if status == .done {
                        Image(systemName: "checkmark")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundStyle(onAccent)
                    } else {
                        Text(String(candidate.rawValue))
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(status == .todo ? MereRunTheme.textMuted : onAccent)
                    }
                }
                .frame(width: 20, height: 20)
                Text(candidate.title)
                    .font(.system(size: 13, weight: status == .active ? .semibold : .medium))
                    .foregroundStyle(status == .todo ? MereRunTheme.textMuted : MereRunTheme.textPrimary)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 10)
            .frame(height: 34)
            .background {
                RoundedRectangle(cornerRadius: 9)
                    .fill(status == .active ? MereRunTheme.accentSoft : Color.clear)
            }
            .contentShape(RoundedRectangle(cornerRadius: 9))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(candidate.title) stage")
        .accessibilityValue(stageAccessibilityValue(status))
        .accessibilityAddTraits(status == .active ? .isSelected : [])
    }

    private func stageCircleFill(_ status: StudioSubjectsStageStatus) -> Color {
        switch status {
        case .done: MereRunTheme.green
        case .active: MereRunTheme.accent
        case .todo: MereRunTheme.surfaceRaised
        }
    }

    private func stageAccessibilityValue(_ status: StudioSubjectsStageStatus) -> String {
        switch status {
        case .done: "done"
        case .active: "current"
        case .todo: "not started"
        }
    }

    // MARK: - Stage content

    private var stageContent: some View {
        GeometryReader { geometry in
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if let notice {
                        MereBanner(severity: notice.severity, text: notice.text) {
                            self.notice = nil
                        }
                    }
                    stageHeader
                    switch stage {
                    case .plan: planBoard
                    case .track: trackBoard(wide: Self.isWide(columnWidth: geometry.size.width))
                    case .animate: animateBoard(wide: Self.isWide(columnWidth: geometry.size.width))
                    }
                }
                .padding(EdgeInsets(top: 22, leading: 24, bottom: 16, trailing: 24))
                .frame(minHeight: geometry.size.height, alignment: .top)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var stageHeader: some View {
        let copy = StudioSubjectsStageCopy.copy(for: stage, progress: progress)
        return HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(copy.title)
                    .font(.system(size: 17, weight: .semibold, design: .serif))
                    .foregroundStyle(MereRunTheme.textPrimary)
                Text(copy.description)
                    .font(.system(size: 12.5))
                    .foregroundStyle(MereRunTheme.textSecondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)
            .accessibilityAddTraits(.isHeader)
            if let secondary = copy.secondary {
                ProjectSecondaryButton(secondary.label) { perform(secondary.action) }
                    .disabled(!secondary.isEnabled)
            }
            ProjectPrimaryButton(copy.primary.label) { perform(copy.primary.action) }
                .disabled(!copy.primary.isEnabled)
        }
    }

    private func perform(_ action: StudioSubjectsStageCopy.Action) {
        switch action {
        case .previewMasks: prepareMasks(previewOnly: true)
        case .trackMasks: prepareMasks(previewOnly: false)
        case .validateRun: animate(preflight: true)
        case .animate: animate(preflight: false)
        case .continueTo(let next): selectedStage = next
        }
    }

    // MARK: Plan board

    private var planBoard: some View {
        VStack(alignment: .leading, spacing: 14) {
            drivingClipPanel
            subjectsPanel
            moreRow("More · mask preparation", isExpanded: $showsPlanMore) {
                maskPreparationControls
            }
        }
    }

    private var drivingClipPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Driving clip")
                    .font(.system(size: 12.5, weight: .semibold))
                Spacer()
                ProjectSegmented(
                    items: [("animation", "Animate reference"), ("replacement", "Replace in scene")],
                    selection: $mode,
                    accessibilityLabel: "Mode"
                )
            }
            Text(
                mode == "animation"
                    ? "Keeps the reference background; the driving clip supplies motion."
                    : "Keeps the driving scene; references replace the masked subjects."
            )
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(MereRunTheme.textMuted)
            HStack(spacing: 8) {
                TextField("/path/to/driving.mp4", text: $drivingVideo)
                    .mereField()
                    .accessibilityLabel("Driving video")
                ProjectSecondaryButton("Choose…") {
                    if let url = StudioSpecialistFiles.chooseFile(
                        title: "Driving video",
                        allowedContentTypes: [.movie]
                    ).first {
                        drivingVideo = url.path
                    }
                }
            }
        }
        .padding(12)
        .projectPanel()
    }

    private var maskPreparationControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Stepper("Width \(width)", value: $width, in: 256...1_536, step: 32)
                Stepper("Height \(height)", value: $height, in: 256...1_536, step: 32)
                Stepper("FPS \(fps)", value: $fps, in: 1...60)
            }
            Toggle("Trim driving clip", isOn: $useTrimRange)
            if useTrimRange {
                valueSlider("In point", value: $inSeconds, range: 0...3_600)
                valueSlider("Out point", value: $outSeconds, range: 0.1...3_600)
            }
            valueSlider("SAM threshold", value: $maskThreshold, range: 0.001...0.5)
            HStack {
                Stepper("SAM resolution \(maskResolution)", value: $maskResolution, in: 256...2_016, step: 16)
                Stepper("Seed search frames \(seedSearchFrames)", value: $seedSearchFrames, in: 1...240)
            }
            labeledField("Mask model", text: $maskModel, placeholder: "vision-segment-sam31")
        }
    }

    // MARK: Track board

    private func trackBoard(wide: Bool) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            boardRow(
                preview: SubjectsPreview(
                    masksURL: overlayURL,
                    beforeURL: beforeURL,
                    frameCount: frameCount ?? 0,
                    fps: manifestFPS,
                    stillFrame: preparedManifest?.previewFrame,
                    showsMasks: $showsMasks,
                    emptyText: progress.planReady
                        ? "Preview a frame or track the clip to see masks here."
                        : "Finish the plan to preview masks."
                ),
                side: subjectsPanel,
                wide: wide
            )
            statsRow
            moreRow("More · corrections & preview frame", isExpanded: $showsTrackMore) {
                trackControls
            }
        }
    }

    @ViewBuilder
    private var statsRow: some View {
        let panels = stats.panels
        if !panels.isEmpty {
            HStack(spacing: 10) {
                ForEach(panels, id: \.label) { panel in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(panel.label)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(MereRunTheme.textMuted)
                        Text(panel.value)
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(MereRunTheme.textPrimary)
                    }
                    .padding(EdgeInsets(top: 10, leading: 12, bottom: 10, trailing: 12))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .projectPanel(cornerRadius: 8)
                    .accessibilityElement(children: .combine)
                }
            }
        }
    }

    private var trackControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            Stepper("Preview frame \(previewFrame)", value: $previewFrame, in: 0...100_000)
            Text("Keyframe corrections refine tracked masks at a frame with a box, positive/negative points, or a painted binary PNG. Re-track to apply them.")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            ForEach($corrections) { $correction in
                correctionEditor($correction)
            }
            ProjectSecondaryButton("Add correction") {
                if let first = subjects.first {
                    corrections.append(StudioSCAILCorrection(subjectID: first.name))
                }
            }
        }
    }

    private func correctionEditor(_ correction: Binding<StudioSCAILCorrection>) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Picker("Subject", selection: correction.subjectID) {
                    ForEach(subjects) { subject in
                        Text(subject.name).tag(subject.name)
                    }
                }
                .fixedSize()
                Stepper("Frame \(correction.wrappedValue.frameIndex)", value: correction.frameIndex, in: 0...100_000)
                Spacer()
                Button(role: .destructive) {
                    corrections.removeAll { $0.id == correction.wrappedValue.id }
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Remove correction")
            }
            TextField("Box x1,y1,x2,y2", text: correction.box)
                .mereField()
            HStack(spacing: 8) {
                TextField("Positive points x,y; x,y", text: correction.positivePoints)
                    .mereField()
                TextField("Negative points x,y; x,y", text: correction.negativePoints)
                    .mereField()
            }
            HStack(spacing: 8) {
                TextField("Painted binary correction PNG", text: correction.paintedMaskPath)
                    .mereField()
                ProjectSecondaryButton("Choose…") {
                    if let url = StudioSpecialistFiles.chooseFile(
                        title: "Painted binary correction",
                        allowedContentTypes: [.png]
                    ).first {
                        correction.wrappedValue.paintedMaskPath = url.path
                    }
                }
            }
        }
        .padding(10)
        .projectPanel(cornerRadius: 9)
    }

    // MARK: Animate board

    private func animateBoard(wide: Bool) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            boardRow(
                preview: SubjectsPreview(
                    masksURL: resultVideoURL ?? overlayURL,
                    beforeURL: beforeURL,
                    frameCount: frameCount ?? 0,
                    fps: manifestFPS,
                    stillFrame: resultVideoURL == nil ? preparedManifest?.previewFrame : nil,
                    showsMasks: $showsMasks,
                    emptyText: "The finished shot appears here.",
                    masksLabel: resultVideoURL == nil ? "Masks" : "Result"
                ),
                side: directionPanel,
                wide: wide
            )
            if let text = animateItem?.outputText, !text.isBlank, resultVideoURL == nil {
                runOutputPanel(text)
            }
            moreRow("More · continuity, model & output", isExpanded: $showsAnimateMore) {
                animateControls
            }
        }
    }

    private var directionPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Direction")
                .font(.system(size: 12.5, weight: .semibold))
            TextField("Describe the finished shot", text: $prompt, axis: .vertical)
                .lineLimit(2...5)
                .mereField()
                .accessibilityLabel("Prompt")
            labeledField("Negative prompt", text: $negativePrompt, placeholder: "Optional")
            HStack {
                Text("Profile")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                Spacer()
                ProjectSegmented(
                    items: [("fast", "Fast · 4-step"), ("quality", "Quality")],
                    selection: $profile,
                    accessibilityLabel: "Render profile"
                )
            }
            if profile == "quality" {
                Stepper("Steps \(steps)", value: $steps, in: 1...100)
                valueSlider("Guidance", value: $guidance, range: 0...15)
                valueSlider("Schedule shift", value: $shift, range: 0...10)
                Picker("Sampler", selection: $sampler) {
                    Text("UniPC").tag("unipc")
                    Text("Euler").tag("euler")
                }
            }
        }
        .padding(12)
        .projectPanel()
    }

    private func runOutputPanel(_ text: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Run output")
                .font(.system(size: 12, weight: .semibold))
                .padding(EdgeInsets(top: 10, leading: 14, bottom: 10, trailing: 14))
            ProjectHairline()
            ScrollView {
                Text(text)
                    .font(.system(size: 11.5, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(EdgeInsets(top: 10, leading: 14, bottom: 10, trailing: 14))
            }
            .frame(maxHeight: 160)
        }
        .projectPanel()
    }

    private var animateControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                labeledField("Seed", text: $seedValue, placeholder: "42")
                    .frame(maxWidth: 160)
                Stepper("Segment length \(segmentLength)", value: $segmentLength, in: 9...401, step: 8)
                Stepper("Overlap \(segmentOverlap)", value: $segmentOverlap, in: 1...77, step: 4)
            }
            HStack {
                Picker("Tail", selection: $tailPolicy) {
                    Text("Drop partial tail").tag("drop")
                    Text("Pad and trim").tag("pad-trim")
                }
                .fixedSize()
                Toggle("Carry driving audio", isOn: $carryDrivingAudio)
            }
            labeledField("Model", text: $model, placeholder: "video-scail2-14b-mlx")
            StudioPathField(
                label: "Model root override",
                placeholder: "Managed model",
                path: $modelRoot,
                picksDirectory: true
            )
            StudioPathField(
                label: "Distilled adapter override",
                placeholder: "Fast profile uses the managed adapter automatically",
                path: $adapterPath
            )
            if !adapterPath.isBlank {
                valueSlider("Adapter strength", value: $adapterStrength, range: 0...2)
            }
            VStack(alignment: .leading, spacing: 5) {
                Text("Output video")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                HStack(spacing: 8) {
                    TextField("/path/to/output.mp4", text: $outputPath)
                        .mereField()
                    ProjectSecondaryButton("Choose…") {
                        if let url = StudioSpecialistFiles.saveFile(
                            title: "Save SCAIL video",
                            suggestedName: "scail.mp4",
                            allowedContentTypes: [.mpeg4Movie]
                        ) {
                            outputPath = url.path
                        }
                    }
                }
            }
        }
    }

    // MARK: - Subjects panel

    private var subjectsPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Subjects")
                    .font(.system(size: 12.5, weight: .semibold))
                Spacer()
                Button(action: addSubject) {
                    Text("+ Add")
                        .font(.system(size: 11.5, weight: .medium))
                        .foregroundStyle(subjects.count < 6 ? MereRunTheme.accent : MereRunTheme.textMuted)
                }
                .buttonStyle(.plain)
                .disabled(subjects.count >= 6)
                .accessibilityLabel("Add subject")
            }
            .padding(EdgeInsets(top: 0, leading: 2, bottom: 4, trailing: 2))
            ForEach($subjects) { $subject in
                subjectRow($subject)
            }
        }
        .padding(12)
        .projectPanel()
    }

    private func subjectRow(_ subject: Binding<StudioSCAILSubject>) -> some View {
        let value = subject.wrappedValue
        let tracked = trackingReport?.subjects.first { $0.id == value.name }
        let trackedFrames = tracked.map { (visible: $0.visibleFrameCount, total: trackingReport?.frameCount ?? 0) }
        let correctionFrames = corrections.filter { $0.subjectID == value.name }.map(\.frameIndex)
        return HStack(spacing: 10) {
            subjectSwatch(value)
            VStack(alignment: .leading, spacing: 2) {
                Text(value.name)
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                Text(StudioSubjectsRowCopy.meta(
                    subject: value, trackedFrames: trackedFrames, correctionFrames: correctionFrames
                ))
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(2)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            ProjectIconButton(systemImage: "ellipsis", label: "Edit \(value.name)") {
                editingSubjectID = value.id
            }
            .popover(
                isPresented: Binding(
                    get: { editingSubjectID == value.id },
                    set: { if !$0 { editingSubjectID = nil } }
                ),
                arrowEdge: .trailing
            ) {
                subjectEditor(subject)
            }
        }
        .padding(EdgeInsets(top: 8, leading: 10, bottom: 8, trailing: 10))
        .background {
            RoundedRectangle(cornerRadius: 9)
                .fill(MereRunTheme.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: 9)
                        .strokeBorder(MereRunTheme.border.opacity(0.4), lineWidth: 1)
                }
        }
    }

    @ViewBuilder
    private func subjectSwatch(_ subject: StudioSCAILSubject) -> some View {
        let referenceURL = preparedManifest?.subjects
            .first { $0.id == subject.name }
            .map { resolve($0.preparedReferenceImagePath) }
            ?? pathURL(subject.referenceImage)
        ZStack {
            RoundedRectangle(cornerRadius: 6)
                .fill(StudioSubjectsPalette.color(named: subject.color).opacity(0.55))
            if let referenceURL, FileManager.default.fileExists(atPath: referenceURL.path) {
                StudioAsyncImagePreview(
                    url: referenceURL,
                    maxPixelSize: 144,
                    contentMode: .fill,
                    fallbackSystemImage: "photo"
                )
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(StudioSubjectsPalette.color(named: subject.color).opacity(0.35))
                }
            }
        }
        .frame(width: 36, height: 36)
        .accessibilityHidden(true)
    }

    private func subjectEditor(_ subject: Binding<StudioSCAILSubject>) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Circle()
                    .fill(StudioSubjectsPalette.color(named: subject.wrappedValue.color).opacity(0.75))
                    .frame(width: 10, height: 10)
                TextField("Subject id", text: subject.name)
                    .mereField()
                    .accessibilityLabel("Subject id")
                if subjects.count > 1 {
                    Button(role: .destructive) {
                        editingSubjectID = nil
                        subjects.removeAll { $0.id == subject.wrappedValue.id }
                        normalizeSubjectColors()
                    } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Remove subject")
                }
            }
            StudioPathField(
                label: "Reference image",
                placeholder: "/path/to/reference.png",
                path: subject.referenceImage,
                allowedContentTypes: [.image]
            )
            labeledField("Reference selector", text: subject.referencePrompt, placeholder: "woman in red")
            labeledField("Driving selector", text: subject.drivingPrompt, placeholder: "dancer")
            DisclosureGroup("Precise selectors") {
                VStack(alignment: .leading, spacing: 7) {
                    Text("Optional box: x1,y1,x2,y2 · points: x,y; x,y")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    TextField("Reference box", text: subject.referenceBox).mereField()
                    TextField("Driving box", text: subject.drivingBox).mereField()
                    TextField("Reference positive points", text: subject.referencePositivePoints).mereField()
                    TextField("Reference negative points", text: subject.referenceNegativePoints).mereField()
                    TextField("Driving positive points", text: subject.drivingPositivePoints).mereField()
                    TextField("Driving negative points", text: subject.drivingNegativePoints).mereField()
                }
                .padding(.top, 7)
            }
        }
        .padding(14)
        .frame(width: 380)
    }

    private func addSubject() {
        guard subjects.count < 6 else { return }
        let index = subjects.count
        subjects.append(
            StudioSCAILSubject(
                name: "subject-\(index + 1)",
                color: StudioSubjectsPalette.names[index],
                referencePrompt: "person",
                drivingPrompt: "person"
            )
        )
    }

    // MARK: - Job bar

    private var jobBar: some View {
        let phase = jobPhase
        let fraction = activeJobRequestID.flatMap { controller.progressByRequestID[$0]?.fractionCompleted }
        return HStack(spacing: 12) {
            Circle()
                .fill(StudioSubjectsJobBar.dotColor(phase: phase))
                .frame(width: 8, height: 8)
            Text("Video · Subjects")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textPrimary)
            Text(StudioSubjectsJobBar.detail(
                phase: phase, subjectCount: subjects.count, frameCount: frameCount, profile: profile
            ))
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(MereRunTheme.textMuted)
            .lineLimit(1)
            if let fraction {
                ProjectProgressBar(fraction: fraction)
                    .frame(maxWidth: 260)
                    .accessibilityLabel("Job progress")
                    .accessibilityValue("\(Int((fraction * 100).rounded())) percent")
            }
            Spacer(minLength: 8)
            ProjectSecondaryButton("Cancel", action: cancelActiveJob)
                .disabled(activeJobRequestID == nil)
            ProjectSecondaryButton("Log") { showsLog.toggle() }
                .disabled(maskRequestID == nil && animateRequestID == nil)
                .popover(isPresented: $showsLog, arrowEdge: .top) { logPopover }
        }
        .padding(.horizontal, 16)
        .frame(height: 40)
        .frame(maxWidth: .infinity)
        .background(MereRunTheme.background)
        .overlay(alignment: .top) {
            Rectangle().fill(MereRunTheme.border.opacity(0.53)).frame(height: 1)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Job bar")
    }

    private var logRequestID: UUID? {
        if let activeJobRequestID { return activeJobRequestID }
        let candidates = [maskItem, animateItem].compactMap { $0 }
        return candidates.max { $0.updatedAt < $1.updatedAt }?.id
    }

    private var logPopover: some View {
        let lines = logRequestID.map { controller.logs(for: $0).map(\.text) } ?? []
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Job log")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                Button("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(lines.joined(separator: "\n"), forType: .string)
                }
                .controlSize(.small)
                .disabled(lines.isEmpty)
            }
            ScrollView {
                Text(lines.isEmpty ? "No output yet." : lines.joined(separator: "\n"))
                    .font(.system(size: 11.5, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(14)
        .frame(width: 560, height: 340)
    }

    private func cancelActiveJob() {
        guard let activeJobRequestID else { return }
        controller.cancel(requestID: activeJobRequestID)
    }

    // MARK: - Shared pieces

    /// The preview beside a 300pt side panel, as the board lays them out; when the column is
    /// too narrow for both (the default window with the Library open), the panel drops below.
    @ViewBuilder
    private func boardRow<Side: View>(preview: SubjectsPreview, side: Side, wide: Bool) -> some View {
        if wide {
            HStack(alignment: .top, spacing: 16) {
                preview
                side.frame(width: Self.subjectsPanelWidth)
            }
            .frame(maxHeight: .infinity, alignment: .top)
        } else {
            VStack(alignment: .leading, spacing: 14) {
                preview
                side
            }
            .frame(maxHeight: .infinity, alignment: .top)
        }
    }

    private func moreRow<Content: View>(
        _ title: String,
        isExpanded: Binding<Bool>,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Button {
                isExpanded.wrappedValue.toggle()
            } label: {
                HStack(spacing: 4) {
                    Text(title)
                    Image(systemName: "chevron.right")
                        .font(.system(size: 9, weight: .semibold))
                        .rotationEffect(.degrees(isExpanded.wrappedValue ? 90 : 0))
                }
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
                .frame(height: 24)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityAddTraits(isExpanded.wrappedValue ? .isSelected : [])
            if isExpanded.wrappedValue {
                content()
                    .padding(12)
                    .projectPanel()
            }
        }
    }

    private func labeledField(_ label: String, text: Binding<String>, placeholder: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            TextField(placeholder, text: text)
                .mereField()
                .accessibilityLabel(label)
        }
    }

    private func valueSlider(_ label: String, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                Spacer()
                Text(value.wrappedValue.formatted(.number.precision(.fractionLength(2))))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .font(MereRunTheme.captionFont)
            Slider(value: value, in: range)
                .accessibilityLabel(label)
        }
    }

    // MARK: - Seed

    private func applySeed() {
        guard !didSeed, let seed else { return }
        didSeed = true
        mode = seed.mode
        drivingVideo = seed.drivingVideo
        width = seed.width
        height = seed.height
        fps = seed.fps
        subjects = seed.subjects
        corrections = seed.corrections
        planSavedAt = seed.planSavedAt
        selectedStage = seed.stage
        if let directory = seed.preparedDirectory {
            preparedDirectory = directory
            loadPreparedManifest()
        }
        if let requestID = seed.maskRequestID {
            maskRequestID = requestID
            maskJob = .track
        }
    }

    // MARK: - Jobs

    private func prepareMasks(previewOnly: Bool) {
        notice = nil
        guard validateInputs(includeRender: false) else { return }
        guard let planURL = writePlan() else { return }
        guard let template = CommandCatalog.template(id: .videoPrepareMasks) else {
            notice = Notice(severity: .error, text: "The mask preparation command is unavailable.")
            return
        }
        let suffix = previewOnly ? "preview-\(previewFrame)" : "tracked"
        let directory = planURL.deletingLastPathComponent()
            .appendingPathComponent(suffix, isDirectory: true)
        try? FileManager.default.removeItem(at: directory)
        preparedDirectory = directory

        var draft = template.defaultDraft()
        draft.inputPath = planURL.path
        draft.outputPath = directory.path
        draft.previewFrame = previewOnly ? String(previewFrame) : ""
        draft.model = maskModel
        maskJob = previewOnly ? .preview(frame: previewFrame) : .track
        maskRequestID = StudioSpecialistRunner.submit(
            templateID: .videoPrepareMasks,
            mode: .video,
            draft: draft,
            controller: controller,
            library: library
        )
        if selectedStage == .plan {
            selectedStage = .track
        }
    }

    private func animate(preflight: Bool) {
        notice = nil
        guard validateInputs(includeRender: true, preflight: preflight),
              let manifest = preparedManifest,
              let first = manifest.subjects.first,
              let drivingMask = manifest.drivingMaskPath else {
            notice = Notice(severity: .error, text: "Track the whole clip before animating.")
            return
        }
        guard let template = CommandCatalog.template(id: .videoAnimate) else {
            notice = Notice(severity: .error, text: "The SCAIL animation command is unavailable.")
            return
        }

        var draft = template.defaultDraft()
        draft.prompt = prompt
        draft.secondaryText = negativePrompt
        draft.inputPath = resolve(first.preparedReferenceImagePath).path
        draft.referenceMaskPath = resolve(first.referenceMaskPath).path
        draft.drivingVideoPath = resolve(manifest.drivingProxyPath ?? manifest.drivingSourcePath).path
        draft.drivingMaskPath = resolve(drivingMask).path
        draft.referenceImagePaths = manifest.subjects.dropFirst()
            .map { resolve($0.preparedReferenceImagePath).path }
            .joined(separator: "\n")
        draft.scailAdditionalReferenceMaskPaths = manifest.subjects.dropFirst()
            .map { resolve($0.referenceMaskPath).path }
            .joined(separator: "\n")
        draft.outputPath = outputPath
        draft.videoTaskMode = mode
        draft.renderProfile = profile
        draft.width = width
        draft.height = height
        draft.fps = fps
        draft.steps = steps
        draft.cfgScale = guidance
        draft.scheduleShift = shift
        draft.sampler = sampler
        draft.seed = seedValue
        draft.segmentLength = segmentLength
        draft.segmentOverlap = segmentOverlap
        draft.tailPolicy = tailPolicy
        draft.audioSource = carryDrivingAudio ? "driving" : "none"
        draft.model = model
        draft.modelRoot = modelRoot
        draft.loraPath = adapterPath
        draft.loraScale = adapterStrength
        draft.preflight = preflight
        draft.json = preflight
        animateJob = preflight ? .validate : .animate
        animateRequestID = StudioSpecialistRunner.submit(
            templateID: .videoAnimate,
            mode: .video,
            draft: draft,
            controller: controller,
            library: library
        )
    }

    private func validateInputs(includeRender: Bool, preflight: Bool = false) -> Bool {
        if let error = planValidationError(includeRender: includeRender, preflight: preflight) {
            notice = Notice(severity: .error, text: error)
            return false
        }
        return true
    }

    /// The first reason the plan (and, with `includeRender`, the render settings) cannot run.
    private func planValidationError(includeRender: Bool, preflight: Bool = false) -> String? {
        guard !drivingVideo.isBlank else {
            return "Choose a driving video."
        }
        guard !subjects.isEmpty, subjects.count <= 6 else {
            return "SCAIL supports one to six subjects."
        }
        for subject in subjects {
            if subject.name.isBlank || subject.referenceImage.isBlank {
                return "Every subject needs an id and reference image."
            }
            if !hasSelector(
                text: subject.referencePrompt,
                box: subject.referenceBox,
                points: subject.referencePositivePoints
            ) || !hasSelector(
                text: subject.drivingPrompt,
                box: subject.drivingBox,
                points: subject.drivingPositivePoints
            ) {
                return "Every subject needs both a reference and driving selector."
            }
            let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
            if subject.name.unicodeScalars.contains(where: { !allowed.contains($0) }) {
                return "Subject ids may contain only letters, numbers, hyphens, and underscores."
            }
        }
        if Set(subjects.map(\.name)).count != subjects.count {
            return "Subject ids must be unique."
        }
        if corrections.contains(where: { correction in
            !subjects.contains(where: { $0.name == correction.subjectID })
        }) {
            return "Every keyframe correction must target a current subject."
        }
        guard width.isMultiple(of: 32), height.isMultiple(of: 32) else {
            return "SCAIL dimensions must be divisible by 32."
        }
        if useTrimRange && outSeconds <= inSeconds {
            return "The driving-video out point must be after the in point."
        }
        if includeRender {
            guard segmentOverlap < segmentLength else {
                return "Continuity overlap must be shorter than the segment."
            }
            guard segmentLength % 4 == 1, segmentOverlap % 4 == 1 else {
                return "Segment length and overlap must equal 1 modulo 4."
            }
            if outputPath.isBlank {
                return "Choose an output video path."
            }
            if !preflight,
               FileManager.default.fileExists(
                   atPath: NSString(string: outputPath).expandingTildeInPath
               ) {
                return "Choose a new output filename so this result remains immutable in Library."
            }
        }
        return nil
    }

    private func writePlan() -> URL? {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MereRun/SCAIL Plans", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let url = base.appendingPathComponent("plan.json")
        do {
            let plan = StudioSCAILMaskPlan(
                mode: mode,
                drivingVideo: drivingVideo,
                inSeconds: useTrimRange ? inSeconds : nil,
                outSeconds: useTrimRange ? outSeconds : nil,
                width: width,
                height: height,
                fps: Double(fps),
                subjects: try subjects.map { subject in
                    StudioSCAILSubjectPlan(
                        id: subject.name,
                        color: subject.color,
                        referenceImage: subject.referenceImage,
                        referenceSelector: try selector(
                            text: subject.referencePrompt,
                            box: subject.referenceBox,
                            positivePoints: subject.referencePositivePoints,
                            negativePoints: subject.referenceNegativePoints,
                            label: "\(subject.name) reference"
                        ),
                        drivingSelector: try selector(
                            text: subject.drivingPrompt,
                            box: subject.drivingBox,
                            positivePoints: subject.drivingPositivePoints,
                            negativePoints: subject.drivingNegativePoints,
                            label: "\(subject.name) driving"
                        )
                    )
                },
                corrections: try corrections.map { correction in
                    StudioSCAILCorrectionPlan(
                        subjectID: correction.subjectID,
                        frameIndex: correction.frameIndex,
                        box: try parseBox(correction.box, label: "correction"),
                        positivePoints: try parsePoints(correction.positivePoints, label: "correction positive"),
                        negativePoints: try parsePoints(correction.negativePoints, label: "correction negative"),
                        paintedBinaryCorrectionPNG: correction.paintedMaskPath.isBlank
                            ? nil
                            : correction.paintedMaskPath
                    )
                },
                threshold: Float(maskThreshold),
                resolution: maskResolution,
                seedFrameSearchLimit: seedSearchFrames
            )
            try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try encoder.encode(plan).write(to: url, options: .atomic)
            planSavedAt = Date()
            return url
        } catch {
            notice = Notice(severity: .error, text: "Could not write mask plan: \(error.localizedDescription)")
            return nil
        }
    }

    private func hasSelector(text: String, box: String, points: String) -> Bool {
        !text.isBlank || !box.isBlank || !points.isBlank
    }

    private func selector(
        text: String,
        box: String,
        positivePoints: String,
        negativePoints: String,
        label: String
    ) throws -> StudioSCAILSelectorPlan {
        StudioSCAILSelectorPlan(
            text: text.isBlank ? nil : text,
            box: try parseBox(box, label: label),
            positivePoints: try parsePoints(positivePoints, label: "\(label) positive"),
            negativePoints: try parsePoints(negativePoints, label: "\(label) negative")
        )
    }

    private func parseBox(_ raw: String, label: String) throws -> StudioSCAILBoxPlan? {
        guard !raw.isBlank else { return nil }
        let values = raw.split(separator: ",").compactMap {
            Float($0.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        guard values.count == 4, values[2] > values[0], values[3] > values[1] else {
            throw StudioSCAILPlanError.invalidBox(label)
        }
        return StudioSCAILBoxPlan(x1: values[0], y1: values[1], x2: values[2], y2: values[3])
    }

    private func parsePoints(_ raw: String, label: String) throws -> [StudioSCAILPointPlan] {
        guard !raw.isBlank else { return [] }
        return try raw
            .replacingOccurrences(of: "\n", with: ";")
            .split(separator: ";")
            .map { pair in
                let values = pair.split(separator: ",").compactMap {
                    Float($0.trimmingCharacters(in: .whitespacesAndNewlines))
                }
                guard values.count == 2 else { throw StudioSCAILPlanError.invalidPoints(label) }
                return StudioSCAILPointPlan(x: values[0], y: values[1])
            }
    }

    private func loadPreparedManifest() {
        guard let preparedDirectory else { return }
        let url = preparedDirectory.appendingPathComponent("manifest.json")
        do {
            let manifest = try JSONDecoder().decode(StudioSCAILManifest.self, from: Data(contentsOf: url))
            preparedManifest = manifest
            preparedFingerprint = maskConfigurationFingerprint
            trackingReport = manifest.trackingPath.flatMap { path in
                try? JSONDecoder().decode(
                    StudioSCAILTrackingReport.self,
                    from: Data(contentsOf: resolve(path))
                )
            }
            qualityReport = manifest.qualityPath.flatMap { path in
                try? JSONDecoder().decode(
                    StudioSCAILQualityReport.self,
                    from: Data(contentsOf: resolve(path))
                )
            }
            if let quality = qualityReport, !quality.blockingErrors.isEmpty {
                notice = Notice(severity: .warning, text: quality.blockingErrors.joined(separator: " "))
            }
        } catch {
            notice = Notice(
                severity: .error,
                text: "Mask run completed, but manifest could not be read: \(error.localizedDescription)"
            )
        }
    }

    private func resolve(_ path: String) -> URL {
        let expanded = NSString(string: path).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded)
        }
        return (preparedDirectory ?? URL(fileURLWithPath: "/"))
            .appendingPathComponent(expanded)
    }

    private func pathURL(_ path: String?) -> URL? {
        guard let path, !path.isBlank else { return nil }
        return URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
    }

    private func normalizeSubjectColors() {
        for index in subjects.indices {
            subjects[index].color = StudioSubjectsPalette.names[index]
        }
    }
}

// MARK: - Preview

/// The 16:9 board: the overlay preview (an image after a frame preview, a video after a full
/// track) or the driving clip, with a transport that scrubs by frame and switches Masks ↔ Before.
private struct SubjectsPreview: View {
    let masksURL: URL?
    let beforeURL: URL?
    let frameCount: Int
    let fps: Double
    /// The frame a still overlay was rendered at; nil while the overlay is a video.
    let stillFrame: Int?
    @Binding var showsMasks: Bool
    let emptyText: String
    var masksLabel = "Masks"

    @StateObject private var playback = SubjectsPlayback()

    private var displayedURL: URL? { showsMasks ? masksURL : beforeURL }

    private var displayedKind: StudioOutputFileKind? {
        guard let displayedURL else { return nil }
        return StudioOutputFileKind.classify(displayedURL)
    }

    private var currentFrame: Int {
        if displayedKind == .image { return stillFrame ?? 0 }
        return playback.frame
    }

    private var fraction: Double {
        guard frameCount > 1 else { return 0 }
        return min(1, max(0, Double(currentFrame) / Double(frameCount - 1)))
    }

    var body: some View {
        ZStack {
            Color(hex: "101010")
            content
            transport
        }
        .aspectRatio(16 / 9, contentMode: .fit)
        .frame(maxWidth: .infinity)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(MereRunTheme.border, lineWidth: 1)
        }
        .onAppear(perform: syncPlayback)
        .onChange(of: displayedURL) { _, _ in syncPlayback() }
        .onChange(of: fps) { _, _ in syncPlayback() }
        .onDisappear { playback.pause() }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(showsMasks ? "Mask preview" : "Driving clip")
    }

    @ViewBuilder
    private var content: some View {
        switch displayedKind {
        case .image:
            if let displayedURL {
                StudioAsyncImagePreview(
                    url: displayedURL,
                    maxPixelSize: 1_600,
                    contentMode: .fit,
                    fallbackSystemImage: "photo"
                )
            }
        case .video:
            SubjectsPlayerLayer(player: playback.player)
        default:
            Text(emptyText)
                .font(.system(size: 12.5))
                .foregroundStyle(Color.white.opacity(0.55))
                .multilineTextAlignment(.center)
                .padding(24)
        }
    }

    private var transport: some View {
        let scrubbable = displayedKind == .video && frameCount > 1
        return VStack {
            Spacer()
            HStack(spacing: 10) {
                Button(action: playback.togglePlaying) {
                    TransportPlayGlyph(playing: playback.isPlaying)
                        .stroke(.white, style: StrokeStyle(lineWidth: 1.7, lineCap: .round, lineJoin: .round))
                        .frame(width: 16, height: 16)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(displayedKind != .video)
                .opacity(displayedKind == .video ? 1 : 0.4)
                .accessibilityLabel(playback.isPlaying ? "Pause" : "Play")
                ProjectScrubber(fraction: fraction, isEnabled: scrubbable) { newFraction in
                    guard frameCount > 1 else { return }
                    playback.seek(frame: Int((newFraction * Double(frameCount - 1)).rounded()))
                }
                .accessibilityLabel("Frame")
                .accessibilityValue("\(currentFrame) of \(frameCount)")
                Text("\(currentFrame) / \(frameCount)")
                    .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                    .fixedSize()
                ProjectSegmented(
                    items: [("masks", masksLabel), ("before", "Before")],
                    selection: Binding(
                        get: { showsMasks ? "masks" : "before" },
                        set: { showsMasks = $0 == "masks" }
                    ),
                    accessibilityLabel: "Preview layer"
                )
                .fixedSize()
            }
            .padding(EdgeInsets(top: 8, leading: 10, bottom: 8, trailing: 10))
            .background {
                RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.55))
            }
            .padding(EdgeInsets(top: 0, leading: 12, bottom: 10, trailing: 12))
        }
    }

    private func syncPlayback() {
        guard displayedKind == .video, let displayedURL else {
            playback.pause()
            return
        }
        playback.load(url: displayedURL, fps: fps, frame: stillFrame ?? playback.frame)
    }
}

/// One `AVPlayer` for the preview; frames are derived from the player clock at the clip's fps.
@MainActor
private final class SubjectsPlayback: ObservableObject {
    @Published private(set) var frame = 0
    @Published private(set) var isPlaying = false

    let player = AVPlayer()
    private var fps = 24.0
    private var url: URL?
    private let observers: Observers

    /// Holds the player's observer tokens outside the actor so they are released when the
    /// playback object goes away, without an isolated deinit.
    private final class Observers: @unchecked Sendable {
        let player: AVPlayer
        var timeObserver: AnyObject?
        var endObserver: NSObjectProtocol?

        init(player: AVPlayer) {
            self.player = player
        }

        func reset() {
            if let timeObserver {
                player.removeTimeObserver(timeObserver)
                self.timeObserver = nil
            }
            if let endObserver {
                NotificationCenter.default.removeObserver(endObserver)
                self.endObserver = nil
            }
        }

        deinit {
            reset()
        }
    }

    init() {
        player.actionAtItemEnd = .pause
        player.isMuted = true
        observers = Observers(player: player)
    }

    func load(url: URL, fps: Double, frame: Int) {
        let fps = fps > 0 ? fps : 24
        guard url != self.url || fps != self.fps else { return }
        self.url = url
        self.fps = fps
        player.pause()
        isPlaying = false
        let item = AVPlayerItem(url: url)
        player.replaceCurrentItem(with: item)
        installObservers(for: item)
        seek(frame: frame)
    }

    func togglePlaying() {
        guard player.currentItem != nil else { return }
        if isPlaying {
            player.pause()
            isPlaying = false
        } else {
            if let duration = player.currentItem?.duration, duration.isNumeric,
               player.currentTime() >= duration {
                player.seek(to: .zero)
            }
            player.play()
            isPlaying = true
        }
    }

    func pause() {
        player.pause()
        isPlaying = false
    }

    func seek(frame: Int) {
        let clamped = max(0, frame)
        self.frame = clamped
        let time = CMTime(seconds: Double(clamped) / fps, preferredTimescale: 600)
        player.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero)
    }

    private func installObservers(for item: AVPlayerItem) {
        observers.reset()
        let interval = CMTime(seconds: 1 / fps, preferredTimescale: 600)
        observers.timeObserver = player.addPeriodicTimeObserver(forInterval: interval, queue: .main) { [weak self] time in
            guard time.isNumeric else { return }
            MainActor.assumeIsolated {
                guard let self else { return }
                let frame = Int((time.seconds * self.fps).rounded())
                if frame != self.frame {
                    self.frame = frame
                }
            }
        } as AnyObject
        observers.endObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: item,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.isPlaying = false
            }
        }
    }
}

private struct SubjectsPlayerLayer: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        view.wantsLayer = true
        let layer = AVPlayerLayer(player: player)
        layer.videoGravity = .resizeAspect
        view.layer = layer
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        (view.layer as? AVPlayerLayer)?.player = player
    }
}

/// Stroke-only play triangle / pause bars on the 24-unit icon grid the mockup uses.
private struct TransportPlayGlyph: Shape {
    let playing: Bool

    func path(in rect: CGRect) -> Path {
        let unit = rect.width / 24
        var path = Path()
        if playing {
            path.addRoundedRect(
                in: CGRect(x: rect.minX + 6 * unit, y: rect.minY + 5 * unit, width: 4 * unit, height: 14 * unit),
                cornerSize: CGSize(width: unit, height: unit)
            )
            path.addRoundedRect(
                in: CGRect(x: rect.minX + 14 * unit, y: rect.minY + 5 * unit, width: 4 * unit, height: 14 * unit),
                cornerSize: CGSize(width: unit, height: unit)
            )
        } else {
            path.move(to: CGPoint(x: rect.minX + 7 * unit, y: rect.minY + 5 * unit))
            path.addLine(to: CGPoint(x: rect.minX + 19 * unit, y: rect.minY + 12 * unit))
            path.addLine(to: CGPoint(x: rect.minX + 7 * unit, y: rect.minY + 19 * unit))
            path.closeSubpath()
        }
        return path
    }
}

/// The transport's 3pt track with an 11pt thumb; drags seek.
private struct ProjectScrubber: View {
    let fraction: Double
    let isEnabled: Bool
    let onSeek: (Double) -> Void

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            ZStack(alignment: .leading) {
                Capsule().fill(Color.white.opacity(0.3)).frame(height: 3)
                Capsule().fill(Color.white).frame(width: width * fraction, height: 3)
                Circle()
                    .fill(Color.white)
                    .frame(width: 11, height: 11)
                    .offset(x: width * fraction - 5.5)
            }
            .frame(height: geometry.size.height)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        guard isEnabled, width > 0 else { return }
                        onSeek(min(1, max(0, value.location.x / width)))
                    }
            )
        }
        .frame(height: 16)
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Board pieces (gen.mjs primitives)

private struct ProjectEyebrow: View {
    let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        Text(text)
            .font(.system(size: 10.5, weight: .semibold))
            .kerning(0.63)
            .textCase(.uppercase)
            .foregroundStyle(MereRunTheme.textMuted)
            .lineLimit(1)
            .accessibilityAddTraits(.isHeader)
    }
}

private struct ProjectHairline: View {
    var body: some View {
        Rectangle()
            .fill(MereRunTheme.border.opacity(0.4))
            .frame(height: 1)
    }
}

/// `btnPrimary`: 28pt, accent fill, radius 6, 13pt semibold on `onAccent`.
private struct ProjectPrimaryButton: View {
    let label: String
    let action: () -> Void
    @Environment(\.isEnabled) private var isEnabled

    init(_ label: String, action: @escaping () -> Void) {
        self.label = label
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(onAccent)
                .lineLimit(1)
                .padding(.horizontal, 14)
                .frame(height: 28)
                .background {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(MereRunTheme.accent.opacity(isEnabled ? 1 : 0.45))
                }
                .contentShape(RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
    }
}

/// `btnSecondary`: 26pt, raised fill, hairline border, 11.5pt medium.
private struct ProjectSecondaryButton: View {
    let label: String
    let action: () -> Void
    @Environment(\.isEnabled) private var isEnabled

    init(_ label: String, action: @escaping () -> Void) {
        self.label = label
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(isEnabled ? MereRunTheme.textPrimary : MereRunTheme.textMuted)
                .lineLimit(1)
                .padding(.horizontal, 10)
                .frame(height: 26)
                .background {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(MereRunTheme.surfaceRaised)
                        .overlay {
                            RoundedRectangle(cornerRadius: 6)
                                .strokeBorder(MereRunTheme.border.opacity(0.6), lineWidth: 1)
                        }
                }
                .contentShape(RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
    }
}

/// `iconBtn`: a 28pt quiet glyph button.
private struct ProjectIconButton: View {
    let systemImage: String
    let label: String
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
                .frame(width: 28, height: 28)
                .background {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(hovering ? MereRunTheme.hoverFill : Color.clear)
                }
                .contentShape(RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .accessibilityLabel(label)
    }
}

/// `segmented`: a 2pt-padded raised pill; the selected segment is a raised 24pt tile.
private struct ProjectSegmented: View {
    let items: [(tag: String, title: String)]
    @Binding var selection: String
    let accessibilityLabel: String

    var body: some View {
        HStack(spacing: 2) {
            ForEach(items, id: \.tag) { item in
                let selected = item.tag == selection
                Button {
                    selection = item.tag
                } label: {
                    Text(item.title)
                        .font(.system(size: 12, weight: selected ? .semibold : .medium))
                        .foregroundStyle(selected ? MereRunTheme.textPrimary : MereRunTheme.textSecondary)
                        .lineLimit(1)
                        .padding(.horizontal, 12)
                        .frame(height: 24)
                        .background {
                            RoundedRectangle(cornerRadius: 5.5)
                                .fill(selected ? segmentedSelection : Color.clear)
                                .shadow(color: selected ? MereRunTheme.shadowColor : .clear, radius: 1, y: 1)
                        }
                        .contentShape(RoundedRectangle(cornerRadius: 5.5))
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(selected ? .isSelected : [])
            }
        }
        .padding(2)
        .background {
            RoundedRectangle(cornerRadius: 7).fill(MereRunTheme.surfaceRaised)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(accessibilityLabel)
    }
}

/// `progress`: a 4pt track in the raised fill with an accent bar.
private struct ProjectProgressBar: View {
    let fraction: Double

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(MereRunTheme.surfaceRaised)
                Capsule()
                    .fill(MereRunTheme.accent)
                    .frame(width: geometry.size.width * min(1, max(0, fraction)))
            }
        }
        .frame(height: 4)
        .frame(maxWidth: .infinity)
    }
}

private extension View {
    /// `panel`: surface fill, `border` at 80%, radius 10 by default.
    func projectPanel(cornerRadius: CGFloat = 10) -> some View {
        background {
            RoundedRectangle(cornerRadius: cornerRadius)
                .fill(MereRunTheme.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: cornerRadius)
                        .strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                }
        }
    }
}
