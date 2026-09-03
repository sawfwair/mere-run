import SwiftUI
import UniformTypeIdentifiers

private struct StudioSCAILSubject: Identifiable, Equatable {
    let id = UUID()
    var name: String
    var color: String
    var referenceImage = ""
    var referencePrompt = ""
    var drivingPrompt = ""
    var referenceBox = ""
    var drivingBox = ""
    var referencePositivePoints = ""
    var drivingPositivePoints = ""
    var referenceNegativePoints = ""
    var drivingNegativePoints = ""
}

private struct StudioSCAILSelectorPlan: Codable {
    let text: String?
    let box: StudioSCAILBoxPlan?
    let positivePoints: [StudioSCAILPointPlan]
    let negativePoints: [StudioSCAILPointPlan]

    enum CodingKeys: String, CodingKey {
        case text
        case box
        case positivePoints = "positive_points"
        case negativePoints = "negative_points"
    }
}

private struct StudioSCAILPointPlan: Codable {
    let x: Float
    let y: Float
}

private struct StudioSCAILBoxPlan: Codable {
    let x1: Float
    let y1: Float
    let x2: Float
    let y2: Float
}

private struct StudioSCAILSubjectPlan: Codable {
    let id: String
    let color: String
    let referenceImage: String
    let referenceSelector: StudioSCAILSelectorPlan
    let drivingSelector: StudioSCAILSelectorPlan

    enum CodingKeys: String, CodingKey {
        case id
        case color
        case referenceImage = "reference_image"
        case referenceSelector = "reference_selector"
        case drivingSelector = "driving_selector"
    }
}

private struct StudioSCAILMaskPlan: Codable {
    let schemaVersion = 1
    let mode: String
    let drivingVideo: String
    let inSeconds: Double?
    let outSeconds: Double?
    let width: Int
    let height: Int
    let fps: Double
    let subjects: [StudioSCAILSubjectPlan]
    let corrections: [StudioSCAILCorrectionPlan]
    let threshold: Float
    let resolution: Int
    let seedFrameSearchLimit: Int
    let paletteTolerance = 192

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case mode
        case drivingVideo = "driving_video"
        case inSeconds = "in_seconds"
        case outSeconds = "out_seconds"
        case width
        case height
        case fps
        case subjects
        case corrections
        case threshold
        case resolution
        case seedFrameSearchLimit = "seed_frame_search_limit"
        case paletteTolerance = "palette_tolerance"
    }
}

private struct StudioSCAILCorrectionPlan: Codable {
    let subjectID: String
    let frameIndex: Int
    let box: StudioSCAILBoxPlan?
    let positivePoints: [StudioSCAILPointPlan]
    let negativePoints: [StudioSCAILPointPlan]
    let paintedBinaryCorrectionPNG: String?

    enum CodingKeys: String, CodingKey {
        case subjectID = "subject_id"
        case frameIndex = "frame_index"
        case box
        case positivePoints = "positive_points"
        case negativePoints = "negative_points"
        case paintedBinaryCorrectionPNG = "painted_binary_correction_png"
    }
}

private struct StudioSCAILCorrection: Identifiable, Equatable {
    let id = UUID()
    var subjectID: String
    var frameIndex = 0
    var box = ""
    var positivePoints = ""
    var negativePoints = ""
    var paintedMaskPath = ""
}

private enum StudioSCAILPlanError: LocalizedError {
    case invalidBox(String)
    case invalidPoints(String)

    var errorDescription: String? {
        switch self {
        case .invalidBox(let label):
            "\(label) box must be four comma-separated numbers with x2>x1 and y2>y1."
        case .invalidPoints(let label):
            "\(label) points must use x,y pairs separated by semicolons."
        }
    }
}

private struct StudioSCAILManifest: Decodable {
    struct Subject: Decodable {
        let id: String
        let preparedReferenceImagePath: String
        let referenceMaskPath: String

        enum CodingKeys: String, CodingKey {
            case id
            case preparedReferenceImagePath = "prepared_reference_image_path"
            case referenceMaskPath = "reference_mask_path"
        }
    }

    let status: String
    let drivingSourcePath: String
    let drivingProxyPath: String?
    let drivingMaskPath: String?
    let overlayPreviewPath: String
    let contactSheetPath: String
    let frameCount: Int
    let subjects: [Subject]

    enum CodingKeys: String, CodingKey {
        case status
        case drivingSourcePath = "driving_source_path"
        case drivingProxyPath = "driving_proxy_path"
        case drivingMaskPath = "driving_mask_path"
        case overlayPreviewPath = "overlay_preview_path"
        case contactSheetPath = "contact_sheet_path"
        case frameCount = "frame_count"
        case subjects
    }
}

struct StudioSCAILView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore

    @State private var mode = "animation"
    @State private var prompt = "a cinematic full-body performance with natural motion"
    @State private var negativePrompt = ""
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
    @State private var fps = 16
    @State private var useTrimRange = false
    @State private var inSeconds = 0.0
    @State private var outSeconds = 10.0
    @State private var maskThreshold = 0.05
    @State private var maskResolution = 1008
    @State private var seedSearchFrames = 48
    @State private var previewFrame = 0
    @State private var maskModel = "vision-segment-sam31"
    @State private var profile = "fast"
    @State private var steps = 40
    @State private var guidance = 5.0
    @State private var shift = 3.0
    @State private var sampler = "unipc"
    @State private var seed = "42"
    @State private var segmentLength = 81
    @State private var segmentOverlap = 5
    @State private var tailPolicy = "drop"
    @State private var carryDrivingAudio = true
    @State private var model = "video-scail2-14b-mlx"
    @State private var modelRoot = ""
    @State private var adapterPath = ""
    @State private var adapterStrength = 1.0
    @State private var corrections: [StudioSCAILCorrection] = []
    @State private var outputPath = StudioSpecialistFiles.timestampedDirectory(component: "SCAIL")
        .deletingLastPathComponent()
        .appendingPathComponent("scail-\(UUID().uuidString.prefix(8)).mp4")
        .path
    @State private var preflight = false
    @State private var preparedManifest: StudioSCAILManifest?
    @State private var preparedDirectory: URL?
    @State private var requestID: UUID?
    @State private var maskRequestID: UUID?
    @State private var errorMessage: String?
    @State private var statusMessage = "Add a reference and driving video, then preview masks."

    private let palette = ["blue", "red", "green", "magenta", "cyan", "yellow"]

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

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(MereRunTheme.border.opacity(0.55))
            HStack(spacing: 0) {
                setupColumn
                    .frame(width: 470)
                Divider().overlay(MereRunTheme.border.opacity(0.55))
                outputColumn
            }
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onReceive(controller.runCompletions) { result in
            guard result.templateID == .videoPrepareMasks, result.requestID == maskRequestID else {
                return
            }
            guard result.exitCode == 0 else {
                statusMessage = "Mask preparation failed. Open the Library row for diagnostics."
                return
            }
            loadPreparedManifest()
        }
        .onChange(of: maskConfigurationFingerprint) { _, _ in
            guard preparedManifest != nil else { return }
            preparedManifest = nil
            preparedDirectory = nil
            statusMessage = "Mask inputs changed. Preview or track masks again."
        }
    }

    /// The stage line: reference segmentation → driving-video tracking → review → animation.
    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "figure.run.square.stack")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
            Text(statusMessage)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1)
            Spacer()
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 10)
    }

    private var setupColumn: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Picker("Mode", selection: $mode) {
                    Text("Animate reference").tag("animation")
                    Text("Replace in scene").tag("replacement")
                }
                .pickerStyle(.segmented)

                Label(
                    mode == "animation"
                        ? "Keeps the reference background; driving supplies motion."
                        : "Keeps the driving scene; references replace masked subjects.",
                    systemImage: mode == "animation" ? "photo.on.rectangle" : "person.crop.rectangle.badge.plus"
                )
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textSecondary)

                StudioPathField(
                    label: "Driving video",
                    placeholder: "/path/to/driving.mp4",
                    path: $drivingVideo,
                    allowedContentTypes: [.movie]
                )

                subjectEditor

                DisclosureGroup("Mask preparation") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Stepper("Width \(width)", value: $width, in: 256...1536, step: 32)
                            Stepper("Height \(height)", value: $height, in: 256...1536, step: 32)
                        }
                        Stepper("FPS \(fps)", value: $fps, in: 1...60)
                        Toggle("Trim driving video", isOn: $useTrimRange)
                        if useTrimRange {
                            valueSlider("In point", value: $inSeconds, range: 0...3_600)
                            valueSlider("Out point", value: $outSeconds, range: 0.1...3_600)
                        }
                        valueSlider("SAM threshold", value: $maskThreshold, range: 0.001...0.5)
                        Stepper("SAM resolution \(maskResolution)", value: $maskResolution, in: 256...2016, step: 16)
                        Stepper("Seed search frames \(seedSearchFrames)", value: $seedSearchFrames, in: 1...240)
                        Stepper("Preview frame \(previewFrame)", value: $previewFrame, in: 0...100_000)
                        labeledField("Mask model", text: $maskModel, placeholder: "vision-segment-sam31")
                    }
                    .padding(.top, 10)
                }

                HStack {
                    Button {
                        prepareMasks(previewOnly: true)
                    } label: {
                        Label("Preview masks", systemImage: "eye")
                    }
                    .buttonStyle(.bordered)
                    Button {
                        prepareMasks(previewOnly: false)
                    } label: {
                        Label("Track full video", systemImage: "scope")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(MereRunTheme.accent)
                }

                correctionEditor

                Divider().overlay(MereRunTheme.border.opacity(0.5))

                VStack(alignment: .leading, spacing: 5) {
                    Text("Direction")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    TextField("Describe the finished shot", text: $prompt, axis: .vertical)
                        .lineLimit(2...5)
                        .mereField()
                }

                labeledField("Negative prompt", text: $negativePrompt, placeholder: "Optional")

                Picker("Profile", selection: $profile) {
                    Text("Fast · 4-step").tag("fast")
                    Text("Quality · manual").tag("quality")
                }
                .pickerStyle(.segmented)

                if profile == "quality" {
                    Stepper("Steps \(steps)", value: $steps, in: 1...100)
                    valueSlider("Guidance", value: $guidance, range: 0...15)
                    valueSlider("Schedule shift", value: $shift, range: 0...10)
                    Picker("Sampler", selection: $sampler) {
                        Text("UniPC").tag("unipc")
                        Text("Euler").tag("euler")
                    }
                }

                DisclosureGroup("Continuity, model & adapter") {
                    VStack(alignment: .leading, spacing: 10) {
                        labeledField("Seed", text: $seed, placeholder: "42")
                        Stepper("Segment length \(segmentLength)", value: $segmentLength, in: 9...401, step: 8)
                        Stepper("Overlap \(segmentOverlap)", value: $segmentOverlap, in: 1...77, step: 4)
                        Picker("Tail", selection: $tailPolicy) {
                            Text("Drop partial tail").tag("drop")
                            Text("Pad and trim").tag("pad-trim")
                        }
                        Toggle("Carry driving audio", isOn: $carryDrivingAudio)
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
                    }
                    .padding(.top, 10)
                }

                VStack(alignment: .leading, spacing: 5) {
                    Text("Output video")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    HStack(spacing: 8) {
                        TextField("/path/to/output.mp4", text: $outputPath)
                            .mereField()
                        Button("Choose…") {
                            if let url = StudioSpecialistFiles.saveFile(
                                title: "Save SCAIL video",
                                suggestedName: "scail.mp4",
                                allowedContentTypes: [.mpeg4Movie]
                            ) {
                                outputPath = url.path
                            }
                        }
                        .buttonStyle(.bordered)
                    }
                }
                Toggle("Preflight only", isOn: $preflight)

                if let errorMessage {
                    Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.red)
                }

                Button {
                    animate()
                } label: {
                    Label(preflight ? "Validate SCAIL run" : "Animate subject", systemImage: "play.rectangle.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(MereRunTheme.accent)
                .controlSize(.large)
                .disabled(preparedManifest?.drivingMaskPath == nil)
            }
            .padding(18)
        }
    }

    private var subjectEditor: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Text("Subjects")
                    .font(MereRunTheme.sectionFont)
                Spacer()
                Button {
                    guard subjects.count < 6 else { return }
                    let index = subjects.count
                    subjects.append(
                        StudioSCAILSubject(
                            name: "subject-\(index + 1)",
                            color: palette[index],
                            referencePrompt: "person",
                            drivingPrompt: "person"
                        )
                    )
                } label: {
                    Label("Add", systemImage: "plus")
                }
                .buttonStyle(.bordered)
                .disabled(subjects.count >= 6)
            }

            ForEach($subjects) { $subject in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Circle()
                            .fill(subjectColor(subject.color))
                            .frame(width: 10, height: 10)
                        TextField("Subject id", text: $subject.name)
                            .textFieldStyle(.plain)
                            .font(.system(size: 12, weight: .semibold))
                        Spacer()
                        if subjects.count > 1 {
                            Button(role: .destructive) {
                                subjects.removeAll { $0.id == subject.id }
                                normalizeSubjectColors()
                            } label: {
                                Image(systemName: "trash")
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    HStack(spacing: 7) {
                        TextField("Reference image", text: $subject.referenceImage)
                            .mereField()
                        Button("Choose…") {
                            subject.referenceImage = StudioSpecialistFiles.chooseFile(
                                title: "Reference for \(subject.name)",
                                allowedContentTypes: [.image]
                            ).first?.path ?? subject.referenceImage
                        }
                        .buttonStyle(.bordered)
                    }
                    HStack {
                        TextField("Reference selector, e.g. woman in red", text: $subject.referencePrompt)
                            .mereField()
                        TextField("Driving selector, e.g. dancer", text: $subject.drivingPrompt)
                            .mereField()
                    }
                    DisclosureGroup("Precise selectors") {
                        VStack(alignment: .leading, spacing: 7) {
                            Text("Optional box: x1,y1,x2,y2 · points: x,y; x,y")
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                            TextField("Reference box", text: $subject.referenceBox)
                                .mereField()
                            TextField("Driving box", text: $subject.drivingBox)
                                .mereField()
                            TextField("Reference positive points", text: $subject.referencePositivePoints)
                                .mereField()
                            TextField("Reference negative points", text: $subject.referenceNegativePoints)
                                .mereField()
                            TextField("Driving positive points", text: $subject.drivingPositivePoints)
                                .mereField()
                            TextField("Driving negative points", text: $subject.drivingNegativePoints)
                                .mereField()
                        }
                        .padding(.top, 7)
                    }
                }
                .padding(10)
                .background(MereRunTheme.surface)
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
            }
        }
    }

    private var correctionEditor: some View {
        DisclosureGroup("Keyframe corrections") {
            VStack(alignment: .leading, spacing: 9) {
                Text("Refine tracked masks at a frame with a box, positive/negative points, or a painted binary PNG.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                ForEach($corrections) { $correction in
                    VStack(alignment: .leading, spacing: 7) {
                        HStack {
                            Picker("Subject", selection: $correction.subjectID) {
                                ForEach(subjects) { subject in
                                    Text(subject.name).tag(subject.name)
                                }
                            }
                            Stepper(
                                "Frame \(correction.frameIndex)",
                                value: $correction.frameIndex,
                                in: 0...100_000
                            )
                            Button(role: .destructive) {
                                corrections.removeAll { $0.id == correction.id }
                            } label: {
                                Image(systemName: "trash")
                            }
                            .buttonStyle(.plain)
                        }
                        TextField("Box x1,y1,x2,y2", text: $correction.box)
                            .mereField()
                        TextField("Positive points x,y; x,y", text: $correction.positivePoints)
                            .mereField()
                        TextField("Negative points x,y; x,y", text: $correction.negativePoints)
                            .mereField()
                        HStack {
                            TextField("Painted binary correction PNG", text: $correction.paintedMaskPath)
                                .mereField()
                            Button("Choose…") {
                                correction.paintedMaskPath = StudioSpecialistFiles.chooseFile(
                                    title: "Painted binary correction",
                                    allowedContentTypes: [.png]
                                ).first?.path ?? correction.paintedMaskPath
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    .padding(9)
                    .background(MereRunTheme.surface)
                    .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
                }
                Button {
                    if let first = subjects.first {
                        corrections.append(StudioSCAILCorrection(subjectID: first.name))
                    }
                } label: {
                    Label("Add correction", systemImage: "plus")
                }
                .buttonStyle(.bordered)
            }
            .padding(.top, 9)
        }
    }

    private var outputColumn: some View {
        VStack(alignment: .leading, spacing: 13) {
            conditioningReview
                .frame(height: 210)
            Divider().overlay(MereRunTheme.border.opacity(0.5))
            Text(requestID == nil ? "Mask preparation & result" : "Animated result")
                .font(MereRunTheme.sectionFont)
            StudioSpecialistResultView(
                requestID: requestID ?? maskRequestID,
                preferredKinds: requestID == nil
                    ? [.image, .video, .text]
                    : [.video, .image, .text]
            )
        }
        .padding(18)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var conditioningReview: some View {
        HStack(spacing: 10) {
            conditioningTile(
                title: "Reference",
                url: preparedReferenceURL ?? pathURL(subjects.first?.referenceImage)
            )
            conditioningTile(
                title: preparedManifest == nil ? "Driving" : "Tracked driving",
                url: preparedOverlayURL ?? pathURL(drivingVideo)
            )
            conditioningTile(
                title: resultVideoURL == nil ? "Mask contact sheet" : "Result",
                url: resultVideoURL ?? preparedContactSheetURL
            )
        }
    }

    private func conditioningTile(title: String, url: URL?) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Group {
                if let url {
                    switch StudioOutputFileKind.classify(url) {
                    case .image:
                        StudioAsyncImagePreview(
                            url: url,
                            maxPixelSize: 900,
                            contentMode: .fit,
                            fallbackSystemImage: "photo"
                        )
                    case .video:
                        StudioVideoPlayerView(url: url)
                    default:
                        Image(systemName: "doc")
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                } else {
                    Image(systemName: "photo.badge.plus")
                        .foregroundStyle(MereRunTheme.textMuted)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .background(MereRunTheme.surface)
            .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
        }
        .frame(maxWidth: .infinity)
    }

    private var preparedReferenceURL: URL? {
        guard let manifest = preparedManifest, let first = manifest.subjects.first else { return nil }
        return resolve(first.preparedReferenceImagePath)
    }

    private var preparedOverlayURL: URL? {
        preparedManifest.map { resolve($0.overlayPreviewPath) }
    }

    private var preparedContactSheetURL: URL? {
        preparedManifest.map { resolve($0.contactSheetPath) }
    }

    private var resultVideoURL: URL? {
        guard let requestID,
              let item = library.items.first(where: { $0.id == requestID }) else {
            return nil
        }
        return item.allArtifactURLs.first {
            StudioOutputFileKind.classify($0) == .video
        }
    }

    private func pathURL(_ path: String?) -> URL? {
        guard let path, !path.isBlank else { return nil }
        return URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
    }

    private func labeledField(
        _ label: String,
        text: Binding<String>,
        placeholder: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            TextField(placeholder, text: text)
                .mereField()
        }
    }

    private func valueSlider(
        _ label: String,
        value: Binding<Double>,
        range: ClosedRange<Double>
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                Spacer()
                Text(value.wrappedValue.formatted(.number.precision(.fractionLength(2))))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .font(MereRunTheme.captionFont)
            Slider(value: value, in: range)
        }
    }

    private func prepareMasks(previewOnly: Bool) {
        errorMessage = nil
        guard validateInputs(includeRender: false) else { return }
        guard let planURL = writePlan() else { return }
        guard let template = CommandCatalog.template(id: .videoPrepareMasks) else {
            errorMessage = "The mask preparation command is unavailable."
            return
        }
        let suffix = previewOnly ? "preview-\(previewFrame)" : "tracked"
        let directory = planURL.deletingLastPathComponent()
            .appendingPathComponent(suffix, isDirectory: true)
        try? FileManager.default.removeItem(at: directory)
        preparedDirectory = directory
        if !previewOnly {
            preparedManifest = nil
        }

        var draft = template.defaultDraft()
        draft.inputPath = planURL.path
        draft.outputPath = directory.path
        draft.previewFrame = previewOnly ? String(previewFrame) : ""
        draft.model = maskModel
        requestID = nil
        maskRequestID = StudioSpecialistRunner.submit(
            templateID: .videoPrepareMasks,
            mode: .video,
            draft: draft,
            controller: controller,
            library: library
        )
        statusMessage = previewOnly ? "Preparing a review frame…" : "Tracking masks through the full video…"
    }

    private func animate() {
        errorMessage = nil
        guard validateInputs(includeRender: true),
              let manifest = preparedManifest,
              let first = manifest.subjects.first,
              let drivingMask = manifest.drivingMaskPath else {
            errorMessage = "Run full-video mask tracking before animation."
            return
        }
        guard let template = CommandCatalog.template(id: .videoAnimate) else {
            errorMessage = "The SCAIL animation command is unavailable."
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
        draft.seed = seed
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
        requestID = StudioSpecialistRunner.submit(
            templateID: .videoAnimate,
            mode: .video,
            draft: draft,
            controller: controller,
            library: library
        )
        statusMessage = preflight
            ? "Validating the complete SCAIL execution plan…"
            : profile == "fast"
            ? "SCAIL is animating with the managed 4-step adapter."
            : "SCAIL quality render started."
    }

    private func validateInputs(includeRender: Bool) -> Bool {
        guard !drivingVideo.isBlank else {
            errorMessage = "Choose a driving video."
            return false
        }
        guard !subjects.isEmpty, subjects.count <= 6 else {
            errorMessage = "SCAIL supports one to six subjects."
            return false
        }
        for subject in subjects {
            if subject.name.isBlank || subject.referenceImage.isBlank {
                errorMessage = "Every subject needs an id and reference image."
                return false
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
                errorMessage = "Every subject needs both a reference and driving selector."
                return false
            }
            let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
            if subject.name.unicodeScalars.contains(where: { !allowed.contains($0) }) {
                errorMessage = "Subject ids may contain only letters, numbers, hyphens, and underscores."
                return false
            }
        }
        if Set(subjects.map(\.name)).count != subjects.count {
            errorMessage = "Subject ids must be unique."
            return false
        }
        if corrections.contains(where: { correction in
            !subjects.contains(where: { $0.name == correction.subjectID })
        }) {
            errorMessage = "Every keyframe correction must target a current subject."
            return false
        }
        guard width.isMultiple(of: 32), height.isMultiple(of: 32) else {
            errorMessage = "SCAIL dimensions must be divisible by 32."
            return false
        }
        if useTrimRange && outSeconds <= inSeconds {
            errorMessage = "The driving-video out point must be after the in point."
            return false
        }
        if includeRender {
            guard segmentOverlap < segmentLength else {
                errorMessage = "Continuity overlap must be shorter than the segment."
                return false
            }
            guard segmentLength % 4 == 1, segmentOverlap % 4 == 1 else {
                errorMessage = "Segment length and overlap must equal 1 modulo 4."
                return false
            }
            if outputPath.isBlank {
                errorMessage = "Choose an output video path."
                return false
            }
            if !preflight,
               FileManager.default.fileExists(
                   atPath: NSString(string: outputPath).expandingTildeInPath
               ) {
                errorMessage = "Choose a new output filename so this result remains immutable in Library."
                return false
            }
        }
        return true
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
            return url
        } catch {
            errorMessage = "Could not write mask plan: \(error.localizedDescription)"
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
            preparedManifest = try JSONDecoder().decode(
                StudioSCAILManifest.self,
                from: Data(contentsOf: url)
            )
            if preparedManifest?.drivingMaskPath == nil {
                statusMessage = "Preview ready. Review the mask, then track the full video."
            } else {
                statusMessage = "Masks tracked and ready for animation."
            }
        } catch {
            errorMessage = "Mask run completed, but manifest could not be read: \(error.localizedDescription)"
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

    private func normalizeSubjectColors() {
        for index in subjects.indices {
            subjects[index].color = palette[index]
        }
    }

    private func subjectColor(_ name: String) -> Color {
        switch name {
        case "blue": .blue
        case "red": .red
        case "green": .green
        case "magenta": .purple
        case "cyan": .cyan
        case "yellow": .yellow
        default: MereRunTheme.textMuted
        }
    }
}
