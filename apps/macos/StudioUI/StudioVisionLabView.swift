import AppKit
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// The Vision Lab's tasks. The shell's task control groups them (Faces covers detect, embed,
/// compare, and batch; Geometry covers single and multi-view); the rail picks the variant.
enum StudioVisionTask: String, CaseIterable, Identifiable {
    case faceDetect = "Face detection"
    case faceEmbed = "Face embedding"
    case faceCompare = "Face comparison"
    case faceBatch = "Face batch"
    case pose = "Pose landmarks"
    case flow = "Optical flow"
    case depthVideo = "Video depth"
    case geometry = "Metric geometry"
    case geometryMultiview = "Multi-view geometry"
    case liveTrack = "Live tracking"

    var id: String { rawValue }

    var templateID: CommandTemplateID {
        switch self {
        case .faceDetect: .visionFaceDetect
        case .faceEmbed: .visionFaceEmbed
        case .faceCompare: .visionFaceCompare
        case .faceBatch: .visionFaceBatch
        case .pose: .visionPose
        case .flow: .visionFlow
        case .depthVideo: .visionDepthVideo
        case .geometry: .visionGeometry
        case .geometryMultiview: .visionGeometryMultiview
        case .liveTrack: .visionTrackLive
        }
    }

    var icon: String {
        switch self {
        case .faceDetect: "face.dashed"
        case .faceEmbed: "person.crop.square"
        case .faceCompare: "person.2"
        case .faceBatch: "person.3.sequence"
        case .pose: "figure.stand"
        case .flow: "arrow.triangle.2.circlepath"
        case .depthVideo: "square.3.layers.3d"
        case .geometry: "view.3d"
        case .geometryMultiview: "camera.metering.multispot"
        case .liveTrack: "video.badge.waveform"
        }
    }

    var subtitle: String {
        switch self {
        case .faceDetect: "Boxes, landmarks, and optional identity vectors"
        case .faceEmbed: "One normalized ArcFace identity vector"
        case .faceCompare: "Cosine similarity across two selected faces"
        case .faceBatch: "Warm-session JSONL analysis of many images"
        case .pose: "Native body, hand, and face landmarks"
        case .flow: "Dense per-pixel motion between equal-size frames"
        case .depthVideo: "Temporally consistent depth frames and review video"
        case .geometry: "Metric depth, normals, cameras, and point cloud"
        case .geometryMultiview: "Joint cameras, confidence, and point cloud"
        case .liveTrack: "Capture and annotate a camera stream"
        }
    }

    var needsPrimaryImage: Bool {
        [.faceDetect, .faceEmbed, .faceCompare, .faceBatch, .pose, .flow, .geometry, .geometryMultiview]
            .contains(self)
    }

    /// The toolbar task this variant belongs to.
    var studioTask: StudioTask {
        switch self {
        case .faceDetect, .faceEmbed, .faceCompare, .faceBatch: .visionFaces
        case .pose: .visionPose
        case .flow: .visionFlow
        case .depthVideo: .visionDepth
        case .geometry, .geometryMultiview: .visionGeometry
        case .liveTrack: .visionLive
        }
    }
}

extension StudioTask {
    /// The Vision Lab variant a toolbar task opens by default, or nil for non-lab tasks.
    var visionLabTask: StudioVisionTask? {
        switch self {
        case .visionFaces: .faceDetect
        case .visionPose: .pose
        case .visionFlow: .flow
        case .visionDepth: .depthVideo
        case .visionGeometry: .geometry
        case .visionLive: .liveTrack
        default: nil
        }
    }
}

struct StudioVisionLabView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore

    /// Owned by the host so the rail and the shell's task control stay in step.
    @Binding var task: StudioVisionTask
    @State private var primaryInput = ""
    @State private var secondaryInput = ""
    @State private var additionalInputs: [String] = []
    @State private var inputListPath = ""
    @State private var outputDirectory = StudioSpecialistFiles
        .timestampedDirectory(component: "Vision")
        .path
    @State private var model = ""
    @State private var faceThreshold = 0.65
    @State private var provider = "auto"
    @State private var maxFaces = 0
    @State private var includeEmbeddings = false
    @State private var faceIndex = 0
    @State private var referenceFaceIndex = 0
    @State private var candidateFaceIndex = 0
    @State private var failFast = false
    @State private var poseBody = true
    @State private var poseHands = true
    @State private var poseFace = true
    @State private var maxHands = 2
    @State private var minimumConfidence = 0.1
    @State private var flowAccuracy = "high"
    @State private var inputSize = 518
    @State private var maxFrames = 240
    @State private var resolutionLevel = 9
    @State private var tokenCount = 0
    @State private var maxPoints = 0
    @State private var camerasPath = ""
    @State private var processResolution = 504
    @State private var referenceView = "saddle-balanced"
    @State private var confidencePercentile = 40.0
    @State private var prompts = "a person"
    @State private var camera = 0
    @State private var duration = 10.0
    @State private var initFrame = 0
    @State private var seedSearchFrames = 30
    @State private var trackingThreshold = 0.05
    @State private var trackingResolution = 1008
    @State private var showBoxes = true
    @State private var showLabels = true
    @State private var dryRun = false
    @State private var requestID: UUID?
    @State private var errorMessage: String?

    private var currentItem: StudioLibraryItem? {
        guard let requestID else { return nil }
        return library.items.first { $0.id == requestID }
    }

    var body: some View {
        HStack(spacing: 0) {
            taskRail
                .frame(width: 210)
            Divider().overlay(MereRunTheme.border.opacity(0.55))
            configuration
                .frame(minWidth: 300, idealWidth: 390, maxWidth: 390)
            Divider().overlay(MereRunTheme.border.opacity(0.55))
            resultPane
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onChange(of: task) { _, newTask in
            model = CommandCatalog.template(id: newTask.templateID)?.defaultDraft().model ?? ""
            outputDirectory = StudioSpecialistFiles.timestampedDirectory(component: "Vision").path
            requestID = nil
            errorMessage = nil
        }
        .onAppear {
            model = CommandCatalog.template(id: task.templateID)?.defaultDraft().model ?? ""
        }
    }

    private var taskRail: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 5) {
                ForEach(StudioVisionTask.allCases) { candidate in
                    Button {
                        task = candidate
                    } label: {
                        HStack(spacing: 9) {
                            Image(systemName: candidate.icon)
                                .frame(width: 20)
                            Text(candidate.rawValue)
                                .lineLimit(1)
                            Spacer()
                        }
                        .font(.system(size: 12.5, weight: task == candidate ? .semibold : .regular))
                        .foregroundStyle(task == candidate ? MereRunTheme.accent : MereRunTheme.textSecondary)
                        .padding(.horizontal, 10)
                        .frame(height: 36)
                        .background {
                            RoundedRectangle(cornerRadius: 8)
                                .fill(task == candidate ? MereRunTheme.accentSoft : .clear)
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(12)
        }
    }

    private var configuration: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 15) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(task.rawValue)
                        .font(MereRunTheme.sectionFont)
                    Text(task.subtitle)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }

                inputControls

                if task != .pose && task != .flow {
                    labeledField("Model override", text: $model, placeholder: "Managed default")
                }

                taskControls

                if task == .depthVideo || task == .geometry || task == .geometryMultiview {
                    Toggle("Preflight only", isOn: $dryRun)
                }

                if let errorMessage {
                    Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.red)
                }

                Button {
                    run()
                } label: {
                    Label(dryRun ? "Run preflight" : "Run \(task.rawValue)", systemImage: task.icon)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(MereRunTheme.accent)
                .controlSize(.large)
            }
            .padding(16)
        }
    }

    @ViewBuilder
    private var inputControls: some View {
        if task == .liveTrack {
            VStack(alignment: .leading, spacing: 5) {
                Text("Tracked prompts · one per line")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                TextField("a person", text: $prompts, axis: .vertical)
                    .lineLimit(2...5)
                    .mereField()
            }
        } else {
            StudioPathField(
                label: task == .depthVideo ? "Input video" : "Primary image",
                placeholder: task == .depthVideo ? "/path/to/video.mp4" : "/path/to/image.png",
                path: $primaryInput,
                allowedContentTypes: task == .depthVideo ? [.movie] : [.image]
            )

            if task == .faceCompare || task == .flow {
                StudioPathField(
                    label: task == .flow ? "Target image" : "Candidate image",
                    placeholder: "/path/to/second.png",
                    path: $secondaryInput,
                    allowedContentTypes: [.image]
                )
            }

            if task == .faceBatch || task == .geometryMultiview {
                multipleInputEditor
            }

            if task == .faceBatch {
                StudioPathField(
                    label: "Input list (optional)",
                    placeholder: "/path/to/images.txt",
                    path: $inputListPath,
                    allowedContentTypes: [.plainText]
                )
            }
        }

        StudioPathField(
            label: "Output directory",
            placeholder: "/path/to/output",
            path: $outputDirectory,
            picksDirectory: true
        )
    }

    private var multipleInputEditor: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text(task == .geometryMultiview ? "Additional ordered views" : "Additional images")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                Spacer()
                Text("\(additionalInputs.count + (primaryInput.isBlank ? 0 : 1)) total")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            ForEach(Array(additionalInputs.enumerated()), id: \.offset) { index, path in
                HStack {
                    Text(path)
                        .font(MereRunTheme.captionFont)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    if task == .geometryMultiview, index > 0 {
                        Button { additionalInputs.swapAt(index, index - 1) } label: {
                            Image(systemName: "arrow.up")
                        }
                        .buttonStyle(.plain)
                    }
                    Button(role: .destructive) { additionalInputs.remove(at: index) } label: {
                        Image(systemName: "xmark")
                    }
                    .buttonStyle(.plain)
                }
            }
            Button {
                additionalInputs.append(contentsOf: StudioSpecialistFiles.chooseFile(
                    title: "Add images",
                    allowedContentTypes: [.image],
                    allowsMultipleSelection: true
                ).map(\.path))
            } label: {
                Label("Add images…", systemImage: "photo.stack")
            }
            .buttonStyle(.bordered)
        }
    }

    @ViewBuilder
    private var taskControls: some View {
        switch task {
        case .faceDetect, .faceEmbed, .faceCompare, .faceBatch:
            faceControls
        case .pose:
            VStack(alignment: .leading, spacing: 10) {
                Toggle("Body landmarks", isOn: $poseBody)
                Toggle("Hand landmarks", isOn: $poseHands)
                Toggle("Face landmarks", isOn: $poseFace)
                Stepper("Maximum hands \(maxHands)", value: $maxHands, in: 1...8)
                valueSlider("Minimum confidence", value: $minimumConfidence, range: 0...1)
            }
        case .flow:
            Picker("Accuracy", selection: $flowAccuracy) {
                Text("Low").tag("low")
                Text("Medium").tag("medium")
                Text("High").tag("high")
                Text("Very high").tag("very-high")
            }
        case .depthVideo:
            VStack(alignment: .leading, spacing: 10) {
                Stepper("Input edge \(inputSize)", value: $inputSize, in: 128...1_536, step: 14)
                Stepper("Maximum frames \(maxFrames)", value: $maxFrames, in: 1...100_000)
            }
        case .geometry:
            VStack(alignment: .leading, spacing: 10) {
                Stepper("Resolution level \(resolutionLevel)", value: $resolutionLevel, in: 0...9)
                Stepper("Token override \(tokenCount)", value: $tokenCount, in: 0...1_000_000, step: 1_000)
                Stepper("Point ceiling \(maxPoints)", value: $maxPoints, in: 0...10_000_000, step: 10_000)
            }
        case .geometryMultiview:
            VStack(alignment: .leading, spacing: 10) {
                StudioPathField(
                    label: "Camera JSON (optional)",
                    placeholder: "/path/to/cameras.json",
                    path: $camerasPath,
                    allowedContentTypes: [.json]
                )
                Stepper("Process resolution \(processResolution)", value: $processResolution, in: 128...2_048, step: 14)
                Picker("Reference view", selection: $referenceView) {
                    Text("First").tag("first")
                    Text("Middle").tag("middle")
                    Text("Saddle balanced").tag("saddle-balanced")
                    Text("Similarity range").tag("saddle-similarity-range")
                }
                valueSlider("Confidence percentile", value: $confidencePercentile, range: 0...100)
                Stepper("Point ceiling \(maxPoints)", value: $maxPoints, in: 0...10_000_000, step: 10_000)
            }
        case .liveTrack:
            VStack(alignment: .leading, spacing: 10) {
                Stepper("Camera \(camera)", value: $camera, in: 0...16)
                valueSlider("Duration", value: $duration, range: 1...3_600, suffix: "s")
                Stepper("Initial frame \(initFrame)", value: $initFrame, in: 0...10_000)
                Stepper("Seed search \(seedSearchFrames)", value: $seedSearchFrames, in: 1...240)
                valueSlider("Threshold", value: $trackingThreshold, range: 0.001...0.5)
                Stepper("Resolution \(trackingResolution)", value: $trackingResolution, in: 256...2_016, step: 16)
                Toggle("Draw boxes", isOn: $showBoxes)
                Toggle("Draw labels", isOn: $showLabels)
            }
        }
    }

    private var faceControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            valueSlider("Detection threshold", value: $faceThreshold, range: 0...1)
            Picker("Execution provider", selection: $provider) {
                Text("Automatic").tag("auto")
                Text("Core ML").tag("coreml")
                Text("CPU").tag("cpu")
            }
            if task == .faceDetect || task == .faceBatch {
                Stepper("Maximum faces \(maxFaces == 0 ? "all" : String(maxFaces))", value: $maxFaces, in: 0...100)
                Toggle("Include embeddings", isOn: $includeEmbeddings)
            }
            if task == .faceEmbed {
                Stepper("Face index \(faceIndex)", value: $faceIndex, in: 0...100)
            }
            if task == .faceCompare {
                Stepper("Reference face \(referenceFaceIndex)", value: $referenceFaceIndex, in: 0...100)
                Stepper("Candidate face \(candidateFaceIndex)", value: $candidateFaceIndex, in: 0...100)
            }
            if task == .faceBatch {
                Toggle("Stop on first failure", isOn: $failFast)
            }
        }
    }

    private var resultPane: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Visual result")
                        .font(MereRunTheme.sectionFont)
                    Text(resultCaption)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
                if let item = currentItem {
                    Text(item.status.rawValue.capitalized)
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(item.status == .failed ? MereRunTheme.red : MereRunTheme.textSecondary)
                }
            }

            if let visualization = bespokeVisualization {
                visualization
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(MereRunTheme.surface.opacity(0.5))
                    .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg))
            } else {
                StudioSpecialistResultView(
                    requestID: requestID,
                    preferredKinds: resultPreferredKinds
                )
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var resultCaption: String {
        switch task {
        case .faceDetect: "Boxes and five-point landmarks render over the source."
        case .pose: "Body, hand, and face points render in native image coordinates."
        case .flow: "Direction-colored vectors visualize the Middlebury flow field."
        case .depthVideo: "Review the depth video and per-frame EXR/PNG artifacts."
        case .geometry, .geometryMultiview: "Orbit the GLB/PLY point cloud and inspect depth/normal maps."
        case .liveTrack: "The annotated camera recording appears as soon as it is written."
        default: "Structured results and sidecars remain available in the Library."
        }
    }

    private var resultPreferredKinds: [StudioOutputFileKind] {
        switch task {
        case .geometry, .geometryMultiview: [.model3D, .image, .text]
        case .depthVideo, .liveTrack: [.video, .image, .text]
        default: [.image, .text, .video, .model3D]
        }
    }

    private var bespokeVisualization: AnyView? {
        guard let item = currentItem, item.status == .completed else { return nil }
        switch task {
        case .faceDetect:
            guard let json = artifact(in: item, extension: "json") else { return nil }
            return AnyView(
                StudioVisionOverlayPreview(
                    imageURL: URL(fileURLWithPath: primaryInput),
                    jsonURL: json,
                    kind: .faces
                )
            )
        case .pose:
            guard let json = artifact(in: item, extension: "json") else { return nil }
            return AnyView(
                StudioVisionOverlayPreview(
                    imageURL: URL(fileURLWithPath: primaryInput),
                    jsonURL: json,
                    kind: .pose
                )
            )
        case .flow:
            guard let flow = artifact(in: item, extension: "flo") else { return nil }
            return AnyView(StudioOpticalFlowPreview(url: flow))
        default:
            return nil
        }
    }

    private func artifact(in item: StudioLibraryItem, extension pathExtension: String) -> URL? {
        item.allArtifactURLs.first { $0.pathExtension.lowercased() == pathExtension }
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
        range: ClosedRange<Double>,
        suffix: String = ""
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                Spacer()
                Text(value.wrappedValue.formatted(.number.precision(.fractionLength(2))) + suffix)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .font(MereRunTheme.captionFont)
            Slider(value: value, in: range)
        }
    }

    private func run() {
        errorMessage = nil
        guard validate() else { return }
        guard let template = CommandCatalog.template(id: task.templateID) else {
            errorMessage = "The selected vision command is unavailable."
            return
        }

        let root = URL(fileURLWithPath: NSString(string: outputDirectory).expandingTildeInPath)
        var draft = template.defaultDraft()
        draft.inputPath = primaryInput
        draft.visionSecondInputPath = secondaryInput
        draft.visionAdditionalInputs = additionalInputs.joined(separator: "\n")
        draft.model = model
        draft.visionFaceScoreThreshold = faceThreshold
        draft.visionExecutionProvider = provider
        draft.visionMaxFaces = maxFaces
        draft.visionIncludeEmbeddings = includeEmbeddings
        draft.visionFaceIndex = String(faceIndex)
        draft.visionReferenceFaceIndex = String(referenceFaceIndex)
        draft.visionCandidateFaceIndex = String(candidateFaceIndex)
        draft.visionInputList = inputListPath
        draft.visionFailFast = failFast
        draft.visionPoseBody = poseBody
        draft.visionPoseHands = poseHands
        draft.visionPoseFace = poseFace
        draft.visionMaxHands = maxHands
        draft.visionMinimumConfidence = minimumConfidence
        draft.visionFlowAccuracy = flowAccuracy
        draft.visionInputSize = inputSize
        draft.visionMaxFrames = maxFrames
        draft.visionResolutionLevel = resolutionLevel
        draft.visionTokenCount = tokenCount
        draft.visionMaxPoints = maxPoints
        draft.camerasPath = camerasPath
        draft.visionProcessResolution = processResolution
        draft.visionReferenceView = referenceView
        draft.visionConfidencePercentile = confidencePercentile
        draft.prompt = prompts
        draft.visionCamera = camera
        draft.durationSeconds = duration
        draft.visionInitFrame = initFrame
        draft.visionSeedSearchFrames = seedSearchFrames
        draft.visionThreshold = trackingThreshold
        draft.visionResolution = trackingResolution
        draft.force = showBoxes
        draft.visionShowLabels = showLabels
        draft.dryRun = dryRun
        draft.json = true

        switch task {
        case .faceDetect, .faceEmbed, .faceCompare, .pose:
            draft.visionJSONOutputPath = root.appendingPathComponent("result.json").path
        case .faceBatch:
            draft.visionJSONLOutput = root.appendingPathComponent("faces.jsonl").path
        case .flow:
            draft.outputPath = root.appendingPathComponent("motion.flo").path
            draft.visionJSONOutputPath = root.appendingPathComponent("motion.json").path
        case .depthVideo, .geometry, .geometryMultiview:
            draft.outputPath = root.path
        case .liveTrack:
            draft.outputPath = root.appendingPathComponent("live-tracking.mp4").path
            draft.visionJSONOutputPath = root.appendingPathComponent("live-tracking.json").path
        }

        requestID = StudioSpecialistRunner.submit(
            templateID: task.templateID,
            mode: task == .liveTrack ? .track : .readImage,
            draft: draft,
            controller: controller,
            library: library
        )
    }

    private func validate() -> Bool {
        guard !outputDirectory.isBlank else {
            errorMessage = "Choose an output directory."
            return false
        }
        let outputURL = URL(fileURLWithPath: NSString(string: outputDirectory).expandingTildeInPath)
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: outputURL.path),
           !contents.isEmpty {
            errorMessage = "Choose a new or empty output directory so this result remains immutable in Library."
            return false
        }
        if task.needsPrimaryImage && primaryInput.isBlank {
            errorMessage = "Choose the primary image."
            return false
        }
        if task == .depthVideo && primaryInput.isBlank {
            errorMessage = "Choose an input video."
            return false
        }
        if (task == .faceCompare || task == .flow) && secondaryInput.isBlank {
            errorMessage = "Choose the second image."
            return false
        }
        if task == .faceBatch && primaryInput.isBlank && additionalInputs.isEmpty && inputListPath.isBlank {
            errorMessage = "Add images or an input-list file."
            return false
        }
        if task == .geometryMultiview && additionalInputs.isEmpty {
            errorMessage = "Add at least one additional ordered view."
            return false
        }
        if task == .liveTrack && prompts.isBlank {
            errorMessage = "Enter at least one tracked prompt."
            return false
        }
        return true
    }
}

private enum StudioVisionOverlayKind {
    case faces
    case pose
}

private struct StudioFaceOverlayResult: Decodable {
    struct Record: Decodable {
        struct Detection: Decodable {
            struct Box: Decodable {
                let x: Double
                let y: Double
                let width: Double
                let height: Double
            }
            struct Point: Decodable {
                let x: Double
                let y: Double
            }
            let score: Double
            let boundingBox: Box
            let landmarks: [Point]

            enum CodingKeys: String, CodingKey {
                case score
                case boundingBox = "boundingBox"
                case landmarks
            }
        }
        let index: Int
        let detection: Detection
    }
    let width: Int
    let height: Int
    let faces: [Record]
}

private struct StudioPoseOverlayResult: Decodable {
    struct Subject: Decodable {
        struct Point: Decodable {
            let name: String
            let x: Double
            let y: Double
            let confidence: Double
        }
        let kind: String
        let index: Int
        let points: [Point]
    }
    let imageWidth: Int
    let imageHeight: Int
    let coordinateSpace: String
    let subjects: [Subject]
}

private struct StudioVisionOverlayPreview: View {
    let imageURL: URL
    let jsonURL: URL
    let kind: StudioVisionOverlayKind

    @State private var image: NSImage?
    @State private var faces: StudioFaceOverlayResult?
    @State private var pose: StudioPoseOverlayResult?
    @State private var error: String?

    var body: some View {
        GeometryReader { geometry in
            if let image {
                ZStack {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                    Canvas { context, size in
                        let rect = aspectFitRect(imageSize: image.size, in: size)
                        if let faces {
                            drawFaces(faces, in: rect, context: &context)
                        }
                        if let pose {
                            drawPose(pose, in: rect, context: &context)
                        }
                    }
                }
            } else {
                ContentUnavailableView(
                    error == nil ? "Loading overlay" : "Overlay unavailable",
                    systemImage: error == nil ? "hourglass" : "exclamationmark.triangle",
                    description: Text(error ?? "")
                )
            }
        }
        .task(id: jsonURL) { load() }
    }

    private func load() {
        image = NSImage(contentsOf: imageURL)
        do {
            let data = try Data(contentsOf: jsonURL)
            switch kind {
            case .faces:
                faces = try JSONDecoder().decode(StudioFaceOverlayResult.self, from: data)
            case .pose:
                pose = try JSONDecoder().decode(StudioPoseOverlayResult.self, from: data)
            }
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func aspectFitRect(imageSize: CGSize, in size: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return .zero }
        let scale = min(size.width / imageSize.width, size.height / imageSize.height)
        let fitted = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        return CGRect(
            x: (size.width - fitted.width) / 2,
            y: (size.height - fitted.height) / 2,
            width: fitted.width,
            height: fitted.height
        )
    }

    private func drawFaces(
        _ result: StudioFaceOverlayResult,
        in rect: CGRect,
        context: inout GraphicsContext
    ) {
        let scaleX = rect.width / CGFloat(max(1, result.width))
        let scaleY = rect.height / CGFloat(max(1, result.height))
        for face in result.faces {
            let box = face.detection.boundingBox
            let frame = CGRect(
                x: rect.minX + CGFloat(box.x) * scaleX,
                y: rect.minY + CGFloat(box.y) * scaleY,
                width: CGFloat(box.width) * scaleX,
                height: CGFloat(box.height) * scaleY
            )
            context.stroke(Path(frame), with: .color(.green), lineWidth: 2)
            for point in face.detection.landmarks {
                let center = CGPoint(
                    x: rect.minX + CGFloat(point.x) * scaleX,
                    y: rect.minY + CGFloat(point.y) * scaleY
                )
                context.fill(
                    Path(ellipseIn: CGRect(x: center.x - 3, y: center.y - 3, width: 6, height: 6)),
                    with: .color(.yellow)
                )
            }
            context.draw(
                Text("#\(face.index) \(Int(face.detection.score * 100))%")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.green),
                at: CGPoint(x: frame.minX, y: max(rect.minY + 8, frame.minY - 8)),
                anchor: .leading
            )
        }
    }

    private func drawPose(
        _ result: StudioPoseOverlayResult,
        in rect: CGRect,
        context: inout GraphicsContext
    ) {
        for subject in result.subjects {
            let color: Color = switch subject.kind {
            case "body": .cyan
            case "hand": .yellow
            default: .pink
            }
            for point in subject.points {
                let x = rect.minX + CGFloat(point.x) * rect.width
                let normalizedY = result.coordinateSpace == "normalized-bottom-left"
                    ? 1 - point.y
                    : point.y
                let y = rect.minY + CGFloat(normalizedY) * rect.height
                context.fill(
                    Path(ellipseIn: CGRect(x: x - 2.5, y: y - 2.5, width: 5, height: 5)),
                    with: .color(color.opacity(max(0.25, point.confidence)))
                )
            }
        }
    }
}

private struct StudioFlowField {
    let width: Int
    let height: Int
    let vectors: [(Float, Float)]

    static func load(url: URL) throws -> StudioFlowField {
        let data = try Data(contentsOf: url)
        guard data.count >= 12 else { throw CocoaError(.fileReadCorruptFile) }
        func uint32(_ offset: Int) -> UInt32 {
            data.withUnsafeBytes { raw in
                UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
            }
        }
        let magic = Float(bitPattern: uint32(0))
        let width = Int(Int32(bitPattern: uint32(4)))
        let height = Int(Int32(bitPattern: uint32(8)))
        guard abs(magic - 202_021.25) < 0.01, width > 0, height > 0,
              data.count >= 12 + width * height * 8 else {
            throw CocoaError(.fileReadCorruptFile)
        }
        var vectors: [(Float, Float)] = []
        vectors.reserveCapacity(width * height)
        var offset = 12
        for _ in 0..<(width * height) {
            vectors.append((
                Float(bitPattern: uint32(offset)),
                Float(bitPattern: uint32(offset + 4))
            ))
            offset += 8
        }
        return StudioFlowField(width: width, height: height, vectors: vectors)
    }
}

private struct StudioOpticalFlowPreview: View {
    let url: URL
    @State private var field: StudioFlowField?
    @State private var error: String?

    var body: some View {
        GeometryReader { geometry in
            if let field {
                Canvas { context, size in
                    context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.black))
                    let step = max(1, min(field.width, field.height) / 28)
                    let sx = size.width / CGFloat(field.width)
                    let sy = size.height / CGFloat(field.height)
                    for y in stride(from: 0, to: field.height, by: step) {
                        for x in stride(from: 0, to: field.width, by: step) {
                            let vector = field.vectors[y * field.width + x]
                            let magnitude = hypot(Double(vector.0), Double(vector.1))
                            guard magnitude.isFinite, magnitude > 0.01 else { continue }
                            let angle = atan2(Double(vector.1), Double(vector.0))
                            let length = min(CGFloat(step) * 0.8, CGFloat(log1p(magnitude)) * 3 + 2)
                            let start = CGPoint(x: (CGFloat(x) + 0.5) * sx, y: (CGFloat(y) + 0.5) * sy)
                            let end = CGPoint(
                                x: start.x + cos(angle) * length,
                                y: start.y + sin(angle) * length
                            )
                            var path = Path()
                            path.move(to: start)
                            path.addLine(to: end)
                            let hue = (angle + .pi) / (2 * .pi)
                            context.stroke(
                                path,
                                with: .color(Color(hue: hue, saturation: 0.9, brightness: 1)),
                                lineWidth: 1.2
                            )
                        }
                    }
                }
                .overlay(alignment: .topLeading) {
                    Text("\(field.width)×\(field.height) dense flow")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(.white)
                        .padding(8)
                        .background(.black.opacity(0.55), in: Capsule())
                        .padding(10)
                }
            } else {
                ContentUnavailableView(
                    error == nil ? "Loading flow field" : "Flow preview unavailable",
                    systemImage: error == nil ? "hourglass" : "exclamationmark.triangle",
                    description: Text(error ?? "")
                )
            }
        }
        .task(id: url) {
            do {
                field = try StudioFlowField.load(url: url)
            } catch {
                self.error = error.localizedDescription
            }
        }
    }
}
