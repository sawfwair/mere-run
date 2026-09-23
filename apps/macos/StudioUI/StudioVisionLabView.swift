import AVFoundation
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
    case depth = "Image depth"
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
        case .depth: .visionDepth
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
        case .depth: "square.3.layers.3d.down.right"
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
        case .depth: "Relative depth for one still image, with a preview to review"
        case .depthVideo: "Temporally consistent depth frames and review video"
        case .geometry: "Metric depth, normals, cameras, and point cloud"
        case .geometryMultiview: "Joint cameras, confidence, and point cloud"
        case .liveTrack: "Capture and annotate a camera stream"
        }
    }

    var needsPrimaryImage: Bool {
        [.faceDetect, .faceEmbed, .faceCompare, .faceBatch, .pose, .flow, .depth, .geometry, .geometryMultiview]
            .contains(self)
    }

    /// The toolbar task this variant belongs to.
    var studioTask: StudioTask {
        switch self {
        case .faceDetect, .faceEmbed, .faceCompare, .faceBatch: .visionFaces
        case .pose: .visionPose
        case .flow: .visionFlow
        case .depth, .depthVideo: .visionDepth
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
        case .visionDepth: .depth
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
    @StudioStoredValue("VisionLab.primaryInput") private var primaryInput = ""
    @StudioStoredValue("VisionLab.secondaryInput") private var secondaryInput = ""
    @StudioStoredValue("VisionLab.additionalInputs") private var additionalInputs: [String] = []
    @StudioStoredValue("VisionLab.inputListPath") private var inputListPath = ""
    @State private var outputDirectory = StudioSpecialistFiles
        .outputDirectory(domain: .vision, name: "vision")
        .path
    @StudioStoredValue("VisionLab.model") private var model = ""
    @StudioStoredValue("VisionLab.faceThreshold") private var faceThreshold = 0.65
    @StudioStoredValue("VisionLab.provider") private var provider = "auto"
    @StudioStoredValue("VisionLab.maxFaces") private var maxFaces = 0
    @StudioStoredValue("VisionLab.includeEmbeddings") private var includeEmbeddings = false
    @StudioStoredValue("VisionLab.faceIndex") private var faceIndex = 0
    @StudioStoredValue("VisionLab.referenceFaceIndex") private var referenceFaceIndex = 0
    @StudioStoredValue("VisionLab.candidateFaceIndex") private var candidateFaceIndex = 0
    @StudioStoredValue("VisionLab.failFast") private var failFast = false
    @StudioStoredValue("VisionLab.poseBody") private var poseBody = true
    @StudioStoredValue("VisionLab.poseHands") private var poseHands = true
    @StudioStoredValue("VisionLab.poseFace") private var poseFace = true
    @StudioStoredValue("VisionLab.maxHands") private var maxHands = 2
    @StudioStoredValue("VisionLab.minimumConfidence") private var minimumConfidence = 0.1
    @StudioStoredValue("VisionLab.flowAccuracy") private var flowAccuracy = "high"
    @StudioStoredValue("VisionLab.inputSize") private var inputSize = 518
    @StudioStoredValue("VisionLab.maxFrames") private var maxFrames = 240
    @StudioStoredValue("VisionLab.depthMaxEdge") private var depthMaxEdge = 1_024
    @StudioStoredValue("VisionLab.depthNative") private var depthNative = false
    @StudioStoredValue("VisionLab.depthCheckpoint") private var depthCheckpoint = ""
    @StudioStoredValue("VisionLab.resolutionLevel") private var resolutionLevel = 9
    @StudioStoredValue("VisionLab.tokenCount") private var tokenCount = 0
    @StudioStoredValue("VisionLab.maxPoints") private var maxPoints = 0
    /// A camera file chosen before the page edited cameras itself; read into `geometryCameras` once.
    @StudioStoredValue("VisionLab.camerasPath") private var legacyCamerasPath = ""
    @StudioStoredValue("VisionLab.suppliesCameras") private var suppliesCameras = false
    @StudioStoredValue("VisionLab.geometryCameras") private var geometryCameras = StudioGeometryCameraDocument()
    /// The saved copy of a valid camera document, for the Command view; empty when cameras are off
    /// or do not match the views. Each run writes its own copy beside its output.
    @State private var draftCamerasPath = ""
    /// Each multi-view image's decoded size by path, read when the list changes.
    @State private var viewSizes: [String: StudioPixelSize] = [:]
    @StudioStoredValue("VisionLab.processResolution") private var processResolution = 504
    @StudioStoredValue("VisionLab.referenceView") private var referenceView = "saddle-balanced"
    @StudioStoredValue("VisionLab.confidencePercentile") private var confidencePercentile = 40.0
    @StudioStoredValue("VisionLab.prompts") private var prompts = "a person"
    @StudioStoredValue("VisionLab.camera") private var camera = 0
    @StudioStoredValue("VisionLab.duration") private var duration = 10.0
    @StudioStoredValue("VisionLab.initFrame") private var initFrame = 0
    @StudioStoredValue("VisionLab.seedSearchFrames") private var seedSearchFrames = 30
    @StudioStoredValue("VisionLab.trackingThreshold") private var trackingThreshold = 0.05
    @StudioStoredValue("VisionLab.trackingResolution") private var trackingResolution = 1008
    @StudioStoredValue("VisionLab.showBoxes") private var showBoxes = true
    @StudioStoredValue("VisionLab.showLabels") private var showLabels = true
    @StudioStoredValue("VisionLab.dryRun") private var dryRun = false
    @StudioStoredValue("requestID") private var requestID: UUID? = nil
    @State private var errorMessage: String?
    /// The cameras on this Mac, in the order the CLI numbers them.
    @State private var cameras: [StudioCamera] = []

    private var currentItem: StudioLibraryItem? {
        guard let requestID else { return nil }
        return library.items.first { $0.id == requestID }
    }

    var body: some View {
        StudioAnalysisLayout { configuration } result: { resultPane }
        .studioTaskCommand(task.templateID, draft: commandDraft)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onChange(of: task) { _, newTask in
            outputDirectory = StudioSpecialistFiles.outputDirectory(domain: .vision, name: "vision").path
            errorMessage = nil
        }
        .onAppear {
            if model.isBlank { model = CommandCatalog.template(id: task.templateID)?.defaultDraft().model ?? "" }
            adoptLegacyCameras()
        }
        .task(id: multiviewPaths) { refreshViewSizes() }
        .task(id: cameraDraftKey) { await saveDraftCameras() }
        .task(id: task) {
            if task == .liveTrack { refreshCameras() }
        }
        .onReceive(NotificationCenter.default.publisher(for: AVCaptureDevice.wasConnectedNotification)) { _ in
            if task == .liveTrack { refreshCameras() }
        }
        .onReceive(NotificationCenter.default.publisher(for: AVCaptureDevice.wasDisconnectedNotification)) { _ in
            if task == .liveTrack { refreshCameras() }
        }
    }

    /// Lists the cameras now attached; an index remembered for a camera that is gone falls back
    /// to the first one, so the picker never shows an empty selection.
    private func refreshCameras() {
        cameras = StudioCamera.connected()
        if !cameras.isEmpty, !cameras.contains(where: { $0.index == camera }) { camera = 0 }
    }

    private var configuration: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 15) {
                let variants = StudioVisionTask.allCases.filter { $0.studioTask == task.studioTask }
                if variants.count > 1 {
                    Picker("Operation", selection: $task) {
                        ForEach(variants) { Text($0.rawValue).tag($0) }
                    }
                }
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

                if task == .depth || task == .depthVideo || task == .geometry || task == .geometryMultiview {
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
                .buttonStyle(.merePrimary)
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
            .buttonStyle(.mereSecondary)
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
        case .depth:
            VStack(alignment: .leading, spacing: 10) {
                Toggle("Run at the source resolution", isOn: $depthNative)
                // `--max-edge` is rounded to a multiple of 16 and cannot go with `--native`.
                Stepper("Longest edge \(depthMaxEdge)", value: $depthMaxEdge, in: 256...4_096, step: 16)
                    .disabled(depthNative)
                Picker("Checkpoint", selection: $depthCheckpoint) {
                    Text("Default").tag("")
                    ForEach(Self.depthCheckpoints, id: \.self) { checkpoint in
                        Text(checkpoint).tag(checkpoint)
                    }
                }
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
                StudioGeometryCameraEditor(
                    enabled: $suppliesCameras,
                    document: $geometryCameras,
                    views: multiviewViews,
                    message: $errorMessage
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
                if cameras.isEmpty {
                    Stepper("Camera \(camera)", value: $camera, in: 0...16)
                } else {
                    // The CLI numbers cameras the way AVFoundation lists them; this is that list.
                    Picker("Camera", selection: $camera) {
                        ForEach(cameras) { device in
                            Text(device.name).tag(device.index)
                        }
                    }
                }
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
                facePicker(label: "Face to embed", selection: $faceIndex)
            }
            if task == .faceCompare {
                facePicker(label: "Reference face", selection: $referenceFaceIndex)
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
        case .depth: "Review the depth preview; the depth map and manifest are beside it."
        case .depthVideo: "Review the depth video and per-frame EXR/PNG artifacts."
        case .geometry, .geometryMultiview: "Orbit the GLB/PLY point cloud and inspect depth/normal maps."
        case .liveTrack: "The annotated camera recording appears as soon as it is written."
        default: "Structured results and sidecars remain available in the Library."
        }
    }

    private var resultPreferredKinds: [StudioOutputFileKind] {
        switch task {
        case .geometry, .geometryMultiview: [.model3D, .image, .text]
        case .depth: [.image, .text]
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

    /// Which face `--face-index` means, chosen by clicking it on the primary image when a Face
    /// detection run has drawn boxes on that image; the stepper stays for an image nobody has
    /// detected faces in yet.
    @ViewBuilder
    private func facePicker(label: String, selection: Binding<Int>) -> some View {
        if let document = faceDetectionDocument(for: primaryInput) {
            VStack(alignment: .leading, spacing: 5) {
                HStack {
                    Text(label)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    Spacer()
                    Text("Face \(selection.wrappedValue)")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                }
                StudioVisionOverlayPreview(
                    imageURL: URL(fileURLWithPath: primaryInput),
                    jsonURL: document,
                    kind: .faces,
                    selectedFaceIndex: selection
                )
                .frame(height: 200)
                .background(MereRunTheme.surfaceRaised)
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                        .strokeBorder(MereRunTheme.border, lineWidth: 1)
                }
                Text("Click a face to choose it.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        } else {
            Stepper("\(label) \(selection.wrappedValue)", value: selection, in: 0...100)
            Text("Run Face detection on this image first to choose a face by clicking it.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
        }
    }

    /// The newest finished Face detection result for `path`, whose boxes the picker draws.
    private func faceDetectionDocument(for path: String) -> URL? {
        guard !path.isBlank else { return nil }
        let input = URL(fileURLWithPath: path).standardizedFileURL
        let detection = library.items
            .filter {
                $0.templateID == .visionFaceDetect && $0.status == .completed
                    && $0.inputURL?.standardizedFileURL == input
            }
            .max { $0.createdAt < $1.createdAt }
        return detection.flatMap { artifact(in: $0, extension: "json") }
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

    private var commandDraft: CommandDraft {
        let root = URL(fileURLWithPath: NSString(string: outputDirectory).expandingTildeInPath)
        var draft = CommandCatalog.template(id: task.templateID)?.defaultDraft() ?? CommandDraft()
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
        draft.visionMaxEdge = depthNative ? nil : depthMaxEdge
        draft.visionNative = depthNative
        draft.visionCheckpoint = depthCheckpoint.isEmpty ? nil : depthCheckpoint
        draft.visionResolutionLevel = resolutionLevel
        draft.visionTokenCount = tokenCount
        draft.visionMaxPoints = maxPoints
        draft.camerasPath = task == .geometryMultiview && suppliesCameras ? draftCamerasPath : ""
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
        case .depth, .depthVideo, .geometry, .geometryMultiview:
            draft.outputPath = root.path
        case .liveTrack:
            draft.outputPath = root.appendingPathComponent("live-tracking.mp4").path
            draft.visionJSONOutputPath = root.appendingPathComponent("live-tracking.json").path
        }

        return draft
    }

    /// `vision depth --checkpoint` names, as `MarigoldV2DepthCheckpoint` spells them.
    private static let depthCheckpoints = [
        "log-stage2", "log-stage1", "log-layered", "uniform-base", "uniform-layered",
        "disparity-base", "disparity-layered",
    ]

    private func run() {
        errorMessage = nil
        guard validate() else { return }
        guard CommandCatalog.template(id: task.templateID) != nil else {
            errorMessage = "The selected vision command is unavailable."
            return
        }
        var draft = commandDraft
        if task == .geometryMultiview, suppliesCameras {
            // The camera file lives beside the run's output folder, which the command fills itself.
            let camerasURL = StudioCameraDocuments.url(besideOutputDirectory: outputDirectory)
            do {
                try FileManager.default.createDirectory(at: camerasURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                try geometryCameras.json().write(to: camerasURL, options: .atomic)
            } catch {
                errorMessage = "Studio could not write the camera file: \(error.localizedDescription)"
                return
            }
            draft.camerasPath = camerasURL.path
        }

        requestID = StudioSpecialistRunner.submit(
            templateID: task.templateID,
            mode: task == .liveTrack ? .track : .readImage,
            draft: draft,
            controller: controller,
            library: library
        )
    }

    /// The images a multi-view run sends, in order.
    private var multiviewPaths: [String] {
        ([primaryInput] + additionalInputs).filter { !$0.isBlank }
    }

    /// Those images with their decoded sizes, for labelling and sizing cameras. Sizes come from
    /// `viewSizes`, read once per change of the list rather than per render.
    private var multiviewViews: [StudioCameraView] {
        multiviewPaths.map { StudioCameraView(name: URL(fileURLWithPath: $0).lastPathComponent, pixelSize: viewSizes[$0]) }
    }

    private func refreshViewSizes() {
        viewSizes = Dictionary(uniqueKeysWithValues: multiviewPaths.compactMap { path in
            StudioPixelSize.of(URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)).map { (path, $0) }
        })
    }

    private struct CameraDraftKey: Equatable {
        let enabled: Bool
        let document: StudioGeometryCameraDocument
        let views: [StudioCameraView]
    }

    private var cameraDraftKey: CameraDraftKey {
        CameraDraftKey(enabled: suppliesCameras, document: geometryCameras, views: multiviewViews)
    }

    /// Keeps the Command view's camera file current, a moment after editing stops; only a document
    /// the CLI would accept is saved, under a name made from its content.
    private func saveDraftCameras() async {
        guard suppliesCameras, geometryCameras.problems(views: multiviewViews).isEmpty else {
            draftCamerasPath = ""
            return
        }
        try? await Task.sleep(for: .milliseconds(300))
        guard !Task.isCancelled else { return }
        do {
            let url = try StudioCameraDocuments.storeDraft(page: "Vision Geometry", content: geometryCameras.json())
            // Rows the Library still names (queued Command-view runs included) keep their files.
            let referenced = Set(library.items.compactMap { $0.commandDraft?.camerasPath })
            StudioCameraDocuments.pruneDrafts(page: "Vision Geometry", current: url, referenced: referenced)
            draftCamerasPath = url.path
        } catch {
            draftCamerasPath = ""
        }
    }

    /// A camera file chosen before this page edited cameras is read into the editor, once. A path
    /// that no longer exists is forgotten quietly; one that will not read is reported once, then
    /// forgotten.
    private func adoptLegacyCameras() {
        guard !legacyCamerasPath.isBlank else { return }
        let url = URL(fileURLWithPath: NSString(string: legacyCamerasPath).expandingTildeInPath)
        legacyCamerasPath = ""
        guard geometryCameras.cameras.isEmpty, FileManager.default.fileExists(atPath: url.path) else { return }
        do {
            geometryCameras = try StudioGeometryCameraDocument.importing(Data(contentsOf: url))
            suppliesCameras = true
        } catch {
            errorMessage = "The camera file at \(url.lastPathComponent) could not be read into the editor: \(error.localizedDescription)"
        }
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
        if task == .geometryMultiview, suppliesCameras,
           let problem = geometryCameras.problems(views: multiviewViews).first {
            errorMessage = problem
            return false
        }
        if task == .liveTrack && prompts.isBlank {
            errorMessage = "Enter at least one tracked prompt."
            return false
        }
        return true
    }
}

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
    /// When set, each face box is a button that picks its index, and the chosen one is drawn in
    /// the accent.
    var selectedFaceIndex: Binding<Int>?

    @State private var image: NSImage?
    /// How the photo is turned to show upright; the documents' coordinates are in stored pixels.
    @State private var orientation = StudioImageOrientation.up
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
                    if let selectedFaceIndex, let faces {
                        faceButtons(faces, selection: selectedFaceIndex, in: aspectFitRect(imageSize: image.size, in: geometry.size))
                    }
                }
                .frame(width: geometry.size.width, height: geometry.size.height)
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
        // The picture is shown upright; the face and pose documents place their boxes and
        // landmarks in the stored pixels the CLI decoded, so they map through the orientation.
        image = StudioImagePreviewLoader.downsampledImage(from: imageURL, maxPixelSize: 1_600)?.image
        orientation = StudioImageMetadata.read(imageURL)?.orientation ?? .up
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

    /// Where a stored-pixel point of a `storedSize` document lands in the fitted upright picture.
    private func viewPoint(_ point: CGPoint, storedSize: CGSize, in rect: CGRect) -> CGPoint {
        let shown = orientation.displaySize(ofStored: storedSize)
        let upright = orientation.displayPoint(fromStored: point, storedSize: storedSize)
        return CGPoint(
            x: rect.minX + upright.x * rect.width / max(1, shown.width),
            y: rect.minY + upright.y * rect.height / max(1, shown.height)
        )
    }

    /// Where `face` lands in the fitted upright picture.
    private func faceFrame(_ face: StudioFaceOverlayResult.Record, result: StudioFaceOverlayResult, in rect: CGRect) -> CGRect {
        let storedSize = CGSize(width: max(1, result.width), height: max(1, result.height))
        let box = face.detection.boundingBox
        let start = viewPoint(CGPoint(x: box.x, y: box.y), storedSize: storedSize, in: rect)
        let end = viewPoint(CGPoint(x: box.x + box.width, y: box.y + box.height), storedSize: storedSize, in: rect)
        return StudioRegionGeometry.rect(from: start, to: end)
    }

    /// One transparent button per detected face, so a click picks it. The Canvas underneath
    /// draws the chosen one in the accent.
    private func faceButtons(_ result: StudioFaceOverlayResult, selection: Binding<Int>, in rect: CGRect) -> some View {
        ForEach(result.faces, id: \.index) { face in
            let frame = faceFrame(face, result: result, in: rect)
            Button {
                selection.wrappedValue = face.index
            } label: {
                Color.clear
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .frame(width: max(frame.width, 8), height: max(frame.height, 8))
            .position(x: frame.midX, y: frame.midY)
            .help("Face \(face.index), \(Int(face.detection.score * 100))% confidence")
            .accessibilityLabel("Face \(face.index), \(Int(face.detection.score * 100)) percent")
            .accessibilityAddTraits(selection.wrappedValue == face.index ? .isSelected : [])
        }
    }

    private func drawFaces(
        _ result: StudioFaceOverlayResult,
        in rect: CGRect,
        context: inout GraphicsContext
    ) {
        if let selectedFaceIndex {
            drawSelectableFaces(result, selected: selectedFaceIndex.wrappedValue, in: rect, context: &context)
            return
        }
        let storedSize = CGSize(width: max(1, result.width), height: max(1, result.height))
        for face in result.faces {
            let frame = faceFrame(face, result: result, in: rect)
            context.stroke(Path(frame), with: .color(.green), lineWidth: 2)
            for point in face.detection.landmarks {
                let center = viewPoint(CGPoint(x: point.x, y: point.y), storedSize: storedSize, in: rect)
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

    /// The picker's rendering: the chosen face in the accent with a soft fill, the rest as quiet
    /// white outlines, each numbered the way `--face-index` counts them.
    private func drawSelectableFaces(
        _ result: StudioFaceOverlayResult,
        selected: Int,
        in rect: CGRect,
        context: inout GraphicsContext
    ) {
        for face in result.faces {
            let frame = faceFrame(face, result: result, in: rect)
            let isSelected = face.index == selected
            let path = Path(roundedRect: frame, cornerRadius: 3)
            if isSelected {
                context.fill(path, with: .color(MereRunTheme.accent.opacity(0.18)))
            }
            context.stroke(
                path,
                with: .color(isSelected ? MereRunTheme.accent : Color.white.opacity(0.85)),
                lineWidth: isSelected ? 2.5 : 1.5
            )
            let tag = Text("\(face.index)")
                .font(.system(size: 10.5, weight: .semibold))
                .foregroundColor(isSelected ? MereRunTheme.onAccent : Color.black.opacity(0.8))
            let tagSize = context.resolve(tag).measure(in: CGSize(width: 60, height: 20))
            let tagRect = CGRect(
                x: frame.minX,
                y: max(rect.minY, frame.minY - tagSize.height - 4),
                width: tagSize.width + 8,
                height: tagSize.height + 3
            )
            context.fill(
                Path(roundedRect: tagRect, cornerRadius: 3),
                with: .color(isSelected ? MereRunTheme.accent : Color.white.opacity(0.85))
            )
            context.draw(tag, at: CGPoint(x: tagRect.midX, y: tagRect.midY), anchor: .center)
        }
    }

    private func drawPose(
        _ result: StudioPoseOverlayResult,
        in rect: CGRect,
        context: inout GraphicsContext
    ) {
        let storedSize = CGSize(width: max(1, result.imageWidth), height: max(1, result.imageHeight))
        for subject in result.subjects {
            let color: Color = switch subject.kind {
            case "body": .cyan
            case "hand": .yellow
            default: .pink
            }
            for point in subject.points {
                let normalizedY = result.coordinateSpace == "normalized-bottom-left"
                    ? 1 - point.y
                    : point.y
                // Normalized in the stored pixels, so scale up, turn upright, then fit.
                let center = viewPoint(
                    CGPoint(x: point.x * storedSize.width, y: normalizedY * storedSize.height),
                    storedSize: storedSize,
                    in: rect
                )
                context.fill(
                    Path(ellipseIn: CGRect(x: center.x - 2.5, y: center.y - 2.5, width: 5, height: 5)),
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
