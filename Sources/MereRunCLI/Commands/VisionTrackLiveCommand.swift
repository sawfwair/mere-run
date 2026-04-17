import ArgumentParser
import Foundation
import MereRunCore

struct VisionTrackLive: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "track-live",
        abstract: "Capture from a camera and track text-prompted objects with the native SAM 3.1 runtime."
    )

    @Option(
        name: [.long],
        parsing: .upToNextOption,
        help: "One or more text prompts used to seed tracked objects from the first captured frame."
    )
    var prompt: [String] = []

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local SAM 3.1 model root.")
    var model: String?

    @Option(name: [.customShort("o"), .long], help: "Annotated output video path.")
    var output: String

    @Option(name: [.customLong("json-output")], help: "Tracking JSON path.")
    var jsonOutput: String?

    @Option(name: [.long], help: "Camera device index (default: 0).")
    var camera: Int = 0

    @Option(name: [.customLong("duration-seconds")], help: "Capture duration in seconds (default: 10).")
    var durationSeconds: Double = 10

    @Option(name: [.long], help: "Score threshold between 0 and 1 (default: 0.3).")
    var threshold: Double = 0.3

    @Option(name: [.long], help: "Square input resolution used for SAM 3.1 preprocessing (default: 1008).")
    var resolution: Int = 1008

    @Flag(name: [.long], help: "Draw bounding boxes over tracked masks in the annotated video.")
    var showBoxes: Bool = false

    @Flag(name: [.customLong("show-labels")], help: "Reserved for labeled video overlays.")
    var showLabels: Bool = false

    func validate() throws {
        guard !prompt.isEmpty else {
            throw ValidationError("Provide at least one --prompt value.")
        }
        guard camera >= 0 else {
            throw ValidationError("--camera must be greater than or equal to 0.")
        }
        guard durationSeconds > 0 else {
            throw ValidationError("--duration-seconds must be greater than 0.")
        }
        guard (0.0...1.0).contains(threshold) else {
            throw ValidationError("--threshold must be between 0 and 1.")
        }
        guard resolution > 0 else {
            throw ValidationError("--resolution must be greater than 0.")
        }
        _ = showLabels
    }

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: false)

        let outputVideoURL = URL(fileURLWithPath: output).standardizedFileURL
        let outputJSONURL = jsonOutput.map { URL(fileURLWithPath: $0).standardizedFileURL }
        let captureURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-sam31-live-\(UUID().uuidString)")
            .appendingPathExtension("mov")

        try SAM31CameraCapture.record(
            cameraIndex: camera,
            durationSeconds: durationSeconds,
            outputURL: captureURL
        )

        let resolvedModel = try VisionSegment.resolveModelRoot(model)
        let segmenter = try SAM31ImageSegmenter(
            modelRootURL: resolvedModel.rootURL,
            expectedModelID: resolvedModel.isManaged ? resolvedModel.modelID : nil
        )
        let tracker = SAM31VideoTracker(segmenter: segmenter)
        let result = try tracker.track(
            videoURL: captureURL,
            promptSet: SAM31PromptSet(textPrompts: prompt),
            outputVideoURL: outputVideoURL,
            jsonOutputURL: outputJSONURL,
            initFrameIndex: 0,
            threshold: Float(threshold),
            resolution: resolution,
            showBoxes: showBoxes,
            showLabels: showLabels
        )

        print("Model: \(result.modelID)")
        print("Objects: \(result.objects.count)")
        print("Frames: \(result.frames.count)")
        print("Video: \(result.annotatedVideoPath)")
        if let jsonOutputPath = result.jsonOutputPath {
            print("JSON: \(jsonOutputPath)")
        }
    }
}
