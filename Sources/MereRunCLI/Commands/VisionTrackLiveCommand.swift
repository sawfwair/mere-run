import ArgumentParser
import Foundation
import MereRunCore
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

struct VisionTrackLive: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "track-live",
        abstract: "Capture from a camera and track text-prompted objects with the native SAM 3.1 runtime."
    )

    @Option(
        name: [.long],
        parsing: .upToNextOption,
        help: "One or more text prompts used to seed tracked objects from the init frame."
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

    @Option(name: [.customLong("init-frame")], help: "Initial frame index used to seed tracking (default: 0).")
    var initFrame: Int = 0

    @Option(
        name: [.customLong("seed-search-frames")],
        help: "Additional frames to search when the init frame finds no objects (default: 30)."
    )
    var seedSearchFrames: Int = 30

    @Option(name: [.long], help: "Score threshold between 0 and 1 (default: 0.05).")
    var threshold: Double = 0.05

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
        guard initFrame >= 0 else {
            throw ValidationError("--init-frame must be greater than or equal to 0.")
        }
        guard seedSearchFrames >= 0 else {
            throw ValidationError("--seed-search-frames must be greater than or equal to 0.")
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
        let captureFile = LiveCaptureFile(url: captureURL)
        defer { captureFile.remove() }

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
        defer { segmenter.unload() }
        let tracker = SAM31VideoTracker(segmenter: segmenter)
        let result = try tracker.track(
            videoURL: captureURL,
            promptSet: SAM31PromptSet(textPrompts: VisionSegment.normalizedTextPrompts(prompt)),
            outputVideoURL: outputVideoURL,
            jsonOutputURL: outputJSONURL,
            initFrameIndex: initFrame,
            threshold: Float(threshold),
            resolution: resolution,
            showBoxes: showBoxes,
            showLabels: showLabels,
            seedFrameSearchLimit: seedSearchFrames
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

/// Owns the camera recording through tracking, including an interrupt or termination while capture is blocked.
/// One dispatch signal source per signal runs cleanup off the POSIX signal handler before the process exits.
final class LiveCaptureFile: @unchecked Sendable {
    static let handledSignals = [SIGINT, SIGTERM]

    let url: URL

    private let lock = NSLock()
    private let sources: [DispatchSourceSignal]
    private let previousSignalHandlers: [sig_t?]
    private var active = true

    init(url: URL) {
        self.url = url
        previousSignalHandlers = Self.handledSignals.map { signal($0, SIG_IGN) }
        sources = Self.handledSignals.map { DispatchSource.makeSignalSource(signal: $0, queue: .global()) }
        for (signo, source) in zip(Self.handledSignals, sources) {
            source.setEventHandler { [weak self] in
                guard self?.remove() == true else { return }
                _exit(128 + signo)
            }
            source.resume()
        }
    }

    @discardableResult
    func remove() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard active else { return false }
        active = false
        try? FileManager.default.removeItem(at: url)
        for (signo, (source, handler)) in zip(Self.handledSignals, zip(sources, previousSignalHandlers)) {
            source.cancel()
            _ = signal(signo, handler)
        }
        return true
    }
}
