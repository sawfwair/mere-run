import ArgumentParser
import Foundation
import MereRunCore

struct VisionTrack: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "track",
        abstract: "Track prompted objects through a video with the native SAM 3.1 runtime."
    )

    @Argument(help: "Video file path.")
    var video: String

    @Option(
        name: [.long],
        parsing: .upToNextOption,
        help: "One or more text prompts used to seed tracked objects on the init frame."
    )
    var prompt: [String] = []

    @Option(name: [.long], help: "Box prompt in image pixels: x1,y1,x2,y2[,label]. Repeat for multiple objects.")
    var box: [String] = []

    @Option(name: [.long], help: "Point prompt in image pixels: x,y,positive[,label] or x,y,negative[,label]. Repeat for multiple objects.")
    var point: [String] = []

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local SAM 3.1 model root.")
    var model: String?

    @Option(name: [.customShort("o"), .long], help: "Annotated output video path (default: <video>_tracked.mp4).")
    var output: String?

    @Option(name: [.customLong("json-output")], help: "Tracking JSON path (default: <video>_tracked.json).")
    var jsonOutput: String?

    @Option(name: [.customLong("mask-output-dir")], help: "Optional directory for per-frame mask PNG exports.")
    var maskOutputDir: String?

    @Option(name: [.customLong("init-frame")], help: "Initial frame index used to seed tracking (default: 0).")
    var initFrame: Int = 0

    @Option(name: [.customLong("end-frame")], help: "Optional inclusive final frame index.")
    var endFrame: Int?

    @Option(name: [.long], help: "Score threshold between 0 and 1 (default: 0.05).")
    var threshold: Double = 0.05

    @Option(name: [.long], help: "Square input resolution used for SAM 3.1 preprocessing (default: 1008).")
    var resolution: Int = 1008

    @Flag(name: [.long], help: "Draw bounding boxes over tracked masks in the annotated video.")
    var showBoxes: Bool = false

    @Flag(name: [.customLong("show-labels")], help: "Reserved for labeled video overlays.")
    var showLabels: Bool = false

    @Flag(name: [.customLong("preflight")], help: "Inspect the tracking request without loading SAM or processing video.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "With --preflight, emit a structured JSON report.")
    var json: Bool = false

    @Flag(name: [.customShort("q"), .long], help: "Suppress normal progress output.")
    var quiet: Bool = false

    @Flag(name: [.customLong(RunReceipt.flagName)], help: RunReceipt.flagHelp)
    var receipt: Bool = false

    func validate() throws {
        if json && !preflight {
            throw ValidationError("--json is only supported with --preflight for vision track.")
        }
        try RunReceipt.validate(receipt: receipt, preflight: preflight)
        if preflight {
            return
        }
        guard !prompt.isEmpty || !box.isEmpty || !point.isEmpty else {
            throw ValidationError("Provide at least one --prompt, --box, or --point value.")
        }
        guard (0.0...1.0).contains(threshold) else {
            throw ValidationError("--threshold must be between 0 and 1.")
        }
        guard resolution > 0 else {
            throw ValidationError("--resolution must be greater than 0.")
        }
        guard initFrame >= 0 else {
            throw ValidationError("--init-frame must be greater than or equal to 0.")
        }
        if let endFrame, endFrame < initFrame {
            throw ValidationError("--end-frame must be greater than or equal to --init-frame.")
        }
        _ = try parsedPromptSet()
        _ = showLabels
    }

    func run() async throws {
        let videoURL = URL(fileURLWithPath: video).standardizedFileURL
        let outputVideoURL = Self.resolveOutputURL(output, inputVideoURL: videoURL)
        let outputJSONURL = Self.resolveJSONOutputURL(jsonOutput, inputVideoURL: videoURL)
        let maskOutputDirectoryURL = VisionSegment.resolveDirectoryURL(maskOutputDir)
        if preflight {
            try runPreflight(
                videoURL: videoURL,
                outputVideoURL: outputVideoURL,
                outputJSONURL: outputJSONURL,
                maskOutputDirectoryURL: maskOutputDirectoryURL
            )
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        guard FileManager.default.fileExists(atPath: videoURL.path) else {
            throw ValidationError("Video not found: \(videoURL.path)")
        }

        let resolvedModel = try VisionSegment.resolveModelRoot(model)
        let promptSet = try parsedPromptSet()

        let segmenter = try SAM31ImageSegmenter(
            modelRootURL: resolvedModel.rootURL,
            expectedModelID: resolvedModel.isManaged ? resolvedModel.modelID : nil
        )
        defer { segmenter.unload() }
        let tracker = SAM31VideoTracker(segmenter: segmenter)
        let result = try tracker.track(
            videoURL: videoURL,
            promptSet: promptSet,
            outputVideoURL: outputVideoURL,
            jsonOutputURL: outputJSONURL,
            initFrameIndex: initFrame,
            endFrameIndex: endFrame,
            threshold: Float(threshold),
            resolution: resolution,
            showBoxes: showBoxes,
            showLabels: showLabels,
            maskOutputDirectoryURL: maskOutputDirectoryURL
        )

        if !quiet {
            print("Model: \(result.modelID)")
            print("Objects: \(result.objects.count)")
            print("Frames: \(result.frames.count)")
            print("Video: \(result.annotatedVideoPath)")
            if let jsonOutputPath = result.jsonOutputPath {
                print("JSON: \(jsonOutputPath)")
            }
            if let maskOutputDirectoryURL {
                print("Masks: \(maskOutputDirectoryURL.path)")
            }
        }
        try RunReceipt.emit(
            RunReceipt.annotatedVideoOutputs(
                videoPath: result.annotatedVideoPath,
                trackingPath: result.jsonOutputPath,
                masks: maskOutputDirectoryURL
            ),
            enabled: receipt
        )
    }

    func parsedPromptSet() throws -> SAM31PromptSet {
        SAM31PromptSet(
            textPrompts: VisionSegment.normalizedTextPrompts(prompt),
            boxPrompts: try box.map(VisionSegment.parseBoxPrompt),
            pointPrompts: try point.map(VisionSegment.parsePointPrompt)
        )
    }

    static func resolveOutputURL(_ rawOutput: String?, inputVideoURL: URL) -> URL {
        guard let rawOutput, !rawOutput.isEmpty else {
            let parent = inputVideoURL.deletingLastPathComponent()
            let stem = inputVideoURL.deletingPathExtension().lastPathComponent
            return parent.appendingPathComponent("\(stem)_tracked").appendingPathExtension("mp4")
        }
        let outputURL = URL(fileURLWithPath: rawOutput).standardizedFileURL
        return outputURL.pathExtension.isEmpty ? outputURL.appendingPathExtension("mp4") : outputURL
    }

    static func resolveJSONOutputURL(_ rawOutput: String?, inputVideoURL: URL) -> URL {
        guard let rawOutput, !rawOutput.isEmpty else {
            let parent = inputVideoURL.deletingLastPathComponent()
            let stem = inputVideoURL.deletingPathExtension().lastPathComponent
            return parent.appendingPathComponent("\(stem)_tracked").appendingPathExtension("json")
        }
        let outputURL = URL(fileURLWithPath: rawOutput).standardizedFileURL
        return outputURL.pathExtension.isEmpty ? outputURL.appendingPathExtension("json") : outputURL
    }

    func makePreflightEnvelope(
        videoURL: URL,
        outputVideoURL: URL,
        outputJSONURL: URL,
        maskOutputDirectoryURL: URL?,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) -> VisionTrackPreflightEnvelope {
        let input = VisionTrackPreflightInput(
            videoURL: videoURL,
            outputVideoURL: outputVideoURL,
            outputJSONURL: outputJSONURL,
            maskOutputDirectoryURL: maskOutputDirectoryURL,
            prompt: prompt,
            box: box,
            point: point,
            model: model,
            initFrame: initFrame,
            endFrame: endFrame,
            threshold: threshold,
            resolution: resolution,
            showBoxes: showBoxes,
            showLabels: showLabels,
            trackArgv: trackActionArguments(
                videoURL: videoURL,
                outputVideoURL: outputVideoURL,
                outputJSONURL: outputJSONURL,
                maskOutputDirectoryURL: maskOutputDirectoryURL
            ),
            cwd: fileManager.currentDirectoryPath
        )
        return VisionTrackPreflightAnalyzer(
            input: input,
            fileManager: fileManager,
            now: now
        ).envelope()
    }

    private func runPreflight(
        videoURL: URL,
        outputVideoURL: URL,
        outputJSONURL: URL,
        maskOutputDirectoryURL: URL?
    ) throws {
        let envelope = makePreflightEnvelope(
            videoURL: videoURL,
            outputVideoURL: outputVideoURL,
            outputJSONURL: outputJSONURL,
            maskOutputDirectoryURL: maskOutputDirectoryURL
        )
        if json {
            print(try StructuredRunOutput.encode(envelope))
        } else {
            print(envelope.summary)
            for diagnostic in envelope.diagnostics {
                print("[\(diagnostic.severity.rawValue)] \(diagnostic.title): \(diagnostic.message)")
            }
        }
        if envelope.status == .blocked {
            throw ExitCode.failure
        }
    }

    private func trackActionArguments(
        videoURL: URL,
        outputVideoURL: URL,
        outputJSONURL: URL,
        maskOutputDirectoryURL: URL?
    ) -> [String] {
        var args = [
            "mere.run",
            "vision",
            "track",
            videoURL.path,
        ]
        if !prompt.isEmpty {
            args.append("--prompt")
            args.append(contentsOf: prompt)
        }
        for boxPrompt in box {
            args += ["--box", boxPrompt]
        }
        for pointPrompt in point {
            args += ["--point", pointPrompt]
        }
        if let model {
            args += ["--model", model]
        }
        args += ["--output", outputVideoURL.path]
        args += ["--json-output", outputJSONURL.path]
        if let maskOutputDirectoryURL {
            args += ["--mask-output-dir", maskOutputDirectoryURL.path]
        }
        args += ["--init-frame", String(initFrame)]
        if let endFrame {
            args += ["--end-frame", String(endFrame)]
        }
        args += ["--threshold", String(threshold), "--resolution", String(resolution)]
        if showBoxes {
            args.append("--show-boxes")
        }
        if showLabels {
            args.append("--show-labels")
        }
        return args
    }
}
