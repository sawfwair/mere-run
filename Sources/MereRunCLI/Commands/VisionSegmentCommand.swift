import ArgumentParser
import Foundation
import MereRunCore

struct VisionSegment: AsyncParsableCommand {
    struct ResolvedModel {
        let modelID: String
        let rootURL: URL
        let isManaged: Bool
    }

    static let defaultManagedModelID: ModelResolver.ModelID = .visionSegmentSAM31

    static let configuration = CommandConfiguration(
        commandName: "segment",
        abstract: "Segment prompted objects in an image with the native SAM 3.1 runtime.",
        discussion: """
        Uses the native Swift/MLX SAM 3.1 detector path in MereRunCore.
        If --model is omitted, this command resolves the managed model id
        `vision-segment-sam31` from the local mere.run model store.
        """
    )

    @Argument(help: "Image file path.")
    var image: String

    @Option(
        name: [.long],
        parsing: .upToNextOption,
        help: "One or more object prompts. Example: --prompt \"a person\" \"a phone\"."
    )
    var prompt: [String] = []

    @Option(
        name: [.long],
        help: "Box prompt in image pixels: x1,y1,x2,y2[,label]. Repeat for multiple objects."
    )
    var box: [String] = []

    @Option(
        name: [.long],
        help: "Point prompt in image pixels: x,y,positive[,label] or x,y,negative[,label]. Repeat to build multi-point prompts."
    )
    var point: [String] = []

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local SAM 3.1 model root.")
    var model: String?

    @Option(name: [.customShort("o"), .long], help: "Annotated output image path (default: <image>_segmented.<ext>).")
    var output: String?

    @Option(name: [.customLong("json-output")], help: "JSON metadata path (default: <image>_segmented.json).")
    var jsonOutput: String?

    @Option(name: [.customLong("mask-output-dir")], help: "Optional directory for per-object PNG mask exports.")
    var maskOutputDir: String?

    @Option(name: [.long], help: "Score threshold between 0 and 1 (default: 0.05).")
    var threshold: Double = 0.05

    @Option(name: [.long], help: "Square input resolution used for SAM 3.1 preprocessing (default: 1008).")
    var resolution: Int = 1008

    @Flag(name: [.long], help: "Draw bounding boxes over the mask overlays in the annotated output image.")
    var showBoxes: Bool = false

    @Flag(name: [.long], help: "For geometry prompts, emit multiple mask candidates per prompted object.")
    var multimask: Bool = false

    @Flag(name: [.customLong("preflight")], help: "Inspect the segmentation request without loading SAM or processing the image.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "With --preflight, emit a structured JSON report.")
    var json: Bool = false

    @Flag(name: [.customShort("q"), .long], help: "Suppress normal progress output.")
    var quiet: Bool = false

    @Flag(name: [.customLong(RunReceipt.flagName)], help: RunReceipt.flagHelp)
    var receipt: Bool = false

    func validate() throws {
        if json && !preflight {
            throw ValidationError("--json is only supported with --preflight for vision segment.")
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
        _ = try parsedPromptSet()
    }

    func run() async throws {
        let imageURL = URL(fileURLWithPath: image).standardizedFileURL
        let outputImageURL = Self.resolveAnnotatedOutputURL(output, inputImageURL: imageURL)
        let outputJSONURL = Self.resolveJSONOutputURL(jsonOutput, inputImageURL: imageURL)
        let maskOutputDirectoryURL = Self.resolveDirectoryURL(maskOutputDir)
        if preflight {
            try runPreflight(
                imageURL: imageURL,
                outputImageURL: outputImageURL,
                outputJSONURL: outputJSONURL,
                maskOutputDirectoryURL: maskOutputDirectoryURL
            )
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: imageURL.path) else {
            throw ValidationError("Image not found: \(imageURL.path)")
        }

        let resolvedModel = try Self.resolveModelRoot(model, fileManager: fileManager)
        let promptSet = try parsedPromptSet()

        let segmenter = try SAM31ImageSegmenter(
            modelRootURL: resolvedModel.rootURL,
            expectedModelID: resolvedModel.isManaged ? resolvedModel.modelID : nil
        )
        defer { segmenter.unload() }
        let result = try segmenter.segment(
            imageURL: imageURL,
            promptSet: promptSet,
            annotatedImageURL: outputImageURL,
            jsonOutputURL: outputJSONURL,
            threshold: Float(threshold),
            resolution: resolution,
            showBoxes: showBoxes,
            multimask: multimask,
            maskOutputDirectoryURL: maskOutputDirectoryURL
        )

        if !quiet {
            print("Model: \(result.modelID)")
            print("Detections: \(result.detections.count)")
            print("Image: \(result.annotatedImageURL.path)")
            print("JSON: \(result.jsonOutputURL.path)")
            if let maskOutputDirectoryURL {
                print("Masks: \(maskOutputDirectoryURL.path)")
            }
        }
        try RunReceipt.emit(
            RunReceipt.annotatedImageOutputs(
                image: result.annotatedImageURL,
                detections: result.jsonOutputURL,
                masks: maskOutputDirectoryURL
            ),
            enabled: receipt
        )
    }

    func makePreflightEnvelope(
        imageURL: URL,
        outputImageURL: URL,
        outputJSONURL: URL,
        maskOutputDirectoryURL: URL?,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) -> VisionSegmentPreflightEnvelope {
        let input = VisionSegmentPreflightInput(
            imageURL: imageURL,
            outputImageURL: outputImageURL,
            outputJSONURL: outputJSONURL,
            maskOutputDirectoryURL: maskOutputDirectoryURL,
            prompt: prompt,
            box: box,
            point: point,
            model: model,
            threshold: threshold,
            resolution: resolution,
            showBoxes: showBoxes,
            multimask: multimask,
            segmentArgv: segmentActionArguments(
                imageURL: imageURL,
                outputImageURL: outputImageURL,
                outputJSONURL: outputJSONURL,
                maskOutputDirectoryURL: maskOutputDirectoryURL
            ),
            cwd: fileManager.currentDirectoryPath
        )
        return VisionSegmentPreflightAnalyzer(
            input: input,
            fileManager: fileManager,
            now: now
        ).envelope()
    }

    private func runPreflight(
        imageURL: URL,
        outputImageURL: URL,
        outputJSONURL: URL,
        maskOutputDirectoryURL: URL?
    ) throws {
        let envelope = makePreflightEnvelope(
            imageURL: imageURL,
            outputImageURL: outputImageURL,
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

    private func segmentActionArguments(
        imageURL: URL,
        outputImageURL: URL,
        outputJSONURL: URL,
        maskOutputDirectoryURL: URL?
    ) -> [String] {
        var args = ["mere.run", "vision", "segment", imageURL.path]
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
        args += ["--output", outputImageURL.path]
        args += ["--json-output", outputJSONURL.path]
        if let maskOutputDirectoryURL {
            args += ["--mask-output-dir", maskOutputDirectoryURL.path]
        }
        args += ["--threshold", String(threshold), "--resolution", String(resolution)]
        if showBoxes {
            args.append("--show-boxes")
        }
        if multimask {
            args.append("--multimask")
        }
        return args
    }

    static func resolveModelRoot(
        _ rawModel: String?,
        fileManager: FileManager = .default,
        resolver: ModelResolver = ModelResolver()
    ) throws -> ResolvedModel {
        if let rawModel, !rawModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let url = URL(fileURLWithPath: rawModel).standardizedFileURL
            var isDirectory: ObjCBool = false
            if fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) {
                guard isDirectory.boolValue else {
                    throw ValidationError("Model path must be a directory: \(url.path)")
                }
                return ResolvedModel(
                    modelID: rawModel,
                    rootURL: url,
                    isManaged: false
                )
            }

            if let modelID = ModelResolver.ModelID(rawValue: rawModel) {
                do {
                    let resolved = try resolver.resolve(modelID)
                    return ResolvedModel(
                        modelID: modelID.rawValue,
                        rootURL: resolved.rootURL,
                        isManaged: true
                    )
                } catch {
                    throw ValidationError(
                        "Model \(modelID.rawValue) not found. Pull it with `\(CLICommandDisplay.modelPullCommand(for: modelID.rawValue))` or point --model at a local path."
                    )
                }
            }

            throw ValidationError("Model path not found: \(rawModel). Pass a local model path or the managed id vision-segment-sam31.")
        }

        do {
            let resolved = try resolver.resolve(defaultManagedModelID)
            return ResolvedModel(
                modelID: defaultManagedModelID.rawValue,
                rootURL: resolved.rootURL,
                isManaged: true
            )
        } catch {
            throw ValidationError(
                "Model \(defaultManagedModelID.rawValue) not found. Pull it with `\(CLICommandDisplay.modelPullCommand(for: defaultManagedModelID.rawValue))` or point --model at a local path."
            )
        }
    }

    static func resolveAnnotatedOutputURL(_ rawOutput: String?, inputImageURL: URL) -> URL {
        guard let rawOutput, !rawOutput.isEmpty else {
            return SAM31ImageSegmenter.defaultAnnotatedOutputURL(for: inputImageURL)
        }

        let outputURL = URL(fileURLWithPath: rawOutput).standardizedFileURL
        if outputURL.pathExtension.isEmpty {
            let ext = inputImageURL.pathExtension.isEmpty ? "png" : inputImageURL.pathExtension
            return outputURL.appendingPathExtension(ext)
        }
        return outputURL
    }

    static func resolveJSONOutputURL(_ rawOutput: String?, inputImageURL: URL) -> URL {
        guard let rawOutput, !rawOutput.isEmpty else {
            return SAM31ImageSegmenter.defaultJSONOutputURL(for: inputImageURL)
        }

        let outputURL = URL(fileURLWithPath: rawOutput).standardizedFileURL
        if outputURL.pathExtension.isEmpty {
            return outputURL.appendingPathExtension("json")
        }
        return outputURL
    }

    static func resolveDirectoryURL(_ rawDirectory: String?) -> URL? {
        guard let rawDirectory, !rawDirectory.isEmpty else { return nil }
        return URL(fileURLWithPath: rawDirectory).standardizedFileURL
    }

    func parsedPromptSet() throws -> SAM31PromptSet {
        SAM31PromptSet(
            textPrompts: Self.normalizedTextPrompts(prompt),
            boxPrompts: try box.map(Self.parseBoxPrompt),
            pointPrompts: try point.map(Self.parsePointPrompt)
        )
    }

    static func normalizedTextPrompts(_ prompts: [String]) -> [String] {
        prompts.compactMap { rawPrompt in
            let words = rawPrompt
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .split { $0.isWhitespace }
                .map(String.init)
            guard !words.isEmpty else { return nil }
            let normalizedWords: ArraySlice<String>
            switch words[0].lowercased() {
            case "a" where words.count > 1,
                 "an" where words.count > 1,
                 "the" where words.count > 1:
                normalizedWords = words.dropFirst()
            default:
                normalizedWords = words[...]
            }
            let normalized = normalizedWords.joined(separator: " ")
            return normalized.isEmpty ? nil : normalized
        }
    }

    static func parseBoxPrompt(_ raw: String) throws -> SAM31PromptBox {
        let parts = raw.split(separator: ",", omittingEmptySubsequences: false).map {
            String($0).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard parts.count == 4 || parts.count == 5 else {
            throw ValidationError("Box prompts must be x1,y1,x2,y2[,label]. Received: \(raw)")
        }
        guard
            let x1 = Float(parts[0]),
            let y1 = Float(parts[1]),
            let x2 = Float(parts[2]),
            let y2 = Float(parts[3])
        else {
            throw ValidationError("Box prompt coordinates must be numeric. Received: \(raw)")
        }
        let label = parts.count == 5 ? parts[4] : nil
        return SAM31PromptBox(x1: x1, y1: y1, x2: x2, y2: y2, label: label)
    }

    static func parsePointPrompt(_ raw: String) throws -> SAM31PromptPoint {
        let parts = raw.split(separator: ",", omittingEmptySubsequences: false).map {
            String($0).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard parts.count == 3 || parts.count == 4 else {
            throw ValidationError("Point prompts must be x,y,positive[,label] or x,y,negative[,label]. Received: \(raw)")
        }
        guard
            let x = Float(parts[0]),
            let y = Float(parts[1])
        else {
            throw ValidationError("Point prompt coordinates must be numeric. Received: \(raw)")
        }
        let polarity = parts[2].lowercased()
        let isPositive: Bool
        switch polarity {
        case "positive", "pos", "p", "1":
            isPositive = true
        case "negative", "neg", "n", "0":
            isPositive = false
        default:
            throw ValidationError("Point polarity must be positive or negative. Received: \(parts[2])")
        }
        let label = parts.count == 4 ? parts[3] : nil
        return SAM31PromptPoint(x: x, y: y, isPositive: isPositive, label: label)
    }
}
