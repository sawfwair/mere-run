import ArgumentParser
import Foundation
import MereRunCore
import MereRunRelayKit

// MARK: - Image Generate Command

struct ImageGenerate: AsyncParsableCommand {
    static let defaultManagedModelID: ModelResolver.ModelID = .zetaNano

    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate images with local image models.",
        discussion: """
        Prints the output file path to stdout.
        Progress and diagnostics are printed to stderr.
        """
    )

    @Option(name: [.customShort("p"), .long], help: "Text prompt.")
    var prompt: String

    @Option(name: [.customShort("n"), .customLong("negative-prompt")], help: "Negative prompt (used when --cfg > 1.0).")
    var negativePrompt: String?

    @Option(
        name: [.customLong("cfg"), .customLong("cfg-scale")],
        help: "CFG scale (uses negative prompt when > 1.0; default is model-specific)."
    )
    var cfgScale: Double?

    @Option(name: [.customLong("sigma-shift")], help: "Sigma shift for the FlowMatch schedule (i2L recommends 8).")
    var sigmaShift: Double?

    @Option(
        name: [.customLong("sigmas")],
        help: "Pre-shifted descending sigma values. A terminal zero is optional."
    )
    var sigmaList: String?

    @Option(name: [.customShort("o"), .long], help: "Output PNG path (default: ./mererun-image-<timestamp>.png).")
    var output: String?

    @Option(name: [.customShort("W"), .long], help: "Output width in pixels.")
    var width: Int = 1024

    @Option(name: [.customShort("H"), .long], help: "Output height in pixels.")
    var height: Int = 1024

    @Option(name: [.customShort("s"), .long], help: "Number of inference steps (default is model-specific).")
    var steps: Int?

    @Option(name: [.long], help: "Random seed (UInt64).")
    var seed: UInt64?

    @Option(name: [.customShort("m"), .long], help: "Model path or canonical model id (default: image-zimage-nano).")
    var model: String?

    @Option(
        name: [.customShort("i"), .long],
        help: "Input image path for image-to-image. SenseNova U1.5 uses it as an editing reference."
    )
    var input: String?

    @Option(
        name: [.customLong("mask")],
        help: "Black/white edit mask. White pixels may change; black pixels are restored from --input."
    )
    var mask: String?

    @Option(
        name: [.customLong("outpaint")],
        help: "Expand the editable canvas as top,right,bottom,left padding inside --width/--height."
    )
    var outpaint: String?

    @Option(name: [.customLong("mask-feather")], help: "Blend radius in pixels at mask/outpaint edges.")
    var maskFeather: Int = 8

    @Option(
        name: [.customLong("ref-image")],
        help: "Reference image path for Klein, Qwen Edit, HiDream O1, or SenseNova U1.5 editing. Repeat for multiple references."
    )
    var referenceImages: [String] = []

    @Flag(name: [.customLong("keep-original-aspect")], help: "For one HiDream reference image, preserve the original aspect ratio.")
    var keepOriginalAspect: Bool = false

    @Option(
        name: [.customLong("strength"), .customLong("str")],
        help: "Image-to-image/reference change strength 0.0–1.0. Defaults to 0.75 for --input and 0.0 for Klein --ref-image."
    )
    var strength: Double?

    @Option(name: [.customLong("max-sequence-length")], help: "Max text sequence length.")
    var maxSequenceLength: Int = 512

    @Flag(
        name: [.customLong("structured-prompt"), .customLong("json-prompt")],
        help: "Expand --prompt into a structured JSON caption with a local text chat model before image generation."
    )
    var structuredPrompt: Bool = false

    @Option(name: [.customLong("structured-prompt-model")], help: "Text chat model id for --structured-prompt.")
    var structuredPromptModel: String = StructuredImagePromptAdapter.defaultModelID

    @Option(name: [.customLong("structured-prompt-model-root")], help: "Optional local model root for --structured-prompt-model.")
    var structuredPromptModelRoot: String?

    @Option(name: [.customLong("structured-prompt-max-tokens")], help: "Max new tokens for the structured prompt adapter.")
    var structuredPromptMaxTokens: Int = StructuredImagePromptAdapter.defaultMaxTokens

    @Option(name: [.customLong("structured-prompt-output")], help: "Write the generated structured JSON caption to this path.")
    var structuredPromptOutput: String?

    @Option(
        name: [.customShort("l"), .customLong("lora")],
        help: "LoRA as PATH_OR_ID[=SCALE]. Repeat to stack FLUX.1 or FLUX.2 adapters."
    )
    var loraArguments: [String] = []

    var lora: String? { loraArguments.first }

    @Option(name: [.long], help: "Default scale for --lora values without an inline scale.")
    var loraScale: Double = 1.0

    @Option(
        name: [.customLong("krea-conditioning-multiplier")],
        help: "Experimental Krea 2 text-conditioning multiplier, applied before denoising."
    )
    var kreaConditioningMultiplier: Double?

    @Option(
        name: [.customLong("krea-conditioning-layer-weights")],
        help: "Experimental comma-separated Krea 2 selected-layer weights, e.g. 1,1,1,1,1,1,1,2.5,5,1.1,4,1."
    )
    var kreaConditioningLayerWeights: String?

    @Option(
        name: [.customLong("krea-base-quantization-bits")],
        help: "Quantize the frozen Krea transformer to 4 or 8 bits and load generation phases sequentially."
    )
    var kreaBaseQuantizationBits: Int?

    @Flag(name: [.customLong("preflight")], help: "Inspect the image generation request without running generation.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "With --preflight, emit a structured JSON report.")
    var json: Bool = false

    @Flag(name: [.short, .long], help: "Print only the output path.")
    var quiet: Bool = false

    @Flag(name: [.customLong(CLIGenerationProgressPrinter.flagName)], help: CLIGenerationProgressPrinter.flagHelp)
    var progressJson: Bool = false

    @Flag(name: [.customLong(RunReceipt.flagName)], help: RunReceipt.flagHelp)
    var receipt: Bool = false

    @Option(name: [.customLong("run-dir")], help: "Create a new durable image run directory with settings, inputs, and output.")
    var runDirectory: String?

    func validate() throws {
        try RunReceipt.validate(receipt: receipt, preflight: preflight)
        if let runDirectory, let structuredPromptOutput {
            let root = URL(fileURLWithPath: runDirectory).standardizedFileURL.resolvingSymlinksInPath().path
            let target = URL(fileURLWithPath: structuredPromptOutput).standardizedFileURL.resolvingSymlinksInPath().path
            if target == root || target.hasPrefix(root + "/") {
                throw ValidationError("Write --structured-prompt-output outside --run-dir. The expanded prompt is included in the run record.")
            }
        }
    }

    func run() async throws {
        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-image", defaultExtension: "png")
        let options = try operationOptions(outputURL: outputURL)
        if let issue = ImageGenerationPlan.issues(options).first { throw ValidationError(issue.message) }
        if preflight {
            try runPreflight(outputURL: outputURL)
            return
        }

        let recording = try runDirectory.map {
            try ImageRunSession(directory: URL(fileURLWithPath: $0), requested: options,
                                modelSelector: model ?? Self.defaultManagedModelID.rawValue)
        }
        do {
            try await generate(options: options, outputURL: outputURL, recording: recording)
        } catch {
            try recording?.fail(error)
            throw error
        }
    }

    private func generate(options suppliedOptions: ImageGenerationOptions, outputURL: URL, recording: ImageRunSession?) async throws {
        var options = suppliedOptions
        let modelRoot = try resolveModelRoot()
        let manifest = try MereRunModelManifest.loadRequired(from: modelRoot)
        let initialPlan = try ImageGenerationPlan.resolve(options, modelRoot: modelRoot, manifest: manifest)
        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        let runEventLogger = try makeRunEventLoggerIfNeeded(
            outputURL: outputURL,
            modelRoot: modelRoot,
            modelManifest: manifest,
            effectiveSteps: initialPlan.request.steps,
            effectiveCFGScale: initialPlan.request.guidanceScale,
            effectiveSigmaShift: initialPlan.request.sigmaShift,
            effectiveSigmas: initialPlan.request.sigmas,
            inputMode: Self.inputMode(
                family: manifest.family,
                inputImage: options.inputImage,
                referenceImages: options.referenceImages
            )
        )

        do {
            var effectivePrompt = prompt
            var effectiveMaxSequenceLength = maxSequenceLength
            if structuredPrompt {
                if !quiet {
                    let backend = StructuredImagePromptAdapter.backendDescription(for: structuredPromptModel)
                    CLIStderr.write("[structured-prompt] Expanding prompt with \(structuredPromptModel) (\(backend))...\n")
                }
                let adapterProgressHandler: (@Sendable (String) -> Void)?
                if quiet {
                    adapterProgressHandler = nil
                } else {
                    adapterProgressHandler = { message in
                        CLIStderr.write("[structured-prompt] \(message)\n")
                    }
                }
                effectivePrompt = try await Self.expandStructuredPromptWithFallback(
                    prompt: prompt,
                    modelID: structuredPromptModel,
                    modelRoot: structuredPromptModelRoot,
                    maxTokens: structuredPromptMaxTokens,
                    progressHandler: adapterProgressHandler
                )
                effectiveMaxSequenceLength = max(
                    effectiveMaxSequenceLength,
                    StructuredImagePromptAdapter.recommendedImagePromptTokens
                )
                if let structuredPromptOutput {
                    let jsonURL = URL(fileURLWithPath: structuredPromptOutput).standardizedFileURL
                    try FileManager.default.createDirectory(
                        at: jsonURL.deletingLastPathComponent(),
                        withIntermediateDirectories: true
                    )
                    try Data(effectivePrompt.utf8).write(to: jsonURL)
                    if !quiet {
                        CLIStderr.write("[structured-prompt] JSON: \(jsonURL.path)\n")
                    }
                }
            }

            options.prompt = effectivePrompt
            options.maxSequenceLength = effectiveMaxSequenceLength
            let plan = try ImageGenerationPlan.resolve(options, modelRoot: modelRoot, manifest: manifest)

            let progressHandler: (@Sendable (GenerationProgress) -> Void)?
            if progressJson {
                progressHandler = CLIGenerationProgressPrinter.makeJSONProgressHandler()
            } else if quiet {
                progressHandler = nil
            } else {
                progressHandler = CLIGenerationProgressPrinter.makeProgressHandler()
            }
            if !quiet {
                CLIStderr.write("[runtime] image backend: \(NativeMLXRuntime.backendDescription)\n")
            }

            let outcome = try await ImageGenerationOperation.execute(plan, recording: recording, progressHandler: progressHandler)
            let result = outcome.result

            try runEventLogger?.record(
                type: "run_finished",
                stage: "finished",
                step: plan.request.steps,
                totalSteps: plan.request.steps,
                fraction: 1,
                path: result.outputURL.path
            )
            // stdout: machine-readable path (easy for scripts)
            print(result.outputURL.path)
            try RunReceipt.emit(
                RunReceipt.generatedImageOutputs(
                    image: result.outputURL,
                    structuredPrompt: structuredPrompt ? structuredPromptOutput.map { URL(fileURLWithPath: $0) } : nil
                ),
                enabled: receipt
            )
        } catch {
            try? runEventLogger?.record(
                type: "run_failed",
                stage: "failed",
                message: error.localizedDescription,
                path: outputURL.path
            )
            throw error
        }
    }

    private func resolveModelRoot() throws -> URL {
        let selection = ImageGenerationModelSelection(model ?? Self.defaultManagedModelID.rawValue)
        switch selection {
        case .local(let url): return url
        case .managed(let id):
            do { return try selection.resolveRoot() } catch {
                let prefix = model == nil ? "Image model" : "Model"
                throw ValidationError(
                    "\(prefix) \(id.rawValue) not found. Pull it with `\(CLICommandDisplay.modelPullCommand(for: id.rawValue))` or point --model at a local path."
                )
            }
        case .unknown(let selector):
            throw ValidationError("Model path not found: \(selector). Pass a local model path or a known model id.")
        }
    }

    func operationOptions(outputURL: URL) throws -> ImageGenerationOptions {
        func url(_ path: String) -> URL { URL(fileURLWithPath: path).standardizedFileURL }
        return try ImageGenerationOptions(
            prompt: prompt, negativePrompt: negativePrompt, outputURL: outputURL,
            width: width, height: height, steps: steps, guidanceScale: cfgScale, seed: seed,
            inputImage: input.map(url), referenceImages: referenceImages.map(url), strength: strength,
            keepOriginalAspect: keepOriginalAspect, maxSequenceLength: maxSequenceLength,
            loras: Self.parseLoRAArguments(loraArguments, defaultScale: loraScale),
            sigmaShift: sigmaShift.map(Float.init), sigmas: Self.parseSigmaList(sigmaList),
            kreaConditioningRebalance: Self.resolveKreaConditioningRebalance(
                multiplier: kreaConditioningMultiplier, layerWeights: kreaConditioningLayerWeights
            ),
            kreaBaseQuantizationBits: kreaBaseQuantizationBits,
            mask: mask.map(url), outpaint: outpaint.map(ImageOutpaintInsets.parse), maskFeather: maskFeather
        )
    }

    typealias LoRAArgument = ImageLoRAReference

    static func parseLoRAArguments(_ arguments: [String], defaultScale: Double) throws -> [LoRAArgument] {
        try ImageLoRAReference.parse(arguments, defaultScale: defaultScale)
    }

    static func resolveLoRAs(
        _ arguments: [LoRAArgument], baseModelID: String, fileManager: FileManager = .default
    ) throws -> [LoRA] {
        try ImageGenerationPlan.resolveLoRAs(arguments, baseModelID: baseModelID, fileManager: fileManager)
    }

    static func parseSigmaList(_ raw: String?) throws -> [Float]? {
        try ImageGenerationSampling.parseSigmas(raw)
    }

    func makePreflightEnvelope(
        outputURL: URL,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init,
        resourceDiagnostics: [PreflightDiagnostic] = []
    ) -> ImageGenerationPreflightEnvelope {
        let input = ImageGenerationPreflightInput(
            operationOptions: Result { try operationOptions(outputURL: outputURL) },
            prompt: prompt,
            negativePrompt: negativePrompt,
            outputURL: outputURL,
            width: width,
            height: height,
            steps: steps,
            seed: seed,
            model: model,
            input: input,
            mask: mask,
            outpaint: outpaint,
            maskFeather: maskFeather,
            referenceImages: referenceImages,
            keepOriginalAspect: keepOriginalAspect,
            strength: strength,
            cfgScale: cfgScale,
            sigmaShift: sigmaShift,
            sigmaList: sigmaList,
            maxSequenceLength: maxSequenceLength,
            structuredPrompt: structuredPrompt,
            structuredPromptModel: structuredPromptModel,
            structuredPromptModelRoot: structuredPromptModelRoot,
            structuredPromptMaxTokens: structuredPromptMaxTokens,
            structuredPromptOutput: structuredPromptOutput,
            loras: loraArguments,
            loraScale: loraScale,
            kreaConditioningMultiplier: kreaConditioningMultiplier,
            kreaConditioningLayerWeights: kreaConditioningLayerWeights,
            kreaBaseQuantizationBits: kreaBaseQuantizationBits,
            generationArgv: generationActionArguments(outputURL: outputURL),
            cwd: fileManager.currentDirectoryPath,
            runDirectory: runDirectory
        )
        return ImageGenerationPreflightAnalyzer(
            input: input,
            fileManager: fileManager,
            now: now
        ).envelope(resourceDiagnostics: resourceDiagnostics)
    }

    private func runPreflight(outputURL: URL) throws {
        let envelope = makePreflightEnvelope(
            outputURL: outputURL,
            resourceDiagnostics: MachineInferencePreflight.diagnostics(
                arguments: generationActionArguments(outputURL: outputURL)
            )
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

    private func generationActionArguments(outputURL: URL) -> [String] {
        var args = ["mere.run", "image", "generate", "--prompt", prompt, "--output", outputURL.path]
        if let runDirectory { args += ["--run-dir", runDirectory] }
        args += ["--width", String(width), "--height", String(height)]
        if let negativePrompt {
            args += ["--negative-prompt", negativePrompt]
        }
        if let cfgScale {
            args += ["--cfg", String(cfgScale)]
        }
        if let sigmaShift {
            args += ["--sigma-shift", String(sigmaShift)]
        }
        if let sigmaList {
            args += ["--sigmas", sigmaList]
        }
        if let steps {
            args += ["--steps", String(steps)]
        }
        if let seed {
            args += ["--seed", String(seed)]
        }
        if let model {
            args += ["--model", model]
        }
        if let input {
            args += ["--input", input]
        }
        if let mask {
            args += ["--mask", mask]
        }
        if let outpaint {
            args += ["--outpaint", outpaint]
        }
        if maskFeather != 8 {
            args += ["--mask-feather", String(maskFeather)]
        }
        for referenceImage in referenceImages {
            args += ["--ref-image", referenceImage]
        }
        if keepOriginalAspect {
            args.append("--keep-original-aspect")
        }
        if let strength {
            args += ["--strength", String(strength)]
        }
        if maxSequenceLength != 512 {
            args += ["--max-sequence-length", String(maxSequenceLength)]
        }
        if structuredPrompt {
            args.append("--structured-prompt")
        }
        if structuredPromptModel != StructuredImagePromptAdapter.defaultModelID {
            args += ["--structured-prompt-model", structuredPromptModel]
        }
        if let structuredPromptModelRoot {
            args += ["--structured-prompt-model-root", structuredPromptModelRoot]
        }
        if structuredPromptMaxTokens != StructuredImagePromptAdapter.defaultMaxTokens {
            args += ["--structured-prompt-max-tokens", String(structuredPromptMaxTokens)]
        }
        if let structuredPromptOutput {
            args += ["--structured-prompt-output", structuredPromptOutput]
        }
        for lora in loraArguments {
            args += ["--lora", lora]
        }
        if loraScale != 1.0 {
            args += ["--lora-scale", String(loraScale)]
        }
        if let kreaConditioningMultiplier {
            args += ["--krea-conditioning-multiplier", String(kreaConditioningMultiplier)]
        }
        if let kreaConditioningLayerWeights {
            args += ["--krea-conditioning-layer-weights", kreaConditioningLayerWeights]
        }
        if let kreaBaseQuantizationBits {
            args += ["--krea-base-quantization-bits", String(kreaBaseQuantizationBits)]
        }
        if quiet {
            args.append("--quiet")
        }
        return args
    }

    func materializedPlanURL(for outputURL: URL) -> URL? {
        let planURL = outputURL.deletingLastPathComponent().appendingPathComponent("plan.json", isDirectory: false)
        guard FileManager.default.fileExists(atPath: planURL.path),
              let plan = try? ImageGenerationRunPlan.decode(from: planURL),
              URL(fileURLWithPath: plan.arguments.output).standardizedFileURL.path == outputURL.standardizedFileURL.path else {
            return nil
        }
        return planURL
    }

    func makeRunEventLoggerIfNeeded(
        outputURL: URL,
        modelRoot: URL,
        modelManifest: MereRunModelManifest,
        effectiveSteps: Int,
        effectiveCFGScale: Double,
        effectiveSigmaShift: Float?,
        effectiveSigmas: [Float]? = nil,
        inputMode: String
    ) throws -> LoRATrainingEventLogger? {
        guard let materializedPlanURL = materializedPlanURL(for: outputURL) else {
            return nil
        }
        let logger = try LoRATrainingEventLogger(
            baseOutputURL: outputURL,
            resumeExisting: true
        )
        var metadata = [
            "model_root": modelRoot.path,
            "model_id": modelManifest.id,
            "model_family": modelManifest.family?.rawValue ?? "unknown",
            "output": outputURL.path,
            "prompt": prompt,
            "width": String(width),
            "height": String(height),
            "steps": String(effectiveSteps),
            "cfg": String(effectiveCFGScale),
            "input_mode": inputMode,
            "plan_file": materializedPlanURL.lastPathComponent,
            "actions_file": "actions.json",
        ]
        if let seed {
            metadata["seed"] = String(seed)
        }
        if let effectiveSigmaShift {
            metadata["sigma_shift"] = String(effectiveSigmaShift)
        }
        if let effectiveSigmas {
            metadata["sigmas"] = effectiveSigmas.map { String($0) }.joined(separator: ",")
        }
        if let input {
            metadata["input"] = input
        }
        if !referenceImages.isEmpty {
            metadata["reference_images"] = referenceImages.joined(separator: "\n")
        }
        if !loraArguments.isEmpty {
            metadata["loras"] = loraArguments.joined(separator: "\n")
            metadata["lora_default_scale"] = String(loraScale)
        }
        try logger.record(
            type: "run_started",
            stage: "starting",
            message: "Image generation started.",
            step: 0,
            totalSteps: effectiveSteps,
            fraction: 0,
            path: outputURL.path,
            metadata: metadata
        )
        return logger
    }

    static func expandStructuredPromptWithFallback(
        prompt: String,
        modelID: String,
        modelRoot: String?,
        maxTokens: Int,
        progressHandler: (@Sendable (String) -> Void)?
    ) async throws -> String {
        do {
            return try await StructuredImagePromptAdapter.expand(
                prompt: prompt,
                modelID: modelID,
                modelRoot: modelRoot,
                maxTokens: maxTokens,
                progressHandler: progressHandler
            )
        } catch {
            guard isStructuredPromptOutputFailure(error) else {
                throw error
            }
            progressHandler?("adapter output rejected; trying Gemma text fallback")
        }

        let defaultModelID = StructuredImagePromptAdapter.defaultModelID
        if modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() != defaultModelID {
            do {
                return try await StructuredImagePromptAdapter.expand(
                    prompt: prompt,
                    modelID: defaultModelID,
                    modelRoot: nil,
                    maxTokens: maxTokens,
                    progressHandler: progressHandler
                )
            } catch {
                progressHandler?("Gemma text fallback unavailable; using deterministic structured prompt")
            }
        } else {
            progressHandler?("using deterministic structured prompt fallback")
        }

        return try StructuredImagePromptAdapter.deterministicCaptionJSON(for: prompt)
    }

    private static func isStructuredPromptOutputFailure(_ error: Error) -> Bool {
        guard let adapterError = error as? StructuredImagePromptAdapterError else { return false }
        switch adapterError {
        case .invalidCaptionJSON:
            return true
        case .invalidMaxTokens, .invalidModelRoot:
            return false
        }
    }

    typealias ConditioningInputs = ImageGenerationConditioning

    static func resolveConditioningInputs(
        family: MereRunModelManifest.Family?, inputImage: URL?, referenceImages: [URL], strength: Double?
    ) -> ConditioningInputs {
        ImageGenerationConditioning.resolve(family: family, inputImage: inputImage, referenceImages: referenceImages, strength: strength)
    }

    static func inputMode(family: MereRunModelManifest.Family?, inputImage: URL?, referenceImages: [URL]) -> String {
        ImageGenerationConditioning.inputMode(family: family, inputImage: inputImage, referenceImages: referenceImages)
    }

    static func resolveKreaConditioningRebalance(
        multiplier: Double?,
        layerWeights rawLayerWeights: String?
    ) throws -> Krea2ConditioningRebalance? {
        guard multiplier != nil || rawLayerWeights != nil else { return nil }

        let resolvedMultiplier = multiplier ?? 1.0
        guard resolvedMultiplier.isFinite else {
            throw ValidationError("--krea-conditioning-multiplier must be finite")
        }

        let layerWeights = try parseKreaConditioningLayerWeights(rawLayerWeights)
        return Krea2ConditioningRebalance(
            multiplier: Float(resolvedMultiplier),
            layerWeights: layerWeights.map(Float.init)
        )
    }

    static func parseKreaConditioningLayerWeights(_ raw: String?) throws -> [Double] {
        guard let raw else { return [] }

        let normalized = raw.replacingOccurrences(of: ";", with: ",")
        let parts = normalized.split(separator: ",", omittingEmptySubsequences: true)
        guard !parts.isEmpty else {
            throw ValidationError("--krea-conditioning-layer-weights must include at least one number")
        }

        return try parts.map { part in
            let trimmed = part.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let value = Double(trimmed), value.isFinite else {
                throw ValidationError("Invalid --krea-conditioning-layer-weights value: \(trimmed)")
            }
            return value
        }
    }
}
