import ArgumentParser
import Foundation
import MereRunCore

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

    func validate() throws {
        try RunReceipt.validate(receipt: receipt, preflight: preflight)
    }

    func run() async throws {
        try validateStaticOptions()
        let kreaConditioningRebalance = try Self.resolveKreaConditioningRebalance(
            multiplier: kreaConditioningMultiplier,
            layerWeights: kreaConditioningLayerWeights
        )
        let explicitSigmas = try Self.parseSigmaList(sigmaList)
        let parsedLoRAs = try Self.parseLoRAArguments(
            loraArguments,
            defaultScale: loraScale
        )

        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-image", defaultExtension: "png")

        if preflight {
            try runPreflight(outputURL: outputURL)
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let inputURL: URL?
        if let input {
            let url = URL(fileURLWithPath: input).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("Input image not found: \(url.path)")
            }
            inputURL = url
        } else {
            inputURL = nil
        }
        let maskURL: URL?
        if let mask {
            let url = URL(fileURLWithPath: mask).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("Mask image not found: \(url.path)")
            }
            maskURL = url
        } else {
            maskURL = nil
        }
        let outpaintInsets = try outpaint.map(ImageOutpaintInsets.parse)
        let editPreparation: ImageEditPreparation?
        if maskURL != nil || outpaintInsets != nil {
            guard let inputURL else {
                throw ValidationError("--mask and --outpaint require --input.")
            }
            editPreparation = try ImageEditPreparation.make(
                inputURL: inputURL,
                maskURL: maskURL,
                outpaint: outpaintInsets,
                width: width,
                height: height,
                featherPixels: maskFeather
            )
        } else {
            editPreparation = nil
        }
        defer { editPreparation?.cleanup() }
        let effectiveInputURL = editPreparation?.generationInputURL ?? inputURL

        let referenceImageURLs = try referenceImages.map { path in
            let url = URL(fileURLWithPath: path).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("Reference image not found: \(url.path)")
            }
            return url
        }

        let resolvedModel: String?
        let resolver = ModelResolver()

        if let model {
            let url = URL(fileURLWithPath: model).standardizedFileURL
            if FileManager.default.fileExists(atPath: url.path) {
                resolvedModel = url.path
            } else if let id = ModelResolver.ModelID(rawValue: model) {
                do {
                    resolvedModel = try resolver.resolve(id).rootURL.path
                } catch {
                    throw ValidationError(
                        "Model \(id.rawValue) not found. Pull it with `\(CLICommandDisplay.modelPullCommand(for: id.rawValue))` or point --model at a local path."
                    )
                }
            } else {
                throw ValidationError(
                    "Model path not found: \(model). Pass a local model path or a known model id."
                )
            }
        } else {
            do {
                resolvedModel = try resolver.resolve(Self.defaultManagedModelID).rootURL.path
            } catch {
                let modelID = Self.defaultManagedModelID.rawValue
                throw ValidationError(
                    "Image model \(modelID) not found. Pull it with `\(CLICommandDisplay.modelPullCommand(for: modelID))` or point --model at a local path."
                )
            }
        }

        let manifest = try MereRunModelManifest.loadRequired(from: URL(fileURLWithPath: resolvedModel!))
        if parsedLoRAs.count > 1, manifest.family != .klein, manifest.family != .flux1 {
            throw ValidationError("Stacked image LoRAs are supported only for FLUX.1 and FLUX.2 models.")
        }
        if explicitSigmas != nil, manifest.family != .klein {
            throw ValidationError("--sigmas is supported only for FLUX.2 models.")
        }
        let loraConfigs = try Self.resolveLoRAs(
            parsedLoRAs,
            baseModelID: manifest.id
        )
        let usesTurboRecipe = parsedLoRAs.contains {
            ManagedAdapterCatalog.spec(for: $0.reference)?.id
                == ManagedAdapterCatalog.flux2DevTurboEightStepID
        }
        let effectiveSigmas = explicitSigmas
            ?? (usesTurboRecipe ? Flux2DevTurboRecipe.sigmas : nil)
        if kreaBaseQuantizationBits != nil, manifest.family != .krea {
            throw ValidationError("--krea-base-quantization-bits is only supported for Krea 2 generation")
        }
        let conditioning = Self.resolveConditioningInputs(
            family: manifest.family,
            inputImage: effectiveInputURL,
            referenceImages: referenceImageURLs,
            strength: strength
        )
        let usesManifestImageDefaults = manifest.engine == .qwenImageEdit
            || manifest.family == .hidream || manifest.family == .senseNova
            || manifest.family == .krea || manifest.family == .ideogram
            || manifest.family == .klein || manifest.family == .flux1
        let effectiveSteps = steps
            ?? effectiveSigmas?.count
            ?? (usesManifestImageDefaults
                ? (manifest.defaults?.steps ?? 4)
                : 4)
        if let effectiveSigmas, effectiveSigmas.count != effectiveSteps {
            throw ValidationError(
                "--steps must equal the number of non-terminal --sigmas values "
                    + "(\(effectiveSigmas.count))."
            )
        }
        let effectiveCFG = cfgScale
            ?? (usesTurboRecipe ? Flux2DevTurboRecipe.guidanceScale : nil)
            ?? (usesManifestImageDefaults
                ? (manifest.defaults?.cfg ?? 1.0)
                : 1.0)
        let effectiveSigmaShift = sigmaShift.map { Float($0) }
            ?? manifest.defaults?.sigmaShift.map { Float($0) }

        let runEventLogger = try makeRunEventLoggerIfNeeded(
            outputURL: outputURL,
            modelRoot: URL(fileURLWithPath: resolvedModel!),
            modelManifest: manifest,
            effectiveSteps: effectiveSteps,
            effectiveCFGScale: effectiveCFG,
            effectiveSigmaShift: effectiveSigmaShift,
            effectiveSigmas: effectiveSigmas,
            inputMode: Self.inputMode(
                family: manifest.family,
                inputImage: inputURL,
                referenceImages: referenceImageURLs
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

            let request = GenerationRequest(
                prompt: effectivePrompt,
                negativePrompt: negativePrompt,
                referenceImages: conditioning.referenceImages,
                referenceStrength: conditioning.referenceStrength,
                width: width,
                height: height,
                steps: effectiveSteps,
                guidanceScale: effectiveCFG,
                seed: seed,
                outputURL: outputURL,
                model: resolvedModel,
                maxSequenceLength: effectiveMaxSequenceLength,
                lora: loraConfigs.first,
                loras: loraConfigs,
                enhancePrompt: false,
                inputImage: conditioning.inputImage,
                strength: conditioning.strength,
                keepOriginalAspect: keepOriginalAspect,
                useBetaSigmas: false,
                sigmaShift: effectiveSigmaShift,
                sigmas: effectiveSigmas,
                kreaConditioningRebalance: kreaConditioningRebalance,
                kreaBaseQuantizationBits: kreaBaseQuantizationBits
            )

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

            let result: GenerationResult
            switch manifest.family {
            case .flux1:
                let generator = Flux1Generator()
                result = try await generator.generate(request, progressHandler: progressHandler)
            case .klein:
                let generator = Flux2KleinGenerator()
                result = try await generator.generate(request, progressHandler: progressHandler)
            case .zimage:
                let generator = ZImageTurboGenerator()
                result = try await generator.generate(request, progressHandler: progressHandler)
            case .hidream:
                let generator = HiDreamO1Generator()
                defer { generator.unload() }
                result = try await generator.generate(request, progressHandler: progressHandler)
            case .senseNova:
                let generator = SenseNovaU15Generator()
                defer { generator.unload() }
                result = try await generator.generate(request, progressHandler: progressHandler)
            case .krea:
                let generator = Krea2Generator()
                defer { generator.unload() }
                result = try await generator.generate(request, progressHandler: progressHandler)
            case .ideogram:
                let generator = Ideogram4Generator()
                defer { generator.unload() }
                result = try await generator.generate(request, progressHandler: progressHandler)
            case .qwen where manifest.engine == .qwenImageEdit:
                let generator = QwenImageEditGenerator()
                result = try await generator.generate(request, progressHandler: progressHandler)
            case .gemma, .laguna, .liquid, .qwen, .sam, .falcon, .terramind, .tessera, .olmoEarth,
                 .face, .geometry, .depth, .threeD,
                 .tts, .asr, .embed, .code, .ocr, .audio, .music, .sfx, .video, .psi, .privacy, .deepseek,
                 .inkling, .muse, .nemotron, nil:
                throw ValidationError("Unsupported image model family for `mere.run image generate`: \(manifest.id)")
            }
            try editPreparation?.finish(generatedURL: result.outputURL)

            try runEventLogger?.record(
                type: "run_finished",
                stage: "finished",
                step: effectiveSteps,
                totalSteps: effectiveSteps,
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

    private func validateStaticOptions() throws {
        if let steps, steps <= 0 {
            throw ValidationError("--steps must be >= 1")
        }
        guard width > 0, height > 0 else {
            throw ValidationError("--width/--height must be > 0")
        }
        if let strength, !(0.0...1.0).contains(strength) {
            throw ValidationError("--strength must be between 0.0 and 1.0")
        }
        if (mask != nil || outpaint != nil), input == nil {
            throw ValidationError("--mask and --outpaint require --input")
        }
        if maskFeather < 0 {
            throw ValidationError("--mask-feather must be >= 0")
        }
        if let outpaint {
            _ = try ImageOutpaintInsets.parse(outpaint)
        }
        if let kreaBaseQuantizationBits, kreaBaseQuantizationBits != 4, kreaBaseQuantizationBits != 8 {
            throw ValidationError("--krea-base-quantization-bits must be 4 or 8")
        }
        guard loraScale.isFinite else {
            throw ValidationError("--lora-scale must be finite")
        }
        if sigmaList != nil, sigmaShift != nil {
            throw ValidationError("--sigmas cannot be combined with --sigma-shift")
        }
    }

    struct LoRAArgument: Equatable {
        let raw: String
        let reference: String
        let scale: Double
    }

    static func parseLoRAArguments(
        _ arguments: [String],
        defaultScale: Double
    ) throws -> [LoRAArgument] {
        guard defaultScale.isFinite else {
            throw ValidationError("--lora-scale must be finite")
        }
        return try arguments.map { raw in
            let separator = raw.lastIndex(of: "=")
            let reference: String
            let scale: Double
            if let separator {
                reference = String(raw[..<separator])
                let rawScale = String(raw[raw.index(after: separator)...])
                guard let parsed = Double(rawScale), parsed.isFinite else {
                    throw ValidationError("--lora scale must be finite (got \(rawScale)).")
                }
                scale = parsed
            } else {
                reference = raw
                scale = defaultScale
            }
            guard !reference.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ValidationError("--lora must be PATH_OR_ID[=SCALE].")
            }
            return LoRAArgument(raw: raw, reference: reference, scale: scale)
        }
    }

    static func resolveLoRAs(
        _ arguments: [LoRAArgument],
        baseModelID: String,
        fileManager: FileManager = .default
    ) throws -> [LoRA] {
        var resolvedPaths = Set<String>()
        return try arguments.map { argument in
            let resolved = try ManagedAdapterArgumentResolver.resolve(
                argument.reference,
                baseModelID: baseModelID,
                fileManager: fileManager
            ) ?? argument.reference
            let url = URL(fileURLWithPath: resolved).standardizedFileURL
            guard fileManager.fileExists(atPath: url.path) else {
                throw ValidationError("LoRA file not found: \(url.path)")
            }
            guard resolvedPaths.insert(url.path).inserted else {
                throw ValidationError("Duplicate LoRA adapter: \(url.path)")
            }
            return .local(path: url.path, scale: argument.scale)
        }
    }

    static func parseSigmaList(_ raw: String?) throws -> [Float]? {
        guard let raw else { return nil }
        var values = try raw.split(separator: ",", omittingEmptySubsequences: false).map { item in
            let trimmed = item.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let value = Float(trimmed), value.isFinite else {
                throw ValidationError("--sigmas must contain finite comma-separated values.")
            }
            return value
        }
        if values.last == 0 {
            values.removeLast()
        }
        guard !values.isEmpty else {
            throw ValidationError("--sigmas must include at least one non-terminal value.")
        }
        do {
            try Flux2EulerScheduler.validateCustomSigmas(values, expectedSteps: values.count)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        return values
    }

    func makePreflightEnvelope(
        outputURL: URL,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) -> ImageGenerationPreflightEnvelope {
        let input = ImageGenerationPreflightInput(
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
            cwd: fileManager.currentDirectoryPath
        )
        return ImageGenerationPreflightAnalyzer(
            input: input,
            fileManager: fileManager,
            now: now
        ).envelope()
    }

    private func runPreflight(outputURL: URL) throws {
        let envelope = makePreflightEnvelope(outputURL: outputURL)
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

    struct ConditioningInputs: Equatable {
        var inputImage: URL?
        var referenceImages: [URL]
        var strength: Double
        var referenceStrength: Double
    }

    static func resolveConditioningInputs(
        family: MereRunModelManifest.Family?,
        inputImage: URL?,
        referenceImages: [URL],
        strength: Double?
    ) -> ConditioningInputs {
        let explicitStrength = strength
        let defaultInputStrength = 0.75

        guard family == .klein else {
            return ConditioningInputs(
                inputImage: inputImage,
                referenceImages: referenceImages,
                strength: explicitStrength ?? defaultInputStrength,
                referenceStrength: 0.0
            )
        }

        var resolvedReferences = referenceImages
        if let inputImage {
            resolvedReferences.insert(inputImage, at: 0)
        }

        return ConditioningInputs(
            inputImage: nil,
            referenceImages: resolvedReferences,
            strength: explicitStrength ?? defaultInputStrength,
            referenceStrength: explicitStrength ?? (inputImage == nil ? 0.0 : defaultInputStrength)
        )
    }

    static func inputMode(
        family: MereRunModelManifest.Family?,
        inputImage: URL?,
        referenceImages: [URL]
    ) -> String {
        let hasInput = inputImage != nil
        let hasReferences = !referenceImages.isEmpty
        guard hasInput || hasReferences else { return "text_to_image" }
        if family == .klein {
            return "reference_image"
        }
        if hasInput && hasReferences {
            return "image_to_image_with_references"
        }
        if hasInput {
            return "image_to_image"
        }
        return "reference_image"
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
