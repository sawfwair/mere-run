import ArgumentParser
import Foundation
import MereRunCore

struct ImageTrainLoRA: AsyncParsableCommand {
    static let defaultManagedModelID: ModelResolver.ModelID = .krea2Raw
    private static let defaultWidth = 1024
    private static let defaultHeight = 1024
    private static let defaultTrainingSteps = 1000
    private static let defaultLearningRate: Float = 1e-4
    private static let defaultRank = 16
    private static let defaultCaptionDropout: Float = 0.05

    static let configuration = CommandConfiguration(
        commandName: "train-lora",
        abstract: "Train a local image LoRA adapter.",
        discussion: """
        Krea 2 LoRAs are trained on image-krea2-raw and can be used with image-krea2-turbo via image generate --lora.
        FLUX.2 Klein LoRAs are trained on a Klein base model, then loaded on distilled Klein models for practical inference. If distilled output drifts, compare against base/checkpoint previews and match the sampling recipe.
        Prints the output LoRA path to stdout. Progress and diagnostics are printed to stderr.
        """
    )

    @Option(name: [.customShort("d"), .long], help: "Dataset directory containing image files with matching .txt captions.")
    var data: String?

    @Option(name: [.customShort("o"), .long], help: "Output .safetensors path.")
    var output: String

    @Option(name: [.customShort("m"), .long], help: "Raw/base model path or canonical model id (default: image-krea2-raw).")
    var model: String?

    @Option(name: [.customShort("W"), .customLong("width")], help: "Training image width in pixels.")
    var widthOverride: Int?

    var width: Int {
        widthOverride ?? Self.defaultWidth
    }

    @Option(name: [.customShort("H"), .customLong("height")], help: "Training image height in pixels.")
    var heightOverride: Int?

    var height: Int {
        heightOverride ?? Self.defaultHeight
    }

    @Option(name: [.customLong("training-steps"), .customLong("steps")], help: "Number of optimizer steps.")
    private var trainingStepsOverride: Int?

    var trainingSteps: Int {
        trainingStepsOverride ?? Self.defaultTrainingSteps
    }

    @Option(name: [.long], help: "Batch size.")
    var batchSize: Int = 1

    @Option(name: [.customLong("learning-rate"), .customLong("lr")], help: "Learning rate.")
    private var learningRateOverride: Float?

    var learningRate: Float {
        learningRateOverride ?? Self.defaultLearningRate
    }

    @Option(name: [.customLong("rank")], help: "LoRA rank.")
    var rankOverride: Int?

    var rank: Int {
        rankOverride ?? Self.defaultRank
    }

    @Option(name: [.customLong("alpha")], help: "LoRA alpha. Defaults to rank.")
    var alphaOverride: Float?

    var alpha: Float? {
        alphaOverride
    }

    @Option(name: [.customLong("max-text-length")], help: "Maximum prompt token length.")
    var maxTextLength: Int = 512

    @Option(name: [.customLong("scheduler-steps")], help: "Number of FlowMatch training timesteps.")
    var schedulerSteps: Int = 1000

    @Option(name: [.customLong("caption-dropout")], help: "Caption dropout probability between 0.0 and 1.0.")
    var captionDropoutOverride: Float?

    var captionDropout: Float {
        captionDropoutOverride ?? Self.defaultCaptionDropout
    }

    @Option(name: [.long], help: "Random seed. Defaults to wall-clock time when omitted or zero.")
    var seed: UInt64 = 0

    @Flag(name: [.customLong("lite")], help: "Train only attention Q/V LoRA layers to reduce memory.")
    var lite: Bool = false

    @Flag(name: [.customLong("exclude-preview-images")], help: "Ignore preview*.png/jpg/webp images in the dataset folder.")
    var excludePreviewImages: Bool = false

    @Option(
        name: [.customLong("checkpoint-interval")],
        help: "Save intermediate Klein LoRA checkpoints every N steps."
    )
    var checkpointInterval: Int?

    @Option(name: [.customLong("max-resolution")], help: "Klein adaptive source-image bucket limit; preserves aspect ratio up to this max side.")
    var maxResolution: Int?

    @Flag(name: [.customLong("progressive")], help: "Klein progressive resolution schedule up to --width/--height.")
    var progressive: Bool = false

    @Flag(name: [.customLong("low-ram")], help: "Klein disk-backed latent cache to reduce peak memory.")
    var lowRam: Bool = false

    @Flag(name: [.customLong("no-compile")], help: "Disable Krea/Klein compiled train-step graph to reduce peak GPU memory or avoid CUDA graph issues.")
    var noCompile: Bool = false

    @Flag(name: [.customLong("gradient-checkpointing")], help: "Checkpoint Klein transformer blocks during backprop to reduce peak GPU memory.")
    var gradientCheckpointing: Bool = false

    @Option(
        name: [.customLong("recipe")],
        help: "Apply a named training recipe: krea-fast-style, krea-cinematic-style, or klein-fast-style."
    )
    var recipe: String?

    @Option(name: [.customLong("benchmark-steps")], help: "Klein benchmark mode: measure N training steps after warmup, report seconds/step, and skip saving.")
    var benchmarkSteps: Int?

    @Option(name: [.customLong("benchmark-warmup-steps")], help: "Klein benchmark warmup steps before timing.")
    var benchmarkWarmupSteps: Int = 5

    @Option(name: [.customLong("sample-interval")], help: "Generate a Klein preview image every N training steps.")
    var sampleInterval: Int?

    @Option(name: [.customLong("sample-prompt")], help: "Klein preview prompt. Defaults to the first caption.")
    var samplePrompt: String?

    @Option(name: [.customLong("sample-model")], help: "Klein preview model path/id. Defaults to image-klein-9b.")
    var sampleModel: String?

    @Option(name: [.customLong("sample-steps")], help: "Klein preview inference steps.")
    var sampleSteps: Int = 8

    @Option(name: [.customLong("sample-cfg")], help: "Klein preview guidance scale.")
    var sampleGuidanceScale: Double = 1.0

    @Option(name: [.customLong("sample-lora-scale")], help: "Klein preview LoRA scale.")
    var sampleLoRAScale: Double = 1.0

    @Option(name: [.customLong("sample-seed")], help: "Klein preview seed.")
    var sampleSeed: UInt64?

    @Flag(name: [.customLong("visualize")], help: "Start a loopback LoRA training dashboard for this run.")
    var visualize: Bool = false

    @Option(name: [.customLong("visualize-port")], help: "Loopback port for --visualize.")
    var visualizePort: Int = 8787

    @Flag(name: [.customLong("preflight")], help: "Inspect the LoRA training request without running training.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "Emit a structured preflight JSON report.")
    var json: Bool = false

    @Option(name: [.customLong("lora-target-ranks")], help: "Klein suffix rank map, e.g. .attn.to_q=128,.ff.linear_in=64.")
    var loraTargetRanks: String?

    @Option(name: [.customLong("lora-rank-preset")], help: "Klein rank preset: flux2-style-128.")
    var loraRankPreset: String?

    @Option(name: [.customLong("lora-target-preset")], help: "Klein exact target preset: fal-klein-fast.")
    var loraTargetPreset: String?

    @Option(name: [.customLong("lora-target-mode")], help: "Klein target mode: suffix or transformer-linear-walk.")
    var loraTargetMode: String?

    @Option(name: [.customLong("timestep-sampling")], help: "Klein timestep sampler: uniform, bellCurve, contentFocused, styleFocused, logitNormal, or shift.")
    var timestepSampling: String?

    @Option(name: [.customLong("timestep-loss-weighting")], help: "Klein timestep loss weighting: none or weighted.")
    var timestepLossWeighting: String?

    @Option(name: [.customLong("loss-weighting")], help: "Klein loss weighting: none, snr, or minSNR.")
    var lossWeighting: String?

    @Option(name: [.customLong("timestep-low")], help: "Klein minimum sampled timestep index, inclusive.")
    var timestepLow: Int?

    @Option(name: [.customLong("timestep-high")], help: "Klein maximum sampled timestep index, exclusive.")
    var timestepHigh: Int?

    @Option(name: [.customLong("lr-warmup-steps")], help: "Krea/Klein cosine schedule warmup steps.")
    var lrWarmupSteps: Int?

    @Flag(name: [.customLong("no-cosine-scheduler")], help: "Disable Krea/Klein cosine LR scheduling.")
    var noCosineScheduler: Bool = false

    @Option(name: [.customLong("lr-min-factor")], help: "Krea/Klein cosine LR floor as a fraction of base LR.")
    var lrMinFactor: Float?

    @Option(name: [.customLong("adam-weight-decay")], help: "Klein AdamW weight decay.")
    var adamWeightDecay: Float?

    @Option(name: [.customLong("synthetic-samples")], help: "Use synthetic training samples for runtime smoke tests.")
    var syntheticSamples: Int?

    @Flag(name: [.short, .long], help: "Print only the output path.")
    var quiet: Bool = false

    func run() async throws {
        let resolvedOptions = try resolvedTrainingOptions()
        try validateResolvedOptions(resolvedOptions)

        if preflight {
            try runPreflight(options: resolvedOptions)
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        let outputURL = URL(fileURLWithPath: output).standardizedFileURL
        guard outputURL.pathExtension.lowercased() == "safetensors" else {
            throw ValidationError("--output must end in .safetensors")
        }
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let modelRoot = try resolveModelRoot(model: resolvedOptions.model)
        let modelManifest = try MereRunModelManifest.loadRequired(from: modelRoot)
        let runContext = try startRunContextIfNeeded(
            outputURL: outputURL,
            modelRoot: modelRoot,
            modelManifest: modelManifest,
            options: resolvedOptions
        )
        defer { runContext?.stop() }

        do {
            switch modelManifest.family {
            case .krea:
                try await runKreaTraining(
                    modelRoot: modelRoot,
                    outputURL: outputURL,
                    options: resolvedOptions,
                    eventLogger: runContext?.logger
                )
            case .klein:
                try await runKleinTraining(
                    modelRoot: modelRoot,
                    outputURL: outputURL,
                    options: resolvedOptions,
                    eventLogger: runContext?.logger
                )
            default:
                let family = modelManifest.family?.rawValue ?? "unknown"
                throw ValidationError("Unsupported LoRA training model family: \(family). Use a Krea 2 Raw or FLUX.2 Klein base model.")
            }
            try runContext?.logger.record(
                type: "run_finished",
                stage: "finished",
                step: resolvedOptions.trainingSteps,
                totalSteps: resolvedOptions.trainingSteps,
                fraction: 1,
                path: outputURL.path
            )
        } catch {
            try? runContext?.logger.record(
                type: "run_failed",
                stage: "failed",
                message: error.localizedDescription,
                path: outputURL.path
            )
            throw error
        }

        if benchmarkSteps == nil {
            print(outputURL.path)
        }
    }

    private func validateResolvedOptions(_ resolvedOptions: ResolvedLoRATrainingOptions) throws {
        guard resolvedOptions.width > 0,
              resolvedOptions.height > 0,
              resolvedOptions.width % 16 == 0,
              resolvedOptions.height % 16 == 0 else {
            throw ValidationError("--width/--height must be > 0 and divisible by 16")
        }
        guard resolvedOptions.trainingSteps >= 1 else {
            throw ValidationError("--training-steps must be >= 1")
        }
        guard batchSize >= 1 else {
            throw ValidationError("--batch-size must be >= 1")
        }
        guard schedulerSteps >= 1 else {
            throw ValidationError("--scheduler-steps must be >= 1")
        }
        guard resolvedOptions.rank >= 1 else {
            throw ValidationError("--rank must be >= 1")
        }
        guard (0.0...1.0).contains(resolvedOptions.captionDropout) else {
            throw ValidationError("--caption-dropout must be between 0.0 and 1.0")
        }
        if let syntheticSamples, syntheticSamples < 1 {
            throw ValidationError("--synthetic-samples must be >= 1")
        }
        if let checkpointInterval = resolvedOptions.checkpointInterval, checkpointInterval < 1 {
            throw ValidationError("--checkpoint-interval must be >= 1")
        }
        if let maxResolution = resolvedOptions.maxResolution, maxResolution < 1 {
            throw ValidationError("--max-resolution must be >= 1")
        }
        if let benchmarkSteps, benchmarkSteps < 1 {
            throw ValidationError("--benchmark-steps must be >= 1")
        }
        guard benchmarkWarmupSteps >= 0 else {
            throw ValidationError("--benchmark-warmup-steps must be >= 0")
        }
        if resolvedOptions.maxResolution != nil, progressive {
            throw ValidationError("--max-resolution cannot be combined with --progressive")
        }
        if let sampleInterval, sampleInterval < 1 {
            throw ValidationError("--sample-interval must be >= 1")
        }
        guard sampleSteps >= 1 else {
            throw ValidationError("--sample-steps must be >= 1")
        }
        guard sampleGuidanceScale >= 0 else {
            throw ValidationError("--sample-cfg must be >= 0")
        }
        guard sampleLoRAScale >= 0 else {
            throw ValidationError("--sample-lora-scale must be >= 0")
        }
        if visualize, !(1...65535).contains(visualizePort) {
            throw ValidationError("--visualize-port must be between 1 and 65535")
        }
        if loraTargetRanks != nil, loraRankPreset != nil {
            throw ValidationError("--lora-target-ranks cannot be combined with --lora-rank-preset")
        }
        if let loraTargetPreset = resolvedOptions.loraTargetPreset {
            if lite {
                throw ValidationError("--lite cannot be combined with --lora-target-preset")
            }
            if loraTargetRanks != nil || loraRankPreset != nil {
                throw ValidationError("--lora-target-preset cannot be combined with --lora-target-ranks or --lora-rank-preset")
            }
            _ = try Self.resolveKleinTargetPreset(loraTargetPreset, rank: resolvedOptions.rank)
        }
        let parsedLoRATargetMode = try Self.resolveKleinLoRATargetMode(loraTargetMode)
        if parsedLoRATargetMode == .transformerLinearWalk {
            if lite {
                throw ValidationError("--lite cannot be combined with --lora-target-mode transformer-linear-walk")
            }
            if loraTargetRanks != nil || loraRankPreset != nil || resolvedOptions.loraTargetPreset != nil {
                throw ValidationError("--lora-target-mode transformer-linear-walk cannot be combined with LoRA target/rank presets")
            }
        }
        if let timestepLow, timestepLow < 0 {
            throw ValidationError("--timestep-low must be >= 0")
        }
        if let timestepHigh, timestepHigh < 1 {
            throw ValidationError("--timestep-high must be >= 1")
        }
        if let timestepLow, let timestepHigh, timestepHigh <= timestepLow {
            throw ValidationError("--timestep-high must be greater than --timestep-low")
        }
        if let timestepHigh, timestepHigh > schedulerSteps {
            throw ValidationError("--timestep-high must be <= --scheduler-steps")
        }
        if let lrWarmupSteps, lrWarmupSteps < 0 {
            throw ValidationError("--lr-warmup-steps must be >= 0")
        }
        if let lrMinFactor, !(0.0...1.0).contains(lrMinFactor) {
            throw ValidationError("--lr-min-factor must be between 0.0 and 1.0")
        }
        if let adamWeightDecay, adamWeightDecay < 0 {
            throw ValidationError("--adam-weight-decay must be >= 0")
        }
    }

    func makePreflightEnvelope(
        options: ResolvedLoRATrainingOptions,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) -> LoRATrainingPreflightEnvelope {
        let input = LoRATrainingPreflightInput(
            data: data,
            output: output,
            recipe: recipe,
            excludePreviewImages: excludePreviewImages,
            syntheticSamples: syntheticSamples,
            requiresKleinModel: options.checkpointInterval != nil || hasKleinOnlyTrainingOptions(options: options),
            options: options,
            trainingArgv: trainingActionArguments(),
            runPlan: makeRunPlan(options: options, fileManager: fileManager, now: now),
            cwd: fileManager.currentDirectoryPath
        )
        return LoRATrainingPreflightAnalyzer(
            input: input,
            fileManager: fileManager,
            now: now
        ).envelope()
    }

    private func runPreflight(options: ResolvedLoRATrainingOptions) throws {
        let envelope = makePreflightEnvelope(options: options)
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

    private func trainingActionArguments() -> [String] {
        var args = ["mere.run", "image", "train-lora"]
        if let data {
            args += ["--data", data]
        }
        args += ["--output", output]
        if let model {
            args += ["--model", model]
        }
        if let widthOverride {
            args += ["--width", String(widthOverride)]
        }
        if let heightOverride {
            args += ["--height", String(heightOverride)]
        }
        if let trainingStepsOverride {
            args += ["--training-steps", String(trainingStepsOverride)]
        }
        if batchSize != 1 {
            args += ["--batch-size", String(batchSize)]
        }
        if let learningRateOverride {
            args += ["--learning-rate", String(learningRateOverride)]
        }
        if let rankOverride {
            args += ["--rank", String(rankOverride)]
        }
        if let alphaOverride {
            args += ["--alpha", String(alphaOverride)]
        }
        if maxTextLength != 512 {
            args += ["--max-text-length", String(maxTextLength)]
        }
        if schedulerSteps != 1000 {
            args += ["--scheduler-steps", String(schedulerSteps)]
        }
        if let captionDropoutOverride {
            args += ["--caption-dropout", String(captionDropoutOverride)]
        }
        if seed != 0 {
            args += ["--seed", String(seed)]
        }
        if lite {
            args.append("--lite")
        }
        if excludePreviewImages {
            args.append("--exclude-preview-images")
        }
        if let checkpointInterval {
            args += ["--checkpoint-interval", String(checkpointInterval)]
        }
        if let maxResolution {
            args += ["--max-resolution", String(maxResolution)]
        }
        if progressive {
            args.append("--progressive")
        }
        if lowRam {
            args.append("--low-ram")
        }
        if noCompile {
            args.append("--no-compile")
        }
        if gradientCheckpointing {
            args.append("--gradient-checkpointing")
        }
        if let recipe {
            args += ["--recipe", recipe]
        }
        if let benchmarkSteps {
            args += ["--benchmark-steps", String(benchmarkSteps)]
        }
        if benchmarkWarmupSteps != 5 {
            args += ["--benchmark-warmup-steps", String(benchmarkWarmupSteps)]
        }
        if let sampleInterval {
            args += ["--sample-interval", String(sampleInterval)]
        }
        if let samplePrompt {
            args += ["--sample-prompt", samplePrompt]
        }
        if let sampleModel {
            args += ["--sample-model", sampleModel]
        }
        if sampleSteps != 8 {
            args += ["--sample-steps", String(sampleSteps)]
        }
        if sampleGuidanceScale != 1.0 {
            args += ["--sample-cfg", String(sampleGuidanceScale)]
        }
        if sampleLoRAScale != 1.0 {
            args += ["--sample-lora-scale", String(sampleLoRAScale)]
        }
        if let sampleSeed {
            args += ["--sample-seed", String(sampleSeed)]
        }
        if visualize {
            args.append("--visualize")
            if visualizePort != 8787 {
                args += ["--visualize-port", String(visualizePort)]
            }
        }
        if let loraTargetRanks {
            args += ["--lora-target-ranks", loraTargetRanks]
        }
        if let loraRankPreset {
            args += ["--lora-rank-preset", loraRankPreset]
        }
        if let loraTargetPreset {
            args += ["--lora-target-preset", loraTargetPreset]
        }
        if let loraTargetMode {
            args += ["--lora-target-mode", loraTargetMode]
        }
        if let timestepSampling {
            args += ["--timestep-sampling", timestepSampling]
        }
        if let timestepLossWeighting {
            args += ["--timestep-loss-weighting", timestepLossWeighting]
        }
        if let lossWeighting {
            args += ["--loss-weighting", lossWeighting]
        }
        if let timestepLow {
            args += ["--timestep-low", String(timestepLow)]
        }
        if let timestepHigh {
            args += ["--timestep-high", String(timestepHigh)]
        }
        if let lrWarmupSteps {
            args += ["--lr-warmup-steps", String(lrWarmupSteps)]
        }
        if noCosineScheduler {
            args.append("--no-cosine-scheduler")
        }
        if let lrMinFactor {
            args += ["--lr-min-factor", String(lrMinFactor)]
        }
        if let adamWeightDecay {
            args += ["--adam-weight-decay", String(adamWeightDecay)]
        }
        if let syntheticSamples {
            args += ["--synthetic-samples", String(syntheticSamples)]
        }
        if quiet {
            args.append("--quiet")
        }
        return args
    }

    private func runKreaTraining(
        modelRoot: URL,
        outputURL: URL,
        options: ResolvedLoRATrainingOptions,
        eventLogger: LoRATrainingEventLogger?
    ) async throws {
        if options.checkpointInterval != nil {
            throw ValidationError("--checkpoint-interval is only supported for FLUX.2 Klein LoRA training")
        }
        if hasKleinOnlyTrainingOptions(options: options) {
            throw ValidationError("Klein training options require a FLUX.2 Klein base model.")
        }

        let examples: [Krea2LoRATrainingExample]
        let datasetRoot: String?
        if syntheticSamples != nil {
            examples = []
            datasetRoot = nil
        } else {
            guard let data else {
                throw ValidationError("--data is required unless --synthetic-samples is set")
            }
            let dataURL = URL(fileURLWithPath: data).standardizedFileURL
            let pairs = try DatasetLoader.loadImageCaptionPairs(
                from: dataURL,
                excludePreviewImages: excludePreviewImages
            )
            examples = pairs.map { pair in
                Krea2LoRATrainingExample(imageURL: pair.imageURL, caption: pair.caption)
            }
            datasetRoot = dataURL.path
        }

        var config = Krea2LoRATrainingConfig()
        config.width = options.width
        config.height = options.height
        config.maxTextLength = maxTextLength
        config.schedulerSteps = schedulerSteps
        config.trainingSteps = options.trainingSteps
        config.batchSize = batchSize
        config.learningRate = options.learningRate
        config.seed = seed
        config.loraRank = options.rank
        config.loraAlpha = options.alpha
        config.captionDropout = options.captionDropout
        config.loraTargetSuffixes = lite ? Krea2LoRAInjector.liteTargetSuffixes : nil
        config.syntheticSampleCount = syntheticSamples
        config.datasetRoot = datasetRoot
        config.useCompile = !options.noCompile
        if let lrWarmupSteps = options.lrWarmupSteps {
            config.lrWarmupSteps = lrWarmupSteps
        }
        if let useCosineScheduler = options.useCosineScheduler {
            config.useCosineScheduler = useCosineScheduler
        }
        if let lrMinFactor = options.lrMinFactor {
            config.lrMinFactor = lrMinFactor
        }

        let stderrProgressHandler = quiet ? nil : Self.makeProgressHandler()
        let progressHandler = Self.makeKreaProgressHandler(
            stderrProgressHandler: stderrProgressHandler,
            eventLogger: eventLogger
        )
        if !quiet {
            CLIStderr.write("[runtime] image training backend: \(NativeMLXRuntime.backendDescription)\n")
        }

        try await Krea2LoRATrainer.train(
            modelPath: modelRoot.path,
            examples: examples,
            outputURL: outputURL,
            config: config,
            progressHandler: progressHandler
        )
    }

    private func runKleinTraining(
        modelRoot: URL,
        outputURL: URL,
        options: ResolvedLoRATrainingOptions,
        eventLogger: LoRATrainingEventLogger?
    ) async throws {
        if syntheticSamples != nil {
            throw ValidationError("--synthetic-samples is only supported for Krea 2 LoRA smoke tests")
        }
        guard let data else {
            throw ValidationError("--data is required")
        }

        let dataURL = URL(fileURLWithPath: data).standardizedFileURL
        let pairs = try DatasetLoader.loadImageCaptionPairs(
            from: dataURL,
            excludePreviewImages: excludePreviewImages
        )
        let examples = pairs.map { pair in
            Flux2KleinLoRATrainingExample(imageURL: pair.imageURL, caption: pair.caption)
        }

        var config = Flux2KleinLoRATrainingConfig()
        let rankPreset = try Self.resolveKleinRankPreset(loraRankPreset)
        let resolvedRank = rankPreset?.rank ?? options.rank
        let targetRanks = try Self.resolveKleinTargetPreset(options.loraTargetPreset, rank: resolvedRank)
        let targetRankSuffixes = try loraTargetRanks.map(Self.parseKleinTargetRankSuffixes) ?? rankPreset?.targetRankSuffixes
        config.width = options.width
        config.height = options.height
        config.maxResolution = options.maxResolution
        config.maxTextLength = maxTextLength
        config.schedulerSteps = schedulerSteps
        config.trainingSteps = benchmarkSteps.map { benchmarkWarmupSteps + $0 } ?? options.trainingSteps
        config.batchSize = batchSize
        config.learningRate = options.learningRate
        config.seed = seed
        config.loraRank = resolvedRank
        config.loraAlpha = options.alpha ?? rankPreset?.alpha ?? Float(config.loraRank)
        config.loraTargetMode = try Self.resolveKleinLoRATargetMode(loraTargetMode)
        config.captionDropout = options.captionDropout
        config.loraTargetSuffixes = lite ? Self.kleinLiteTargetSuffixes : nil
        config.loraTargetRanks = targetRanks
        config.loraTargetRankSuffixes = targetRankSuffixes
        config.checkpointInterval = options.checkpointInterval
        config.sampleInterval = sampleInterval
        config.samplePrompt = samplePrompt
        config.progressive = progressive
        config.lowRam = options.lowRam
        config.gradientCheckpointing = gradientCheckpointing
        config.benchmarkSteps = benchmarkSteps
        config.benchmarkWarmupSteps = benchmarkWarmupSteps
        if options.noCompile || gradientCheckpointing {
            config.useCompile = false
        }
        config.datasetRoot = dataURL.path
        if let raw = timestepSampling {
            guard let parsed = Flux2TimestepSamplingStrategy(rawValue: raw) else {
                throw ValidationError("Unsupported --timestep-sampling '\(raw)'")
            }
            config.timestepSampling = parsed
        }
        if let raw = timestepLossWeighting {
            guard let parsed = Flux2TimestepLossWeightingStrategy(rawValue: raw) else {
                throw ValidationError("Unsupported --timestep-loss-weighting '\(raw)'")
            }
            config.timestepLossWeighting = parsed
        }
        if let raw = lossWeighting {
            guard let parsed = Flux2LossWeightingStrategy(rawValue: raw) else {
                throw ValidationError("Unsupported --loss-weighting '\(raw)'")
            }
            config.lossWeighting = parsed
        }
        if let timestepLow {
            config.timestepLow = timestepLow
        }
        if let timestepHigh {
            config.timestepHigh = timestepHigh
        }
        if let lrWarmupSteps {
            config.lrWarmupSteps = lrWarmupSteps
        }
        if noCosineScheduler {
            config.useCosineScheduler = false
        }
        if let lrMinFactor {
            config.lrMinFactor = lrMinFactor
        }
        if let adamWeightDecay {
            config.adamWeightDecay = adamWeightDecay
        }

        let stderrProgressHandler = quiet ? nil : Self.makeKleinProgressHandler()
        let progressHandler = Self.makeKleinProgressHandler(
            stderrProgressHandler: stderrProgressHandler,
            eventLogger: eventLogger
        )
        let sampleHandler = try makeKleinSampleHandler(
            outputURL: outputURL,
            fallbackPrompt: examples.first?.caption ?? "",
            width: options.width,
            height: options.height,
            eventLogger: eventLogger
        )
        if !quiet {
            CLIStderr.write("[runtime] image training backend: \(NativeMLXRuntime.backendDescription)\n")
        }

        try await Flux2KleinLoRATrainer.train(
            modelPath: modelRoot.path,
            examples: examples,
            outputURL: outputURL,
            config: config,
            progressHandler: progressHandler,
            sampleHandler: sampleHandler
        )
    }

    struct ResolvedLoRATrainingOptions {
        let model: String?
        let width: Int
        let height: Int
        let trainingSteps: Int
        let learningRate: Float
        let rank: Int
        let alpha: Float?
        let captionDropout: Float
        let checkpointInterval: Int?
        let maxResolution: Int?
        let lowRam: Bool
        let noCompile: Bool
        let loraTargetPreset: String?
        let lrWarmupSteps: Int?
        let useCosineScheduler: Bool?
        let lrMinFactor: Float?
    }

    private struct LoRATrainingRecipe {
        let model: String?
        let width: Int?
        let height: Int?
        let trainingSteps: Int?
        let learningRate: Float?
        let rank: Int?
        let alpha: Float?
        let captionDropout: Float?
        let checkpointInterval: Int?
        let maxResolution: Int?
        let lowRam: Bool
        let noCompile: Bool
        let loraTargetPreset: String?
        let lrWarmupSteps: Int?
        let useCosineScheduler: Bool?
        let lrMinFactor: Float?
    }

    func resolvedTrainingOptions() throws -> ResolvedLoRATrainingOptions {
        let recipe = try Self.resolveLoRATrainingRecipe(recipe)
        let explicitSchedulerRequested = lrWarmupSteps != nil || lrMinFactor != nil
        return ResolvedLoRATrainingOptions(
            model: model ?? recipe?.model,
            width: widthOverride ?? recipe?.width ?? Self.defaultWidth,
            height: heightOverride ?? recipe?.height ?? Self.defaultHeight,
            trainingSteps: trainingStepsOverride ?? recipe?.trainingSteps ?? Self.defaultTrainingSteps,
            learningRate: learningRateOverride ?? recipe?.learningRate ?? Self.defaultLearningRate,
            rank: rankOverride ?? recipe?.rank ?? Self.defaultRank,
            alpha: alphaOverride ?? recipe?.alpha,
            captionDropout: captionDropoutOverride ?? recipe?.captionDropout ?? Self.defaultCaptionDropout,
            checkpointInterval: checkpointInterval ?? recipe?.checkpointInterval,
            maxResolution: maxResolution ?? recipe?.maxResolution,
            lowRam: lowRam || recipe?.lowRam == true,
            noCompile: noCompile || recipe?.noCompile == true,
            loraTargetPreset: loraTargetPreset ?? recipe?.loraTargetPreset,
            lrWarmupSteps: lrWarmupSteps ?? recipe?.lrWarmupSteps,
            useCosineScheduler: noCosineScheduler ? false : recipe?.useCosineScheduler ?? (explicitSchedulerRequested ? true : nil),
            lrMinFactor: lrMinFactor ?? recipe?.lrMinFactor
        )
    }

    private static func resolveLoRATrainingRecipe(_ raw: String?) throws -> LoRATrainingRecipe? {
        guard let raw else { return nil }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "krea-fast-style", "local-krea-style", "fal-krea-style", "krea2-fast-style":
            return LoRATrainingRecipe(
                model: ModelResolver.ModelID.krea2Raw.rawValue,
                width: 768,
                height: 768,
                trainingSteps: 100,
                learningRate: 0.0005,
                rank: 32,
                alpha: 32,
                captionDropout: Self.defaultCaptionDropout,
                checkpointInterval: nil,
                maxResolution: nil,
                lowRam: false,
                noCompile: false,
                loraTargetPreset: nil,
                lrWarmupSteps: 10,
                useCosineScheduler: true,
                lrMinFactor: 0
            )
        case "krea-cinematic-style", "krea-movie-style", "krea-wide-style":
            return LoRATrainingRecipe(
                model: ModelResolver.ModelID.krea2Raw.rawValue,
                width: 768,
                height: 416,
                trainingSteps: 200,
                learningRate: 0.0001,
                rank: 32,
                alpha: 32,
                captionDropout: Self.defaultCaptionDropout,
                checkpointInterval: nil,
                maxResolution: nil,
                lowRam: false,
                noCompile: true,
                loraTargetPreset: nil,
                lrWarmupSteps: 20,
                useCosineScheduler: true,
                lrMinFactor: 0
            )
        case "klein-fast-style", "local-klein-style", "flux2-klein-fast-style":
            return LoRATrainingRecipe(
                model: ModelResolver.ModelID.kleinBase9B.rawValue,
                width: nil,
                height: nil,
                trainingSteps: 1000,
                learningRate: 0.00005,
                rank: nil,
                alpha: nil,
                captionDropout: nil,
                checkpointInterval: 250,
                maxResolution: 512,
                lowRam: true,
                noCompile: true,
                loraTargetPreset: "fal-klein-fast",
                lrWarmupSteps: nil,
                useCosineScheduler: nil,
                lrMinFactor: nil
            )
        default:
            throw ValidationError(
                "Unsupported --recipe '\(raw)'. Supported recipes: krea-fast-style, krea-cinematic-style, klein-fast-style"
            )
        }
    }

    private struct KleinRankPreset {
        let rank: Int
        let alpha: Float
        let targetRankSuffixes: [String: Int]
    }

    private static func resolveKleinRankPreset(_ raw: String?) throws -> KleinRankPreset? {
        guard let raw else { return nil }
        switch raw.lowercased() {
        case "flux2-style-128", "style-128":
            return KleinRankPreset(
                rank: 128,
                alpha: 64,
                targetRankSuffixes: Dictionary(
                    uniqueKeysWithValues: Flux2LoRAInjector.defaultTargetSuffixes.map { ($0, 128) }
                )
            )
        default:
            throw ValidationError("Unsupported --lora-rank-preset '\(raw)'. Supported preset: flux2-style-128")
        }
    }

    private static func resolveKleinLoRATargetMode(_ raw: String?) throws -> Flux2LoRAInjector.TargetMode {
        guard let raw else { return .suffix }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "suffix", "default":
            return .suffix
        case "transformer-linear-walk", "linear-walk", "all-linear", "walk":
            return .transformerLinearWalk
        default:
            throw ValidationError("Unsupported --lora-target-mode '\(raw)'. Supported modes: suffix, transformer-linear-walk")
        }
    }

    static func resolveKleinTargetPreset(_ raw: String?, rank: Int) throws -> [String: Int]? {
        guard let raw else { return nil }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "fal-klein-fast", "fal-fast", "flux2-klein-fal-fast":
            return falKleinFastTargetRanks(rank: rank)
        default:
            throw ValidationError("Unsupported --lora-target-preset '\(raw)'. Supported preset: fal-klein-fast")
        }
    }

    private static func falKleinFastTargetRanks(rank: Int) -> [String: Int] {
        var ranks: [String: Int] = [
            "x_embedder": rank,
            "context_embedder": rank,
            "time_guidance_embed.timestep_embedder.linear_1": rank,
            "time_guidance_embed.timestep_embedder.linear_2": rank,
            "double_stream_modulation_img.linear": rank,
            "double_stream_modulation_txt.linear": rank,
            "single_stream_modulation.linear": rank,
            "proj_out": rank,
        ]

        for block in 0..<8 {
            for suffix in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
            ] {
                ranks["transformer_blocks.\(block).\(suffix)"] = rank
            }
        }

        for block in 0..<24 {
            ranks["single_transformer_blocks.\(block).attn.to_qkv_mlp_proj"] = rank
            ranks["single_transformer_blocks.\(block).attn.to_out"] = rank
        }

        return ranks
    }

    private static func parseKleinTargetRankSuffixes(_ raw: String) throws -> [String: Int] {
        let entries = raw
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !entries.isEmpty else {
            throw ValidationError("--lora-target-ranks cannot be empty")
        }

        var ranks: [String: Int] = [:]
        for entry in entries {
            let parts = entry
                .split(separator: "=", maxSplits: 1)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            guard parts.count == 2, !parts[0].isEmpty, let rank = Int(parts[1]) else {
                throw ValidationError("Invalid --lora-target-ranks entry '\(entry)'; use suffix=rank")
            }
            guard rank >= 1 else {
                throw ValidationError("--lora-target-ranks entry '\(entry)' must use rank >= 1")
            }
            ranks[parts[0]] = rank
        }
        return ranks
    }

    private func makeKleinSampleHandler(
        outputURL: URL,
        fallbackPrompt: String,
        width: Int,
        height: Int,
        eventLogger: LoRATrainingEventLogger?
    ) throws -> (@Sendable (Int, URL) async -> Void)? {
        guard sampleInterval != nil else { return nil }

        let modelPath = try resolveKleinSampleModelPath()
        let prompt = (samplePrompt ?? fallbackPrompt).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw ValidationError("--sample-prompt is empty and the first caption is empty")
        }

        let generator = Flux2KleinGenerator()
        let outputBaseName = outputURL.deletingPathExtension().lastPathComponent
        let sampleDirectory = outputURL.deletingLastPathComponent().appendingPathComponent("samples", isDirectory: true)
        let sampleWidth = width
        let sampleHeight = height
        let steps = sampleSteps
        let guidanceScale = sampleGuidanceScale
        let loraScale = sampleLoRAScale
        let seedValue = sampleSeed ?? (seed == 0 ? 42 : seed)
        let quietMode = quiet

        return { step, checkpointURL in
            do {
                try FileManager.default.createDirectory(at: sampleDirectory, withIntermediateDirectories: true)
                let sampleURL = sampleDirectory.appendingPathComponent("\(outputBaseName)-step\(step)-sample.png")
                let request = GenerationRequest(
                    prompt: prompt,
                    width: sampleWidth,
                    height: sampleHeight,
                    steps: steps,
                    guidanceScale: guidanceScale,
                    seed: seedValue,
                    outputURL: sampleURL,
                    model: modelPath,
                    lora: .local(path: checkpointURL.path, scale: loraScale)
                )
                _ = try await generator.generate(request, progressHandler: nil)
                try? eventLogger?.record(
                    type: "sample_saved",
                    stage: "sampling",
                    step: step,
                    path: sampleURL.path,
                    metadata: ["checkpoint": checkpointURL.path]
                )
                if !quietMode {
                    CLIStderr.write("[sample] \(sampleURL.path)\n")
                }
            } catch {
                try? eventLogger?.record(
                    type: "sample_failed",
                    stage: "sampling",
                    message: error.localizedDescription,
                    step: step,
                    metadata: ["checkpoint": checkpointURL.path]
                )
                CLIStderr.write("[sample] step \(step) failed: \(error.localizedDescription)\n")
            }
        }
    }

    func materializedPlanURL(for outputURL: URL) -> URL? {
        let planURL = outputURL.deletingLastPathComponent().appendingPathComponent("plan.json", isDirectory: false)
        guard FileManager.default.fileExists(atPath: planURL.path),
              let plan = try? LoRATrainingRunPlan.decode(from: planURL),
              URL(fileURLWithPath: plan.arguments.output).standardizedFileURL.path == outputURL.standardizedFileURL.path else {
            return nil
        }
        return planURL
    }

    func makeRunEventLoggerIfNeeded(
        outputURL: URL,
        modelRoot: URL,
        modelManifest: MereRunModelManifest,
        options: ResolvedLoRATrainingOptions
    ) throws -> LoRATrainingEventLogger? {
        let materializedPlanURL = materializedPlanURL(for: outputURL)
        guard visualize || materializedPlanURL != nil else {
            return nil
        }
        let logger = try LoRATrainingEventLogger(
            baseOutputURL: outputURL,
            resumeExisting: materializedPlanURL != nil
        )
        var metadata = [
            "model_root": modelRoot.path,
            "model_id": modelManifest.id,
            "model_family": modelManifest.family?.rawValue ?? "unknown",
            "recipe": recipe ?? "",
            "data": data ?? "",
            "output": outputURL.path,
        ]
        if let materializedPlanURL {
            metadata["plan_file"] = materializedPlanURL.lastPathComponent
            metadata["actions_file"] = "actions.json"
        }
        try logger.record(
            type: "run_started",
            stage: "starting",
            message: "LoRA training started.",
            step: 0,
            totalSteps: options.trainingSteps,
            fraction: 0,
            path: outputURL.path,
            metadata: metadata
        )
        return logger
    }

    private func startRunContextIfNeeded(
        outputURL: URL,
        modelRoot: URL,
        modelManifest: MereRunModelManifest,
        options: ResolvedLoRATrainingOptions
    ) throws -> LoRATrainingRunContext? {
        guard let logger = try makeRunEventLoggerIfNeeded(
            outputURL: outputURL,
            modelRoot: modelRoot,
            modelManifest: modelManifest,
            options: options
        ) else {
            return nil
        }
        guard visualize else {
            return LoRATrainingRunContext(logger: logger, serverTask: nil)
        }

        let runDirectoryURL = outputURL.deletingLastPathComponent()
        let viewer = LoRATrainingRunViewer(runDirectoryURL: runDirectoryURL)
        let host = "127.0.0.1"
        let port = visualizePort
        let task = Task {
            do {
                try await viewer.run(host: host, port: port)
            } catch is CancellationError {
                // The training command owns this helper server and cancels it when the run exits.
            } catch {
                CLIStderr.write("[visualize] server stopped: \(error.localizedDescription)\n")
            }
        }
        try logger.record(
            type: "viewer_started",
            stage: "visualizing",
            message: "LoRA training viewer started.",
            path: runDirectoryURL.path,
            metadata: ["viewer_url": "http://\(host):\(port)"]
        )
        return LoRATrainingRunContext(logger: logger, serverTask: task)
    }

    private func resolveKleinSampleModelPath() throws -> String {
        let spec = sampleModel ?? ModelResolver.ModelID.klein9B.rawValue
        let url = URL(fileURLWithPath: spec).standardizedFileURL
        if FileManager.default.fileExists(atPath: url.path) {
            return url.path
        }

        if let id = ModelResolver.ModelID(rawValue: spec) {
            do {
                return try ModelResolver().resolve(id).rootURL.path
            } catch {
                throw ValidationError(
                    "Sample model \(id.rawValue) not found. Pull it with `mere.run model pull \(id.rawValue)` or pass --sample-model."
                )
            }
        }

        throw ValidationError("Sample model path not found: \(spec). Pass a local path or known model id.")
    }

    private func hasKleinOnlyTrainingOptions(options: ResolvedLoRATrainingOptions) -> Bool {
        options.maxResolution != nil ||
            progressive ||
            options.lowRam ||
            gradientCheckpointing ||
            benchmarkSteps != nil ||
            benchmarkWarmupSteps != 5 ||
            sampleInterval != nil ||
            samplePrompt != nil ||
            sampleModel != nil ||
            sampleSteps != 8 ||
            sampleGuidanceScale != 1.0 ||
            sampleLoRAScale != 1.0 ||
            sampleSeed != nil ||
            loraTargetRanks != nil ||
            loraRankPreset != nil ||
            options.loraTargetPreset != nil ||
            loraTargetMode != nil ||
            timestepSampling != nil ||
            timestepLossWeighting != nil ||
            lossWeighting != nil ||
            timestepLow != nil ||
            timestepHigh != nil ||
            adamWeightDecay != nil
    }

    private func resolveModelRoot(model: String?) throws -> URL {
        let resolver = ModelResolver()
        guard let model else {
            do {
                return try resolver.resolve(Self.defaultManagedModelID).rootURL
            } catch {
                let modelID = Self.defaultManagedModelID.rawValue
                throw ValidationError(
                    "Krea 2 Raw model \(modelID) not found. Pull it with `mere.run model pull \(modelID)` or point --model at a local Raw model path."
                )
            }
        }

        let url = URL(fileURLWithPath: model).standardizedFileURL
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        if let id = ModelResolver.ModelID(rawValue: model) {
            do {
                return try resolver.resolve(id).rootURL
            } catch {
                throw ValidationError(
                    "Model \(id.rawValue) not found. Pull it with `mere.run model pull \(id.rawValue)` or point --model at a local path."
                )
            }
        }
        throw ValidationError("Model path not found: \(model). Pass a local model path or a known model id.")
    }

    private static let kleinLiteTargetSuffixes: [String] = [
        ".attn.to_q",
        ".attn.to_v",
        ".attn.add_q_proj",
        ".attn.add_v_proj",
    ]

    private static func makeProgressHandler() -> (@Sendable (Krea2LoRATrainingProgress) -> Void) {
        { progress in
            switch progress.stage {
            case .loadingModels:
                CLIStderr.write("Loading Krea 2 Raw models...\n")
            case .encodingDataset(let current, let total):
                CLIStderr.write("\rEncoding dataset (\(current)/\(total))")
                if current >= total {
                    CLIStderr.write("\n")
                }
            case .injectingLoRA(let layerCount):
                CLIStderr.write("Injected LoRA into \(layerCount) Krea 2 layers.\n")
            case .training(let step, let total, let loss):
                if let loss {
                    CLIStderr.write(String(format: "\rTraining (%d/%d) loss %.6f\n", step, total, loss))
                } else {
                    CLIStderr.write("\rTraining (\(step)/\(total))")
                }
            case .saving:
                CLIStderr.write("Saving LoRA artifacts...\n")
            }
        }
    }

    private static func makeKleinProgressHandler() -> (@Sendable (Flux2KleinLoRATrainingProgress) -> Void) {
        { progress in
            switch progress.stage {
            case .loadingModels:
                CLIStderr.write("Loading FLUX.2 Klein models...\n")
            case .encodingDataset(let current, let total):
                CLIStderr.write("\rEncoding dataset (\(current)/\(total))")
                if current >= total {
                    CLIStderr.write("\n")
                }
            case .injectingLoRA(let layerCount):
                CLIStderr.write("Injected LoRA into \(layerCount) FLUX.2 Klein layers.\n")
            case .training(let step, let total, let loss):
                if let loss {
                    CLIStderr.write(String(format: "\rTraining (%d/%d) loss %.6f\n", step, total, loss))
                } else {
                    CLIStderr.write("\rTraining (\(step)/\(total))")
                }
            case .sampling(let step):
                CLIStderr.write("Sampling preview at step \(step)...\n")
            case .saving:
                CLIStderr.write("Saving LoRA artifacts...\n")
            }
        }
    }

    private static func makeKreaProgressHandler(
        stderrProgressHandler: (@Sendable (Krea2LoRATrainingProgress) -> Void)?,
        eventLogger: LoRATrainingEventLogger?
    ) -> (@Sendable (Krea2LoRATrainingProgress) -> Void)? {
        guard stderrProgressHandler != nil || eventLogger != nil else { return nil }
        return { progress in
            stderrProgressHandler?(progress)
            recordKreaProgress(progress, to: eventLogger)
        }
    }

    private static func makeKleinProgressHandler(
        stderrProgressHandler: (@Sendable (Flux2KleinLoRATrainingProgress) -> Void)?,
        eventLogger: LoRATrainingEventLogger?
    ) -> (@Sendable (Flux2KleinLoRATrainingProgress) -> Void)? {
        guard stderrProgressHandler != nil || eventLogger != nil else { return nil }
        return { progress in
            stderrProgressHandler?(progress)
            recordKleinProgress(progress, to: eventLogger)
        }
    }

    private static func recordKreaProgress(
        _ progress: Krea2LoRATrainingProgress,
        to eventLogger: LoRATrainingEventLogger?
    ) {
        guard let eventLogger else { return }
        switch progress.stage {
        case .loadingModels:
            try? eventLogger.record(
                type: "progress",
                stage: "loading_models",
                message: "Loading Krea 2 Raw models.",
                fraction: progress.fraction
            )
        case .encodingDataset(let current, let total):
            try? eventLogger.record(
                type: "progress",
                stage: "encoding_dataset",
                message: "Encoding dataset.",
                step: current,
                totalSteps: total,
                fraction: progress.fraction
            )
        case .injectingLoRA(let layerCount):
            try? eventLogger.record(
                type: "progress",
                stage: "injecting_lora",
                message: "Injected LoRA layers.",
                fraction: progress.fraction,
                metadata: ["layer_count": "\(layerCount)"]
            )
        case .training(let step, let total, let loss):
            try? eventLogger.record(
                type: "progress",
                stage: "training",
                message: "Training.",
                step: step,
                totalSteps: total,
                loss: loss,
                fraction: progress.fraction
            )
        case .saving:
            try? eventLogger.record(
                type: "progress",
                stage: "saving",
                message: "Saving LoRA artifacts.",
                fraction: progress.fraction
            )
        }
    }

    private static func recordKleinProgress(
        _ progress: Flux2KleinLoRATrainingProgress,
        to eventLogger: LoRATrainingEventLogger?
    ) {
        guard let eventLogger else { return }
        switch progress.stage {
        case .loadingModels:
            try? eventLogger.record(
                type: "progress",
                stage: "loading_models",
                message: "Loading FLUX.2 Klein models.",
                fraction: progress.fraction
            )
        case .encodingDataset(let current, let total):
            try? eventLogger.record(
                type: "progress",
                stage: "encoding_dataset",
                message: "Encoding dataset.",
                step: current,
                totalSteps: total,
                fraction: progress.fraction
            )
        case .injectingLoRA(let layerCount):
            try? eventLogger.record(
                type: "progress",
                stage: "injecting_lora",
                message: "Injected LoRA layers.",
                fraction: progress.fraction,
                metadata: ["layer_count": "\(layerCount)"]
            )
        case .training(let step, let total, let loss):
            try? eventLogger.record(
                type: "progress",
                stage: "training",
                message: "Training.",
                step: step,
                totalSteps: total,
                loss: loss,
                fraction: progress.fraction
            )
        case .sampling(let step):
            try? eventLogger.record(
                type: "progress",
                stage: "sampling",
                message: "Sampling preview.",
                step: step,
                fraction: progress.fraction
            )
        case .saving:
            try? eventLogger.record(
                type: "progress",
                stage: "saving",
                message: "Saving LoRA artifacts.",
                fraction: progress.fraction
            )
        }
    }
}

private struct LoRATrainingRunContext {
    let logger: LoRATrainingEventLogger
    let serverTask: Task<Void, Never>?

    func stop() {
        serverTask?.cancel()
    }
}
