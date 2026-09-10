import ArgumentParser
import Foundation
import MereRunCore

struct ImageTrainLoRA: AsyncParsableCommand {
    static let defaultManagedModelID: ModelResolver.ModelID = ImageLoRATrainingOptions.defaultManagedModelID
    private static let defaultWidth = ImageLoRATrainingOptions.defaultWidth
    private static let defaultHeight = ImageLoRATrainingOptions.defaultHeight
    private static let defaultTrainingSteps = ImageLoRATrainingOptions.defaultTrainingSteps
    private static let defaultLearningRate: Float = ImageLoRATrainingOptions.defaultLearningRate
    private static let defaultRank = ImageLoRATrainingOptions.defaultRank
    private static let defaultCaptionDropout: Float = ImageLoRATrainingOptions.defaultCaptionDropout

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

    @Option(
        name: [.customLong("base-quantization-bits")],
        help: "Quantize the frozen Krea base transformer to 4 or 8 bits while training."
    )
    var baseQuantizationBits: Int?

    @Flag(name: [.customLong("exclude-preview-images")], help: "Ignore preview*.png/jpg/webp images in the dataset folder.")
    var excludePreviewImages: Bool = false

    @Option(
        name: [.customLong("checkpoint-interval")],
        help: "Save intermediate Klein LoRA checkpoints every N steps."
    )
    var checkpointInterval: Int?

    @Option(
        name: [.customLong("resume-from")],
        help: "Resume FLUX.2 Klein training from a .safetensors checkpoint or checkpoint archive."
    )
    var resumeFrom: String?

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

    func trainingOptions() -> ImageLoRATrainingOptions {
        var options = ImageLoRATrainingOptions(data: data, output: output)
        options.model = model
        options.widthOverride = widthOverride
        options.heightOverride = heightOverride
        options.trainingStepsOverride = trainingStepsOverride
        options.batchSize = batchSize
        options.learningRateOverride = learningRateOverride
        options.rankOverride = rankOverride
        options.alphaOverride = alphaOverride
        options.maxTextLength = maxTextLength
        options.schedulerSteps = schedulerSteps
        options.captionDropoutOverride = captionDropoutOverride
        options.seed = seed
        options.lite = lite
        options.baseQuantizationBits = baseQuantizationBits
        options.excludePreviewImages = excludePreviewImages
        options.checkpointInterval = checkpointInterval
        options.resumeFrom = resumeFrom
        options.maxResolution = maxResolution
        options.progressive = progressive
        options.lowRam = lowRam
        options.noCompile = noCompile
        options.gradientCheckpointing = gradientCheckpointing
        options.recipe = recipe
        options.benchmarkSteps = benchmarkSteps
        options.benchmarkWarmupSteps = benchmarkWarmupSteps
        options.sampleInterval = sampleInterval
        options.samplePrompt = samplePrompt
        options.sampleModel = sampleModel
        options.sampleSteps = sampleSteps
        options.sampleGuidanceScale = sampleGuidanceScale
        options.sampleLoRAScale = sampleLoRAScale
        options.sampleSeed = sampleSeed
        options.loraTargetRanks = loraTargetRanks
        options.loraRankPreset = loraRankPreset
        options.loraTargetPreset = loraTargetPreset
        options.loraTargetMode = loraTargetMode
        options.timestepSampling = timestepSampling
        options.timestepLossWeighting = timestepLossWeighting
        options.lossWeighting = lossWeighting
        options.timestepLow = timestepLow
        options.timestepHigh = timestepHigh
        options.lrWarmupSteps = lrWarmupSteps
        options.noCosineScheduler = noCosineScheduler
        options.lrMinFactor = lrMinFactor
        options.adamWeightDecay = adamWeightDecay
        options.syntheticSamples = syntheticSamples
        return options
    }

    func run() async throws {
        do {
            try await runTraining()
        } catch let error as ImageLoRATrainingIssue {
            throw ValidationError(error.message(modelPullCommand: CLICommandDisplay.modelPullCommand))
        }
    }

    private func runTraining() async throws {
        let input = trainingOptions()
        let resolvedOptions = try input.resolve()
        try input.validate(resolvedOptions)
        if visualize, !(1...65535).contains(visualizePort) {
            throw ValidationError("--visualize-port must be between 1 and 65535")
        }
        if preflight {
            try runPreflight(options: resolvedOptions)
            return
        }
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let outputURL = try ImageLoRATrainingPlan.outputURL(for: output)
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let plan = try ImageLoRATrainingPlan.resolve(input)
        let runContext = try startRunContextIfNeeded(
            outputURL: outputURL, modelRoot: plan.modelRoot,
            modelManifest: plan.modelManifest, options: resolvedOptions
        )
        let outcome: ImageLoRATrainingOutcome
        do {
            if !quiet {
                CLIStderr.write("[runtime] image training backend: \(NativeMLXRuntime.backendDescription)\n")
            }
            outcome = try await ImageLoRATrainingOperation.execute(
                plan, progress: makeTrainingProgressHandler(eventLogger: runContext?.logger)
            )
            try runContext?.logger.record(
                type: "run_finished", stage: "finished", step: resolvedOptions.trainingSteps,
                totalSteps: resolvedOptions.trainingSteps, fraction: 1, path: outputURL.path
            )
        } catch {
            try? runContext?.logger.record(
                type: "run_failed", stage: "failed", message: error.localizedDescription, path: outputURL.path
            )
            await runContext?.stop()
            throw error
        }
        await runContext?.stop()
        if case .saved(let url) = outcome { print(url.path) }
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
        if let baseQuantizationBits {
            args += ["--base-quantization-bits", String(baseQuantizationBits)]
        }
        if excludePreviewImages {
            args.append("--exclude-preview-images")
        }
        if let checkpointInterval {
            args += ["--checkpoint-interval", String(checkpointInterval)]
        }
        if let resumeFrom {
            args += ["--resume-from", resumeFrom]
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

    typealias ResolvedLoRATrainingOptions = ImageLoRATrainingOptions.Resolved

    func resolvedTrainingOptions() throws -> ResolvedLoRATrainingOptions {
        do {
            return try trainingOptions().resolve()
        } catch let error as ImageLoRATrainingIssue {
            throw ValidationError(error.message(modelPullCommand: CLICommandDisplay.modelPullCommand))
        }
    }

    static func resolveKleinTargetPreset(_ raw: String?, rank: Int) throws -> [String: Int]? {
        do {
            return try ImageLoRATrainingOptions.resolveKleinTargetPreset(raw, rank: rank)
        } catch let error as ImageLoRATrainingIssue {
            throw ValidationError(error.message(modelPullCommand: CLICommandDisplay.modelPullCommand))
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
        try logger.record(
            type: "viewer_started",
            stage: "visualizing",
            message: "LoRA training viewer started.",
            path: runDirectoryURL.path,
            metadata: ["viewer_url": "http://\(host):\(port)"]
        )
        let task = Task {
            do {
                try await viewer.run(host: host, port: port)
            } catch is CancellationError {
                // The training command owns this helper server and cancels it when the run exits.
            } catch {
                CLIStderr.write("[visualize] server stopped: \(error.localizedDescription)\n")
            }
        }
        return LoRATrainingRunContext(logger: logger, serverTask: task)
    }

    private func hasKleinOnlyTrainingOptions(options: ResolvedLoRATrainingOptions) -> Bool {
        trainingOptions().hasKleinOnlyTrainingOptions(options: options)
    }

    private func makeTrainingProgressHandler(
        eventLogger: LoRATrainingEventLogger?
    ) -> @Sendable (ImageLoRATrainingProgress) -> Void {
        let krea = Self.makeKreaProgressHandler(
            stderrProgressHandler: quiet ? nil : Self.makeProgressHandler(), eventLogger: eventLogger
        )
        let klein = Self.makeKleinProgressHandler(
            stderrProgressHandler: quiet ? nil : Self.makeKleinProgressHandler(), eventLogger: eventLogger
        )
        return { progress in
            switch progress {
            case .krea(let update): krea?(update)
            case .klein(let update): klein?(update)
            case .sampleSaved(let step, let url, let checkpoint):
                try? eventLogger?.record(
                    type: "sample_saved", stage: "sampling", step: step,
                    path: url.path, metadata: ["checkpoint": checkpoint.path]
                )
                if !quiet { CLIStderr.write("[sample] \(url.path)\n") }
            case .sampleFailed(let step, let message, let checkpoint):
                try? eventLogger?.record(
                    type: "sample_failed", stage: "sampling", message: message,
                    step: step, metadata: ["checkpoint": checkpoint.path]
                )
                CLIStderr.write("[sample] step \(step) failed: \(message)\n")
            }
        }
    }

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

    func stop() async {
        serverTask?.cancel()
        await serverTask?.value
    }
}
