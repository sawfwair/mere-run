import Foundation
import MLX
import MLXNN
import MLXRandom

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

/// Timestep sampling strategy for training.
public enum Flux2TimestepSamplingStrategy: String, Sendable {
    /// Uniform random sampling across all timesteps
    case uniform
    /// Bell-shaped weighting that favors middle timesteps where most learning happens
    case bellCurve
    /// Favors earlier timesteps (content/structure learning)
    case contentFocused
    /// Favors later timesteps (style/detail learning)
    case styleFocused
    /// Logit-normal distribution (OneTrainer default for FLUX.2) - concentrates on mid-range timesteps
    case logitNormal
    /// Uniform sampling with dynamic FlowMatch sigma shift (ai-toolkit `timestep_type: shift`)
    case shift
}

/// Timestep loss weighting strategy (multiplies per-step loss by a timestep-dependent weight).
///
/// This is distinct from timestep *sampling* and mirrors ai-toolkit's `timestep_type: weighted`
/// behavior for FlowMatch schedulers.
public enum Flux2TimestepLossWeightingStrategy: String, Sendable {
    /// No weighting (all timesteps contribute equally).
    case none
    /// Bell-shaped, mean-normalized weighting (ai-toolkit default).
    case weighted
}

/// Loss weighting strategy for training.
public enum Flux2LossWeightingStrategy: String, Sendable {
    /// No weighting - all timesteps contribute equally to loss
    case none
    /// SNR (Signal-to-Noise Ratio) weighting - weight by SNR to prevent high-noise steps from dominating
    case snr
    /// Min-SNR weighting with gamma=5 (from "Efficient Diffusion Training via Min-SNR Weighting")
    case minSNR
}

public enum Flux2KleinLoRATrainerError: Error, LocalizedError {
    case datasetEmpty
    case mixedDatasetModes
    case editTrainingRequiresBaseModel
    case invalidDimensions(width: Int, height: Int)
    case invalidTrainingSteps(Int)
    case invalidBatchSize(Int)
    case invalidSchedulerSteps(Int)
    case outputMustBeSafetensors(URL)
    case imageNotFound(URL)
    case captionEmpty(URL)
    case imageDecodeFailed(URL)
    case missingBatchNormStats

    public var errorDescription: String? {
        switch self {
        case .datasetEmpty:
            return "Training dataset is empty."
        case .mixedDatasetModes:
            return "Dataset mixes edit and txt2img examples. Use a single mode per run."
        case .editTrainingRequiresBaseModel:
            return "Edit training currently requires a FLUX.2 Klein base model."
        case .invalidDimensions(let width, let height):
            return "Invalid image dimensions (\(width)x\(height)); width/height must be >0 and divisible by 16."
        case .invalidTrainingSteps(let steps):
            return "Training steps must be >= 1 (got \(steps))."
        case .invalidBatchSize(let batchSize):
            return "Batch size must be >= 1 (got \(batchSize))."
        case .invalidSchedulerSteps(let steps):
            return "Scheduler steps must be >= 1 (got \(steps))."
        case .outputMustBeSafetensors(let url):
            return "Output must be a .safetensors file: \(url.path)"
        case .imageNotFound(let url):
            return "Image not found: \(url.path)"
        case .captionEmpty(let url):
            return "Caption is empty for: \(url.path)"
        case .imageDecodeFailed(let url):
            return "Failed to decode image: \(url.path)"
        case .missingBatchNormStats:
            return "Missing VAE BatchNorm stats (bn.running_mean / bn.running_var)."
        }
    }
}

public struct Flux2KleinLoRATrainingExample: Hashable, Sendable {
    public let imageURL: URL
    public let inputImageURL: URL?
    public let caption: String

    public init(imageURL: URL, inputImageURL: URL? = nil, caption: String) {
        self.imageURL = imageURL
        self.inputImageURL = inputImageURL
        self.caption = caption
    }
}

public struct Flux2KleinLoRATrainingConfig: Sendable {
    public var width: Int
    public var height: Int
    public var maxResolution: Int?
    
    /// Max token length for the text encoder (lower = faster).
    public var maxTextLength: Int

    /// Number of discrete training timesteps to sample from (ai-toolkit default: 1000).
    public var schedulerSteps: Int

    /// Number of gradient updates to run.
    public var trainingSteps: Int

    public var batchSize: Int
    public var learningRate: Float
    public var seed: UInt64

    public var loraRank: Int
    public var loraAlpha: Float
    public var loraTargetMode: Flux2LoRAInjector.TargetMode
    public var loraTargetSuffixes: [String]?
    public var loraTargetRanks: [String: Int]?
    public var loraTargetRankSuffixes: [String: Int]?
    public var datasetRoot: String?

    /// Probability of dropping caption during training (0.0-1.0).
    /// Helps model learn to generate without text conditioning.
    public var captionDropout: Float

    public var saveDType: DType
    public var logEvery: Int

    /// Interval at which to save checkpoint artifacts.
    public var checkpointInterval: Int?

    /// Interval at which to generate sample previews.
    public var sampleInterval: Int?

    /// Prompt to use for sample generation. If nil, uses first training caption.
    public var samplePrompt: String?

    /// Timestep sampling strategy.
    public var timestepSampling: Flux2TimestepSamplingStrategy

    /// Optional timestep loss weighting (ai-toolkit `timestep_type: weighted`).
    public var timestepLossWeighting: Flux2TimestepLossWeightingStrategy

    /// Loss weighting strategy (SNR-based weighting).
    public var lossWeighting: Flux2LossWeightingStrategy

    /// EMA decay rate for model weights (0 = disabled, 0.9999 = typical).
    /// When enabled, saves both regular and EMA weights.
    public var emaDecay: Float
    
    /// Enable progressive resolution training (e.g. 512 -> 768 -> target).
    public var progressive: Bool
    
    /// Compile the train step with MLX (can speed up steady-state).
    public var useCompile: Bool

    /// Use disk-backed cache for encoded data to reduce peak memory usage.
    public var lowRam: Bool

    /// Recompute transformer block activations during backprop to reduce peak memory.
    public var gradientCheckpointing: Bool
    
    /// Benchmark mode: if set, measure these steps (after warmup) and then exit without saving.
    public var benchmarkSteps: Int?
    
    /// Warmup steps before benchmark timing starts.
    public var benchmarkWarmupSteps: Int
    
    /// AdamW hyperparameters for manual LoRA-only optimization.
    public var adamBeta1: Float
    public var adamBeta2: Float
    public var adamEps: Float
    public var adamWeightDecay: Float

    /// Minimum timestep index to sample (inclusive). Default 0.
    public var timestepLow: Int

    /// Maximum timestep index to sample (exclusive). Default nil = schedulerSteps.
    public var timestepHigh: Int?

    /// Number of warmup steps for learning rate scheduler (OneTrainer default: 200).
    public var lrWarmupSteps: Int

    /// Use cosine annealing LR scheduler (OneTrainer default: true).
    public var useCosineScheduler: Bool

    /// Minimum LR as fraction of base LR at end of cosine decay (0.0 = decay to zero).
    public var lrMinFactor: Float

    public init(
        width: Int = 1024,
        height: Int = 1024,
        maxResolution: Int? = nil,
        maxTextLength: Int = 512,
        schedulerSteps: Int = 1000,
        trainingSteps: Int = 1000,
        batchSize: Int = 1,
        learningRate: Float = 2e-4,
        seed: UInt64 = 0,
        loraRank: Int = 16,
        loraAlpha: Float = 1.0,
        loraTargetMode: Flux2LoRAInjector.TargetMode = .suffix,
        loraTargetSuffixes: [String]? = nil,
        loraTargetRanks: [String: Int]? = nil,
        loraTargetRankSuffixes: [String: Int]? = nil,
        datasetRoot: String? = nil,
        captionDropout: Float = 0.05,
        saveDType: DType = .float16,
        logEvery: Int = 10,
        checkpointInterval: Int? = nil,
        sampleInterval: Int? = nil,
        samplePrompt: String? = nil,
        timestepSampling: Flux2TimestepSamplingStrategy = .logitNormal,
        timestepLossWeighting: Flux2TimestepLossWeightingStrategy = .none,
        lossWeighting: Flux2LossWeightingStrategy = .none,
        emaDecay: Float = 0,
        progressive: Bool = false,
        useCompile: Bool = true,
        lowRam: Bool = false,
        gradientCheckpointing: Bool = false,
        benchmarkSteps: Int? = nil,
        benchmarkWarmupSteps: Int = 5,
        adamBeta1: Float = 0.9,
        adamBeta2: Float = 0.999,
        adamEps: Float = 1e-8,
        adamWeightDecay: Float = 0.01,
        timestepLow: Int = 0,
        timestepHigh: Int? = nil,
        lrWarmupSteps: Int = 200,
        useCosineScheduler: Bool = true,
        lrMinFactor: Float = 0.0
    ) {
        self.width = width
        self.height = height
        self.maxResolution = maxResolution
        self.maxTextLength = maxTextLength
        self.schedulerSteps = schedulerSteps
        self.trainingSteps = trainingSteps
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.seed = seed
        self.loraRank = loraRank
        self.loraAlpha = loraAlpha
        self.loraTargetMode = loraTargetMode
        self.loraTargetSuffixes = loraTargetSuffixes
        self.loraTargetRanks = loraTargetRanks
        self.loraTargetRankSuffixes = loraTargetRankSuffixes
        self.datasetRoot = datasetRoot
        self.captionDropout = captionDropout
        self.saveDType = saveDType
        self.logEvery = logEvery
        self.checkpointInterval = checkpointInterval
        self.sampleInterval = sampleInterval
        self.samplePrompt = samplePrompt
        self.timestepSampling = timestepSampling
        self.timestepLossWeighting = timestepLossWeighting
        self.lossWeighting = lossWeighting
        self.emaDecay = emaDecay
        self.progressive = progressive
        self.useCompile = useCompile
        self.lowRam = lowRam
        self.gradientCheckpointing = gradientCheckpointing
        self.benchmarkSteps = benchmarkSteps
        self.benchmarkWarmupSteps = benchmarkWarmupSteps
        self.adamBeta1 = adamBeta1
        self.adamBeta2 = adamBeta2
        self.adamEps = adamEps
        self.adamWeightDecay = adamWeightDecay
        self.timestepLow = timestepLow
        self.timestepHigh = timestepHigh
        self.lrWarmupSteps = lrWarmupSteps
        self.useCosineScheduler = useCosineScheduler
        self.lrMinFactor = lrMinFactor
    }
}

public struct Flux2KleinLoRATrainingProgress: Sendable {
    public enum Stage: Sendable {
        case loadingModels
        case encodingDataset(current: Int, total: Int)
        case injectingLoRA(layerCount: Int)
        case training(step: Int, total: Int, loss: Float?)
        case sampling(step: Int)
        case saving
    }

    public let stage: Stage
    public let fraction: Float

    public init(stage: Stage, fraction: Float) {
        self.stage = stage
        self.fraction = fraction
    }
}

public enum Flux2KleinLoRATrainer {
    /// Trains a LoRA on the given examples.
    /// - Parameters:
    ///   - modelPath: Path to the base model directory.
    ///   - examples: Training examples (image + caption pairs).
    ///   - outputURL: Where to save the final LoRA safetensors file.
    ///   - config: Training configuration.
    ///   - resumeFromLoRA: Optional URL to an existing LoRA safetensors file to resume training from.
    ///   - progressHandler: Called periodically with training progress.
    ///   - sampleHandler: Called at sample intervals with checkpoint URL. The handler should generate a sample image and return.
    ///   - cancellationHandler: Called when training is cancelled. Returns true to save checkpoint, false to discard.
    ///                          Receives current step number. If handler returns true, checkpoint is saved to outputURL.
    public static func train(
        modelPath: String,
        examples: [Flux2KleinLoRATrainingExample],
        outputURL: URL,
        config: Flux2KleinLoRATrainingConfig = Flux2KleinLoRATrainingConfig(),
        resumeFromLoRA: URL? = nil,
        progressHandler: (@Sendable (Flux2KleinLoRATrainingProgress) -> Void)? = nil,
        sampleHandler: (@Sendable (Int, URL) async -> Void)? = nil,
        cancellationHandler: (@Sendable (Int) async -> Bool)? = nil
    ) async throws {
        guard !examples.isEmpty else {
            throw Flux2KleinLoRATrainerError.datasetEmpty
        }
        guard config.width > 0, config.height > 0, config.width % 16 == 0, config.height % 16 == 0 else {
            throw Flux2KleinLoRATrainerError.invalidDimensions(width: config.width, height: config.height)
        }
        guard config.trainingSteps >= 1 else {
            throw Flux2KleinLoRATrainerError.invalidTrainingSteps(config.trainingSteps)
        }
        guard config.batchSize >= 1 else {
            throw Flux2KleinLoRATrainerError.invalidBatchSize(config.batchSize)
        }
        guard config.schedulerSteps >= 1 else {
            throw Flux2KleinLoRATrainerError.invalidSchedulerSteps(config.schedulerSteps)
        }
        if let maxResolution = config.maxResolution, maxResolution <= 0 {
            throw NSError(
                domain: "Flux2KleinLoRATrainer",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "--max-resolution must be >= 1 (got \(maxResolution))"]
            )
        }
        if config.maxResolution != nil, config.progressive {
            throw NSError(
                domain: "Flux2KleinLoRATrainer",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "--max-resolution cannot be combined with progressive training."]
            )
        }
        guard config.maxTextLength >= 1, config.maxTextLength <= 512 else {
            throw NSError(
                domain: "Flux2KleinLoRATrainer",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "--max-text-len must be between 1 and 512 (got \(config.maxTextLength))"]
            )
        }
        if let benchmarkSteps = config.benchmarkSteps {
            guard benchmarkSteps >= 1 else {
                throw NSError(
                    domain: "Flux2KleinLoRATrainer",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "--benchmark must be >= 1 (got \(benchmarkSteps))"]
                )
            }
            guard config.benchmarkWarmupSteps >= 0 else {
                throw NSError(
                    domain: "Flux2KleinLoRATrainer",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "--benchmark-warmup must be >= 0 (got \(config.benchmarkWarmupSteps))"]
                )
            }
        }
        guard outputURL.pathExtension.lowercased() == "safetensors" else {
            throw Flux2KleinLoRATrainerError.outputMustBeSafetensors(outputURL)
        }

        progressHandler?(Flux2KleinLoRATrainingProgress(stage: .loadingModels, fraction: 0))

        let hasEditExamples = examples.contains { $0.inputImageURL != nil }
        if hasEditExamples, examples.contains(where: { $0.inputImageURL == nil }) {
            throw Flux2KleinLoRATrainerError.mixedDatasetModes
        }
        let datasetFingerprint = LoRATrainingFingerprint.sha256Hex(
            examples
                .map { example in
                    [
                        example.imageURL.standardizedFileURL.path,
                        example.inputImageURL?.standardizedFileURL.path ?? "",
                        example.caption
                    ].joined(separator: "|")
                }
                .joined(separator: "\n")
        )
        let runDataFingerprint = Self.makeRunDataFingerprint(
            examples: examples,
            dataRootPath: config.datasetRoot,
            isEdit: hasEditExamples
        )
        let resolvedSeed = config.seed == 0 ? UInt64(Date().timeIntervalSince1970) : config.seed

        let modelURL = URL(fileURLWithPath: modelPath).standardizedFileURL
        let modelManifest = try MereRunModelManifest.loadRequired(from: modelURL)
        if hasEditExamples {
            guard modelManifest.variant == .base else {
                throw Flux2KleinLoRATrainerError.editTrainingRequiresBaseModel
            }
        }
        let serializedTargetRanks = Self.serializedLoRATargetRanks(config.loraTargetRanks)
        let serializedTargetRankSuffixes = Self.serializedLoRATargetRanks(config.loraTargetRankSuffixes)
        let configFingerprintInput: String = [
            "model:\(modelURL.path)",
            "variant:\(modelManifest.variant?.rawValue ?? "")",
            "size:\(config.width)x\(config.height)",
            "max_resolution:\(config.maxResolution.map { "\($0)" } ?? "")",
            "scheduler_steps:\(config.schedulerSteps)",
            "training_steps:\(config.trainingSteps)",
            "batch_size:\(config.batchSize)",
            "learning_rate:\(config.learningRate)",
            "rank:\(config.loraRank)",
            "alpha:\(config.loraAlpha)",
            "lora_target_mode:\(config.loraTargetMode.rawValue)",
            "caption_dropout:\(config.captionDropout)",
            "max_text_length:\(config.maxTextLength)",
            "checkpoint_interval:\(config.checkpointInterval.map { "\($0)" } ?? "")",
            "sample_interval:\(config.sampleInterval.map { "\($0)" } ?? "")",
            "progressive:\(config.progressive)",
            "gradient_checkpointing:\(config.gradientCheckpointing)",
            "low_ram:\(config.lowRam)",
            "timestep_sampling:\(config.timestepSampling.rawValue)",
            "timestep_loss_weighting:\(config.timestepLossWeighting.rawValue)",
            "loss_weighting:\(config.lossWeighting.rawValue)",
            "timestep_low:\(config.timestepLow)",
            "timestep_high:\(config.timestepHigh.map { "\($0)" } ?? "")",
            "lora_target_suffixes:\((config.loraTargetSuffixes ?? Flux2LoRAInjector.defaultTargetSuffixes).joined(separator: ","))",
            "lora_target_ranks:\(serializedTargetRanks)",
            "lora_target_rank_suffixes:\(serializedTargetRankSuffixes)",
            "edit_mode:\(hasEditExamples)",
        ].joined(separator: "\n")
        let configFingerprint = LoRATrainingFingerprint.sha256Hex(configFingerprintInput)
        let componentResolver = ModelComponentResolver(modelRootURL: modelURL, manifest: modelManifest)

        let transformerComponent = try componentResolver.resolveDirectory(for: .transformer, fallbackLocalPath: "transformer")
        let tokenizerComponent = try componentResolver.resolveDirectory(for: .tokenizer, fallbackLocalPath: "tokenizer")
        let textEncoderComponent = try componentResolver.resolveDirectory(for: .textEncoder, fallbackLocalPath: "text_encoder")
        let vaeComponent = try componentResolver.resolveDirectory(for: .vae, fallbackLocalPath: "vae")

        let transformerQuantization = try ModelWeightsLoader.QuantizationParams.fromManifest(transformerComponent.sourceManifest)
        let textEncoderQuantization = try ModelWeightsLoader.QuantizationParams.fromManifest(textEncoderComponent.sourceManifest)

        // 1) Load transformer
        // Shared shard discovery resolves symlinked component directories before listing them.
        let transformerConfig = try loadTransformerConfig(from: transformerComponent.directoryURL)
        let transformer = Flux2Transformer2DModel(config: transformerConfig)
        try loadTransformerWeights(from: transformerComponent.directoryURL, to: transformer, quantization: transformerQuantization)

        // 2) Load tokenizer + text encoder + VAE for dataset prep
        let tokenizer = try loadTokenizer(from: tokenizerComponent.directoryURL)

        let textEncoderConfig = try loadTextEncoderConfig(from: textEncoderComponent.directoryURL)
        let textEncoder = QwenTextEncoder(configuration: textEncoderConfig)
        try await loadTextEncoderWeights(
            from: textEncoderComponent.directoryURL,
            to: textEncoder,
            quantization: textEncoderQuantization
        )

        let vaeConfig = try loadVAEConfig(from: vaeComponent.directoryURL)
        let vae = AutoencoderKL(configuration: vaeConfig)
        let vaeWeightsURL = Flux2KleinGenerator.checkpointFileURL(
            in: vaeComponent.directoryURL,
            filename: "diffusion_pytorch_model.safetensors"
        )
        let (bnMean, bnVar) = try loadBatchNormStats(from: vaeWeightsURL)
        try loadVAEWeights(from: vaeWeightsURL, to: vae)

        progressHandler?(Flux2KleinLoRATrainingProgress(stage: .loadingModels, fraction: 1))

        // 3) Encode dataset (prompt embeds + latents). Keep everything on-device.
        struct PreparedPrompt {
            let imageURL: URL
            let inputImageURL: URL?
            let promptEmbeds: MLXArray
        }

        struct PreparedExample {
            let cleanLatents: MLXArray
            let promptEmbeds: MLXArray
            let referenceLatents: MLXArray?
        }

        struct PhasePlan {
            let width: Int
            let height: Int
            let steps: Int
            let sampleIndices: [Int]
        }

        struct PhaseData {
            let phase: PhasePlan
            let patchedHeight: Int
            let patchedWidth: Int
            let seqLen: Int
            let imgIds: MLXArray
            let isEdit: Bool
            let prepared: [PreparedExample]?
            let preparedCount: Int
            let cache: TrainingDataCache?
        }

        let adaptiveResolution = config.maxResolution != nil

        // Encode prompt embeds once (resolution-independent).
        var preparedPrompts: [PreparedPrompt] = []
        preparedPrompts.reserveCapacity(examples.count)
        for example in examples {
            let caption = example.caption.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !caption.isEmpty else {
                throw Flux2KleinLoRATrainerError.captionEmpty(example.imageURL)
            }
            let promptEmbeds = try encodePromptEmbeds(
                prompt: caption,
                tokenizer: tokenizer,
                textEncoder: textEncoder,
                maxLength: config.maxTextLength
            ).asType(.bfloat16)
            MLX.eval(promptEmbeds)
            preparedPrompts.append(
                PreparedPrompt(
                    imageURL: example.imageURL,
                    inputImageURL: example.inputImageURL,
                    promptEmbeds: promptEmbeds
                )
            )
        }

        let allSampleIndices = Array(preparedPrompts.indices)
        let phasePlans: [PhasePlan] = try {
            if adaptiveResolution {
                var buckets: [LoRAResolvedResolution: [Int]] = [:]
                for (index, item) in preparedPrompts.enumerated() {
                    let dims = try LoRATrainingResolution.resolveFromImage(
                        at: item.imageURL,
                        maxResolution: config.maxResolution,
                        multiple: 16
                    )
                    buckets[dims, default: []].append(index)
                }

                let orderedBuckets = buckets.keys.sorted { lhs, rhs in
                    let lhsArea = Int64(lhs.width) * Int64(lhs.height)
                    let rhsArea = Int64(rhs.width) * Int64(rhs.height)
                    if lhsArea == rhsArea {
                        if lhs.width == rhs.width {
                            return lhs.height > rhs.height
                        }
                        return lhs.width > rhs.width
                    }
                    return lhsArea > rhsArea
                }
                let bucketSizes = orderedBuckets.map { buckets[$0]?.count ?? 0 }
                let bucketSteps = LoRATrainingResolution.allocateSteps(
                    totalSteps: config.trainingSteps,
                    bucketSizes: bucketSizes
                )

                return zip(orderedBuckets, bucketSteps).compactMap { bucket, steps in
                    guard steps > 0, let indices = buckets[bucket], !indices.isEmpty else { return nil }
                    return PhasePlan(width: bucket.width, height: bucket.height, steps: steps, sampleIndices: indices)
                }
            }

            let progressivePhases = TrainingSchedule.progressivePhases(
                targetWidth: config.width,
                targetHeight: config.height,
                totalSteps: config.trainingSteps,
                enabled: config.progressive,
                benchmarkSteps: config.benchmarkSteps,
                multiple: 16
            )
            return progressivePhases.map {
                PhasePlan(
                    width: $0.width,
                    height: $0.height,
                    steps: $0.steps,
                    sampleIndices: allSampleIndices
                )
            }
        }()
        guard !phasePlans.isEmpty else {
            throw NSError(
                domain: "Flux2KleinLoRATrainer",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No training phases were produced from dataset resolution bucketing."]
            )
        }

        if progressHandler != nil, phasePlans.count > 1 {
            let label = adaptiveResolution ? "Adaptive resolution buckets" : "Progressive schedule"
            let schedule = phasePlans
                .map { "\($0.width)x\($0.height)@\($0.steps)" }
                .joined(separator: " -> ")
            FileHandle.standardError.write(Data("[LoRATrainer] \(label): \(schedule)\n".utf8))
        }

        let resolutionBucketSummary = phasePlans
            .map { "\($0.width)x\($0.height):\($0.sampleIndices.count)@\($0.steps)" }
            .joined(separator: ";")
        let checkpointPhaseSchedule = phasePlans.map {
            LoRATrainingCheckpointState.Phase(
                width: $0.width,
                height: $0.height,
                steps: $0.steps,
                sampleCount: $0.sampleIndices.count
            )
        }
        let checkpointConfigSnapshot: [String: String] = [
            "model": modelURL.path,
            "variant": modelManifest.variant?.rawValue ?? "",
            "dataset_root": config.datasetRoot ?? "",
            "size": "\(config.width)x\(config.height)",
            "max_resolution": config.maxResolution.map { "\($0)" } ?? "",
            "resolution_buckets": resolutionBucketSummary,
            "scheduler_steps": "\(config.schedulerSteps)",
            "training_steps": "\(config.trainingSteps)",
            "batch_size": "\(config.batchSize)",
            "learning_rate": "\(config.learningRate)",
            "rank": "\(config.loraRank)",
            "alpha": "\(config.loraAlpha)",
            "lora_target_mode": config.loraTargetMode.rawValue,
            "caption_dropout": "\(config.captionDropout)",
            "max_text_length": "\(config.maxTextLength)",
            "checkpoint_interval": config.checkpointInterval.map { "\($0)" } ?? "",
            "sample_interval": config.sampleInterval.map { "\($0)" } ?? "",
            "progressive": "\(config.progressive)",
            "gradient_checkpointing": "\(config.gradientCheckpointing)",
            "low_ram": "\(config.lowRam)",
            "timestep_sampling": config.timestepSampling.rawValue,
            "timestep_loss_weighting": config.timestepLossWeighting.rawValue,
            "loss_weighting": config.lossWeighting.rawValue,
            "timestep_low": "\(config.timestepLow)",
            "timestep_high": config.timestepHigh.map { "\($0)" } ?? "",
            "lora_target_suffixes": (config.loraTargetSuffixes ?? Flux2LoRAInjector.defaultTargetSuffixes).joined(separator: ","),
            "lora_target_ranks": serializedTargetRanks,
            "lora_target_rank_suffixes": serializedTargetRankSuffixes,
            "edit_mode": "\(hasEditExamples)",
            "config_fingerprint": configFingerprint,
        ]

        let totalEncodingSteps = preparedPrompts.count + phasePlans.reduce(0) { $0 + $1.sampleIndices.count }
        var encodingStep = 0

        // Pre-compute empty prompt embedding for caption dropout
        let emptyPromptEmbeds: MLXArray? = config.captionDropout > 0 ? try {
            let embeds = try encodePromptEmbeds(
                prompt: "",
                tokenizer: tokenizer,
                textEncoder: textEncoder,
                maxLength: config.maxTextLength
            ).asType(.bfloat16)
            MLX.eval(embeds)
            return embeds
        }() : nil

        // Count prompt-encode progress after prompt encoding has completed.
        if totalEncodingSteps > 0 {
            encodingStep = preparedPrompts.count
            progressHandler?(Flux2KleinLoRATrainingProgress(
                stage: .encodingDataset(current: encodingStep, total: totalEncodingSteps),
                fraction: Float(encodingStep) / Float(totalEncodingSteps)
            ))
        }

        // Prepare per-phase latents (these are the only shape-dependent dataset artifacts).
        var phaseData: [PhaseData] = []
        phaseData.reserveCapacity(phasePlans.count)
        var lowRamCaches: [TrainingDataCache] = []
        defer {
            for cache in lowRamCaches {
                try? cache.clear()
            }
        }

        for (phaseIndex, phase) in phasePlans.enumerated() {
            let patchedHeight = phase.height / 16
            let patchedWidth = phase.width / 16
            let seqLen = patchedHeight * patchedWidth

            let imgIds: MLXArray = hasEditExamples
                ? Flux2PosEmbed.prepareMultiImageIds(
                    imageCount: 2,
                    height: patchedHeight,
                    width: patchedWidth,
                    tCoords: [0, 10]
                )
                : Flux2PosEmbed.prepareImageIds(height: patchedHeight, width: patchedWidth)
            let phaseCache: TrainingDataCache? = try {
                guard config.lowRam else { return nil }
                let cacheDir = outputURL.deletingLastPathComponent()
                    .appendingPathComponent(".zero_cache", isDirectory: true)
                    .appendingPathComponent("flux2-phase\(phaseIndex)-\(UUID().uuidString)", isDirectory: true)
                let cache = TrainingDataCache(cacheDir: cacheDir)
                try cache.initialize(wipe: true)
                lowRamCaches.append(cache)
                return cache
            }()

            var prepared: [PreparedExample] = []
            if phaseCache == nil {
                prepared.reserveCapacity(phase.sampleIndices.count)
            }

            for (localIndex, promptIndex) in phase.sampleIndices.enumerated() {
                try Task.checkCancellation()
                encodingStep += 1
                progressHandler?(Flux2KleinLoRATrainingProgress(
                    stage: .encodingDataset(current: encodingStep, total: totalEncodingSteps),
                    fraction: Float(encodingStep) / Float(max(totalEncodingSteps, 1))
                ))

                let item = preparedPrompts[promptIndex]
                let cleanLatents = try encodeTrainingImage(
                    item.imageURL,
                    vae: vae,
                    width: phase.width,
                    height: phase.height,
                    patchedHeight: patchedHeight,
                    patchedWidth: patchedWidth,
                    bnMean: bnMean,
                    bnVar: bnVar
                )
                MLX.eval(cleanLatents)
                let referenceLatents: MLXArray? = try {
                    guard hasEditExamples else { return nil }
                    guard let inputImageURL = item.inputImageURL else {
                        throw Flux2KleinLoRATrainerError.mixedDatasetModes
                    }
                    let reference = try encodeTrainingImage(
                        inputImageURL,
                        vae: vae,
                        width: phase.width,
                        height: phase.height,
                        patchedHeight: patchedHeight,
                        patchedWidth: patchedWidth,
                        bnMean: bnMean,
                        bnVar: bnVar
                    )
                    MLX.eval(reference)
                    return reference
                }()
                if let phaseCache {
                    try phaseCache.save(
                        id: localIndex,
                        latents: cleanLatents,
                        cond: item.promptEmbeds,
                        referenceLatents: referenceLatents,
                        width: phase.width,
                        height: phase.height
                    )
                    Memory.clearCache()
                } else {
                    prepared.append(
                        PreparedExample(
                            cleanLatents: cleanLatents,
                            promptEmbeds: item.promptEmbeds,
                            referenceLatents: referenceLatents
                        )
                    )
                }
            }

            phaseData.append(
                PhaseData(
                    phase: phase,
                    patchedHeight: patchedHeight,
                    patchedWidth: patchedWidth,
                    seqLen: seqLen,
                    imgIds: imgIds,
                    isEdit: hasEditExamples,
                    prepared: phaseCache == nil ? prepared : nil,
                    preparedCount: phase.sampleIndices.count,
                    cache: phaseCache
                )
            )
        }

        progressHandler?(Flux2KleinLoRATrainingProgress(
            stage: .encodingDataset(current: totalEncodingSteps, total: totalEncodingSteps),
            fraction: 1
        ))

        // 4) Inject LoRA wrappers (zero-init B matrix like ai-toolkit/kohya)
        let targetSuffixes = config.loraTargetSuffixes ?? Flux2LoRAInjector.defaultTargetSuffixes
        let resolvedTargetRanks = try config.loraTargetRanks ?? config.loraTargetRankSuffixes.map { targetRankSuffixes in
            try Flux2LoRAInjector.resolveTargetRanks(
                in: transformer,
                defaultRank: config.loraRank,
                targetSuffixes: targetSuffixes,
                targetRankSuffixes: targetRankSuffixes
            )
        }
        let loraLayers = try Flux2LoRAInjector.inject(
            into: transformer,
            rank: config.loraRank,
            alpha: config.loraAlpha,
            targetMode: config.loraTargetMode,
            targetSuffixes: targetSuffixes,
            targetRanks: resolvedTargetRanks,
            zeroInitUp: true
        )

        var resumeStep = 0
        var resumedRNGState: UInt64? = nil
        var resumedSeed: UInt64? = nil
        var resumeIteratorCursor: MFluxResumeIteratorCompat.Cursor? = nil
        var resolvedResumeCheckpoint: LoRAResolvedCheckpoint? = nil
        var resumeSidecar: LoRATrainingCheckpointState? = nil
        var resumeRunManifest: LoRATrainingRunManifest? = nil

        // Load existing weights if resuming from a previous LoRA checkpoint.
        if let resumeFromLoRA {
            let resolvedCheckpoint = try LoRACheckpointResolver.resolve(resumeFromLoRA)
            resolvedResumeCheckpoint = resolvedCheckpoint
            let resumeURL = resolvedCheckpoint.checkpointURL
            let loadedCount = try Flux2LoRAInjector.loadWeights(
                from: resumeURL,
                into: loraLayers,
                optimizerStateURL: resolvedCheckpoint.optimizerStateURL
            )
            let metadata: [String: String]? = {
                guard let tuple = try? MLX.loadArraysAndMetadata(url: resumeURL) else { return nil }
                return tuple.1
            }()
            let sidecar = try LoRATrainingCheckpointState.load(nextTo: resumeURL)
            resumeSidecar = sidecar
            let runManifest = Self.loadRunManifestLenient(nextTo: resumeURL)
            resumeRunManifest = runManifest
            let iteratorState: MFluxResumeIteratorCompat.State? = {
                guard let iteratorStateURL = resolvedCheckpoint.iteratorStateURL else {
                    return nil
                }
                return MFluxResumeIteratorCompat.loadState(from: iteratorStateURL)
            }()

            if let runManifest {
                try Self.validateResumeRunManifest(
                    runManifest,
                    expectedModel: "flux2-klein",
                    expectedIsEdit: hasEditExamples,
                    expectedDataFingerprint: runDataFingerprint,
                    expectedDatasetFingerprint: datasetFingerprint,
                    expectedConfigFingerprint: configFingerprint,
                    expectedPhaseSchedule: checkpointPhaseSchedule,
                    checkpointName: resumeURL.lastPathComponent
                )
                resumeStep = runManifest.step
                resumedRNGState = runManifest.rngState
                resumedSeed = runManifest.seed
            } else if let sidecar {
                if let fingerprint = sidecar.datasetFingerprint, fingerprint != datasetFingerprint {
                    throw NSError(
                        domain: "Flux2KleinLoRATrainer",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "Resume dataset fingerprint mismatch for \(resumeURL.lastPathComponent)."]
                    )
                }
                if let fingerprint = sidecar.configFingerprint, fingerprint != configFingerprint {
                    throw NSError(
                        domain: "Flux2KleinLoRATrainer",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "Resume config fingerprint mismatch for \(resumeURL.lastPathComponent)."]
                    )
                }
                if let schedule = sidecar.phaseSchedule,
                   !LoRATrainingCheckpointState.scheduleMatches(expected: checkpointPhaseSchedule, actual: schedule) {
                    throw NSError(
                        domain: "Flux2KleinLoRATrainer",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "Resume phase schedule mismatch for \(resumeURL.lastPathComponent)."]
                    )
                }
                if let cursor = sidecar.phaseCursor,
                   let expectedCursor = LoRATrainingCheckpointState.cursor(
                    forCompletedStep: sidecar.step,
                    phaseSchedule: checkpointPhaseSchedule
                   ),
                   cursor != expectedCursor {
                    throw NSError(
                        domain: "Flux2KleinLoRATrainer",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "Resume iterator cursor mismatch for \(resumeURL.lastPathComponent)."]
                    )
                }
                resumeStep = sidecar.step
                resumedRNGState = sidecar.rngState
                resumedSeed = sidecar.seed
            } else if let iteratorState {
                resumeStep = iteratorState.step
                resumedSeed = iteratorState.seed
            } else if let metadata,
                      let stepString = metadata["step"],
                      let parsedStep = Int(stepString) {
                resumeStep = parsedStep
                if let seedString = metadata["seed"] {
                    resumedSeed = UInt64(seedString)
                }
            }

            if let iteratorState,
               iteratorState.step == resumeStep {
                if resumedSeed == nil {
                    resumedSeed = iteratorState.seed
                }
                resumeIteratorCursor = iteratorState.cursor
            }

            if resumeStep >= config.trainingSteps {
                throw NSError(
                    domain: "Flux2KleinLoRATrainer",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Resume step \(resumeStep) is already >= requested total steps \(config.trainingSteps)."]
                )
            }

            print("[LoRATrainer] Resumed from \(resumeURL.lastPathComponent): loaded \(loadedCount)/\(loraLayers.count) layers, start step \(resumeStep)")
        }
        defer { resolvedResumeCheckpoint?.cleanup() }

        // Freeze base model; unfreeze only LoRA params.
        try transformer.freeze(recursive: true, keys: nil, strict: false)
        for layer in loraLayers.values {
            guard let module = layer as? Module else { continue }
            try module.unfreeze(recursive: false, keys: ["loraDown", "loraUp"], strict: true)
        }
        // Gradient checkpointing follows the explicit config/CLI flag, with
        // the shared resolution-aware default as fallback (peak training
        // pixels decide: capped-resolution recipe runs keep the uncompiled
        // recompute path off). The cache cap bounds the MLX buffer pool,
        // which otherwise pins the footprint at the worst transient spike.
        let peakTrainingPixels: Int = {
            let target = config.width * config.height
            if let maxResolution = config.maxResolution {
                return min(target, maxResolution * maxResolution)
            }
            return target
        }()
        let effectiveGradientCheckpointing = config.gradientCheckpointing
            || LoRATrainingEnvironment.gradientCheckpointingEnabled(trainingPixels: peakTrainingPixels)
        transformer.gradientCheckpointing = effectiveGradientCheckpointing
        if LoRATrainingEnvironment.trainingCacheLimitGB > 0 {
            MLX.Memory.cacheLimit = LoRATrainingEnvironment.trainingCacheLimitGB * 1_073_741_824
        }
        FileHandle.standardError.write(Data(
            "[flux2-lora-train] grad_checkpoint=\(effectiveGradientCheckpointing) peak_pixels=\(peakTrainingPixels) cache_limit_gb=\(LoRATrainingEnvironment.trainingCacheLimitGB)\n".utf8
        ))

        MLX.eval(transformer)
        progressHandler?(Flux2KleinLoRATrainingProgress(stage: .injectingLoRA(layerCount: loraLayers.count), fraction: 1))

        // Initialize per-LoRA Adam state for fast, resumable training.
        Self.initializeAdamStateIfNeeded(for: loraLayers)
        let loraState = LoRAState(loraLayers: loraLayers)
        MLX.eval(loraState)
        let loraLayerList = loraState.layers

        // Initialize EMA if enabled
        var emaState: LoRAEMAState? = nil
        if config.emaDecay > 0 {
            emaState = LoRAEMAState(decay: config.emaDecay)
            emaState?.initialize(from: loraLayers)
        }

        // 5) Training loop (FlowMatch / rectified flow loss)
        let txtIds = Flux2PosEmbed.prepareTextIds(seqLen: config.maxTextLength, numAxes: 4)

        // Pre-compute timestep sampling weights (independent of resolution)
        let timestepWeights = computeTimestepWeights(
            numSteps: config.schedulerSteps,
            strategy: config.timestepSampling
        )

        // Optional timestep loss weighting (ai-toolkit `timestep_type: weighted`).
        let timestepLossWeights = computeTimestepLossWeights(
            numSteps: config.schedulerSteps,
            strategy: config.timestepLossWeighting
        )

        let benchmarkSteps = config.benchmarkSteps
        let benchmarkWarmupSteps = config.benchmarkWarmupSteps
        let benchmarkEndStep = benchmarkSteps.map { benchmarkWarmupSteps + $0 }
        var benchmarkStartTime: CFTimeInterval? = nil
        if let benchmarkSteps {
            FileHandle.standardError.write(
                Data("[LoRATrainer] Benchmark running: warmup \(benchmarkWarmupSteps) + measure \(benchmarkSteps)\n".utf8)
            )
        }

        let effectiveSeed = resumedSeed ?? resolvedSeed
        var rng = resumedRNGState.map { SplitMix64(rawState: $0) } ?? SplitMix64(seed: effectiveSeed)
        var lastRNGState = rng.rawState
        var currentStep = resumeStep

        if resumeStep > 0,
           let resumeCheckpointURL = resolvedResumeCheckpoint?.checkpointURL {
            LoRATrainingResumeArtifacts.restore(
                from: resumeCheckpointURL,
                sidecar: Self.syntheticSidecar(from: resumeRunManifest) ?? resumeSidecar,
                lossStateURL: resolvedResumeCheckpoint?.lossStateURL,
                to: outputURL
            )
        }

        let metricsLogger = try LoRATrainingMetricsLogger(
            baseOutputURL: outputURL,
            resumeExisting: resumeStep > 0
        )
        let lossCSVFile = metricsLogger.csvURL.lastPathComponent
        let lossHTMLFile = metricsLogger.htmlURL.lastPathComponent

        // These are constants for the whole run (captured by the compiled closure).
        let lossWeighting = config.lossWeighting

        let beta1 = MLXArray(config.adamBeta1)
        let beta2 = MLXArray(config.adamBeta2)
        let oneMinusBeta1 = MLXArray(1.0 - config.adamBeta1)
        let oneMinusBeta2 = MLXArray(1.0 - config.adamBeta2)
        let baseLR = config.learningRate
        let eps = MLXArray(config.adamEps)
        let useWeightDecay = config.adamWeightDecay != 0
        let weightDecayFactor = config.adamWeightDecay

        // LR scheduling helper (computes scheduled LR for a given step)
        let useCosineScheduler = config.useCosineScheduler
        let lrWarmupSteps = config.lrWarmupSteps
        let lrMinFactor = config.lrMinFactor
        let totalSteps = config.trainingSteps
        func computeScheduledLR(step: Int) -> Float {
            guard useCosineScheduler else { return baseLR }
            if step < lrWarmupSteps {
                // Linear warmup
                return baseLR * Float(step + 1) / Float(max(lrWarmupSteps, 1))
            } else {
                // Cosine annealing
                let progress = Float(step - lrWarmupSteps) / Float(max(totalSteps - lrWarmupSteps, 1))
                let cosineDecay = 0.5 * (1.0 + cos(Float.pi * progress))
                return baseLR * (lrMinFactor + (1.0 - lrMinFactor) * cosineDecay)
            }
        }

        do {
            var globalStep = resumeStep
            var remainingSkippedSteps = resumeStep
            var resumeIteratorCursorConsumed = false

            for (phaseIndex, phase) in phaseData.enumerated() {
                if globalStep >= config.trainingSteps { break }
                if phase.phase.steps <= 0 { continue }
                if remainingSkippedSteps >= phase.phase.steps {
                    remainingSkippedSteps -= phase.phase.steps
                    continue
                }
                let phaseStartStep = remainingSkippedSteps
                remainingSkippedSteps = 0

                let seqLen = phase.seqLen
                let trainingSigmaValues = computeTrainingSigmas(
                    numSteps: config.schedulerSteps,
                    numTrainTimesteps: 1000,
                    imageSeqLen: config.timestepSampling == .shift ? seqLen : nil
                )
                let prepared = phase.prepared
                let phaseCache = phase.cache
                let preparedCount = phase.preparedCount
                let imgIds = phase.imgIds
                let isEditPhase = phase.isEdit
                let canUseResumeCursor = !resumeIteratorCursorConsumed &&
                    (resumeIteratorCursor?.permutation.count == preparedCount)
                var phaseIteratorCursor: MFluxResumeIteratorCompat.Cursor?
                if canUseResumeCursor {
                    phaseIteratorCursor = resumeIteratorCursor
                    resumeIteratorCursorConsumed = true
                } else {
                    let phaseSeed = effectiveSeed &+ (UInt64(phaseIndex + 1) &* 0x9E3779B97F4A7C15)
                    phaseIteratorCursor = MFluxResumeIteratorCompat.makePythonCursor(
                        sampleCount: preparedCount,
                        seed: phaseSeed
                    ) ?? MFluxResumeIteratorCompat.makeLocalCursor(
                        sampleCount: preparedCount,
                        seed: phaseSeed
                    )
                }
                if prepared == nil, phaseCache == nil {
                    throw NSError(
                        domain: "Flux2KleinLoRATrainer",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "Missing phase training data at runtime."]
                    )
                }

                // Build loss+grad function (resolution-specific via imgIds/seqLen)
                let lossAndGrad = valueAndGrad(model: transformer) { model, arrays in
                    let clean = arrays[0]
                    let noise = arrays[1]
                    let promptEmbeds = arrays[2]
                    let sigmaIndex = isEditPhase ? 4 : 3
                    let timestepLossIndex = isEditPhase ? 5 : 4
                    let sigmaInput = arrays[sigmaIndex].asType(.float32)
                    let timestepLossInput = arrays[timestepLossIndex].asType(.float32)
                    let sigmaBroadcast = sigmaInput.ndim == 0
                        ? sigmaInput
                        : sigmaInput.expandedDimensions(axis: 1).expandedDimensions(axis: 2)
                    let timestepLossBroadcast = timestepLossInput.ndim == 0
                        ? timestepLossInput
                        : timestepLossInput.expandedDimensions(axis: 1).expandedDimensions(axis: 2)

                    let latentsT = ((MLXArray(1.0) - sigmaBroadcast) * clean.asType(.float32) + sigmaBroadcast * noise.asType(.float32))
                        .asType(.bfloat16)

                    // FLUX.2 expects timesteps in 0-1000 range, not 0-1
                    let timestep = sigmaInput * MLXArray(1000.0)
                    let hiddenStates: MLXArray
                    if isEditPhase {
                        let referenceLatents = arrays[3]
                        hiddenStates = MLX.concatenated([latentsT, referenceLatents], axis: 1)
                    } else {
                        hiddenStates = latentsT
                    }

                    let predAll = model(
                        hiddenStates: hiddenStates,
                        encoderHiddenStates: promptEmbeds,
                        timestep: timestep,
                        imgIds: imgIds,
                        txtIds: txtIds
                    )
                    let pred = isEditPhase ? predAll[0..., 0..<seqLen, 0...] : predAll

                    let error = (clean.asType(.float32) + pred.asType(.float32) - noise.asType(.float32)).square()
                    let snrWeight = Self.computeSNRWeight(sigma: sigmaInput, strategy: lossWeighting)
                    let snrWeightBroadcast = snrWeight.ndim == 0
                        ? snrWeight
                        : snrWeight.expandedDimensions(axis: 1).expandedDimensions(axis: 2)
                    let weightedError = error * snrWeightBroadcast * timestepLossBroadcast
                    let loss = weightedError.mean()

                    return [loss]
                }

                // Compile the full training step (forward + backward + LoRA update) for this shape.
                // Checkpointed blocks stay uncompiled: nesting checkpoint
                // closures inside a compiled step is unproven with this
                // mlx-swift.
                var compiledTrainStep: (([MLXArray]) -> [MLXArray])? = nil
                if config.useCompile, !transformer.gradientCheckpointing {
                    let state: [any Updatable] = [loraState]
                    if isEditPhase {
                        // Input arrays: [clean, noise, promptEmbeds, referenceLatents, sigma, timestepLossWeight, lr, oneMinusLrWd]
                        compiledTrainStep = compile(inputs: state, outputs: state) { inputs -> [MLXArray] in
                            let lrInput = inputs[6]
                            let oneMinusLrWdInput = inputs[7]
                            let (values, grads) = lossAndGrad(transformer, Array(inputs[0..<6]))
                            let gradMap = Dictionary(uniqueKeysWithValues: grads.flattened())
                            Self.applyAdamW(
                                loraLayers: loraLayerList,
                                gradMap: gradMap,
                                lr: lrInput,
                                beta1: beta1,
                                beta2: beta2,
                                oneMinusBeta1: oneMinusBeta1,
                                oneMinusBeta2: oneMinusBeta2,
                                eps: eps,
                                oneMinusLrWd: oneMinusLrWdInput,
                                useWeightDecay: useWeightDecay
                            )
                            return values
                        }
                    } else {
                        // Input arrays: [clean, noise, promptEmbeds, sigma, timestepLossWeight, lr, oneMinusLrWd]
                        compiledTrainStep = compile(inputs: state, outputs: state) { inputs -> [MLXArray] in
                            let lrInput = inputs[5]
                            let oneMinusLrWdInput = inputs[6]
                            let (values, grads) = lossAndGrad(transformer, Array(inputs[0..<5]))
                            let gradMap = Dictionary(uniqueKeysWithValues: grads.flattened())
                            Self.applyAdamW(
                                loraLayers: loraLayerList,
                                gradMap: gradMap,
                                lr: lrInput,
                                beta1: beta1,
                                beta2: beta2,
                                oneMinusBeta1: oneMinusBeta1,
                                oneMinusBeta2: oneMinusBeta2,
                                eps: eps,
                                oneMinusLrWd: oneMinusLrWdInput,
                                useWeightDecay: useWeightDecay
                            )
                            return values
                        }
                    }
                }

                for _ in phaseStartStep..<phase.phase.steps {
                    if globalStep >= config.trainingSteps { break }

                    currentStep = globalStep
                    try Task.checkCancellation()

                    if benchmarkStartTime == nil, let end = benchmarkEndStep, globalStep == benchmarkWarmupSteps, globalStep < end {
                        // Ensure warmup work is finished so benchmark timing reflects steady-state throughput.
                        Stream.gpu.synchronize()
                        benchmarkStartTime = CFAbsoluteTimeGetCurrent()
                    }

                    let requestedBatchSize = min(config.batchSize, preparedCount)
                    let sampledIndices = MFluxResumeIteratorCompat.nextBatchIndices(
                            requestedBatchSize: requestedBatchSize,
                            sampleCount: preparedCount,
                            cursor: &phaseIteratorCursor
                        ) ?? []
                    guard !sampledIndices.isEmpty else {
                        throw NSError(
                            domain: "Flux2KleinLoRATrainer",
                            code: 1,
                            userInfo: [NSLocalizedDescriptionKey: "Failed to sample a training batch from phase iterator state."]
                        )
                    }
                    let batchSize = sampledIndices.count
                    let trainingSeedPairs = MFluxResumeIteratorCompat.nextTrainingSeedPairs(
                        batchSize: batchSize,
                        cursor: &phaseIteratorCursor
                    )
                    let stepSeedPairs: [(time: UInt64, noise: UInt64)] = {
                        if let trainingSeedPairs, trainingSeedPairs.count == batchSize {
                            return trainingSeedPairs
                        }
                        // Fall back to local trainer RNG if iterator-compatible RNG state is unavailable.
                        return (0..<batchSize).map { _ in
                            (
                                time: UInt64(UInt32(truncatingIfNeeded: rng.next())),
                                noise: UInt64(UInt32(truncatingIfNeeded: rng.next()))
                            )
                        }
                    }()
                    resumeIteratorCursor = phaseIteratorCursor

                    let cleanBatch: MLXArray
                    let promptBatch: MLXArray
                    let referenceBatch: MLXArray?

                    if batchSize == 1 {
                        // Fast path: avoid concatenation.
                        let idx = sampledIndices[0]
                        let item: PreparedExample
                        if let phaseCache {
                            let cached = try phaseCache.load(id: idx)
                            item = PreparedExample(
                                cleanLatents: cached.latents,
                                promptEmbeds: cached.cond,
                                referenceLatents: cached.referenceLatents
                            )
                        } else if let prepared {
                            item = prepared[idx]
                        } else {
                            throw NSError(
                                domain: "Flux2KleinLoRATrainer",
                                code: 1,
                                userInfo: [NSLocalizedDescriptionKey: "Missing phase training data at runtime."]
                            )
                        }
                        cleanBatch = item.cleanLatents
                        referenceBatch = item.referenceLatents

                        let useEmptyPrompt = emptyPromptEmbeds != nil &&
                            Float(rng.next() % 1000) / 1000.0 < config.captionDropout
                        if useEmptyPrompt, let empty = emptyPromptEmbeds {
                            promptBatch = empty
                        } else {
                            promptBatch = item.promptEmbeds
                        }
                    } else {
                        var cleanBatchParts: [MLXArray] = []
                        var promptBatchParts: [MLXArray] = []
                        var referenceBatchParts: [MLXArray] = []
                        cleanBatchParts.reserveCapacity(batchSize)
                        promptBatchParts.reserveCapacity(batchSize)
                        if isEditPhase { referenceBatchParts.reserveCapacity(batchSize) }

                        for idx in sampledIndices {
                            let item: PreparedExample
                            if let phaseCache {
                                let cached = try phaseCache.load(id: idx)
                                item = PreparedExample(
                                    cleanLatents: cached.latents,
                                    promptEmbeds: cached.cond,
                                    referenceLatents: cached.referenceLatents
                                )
                            } else if let prepared {
                                item = prepared[idx]
                            } else {
                                throw NSError(
                                    domain: "Flux2KleinLoRATrainer",
                                    code: 1,
                                    userInfo: [NSLocalizedDescriptionKey: "Missing phase training data at runtime."]
                                )
                            }
                            cleanBatchParts.append(item.cleanLatents)
                            if isEditPhase {
                                guard let reference = item.referenceLatents else {
                                    throw Flux2KleinLoRATrainerError.mixedDatasetModes
                                }
                                referenceBatchParts.append(reference)
                            }

                            let useEmptyPrompt = emptyPromptEmbeds != nil &&
                                Float(rng.next() % 1000) / 1000.0 < config.captionDropout
                            if useEmptyPrompt, let empty = emptyPromptEmbeds {
                                promptBatchParts.append(empty)
                            } else {
                                promptBatchParts.append(item.promptEmbeds)
                            }
                        }

                        cleanBatch = MLX.concatenated(cleanBatchParts, axis: 0)
                        promptBatch = MLX.concatenated(promptBatchParts, axis: 0)
                        referenceBatch = isEditPhase ? MLX.concatenated(referenceBatchParts, axis: 0) : nil
                    }

                    if isEditPhase, referenceBatch == nil {
                        throw Flux2KleinLoRATrainerError.mixedDatasetModes
                    }

                    let timestepHigh = config.timestepHigh ?? config.schedulerSteps
                    var timestepIndices: [Int] = []
                    var noiseParts: [MLXArray] = []
                    timestepIndices.reserveCapacity(batchSize)
                    noiseParts.reserveCapacity(batchSize)
                    for seedPair in stepSeedPairs {
                        let t: Int = {
                            var timestepRNG = SplitMix64(rawState: seedPair.time)
                            return sampleWeightedTimestep(
                                weights: timestepWeights,
                                rng: &timestepRNG,
                                low: config.timestepLow,
                                high: timestepHigh
                            )
                        }()
                        timestepIndices.append(t)
                        let noisePart = MLXRandom.normal(
                            [1, seqLen, 128],
                            key: MLXRandom.key(seedPair.noise)
                        ).asType(.bfloat16)
                        noiseParts.append(noisePart)
                    }
                    let noise = batchSize == 1
                        ? noiseParts[0]
                        : MLX.concatenated(noiseParts, axis: 0)
                    let sigma = MLXArray(timestepIndices.map { trainingSigmaValues[$0] }).asType(.float32)
                    let timestepLossWeight = MLXArray(timestepIndices.map { timestepLossWeights[$0] }).asType(.float32)
                    lastRNGState = rng.rawState

                    // Compute scheduled learning rate for this step
                    let currentLR = computeScheduledLR(step: globalStep)
                    let lr = MLXArray(currentLR)
                    let oneMinusLrWd = useWeightDecay
                        ? MLXArray(1.0 - currentLR * weightDecayFactor)
                        : MLXArray(1.0)

                    // Run training step (compiled or not)
                    let loss: MLXArray
                    if let compiledStep = compiledTrainStep {
                        if isEditPhase {
                            guard let referenceBatch else {
                                throw Flux2KleinLoRATrainerError.mixedDatasetModes
                            }
                            loss = compiledStep([cleanBatch, noise, promptBatch, referenceBatch, sigma, timestepLossWeight, lr, oneMinusLrWd])[0]
                        } else {
                            loss = compiledStep([cleanBatch, noise, promptBatch, sigma, timestepLossWeight, lr, oneMinusLrWd])[0]
                        }
                    } else {
                        let values: [MLXArray]
                        let gradMap: [String: MLXArray]
                        if isEditPhase {
                            guard let referenceBatch else {
                                throw Flux2KleinLoRATrainerError.mixedDatasetModes
                            }
                            let (v, g) = lossAndGrad(transformer, [cleanBatch, noise, promptBatch, referenceBatch, sigma, timestepLossWeight])
                            values = v
                            gradMap = Dictionary(uniqueKeysWithValues: g.flattened())
                        } else {
                            let (v, g) = lossAndGrad(transformer, [cleanBatch, noise, promptBatch, sigma, timestepLossWeight])
                            values = v
                            gradMap = Dictionary(uniqueKeysWithValues: g.flattened())
                        }
                        loss = values[0]
                        Self.applyAdamW(
                            loraLayers: loraLayerList,
                            gradMap: gradMap,
                            lr: lr,
                            beta1: beta1,
                            beta2: beta2,
                            oneMinusBeta1: oneMinusBeta1,
                            oneMinusBeta2: oneMinusBeta2,
                            eps: eps,
                            oneMinusLrWd: oneMinusLrWd,
                            useWeightDecay: useWeightDecay
                        )
                    }

                    // Update EMA weights (optional). Ensure it doesn't build a lazy graph chain.
                    if let emaState {
                        emaState.update(from: loraLayers)
                    }

                    // Materialize loss + LoRA state (and EMA state) without a full CPU sync each step.
                    if let emaState {
                        MLX.asyncEval(loss, loraState, emaState)
                    } else {
                        MLX.asyncEval(loss, loraState)
                    }

                    // Benchmark mode: avoid per-step syncing/logging/saving to get clean throughput.
                    if benchmarkSteps == nil {
                        let shouldLog = (globalStep + 1) % max(config.logEvery, 1) == 0 || globalStep == config.trainingSteps - 1

                        let lossValue: Float? = {
                            guard shouldLog else { return nil }
                            // Sync only when we actually need to read scalar values.
                            Stream.gpu.synchronize()
                            return loss.item(Float.self)
                        }()

                        if let lossValue {
                            try metricsLogger.record(step: globalStep + 1, loss: lossValue)
                            var diagnostics = String(
                                format: "[flux2-lora-train] step=%d/%d loss=%.6f",
                                globalStep + 1, config.trainingSteps, lossValue
                            )
                            if let footprint = LoRATrainingEnvironment.currentPhysicalFootprintGB() {
                                diagnostics += String(format: " footprint_gb=%.1f", footprint)
                            }
                            FileHandle.standardError.write(Data((diagnostics + "\n").utf8))
                            progressHandler?(Flux2KleinLoRATrainingProgress(
                                stage: .training(step: globalStep + 1, total: config.trainingSteps, loss: lossValue),
                                fraction: Float(globalStep + 1) / Float(config.trainingSteps)
                            ))
                        } else {
                            progressHandler?(Flux2KleinLoRATrainingProgress(
                                stage: .training(step: globalStep + 1, total: config.trainingSteps, loss: nil),
                                fraction: Float(globalStep + 1) / Float(config.trainingSteps)
                            ))
                        }

                        // Save checkpoints and preview samples independently.
                        let step = globalStep + 1
                        let shouldCheckpoint = config.checkpointInterval.map { step % $0 == 0 } ?? false
                        let shouldGenerateSample = (config.sampleInterval.map { step % $0 == 0 } ?? false) && sampleHandler != nil
                        if (shouldCheckpoint || shouldGenerateSample), step < config.trainingSteps {
                            if shouldGenerateSample {
                                progressHandler?(Flux2KleinLoRATrainingProgress(
                                    stage: .sampling(step: step),
                                    fraction: Float(step) / Float(config.trainingSteps)
                                ))
                            }

                            let outputBaseName = outputURL.deletingPathExtension().lastPathComponent
                            let checkpointURL: URL
                            if shouldCheckpoint {
                                let checkpointDir = outputURL.deletingLastPathComponent()
                                    .appendingPathComponent("checkpoints", isDirectory: true)
                                try? FileManager.default.createDirectory(at: checkpointDir, withIntermediateDirectories: true)
                                checkpointURL = checkpointDir.appendingPathComponent("\(outputBaseName)-step\(step).safetensors")
                            } else {
                                let previewCheckpointDir = FileManager.default.temporaryDirectory
                                    .appendingPathComponent("mererun-lora-preview-checkpoints", isDirectory: true)
                                try? FileManager.default.createDirectory(at: previewCheckpointDir, withIntermediateDirectories: true)
                                checkpointURL = previewCheckpointDir.appendingPathComponent("\(outputBaseName)-step\(step).safetensors")
                            }

                            try LoRASafetensorsWriter.save(
                                loraLayers: loraLayers,
                                to: checkpointURL,
                                dtype: config.saveDType,
                                includeOptimizerState: true,
                                metadata: [
                                    "format": "mererun.flux2.lora",
                                    "base_model": "flux2-klein",
                                    "model_variant": modelManifest.variant?.rawValue ?? "",
                                    "step": "\(step)",
                                    "total_steps": "\(config.trainingSteps)",
                                    "seed": "\(effectiveSeed)",
                                    "rng_state": "\(lastRNGState)",
                                    "dataset_fingerprint": datasetFingerprint,
                                    "config_fingerprint": configFingerprint,
                                ]
                            )

                            if shouldCheckpoint {
                                let checkpointArchiveFile = LoRACheckpointArchive.archiveURL(for: checkpointURL).lastPathComponent
                                let checkpointState = LoRATrainingCheckpointState(
                                    format: "mererun.flux2.lora",
                                    baseModel: "flux2-klein",
                                    checkpointFile: checkpointURL.lastPathComponent,
                                    step: step,
                                    totalSteps: config.trainingSteps,
                                    seed: effectiveSeed,
                                    rngState: lastRNGState,
                                    datasetFingerprint: datasetFingerprint,
                                    configFingerprint: configFingerprint,
                                    phaseSchedule: checkpointPhaseSchedule,
                                    phaseCursor: LoRATrainingCheckpointState.cursor(
                                        forCompletedStep: step,
                                        phaseSchedule: checkpointPhaseSchedule
                                    ),
                                    configSnapshot: checkpointConfigSnapshot,
                                    lossCSVFile: lossCSVFile,
                                    lossHTMLFile: lossHTMLFile,
                                    manifestFile: LoRATrainingManifest.url(nextTo: checkpointURL).lastPathComponent
                                )
                                try checkpointState.write(nextTo: checkpointURL)
                                if shouldGenerateSample {
                                    await sampleHandler?(step, checkpointURL)
                                }
                                let checkpointSampleArtifacts = shouldGenerateSample
                                    ? Self.sampleArtifacts(
                                        in: outputURL.deletingLastPathComponent(),
                                        step: step
                                    )
                                    : []
                                let checkpointSampleFiles = checkpointSampleArtifacts
                                    .map(\.lastPathComponent)
                                    .joined(separator: ",")
                                let sidecarURL = LoRATrainingCheckpointState.url(nextTo: checkpointURL)
                                let manifestURL = LoRATrainingManifest.url(nextTo: checkpointURL)
                                let runManifestURL = LoRATrainingRunManifest.url(nextTo: checkpointURL)
                                let mfluxCompatArtifacts = try MFluxCheckpointCompatArtifactsWriter.write(
                                    checkpointURL: checkpointURL,
                                    step: step,
                                    seed: effectiveSeed,
                                    batchSize: config.batchSize,
                                    datasetCount: examples.count,
                                    loraAdapterFileName: checkpointURL.lastPathComponent,
                                    optimizerFileName: checkpointURL.lastPathComponent,
                                    iteratorCursor: resumeIteratorCursor,
                                    lossPoints: metricsLogger.allPoints(),
                                    configSnapshot: checkpointConfigSnapshot
                                )
                                let checkpointManifest = LoRATrainingManifest(
                                    format: "mererun.flux2.lora",
                                    baseModel: "flux2-klein",
                                    outputFile: checkpointURL.lastPathComponent,
                                    emaOutputFile: nil,
                                    training: LoRATrainingManifest.Training(
                                        width: config.width,
                                        height: config.height,
                                        trainingSteps: config.trainingSteps,
                                        batchSize: config.batchSize,
                                        learningRate: config.learningRate,
                                        seed: effectiveSeed,
                                        datasetCount: examples.count,
                                        checkpointInterval: config.checkpointInterval,
                                        sampleInterval: config.sampleInterval,
                                        samplePrompt: config.samplePrompt,
                                        emaDecay: config.emaDecay
                                    ),
                                    lora: LoRATrainingManifest.LoRA(
                                        rank: config.loraRank,
                                        alpha: config.loraAlpha,
                                        saveDType: String(describing: config.saveDType),
                                        includesOptimizerState: true
                                    ),
                                    extras: [
                                        "checkpoint": "true",
                                        "step": "\(step)",
                                        "max_text_length": "\(config.maxTextLength)",
                                        "scheduler_steps": "\(config.schedulerSteps)",
                                        "caption_dropout": "\(config.captionDropout)",
                                        "timestep_sampling": config.timestepSampling.rawValue,
                                        "timestep_loss_weighting": config.timestepLossWeighting.rawValue,
                                        "loss_weighting": config.lossWeighting.rawValue,
                                        "timestep_low": "\(config.timestepLow)",
                                        "timestep_high": config.timestepHigh.map { "\($0)" } ?? "",
                                        "progressive": "\(config.progressive)",
                                        "adaptive_resolution": "\(adaptiveResolution)",
                                        "max_resolution": config.maxResolution.map { "\($0)" } ?? "",
                                        "resolution_buckets": resolutionBucketSummary,
                                        "use_compile": "\(config.useCompile)",
                                        "gradient_checkpointing": "\(config.gradientCheckpointing)",
                                        "low_ram": "\(config.lowRam)",
                                        "edit_mode": "\(hasEditExamples)",
                                        "lora_target_mode": config.loraTargetMode.rawValue,
                                        "lora_target_suffixes": config.loraTargetSuffixes?.joined(separator: ",") ?? "",
                                        "lora_target_ranks": serializedTargetRanks,
                                        "lora_target_rank_suffixes": serializedTargetRankSuffixes,
                                        "adam_beta1": "\(config.adamBeta1)",
                                        "adam_beta2": "\(config.adamBeta2)",
                                        "adam_eps": "\(config.adamEps)",
                                        "adam_weight_decay": "\(config.adamWeightDecay)",
                                        "lr_warmup_steps": "\(config.lrWarmupSteps)",
                                        "use_cosine_scheduler": "\(config.useCosineScheduler)",
                                        "lr_min_factor": "\(config.lrMinFactor)",
                                        "dataset_root": config.datasetRoot ?? "",
                                        "seed": "\(effectiveSeed)",
                                        "rng_state": "\(lastRNGState)",
                                        "dataset_fingerprint": datasetFingerprint,
                                        "config_fingerprint": configFingerprint,
                                        "checkpoint_sidecar_file": sidecarURL.lastPathComponent,
                                        "checkpoint_archive_file": checkpointArchiveFile,
                                        "loss_csv_file": lossCSVFile,
                                        "loss_html_file": lossHTMLFile,
                                        "sample_files": checkpointSampleFiles,
                                        "run_manifest_file": runManifestURL.lastPathComponent,
                                        "mflux_iterator_file": mfluxCompatArtifacts.iteratorURL.lastPathComponent,
                                        "mflux_loss_file": mfluxCompatArtifacts.lossURL.lastPathComponent,
                                        "mflux_config_file": mfluxCompatArtifacts.configURL.lastPathComponent,
                                        "mflux_checkpoint_file": mfluxCompatArtifacts.checkpointManifestURL.lastPathComponent,
                                    ]
                                )
                                try checkpointManifest.write(nextTo: checkpointURL)
                                let checkpointRunManifest = LoRATrainingRunManifest(
                                    format: "mererun.flux2.lora",
                                    model: "flux2-klein",
                                    isEdit: hasEditExamples,
                                    dataRoot: config.datasetRoot,
                                    dataRootRelative: LoRATrainingRunManifest.relativePath(
                                        from: checkpointURL.deletingLastPathComponent(),
                                        to: config.datasetRoot
                                    ),
                                    dataFingerprint: runDataFingerprint,
                                    checkpointFiles: Self.runManifestCheckpointFiles([
                                        "lora_adapter": checkpointURL.lastPathComponent,
                                        "checkpoint_state": sidecarURL.lastPathComponent,
                                        "manifest": manifestURL.lastPathComponent,
                                        "loss_csv": lossCSVFile,
                                        "loss_html": lossHTMLFile,
                                        "sample_files": checkpointSampleFiles,
                                        "optimizer": checkpointURL.lastPathComponent,
                                        "iterator": mfluxCompatArtifacts.iteratorURL.lastPathComponent,
                                        "loss": mfluxCompatArtifacts.lossURL.lastPathComponent,
                                        "config": mfluxCompatArtifacts.configURL.lastPathComponent,
                                    ]),
                                    step: step,
                                    totalSteps: config.trainingSteps,
                                    seed: effectiveSeed,
                                    rngState: lastRNGState,
                                    datasetFingerprint: datasetFingerprint,
                                    configFingerprint: configFingerprint,
                                    phaseSchedule: checkpointPhaseSchedule,
                                    phaseCursor: LoRATrainingCheckpointState.cursor(
                                        forCompletedStep: step,
                                        phaseSchedule: checkpointPhaseSchedule
                                    ),
                                    configSnapshot: checkpointConfigSnapshot
                                )
                                try checkpointRunManifest.write(nextTo: checkpointURL)
                                try metricsLogger.writeArtifacts()
                                do {
                                    var checkpointFiles: [URL] = [sidecarURL, manifestURL, runManifestURL, metricsLogger.csvURL, metricsLogger.htmlURL]
                                    checkpointFiles.append(contentsOf: checkpointSampleArtifacts)
                                    checkpointFiles.append(contentsOf: mfluxCompatArtifacts.additionalFiles)
                                    _ = try LoRACheckpointArchive.createZipBundle(
                                        primaryFile: checkpointURL,
                                        additionalFiles: checkpointFiles
                                    )
                                } catch {
                                    print("[LoRATrainer] Warning: failed to archive checkpoint \(checkpointURL.lastPathComponent): \(error.localizedDescription)")
                                }
                            } else if shouldGenerateSample {
                                await sampleHandler?(step, checkpointURL)
                                try? FileManager.default.removeItem(at: checkpointURL)
                            }
                        }
                    }

                    globalStep += 1
                }
            }

            if let benchmarkSteps {
                // Flush queued work and report throughput.
                Stream.gpu.synchronize()
                let end = CFAbsoluteTimeGetCurrent()
                let start = benchmarkStartTime ?? end
                let elapsed = max(0, end - start)
                let avg = elapsed / Double(max(benchmarkSteps, 1))
                let message = String(format: "[LoRATrainer] Benchmark: %d steps avg %.3fs/step (elapsed %.2fs)\n", benchmarkSteps, avg, elapsed)
                FileHandle.standardError.write(Data(message.utf8))
                return
            }
        } catch is CancellationError {
            // Training was cancelled - ask caller if we should save
            if let handler = cancellationHandler {
                let shouldSave = await handler(currentStep)
                if shouldSave && currentStep > 0 {
                    // Save checkpoint at current step
                    progressHandler?(Flux2KleinLoRATrainingProgress(stage: .saving, fraction: 0))
                    try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try LoRASafetensorsWriter.save(
                        loraLayers: loraLayers,
                        to: outputURL,
                        dtype: config.saveDType,
                        includeOptimizerState: true,
                        metadata: [
                            "format": "mererun.flux2.lora",
                            "base_model": "flux2-klein",
                            "model_variant": modelManifest.variant?.rawValue ?? "",
                            "step": "\(currentStep)",
                            "total_steps": "\(config.trainingSteps)",
                            "cancelled": "true",
                            "seed": "\(effectiveSeed)",
                            "rng_state": "\(lastRNGState)",
                            "dataset_fingerprint": datasetFingerprint,
                            "config_fingerprint": configFingerprint,
                        ]
                    )
                    let checkpointState = LoRATrainingCheckpointState(
                        format: "mererun.flux2.lora",
                        baseModel: "flux2-klein",
                        checkpointFile: outputURL.lastPathComponent,
                        step: currentStep,
                        totalSteps: config.trainingSteps,
                        seed: effectiveSeed,
                        rngState: lastRNGState,
                        datasetFingerprint: datasetFingerprint,
                        configFingerprint: configFingerprint,
                        phaseSchedule: checkpointPhaseSchedule,
                        phaseCursor: LoRATrainingCheckpointState.cursor(
                            forCompletedStep: currentStep,
                            phaseSchedule: checkpointPhaseSchedule
                        ),
                        configSnapshot: checkpointConfigSnapshot,
                        lossCSVFile: lossCSVFile,
                        lossHTMLFile: lossHTMLFile,
                        manifestFile: LoRATrainingManifest.url(nextTo: outputURL).lastPathComponent
                    )
                    try checkpointState.write(nextTo: outputURL)

                    var emaOutputFile: String? = nil
                    var emaOutputURL: URL? = nil
                    if let emaState {
                        let emaURL = outputURL.deletingPathExtension()
                            .appendingPathExtension("ema")
                            .appendingPathExtension("safetensors")
                        emaOutputFile = emaURL.lastPathComponent
                        emaOutputURL = emaURL
                        try LoRASafetensorsWriter.saveEMASnapshot(
                            emaState: emaState,
                            loraLayers: loraLayers,
                            to: emaURL,
                            dtype: config.saveDType,
                            metadata: [
                                "format": "mererun.flux2.lora",
                                "base_model": "flux2-klein",
                                "model_variant": modelManifest.variant?.rawValue ?? "",
                                "ema_decay": "\(config.emaDecay)",
                                "cancelled": "true",
                            ]
                        )
                    }
                    let checkpointArchiveFile = LoRACheckpointArchive.archiveURL(for: outputURL).lastPathComponent
                    let sampleArtifacts = Self.sampleArtifacts(in: outputURL.deletingLastPathComponent())
                    let sampleFiles = sampleArtifacts.map(\.lastPathComponent).joined(separator: ",")
                    let sidecarURL = LoRATrainingCheckpointState.url(nextTo: outputURL)
                    let manifestURL = LoRATrainingManifest.url(nextTo: outputURL)
                    let runManifestURL = LoRATrainingRunManifest.url(nextTo: outputURL)
                    let mfluxCompatArtifacts = try MFluxCheckpointCompatArtifactsWriter.write(
                        checkpointURL: outputURL,
                        step: currentStep,
                        seed: effectiveSeed,
                        batchSize: config.batchSize,
                        datasetCount: examples.count,
                        loraAdapterFileName: outputURL.lastPathComponent,
                        optimizerFileName: outputURL.lastPathComponent,
                        iteratorCursor: resumeIteratorCursor,
                        lossPoints: metricsLogger.allPoints(),
                        configSnapshot: checkpointConfigSnapshot
                    )

                    let trainingManifest = LoRATrainingManifest(
                        format: "mererun.flux2.lora",
                        baseModel: "flux2-klein",
                        outputFile: outputURL.lastPathComponent,
                        emaOutputFile: emaOutputFile,
                        training: LoRATrainingManifest.Training(
                            width: config.width,
                            height: config.height,
                            trainingSteps: config.trainingSteps,
                            batchSize: config.batchSize,
                            learningRate: config.learningRate,
                            seed: effectiveSeed,
                            datasetCount: examples.count,
                            checkpointInterval: config.checkpointInterval,
                            sampleInterval: config.sampleInterval,
                            samplePrompt: config.samplePrompt,
                            emaDecay: config.emaDecay
                        ),
                        lora: LoRATrainingManifest.LoRA(
                            rank: config.loraRank,
                            alpha: config.loraAlpha,
                            saveDType: String(describing: config.saveDType),
                            includesOptimizerState: true
                        ),
                        extras: [
                            "cancelled": "true",
                            "step": "\(currentStep)",
                            "max_text_length": "\(config.maxTextLength)",
                            "scheduler_steps": "\(config.schedulerSteps)",
                            "caption_dropout": "\(config.captionDropout)",
                            "timestep_sampling": config.timestepSampling.rawValue,
                            "timestep_loss_weighting": config.timestepLossWeighting.rawValue,
                            "loss_weighting": config.lossWeighting.rawValue,
                            "timestep_low": "\(config.timestepLow)",
                            "timestep_high": config.timestepHigh.map { "\($0)" } ?? "",
                            "progressive": "\(config.progressive)",
                            "adaptive_resolution": "\(adaptiveResolution)",
                            "max_resolution": config.maxResolution.map { "\($0)" } ?? "",
                            "resolution_buckets": resolutionBucketSummary,
                            "use_compile": "\(config.useCompile)",
                            "gradient_checkpointing": "\(config.gradientCheckpointing)",
                            "low_ram": "\(config.lowRam)",
                            "edit_mode": "\(hasEditExamples)",
                            "lora_target_mode": config.loraTargetMode.rawValue,
                            "lora_target_suffixes": config.loraTargetSuffixes?.joined(separator: ",") ?? "",
                            "lora_target_ranks": serializedTargetRanks,
                            "lora_target_rank_suffixes": serializedTargetRankSuffixes,
                            "adam_beta1": "\(config.adamBeta1)",
                            "adam_beta2": "\(config.adamBeta2)",
                            "adam_eps": "\(config.adamEps)",
                            "adam_weight_decay": "\(config.adamWeightDecay)",
                            "lr_warmup_steps": "\(config.lrWarmupSteps)",
                            "use_cosine_scheduler": "\(config.useCosineScheduler)",
                            "lr_min_factor": "\(config.lrMinFactor)",
                            "dataset_root": config.datasetRoot ?? "",
                            "seed": "\(effectiveSeed)",
                            "rng_state": "\(lastRNGState)",
                            "dataset_fingerprint": datasetFingerprint,
                            "config_fingerprint": configFingerprint,
                            "checkpoint_sidecar_file": sidecarURL.lastPathComponent,
                            "checkpoint_archive_file": checkpointArchiveFile,
                            "loss_csv_file": lossCSVFile,
                            "loss_html_file": lossHTMLFile,
                            "sample_files": sampleFiles,
                            "run_manifest_file": runManifestURL.lastPathComponent,
                            "mflux_iterator_file": mfluxCompatArtifacts.iteratorURL.lastPathComponent,
                            "mflux_loss_file": mfluxCompatArtifacts.lossURL.lastPathComponent,
                            "mflux_config_file": mfluxCompatArtifacts.configURL.lastPathComponent,
                            "mflux_checkpoint_file": mfluxCompatArtifacts.checkpointManifestURL.lastPathComponent,
                        ]
                    )
                    try trainingManifest.write(nextTo: outputURL)
                    let cancelledRunManifest = LoRATrainingRunManifest(
                        format: "mererun.flux2.lora",
                        model: "flux2-klein",
                        isEdit: hasEditExamples,
                        dataRoot: config.datasetRoot,
                        dataRootRelative: LoRATrainingRunManifest.relativePath(
                            from: outputURL.deletingLastPathComponent(),
                            to: config.datasetRoot
                        ),
                        dataFingerprint: runDataFingerprint,
                        checkpointFiles: Self.runManifestCheckpointFiles([
                            "lora_adapter": outputURL.lastPathComponent,
                            "checkpoint_state": sidecarURL.lastPathComponent,
                            "manifest": manifestURL.lastPathComponent,
                            "loss_csv": lossCSVFile,
                            "loss_html": lossHTMLFile,
                            "sample_files": sampleFiles,
                            "ema_lora_adapter": emaOutputFile,
                            "optimizer": outputURL.lastPathComponent,
                            "iterator": mfluxCompatArtifacts.iteratorURL.lastPathComponent,
                            "loss": mfluxCompatArtifacts.lossURL.lastPathComponent,
                            "config": mfluxCompatArtifacts.configURL.lastPathComponent,
                        ]),
                        step: currentStep,
                        totalSteps: config.trainingSteps,
                        seed: effectiveSeed,
                        rngState: lastRNGState,
                        datasetFingerprint: datasetFingerprint,
                        configFingerprint: configFingerprint,
                        phaseSchedule: checkpointPhaseSchedule,
                        phaseCursor: LoRATrainingCheckpointState.cursor(
                            forCompletedStep: currentStep,
                            phaseSchedule: checkpointPhaseSchedule
                        ),
                        configSnapshot: checkpointConfigSnapshot
                    )
                    try cancelledRunManifest.write(nextTo: outputURL)
                    try metricsLogger.writeArtifacts()
                    do {
                        var files: [URL] = [sidecarURL, manifestURL, runManifestURL, metricsLogger.csvURL, metricsLogger.htmlURL]
                        if let emaOutputURL {
                            files.append(emaOutputURL)
                        }
                        files.append(contentsOf: sampleArtifacts)
                        files.append(contentsOf: mfluxCompatArtifacts.additionalFiles)
                        _ = try LoRACheckpointArchive.createZipBundle(
                            primaryFile: outputURL,
                            additionalFiles: files
                        )
                    } catch {
                        print("[LoRATrainer] Warning: failed to archive checkpoint \(outputURL.lastPathComponent): \(error.localizedDescription)")
                    }

                    progressHandler?(Flux2KleinLoRATrainingProgress(stage: .saving, fraction: 1))
                    return  // Don't re-throw - training completed with early save
                }
            }
            throw CancellationError()  // Re-throw if not saving
        }

        // 6) Save LoRA safetensors
        progressHandler?(Flux2KleinLoRATrainingProgress(stage: .saving, fraction: 0))
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        try LoRASafetensorsWriter.save(
            loraLayers: loraLayers,
            to: outputURL,
            dtype: config.saveDType,
            includeOptimizerState: true,
            metadata: [
                "format": "mererun.flux2.lora",
                "base_model": "flux2-klein",
                "model_variant": modelManifest.variant?.rawValue ?? "",
                "step": "\(config.trainingSteps)",
                "total_steps": "\(config.trainingSteps)",
                "seed": "\(effectiveSeed)",
                "rng_state": "\(lastRNGState)",
                "dataset_fingerprint": datasetFingerprint,
                "config_fingerprint": configFingerprint,
            ]
        )
        let checkpointState = LoRATrainingCheckpointState(
            format: "mererun.flux2.lora",
            baseModel: "flux2-klein",
            checkpointFile: outputURL.lastPathComponent,
            step: config.trainingSteps,
            totalSteps: config.trainingSteps,
            seed: effectiveSeed,
            rngState: lastRNGState,
            datasetFingerprint: datasetFingerprint,
            configFingerprint: configFingerprint,
            phaseSchedule: checkpointPhaseSchedule,
            phaseCursor: LoRATrainingCheckpointState.cursor(
                forCompletedStep: config.trainingSteps,
                phaseSchedule: checkpointPhaseSchedule
            ),
            configSnapshot: checkpointConfigSnapshot,
            lossCSVFile: lossCSVFile,
            lossHTMLFile: lossHTMLFile,
            manifestFile: LoRATrainingManifest.url(nextTo: outputURL).lastPathComponent
        )
        try checkpointState.write(nextTo: outputURL)

        // Save EMA weights if enabled
        var emaOutputFile: String? = nil
        var emaOutputURL: URL? = nil
        if let emaState {
            let emaURL = outputURL.deletingPathExtension()
                .appendingPathExtension("ema")
                .appendingPathExtension("safetensors")
            emaOutputFile = emaURL.lastPathComponent
            emaOutputURL = emaURL
            try LoRASafetensorsWriter.saveEMASnapshot(
                emaState: emaState,
                loraLayers: loraLayers,
                to: emaURL,
                dtype: config.saveDType,
                metadata: [
                    "format": "mererun.flux2.lora",
                    "base_model": "flux2-klein",
                    "model_variant": modelManifest.variant?.rawValue ?? "",
                    "ema_decay": "\(config.emaDecay)",
                ]
            )
        }
        let checkpointArchiveFile = LoRACheckpointArchive.archiveURL(for: outputURL).lastPathComponent
        let sampleArtifacts = Self.sampleArtifacts(in: outputURL.deletingLastPathComponent())
        let sampleFiles = sampleArtifacts.map(\.lastPathComponent).joined(separator: ",")
        let sidecarURL = LoRATrainingCheckpointState.url(nextTo: outputURL)
        let manifestURL = LoRATrainingManifest.url(nextTo: outputURL)
        let runManifestURL = LoRATrainingRunManifest.url(nextTo: outputURL)
        let mfluxCompatArtifacts = try MFluxCheckpointCompatArtifactsWriter.write(
            checkpointURL: outputURL,
            step: config.trainingSteps,
            seed: effectiveSeed,
            batchSize: config.batchSize,
            datasetCount: examples.count,
            loraAdapterFileName: outputURL.lastPathComponent,
            optimizerFileName: outputURL.lastPathComponent,
            iteratorCursor: resumeIteratorCursor,
            lossPoints: metricsLogger.allPoints(),
            configSnapshot: checkpointConfigSnapshot
        )

        let trainingManifest = LoRATrainingManifest(
            format: "mererun.flux2.lora",
            baseModel: "flux2-klein",
            outputFile: outputURL.lastPathComponent,
            emaOutputFile: emaOutputFile,
            training: LoRATrainingManifest.Training(
                width: config.width,
                height: config.height,
                trainingSteps: config.trainingSteps,
                batchSize: config.batchSize,
                learningRate: config.learningRate,
                seed: effectiveSeed,
                datasetCount: examples.count,
                checkpointInterval: config.checkpointInterval,
                sampleInterval: config.sampleInterval,
                samplePrompt: config.samplePrompt,
                emaDecay: config.emaDecay
            ),
            lora: LoRATrainingManifest.LoRA(
                rank: config.loraRank,
                alpha: config.loraAlpha,
                saveDType: String(describing: config.saveDType),
                includesOptimizerState: true
            ),
            extras: [
                "max_text_length": "\(config.maxTextLength)",
                "scheduler_steps": "\(config.schedulerSteps)",
                "caption_dropout": "\(config.captionDropout)",
                "timestep_sampling": config.timestepSampling.rawValue,
                "timestep_loss_weighting": config.timestepLossWeighting.rawValue,
                "loss_weighting": config.lossWeighting.rawValue,
                "timestep_low": "\(config.timestepLow)",
                "timestep_high": config.timestepHigh.map { "\($0)" } ?? "",
                "progressive": "\(config.progressive)",
                "adaptive_resolution": "\(adaptiveResolution)",
                "max_resolution": config.maxResolution.map { "\($0)" } ?? "",
                "resolution_buckets": resolutionBucketSummary,
                "use_compile": "\(config.useCompile)",
                "gradient_checkpointing": "\(config.gradientCheckpointing)",
                "low_ram": "\(config.lowRam)",
                "edit_mode": "\(hasEditExamples)",
                "lora_target_mode": config.loraTargetMode.rawValue,
                "lora_target_suffixes": config.loraTargetSuffixes?.joined(separator: ",") ?? "",
                "lora_target_ranks": serializedTargetRanks,
                "lora_target_rank_suffixes": serializedTargetRankSuffixes,
                "adam_beta1": "\(config.adamBeta1)",
                "adam_beta2": "\(config.adamBeta2)",
                "adam_eps": "\(config.adamEps)",
                "adam_weight_decay": "\(config.adamWeightDecay)",
                "lr_warmup_steps": "\(config.lrWarmupSteps)",
                "use_cosine_scheduler": "\(config.useCosineScheduler)",
                "lr_min_factor": "\(config.lrMinFactor)",
                "dataset_root": config.datasetRoot ?? "",
                "seed": "\(effectiveSeed)",
                "rng_state": "\(lastRNGState)",
                "dataset_fingerprint": datasetFingerprint,
                "config_fingerprint": configFingerprint,
                "checkpoint_sidecar_file": sidecarURL.lastPathComponent,
                "checkpoint_archive_file": checkpointArchiveFile,
                "loss_csv_file": lossCSVFile,
                "loss_html_file": lossHTMLFile,
                "sample_files": sampleFiles,
                "run_manifest_file": runManifestURL.lastPathComponent,
                "mflux_iterator_file": mfluxCompatArtifacts.iteratorURL.lastPathComponent,
                "mflux_loss_file": mfluxCompatArtifacts.lossURL.lastPathComponent,
                "mflux_config_file": mfluxCompatArtifacts.configURL.lastPathComponent,
                "mflux_checkpoint_file": mfluxCompatArtifacts.checkpointManifestURL.lastPathComponent,
            ]
        )
        try trainingManifest.write(nextTo: outputURL)
        let finalRunManifest = LoRATrainingRunManifest(
            format: "mererun.flux2.lora",
            model: "flux2-klein",
            isEdit: hasEditExamples,
            dataRoot: config.datasetRoot,
            dataRootRelative: LoRATrainingRunManifest.relativePath(
                from: outputURL.deletingLastPathComponent(),
                to: config.datasetRoot
            ),
            dataFingerprint: runDataFingerprint,
            checkpointFiles: Self.runManifestCheckpointFiles([
                "lora_adapter": outputURL.lastPathComponent,
                "checkpoint_state": sidecarURL.lastPathComponent,
                "manifest": manifestURL.lastPathComponent,
                "loss_csv": lossCSVFile,
                "loss_html": lossHTMLFile,
                "sample_files": sampleFiles,
                "ema_lora_adapter": emaOutputFile,
                "optimizer": outputURL.lastPathComponent,
                "iterator": mfluxCompatArtifacts.iteratorURL.lastPathComponent,
                "loss": mfluxCompatArtifacts.lossURL.lastPathComponent,
                "config": mfluxCompatArtifacts.configURL.lastPathComponent,
            ]),
            step: config.trainingSteps,
            totalSteps: config.trainingSteps,
            seed: effectiveSeed,
            rngState: lastRNGState,
            datasetFingerprint: datasetFingerprint,
            configFingerprint: configFingerprint,
            phaseSchedule: checkpointPhaseSchedule,
            phaseCursor: LoRATrainingCheckpointState.cursor(
                forCompletedStep: config.trainingSteps,
                phaseSchedule: checkpointPhaseSchedule
            ),
            configSnapshot: checkpointConfigSnapshot
        )
        try finalRunManifest.write(nextTo: outputURL)
        try metricsLogger.writeArtifacts()
        do {
            var files: [URL] = [sidecarURL, manifestURL, runManifestURL, metricsLogger.csvURL, metricsLogger.htmlURL]
            if let emaOutputURL {
                files.append(emaOutputURL)
            }
            files.append(contentsOf: sampleArtifacts)
            files.append(contentsOf: mfluxCompatArtifacts.additionalFiles)
            _ = try LoRACheckpointArchive.createZipBundle(
                primaryFile: outputURL,
                additionalFiles: files
            )
        } catch {
            print("[LoRATrainer] Warning: failed to archive checkpoint \(outputURL.lastPathComponent): \(error.localizedDescription)")
        }

        progressHandler?(Flux2KleinLoRATrainingProgress(stage: .saving, fraction: 1))
    }

    private static func sampleArtifacts(in outputDirectory: URL, step: Int? = nil) -> [URL] {
        let sampleDirectory = outputDirectory.appendingPathComponent("samples", isDirectory: true)
        let fileManager = FileManager.default
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: sampleDirectory.path, isDirectory: &isDir), isDir.boolValue else {
            return []
        }

        let entries: [URL]
        do {
            entries = try fileManager.contentsOfDirectoryResolvingSymlinks(
                at: sampleDirectory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
        } catch {
            return []
        }

        let requiredLegacyPrefix = step.map { "step-\($0)" }
        let requiredMfluxPrefix = step.map { String(format: "%07d_", $0) }
        return entries
            .filter { entry in
                guard (try? entry.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else {
                    return false
                }
                guard let requiredLegacyPrefix, let requiredMfluxPrefix else {
                    return true
                }
                let filename = entry.lastPathComponent
                return filename.hasPrefix(requiredLegacyPrefix) || filename.hasPrefix(requiredMfluxPrefix)
            }
            .sorted { lhs, rhs in lhs.lastPathComponent < rhs.lastPathComponent }
    }

    private static func makeRunDataFingerprint(
        examples: [Flux2KleinLoRATrainingExample],
        dataRootPath: String?,
        isEdit: Bool
    ) -> LoRATrainingRunManifest.DataFingerprint {
        let imagePaths = examples.map { example in
            LoRATrainingRunManifest.dataPath(
                for: example.imageURL,
                dataRootPath: dataRootPath
            )
        }
        let inputImagePaths = examples.compactMap { example -> String? in
            guard let inputImageURL = example.inputImageURL else { return nil }
            return LoRATrainingRunManifest.dataPath(
                for: inputImageURL,
                dataRootPath: dataRootPath
            )
        }
        return LoRATrainingRunManifest.DataFingerprint(
            count: examples.count,
            images: imagePaths,
            inputImages: inputImagePaths,
            isEdit: isEdit
        )
    }

    private static func validateResumeRunManifest(
        _ manifest: LoRATrainingRunManifest,
        expectedModel: String,
        expectedIsEdit: Bool,
        expectedDataFingerprint: LoRATrainingRunManifest.DataFingerprint,
        expectedDatasetFingerprint: String,
        expectedConfigFingerprint: String,
        expectedPhaseSchedule: [LoRATrainingCheckpointState.Phase],
        checkpointName: String
    ) throws {
        if manifest.model.caseInsensitiveCompare(expectedModel) != .orderedSame {
            throw NSError(
                domain: "Flux2KleinLoRATrainer",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Resume run manifest model mismatch for \(checkpointName). Expected \(expectedModel), got \(manifest.model).",
                ]
            )
        }
        if manifest.isEdit != expectedIsEdit {
            throw NSError(
                domain: "Flux2KleinLoRATrainer",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Resume run manifest edit-mode mismatch for \(checkpointName).",
                ]
            )
        }
        if let actualDataFingerprint = manifest.dataFingerprint,
           actualDataFingerprint != expectedDataFingerprint {
            throw NSError(
                domain: "Flux2KleinLoRATrainer",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Resume run manifest data fingerprint mismatch for \(checkpointName).",
                ]
            )
        }
        if let actualDatasetFingerprint = manifest.datasetFingerprint,
           actualDatasetFingerprint != expectedDatasetFingerprint {
            throw NSError(
                domain: "Flux2KleinLoRATrainer",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Resume run manifest dataset fingerprint mismatch for \(checkpointName).",
                ]
            )
        }
        if let actualConfigFingerprint = manifest.configFingerprint,
           actualConfigFingerprint != expectedConfigFingerprint {
            throw NSError(
                domain: "Flux2KleinLoRATrainer",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Resume run manifest config fingerprint mismatch for \(checkpointName).",
                ]
            )
        }
        if let schedule = manifest.phaseSchedule,
           !LoRATrainingCheckpointState.scheduleMatches(expected: expectedPhaseSchedule, actual: schedule) {
            throw NSError(
                domain: "Flux2KleinLoRATrainer",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Resume run manifest phase schedule mismatch for \(checkpointName).",
                ]
            )
        }
        if let cursor = manifest.phaseCursor {
            guard let expectedCursor = LoRATrainingCheckpointState.cursor(
                forCompletedStep: manifest.step,
                phaseSchedule: expectedPhaseSchedule
            ), cursor == expectedCursor else {
                throw NSError(
                    domain: "Flux2KleinLoRATrainer",
                    code: 1,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Resume run manifest iterator cursor mismatch for \(checkpointName).",
                    ]
                )
            }
        }
    }

    private static func loadRunManifestLenient(nextTo checkpointURL: URL) -> LoRATrainingRunManifest? {
        do {
            return try LoRATrainingRunManifest.load(nextTo: checkpointURL)
        } catch {
            FileHandle.standardError.write(
                Data("[LoRATrainer] Warning: failed to decode run.json beside \(checkpointURL.lastPathComponent): \(error.localizedDescription)\n".utf8)
            )
            return nil
        }
    }

    private static func syntheticSidecar(from manifest: LoRATrainingRunManifest?) -> LoRATrainingCheckpointState? {
        guard let manifest else { return nil }

        let checkpointFiles = manifest.checkpointFiles
        let checkpointFile = (checkpointFiles["lora_adapter"] as NSString?)?.lastPathComponent ?? ""
        let resolvedCheckpointFile = checkpointFile.isEmpty ? "checkpoint.safetensors" : checkpointFile
        let manifestFile = (checkpointFiles["manifest"] as NSString?)?.lastPathComponent
        let lossCSVFile = (checkpointFiles["loss_csv"] as NSString?)?.lastPathComponent
        let lossHTMLFile = (checkpointFiles["loss_html"] as NSString?)?.lastPathComponent

        return LoRATrainingCheckpointState(
            format: manifest.format,
            baseModel: manifest.model,
            checkpointFile: resolvedCheckpointFile,
            step: manifest.step,
            totalSteps: manifest.totalSteps,
            seed: manifest.seed,
            rngState: manifest.rngState,
            datasetFingerprint: manifest.datasetFingerprint,
            configFingerprint: manifest.configFingerprint,
            phaseSchedule: manifest.phaseSchedule,
            phaseCursor: manifest.phaseCursor,
            configSnapshot: manifest.configSnapshot,
            lossCSVFile: lossCSVFile,
            lossHTMLFile: lossHTMLFile,
            manifestFile: manifestFile
        )
    }

    private static func runManifestCheckpointFiles(_ rawFiles: [String: String?]) -> [String: String] {
        var normalized: [String: String] = [:]
        normalized.reserveCapacity(rawFiles.count)

        for (key, rawValue) in rawFiles {
            guard let rawValue else { continue }
            let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            if key == "sample_files" {
                let files = trimmed
                    .split(separator: ",")
                    .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                    .map { ($0 as NSString).lastPathComponent }
                    .filter { !$0.isEmpty }
                guard !files.isEmpty else { continue }
                normalized[key] = files.joined(separator: ",")
                continue
            }

            let fileName = (trimmed as NSString).lastPathComponent
            guard !fileName.isEmpty else { continue }
            normalized[key] = fileName
        }

        return normalized
    }

    private static func serializedLoRATargetRanks(_ targetRanks: [String: Int]?) -> String {
        guard let targetRanks, !targetRanks.isEmpty else { return "" }
        return targetRanks
            .sorted { lhs, rhs in lhs.key < rhs.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: ";")
    }
}
