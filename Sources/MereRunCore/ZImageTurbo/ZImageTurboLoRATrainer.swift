import Foundation
import MLX
import MLXNN
import MLXRandom

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

/// Timestep sampling strategy for training.
public enum TimestepSamplingStrategy: String, Sendable {
    /// Uniform random sampling across all timesteps
    case uniform
    /// Bell-shaped weighting that favors middle timesteps where most learning happens
    case bellCurve
    /// Favors earlier timesteps (content/structure learning)
    case contentFocused
    /// Favors later timesteps (style/detail learning)
    case styleFocused
    /// Sigmoid-based sampling that clusters around middle timesteps (ai-toolkit default)
    case sigmoid
}

/// Loss weighting strategy for training.
public enum LossWeightingStrategy: String, Sendable {
    /// No weighting - all timesteps contribute equally to loss
    case none
    /// SNR (Signal-to-Noise Ratio) weighting - weight by SNR to prevent high-noise steps from dominating
    case snr
    /// Min-SNR weighting with gamma=5 (from "Efficient Diffusion Training via Min-SNR Weighting")
    case minSNR
}

/// Timestep loss weighting strategy for FlowMatch training.
public enum ZImageTimestepLossWeightingStrategy: String, Sendable {
    /// No weighting (all timesteps contribute equally).
    case none
    /// Bell-shaped mean-normalized weighting (ai-toolkit default).
    case weighted
    /// Half-bell variant (ai-toolkit v2 / linear_timesteps2).
    case weighted2
}

public enum ZImageTurboLoRATrainerError: Error, LocalizedError {
    case datasetEmpty
    case invalidDimensions(width: Int, height: Int)
    case invalidTrainingSteps(Int)
    case invalidBatchSize(Int)
    case invalidSchedulerSteps(Int)
    case outputMustBeSafetensors(URL)
    case imageNotFound(URL)
    case captionEmpty(URL)
    case imageDecodeFailed(URL)
    case modelPathNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .datasetEmpty:
            return "Training dataset is empty."
        case .invalidDimensions(let width, let height):
            return "Invalid image dimensions (\(width)x\(height)); width/height must be >0 and divisible by 32."
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
        case .modelPathNotFound(let path):
            return "Model path not found: \(path)"
        }
    }
}

public struct ZImageTurboLoRATrainingExample: Hashable, Sendable {
    public let imageURL: URL
    public let caption: String

    public init(imageURL: URL, caption: String) {
        self.imageURL = imageURL
        self.caption = caption
    }
}

public struct ZImageTurboLoRATrainingConfig: Sendable {
    public var width: Int
    public var height: Int
    public var maxResolution: Int?

    /// Max tokens for text encoder (lower = faster / less memory).
    public var maxTextLength: Int

    /// Number of FlowMatch training timesteps (ai-toolkit default: 1000).
    public var schedulerSteps: Int

    /// Number of gradient updates to run.
    public var trainingSteps: Int

    public var batchSize: Int
    public var learningRate: Float
    public var seed: UInt64

    public var loraRank: Int
    public var loraAlpha: Float?
    public var loraTargetPrefixes: [String]?
    public var loraTargetSuffixes: [String]?
    public var loraTargetRanks: [String: Int]?
    public var datasetRoot: String?

    /// Probability of dropping caption during training (0.0-1.0).
    public var captionDropout: Float

    public var saveDType: DType
    public var logEvery: Int

    /// Interval at which to save checkpoint artifacts.
    public var checkpointInterval: Int?

    /// Interval at which to generate sample previews.
    public var sampleInterval: Int?

    /// Prompt to use for sample generation.
    public var samplePrompt: String?

    /// Use lite mode (only Q/V in main layers) to reduce memory usage.
    public var liteMode: Bool

    /// Enable gradient checkpointing to reduce memory during backward pass.
    /// Trades compute for memory by recomputing activations during backward.
    public var gradientCheckpointing: Bool

    /// Use disk-backed cache for encoded data to reduce peak memory usage.
    public var lowRam: Bool

    /// Timestep sampling strategy.
    public var timestepSampling: TimestepSamplingStrategy

    /// Optional timestep loss weighting (ai-toolkit `timestep_type: weighted`).
    public var timestepLossWeighting: ZImageTimestepLossWeightingStrategy

    /// Loss weighting strategy (SNR-based weighting).
    public var lossWeighting: LossWeightingStrategy

    /// EMA decay rate for model weights (0 = disabled, 0.9999 = typical).
    /// When enabled, saves both regular and EMA weights.
    public var emaDecay: Float

    /// Include wavelet loss for improved high-frequency detail (experimental).
    public var waveletLoss: Bool

    /// Weight for wavelet loss component (only used if waveletLoss is true).
    public var waveletLossWeight: Float

    /// Enable differential guidance during training (ai-toolkit default for base models).
    /// Modifies target: target = pred + scale * (target - pred)
    public var doDifferentialGuidance: Bool

    /// Scale for differential guidance (ai-toolkit default: 4.0).
    public var differentialGuidanceScale: Float

    /// Whether to apply dynamic sigma shifting based on resolution.
    public var useDynamicSigmaShift: Bool

    /// Optional fixed sigma shift (e.g. 3.0 for base models).
    public var sigmaShift: Float?

    /// Optional assistant LoRA path to merge into the base transformer for training.
    public var assistantLoRAPath: String?

    /// Use the default training adapter from mere.run's model storage if no custom assistant LoRA is provided.
    /// Default: true. Set to false to disable the training adapter entirely.
    public var useTrainingAdapter: Bool

    /// Base model identifier for metadata.
    public var baseModelId: String

    /// Benchmark mode: if set, measure these steps (after warmup) and then exit without saving.
    public var benchmarkSteps: Int?

    /// Warmup steps before benchmark timing starts.
    public var benchmarkWarmupSteps: Int

    /// Print per-step timing breakdown.
    public var timingEnabled: Bool

    /// Use synthetic latents/embeds instead of dataset encoding.
    public var syntheticSampleCount: Int?

    /// Minimum timestep index to sample (inclusive). Default 0.
    /// For turbo models (9-step), mflux recommends timestepLow=4, timestepHigh=9 to focus on later timesteps.
    public var timestepLow: Int

    /// Maximum timestep index to sample (exclusive). Default nil = schedulerSteps.
    public var timestepHigh: Int?

    public init(
        width: Int = 1024,
        height: Int = 1024,
        maxResolution: Int? = nil,
        maxTextLength: Int = 256,
        schedulerSteps: Int = 1000,
        trainingSteps: Int = 1000,
        batchSize: Int = 1,
        learningRate: Float = 1e-4,
        seed: UInt64 = 0,
        loraRank: Int = 16,
        loraAlpha: Float? = nil,
        loraTargetPrefixes: [String]? = nil,
        loraTargetSuffixes: [String]? = nil,
        loraTargetRanks: [String: Int]? = nil,
        datasetRoot: String? = nil,
        captionDropout: Float = 0.05,
        saveDType: DType = .float16,
        logEvery: Int = 10,
        checkpointInterval: Int? = nil,
        sampleInterval: Int? = nil,
        samplePrompt: String? = nil,
        liteMode: Bool = false,
        gradientCheckpointing: Bool = false,
        lowRam: Bool = false,
        timestepSampling: TimestepSamplingStrategy = .sigmoid,
        timestepLossWeighting: ZImageTimestepLossWeightingStrategy = .none,
        lossWeighting: LossWeightingStrategy = .none,
        emaDecay: Float = 0,
        waveletLoss: Bool = false,
        waveletLossWeight: Float = 0.1,
        doDifferentialGuidance: Bool = true,
        differentialGuidanceScale: Float = 4.0,
        useDynamicSigmaShift: Bool = true,
        sigmaShift: Float? = nil,
        assistantLoRAPath: String? = nil,
        useTrainingAdapter: Bool = true,
        baseModelId: String = "z-image-turbo",
        benchmarkSteps: Int? = nil,
        benchmarkWarmupSteps: Int = 5,
        timingEnabled: Bool = false,
        syntheticSampleCount: Int? = nil,
        timestepLow: Int = 0,
        timestepHigh: Int? = nil
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
        self.loraTargetPrefixes = loraTargetPrefixes
        self.loraTargetSuffixes = loraTargetSuffixes
        self.loraTargetRanks = loraTargetRanks
        self.datasetRoot = datasetRoot
        self.captionDropout = captionDropout
        self.saveDType = saveDType
        self.logEvery = logEvery
        self.checkpointInterval = checkpointInterval
        self.sampleInterval = sampleInterval
        self.samplePrompt = samplePrompt
        self.liteMode = liteMode
        self.gradientCheckpointing = gradientCheckpointing
        self.lowRam = lowRam
        self.timestepSampling = timestepSampling
        self.timestepLossWeighting = timestepLossWeighting
        self.lossWeighting = lossWeighting
        self.emaDecay = emaDecay
        self.waveletLoss = waveletLoss
        self.waveletLossWeight = waveletLossWeight
        self.doDifferentialGuidance = doDifferentialGuidance
        self.differentialGuidanceScale = differentialGuidanceScale
        self.useDynamicSigmaShift = useDynamicSigmaShift
        self.sigmaShift = sigmaShift
        self.assistantLoRAPath = assistantLoRAPath
        self.useTrainingAdapter = useTrainingAdapter
        self.baseModelId = baseModelId
        self.benchmarkSteps = benchmarkSteps
        self.benchmarkWarmupSteps = benchmarkWarmupSteps
        self.timingEnabled = timingEnabled
        self.syntheticSampleCount = syntheticSampleCount
        self.timestepLow = timestepLow
        self.timestepHigh = timestepHigh
    }
}

public struct ZImageTurboLoRATrainingProgress: Sendable {
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

public enum ZImageTurboLoRATrainer {
    /// Default training adapter that provides a better starting point for rapid Z-Image-Turbo training.
    /// Resolved through Hugging Face Hub.
    public static let defaultTrainingAdapterRef =
        "ostris/zimage_turbo_training_adapter:zimage_turbo_training_adapter_v2.safetensors"

    /// Trains a LoRA on the given examples.
    public static func train(
        modelPath: String,
        examples: [ZImageTurboLoRATrainingExample],
        outputURL: URL,
        config: ZImageTurboLoRATrainingConfig = ZImageTurboLoRATrainingConfig(),
        resumeFromLoRA: URL? = nil,
        progressHandler: (@Sendable (ZImageTurboLoRATrainingProgress) -> Void)? = nil,
        sampleHandler: (@Sendable (Int, URL) async -> Void)? = nil,
        cancellationHandler: (@Sendable (Int) async -> Bool)? = nil
    ) async throws {
        guard !examples.isEmpty else {
            throw ZImageTurboLoRATrainerError.datasetEmpty
        }
        guard config.width > 0, config.height > 0, config.width % 32 == 0, config.height % 32 == 0 else {
            throw ZImageTurboLoRATrainerError.invalidDimensions(width: config.width, height: config.height)
        }
        guard config.trainingSteps >= 1 else {
            throw ZImageTurboLoRATrainerError.invalidTrainingSteps(config.trainingSteps)
        }
        guard config.batchSize >= 1 else {
            throw ZImageTurboLoRATrainerError.invalidBatchSize(config.batchSize)
        }
        guard config.schedulerSteps >= 1 else {
            throw ZImageTurboLoRATrainerError.invalidSchedulerSteps(config.schedulerSteps)
        }
        if let maxResolution = config.maxResolution, maxResolution <= 0 {
            throw NSError(
                domain: "ZImageTurboLoRATrainer",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "--max-resolution must be >= 1 (got \(maxResolution))."]
            )
        }
        guard outputURL.pathExtension.lowercased() == "safetensors" else {
            throw ZImageTurboLoRATrainerError.outputMustBeSafetensors(outputURL)
        }

        progressHandler?(ZImageTurboLoRATrainingProgress(stage: .loadingModels, fraction: 0))

        let datasetFingerprint = LoRATrainingFingerprint.sha256Hex(
            examples
                .map { example in
                    [
                        example.imageURL.standardizedFileURL.path,
                        example.caption
                    ].joined(separator: "|")
                }
                .joined(separator: "\n")
        )
        let runDataFingerprint = Self.makeRunDataFingerprint(
            examples: examples,
            dataRootPath: config.datasetRoot
        )
        let resolvedSeed = config.seed == 0 ? UInt64(Date().timeIntervalSince1970) : config.seed

        let modelURL = URL(fileURLWithPath: modelPath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw ZImageTurboLoRATrainerError.modelPathNotFound(modelPath)
        }

        let modelManifest = try MereRunModelManifest.loadRequired(from: modelURL)
        let inferredBaseModel = modelManifest.variant == .base
        let effectiveUseTrainingAdapter = config.useTrainingAdapter && !inferredBaseModel
        let serializedTargetRanks = Self.serializedLoRATargetRanks(config.loraTargetRanks)
        let configFingerprintInput: String = [
            "model:\(modelURL.path)",
            "model_id:\(config.baseModelId)",
            "manifest_variant:\(modelManifest.variant?.rawValue ?? "")",
            "size:\(config.width)x\(config.height)",
            "max_resolution:\(config.maxResolution.map { "\($0)" } ?? "")",
            "scheduler_steps:\(config.schedulerSteps)",
            "training_steps:\(config.trainingSteps)",
            "batch_size:\(config.batchSize)",
            "learning_rate:\(config.learningRate)",
            "rank:\(config.loraRank)",
            "alpha:\(config.loraAlpha.map { "\($0)" } ?? "")",
            "max_text_length:\(config.maxTextLength)",
            "caption_dropout:\(config.captionDropout)",
            "checkpoint_interval:\(config.checkpointInterval.map { "\($0)" } ?? "")",
            "sample_interval:\(config.sampleInterval.map { "\($0)" } ?? "")",
            "lite_mode:\(config.liteMode)",
            "gradient_checkpointing:\(config.gradientCheckpointing)",
            "low_ram:\(config.lowRam)",
            "timestep_sampling:\(config.timestepSampling.rawValue)",
            "timestep_loss_weighting:\(config.timestepLossWeighting.rawValue)",
            "loss_weighting:\(config.lossWeighting.rawValue)",
            "wavelet_loss:\(config.waveletLoss)",
            "wavelet_loss_weight:\(config.waveletLossWeight)",
            "differential_guidance:\(config.doDifferentialGuidance)",
            "differential_guidance_scale:\(config.differentialGuidanceScale)",
            "use_dynamic_sigma_shift:\(config.useDynamicSigmaShift)",
            "sigma_shift:\(config.sigmaShift.map { "\($0)" } ?? "")",
            "timestep_low:\(config.timestepLow)",
            "timestep_high:\(config.timestepHigh.map { "\($0)" } ?? "")",
            "synthetic_sample_count:\(config.syntheticSampleCount.map { "\($0)" } ?? "")",
            "lora_target_prefixes:\((config.loraTargetPrefixes ?? ZImageLoRAInjector.aiToolkitCompatiblePrefixes).joined(separator: ","))",
            "lora_target_suffixes:\((config.loraTargetSuffixes ?? (config.liteMode ? ZImageLoRAInjector.liteTargetSuffixes : ZImageLoRAInjector.aiToolkitCompatibleSuffixes)).joined(separator: ","))",
            "lora_target_ranks:\(serializedTargetRanks)",
            "assistant_lora_path:\(config.assistantLoRAPath ?? "")",
            "use_training_adapter:\(effectiveUseTrainingAdapter)",
        ].joined(separator: "\n")
        let configFingerprint = LoRATrainingFingerprint.sha256Hex(configFingerprintInput)

        let componentResolver = ModelComponentResolver(modelRootURL: modelURL, manifest: modelManifest)
        let tokenizerComponent = try componentResolver.resolveDirectory(for: .tokenizer, fallbackLocalPath: "tokenizer")
        let textEncoderComponent = try componentResolver.resolveDirectory(for: .textEncoder, fallbackLocalPath: "text_encoder")
        let transformerComponent = try componentResolver.resolveDirectory(for: .transformer, fallbackLocalPath: "transformer")
        let vaeComponent = try componentResolver.resolveDirectory(for: .vae, fallbackLocalPath: "vae")
        let schedulerComponent = try componentResolver.resolveDirectory(for: .scheduler, fallbackLocalPath: "scheduler")

        let resources = ZImageTurboResources(
            modelRootURL: modelURL,
            tokenizerDirURL: tokenizerComponent.directoryURL,
            textEncoderDirURL: textEncoderComponent.directoryURL,
            transformerDirURL: transformerComponent.directoryURL,
            vaeDirURL: vaeComponent.directoryURL,
            schedulerDirURL: schedulerComponent.directoryURL
        )
        let configs = try ZImageTurboModelConfigs.load(from: resources)
        let textEncoderQuantization = try ModelWeightsLoader.QuantizationParams.fromManifest(textEncoderComponent.sourceManifest)
        let transformerQuantization = try ModelWeightsLoader.QuantizationParams.fromManifest(transformerComponent.sourceManifest)

        // Load tokenizer
        let tokenizerDir = resources.tokenizerDirURL
        let maxLen = min(config.maxTextLength, configs.textEncoder.maxPositionEmbeddings)
        let tokenizer = try QwenTokenizer.load(from: tokenizerDir, maxLengthOverride: maxLen)

        // Load text encoder
        let textEncoder = QwenTextEncoder(configuration: .init(
            vocabSize: configs.textEncoder.vocabSize,
            hiddenSize: configs.textEncoder.hiddenSize,
            numHiddenLayers: configs.textEncoder.numHiddenLayers,
            numAttentionHeads: configs.textEncoder.numAttentionHeads,
            numKeyValueHeads: configs.textEncoder.numKeyValueHeads,
            intermediateSize: configs.textEncoder.intermediateSize,
            ropeTheta: configs.textEncoder.ropeTheta,
            maxPositionEmbeddings: configs.textEncoder.maxPositionEmbeddings,
            rmsNormEps: configs.textEncoder.rmsNormEps,
            headDim: configs.textEncoder.headDim
        ))
        try loadTextEncoderWeights(from: resources, into: textEncoder, quantization: textEncoderQuantization)

        // Load transformer
        let transformer = ZImageTransformer2DModel(configuration: configs.transformer)
        try loadTransformerWeights(from: resources, into: transformer, quantization: transformerQuantization)

        // Apply assistant LoRA: use explicit path, or default training adapter if enabled
        let effectiveAssistantPath: String? = {
            if let explicit = config.assistantLoRAPath, !explicit.isEmpty {
                return explicit
            }
            if effectiveUseTrainingAdapter {
                return Self.defaultTrainingAdapterRef
            }
            return nil
        }()
        if let assistantPath = effectiveAssistantPath {
            let resolvedURL = try await resolveAssistantLoRAPath(assistantPath)
            try applyAssistantLoRA(from: resolvedURL.path, to: transformer)
        }
        transformer.gradientCheckpointing = config.gradientCheckpointing

        // Load VAE
        let vae = AutoencoderKL(configuration: .init(
            inChannels: configs.vae.inChannels,
            outChannels: configs.vae.outChannels,
            latentChannels: configs.vae.latentChannels,
            scalingFactor: configs.vae.scalingFactor,
            shiftFactor: configs.vae.shiftFactor,
            blockOutChannels: configs.vae.blockOutChannels,
            layersPerBlock: configs.vae.layersPerBlock,
            normNumGroups: configs.vae.normNumGroups,
            sampleSize: configs.vae.sampleSize ?? 1024,
            midBlockAddAttention: configs.vae.midBlockAddAttention
        ))
        try loadVAEWeights(from: resources, into: vae)

        // Ensure all model weights are evaluated before use
        MLX.eval(textEncoder, transformer, vae)

        if config.timingEnabled {
            let memMB = Double(Memory.activeMemory) / 1024.0 / 1024.0
            print(String(format: "[ZImageLoRATrainer] GPU active memory after load: %.1f MB", memMB))
        }

        progressHandler?(ZImageTurboLoRATrainingProgress(stage: .loadingModels, fraction: 1))

        // Encode dataset
        struct PreparedPrompt {
            let imageURL: URL
            let promptEmbeds: MLXArray
        }

        struct PreparedExample {
            let cleanLatents: MLXArray
            let promptEmbeds: MLXArray
        }

        struct PhasePlan {
            let width: Int
            let height: Int
            let steps: Int
            let sampleIndices: [Int]
        }

        struct PhaseData {
            let phase: PhasePlan
            let latentHeight: Int
            let latentWidth: Int
            let trainingSigmas: [Float]
            let prepared: [PreparedExample]?
            let preparedCount: Int
            let cache: TrainingDataCache?
        }

        let syntheticCount = config.syntheticSampleCount ?? 0
        let adaptiveResolution = config.maxResolution != nil && syntheticCount == 0
        let useLowRam = config.lowRam && syntheticCount == 0

        let phasePlans: [PhasePlan] = try {
            if syntheticCount > 0 {
                return [
                    PhasePlan(
                        width: config.width,
                        height: config.height,
                        steps: config.trainingSteps,
                        sampleIndices: Array(0..<syntheticCount)
                    )
                ]
            }

            if adaptiveResolution {
                var buckets: [LoRAResolvedResolution: [Int]] = [:]
                for (index, example) in examples.enumerated() {
                    let dims = try LoRATrainingResolution.resolveFromImage(
                        at: example.imageURL,
                        maxResolution: config.maxResolution,
                        multiple: 32
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

            return [
                PhasePlan(
                    width: config.width,
                    height: config.height,
                    steps: config.trainingSteps,
                    sampleIndices: Array(examples.indices)
                )
            ]
        }()

        guard !phasePlans.isEmpty else {
            throw NSError(
                domain: "ZImageTurboLoRATrainer",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No training phases were produced from dataset resolution bucketing."]
            )
        }

        if progressHandler != nil, phasePlans.count > 1 {
            let label = adaptiveResolution ? "Adaptive resolution buckets" : "Training schedule"
            let schedule = phasePlans
                .map { "\($0.width)x\($0.height)@\($0.steps)" }
                .joined(separator: " -> ")
            FileHandle.standardError.write(Data("[ZImageLoRATrainer] \(label): \(schedule)\n".utf8))
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
            "model_id": config.baseModelId,
            "manifest_variant": modelManifest.variant?.rawValue ?? "",
            "dataset_root": config.datasetRoot ?? "",
            "size": "\(config.width)x\(config.height)",
            "max_resolution": config.maxResolution.map { "\($0)" } ?? "",
            "resolution_buckets": resolutionBucketSummary,
            "scheduler_steps": "\(config.schedulerSteps)",
            "training_steps": "\(config.trainingSteps)",
            "batch_size": "\(config.batchSize)",
            "learning_rate": "\(config.learningRate)",
            "rank": "\(config.loraRank)",
            "alpha": config.loraAlpha.map { "\($0)" } ?? "",
            "max_text_length": "\(config.maxTextLength)",
            "caption_dropout": "\(config.captionDropout)",
            "checkpoint_interval": config.checkpointInterval.map { "\($0)" } ?? "",
            "sample_interval": config.sampleInterval.map { "\($0)" } ?? "",
            "lite_mode": "\(config.liteMode)",
            "gradient_checkpointing": "\(config.gradientCheckpointing)",
            "low_ram": "\(config.lowRam)",
            "timestep_sampling": config.timestepSampling.rawValue,
            "timestep_loss_weighting": config.timestepLossWeighting.rawValue,
            "loss_weighting": config.lossWeighting.rawValue,
            "wavelet_loss": "\(config.waveletLoss)",
            "wavelet_loss_weight": "\(config.waveletLossWeight)",
            "differential_guidance": "\(config.doDifferentialGuidance)",
            "differential_guidance_scale": "\(config.differentialGuidanceScale)",
            "use_dynamic_sigma_shift": "\(config.useDynamicSigmaShift)",
            "sigma_shift": config.sigmaShift.map { "\($0)" } ?? "",
            "timestep_low": "\(config.timestepLow)",
            "timestep_high": config.timestepHigh.map { "\($0)" } ?? "",
            "synthetic_sample_count": config.syntheticSampleCount.map { "\($0)" } ?? "",
            "lora_target_prefixes": (config.loraTargetPrefixes ?? ZImageLoRAInjector.aiToolkitCompatiblePrefixes).joined(separator: ","),
            "lora_target_suffixes": (config.loraTargetSuffixes ?? (config.liteMode ? ZImageLoRAInjector.liteTargetSuffixes : ZImageLoRAInjector.aiToolkitCompatibleSuffixes)).joined(separator: ","),
            "lora_target_ranks": serializedTargetRanks,
            "assistant_lora_path": config.assistantLoRAPath ?? "",
            "use_training_adapter": "\(effectiveUseTrainingAdapter)",
            "config_fingerprint": configFingerprint,
        ]

        var phaseData: [PhaseData] = []
        phaseData.reserveCapacity(phasePlans.count)

        var lowRamCaches: [TrainingDataCache] = []
        defer {
            for cache in lowRamCaches {
                try? cache.clear()
            }
        }

        if syntheticCount > 0 {
            guard let phase = phasePlans.first else {
                throw NSError(
                    domain: "ZImageTurboLoRATrainer",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Missing synthetic training phase."]
                )
            }
            let latentHeight = phase.height / 8
            let latentWidth = phase.width / 8
            let seqLen = maxLen
            let hidden = configs.textEncoder.hiddenSize
            var prepared: [PreparedExample] = []
            prepared.reserveCapacity(syntheticCount)
            for _ in 0..<syntheticCount {
                let cleanLatents = MLXRandom.normal([1, 16, latentHeight, latentWidth]).asType(.bfloat16)
                let promptEmbeds = MLXRandom.normal([1, seqLen, hidden]).asType(.bfloat16)
                prepared.append(PreparedExample(cleanLatents: cleanLatents, promptEmbeds: promptEmbeds))
            }

            let trainingSigmas: [Float] = {
                if let fixedShift = config.sigmaShift {
                    return computeTrainingSigmas(numSteps: config.schedulerSteps, sigmaShift: fixedShift)
                }
                if config.useDynamicSigmaShift {
                    return computeTrainingSigmas(numSteps: config.schedulerSteps, width: phase.width, height: phase.height)
                }
                return computeTrainingSigmas(numSteps: config.schedulerSteps)
            }()

            phaseData.append(
                PhaseData(
                    phase: phase,
                    latentHeight: latentHeight,
                    latentWidth: latentWidth,
                    trainingSigmas: trainingSigmas,
                    prepared: prepared,
                    preparedCount: prepared.count,
                    cache: nil
                )
            )

            progressHandler?(ZImageTurboLoRATrainingProgress(
                stage: .encodingDataset(current: syntheticCount, total: syntheticCount),
                fraction: 1
            ))
        } else {
            var preparedPrompts: [PreparedPrompt] = []
            preparedPrompts.reserveCapacity(examples.count)
            for example in examples {
                let caption = example.caption.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !caption.isEmpty else {
                    throw ZImageTurboLoRATrainerError.captionEmpty(example.imageURL)
                }

                let promptEmbeds = try encodePromptEmbeds(
                    prompt: caption,
                    tokenizer: tokenizer,
                    textEncoder: textEncoder
                ).asType(.bfloat16)
                MLX.eval(promptEmbeds)
                preparedPrompts.append(
                    PreparedPrompt(
                        imageURL: example.imageURL,
                        promptEmbeds: promptEmbeds
                    )
                )
            }

            let totalEncodingSteps = preparedPrompts.count + phasePlans.reduce(0) { $0 + $1.sampleIndices.count }
            var encodingStep = preparedPrompts.count
            if totalEncodingSteps > 0 {
                progressHandler?(ZImageTurboLoRATrainingProgress(
                    stage: .encodingDataset(current: encodingStep, total: totalEncodingSteps),
                    fraction: Float(encodingStep) / Float(totalEncodingSteps)
                ))
            }

            for (phaseIndex, phase) in phasePlans.enumerated() {
                let latentHeight = phase.height / 8
                let latentWidth = phase.width / 8
                let phaseCache: TrainingDataCache? = try {
                    guard useLowRam else { return nil }
                    let cacheDir = outputURL.deletingLastPathComponent()
                        .appendingPathComponent(".zero_cache", isDirectory: true)
                        .appendingPathComponent("zimage-phase\(phaseIndex)-\(UUID().uuidString)", isDirectory: true)
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
                    progressHandler?(ZImageTurboLoRATrainingProgress(
                        stage: .encodingDataset(current: encodingStep, total: totalEncodingSteps),
                        fraction: Float(encodingStep) / Float(max(totalEncodingSteps, 1))
                    ))

                    let item = preparedPrompts[promptIndex]
                    let cleanLatents = try encodeTrainingImage(
                        item.imageURL,
                        vae: vae,
                        width: phase.width,
                        height: phase.height,
                        latentHeight: latentHeight,
                        latentWidth: latentWidth
                    )
                    if let phaseCache {
                        MLX.eval(cleanLatents, item.promptEmbeds)
                        try phaseCache.save(
                            id: localIndex,
                            latents: cleanLatents,
                            cond: item.promptEmbeds,
                            width: phase.width,
                            height: phase.height
                        )
                        Memory.clearCache()
                    } else {
                        prepared.append(PreparedExample(cleanLatents: cleanLatents, promptEmbeds: item.promptEmbeds))
                    }
                }

                let trainingSigmas: [Float] = {
                    if let fixedShift = config.sigmaShift {
                        return computeTrainingSigmas(numSteps: config.schedulerSteps, sigmaShift: fixedShift)
                    }
                    if config.useDynamicSigmaShift {
                        return computeTrainingSigmas(numSteps: config.schedulerSteps, width: phase.width, height: phase.height)
                    }
                    return computeTrainingSigmas(numSteps: config.schedulerSteps)
                }()

                phaseData.append(
                    PhaseData(
                        phase: phase,
                        latentHeight: latentHeight,
                        latentWidth: latentWidth,
                        trainingSigmas: trainingSigmas,
                        prepared: phaseCache == nil ? prepared : nil,
                        preparedCount: phase.sampleIndices.count,
                        cache: phaseCache
                    )
                )
            }

            progressHandler?(ZImageTurboLoRATrainingProgress(
                stage: .encodingDataset(current: totalEncodingSteps, total: totalEncodingSteps),
                fraction: 1
            ))
        }

        if config.timingEnabled {
            let memMB = Double(Memory.activeMemory) / 1024.0 / 1024.0
            print(String(format: "[ZImageLoRATrainer] GPU active memory after encode: %.1f MB", memMB))
        }

        // Pre-compute empty prompt embedding for caption dropout
        let emptyPromptEmbeds: MLXArray? = config.captionDropout > 0 ? try {
            if syntheticCount > 0 {
                let seqLen = maxLen
                let hidden = configs.textEncoder.hiddenSize
                return MLXArray.zeros([1, seqLen, hidden], dtype: .bfloat16)
            }
            let embeds = try encodePromptEmbeds(
                prompt: "",
                tokenizer: tokenizer,
                textEncoder: textEncoder
            ).asType(.bfloat16)
            return embeds
        }() : nil

        // Inject LoRA wrappers
        // Use ai-toolkit compatible settings: only train main layers block (not refiners)
        // and include adaLN_modulation.0 in addition to attention/feedforward projections
        let targetPrefixes = config.loraTargetPrefixes ?? ZImageLoRAInjector.aiToolkitCompatiblePrefixes
        let targetSuffixes = config.loraTargetSuffixes ?? (
            config.liteMode
                ? ZImageLoRAInjector.liteTargetSuffixes
                : ZImageLoRAInjector.aiToolkitCompatibleSuffixes
        )
        // Default alpha to rank/3 for better inference scaling (works at lora-scale=1.0)
        let effectiveAlpha = config.loraAlpha ?? (Float(config.loraRank) / 3.0)
        let loraLayers = try ZImageLoRAInjector.inject(
            into: transformer,
            rank: config.loraRank,
            alpha: effectiveAlpha,
            targetPrefixes: targetPrefixes,
            targetSuffixes: targetSuffixes,
            targetRanks: config.loraTargetRanks,
            zeroInitUp: true
        )

        var resumeStep = 0
        var resumedRNGState: UInt64? = nil
        var resumedSeed: UInt64? = nil
        var resumeIteratorCursor: MFluxResumeIteratorCompat.Cursor? = nil
        var resolvedResumeCheckpoint: LoRAResolvedCheckpoint? = nil
        var resumeSidecar: LoRATrainingCheckpointState? = nil
        var resumeRunManifest: LoRATrainingRunManifest? = nil

        // Load existing weights if resuming
        if let resumeFromLoRA {
            let resolvedCheckpoint = try LoRACheckpointResolver.resolve(resumeFromLoRA)
            resolvedResumeCheckpoint = resolvedCheckpoint
            let resumeURL = resolvedCheckpoint.checkpointURL
            let loadedCount = try ZImageLoRAInjector.loadWeights(
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
                    expectedModel: config.baseModelId,
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
                        domain: "ZImageTurboLoRATrainer",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "Resume dataset fingerprint mismatch for \(resumeURL.lastPathComponent)."]
                    )
                }
                if let fingerprint = sidecar.configFingerprint, fingerprint != configFingerprint {
                    throw NSError(
                        domain: "ZImageTurboLoRATrainer",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "Resume config fingerprint mismatch for \(resumeURL.lastPathComponent)."]
                    )
                }
                if let schedule = sidecar.phaseSchedule,
                   !LoRATrainingCheckpointState.scheduleMatches(expected: checkpointPhaseSchedule, actual: schedule) {
                    throw NSError(
                        domain: "ZImageTurboLoRATrainer",
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
                        domain: "ZImageTurboLoRATrainer",
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
                    domain: "ZImageTurboLoRATrainer",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Resume step \(resumeStep) is already >= requested total steps \(config.trainingSteps)."]
                )
            }

            print("[ZImageLoRATrainer] Resumed from \(resumeURL.lastPathComponent): loaded \(loadedCount)/\(loraLayers.count) layers, start step \(resumeStep)")
        }
        defer { resolvedResumeCheckpoint?.cleanup() }

        // Freeze base model; unfreeze only LoRA params
        try transformer.freeze(recursive: true, keys: nil, strict: false)
        for layer in loraLayers.values {
            guard let module = layer as? Module else { continue }
            try module.unfreeze(recursive: false, keys: ["loraDown", "loraUp"], strict: true)
        }

        MLX.eval(transformer)
        progressHandler?(ZImageTurboLoRATrainingProgress(stage: .injectingLoRA(layerCount: loraLayers.count), fraction: 1))

        // Sanity check: ensure LoRA params are trainable (otherwise grads will be empty).
        let trainableKeys = Set(transformer.trainableParameters().flattened().map { $0.0 })
        let trainableLoRACount = loraLayers.keys.reduce(into: 0) { count, path in
            if trainableKeys.contains("\(path).loraDown") && trainableKeys.contains("\(path).loraUp") {
                count += 1
            }
        }
        if trainableLoRACount == 0 {
            print("[ZImageLoRATrainer] Warning: LoRA parameters not found in trainableParameters(). Training updates will be zero.")
        }

        // Initialize per-LoRA Adam state for compiled training.
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

        // Adam hyperparameters (ai-toolkit uses weight_decay=0.0001)
        let beta1: Float = 0.9
        let beta2: Float = 0.999
        let adamEps: Float = 1e-8
        let weightDecay: Float = 0.0001

        // Capture config values for use in closure
        let useWaveletLoss = config.waveletLoss
        let waveletWeight = config.waveletLossWeight
        let useDifferentialGuidance = config.doDifferentialGuidance
        let differentialGuidanceScale = config.differentialGuidanceScale

        // Training loop using flow matching loss
        let lossAndGrad = valueAndGrad(model: transformer) { model, arrays in
            let clean = arrays[0]  // [B, 16, H/8, W/8]
            let noise = arrays[1]  // [B, 16, H/8, W/8]
            let promptEmbeds = arrays[2]  // [B, seqLen, dim]
            let sigmaInput = arrays[3].asType(.float32)  // scalar or [B]
            let timestepLossWeightInput = arrays[4].asType(.float32)  // scalar or [B]
            let snrWeightInput = arrays[5].asType(.float32)  // scalar or [B]
            let sigmaBroadcast = sigmaInput.ndim == 0
                ? sigmaInput
                : sigmaInput.expandedDimensions(axis: 1).expandedDimensions(axis: 2).expandedDimensions(axis: 3)
            let timestepLossWeight = timestepLossWeightInput.ndim == 0
                ? timestepLossWeightInput
                : timestepLossWeightInput.expandedDimensions(axis: 1).expandedDimensions(axis: 2).expandedDimensions(axis: 3)
            let snrWeight = snrWeightInput.ndim == 0
                ? snrWeightInput
                : snrWeightInput.expandedDimensions(axis: 1).expandedDimensions(axis: 2).expandedDimensions(axis: 3)

            // Interpolate between clean and noise
            let latentsT = ((MLXArray(1.0) - sigmaBroadcast) * clean.asType(.float32) + sigmaBroadcast * noise.asType(.float32))
                .asType(.bfloat16)

            // Z-Image uses (1 - sigma) * t_scale for timestep embedding
            let tInput = (MLXArray(1.0) - sigmaInput).asType(.float32)

            let pred = model.forward(latents: latentsT, timestep: tInput, promptEmbeds: promptEmbeds)

            // Flow matching loss: target is (noise - clean) per ai-toolkit convention
            // Model predicts negative of this, so we negate pred
            let velocity = -pred.asType(.float32)
            var target = noise.asType(.float32) - clean.asType(.float32)

            // Apply differential guidance (ai-toolkit: target = pred + scale * (target - pred))
            // This amplifies the error signal to encourage stronger corrections
            if useDifferentialGuidance {
                let scale = MLXArray(differentialGuidanceScale)
                target = velocity + scale * (target - velocity)
            }

            var weightedError = (velocity - target).square() * timestepLossWeight * snrWeight

            // Add wavelet loss if enabled, scaled by average timestep/SNR weighting.
            if useWaveletLoss {
                let waveletLoss = computeWaveletLoss(prediction: velocity, target: target)
                let timestepScale = timestepLossWeightInput.ndim == 0
                    ? timestepLossWeightInput
                    : timestepLossWeightInput.mean()
                let snrScale = snrWeightInput.ndim == 0
                    ? snrWeightInput
                    : snrWeightInput.mean()
                let waveletTerm = MLXArray(waveletWeight) * waveletLoss * timestepScale * snrScale
                weightedError = weightedError + waveletTerm
            }

            return [weightedError.mean()]
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

        // Pre-compute timestep sampling weights
        let timestepWeights = computeTimestepWeights(
            numSteps: config.schedulerSteps,
            strategy: config.timestepSampling
        )

        // Optional timestep loss weights (ai-toolkit `timestep_type: weighted` / linear_timesteps)
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
                Data("[ZImageLoRATrainer] Benchmark running: warmup \(benchmarkWarmupSteps) + measure \(benchmarkSteps)\n".utf8)
            )
        }

        let timingEnabled = config.timingEnabled
        let logTiming = { (step: Int, prep: Double, fwd: Double, opt: Double, total: Double) in
            let msg = String(format: "[ZImageLoRATrainer] step %d timing: prep=%.3fs fwd+bwd=%.3fs opt=%.3fs total=%.3fs\n", step, prep, fwd, opt, total)
            FileHandle.standardError.write(Data(msg.utf8))
        }

        do {
            var globalStep = resumeStep
            var remainingSkippedSteps = resumeStep
            var resumeIteratorCursorConsumed = false

            for (phaseIndex, phase) in phaseData.enumerated() {
                if globalStep >= config.trainingSteps { break }
                if phase.phase.steps <= 0 || phase.preparedCount <= 0 { continue }
                if remainingSkippedSteps >= phase.phase.steps {
                    remainingSkippedSteps -= phase.phase.steps
                    continue
                }

                let phaseStartStep = remainingSkippedSteps
                remainingSkippedSteps = 0

                let prepared = phase.prepared
                let phaseCache = phase.cache
                let preparedCount = phase.preparedCount
                let latentHeight = phase.latentHeight
                let latentWidth = phase.latentWidth
                let trainingSigmas = phase.trainingSigmas
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
                        domain: "ZImageTurboLoRATrainer",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "Missing phase training data at runtime."]
                    )
                }

                var compiledTrainStep: (([MLXArray]) -> [MLXArray])? = nil

                for _ in phaseStartStep..<phase.phase.steps {
                    if globalStep >= config.trainingSteps { break }
                    currentStep = globalStep
                    try Task.checkCancellation()

                    if benchmarkStartTime == nil,
                       let end = benchmarkEndStep,
                       globalStep == benchmarkWarmupSteps,
                       globalStep < end {
                        Stream.gpu.synchronize()
                        benchmarkStartTime = CFAbsoluteTimeGetCurrent()
                    }

                    let stepStart = CFAbsoluteTimeGetCurrent()
                    let requestedBatchSize = min(config.batchSize, preparedCount)
                    let sampledIndices = MFluxResumeIteratorCompat.nextBatchIndices(
                            requestedBatchSize: requestedBatchSize,
                            sampleCount: preparedCount,
                            cursor: &phaseIteratorCursor
                        ) ?? []
                    guard !sampledIndices.isEmpty else {
                        throw NSError(
                            domain: "ZImageTurboLoRATrainer",
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

                    var cleanBatchParts: [MLXArray] = []
                    var promptBatchParts: [MLXArray] = []
                    cleanBatchParts.reserveCapacity(batchSize)
                    promptBatchParts.reserveCapacity(batchSize)

                    for idx in sampledIndices {
                        let item: PreparedExample
                        if let phaseCache {
                            let cached = try phaseCache.load(id: idx)
                            item = PreparedExample(
                                cleanLatents: cached.latents,
                                promptEmbeds: cached.cond
                            )
                        } else if let prepared {
                            item = prepared[idx]
                        } else {
                            throw NSError(
                                domain: "ZImageTurboLoRATrainer",
                                code: 1,
                                userInfo: [NSLocalizedDescriptionKey: "Missing phase training data at runtime."]
                            )
                        }
                        cleanBatchParts.append(item.cleanLatents)

                        // Caption dropout
                        let useEmptyPrompt = emptyPromptEmbeds != nil &&
                            Float(rng.next() % 1000) / 1000.0 < config.captionDropout
                        if useEmptyPrompt, let empty = emptyPromptEmbeds {
                            promptBatchParts.append(empty)
                        } else {
                            promptBatchParts.append(item.promptEmbeds)
                        }
                    }

                    let cleanBatch = MLX.concatenated(cleanBatchParts, axis: 0)
                    let promptBatch = MLX.concatenated(promptBatchParts, axis: 0)

                    // Sample timestep (respecting timestepLow/timestepHigh range)
                    let timestepHigh = config.timestepHigh ?? config.schedulerSteps
                    let timestepLow = config.timestepLow
                    var timestepIndices: [Int] = []
                    var sigmaValues: [Float] = []
                    var snrWeights: [Float] = []
                    var noiseParts: [MLXArray] = []
                    timestepIndices.reserveCapacity(batchSize)
                    sigmaValues.reserveCapacity(batchSize)
                    snrWeights.reserveCapacity(batchSize)
                    noiseParts.reserveCapacity(batchSize)
                    for seedPair in stepSeedPairs {
                        let timestepIndex: Int
                        let sigmaValue: Float
                        if config.timestepSampling == .sigmoid {
                            var timestepRNG = SplitMix64(rawState: seedPair.time)
                            let sigmaSample = sampleSigmoidTimestep(
                                rng: &timestepRNG,
                                low: timestepLow,
                                high: timestepHigh,
                                numSteps: config.schedulerSteps
                            )
                            sigmaValue = sigmaSample.item(Float.self)
                            timestepIndex = Int((sigmaValue * Float(config.schedulerSteps)).rounded())
                                .clamped(to: timestepLow..<timestepHigh)
                        } else {
                            var timestepRNG = SplitMix64(rawState: seedPair.time)
                            timestepIndex = sampleWeightedTimestep(
                                weights: timestepWeights,
                                rng: &timestepRNG,
                                low: timestepLow,
                                high: timestepHigh
                            )
                            sigmaValue = trainingSigmas[timestepIndex]
                        }

                        timestepIndices.append(timestepIndex)
                        sigmaValues.append(sigmaValue)
                        snrWeights.append(computeSNRWeight(sigma: sigmaValue, strategy: config.lossWeighting))

                        let noisePart = MLXRandom.normal(
                            [1, 16, latentHeight, latentWidth],
                            key: MLXRandom.key(seedPair.noise)
                        ).asType(.bfloat16)
                        noiseParts.append(noisePart)
                    }
                    let noise = batchSize == 1
                        ? noiseParts[0]
                        : MLX.concatenated(noiseParts, axis: 0)
                    let sigma = MLXArray(sigmaValues).asType(.float32)
                    lastRNGState = rng.rawState

                    // Timestep loss weight
                    let timestepLossWeight = MLXArray(
                        timestepIndices.map { timestepLossWeights[$0.clamped(to: 0..<timestepLossWeights.count)] }
                    ).asType(.float32)

                    // Compute SNR weight for this timestep
                    let snrWeightArray = MLXArray(snrWeights).asType(.float32)

                    let prepEnd = CFAbsoluteTimeGetCurrent()

                    let loss: MLXArray
                    let fwdEnd: CFTimeInterval
                    if compiledTrainStep == nil {
                        let state: [any Updatable] = [loraState]
                        let lr = MLXArray(config.learningRate)
                        let beta1Arr = MLXArray(beta1)
                        let beta2Arr = MLXArray(beta2)
                        let oneMinusBeta1 = MLXArray(1.0 - beta1)
                        let oneMinusBeta2 = MLXArray(1.0 - beta2)
                        let epsArr = MLXArray(adamEps)
                        let oneMinusLrWd = MLXArray(1.0 - config.learningRate * weightDecay)

                        compiledTrainStep = compile(inputs: state, outputs: state) { inputs -> [MLXArray] in
                            let (values, grads) = lossAndGrad(transformer, inputs)
                            let gradMap = Dictionary(uniqueKeysWithValues: grads.flattened())
                            Self.applyAdamW(
                                loraLayers: loraLayerList,
                                gradMap: gradMap,
                                lr: lr,
                                beta1: beta1Arr,
                                beta2: beta2Arr,
                                oneMinusBeta1: oneMinusBeta1,
                                oneMinusBeta2: oneMinusBeta2,
                                eps: epsArr,
                                oneMinusLrWd: oneMinusLrWd,
                                useWeightDecay: true
                            )
                            return values
                        }
                    }

                    if let compiled = compiledTrainStep {
                        loss = compiled([cleanBatch, noise, promptBatch, sigma, timestepLossWeight, snrWeightArray])[0]
                        fwdEnd = CFAbsoluteTimeGetCurrent()
                    } else {
                        let (values, grads) = lossAndGrad(
                            transformer,
                            [cleanBatch, noise, promptBatch, sigma, timestepLossWeight, snrWeightArray]
                        )
                        loss = values.first!
                        fwdEnd = CFAbsoluteTimeGetCurrent()

                        // Manual AdamW update (fallback when compile is unavailable)
                        var flatGrads = grads.flattened()

                        // Gradient clipping (max_grad_norm = 1.0, matches ai-toolkit)
                        let maxGradNorm: Float = 1.0
                        let gradNormSquared = flatGrads.reduce(MLXArray(0.0)) { acc, kv in
                            acc + kv.1.square().sum()
                        }
                        let gradNorm = gradNormSquared.sqrt()
                        let clipCoef = MLXArray(maxGradNorm) / (gradNorm + MLXArray(1e-6))
                        let shouldClip = gradNorm .> MLXArray(maxGradNorm)
                        flatGrads = flatGrads.map { (key, grad) in
                            (key, MLX.where(shouldClip, grad * clipCoef, grad))
                        }
                        let stepF = Float(globalStep + 1)
                        let biasCorrection1 = 1 - powf(beta1, stepF)
                        let biasCorrection2 = 1 - powf(beta2, stepF)
                        for (layerPath, layer) in loraLayers {
                            guard let downGrad = flatGrads.first(where: { $0.0 == "\(layerPath).loraDown" })?.1,
                                  let upGrad = flatGrads.first(where: { $0.0 == "\(layerPath).loraUp" })?.1 else {
                                continue
                            }

                            var mDown = layer.loraDownM ?? MLXArray.zeros(like: downGrad)
                            var vDown = layer.loraDownV ?? MLXArray.zeros(like: downGrad)
                            mDown = beta1 * mDown + (1 - beta1) * downGrad
                            vDown = beta2 * vDown + (1 - beta2) * downGrad.square()
                            let mDownHat = mDown / biasCorrection1
                            let vDownHat = vDown / biasCorrection2
                            let downUpdate = mDownHat / (vDownHat.sqrt() + adamEps)
                            if weightDecay != 0 {
                                layer.loraDown = layer.loraDown * (1 - config.learningRate * weightDecay)
                            }
                            layer.loraDown = layer.loraDown - config.learningRate * downUpdate
                            layer.loraDownM = mDown
                            layer.loraDownV = vDown

                            var mUp = layer.loraUpM ?? MLXArray.zeros(like: upGrad)
                            var vUp = layer.loraUpV ?? MLXArray.zeros(like: upGrad)
                            mUp = beta1 * mUp + (1 - beta1) * upGrad
                            vUp = beta2 * vUp + (1 - beta2) * upGrad.square()
                            let mUpHat = mUp / biasCorrection1
                            let vUpHat = vUp / biasCorrection2
                            let upUpdate = mUpHat / (vUpHat.sqrt() + adamEps)
                            if weightDecay != 0 {
                                layer.loraUp = layer.loraUp * (1 - config.learningRate * weightDecay)
                            }
                            layer.loraUp = layer.loraUp - config.learningRate * upUpdate
                            layer.loraUpM = mUp
                            layer.loraUpV = vUp
                        }
                    }

                    if let emaState {
                        MLX.asyncEval(loss, loraState, emaState)
                    } else {
                        MLX.asyncEval(loss, loraState)
                    }
                    let optEnd = CFAbsoluteTimeGetCurrent()

                    // Update EMA weights
                    emaState?.update(from: loraLayers)

                    if timingEnabled {
                        logTiming(globalStep + 1, prepEnd - stepStart, fwdEnd - prepEnd, optEnd - fwdEnd, optEnd - stepStart)
                    }

                    if benchmarkSteps == nil {
                        let lossValue: Float? = {
                            if (globalStep + 1) % max(config.logEvery, 1) == 0 || globalStep == config.trainingSteps - 1 {
                                return loss.item(Float.self)
                            }
                            return nil
                        }()

                        if let lossValue {
                            try metricsLogger.record(step: globalStep + 1, loss: lossValue)
                            progressHandler?(ZImageTurboLoRATrainingProgress(
                                stage: .training(step: globalStep + 1, total: config.trainingSteps, loss: lossValue),
                                fraction: Float(globalStep + 1) / Float(config.trainingSteps)
                            ))
                        } else {
                            progressHandler?(ZImageTurboLoRATrainingProgress(
                                stage: .training(step: globalStep + 1, total: config.trainingSteps, loss: nil),
                                fraction: Float(globalStep + 1) / Float(config.trainingSteps)
                            ))
                        }
                    }

                    // Save checkpoints and preview samples independently.
                    let step = globalStep + 1
                    let shouldCheckpoint = config.checkpointInterval.map { step % $0 == 0 } ?? false
                    let shouldGenerateSample = (config.sampleInterval.map { step % $0 == 0 } ?? false) && sampleHandler != nil
                    if (shouldCheckpoint || shouldGenerateSample), step < config.trainingSteps {
                        if shouldGenerateSample {
                            progressHandler?(ZImageTurboLoRATrainingProgress(
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
                                "format": "mererun.zimage.lora",
                                "base_model": config.baseModelId,
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
                                format: "mererun.zimage.lora",
                                baseModel: config.baseModelId,
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
                                format: "mererun.zimage.lora",
                                baseModel: config.baseModelId,
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
                                    alpha: effectiveAlpha,
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
                                    "adaptive_resolution": "\(adaptiveResolution)",
                                    "max_resolution": config.maxResolution.map { "\($0)" } ?? "",
                                    "resolution_buckets": resolutionBucketSummary,
                                    "lite_mode": "\(config.liteMode)",
                                    "gradient_checkpointing": "\(config.gradientCheckpointing)",
                                    "wavelet_loss": "\(config.waveletLoss)",
                                    "wavelet_loss_weight": "\(config.waveletLossWeight)",
                                    "differential_guidance": "\(config.doDifferentialGuidance)",
                                    "differential_guidance_scale": "\(config.differentialGuidanceScale)",
                                    "use_dynamic_sigma_shift": "\(config.useDynamicSigmaShift)",
                                    "sigma_shift": config.sigmaShift.map { "\($0)" } ?? "",
                                    "assistant_lora_path": config.assistantLoRAPath ?? "",
                                    "use_training_adapter": "\(effectiveUseTrainingAdapter)",
                                    "low_ram": "\(config.lowRam)",
                                    "timing_enabled": "\(config.timingEnabled)",
                                    "synthetic_sample_count": config.syntheticSampleCount.map { "\($0)" } ?? "",
                                    "lora_target_prefixes": config.loraTargetPrefixes?.joined(separator: ",") ?? "",
                                    "lora_target_suffixes": config.loraTargetSuffixes?.joined(separator: ",") ?? "",
                                    "lora_target_ranks": serializedTargetRanks,
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
                                format: "mererun.zimage.lora",
                                model: config.baseModelId,
                                isEdit: false,
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
                                print("[ZImageLoRATrainer] Warning: failed to archive checkpoint \(checkpointURL.lastPathComponent): \(error.localizedDescription)")
                            }
                        } else if shouldGenerateSample {
                            await sampleHandler?(step, checkpointURL)
                            try? FileManager.default.removeItem(at: checkpointURL)
                        }
                    }

                    if let end = benchmarkEndStep, globalStep + 1 == end {
                        Stream.gpu.synchronize()
                        if let start = benchmarkStartTime, let steps = benchmarkSteps {
                            let elapsed = CFAbsoluteTimeGetCurrent() - start
                            let avg = elapsed / Double(steps)
                            FileHandle.standardError.write(
                                Data(String(format: "[ZImageLoRATrainer] Benchmark: %d steps avg %.3fs/step (elapsed %.2fs)\n", steps, avg, elapsed).utf8)
                            )
                        }
                        return
                    }

                    globalStep += 1
                }
            }
        } catch is CancellationError {
            if let handler = cancellationHandler {
                let shouldSave = await handler(currentStep)
                if shouldSave && currentStep > 0 {
                    progressHandler?(ZImageTurboLoRATrainingProgress(stage: .saving, fraction: 0))
                    try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try LoRASafetensorsWriter.save(
                        loraLayers: loraLayers,
                        to: outputURL,
                        dtype: config.saveDType,
                        includeOptimizerState: true,
                        metadata: [
                            "format": "mererun.zimage.lora",
                            "base_model": config.baseModelId,
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
                        format: "mererun.zimage.lora",
                        baseModel: config.baseModelId,
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
                                "format": "mererun.zimage.lora",
                                "base_model": config.baseModelId,
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
                    let manifest = LoRATrainingManifest(
                        format: "mererun.zimage.lora",
                        baseModel: config.baseModelId,
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
                            alpha: effectiveAlpha,
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
                            "adaptive_resolution": "\(adaptiveResolution)",
                            "max_resolution": config.maxResolution.map { "\($0)" } ?? "",
                            "resolution_buckets": resolutionBucketSummary,
                            "lite_mode": "\(config.liteMode)",
                            "gradient_checkpointing": "\(config.gradientCheckpointing)",
                            "wavelet_loss": "\(config.waveletLoss)",
                            "wavelet_loss_weight": "\(config.waveletLossWeight)",
                            "differential_guidance": "\(config.doDifferentialGuidance)",
                            "differential_guidance_scale": "\(config.differentialGuidanceScale)",
                            "use_dynamic_sigma_shift": "\(config.useDynamicSigmaShift)",
                            "sigma_shift": config.sigmaShift.map { "\($0)" } ?? "",
                            "assistant_lora_path": config.assistantLoRAPath ?? "",
                            "use_training_adapter": "\(effectiveUseTrainingAdapter)",
                            "low_ram": "\(config.lowRam)",
                            "timing_enabled": "\(config.timingEnabled)",
                            "synthetic_sample_count": config.syntheticSampleCount.map { "\($0)" } ?? "",
                            "lora_target_prefixes": config.loraTargetPrefixes?.joined(separator: ",") ?? "",
                            "lora_target_suffixes": config.loraTargetSuffixes?.joined(separator: ",") ?? "",
                            "lora_target_ranks": serializedTargetRanks,
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
                    try manifest.write(nextTo: outputURL)
                    let cancelledRunManifest = LoRATrainingRunManifest(
                        format: "mererun.zimage.lora",
                        model: config.baseModelId,
                        isEdit: false,
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
                        print("[ZImageLoRATrainer] Warning: failed to archive checkpoint \(outputURL.lastPathComponent): \(error.localizedDescription)")
                    }

                    progressHandler?(ZImageTurboLoRATrainingProgress(stage: .saving, fraction: 1))
                    return
                }
            }
            throw CancellationError()
        }

        // Save final LoRA
        progressHandler?(ZImageTurboLoRATrainingProgress(stage: .saving, fraction: 0))
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        try LoRASafetensorsWriter.save(
            loraLayers: loraLayers,
            to: outputURL,
            dtype: config.saveDType,
            includeOptimizerState: true,
            metadata: [
                "format": "mererun.zimage.lora",
                "base_model": config.baseModelId,
                "step": "\(config.trainingSteps)",
                "total_steps": "\(config.trainingSteps)",
                "seed": "\(effectiveSeed)",
                "rng_state": "\(lastRNGState)",
                "dataset_fingerprint": datasetFingerprint,
                "config_fingerprint": configFingerprint,
            ]
        )
        let checkpointState = LoRATrainingCheckpointState(
            format: "mererun.zimage.lora",
            baseModel: config.baseModelId,
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
                    "format": "mererun.zimage.lora",
                    "base_model": config.baseModelId,
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
            format: "mererun.zimage.lora",
            baseModel: config.baseModelId,
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
                alpha: effectiveAlpha,
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
                "adaptive_resolution": "\(adaptiveResolution)",
                "max_resolution": config.maxResolution.map { "\($0)" } ?? "",
                "resolution_buckets": resolutionBucketSummary,
                "lite_mode": "\(config.liteMode)",
                "gradient_checkpointing": "\(config.gradientCheckpointing)",
                "wavelet_loss": "\(config.waveletLoss)",
                "wavelet_loss_weight": "\(config.waveletLossWeight)",
                "differential_guidance": "\(config.doDifferentialGuidance)",
                "differential_guidance_scale": "\(config.differentialGuidanceScale)",
                "use_dynamic_sigma_shift": "\(config.useDynamicSigmaShift)",
                "sigma_shift": config.sigmaShift.map { "\($0)" } ?? "",
                "assistant_lora_path": config.assistantLoRAPath ?? "",
                "use_training_adapter": "\(effectiveUseTrainingAdapter)",
                "low_ram": "\(config.lowRam)",
                "timing_enabled": "\(config.timingEnabled)",
                "synthetic_sample_count": config.syntheticSampleCount.map { "\($0)" } ?? "",
                "lora_target_prefixes": config.loraTargetPrefixes?.joined(separator: ",") ?? "",
                "lora_target_suffixes": config.loraTargetSuffixes?.joined(separator: ",") ?? "",
                "lora_target_ranks": serializedTargetRanks,
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
            format: "mererun.zimage.lora",
            model: config.baseModelId,
            isEdit: false,
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
            print("[ZImageLoRATrainer] Warning: failed to archive checkpoint \(outputURL.lastPathComponent): \(error.localizedDescription)")
        }

        progressHandler?(ZImageTurboLoRATrainingProgress(stage: .saving, fraction: 1))
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
            entries = try fileManager.contentsOfDirectory(
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
        examples: [ZImageTurboLoRATrainingExample],
        dataRootPath: String?
    ) -> LoRATrainingRunManifest.DataFingerprint {
        let imagePaths = examples.map { example in
            LoRATrainingRunManifest.dataPath(
                for: example.imageURL,
                dataRootPath: dataRootPath
            )
        }
        return LoRATrainingRunManifest.DataFingerprint(
            count: examples.count,
            images: imagePaths,
            inputImages: [],
            isEdit: false
        )
    }

    private static func validateResumeRunManifest(
        _ manifest: LoRATrainingRunManifest,
        expectedModel: String,
        expectedDataFingerprint: LoRATrainingRunManifest.DataFingerprint,
        expectedDatasetFingerprint: String,
        expectedConfigFingerprint: String,
        expectedPhaseSchedule: [LoRATrainingCheckpointState.Phase],
        checkpointName: String
    ) throws {
        if manifest.model.caseInsensitiveCompare(expectedModel) != .orderedSame {
            throw NSError(
                domain: "ZImageTurboLoRATrainer",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Resume run manifest model mismatch for \(checkpointName). Expected \(expectedModel), got \(manifest.model).",
                ]
            )
        }
        if manifest.isEdit {
            throw NSError(
                domain: "ZImageTurboLoRATrainer",
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
                domain: "ZImageTurboLoRATrainer",
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
                domain: "ZImageTurboLoRATrainer",
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
                domain: "ZImageTurboLoRATrainer",
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
                domain: "ZImageTurboLoRATrainer",
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
                    domain: "ZImageTurboLoRATrainer",
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
                Data("[ZImageLoRATrainer] Warning: failed to decode run.json beside \(checkpointURL.lastPathComponent): \(error.localizedDescription)\n".utf8)
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

// MARK: - Extensions

private extension Int {
    func clamped(to range: Range<Int>) -> Int {
        if self < range.lowerBound { return range.lowerBound }
        if self >= range.upperBound { return range.upperBound - 1 }
        return self
    }
}
