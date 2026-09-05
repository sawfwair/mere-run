import Foundation

public enum Cosmos3ResourcesError: LocalizedError, Sendable {
    case invalidConfiguration(URL, String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let url, let reason):
            return "Invalid Cosmos3 configuration at \(url.path): \(reason)"
        }
    }
}

public enum Cosmos3FeedForwardActivation: String, Codable, Hashable, Sendable {
    case reluSquared = "relu2"
    case siluGated = "silu"
}

public struct Cosmos3QuantizationConfiguration: Codable, Hashable, Sendable {
    public let bits: Int
    public let groupSize: Int
    public let mode: String

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
        case mode
    }
}

public struct Cosmos3TransformerConfiguration: Decodable, Hashable, Sendable {
    public let actionDimension: Int?
    public let generatesActions: Bool
    public let attentionBias: Bool
    public let feedForwardActivation: Cosmos3FeedForwardActivation
    public let baseFPS: Int
    public let enablesFPSModulation: Bool
    public let headDimension: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let latentChannels: Int
    public let latentPatchSize: Int
    public let attentionHeadCount: Int
    public let embodimentDomainCount: Int
    public let layerCount: Int
    public let keyValueHeadCount: Int
    public let patchLatentDimension: Int
    public let normalizesUnderstandingQueriesAndKeys: Bool
    public let rmsNormEpsilon: Float
    public let ropeAxesDimensions: [Int]
    public let ropeTheta: Float
    public let soundDimension: Int?
    public let generatesSound: Bool
    public let temporalCompressionFactor: Int
    public let timestepScale: Float
    public let normalizesUnderstandingKeysForGeneration: Bool
    public let quantization: Cosmos3QuantizationConfiguration?
    public let includesLanguageModelHead: Bool
    public let resetsSpatialPositionIDs: Bool
    public let temporalModalityMargin: Int
    public let vocabularySize: Int

    enum CodingKeys: String, CodingKey {
        case actionDimension = "action_dim"
        case generatesActions = "action_gen"
        case attentionBias = "attention_bias"
        case feedForwardActivation = "hidden_act"
        case baseFPS = "base_fps"
        case enablesFPSModulation = "enable_fps_modulation"
        case headDimension = "head_dim"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case latentChannels = "latent_channel"
        case latentPatchSize = "latent_patch_size"
        case attentionHeadCount = "num_attention_heads"
        case embodimentDomainCount = "num_embodiment_domains"
        case layerCount = "num_hidden_layers"
        case keyValueHeadCount = "num_key_value_heads"
        case patchLatentDimension = "patch_latent_dim"
        case normalizesUnderstandingQueriesAndKeys = "qk_norm_for_text"
        case rmsNormEpsilon = "rms_norm_eps"
        case ropeAxesDimensions = "rope_axes_dim"
        case ropeTheta = "rope_theta"
        case soundDimension = "sound_dim"
        case generatesSound = "sound_gen"
        case temporalCompressionFactor = "temporal_compression_factor"
        case timestepScale = "timestep_scale"
        case normalizesUnderstandingKeysForGeneration = "use_und_k_norm_for_gen"
        case quantization
        case quantizationConfig = "quantization_config"
        case textToImageOnly = "mererun_text_to_image_only"
        case resetsSpatialPositionIDs = "unified_3d_mrope_reset_spatial_ids"
        case temporalModalityMargin = "unified_3d_mrope_temporal_modality_margin"
        case vocabularySize = "vocab_size"
    }

    public init(
        actionDimension: Int? = 64,
        generatesActions: Bool = true,
        attentionBias: Bool = false,
        feedForwardActivation: Cosmos3FeedForwardActivation = .reluSquared,
        baseFPS: Int = 24,
        enablesFPSModulation: Bool = true,
        headDimension: Int = 128,
        hiddenSize: Int = 2_048,
        intermediateSize: Int = 9_216,
        latentChannels: Int = 48,
        latentPatchSize: Int = 2,
        attentionHeadCount: Int = 16,
        embodimentDomainCount: Int = 32,
        layerCount: Int = 28,
        keyValueHeadCount: Int = 8,
        patchLatentDimension: Int = 192,
        normalizesUnderstandingQueriesAndKeys: Bool = false,
        rmsNormEpsilon: Float = 1e-5,
        ropeAxesDimensions: [Int] = [24, 20, 20],
        ropeTheta: Float = 100_000_000,
        soundDimension: Int? = nil,
        generatesSound: Bool = false,
        temporalCompressionFactor: Int = 4,
        timestepScale: Float = 0.001,
        normalizesUnderstandingKeysForGeneration: Bool = true,
        quantization: Cosmos3QuantizationConfiguration? = nil,
        includesLanguageModelHead: Bool = true,
        resetsSpatialPositionIDs: Bool = true,
        temporalModalityMargin: Int = 15_000,
        vocabularySize: Int = 131_072
    ) {
        self.actionDimension = actionDimension
        self.generatesActions = generatesActions
        self.attentionBias = attentionBias
        self.feedForwardActivation = feedForwardActivation
        self.baseFPS = baseFPS
        self.enablesFPSModulation = enablesFPSModulation
        self.headDimension = headDimension
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.latentChannels = latentChannels
        self.latentPatchSize = latentPatchSize
        self.attentionHeadCount = attentionHeadCount
        self.embodimentDomainCount = embodimentDomainCount
        self.layerCount = layerCount
        self.keyValueHeadCount = keyValueHeadCount
        self.patchLatentDimension = patchLatentDimension
        self.normalizesUnderstandingQueriesAndKeys = normalizesUnderstandingQueriesAndKeys
        self.rmsNormEpsilon = rmsNormEpsilon
        self.ropeAxesDimensions = ropeAxesDimensions
        self.ropeTheta = ropeTheta
        self.soundDimension = soundDimension
        self.generatesSound = generatesSound
        self.temporalCompressionFactor = temporalCompressionFactor
        self.timestepScale = timestepScale
        self.normalizesUnderstandingKeysForGeneration = normalizesUnderstandingKeysForGeneration
        self.quantization = quantization
        self.includesLanguageModelHead = includesLanguageModelHead
        self.resetsSpatialPositionIDs = resetsSpatialPositionIDs
        self.temporalModalityMargin = temporalModalityMargin
        self.vocabularySize = vocabularySize
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.actionDimension = try container.decodeIfPresent(Int.self, forKey: .actionDimension)
        self.generatesActions = try container.decodeIfPresent(Bool.self, forKey: .generatesActions) ?? false
        self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.baseFPS = try container.decodeIfPresent(Int.self, forKey: .baseFPS) ?? 24
        self.enablesFPSModulation = try container.decodeIfPresent(Bool.self, forKey: .enablesFPSModulation) ?? true
        self.headDimension = try container.decode(Int.self, forKey: .headDimension)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.latentChannels = try container.decode(Int.self, forKey: .latentChannels)
        self.latentPatchSize = try container.decode(Int.self, forKey: .latentPatchSize)
        self.attentionHeadCount = try container.decode(Int.self, forKey: .attentionHeadCount)
        self.embodimentDomainCount = try container.decodeIfPresent(Int.self, forKey: .embodimentDomainCount) ?? 32
        self.layerCount = try container.decode(Int.self, forKey: .layerCount)
        self.keyValueHeadCount = try container.decode(Int.self, forKey: .keyValueHeadCount)
        self.patchLatentDimension = try container.decode(Int.self, forKey: .patchLatentDimension)
        self.normalizesUnderstandingQueriesAndKeys = try container.decodeIfPresent(
            Bool.self,
            forKey: .normalizesUnderstandingQueriesAndKeys
        ) ?? true
        self.feedForwardActivation = try container.decodeIfPresent(
            Cosmos3FeedForwardActivation.self,
            forKey: .feedForwardActivation
        ) ?? (normalizesUnderstandingQueriesAndKeys ? .siluGated : .reluSquared)
        self.rmsNormEpsilon = try container.decode(Float.self, forKey: .rmsNormEpsilon)
        self.ropeAxesDimensions = try container.decode([Int].self, forKey: .ropeAxesDimensions)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self.soundDimension = try container.decodeIfPresent(Int.self, forKey: .soundDimension)
        self.generatesSound = try container.decodeIfPresent(Bool.self, forKey: .generatesSound) ?? false
        self.temporalCompressionFactor = try container.decodeIfPresent(
            Int.self,
            forKey: .temporalCompressionFactor
        ) ?? 4
        self.timestepScale = try container.decodeIfPresent(Float.self, forKey: .timestepScale) ?? 0.001
        self.normalizesUnderstandingKeysForGeneration = try container.decodeIfPresent(
            Bool.self,
            forKey: .normalizesUnderstandingKeysForGeneration
        ) ?? false
        self.quantization = try container.decodeIfPresent(
            Cosmos3QuantizationConfiguration.self,
            forKey: .quantizationConfig
        ) ?? container.decodeIfPresent(Cosmos3QuantizationConfiguration.self, forKey: .quantization)
        self.includesLanguageModelHead = !(try container.decodeIfPresent(
            Bool.self,
            forKey: .textToImageOnly
        ) ?? false)
        self.resetsSpatialPositionIDs = try container.decodeIfPresent(
            Bool.self,
            forKey: .resetsSpatialPositionIDs
        ) ?? true
        self.temporalModalityMargin = try container.decodeIfPresent(
            Int.self,
            forKey: .temporalModalityMargin
        ) ?? 15_000
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
    }

    public func validationIssues() -> [String] {
        var issues: [String] = []
        if attentionHeadCount % keyValueHeadCount != 0 {
            issues.append("num_attention_heads must be divisible by num_key_value_heads")
        }
        if ropeAxesDimensions.reduce(0, +) != headDimension / 2 {
            issues.append("rope_axes_dim must sum to head_dim / 2")
        }
        if patchLatentDimension != latentChannels * latentPatchSize * latentPatchSize {
            issues.append("patch_latent_dim must equal latent_channel * latent_patch_size^2")
        }
        if generatesActions && actionDimension == nil {
            issues.append("action_dim is required when action_gen is true")
        }
        if generatesSound && soundDimension == nil {
            issues.append("sound_dim is required when sound_gen is true")
        }
        if normalizesUnderstandingQueriesAndKeys == normalizesUnderstandingKeysForGeneration {
            issues.append(
                "exactly one understanding key-normalization path must be enabled"
            )
        }
        if feedForwardActivation == .reluSquared && normalizesUnderstandingQueriesAndKeys {
            issues.append("relu2 checkpoints require qk_norm_for_text=false")
        }
        if feedForwardActivation == .siluGated && !normalizesUnderstandingQueriesAndKeys {
            issues.append("silu checkpoints require qk_norm_for_text=true")
        }
        if let quantization {
            if !(2...8).contains(quantization.bits) {
                issues.append("quantization bits must be between 2 and 8")
            }
            if quantization.groupSize < 1 {
                issues.append("quantization group_size must be positive")
            }
            if quantization.mode != "affine" {
                issues.append("Cosmos3 MLX checkpoints require affine quantization")
            }
        }
        if layerCount < 1 { issues.append("num_hidden_layers must be positive") }
        if embodimentDomainCount < 1 { issues.append("num_embodiment_domains must be positive") }
        if temporalCompressionFactor < 1 { issues.append("temporal_compression_factor must be positive") }
        return issues
    }
}

public struct Cosmos3DistilledConfiguration: Decodable, Hashable, Sendable {
    public let isDistilled: Bool
    public let sigmas: [Float]

    enum CodingKeys: String, CodingKey {
        case isDistilled = "is_distilled"
        case sigmas = "distilled_sigmas"
    }

    public func validationIssues() -> [String] {
        var issues: [String] = []
        if !isDistilled { issues.append("is_distilled must be true") }
        if sigmas.isEmpty { issues.append("distilled_sigmas must not be empty") }
        if sigmas.contains(where: { !$0.isFinite || $0 <= 0 || $0 > 1 }) {
            issues.append("distilled_sigmas must contain values in (0, 1]")
        }
        if zip(sigmas, sigmas.dropFirst()).contains(where: { pair in pair.0 <= pair.1 }) {
            issues.append("distilled_sigmas must be strictly descending")
        }
        return issues
    }
}

private struct Cosmos3DistilledMarker: Decodable {
    let isDistilled: Bool?

    enum CodingKeys: String, CodingKey {
        case isDistilled = "is_distilled"
    }
}

public struct Cosmos3VAEConfiguration: Codable, Hashable, Sendable {
    public let baseDimension: Int
    public let decoderBaseDimension: Int
    public let dimensionMultipliers: [Int]
    public let inputChannels: Int
    public let latentMeans: [Float]
    public let latentStandardDeviations: [Float]
    public let outputChannels: Int
    public let patchSize: Int
    public let spatialScaleFactor: Int
    public let temporalScaleFactor: Int
    public let latentDimension: Int

    enum CodingKeys: String, CodingKey {
        case baseDimension = "base_dim"
        case decoderBaseDimension = "decoder_base_dim"
        case dimensionMultipliers = "dim_mult"
        case inputChannels = "in_channels"
        case latentMeans = "latents_mean"
        case latentStandardDeviations = "latents_std"
        case outputChannels = "out_channels"
        case patchSize = "patch_size"
        case spatialScaleFactor = "scale_factor_spatial"
        case temporalScaleFactor = "scale_factor_temporal"
        case latentDimension = "z_dim"
    }

    public func validationIssues() -> [String] {
        var issues: [String] = []
        if latentMeans.count != latentDimension {
            issues.append("latents_mean count must equal z_dim")
        }
        if latentStandardDeviations.count != latentDimension {
            issues.append("latents_std count must equal z_dim")
        }
        if latentStandardDeviations.contains(where: { $0 <= 0 }) {
            issues.append("latents_std values must be positive")
        }
        if spatialScaleFactor != 16 || temporalScaleFactor != 4 {
            issues.append("Cosmos3-Edge requires the Wan 4x16x16 VAE")
        }
        return issues
    }
}

public struct Cosmos3ReasonerConfiguration: Codable, Hashable, Sendable {
    public struct Vision: Codable, Hashable, Sendable {
        public let attentionDropout: Float
        public let hiddenActivation: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let layerNormEpsilon: Float
        public let attentionHeadCount: Int
        public let channelCount: Int
        public let layerCount: Int
        public let patchCount: Int
        public let patchSize: Int
        public let spatialMergeSize: Int

        enum CodingKeys: String, CodingKey {
            case attentionDropout = "attention_dropout"
            case hiddenActivation = "hidden_act"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case layerNormEpsilon = "layer_norm_eps"
            case attentionHeadCount = "num_attention_heads"
            case channelCount = "num_channels"
            case layerCount = "num_hidden_layers"
            case patchCount = "num_patches"
            case patchSize = "patch_size"
            case spatialMergeSize = "spatial_merge_size"
        }
    }

    public struct Projector: Codable, Hashable, Sendable {
        public let inputHiddenSize: Int
        public let intermediateSize: Int
        public let outputHiddenSize: Int
        public let spatialMergeSize: Int
        public let usesPostShuffleNorm: Bool

        enum CodingKeys: String, CodingKey {
            case inputHiddenSize = "input_hidden_size"
            case intermediateSize = "merger_intermediate_size"
            case outputHiddenSize = "out_hidden_size"
            case spatialMergeSize = "spatial_merge_size"
            case usesPostShuffleNorm = "use_postshuffle_norm"
        }
    }

    public let vision: Vision
    public let projector: Projector
    public let imageTokenID: Int
    public let videoTokenID: Int
    public let visionStartTokenID: Int
    public let visionEndTokenID: Int

    enum CodingKeys: String, CodingKey {
        case vision = "vision_config"
        case projector = "projector_config"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case visionStartTokenID = "vision_start_token_id"
        case visionEndTokenID = "vision_end_token_id"
    }

    public func validationIssues() -> [String] {
        var issues: [String] = []
        if vision.hiddenSize % vision.attentionHeadCount != 0 {
            issues.append("vision hidden_size must be divisible by num_attention_heads")
        }
        if Int(Double(vision.patchCount).squareRoot()).squared != vision.patchCount {
            issues.append("vision num_patches must describe a square position grid")
        }
        if vision.hiddenActivation != "gelu_pytorch_tanh" {
            issues.append("Cosmos3-Edge vision requires gelu_pytorch_tanh")
        }
        if projector.inputHiddenSize != vision.hiddenSize {
            issues.append("projector input_hidden_size must equal vision hidden_size")
        }
        if projector.spatialMergeSize != vision.spatialMergeSize {
            issues.append("projector and vision spatial_merge_size must match")
        }
        return issues
    }
}

private extension Int {
    var squared: Int { self * self }
}

public struct Cosmos3Resources: Hashable, Sendable {
    public static let modelID = "video-cosmos3-edge-mlx"
    public static let officialRepoID = "nvidia/Cosmos3-Edge"
    public static let officialRevision = "6f58f6b4c91288838e60b6bcb2cc45d997e961de"
    public static let superTextToImageRepository = "nvidia/Cosmos3-Super-Text2Image-4Step"
    public static let superTextToImageRevision = "aa0d5a57b7b045d68daa60fbacd84ec723c7cb7b"
    public static let snapshotPatterns = [
        "LICENSE*",
        "README.md",
        "config.json",
        "model_index.json",
        "modular_model_index.json",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "generation_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "scheduler/*",
        "text_tokenizer/*",
        "transformer/*",
        "vae/*",
        "vision_encoder/*",
    ]

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var transformerRootURL: URL { rootURL.appendingPathComponent("transformer") }
    public var transformerConfigURL: URL { transformerRootURL.appendingPathComponent("config.json") }
    public var transformerIndexURL: URL {
        transformerRootURL.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
    }
    public var transformerURL: URL {
        transformerRootURL.appendingPathComponent("diffusion_pytorch_model.safetensors")
    }
    public var vaeRootURL: URL { rootURL.appendingPathComponent("vae") }
    public var vaeConfigURL: URL { vaeRootURL.appendingPathComponent("config.json") }
    public var vaeIndexURL: URL {
        vaeRootURL.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
    }
    public var vaeURL: URL { vaeRootURL.appendingPathComponent("diffusion_pytorch_model.safetensors") }
    public var tokenizerRootURL: URL { rootURL.appendingPathComponent("text_tokenizer") }
    public var reasonerConfigURL: URL { rootURL.appendingPathComponent("config.json") }
    public var reasonerIndexURL: URL {
        rootURL.appendingPathComponent("model.safetensors.index.json")
    }
    public var reasonerTokenizerConfigURL: URL {
        rootURL.appendingPathComponent("tokenizer_config.json")
    }
    public var reasonerTokenizerURL: URL {
        rootURL.appendingPathComponent("tokenizer.json")
    }
    public var reasonerChatTemplateURL: URL {
        rootURL.appendingPathComponent("chat_template.jinja")
    }
    public var processorConfigURL: URL {
        rootURL.appendingPathComponent("processor_config.json")
    }
    public var imageProcessorConfigURL: URL {
        rootURL.appendingPathComponent("preprocessor_config.json")
    }
    public var videoProcessorConfigURL: URL {
        rootURL.appendingPathComponent("video_preprocessor_config.json")
    }
    public var visionEncoderRootURL: URL {
        rootURL.appendingPathComponent("vision_encoder")
    }
    public var visionEncoderURL: URL {
        visionEncoderRootURL.appendingPathComponent("model.safetensors")
    }
    public var schedulerConfigURL: URL {
        rootURL.appendingPathComponent("scheduler").appendingPathComponent("scheduler_config.json")
    }
    public var modularModelIndexURL: URL {
        rootURL.appendingPathComponent("modular_model_index.json")
    }
    public var tokenizerConfigURL: URL {
        tokenizerRootURL.appendingPathComponent("tokenizer_config.json")
    }
    public var tokenizerURL: URL {
        tokenizerRootURL.appendingPathComponent("tokenizer.json")
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing = [
            transformerConfigURL,
            vaeConfigURL,
            schedulerConfigURL,
            tokenizerConfigURL,
            tokenizerURL,
        ].filter { !fileManager.fileExists(atPath: $0.path) }

        if !fileManager.fileExists(atPath: transformerURL.path) {
            missing.append(contentsOf: missingIndexOrShards(
                indexURL: transformerIndexURL,
                fileManager: fileManager
            ))
        }
        if !fileManager.fileExists(atPath: vaeURL.path) {
            missing.append(contentsOf: missingIndexOrShards(
                indexURL: vaeIndexURL,
                fileManager: fileManager
            ))
        }
        return missing
    }

    public func validateReasoner(fileManager: FileManager = .default) -> [URL] {
        [
            reasonerConfigURL,
            reasonerIndexURL,
            reasonerTokenizerConfigURL,
            reasonerTokenizerURL,
            reasonerChatTemplateURL,
            processorConfigURL,
            imageProcessorConfigURL,
            videoProcessorConfigURL,
            visionEncoderURL,
        ].filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public func loadTransformerConfiguration() throws -> Cosmos3TransformerConfiguration {
        try Self.loadConfiguration(
            Cosmos3TransformerConfiguration.self,
            from: transformerConfigURL,
            validate: { $0.validationIssues() }
        )
    }

    public func loadVAEConfiguration() throws -> Cosmos3VAEConfiguration {
        try Self.loadConfiguration(
            Cosmos3VAEConfiguration.self,
            from: vaeConfigURL,
            validate: { $0.validationIssues() }
        )
    }

    public func loadDistilledConfiguration() throws -> Cosmos3DistilledConfiguration? {
        guard FileManager.default.fileExists(atPath: modularModelIndexURL.path) else {
            return nil
        }
        let marker = try Self.loadConfiguration(
            Cosmos3DistilledMarker.self,
            from: modularModelIndexURL,
            validate: { _ in [] }
        )
        guard marker.isDistilled == true else {
            return nil
        }
        return try Self.loadConfiguration(
            Cosmos3DistilledConfiguration.self,
            from: modularModelIndexURL,
            validate: { $0.validationIssues() }
        )
    }

    public func loadReasonerConfiguration() throws -> Cosmos3ReasonerConfiguration {
        try Self.loadConfiguration(
            Cosmos3ReasonerConfiguration.self,
            from: reasonerConfigURL,
            validate: { $0.validationIssues() }
        )
    }

    private static func loadConfiguration<T: Decodable>(
        _ type: T.Type,
        from url: URL,
        validate: (T) -> [String]
    ) throws -> T {
        do {
            let value = try JSONDecoder().decode(type, from: Data(contentsOf: url))
            let issues = validate(value)
            guard issues.isEmpty else {
                throw Cosmos3ResourcesError.invalidConfiguration(url, issues.joined(separator: "; "))
            }
            return value
        } catch let error as Cosmos3ResourcesError {
            throw error
        } catch {
            throw Cosmos3ResourcesError.invalidConfiguration(url, error.localizedDescription)
        }
    }

    private func missingIndexOrShards(
        indexURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        guard fileManager.fileExists(atPath: indexURL.path),
              let data = try? Data(contentsOf: indexURL),
              let index = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: data),
              !index.shardFilenames.isEmpty else {
            return [indexURL]
        }
        let root = indexURL.deletingLastPathComponent()
        return index.shardFilenames
            .map { root.appendingPathComponent($0, isDirectory: false) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }
}
