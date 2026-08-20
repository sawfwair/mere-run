import Foundation

public struct NemotronOmniVisionConfig: Decodable, Sendable, Hashable {
    public let version: String
    public let patchSize: Int
    public let minNumPatches: Int
    public let maxNumPatches: Int
    public let videoTargetNumPatches: Int
    public let videoTemporalPatchSize: Int
    public let separateVideoEmbedder: Bool

    private enum CodingKeys: String, CodingKey {
        case version
        case patchSize = "patch_size"
        case minNumPatches = "min_num_patches"
        case maxNumPatches = "max_num_patches"
        case videoTargetNumPatches = "video_target_num_patches"
        case videoTemporalPatchSize = "video_temporal_patch_size"
        case separateVideoEmbedder = "separate_video_embedder"
    }
}

public struct NemotronOmniSoundConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let hiddenSize: Int
    public let numAttentionHeads: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let convKernelSize: Int
    public let subsamplingFactor: Int
    public let subsamplingConvChannels: Int
    public let subsamplingConvKernelSize: Int
    public let subsamplingConvStride: Int
    public let numMelBins: Int
    public let projectionHiddenSize: Int
    public let samplingRate: Int

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case convKernelSize = "conv_kernel_size"
        case subsamplingFactor = "subsampling_factor"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case subsamplingConvKernelSize = "subsampling_conv_kernel_size"
        case subsamplingConvStride = "subsampling_conv_stride"
        case numMelBins = "num_mel_bins"
        case projectionHiddenSize = "projection_hidden_size"
        case samplingRate = "sampling_rate"
    }
}

public struct NemotronOmniLanguageConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let hiddenSize: Int
    public let vocabSize: Int
    public let numHiddenLayers: Int
    public let hybridOverridePattern: String
    public let maxPositionEmbeddings: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let normEps: Float
    public let mambaHeadDim: Int
    public let mambaNumHeads: Int
    public let ssmStateSize: Int
    public let nGroups: Int
    public let convKernel: Int
    public let timeStepMin: Float
    public let timeStepMax: Float
    public let nRoutedExperts: Int
    public let nSharedExperts: Int
    public let numExpertsPerToken: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let routedScalingFactor: Float
    public let normTopKProbability: Bool
    public let nGroup: Int
    public let topKGroup: Int
    public let eosTokenID: Int

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case vocabSize = "vocab_size"
        case numHiddenLayers = "num_hidden_layers"
        case hybridOverridePattern = "hybrid_override_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case normEps = "norm_eps"
        case mambaHeadDim = "mamba_head_dim"
        case mambaNumHeads = "mamba_num_heads"
        case ssmStateSize = "ssm_state_size"
        case nGroups = "n_groups"
        case convKernel = "conv_kernel"
        case timeStepMin = "time_step_min"
        case timeStepMax = "time_step_max"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopKProbability = "norm_topk_prob"
        case nGroup = "n_group"
        case topKGroup = "topk_group"
        case eosTokenID = "eos_token_id"
    }

    public var layerBlockTypes: [String] {
        hybridOverridePattern.map { value in
            switch value {
            case "M": "mamba"
            case "E": "moe"
            case "*": "attention"
            default: "unsupported"
            }
        }
    }

    var runtimeConfig: NemotronHConfig {
        NemotronHConfig(
            modelType: modelType,
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            numHiddenLayers: numHiddenLayers,
            layersBlockType: layerBlockTypes,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: headDim,
            maxPositionEmbeddings: maxPositionEmbeddings,
            normEps: normEps,
            mambaHeadDim: mambaHeadDim,
            mambaNumHeads: mambaNumHeads,
            ssmStateSize: ssmStateSize,
            nGroups: nGroups,
            convKernel: convKernel,
            timeStepMin: timeStepMin,
            timeStepMax: timeStepMax,
            nRoutedExperts: nRoutedExperts,
            nSharedExperts: nSharedExperts,
            numExpertsPerToken: numExpertsPerToken,
            moeIntermediateSize: moeIntermediateSize,
            sharedExpertIntermediateSize: sharedExpertIntermediateSize,
            routedScalingFactor: routedScalingFactor,
            normTopKProbability: normTopKProbability,
            nGroup: nGroup,
            topKGroup: topKGroup,
            eosTokenIDs: [eosTokenID],
            quantization: NemotronHQuantizationConfig(
                bits: 4,
                groupSize: 16,
                mode: "nvfp4",
                globalScale: true
            )
        )
    }
}

public struct NemotronOmniConfig: Decodable, Sendable, Hashable {
    public let architectures: [String]
    public let modelType: String
    public let maximumSequenceLength: Int
    public let imageContextTokenID: Int
    public let videoContextTokenID: Int
    public let soundContextTokenID: Int
    public let downsampleRatio: Double
    public let projectorHiddenSize: Int
    public let visionHiddenSize: Int
    public let videoPruningRate: Double
    public let vision: NemotronOmniVisionConfig
    public let sound: NemotronOmniSoundConfig
    public let language: NemotronOmniLanguageConfig

    private enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case maximumSequenceLength = "max_sequence_length"
        case imageContextTokenID = "img_context_token_id"
        case videoContextTokenID = "video_context_token_id"
        case soundContextTokenID = "sound_context_token_id"
        case downsampleRatio = "downsample_ratio"
        case projectorHiddenSize = "projector_hidden_size"
        case visionHiddenSize = "vit_hidden_size"
        case videoPruningRate = "video_pruning_rate"
        case vision = "vision_config"
        case sound = "sound_config"
        case language = "llm_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        architectures = try container.decode([String].self, forKey: .architectures)
        modelType = try container.decode(String.self, forKey: .modelType)
        maximumSequenceLength = try container.decode(Int.self, forKey: .maximumSequenceLength)
        imageContextTokenID = try container.decode(Int.self, forKey: .imageContextTokenID)
        videoContextTokenID = try container.decode(Int.self, forKey: .videoContextTokenID)
        soundContextTokenID = try container.decode(Int.self, forKey: .soundContextTokenID)
        downsampleRatio = try container.decode(Double.self, forKey: .downsampleRatio)
        projectorHiddenSize = try container.decode(Int.self, forKey: .projectorHiddenSize)
        visionHiddenSize = try container.decode(Int.self, forKey: .visionHiddenSize)
        videoPruningRate = try container.decode(Double.self, forKey: .videoPruningRate)
        vision = try container.decode(NemotronOmniVisionConfig.self, forKey: .vision)
        sound = try container.decode(NemotronOmniSoundConfig.self, forKey: .sound)
        language = try container.decode(NemotronOmniLanguageConfig.self, forKey: .language)

        let supportedLayerTypes = Set(language.layerBlockTypes)
        guard architectures == ["NemotronH_Nano_Omni_Reasoning_V3"],
              modelType == "NemotronH_Nano_Omni_Reasoning_V3",
              maximumSequenceLength == NemotronOmniResources.maximumContextLength,
              imageContextTokenID == 18,
              videoContextTokenID == 131_081,
              soundContextTokenID == 27,
              downsampleRatio == 0.5,
              projectorHiddenSize == 20_480,
              visionHiddenSize == 1_280,
              videoPruningRate == 0.7,
              language.modelType == "nemotron_h",
              language.hiddenSize == 2_688,
              language.vocabSize == 131_072,
              language.numHiddenLayers == 52,
              language.maxPositionEmbeddings == 262_144,
              language.layerBlockTypes.count == language.numHiddenLayers,
              supportedLayerTypes.isSubset(of: ["mamba", "moe", "attention"]),
              vision.version == "c-radio_v4-h",
              vision.patchSize == 16,
              vision.minNumPatches == 1_024,
              vision.maxNumPatches == 13_312,
              vision.videoTargetNumPatches == 1_024,
              vision.videoTemporalPatchSize == 2,
              vision.separateVideoEmbedder,
              sound.modelType == "parakeet",
              sound.hiddenSize == 1_024,
              sound.numAttentionHeads == 8,
              sound.numHiddenLayers == 24,
              sound.intermediateSize == 4_096,
              sound.convKernelSize == 9,
              sound.subsamplingFactor == 8,
              sound.subsamplingConvChannels == 256,
              sound.subsamplingConvKernelSize == 3,
              sound.subsamplingConvStride == 2,
              sound.numMelBins == 128,
              sound.projectionHiddenSize == 4_096,
              sound.samplingRate == NemotronOmniResources.audioSampleRate else {
            throw DecodingError.dataCorruptedError(
                forKey: .modelType,
                in: container,
                debugDescription: "Unsupported Nemotron 3 Nano Omni checkpoint contract."
            )
        }
    }
}

public struct NemotronOmniPreprocessorConfig: Decodable, Sendable, Hashable {
    public let patchSize: Int
    public let downsampleRatio: Double
    public let normalizationMean: [Float]
    public let normalizationStandardDeviation: [Float]
    public let minNumPatches: Int
    public let maxNumPatches: Int

    private enum CodingKeys: String, CodingKey {
        case patchSize = "patch_size"
        case downsampleRatio = "downsample_ratio"
        case normalizationMean = "norm_mean"
        case normalizationStandardDeviation = "norm_std"
        case minNumPatches = "min_num_patches"
        case maxNumPatches = "max_num_patches"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        patchSize = try container.decode(Int.self, forKey: .patchSize)
        downsampleRatio = try container.decode(Double.self, forKey: .downsampleRatio)
        normalizationMean = try container.decode([Float].self, forKey: .normalizationMean)
        normalizationStandardDeviation = try container.decode(
            [Float].self,
            forKey: .normalizationStandardDeviation
        )
        minNumPatches = try container.decode(Int.self, forKey: .minNumPatches)
        maxNumPatches = try container.decode(Int.self, forKey: .maxNumPatches)

        guard patchSize == 16,
              downsampleRatio == 0.5,
              normalizationMean == [0.48145466, 0.4578275, 0.40821073],
              normalizationStandardDeviation == [0.26862954, 0.26130258, 0.27577711],
              minNumPatches == 1_024,
              maxNumPatches == 13_312 else {
            throw DecodingError.dataCorruptedError(
                forKey: .patchSize,
                in: container,
                debugDescription: "Unsupported Nemotron 3 Nano Omni media processor contract."
            )
        }
    }
}

public enum NemotronOmniPlaceholderPlanner {
    /// C-RADIO patch tokens are pixel-shuffled by a 0.5 downsample ratio, so
    /// every four source patches become one language-model placeholder.
    public static func imageTokenCount(sourcePatchCount: Int, downsampleRatio: Double = 0.5) -> Int {
        precondition(sourcePatchCount > 0)
        let divisor = Int((1 / downsampleRatio) * (1 / downsampleRatio))
        return sourcePatchCount / divisor
    }

    /// Audio uses a three-stage stride-2 convolutional subsampler after the
    /// 10 ms mel hop. This mirrors the pinned upstream processor exactly.
    public static func audioTokenCount(
        sampleCount: Int,
        hopLength: Int = 160,
        kernelSize: Int = 3,
        stride: Int = 2,
        subsamplingFactor: Int = 8
    ) -> Int {
        precondition(sampleCount >= 0)
        var frames = 1 + sampleCount / hopLength
        var remainingFactor = subsamplingFactor
        let padding = (kernelSize - 1) / 2
        while remainingFactor > 1 {
            frames = max(0, (frames + 2 * padding - kernelSize) / stride + 1)
            remainingFactor /= stride
        }
        return max(1, frames)
    }
}
