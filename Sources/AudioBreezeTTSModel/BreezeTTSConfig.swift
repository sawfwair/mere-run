// Adapted from Blaizzy/mlx-audio-swift at 01dec7c9bdce3088a6b6b7ab9f2e403458195efb.
// Copyright (c) 2025 Prince Canuma. Licensed under MIT; see THIRD_PARTY_NOTICES.md.
import Foundation

struct BreezeRopeScaling: Decodable, Sendable {
    let factor: Float
    let highFreqFactor: Float?
    let lowFreqFactor: Float?
    let originalMaxPositionEmbeddings: Int?
    let ropeType: String

    enum CodingKeys: String, CodingKey {
        case factor
        case highFreqFactor = "high_freq_factor"
        case lowFreqFactor = "low_freq_factor"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case ropeType = "rope_type"
    }
}

public struct BreezeBackboneConfig: Decodable, Sendable {
    var modelType: String
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var ropeScaling: BreezeRopeScaling?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        ropeScaling = try c.decodeIfPresent(BreezeRopeScaling.self, forKey: .ropeScaling)
    }
}

public struct BreezeDepthDecoderConfig: Decodable, Sendable {
    var vocabSize: Int
    var numCodebooks: Int
    var audioEmbedSize: Int
    var backboneHiddenSize: Int
    var hiddenSize: Int
    var numHiddenLayers: Int
    var intermediateSize: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var ropeScaling: BreezeRopeScaling?

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case numCodebooks = "num_codebooks"
        case audioEmbedSize = "audio_embed_size"
        case backboneHiddenSize = "backbone_hidden_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        numCodebooks = try c.decode(Int.self, forKey: .numCodebooks)
        audioEmbedSize = try c.decode(Int.self, forKey: .audioEmbedSize)
        backboneHiddenSize = try c.decode(Int.self, forKey: .backboneHiddenSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        ropeScaling = try c.decodeIfPresent(BreezeRopeScaling.self, forKey: .ropeScaling)
    }
}

public struct BreezeTextEncoderConfig: Decodable, Sendable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var queryPreAttentionScalar: Float
    var maxPositionEmbeddings: Int
    var slidingWindow: Int
    var eoiTokenIndex: Int
    var layerTypes: [String]

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case queryPreAttentionScalar = "query_pre_attn_scalar"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case eoiTokenIndex = "eoi_token_index"
        case layerTypes = "layer_types"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        queryPreAttentionScalar = try c.decode(Float.self, forKey: .queryPreAttentionScalar)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        eoiTokenIndex = try c.decode(Int.self, forKey: .eoiTokenIndex)
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
    }
}

public struct BreezeCodecConfig: Decodable, Sendable {
    var samplingRate: Int
    var codebookSize: Int

    enum CodingKeys: String, CodingKey {
        case samplingRate = "sampling_rate"
        case codebookSize = "codebook_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        samplingRate = try c.decode(Int.self, forKey: .samplingRate)
        codebookSize = try c.decode(Int.self, forKey: .codebookSize)
    }
}

public struct BreezeTTSConfig: Decodable, Sendable {
    var modelType: String
    var audioNumCodebooks: Int
    var audioVocabSize: Int
    var audioEmbedSize: Int
    var textVocabSize: Int
    var audioTokenID: Int
    var audioEOSTokenID: Int
    var codebookPadTokenID: Int
    var codebookEOSTokenID: Int
    var tieCodebooksEmbeddings: Bool
    var backboneConfig: BreezeBackboneConfig
    var codecConfig: BreezeCodecConfig
    var depthDecoderConfig: BreezeDepthDecoderConfig
    var textEncoderConfig: BreezeTextEncoderConfig
    var ropeScaling: BreezeRopeScaling?

    var numCodebooks: Int { audioNumCodebooks }
    var codecVocabSize: Int { codecConfig.codebookSize }
    var sampleRate: Int { codecConfig.samplingRate }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioNumCodebooks = "audio_num_codebooks"
        case audioVocabSize = "audio_vocab_size"
        case audioEmbedSize = "audio_embed_size"
        case textVocabSize = "text_vocab_size"
        case audioTokenID = "audio_token_id"
        case audioEOSTokenID = "audio_eos_token_id"
        case codebookPadTokenID = "codebook_pad_token_id"
        case codebookEOSTokenID = "codebook_eos_token_id"
        case tieCodebooksEmbeddings = "tie_codebooks_embeddings"
        case backboneConfig = "backbone_config"
        case codecConfig = "codec_config"
        case depthDecoderConfig = "depth_decoder_config"
        case textEncoderConfig = "text_encoder_config"
        case ropeScaling = "rope_scaling"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        audioNumCodebooks = try c.decode(Int.self, forKey: .audioNumCodebooks)
        audioVocabSize = try c.decode(Int.self, forKey: .audioVocabSize)
        backboneConfig = try c.decode(BreezeBackboneConfig.self, forKey: .backboneConfig)
        codecConfig = try c.decode(BreezeCodecConfig.self, forKey: .codecConfig)
        depthDecoderConfig = try c.decode(BreezeDepthDecoderConfig.self, forKey: .depthDecoderConfig)
        textEncoderConfig = try c.decode(BreezeTextEncoderConfig.self, forKey: .textEncoderConfig)
        audioEmbedSize = try c.decode(Int.self, forKey: .audioEmbedSize)
        textVocabSize = try c.decode(Int.self, forKey: .textVocabSize)
        audioTokenID = try c.decode(Int.self, forKey: .audioTokenID)
        audioEOSTokenID = try c.decode(Int.self, forKey: .audioEOSTokenID)
        codebookPadTokenID = try c.decode(Int.self, forKey: .codebookPadTokenID)
        codebookEOSTokenID = try c.decode(Int.self, forKey: .codebookEOSTokenID)
        tieCodebooksEmbeddings = try c.decode(Bool.self, forKey: .tieCodebooksEmbeddings)

        ropeScaling = try c.decodeIfPresent(BreezeRopeScaling.self, forKey: .ropeScaling)
    }
}
