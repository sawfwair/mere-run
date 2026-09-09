import Foundation

public struct Qwen3TTSTokenizerDecoderConfig: Decodable, Sendable, Hashable {
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let latentDim: Int
    public let codebookDim: Int
    public let codebookSize: Int
    public let decoderDim: Int
    public let hiddenAct: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let layerScaleInitialScale: Float
    public let maxPositionEmbeddings: Int
    public let headDim: Int
    public let numAttentionHeads: Int
    public let numHiddenLayers: Int
    public let numKeyValueHeads: Int
    public let numQuantizers: Int
    public let numSemanticQuantizers: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let semanticCodebookSize: Int
    public let slidingWindow: Int
    public let upsampleRates: [Int]
    public let upsamplingRatios: [Int]
    public let vectorQuantizationHiddenDimension: Int

    enum CodingKeys: String, CodingKey {
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case latentDim = "latent_dim"
        case codebookDim = "codebook_dim"
        case codebookSize = "codebook_size"
        case decoderDim = "decoder_dim"
        case hiddenAct = "hidden_act"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case layerScaleInitialScale = "layer_scale_initial_scale"
        case maxPositionEmbeddings = "max_position_embeddings"
        case headDim = "head_dim"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case numQuantizers = "num_quantizers"
        case numSemanticQuantizers = "num_semantic_quantizers"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case semanticCodebookSize = "semantic_codebook_size"
        case slidingWindow = "sliding_window"
        case upsampleRates = "upsample_rates"
        case upsamplingRatios = "upsampling_ratios"
        case vectorQuantizationHiddenDimension = "vector_quantization_hidden_dimension"
    }

    public init(
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0,
        latentDim: Int = 1024,
        codebookDim: Int = 512,
        codebookSize: Int = 2048,
        decoderDim: Int = 1536,
        hiddenAct: String = "silu",
        hiddenSize: Int = 512,
        intermediateSize: Int = 1024,
        layerScaleInitialScale: Float = 0.01,
        maxPositionEmbeddings: Int = 8000,
        headDim: Int = 64,
        numAttentionHeads: Int = 16,
        numHiddenLayers: Int = 8,
        numKeyValueHeads: Int = 16,
        numQuantizers: Int = 16,
        numSemanticQuantizers: Int = 1,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10000.0,
        semanticCodebookSize: Int = 4096,
        slidingWindow: Int = 72,
        upsampleRates: [Int] = [8, 5, 4, 3],
        upsamplingRatios: [Int] = [2, 2],
        vectorQuantizationHiddenDimension: Int = 512
    ) {
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.latentDim = latentDim
        self.codebookDim = codebookDim
        self.codebookSize = codebookSize
        self.decoderDim = decoderDim
        self.hiddenAct = hiddenAct
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.layerScaleInitialScale = layerScaleInitialScale
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.headDim = headDim
        self.numAttentionHeads = numAttentionHeads
        self.numHiddenLayers = numHiddenLayers
        self.numKeyValueHeads = numKeyValueHeads
        self.numQuantizers = numQuantizers
        self.numSemanticQuantizers = numSemanticQuantizers
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.semanticCodebookSize = semanticCodebookSize
        self.slidingWindow = slidingWindow
        self.upsampleRates = upsampleRates
        self.upsamplingRatios = upsamplingRatios
        self.vectorQuantizationHiddenDimension = vectorQuantizationHiddenDimension
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        latentDim = try container.decodeIfPresent(Int.self, forKey: .latentDim) ?? 1024
        codebookDim = try container.decodeIfPresent(Int.self, forKey: .codebookDim) ?? 512
        codebookSize = try container.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 2048
        decoderDim = try container.decodeIfPresent(Int.self, forKey: .decoderDim) ?? 1536
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 512
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 1024
        layerScaleInitialScale = try container.decodeIfPresent(Float.self, forKey: .layerScaleInitialScale) ?? 0.01
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8000
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 8
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 16
        numQuantizers = try container.decodeIfPresent(Int.self, forKey: .numQuantizers) ?? 16
        numSemanticQuantizers = try container.decodeIfPresent(Int.self, forKey: .numSemanticQuantizers) ?? 1
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        semanticCodebookSize = try container.decodeIfPresent(Int.self, forKey: .semanticCodebookSize) ?? 4096
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 72
        upsampleRates = try container.decodeIfPresent([Int].self, forKey: .upsampleRates) ?? [8, 5, 4, 3]
        upsamplingRatios = try container.decodeIfPresent([Int].self, forKey: .upsamplingRatios) ?? [2, 2]
        vectorQuantizationHiddenDimension = try container.decodeIfPresent(Int.self, forKey: .vectorQuantizationHiddenDimension) ?? 512
    }
}
