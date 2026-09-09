import Foundation

public struct Qwen3TTSTokenizerEncoderConfig: Decodable, Sendable, Hashable {
    public let frameRate: Float
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let audioChannels: Int
    public let codebookDim: Int
    public let codebookSize: Int
    public let compress: Int
    public let dilationGrowthRate: Int
    public let headDim: Int
    public let hiddenAct: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let kernelSize: Int
    public let lastKernelSize: Int
    public let layerScaleInitialScale: Float
    public let maxPositionEmbeddings: Int
    public let normEps: Float
    public let numAttentionHeads: Int
    public let numFilters: Int
    public let numHiddenLayers: Int
    public let numKeyValueHeads: Int
    public let numQuantizers: Int
    public let numResidualLayers: Int
    public let numSemanticQuantizers: Int
    public let residualKernelSize: Int
    public let ropeTheta: Float
    public let samplingRate: Int
    public let slidingWindow: Int
    public let upsamplingRatios: [Int]
    public let useCausalConv: Bool
    public let useConvShortcut: Bool
    public let vectorQuantizationHiddenDimension: Int

    enum CodingKeys: String, CodingKey {
        case frameRate = "frame_rate"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case audioChannels = "audio_channels"
        case codebookDim = "codebook_dim"
        case codebookSize = "codebook_size"
        case compress
        case dilationGrowthRate = "dilation_growth_rate"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case kernelSize = "kernel_size"
        case lastKernelSize = "last_kernel_size"
        case layerScaleInitialScale = "layer_scale_initial_scale"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "norm_eps"
        case numAttentionHeads = "num_attention_heads"
        case numFilters = "num_filters"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case numQuantizers = "num_quantizers"
        case numResidualLayers = "num_residual_layers"
        case numSemanticQuantizers = "num_semantic_quantizers"
        case residualKernelSize = "residual_kernel_size"
        case ropeTheta = "rope_theta"
        case samplingRate = "sampling_rate"
        case slidingWindow = "sliding_window"
        case upsamplingRatios = "upsampling_ratios"
        case useCausalConv = "use_causal_conv"
        case useConvShortcut = "use_conv_shortcut"
        case vectorQuantizationHiddenDimension = "vector_quantization_hidden_dimension"
    }

    public init(
        frameRate: Float = 12.5,
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0,
        audioChannels: Int = 1,
        codebookDim: Int = 256,
        codebookSize: Int = 2048,
        compress: Int = 2,
        dilationGrowthRate: Int = 2,
        headDim: Int = 64,
        hiddenAct: String = "gelu",
        hiddenSize: Int = 512,
        intermediateSize: Int = 2048,
        kernelSize: Int = 7,
        lastKernelSize: Int = 3,
        layerScaleInitialScale: Float = 0.01,
        maxPositionEmbeddings: Int = 8000,
        normEps: Float = 1e-5,
        numAttentionHeads: Int = 8,
        numFilters: Int = 64,
        numHiddenLayers: Int = 8,
        numKeyValueHeads: Int = 8,
        numQuantizers: Int = 32,
        numResidualLayers: Int = 1,
        numSemanticQuantizers: Int = 1,
        residualKernelSize: Int = 3,
        ropeTheta: Float = 10000.0,
        samplingRate: Int = 24000,
        slidingWindow: Int = 250,
        upsamplingRatios: [Int] = [8, 6, 5, 4],
        useCausalConv: Bool = true,
        useConvShortcut: Bool = false,
        vectorQuantizationHiddenDimension: Int = 256
    ) {
        self.frameRate = frameRate
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.audioChannels = audioChannels
        self.codebookDim = codebookDim
        self.codebookSize = codebookSize
        self.compress = compress
        self.dilationGrowthRate = dilationGrowthRate
        self.headDim = headDim
        self.hiddenAct = hiddenAct
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.kernelSize = kernelSize
        self.lastKernelSize = lastKernelSize
        self.layerScaleInitialScale = layerScaleInitialScale
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.normEps = normEps
        self.numAttentionHeads = numAttentionHeads
        self.numFilters = numFilters
        self.numHiddenLayers = numHiddenLayers
        self.numKeyValueHeads = numKeyValueHeads
        self.numQuantizers = numQuantizers
        self.numResidualLayers = numResidualLayers
        self.numSemanticQuantizers = numSemanticQuantizers
        self.residualKernelSize = residualKernelSize
        self.ropeTheta = ropeTheta
        self.samplingRate = samplingRate
        self.slidingWindow = slidingWindow
        self.upsamplingRatios = upsamplingRatios
        self.useCausalConv = useCausalConv
        self.useConvShortcut = useConvShortcut
        self.vectorQuantizationHiddenDimension = vectorQuantizationHiddenDimension
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        frameRate = try container.decodeIfPresent(Float.self, forKey: .frameRate) ?? 12.5
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        audioChannels = try container.decodeIfPresent(Int.self, forKey: .audioChannels) ?? 1
        codebookDim = try container.decodeIfPresent(Int.self, forKey: .codebookDim) ?? 256
        codebookSize = try container.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 2048
        compress = try container.decodeIfPresent(Int.self, forKey: .compress) ?? 2
        dilationGrowthRate = try container.decodeIfPresent(Int.self, forKey: .dilationGrowthRate) ?? 2
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 512
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2048
        kernelSize = try container.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 7
        lastKernelSize = try container.decodeIfPresent(Int.self, forKey: .lastKernelSize) ?? 3
        layerScaleInitialScale = try container.decodeIfPresent(Float.self, forKey: .layerScaleInitialScale) ?? 0.01
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8000
        normEps = try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        numFilters = try container.decodeIfPresent(Int.self, forKey: .numFilters) ?? 64
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 8
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        numQuantizers = try container.decodeIfPresent(Int.self, forKey: .numQuantizers) ?? 32
        numResidualLayers = try container.decodeIfPresent(Int.self, forKey: .numResidualLayers) ?? 1
        numSemanticQuantizers = try container.decodeIfPresent(Int.self, forKey: .numSemanticQuantizers) ?? 1
        residualKernelSize = try container.decodeIfPresent(Int.self, forKey: .residualKernelSize) ?? 3
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        samplingRate = try container.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24000
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 250
        upsamplingRatios = try container.decodeIfPresent([Int].self, forKey: .upsamplingRatios) ?? [8, 6, 5, 4]
        useCausalConv = try container.decodeIfPresent(Bool.self, forKey: .useCausalConv) ?? true
        useConvShortcut = try container.decodeIfPresent(Bool.self, forKey: .useConvShortcut) ?? false
        vectorQuantizationHiddenDimension = try container.decodeIfPresent(Int.self, forKey: .vectorQuantizationHiddenDimension) ?? 256
    }
}
