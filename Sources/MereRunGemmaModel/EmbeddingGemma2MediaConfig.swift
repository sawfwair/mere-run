import Foundation

public struct EmbeddingGemma2VisionConfig: Decodable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let positionEmbeddingSize: Int
    public let rmsNormEps: Float
    public let standardize: Bool
    public let useClippedLinears: Bool
    public let hiddenActivation: String
    public let ropeParameters: Rope
    public struct Rope: Decodable, Sendable {
        public let ropeTheta: Float
        public let ropeType: String
        enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta", ropeType = "rope_type" }
    }
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", hiddenSize = "hidden_size", intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers", numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads", headDim = "head_dim", patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size", positionEmbeddingSize = "position_embedding_size"
        case rmsNormEps = "rms_norm_eps", standardize, useClippedLinears = "use_clipped_linears"
        case hiddenActivation = "hidden_activation", ropeParameters = "rope_parameters"
    }
    public func validate() throws {
        guard modelType == "gemma4_vision", hiddenActivation == "gelu_pytorch_tanh", !standardize,
              !useClippedLinears, ropeParameters.ropeType == "axial", ropeParameters.ropeTheta > 0,
              [hiddenSize, intermediateSize, numHiddenLayers, numAttentionHeads, numKeyValueHeads,
               headDim, patchSize, poolingKernelSize, positionEmbeddingSize].allSatisfy({ $0 > 0 }),
              headDim.isMultiple(of: 4), numAttentionHeads.isMultiple(of: numKeyValueHeads),
              rmsNormEps.isFinite, rmsNormEps > 0 else {
            throw EmbeddingGemma2Error.invalidConfiguration("Unsupported EmbeddingGemma 2 vision configuration.")
        }
    }
}

public struct EmbeddingGemma2AudioConfig: Decodable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let outputProjDims: Int
    public let subsamplingConvChannels: [Int]
    public let convKernelSize: Int
    public let rmsNormEps: Float
    public let residualWeight: Float
    public let gradientClipping: Float
    public let attentionChunkSize: Int
    public let attentionContextLeft: Int
    public let attentionContextRight: Int
    public let attentionLogitCap: Float
    public let attentionInvalidLogitsValue: Float
    public let useClippedLinears: Bool
    public let hiddenAct: String
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", hiddenSize = "hidden_size", numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads", outputProjDims = "output_proj_dims"
        case subsamplingConvChannels = "subsampling_conv_channels", convKernelSize = "conv_kernel_size"
        case rmsNormEps = "rms_norm_eps", residualWeight = "residual_weight", gradientClipping = "gradient_clipping"
        case attentionChunkSize = "attention_chunk_size", attentionContextLeft = "attention_context_left"
        case attentionContextRight = "attention_context_right", attentionLogitCap = "attention_logit_cap"
        case attentionInvalidLogitsValue = "attention_invalid_logits_value", useClippedLinears = "use_clipped_linears"
        case hiddenAct = "hidden_act"
    }
    public func validate() throws {
        guard modelType == "gemma4_audio", hiddenAct == "silu", useClippedLinears,
              [hiddenSize, numHiddenLayers, numAttentionHeads, outputProjDims, convKernelSize,
               attentionChunkSize, attentionContextLeft].allSatisfy({ $0 > 0 }),
              hiddenSize.isMultiple(of: numAttentionHeads), hiddenSize.isMultiple(of: 2),
              subsamplingConvChannels.count == 2, subsamplingConvChannels.allSatisfy({ $0 > 0 }),
              subsamplingConvChannels[0] == 128, attentionContextRight == 0,
              attentionContextLeft == attentionChunkSize + 1, rmsNormEps.isFinite, rmsNormEps > 0,
              gradientClipping.isFinite, gradientClipping > 0, attentionLogitCap.isFinite, attentionLogitCap > 0,
              residualWeight.isFinite, attentionInvalidLogitsValue.isFinite, attentionInvalidLogitsValue < 0 else {
            throw EmbeddingGemma2Error.invalidConfiguration("Unsupported EmbeddingGemma 2 audio configuration.")
        }
    }
}
