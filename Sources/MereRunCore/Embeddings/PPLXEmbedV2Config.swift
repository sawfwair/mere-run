import Foundation
import MereRunQwenModel

public struct PPLXEmbedV2Config: Decodable {
    public let backbone: Q35Config
    public let quantization: PPLXEmbedV2QuantizationConfig?
    public let embeddingDim: Int
    public let queryPrefix: String
    public let documentPrefix: String
    public let boundaryMarker: String
    public let queryLength: Int
    public let documentLength: Int
    public var isContextual: Bool { backbone.modelType == "pplx_contextual_qwen3_5" }

    private enum CodingKeys: String, CodingKey {
        case textConfig = "text_config", embeddingDim = "embedding_dim", quantization
        case queryPrefix = "query_prefix", documentPrefix = "document_prefix", boundaryMarker = "boundary_marker"
        case queryLength = "query_length", documentLength = "document_length"
    }
    private struct TextSemantics: Decodable {
        let isCausal: Bool
        let hiddenAct: String
        enum CodingKeys: String, CodingKey { case isCausal = "is_causal", hiddenAct = "hidden_act" }
    }

    public init(from decoder: Decoder) throws {
        backbone = try Q35Config(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        quantization = try container.decodeIfPresent(PPLXEmbedV2QuantizationConfig.self, forKey: .quantization)
        try quantization?.validate()
        let semantics = try container.decode(TextSemantics.self, forKey: .textConfig)
        guard ["qwen3_5", "pplx_contextual_qwen3_5"].contains(backbone.modelType), !semantics.isCausal,
              semantics.hiddenAct == "silu", (backbone.quantization == nil) == (quantization == nil) else {
            throw PPLXEmbedV2Error.invalidConfiguration("PPLX Embed v2 requires bidirectional Qwen3.5 and explicit native quantization metadata when packed.")
        }
        let text = backbone.textConfig
        guard text.hiddenSize > 0, text.intermediateSize > 0, text.vocabSize > 0,
              text.numHiddenLayers > 0, text.layerTypes.count == text.numHiddenLayers,
              text.layerTypes.allSatisfy({ ["linear_attention", "full_attention"].contains($0) }),
              text.numAttentionHeads > 0, text.numKeyValueHeads > 0,
              text.numAttentionHeads.isMultiple(of: text.numKeyValueHeads), text.headDim > 0,
              text.linearNumKeyHeads > 0, text.linearNumValueHeads > 0,
              text.linearNumValueHeads.isMultiple(of: text.linearNumKeyHeads),
              text.linearKeyHeadDim > 0, text.linearValueHeadDim > 0, text.linearConvKernelDim > 1,
              text.maxPositionEmbeddings > 0, text.rmsNormEps > 0,
              !text.usesMoE, text.mlpOnlyLayers.isEmpty, !text.attentionBias, text.attnOutputGate,
              text.ropeParameters.ropeTheta > 0, text.ropeParameters.ropeType == "default" else {
            throw PPLXEmbedV2Error.invalidConfiguration("Unsupported PPLX Embed v2 transformer geometry.")
        }
        let rotaryDimensions = Int(Float(text.headDim) * text.ropeParameters.partialRotaryFactor)
        guard rotaryDimensions > 0, rotaryDimensions <= text.headDim, rotaryDimensions.isMultiple(of: 2) else {
            throw PPLXEmbedV2Error.invalidConfiguration("Invalid PPLX Embed v2 rotary dimensions.")
        }
        embeddingDim = try container.decodeIfPresent(Int.self, forKey: .embeddingDim) ?? 128
        queryPrefix = try container.decodeIfPresent(String.self, forKey: .queryPrefix) ?? "[Q] "
        documentPrefix = try container.decodeIfPresent(String.self, forKey: .documentPrefix) ?? "[D] "
        boundaryMarker = try container.decodeIfPresent(String.self, forKey: .boundaryMarker) ?? "<|chunk_sep|>"
        queryLength = try container.decodeIfPresent(Int.self, forKey: .queryLength) ?? 1_024
        documentLength = try container.decodeIfPresent(Int.self, forKey: .documentLength) ?? 4_096
        guard embeddingDim == (backbone.modelType == "qwen3_5" ? 128 : 2_048),
              queryLength > 0, documentLength > 0, queryLength <= text.maxPositionEmbeddings,
              documentLength <= text.maxPositionEmbeddings,
              queryPrefix == "[Q] ", documentPrefix == "[D] ", boundaryMarker == "<|chunk_sep|>" else {
            throw PPLXEmbedV2Error.invalidConfiguration("Unsupported PPLX Embed v2 projection or input contract.")
        }
    }
}
