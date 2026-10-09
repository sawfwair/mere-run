// Native Swift/MLX reimplementation of the pinned LiquidAI D1 references.
// Modified for mere.run; see THIRD_PARTY_NOTICES.md and licenses/d1-LFM-OPEN-LICENSE.txt.
import Foundation

public enum D1Error: LocalizedError {
    case invalid(String)
    public var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}

public struct D1TextConfiguration: Decodable, Sendable {
    public let vocab_size: Int
    public let hidden_size: Int
    public let intermediate_size: Int
    public let num_hidden_layers: Int
    public let num_attention_heads: Int
    public let num_key_value_heads: Int
    public let layer_types: [String]
    public let norm_eps: Float
    public let conv_L_cache: Int
    public let max_position_embeddings: Int
    public let rope_theta: Float?
    public let rope_parameters: Rope?
    public let block_auto_adjust_ff_dim: Bool?
    public let block_ffn_dim_multiplier: Double?
    public let block_multiple_of: Int?
    public struct Rope: Decodable, Sendable { public let rope_theta: Float }
    var headDim: Int { hidden_size / num_attention_heads }
    var theta: Float { rope_parameters?.rope_theta ?? rope_theta ?? 1_000_000 }
    func ffnWidth(omni: Bool) -> Int {
        guard omni || block_auto_adjust_ff_dim == true else { return intermediate_size }
        let adjusted = Int(Double(2 * intermediate_size / 3) * (block_ffn_dim_multiplier ?? 1))
        let multiple = block_multiple_of ?? 256
        return multiple * ((adjusted + multiple - 1) / multiple)
    }
    func validate() throws {
        guard vocab_size > 0, hidden_size > 0, intermediate_size > 0, num_hidden_layers > 0,
              num_attention_heads > 0, num_key_value_heads > 0,
              hidden_size % num_attention_heads == 0, num_attention_heads % num_key_value_heads == 0,
              headDim % 2 == 0, layer_types.count == num_hidden_layers,
              layer_types.allSatisfy({ ["conv", "full_attention"].contains($0) }),
              conv_L_cache == 3, norm_eps > 0, theta > 0, max_position_embeddings > 0,
              (block_multiple_of ?? 256) > 0, (block_ffn_dim_multiplier ?? 1) > 0 else {
            throw D1Error.invalid("Unsupported D1 text configuration.")
        }
    }
}

public struct D1Configuration: Decodable, Sendable {
    public let model_type: String
    public let auto_map: [String: String]?
    public let text_config: D1TextConfiguration
    public let vision_config: D1VLVisionConfig
    public let audio_config: D1AudioConfiguration?
    public let head_layers: Int?
    public let max_length: Int?
    public let image_text_length: Int?
    public let audio_text_length: Int?
    public let temperatures: [String: Float]?
    public let projector_hidden_size: Int
    public let downsample_factor: Int?
    public let projector_bias: Bool?
    public let projector_hidden_act: String?
    public let projector_use_layernorm: Bool?
    public let bos_token_id: Int
    public let image_token_id: Int?
    public var isOmni: Bool { model_type == "d1_omni" }
    public var maxLength: Int { isOmni ? max_length ?? 16_384 : text_config.max_position_embeddings }
    public func validate() throws {
        try text_config.validate()
        guard isOmni || (model_type == "lfm2_vl" && auto_map?["AutoModel"] == "modeling_d1.D1Model"),
              projector_hidden_size > 0, maxLength > 0, bos_token_id >= 0, bos_token_id < text_config.vocab_size,
              vision_config.hiddenSize > 0, vision_config.intermediateSize > 0, vision_config.numAttentionHeads > 0,
              vision_config.modelType == "siglip2_vision_model", vision_config.visionUseHead != true,
              (downsample_factor ?? 2) == 2, projector_bias != false,
              (projector_hidden_act ?? "gelu") == "gelu", projector_use_layernorm != true,
              vision_config.hiddenSize % vision_config.numAttentionHeads == 0, vision_config.patchSize == 16,
              vision_config.numChannels == 3, vision_config.numHiddenLayers > 0,
              vision_config.numPatches == 256, vision_config.layerNormEpsilon > 0 else {
            throw D1Error.invalid("This directory is not a supported D1 decision checkpoint.")
        }
        if isOmni {
            guard let head_layers, head_layers > 0, text_config.hidden_size % 64 == 0,
                  (temperatures ?? [:]).values.allSatisfy({ $0.isFinite && $0 > 0 }),
                  (image_text_length ?? 896) > 0, (audio_text_length ?? 15_360) > 0,
                  let audio_config else { throw D1Error.invalid("Invalid D1 omni head or media configuration.") }
            try audio_config.validate()
        } else if image_token_id.map({ !(0..<text_config.vocab_size).contains($0) }) ?? true {
            throw D1Error.invalid("D1-3B requires its image token id.")
        }
    }
}

public struct D1AudioConfiguration: Decodable, Sendable {
    public let feat_in: Int
    public let n_layers: Int
    public let d_model: Int
    public let subsampling_conv_channels: Int
    public let ff_expansion_factor: Int
    public let n_heads: Int
    public let conv_kernel_size: Int
    public let residual_width: Int
    func validate() throws {
        guard feat_in == 128, n_layers > 0, d_model > 0, d_model % 2 == 0, n_heads > 0,
              d_model % n_heads == 0, subsampling_conv_channels > 0, ff_expansion_factor > 0,
              conv_kernel_size > 0, conv_kernel_size % 2 == 1, residual_width > 0 else {
            throw D1Error.invalid("Unsupported D1 audio configuration.")
        }
    }
}
