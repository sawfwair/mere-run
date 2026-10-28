import Foundation
import MereRunQwenModel

public enum Drama3Error: LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let reason): "DRAMA-3: \(reason)" }
    }
}

public struct Drama3Config: Decodable, Sendable {
    public let text: Q35TextConfig
    public let audio: Audio
    public let eos: Int
    public let semanticStart: Int
    public let semanticEnd: Int
    public let quantization: Q35QuantizationConfig?

    public struct Audio: Decodable, Sendable {
        public let dim: Int
        public let heads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let intermediate: Int
        public let codebooks: Int
        public let vocab: Int
        public let sharedLayers: Int
        public let backboneCodebooks: Int
        public let eps: Float
        public let ropeBase: Float
        public let rotaryPercent: Float
        let decoderType: String
        let rotaryInterleaved: Bool
        let outputGate: Bool
        let zeroCentered: Bool
        enum CodingKeys: String, CodingKey {
            case dim, heads = "n_head", kvHeads = "n_local_heads", headDim = "head_dim"
            case intermediate = "intermediate_size", codebooks = "num_codebooks", vocab = "vocab_size"
            case sharedLayers = "shared_enrichment_layers", backboneCodebooks = "vq_backbone_codebooks"
            case eps = "norm_eps", ropeBase = "rope_base", rotaryPercent = "rotary_percent"
            case decoderType = "decoder_type", rotaryInterleaved = "rotary_interleaved"
            case outputGate = "attention_output_gate", zeroCentered = "layernorm_zero_centered_gamma"
        }
    }

    enum CodingKeys: String, CodingKey {
        case text = "text_config", audio = "audio_decoder_config", eos = "eos_token_id"
        case semanticStart = "semantic_start_token_id", semanticEnd = "semantic_end_token_id"
        case quantization = "mere_run_drama3_quantization", modelType = "model_type"
        case template = "speech_template_version"
    }

    private struct TextInvariants: Decodable {
        let rotary_interleaved: Bool
        let use_qk_norm: Bool
        let layernorm_zero_centered_gamma: Bool
        let shared_expert_gate: Bool
        let mlp_layer_types: [String]
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let invariants = try c.decode(TextInvariants.self, forKey: .text)
        text = try c.decode(Q35TextConfig.self, forKey: .text)
        audio = try c.decode(Audio.self, forKey: .audio)
        eos = try c.decode(Int.self, forKey: .eos)
        semanticStart = try c.decode(Int.self, forKey: .semanticStart)
        semanticEnd = try c.decode(Int.self, forKey: .semanticEnd)
        quantization = try c.decodeIfPresent(Q35QuantizationConfig.self, forKey: .quantization)
        guard try c.decode(String.self, forKey: .modelType) == "fish_omni",
              try c.decode(String.self, forKey: .template) == "reference-bracket-v1",
              text.modelType == "fish_omni_text", text.attnOutputGate,
              !invariants.rotary_interleaved, invariants.use_qk_norm,
              invariants.layernorm_zero_centered_gamma, invariants.shared_expert_gate,
              invariants.mlp_layer_types == Array(repeating: "moe", count: text.numHiddenLayers),
              text.layerTypes.count == text.numHiddenLayers,
              Set(text.layerTypes).isSubset(of: ["full_attention", "linear_attention"]),
              text.numExperts > 0, text.numExpertsPerTok > 0, text.numExpertsPerTok <= text.numExperts,
              audio.dim == text.hiddenSize, audio.decoderType == "codebook_mtp_shared",
              audio.codebooks == 10, audio.backboneCodebooks == 1, audio.sharedLayers > 0,
              audio.rotaryInterleaved, audio.outputGate, audio.zeroCentered,
              audio.heads > 0, audio.kvHeads > 0, audio.heads.isMultiple(of: audio.kvHeads),
              audio.headDim > 0, audio.vocab > 0,
              semanticEnd - semanticStart + 1 == audio.vocab,
              (0..<text.vocabSize).contains(eos), semanticStart >= 0, semanticEnd < text.vocabSize else {
            throw Drama3Error.invalid("unsupported architecture or prompt schema")
        }
        if let quantization {
            guard quantization.bits == 4, quantization.groupSize == 64, quantization.mode == "affine" else {
                throw Drama3Error.invalid("expected expert-only affine Q4/group-64 pack")
            }
        }
    }

    package func backboneConfig() throws -> Q35Config {
        struct Bridge: Encodable {
            let model_type = "fish_omni_text"
            let text_config: Q35TextConfig
            let quantization: Q35QuantizationConfig
        }
        // Only routed experts are constructed packed. All other modules stay BF16.
        return try JSONDecoder().decode(Q35Config.self, from: JSONEncoder().encode(
            Bridge(text_config: text, quantization: .init(groupSize: 64, bits: 4))
        ))
    }
}
