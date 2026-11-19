import Foundation
import MereRunQwenModel

package enum HaruError: LocalizedError {
    case unsupportedConfiguration
    case checkpointParameter(String)
    case invalidFrame(Int)
    case sessionNotStarted
    case sessionAlreadyStarted
    case contextLimit

    package var errorDescription: String? {
        switch self {
        case .unsupportedConfiguration: "Unsupported Haru checkpoint configuration."
        case .checkpointParameter(let name): "Missing or incompatible Haru checkpoint parameter: \(name)."
        case .invalidFrame(let count): "Expected one Haru audio frame of 3,200 samples, got \(count)."
        case .sessionNotStarted: "Start the Haru session before sending audio."
        case .sessionAlreadyStarted: "This Haru session has already started."
        case .contextLimit: "The Haru conversation reached the model's context limit."
        }
    }
}

package struct HaruConfiguration: Decodable {
    package static let sampleRate = 24_000
    package static let frameSamples = 3_200

    struct Codec: Decodable {
        let latentDim: Int
        let baseChannels: Int
        let ratios: [Int]
        let encoderDepths: [Int]
        let decoderDepths: [Int]
        let normEps: Float
        let latentStd: Float

        private enum CodingKeys: String, CodingKey {
            case latentDim = "latent_dim"
            case baseChannels = "base_channels"
            case ratios
            case encoderDepths = "encoder_depths"
            case decoderDepths = "decoder_depths"
            case normEps = "norm_eps"
            case latentStd = "latent_std"
        }
    }

    struct Head: Decodable {
        let hiddenSize: Int
        let latentDim: Int
        let numLayers: Int
        let ffnRatio: Float
        let normEps: Float
        let intervalEmbed: Float
        let numSteps: Int
        let guidanceScale: Float

        private enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case latentDim = "latent_dim"
            case numLayers = "num_layers"
            case ffnRatio = "ffn_ratio"
            case normEps = "norm_eps"
            case intervalEmbed = "interval_embed"
            case numSteps = "num_steps"
            case guidanceScale = "guidance_scale"
        }
    }

    let padTokenId: Int
    let eopTokenId: Int
    let latentDim: Int
    let lm: Q35Config
    let codec: Codec
    let head: Head

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case sampleRate = "sample_rate"
        case frameSamples = "frame_samples"
        case padTokenId = "pad_token_id"
        case eopTokenId = "eop_token_id"
        case latentDim = "latent_dim"
        case lm, codec, head
    }

    private enum BackboneKeys: String, CodingKey {
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    package init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let backbone = try values.nestedContainer(keyedBy: BackboneKeys.self, forKey: .lm)
        let rawText = try values.decode(Q35TextConfig.self, forKey: .lm)
        self.lm = Q35Config(haruTextConfig: rawText)
        self.codec = try values.decode(Codec.self, forKey: .codec)
        self.head = try values.decode(Head.self, forKey: .head)
        self.padTokenId = try values.decode(Int.self, forKey: .padTokenId)
        self.eopTokenId = try values.decode(Int.self, forKey: .eopTokenId)
        self.latentDim = try values.decode(Int.self, forKey: .latentDim)
        guard try values.decode(String.self, forKey: .modelType) == "haru",
              try values.decode(Int.self, forKey: .sampleRate) == Self.sampleRate,
              try values.decode(Int.self, forKey: .frameSamples) == Self.frameSamples,
              try backbone.decode(Bool.self, forKey: .tieWordEmbeddings) else {
            throw HaruError.unsupportedConfiguration
        }
        try validate()
    }

    private func validate() throws {
        let text = lm.textConfig
        guard text.modelType == "qwen3_5_text", !text.usesMoE, text.attnOutputGate,
              text.layerTypes.count == text.numHiddenLayers,
              text.layerTypes.allSatisfy({ $0 == "linear_attention" || $0 == "full_attention" }),
              padTokenId != eopTokenId, (0..<text.vocabSize).contains(padTokenId),
              (0..<text.vocabSize).contains(eopTokenId), latentDim == codec.latentDim,
              latentDim == head.latentDim, head.hiddenSize == text.hiddenSize,
              codec.ratios == [8, 5, 5, 4, 2, 2], codec.baseChannels > 0,
              codec.encoderDepths.count == 7, codec.decoderDepths.count == 7,
              codec.encoderDepths.allSatisfy({ $0 >= 0 }), codec.decoderDepths.allSatisfy({ $0 >= 0 }),
              head.numLayers > 0, head.numSteps == 1, (1...2).contains(head.guidanceScale) else {
            throw HaruError.unsupportedConfiguration
        }
    }

    package static func load(from root: URL) throws -> Self {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: root.appending(path: "config.json")))
    }
}
