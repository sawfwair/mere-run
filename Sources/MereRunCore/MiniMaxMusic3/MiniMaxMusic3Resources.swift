import Foundation

public struct MiniMaxMusic3LanguageConfiguration: Codable, Hashable, Sendable {
    public struct RopeParameters: Codable, Hashable, Sendable {
        public let ropeTheta: Float

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
        }
    }

    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeParameters: RopeParameters

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
    }
}

public struct MiniMaxMusic3DepthConfiguration: Codable, Hashable, Sendable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let audioVocabSize: Int
    public let numCodebooks: Int
    public let maxPositionEmbeddings: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numLayers = "num_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case audioVocabSize = "audio_vocab_size"
        case numCodebooks = "num_codebooks"
        case maxPositionEmbeddings = "max_position_embeddings"
    }
}

public struct MiniMaxMusic3ConditionConfiguration: Codable, Hashable, Sendable {
    public let conditionHiddenDim: Int
    public let numConditionLayers: Int
    public let outDim: Int
    public let inputSamplingRate: Int
    public let inputHopLength: Int
    public let outputSamplingRate: Int
    public let outputHopLength: Int

    enum CodingKeys: String, CodingKey {
        case conditionHiddenDim = "condition_hidden_dim"
        case numConditionLayers = "num_condition_layers"
        case outDim = "out_dim"
        case inputSamplingRate = "input_sampling_rate"
        case inputHopLength = "input_hop_length"
        case outputSamplingRate = "output_sampling_rate"
        case outputHopLength = "output_hop_length"
    }
}

public struct MiniMaxMusic3TransformerConfiguration: Codable, Hashable, Sendable {
    public let inChannels: Int
    public let conditionDim: Int
    public let numLayers: Int
    public let numAttentionHeads: Int
    public let attentionHeadDim: Int
    public let ffInnerDim: Int
    public let rotaryDim: Int
    public let fourierEmbeddingDim: Int

    enum CodingKeys: String, CodingKey {
        case inChannels = "in_channels"
        case conditionDim = "condition_dim"
        case numLayers = "num_layers"
        case numAttentionHeads = "num_attention_heads"
        case attentionHeadDim = "attention_head_dim"
        case ffInnerDim = "ff_inner_dim"
        case rotaryDim = "rotary_dim"
        case fourierEmbeddingDim = "fourier_embedding_dim"
    }
}

public struct MiniMaxMusic3VocoderConfiguration: Codable, Hashable, Sendable {
    public let latentChannels: Int
    public let decoderInputDim: Int
    public let decoderHiddenDim: Int
    public let upsamplingRatios: [Int]
    public let samplingRate: Int

    enum CodingKeys: String, CodingKey {
        case latentChannels = "latent_channels"
        case decoderInputDim = "decoder_input_dim"
        case decoderHiddenDim = "decoder_hidden_dim"
        case upsamplingRatios = "upsampling_ratios"
        case samplingRate = "sampling_rate"
    }
}

public struct MiniMaxMusic3Resources: Sendable, Hashable {
    public static let modelID = "music-minimax-music3"
    public static let repository = "MiniMaxAI/MiniMax-Music3"
    public static let revision = "bd348f9c49ea3c1b39f33ace3436f8fad435f24e"
    public static let estimatedDownloadBytes: Int64 = 28_517_620_807

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var tokenizerURL: URL { rootURL.appendingPathComponent("tokenizer", isDirectory: true) }
    public var languageModelURL: URL { rootURL.appendingPathComponent("language_model", isDirectory: true) }
    public var depthDecoderURL: URL { rootURL.appendingPathComponent("rvq_depth_decoder", isDirectory: true) }
    public var conditionEncoderURL: URL { rootURL.appendingPathComponent("condition_encoder", isDirectory: true) }
    public var transformerURL: URL { rootURL.appendingPathComponent("transformer", isDirectory: true) }
    public var vocoderURL: URL { rootURL.appendingPathComponent("vocoder", isDirectory: true) }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing = Self.requiredFiles.map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
        for directory in [languageModelURL, transformerURL] {
            let index = directory.appendingPathComponent("model.safetensors.index.json")
            let diffusersIndex = directory.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
            let single = directory.appendingPathComponent("model.safetensors")
            let diffusersSingle = directory.appendingPathComponent("diffusion_pytorch_model.safetensors")
            if ![index, diffusersIndex, single, diffusersSingle].contains(where: {
                fileManager.fileExists(atPath: $0.path)
            }) {
                missing.append(directory.appendingPathComponent("*.safetensors"))
            }
        }
        return missing
    }

    public func loadLanguageConfiguration() throws -> MiniMaxMusic3LanguageConfiguration {
        try Self.decode(MiniMaxMusic3LanguageConfiguration.self, at: languageModelURL.appendingPathComponent("config.json"))
    }

    public func loadDepthConfiguration() throws -> MiniMaxMusic3DepthConfiguration {
        try Self.decode(MiniMaxMusic3DepthConfiguration.self, at: depthDecoderURL.appendingPathComponent("config.json"))
    }

    public func loadConditionConfiguration() throws -> MiniMaxMusic3ConditionConfiguration {
        try Self.decode(MiniMaxMusic3ConditionConfiguration.self, at: conditionEncoderURL.appendingPathComponent("config.json"))
    }

    public func loadTransformerConfiguration() throws -> MiniMaxMusic3TransformerConfiguration {
        try Self.decode(MiniMaxMusic3TransformerConfiguration.self, at: transformerURL.appendingPathComponent("config.json"))
    }

    public func loadVocoderConfiguration() throws -> MiniMaxMusic3VocoderConfiguration {
        try Self.decode(MiniMaxMusic3VocoderConfiguration.self, at: vocoderURL.appendingPathComponent("config.json"))
    }

    public static func looksLikeRoot(_ rootURL: URL, fileManager: FileManager = .default) -> Bool {
        fileManager.fileExists(atPath: rootURL.appendingPathComponent("modular_model_index.json").path)
            && fileManager.fileExists(atPath: rootURL.appendingPathComponent("language_model/config.json").path)
            && fileManager.fileExists(atPath: rootURL.appendingPathComponent("vocoder/config.json").path)
    }

    private static let requiredFiles = [
        "LICENSE",
        "config.json",
        "modular_model_index.json",
        "condition_encoder/config.json",
        "condition_encoder/diffusion_pytorch_model.safetensors",
        "language_model/config.json",
        "rvq_depth_decoder/config.json",
        "rvq_depth_decoder/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "vocoder/config.json",
        "vocoder/diffusion_pytorch_model.safetensors",
    ]

    private static func decode<T: Decodable>(_ type: T.Type, at url: URL) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: url))
    }
}
