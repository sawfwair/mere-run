import Foundation

public struct LingBotVideoTransformerConfig: Decodable, Sendable, Hashable {
    public let axesDims: [Int]
    public let axesLens: [Int]
    public let depth: Int
    public let freqDim: Int
    public let hiddenSize: Int
    public let inChannels: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let normEps: Float
    public let normTopKProb: Bool
    public let numAttentionHeads: Int
    public let numExperts: Int
    public let numExpertsPerTok: Int
    public let nGroup: Int?
    public let nSharedExperts: Int?
    public let outBias: Bool
    public let outChannels: Int
    public let patchEmbedBias: Bool
    public let patchSize: [Int]
    public let qkvBias: Bool
    public let ropeTheta: Float
    public let routedScalingFactor: Float
    public let scoreFunc: String
    public let textDim: Int
    public let timestepMLPBias: Bool
    public let topKGroup: Int?

    enum CodingKeys: String, CodingKey {
        case axesDims = "axes_dims"
        case axesLens = "axes_lens"
        case depth
        case freqDim = "freq_dim"
        case hiddenSize = "hidden_size"
        case inChannels = "in_channels"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normEps = "norm_eps"
        case normTopKProb = "norm_topk_prob"
        case numAttentionHeads = "num_attention_heads"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case nGroup = "n_group"
        case nSharedExperts = "n_shared_experts"
        case outBias = "out_bias"
        case outChannels = "out_channels"
        case patchEmbedBias = "patch_embed_bias"
        case patchSize = "patch_size"
        case qkvBias = "qkv_bias"
        case ropeTheta = "rope_theta"
        case routedScalingFactor = "routed_scaling_factor"
        case scoreFunc = "score_func"
        case textDim = "text_dim"
        case timestepMLPBias = "timestep_mlp_bias"
        case topKGroup = "topk_group"
    }

    public init(
        axesDims: [Int] = [32, 48, 48],
        axesLens: [Int] = [8192, 1024, 1024],
        depth: Int = 24,
        freqDim: Int = 256,
        hiddenSize: Int = 2048,
        inChannels: Int = 16,
        intermediateSize: Int = 6144,
        moeIntermediateSize: Int = 768,
        normEps: Float = 1e-6,
        normTopKProb: Bool = true,
        numAttentionHeads: Int = 16,
        numExperts: Int = 0,
        numExpertsPerTok: Int = 8,
        nGroup: Int? = nil,
        nSharedExperts: Int? = nil,
        outBias: Bool = true,
        outChannels: Int = 16,
        patchEmbedBias: Bool = true,
        patchSize: [Int] = [1, 2, 2],
        qkvBias: Bool = false,
        ropeTheta: Float = 256,
        routedScalingFactor: Float = 1,
        scoreFunc: String = "sigmoid",
        textDim: Int = 2560,
        timestepMLPBias: Bool = true,
        topKGroup: Int? = nil
    ) {
        self.axesDims = axesDims
        self.axesLens = axesLens
        self.depth = depth
        self.freqDim = freqDim
        self.hiddenSize = hiddenSize
        self.inChannels = inChannels
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.normEps = normEps
        self.normTopKProb = normTopKProb
        self.numAttentionHeads = numAttentionHeads
        self.numExperts = numExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.nGroup = nGroup
        self.nSharedExperts = nSharedExperts
        self.outBias = outBias
        self.outChannels = outChannels
        self.patchEmbedBias = patchEmbedBias
        self.patchSize = patchSize
        self.qkvBias = qkvBias
        self.ropeTheta = ropeTheta
        self.routedScalingFactor = routedScalingFactor
        self.scoreFunc = scoreFunc
        self.textDim = textDim
        self.timestepMLPBias = timestepMLPBias
        self.topKGroup = topKGroup
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.axesDims = try container.decode([Int].self, forKey: .axesDims)
        self.axesLens = try container.decode([Int].self, forKey: .axesLens)
        self.depth = try container.decode(Int.self, forKey: .depth)
        self.freqDim = try container.decode(Int.self, forKey: .freqDim)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.inChannels = try container.decode(Int.self, forKey: .inChannels)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 768
        self.normEps = try container.decode(Float.self, forKey: .normEps)
        self.normTopKProb = try container.decodeIfPresent(Bool.self, forKey: .normTopKProb) ?? true
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numExperts = try container.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 8
        self.nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup)
        self.nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        self.outBias = try container.decode(Bool.self, forKey: .outBias)
        self.outChannels = try container.decode(Int.self, forKey: .outChannels)
        self.patchEmbedBias = try container.decode(Bool.self, forKey: .patchEmbedBias)
        self.patchSize = try container.decode([Int].self, forKey: .patchSize)
        self.qkvBias = try container.decode(Bool.self, forKey: .qkvBias)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self.routedScalingFactor = try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
        self.scoreFunc = try container.decodeIfPresent(String.self, forKey: .scoreFunc) ?? "sigmoid"
        self.textDim = try container.decode(Int.self, forKey: .textDim)
        self.timestepMLPBias = try container.decode(Bool.self, forKey: .timestepMLPBias)
        self.topKGroup = try container.decodeIfPresent(Int.self, forKey: .topKGroup)
    }

    public var headDim: Int {
        hiddenSize / numAttentionHeads
    }
}

public struct LingBotVideoQuantizationConfig: Codable, Sendable, Hashable {
    public static let filename = "mererun_lingbot_quantization.json"

    public let schemaVersion: Int
    public let bits: Int
    public let groupSize: Int
    public let mode: String
    public let expertsOnly: Bool
    public let includesRefiner: Bool
    public let sourceModelID: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case bits
        case groupSize = "group_size"
        case mode
        case expertsOnly = "experts_only"
        case includesRefiner = "includes_refiner"
        case sourceModelID = "source_model_id"
    }

    public init(
        schemaVersion: Int = 1,
        bits: Int,
        groupSize: Int,
        mode: String = "affine",
        expertsOnly: Bool = true,
        includesRefiner: Bool,
        sourceModelID: String
    ) {
        self.schemaVersion = schemaVersion
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode
        self.expertsOnly = expertsOnly
        self.includesRefiner = includesRefiner
        self.sourceModelID = sourceModelID
    }
}

public struct LingBotVideoVAEConfig: Decodable, Sendable, Hashable {
    public let baseDim: Int
    public let dimMult: [Int]
    public let latentsMean: [Float]
    public let latentsStd: [Float]
    public let numResBlocks: Int
    public let temporalDownsample: [Bool]
    public let zDim: Int

    enum CodingKeys: String, CodingKey {
        case baseDim = "base_dim"
        case dimMult = "dim_mult"
        case latentsMean = "latents_mean"
        case latentsStd = "latents_std"
        case numResBlocks = "num_res_blocks"
        case temporalDownsample = "temperal_downsample"
        case zDim = "z_dim"
    }

    public var blockOutChannels: [Int] {
        dimMult.map { baseDim * $0 }
    }

    public var temporalCompressionRatio: Int {
        1 << temporalDownsample.filter { $0 }.count
    }
}

public struct LingBotVideoTextConfig: Decodable, Sendable, Hashable {
    public struct Text: Decodable, Sendable, Hashable {
        public let headDim: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let maxPositionEmbeddings: Int
        public let numAttentionHeads: Int
        public let numHiddenLayers: Int
        public let numKeyValueHeads: Int
        public let rmsNormEps: Float
        public let ropeScaling: RopeScaling?
        public let ropeTheta: Float
        public let vocabSize: Int

        enum CodingKeys: String, CodingKey {
            case headDim = "head_dim"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case maxPositionEmbeddings = "max_position_embeddings"
            case numAttentionHeads = "num_attention_heads"
            case numHiddenLayers = "num_hidden_layers"
            case numKeyValueHeads = "num_key_value_heads"
            case rmsNormEps = "rms_norm_eps"
            case ropeScaling = "rope_scaling"
            case ropeTheta = "rope_theta"
            case vocabSize = "vocab_size"
        }
    }

    public struct RopeScaling: Decodable, Sendable, Hashable {
        public let mropeInterleaved: Bool?
        public let mropeSection: [Int]?

        enum CodingKeys: String, CodingKey {
            case mropeInterleaved = "mrope_interleaved"
            case mropeSection = "mrope_section"
        }
    }

    public let textConfig: Text

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }
}

public struct LingBotVideoResources: Sendable {
    public enum ResourceError: LocalizedError {
        case missingFile(URL)
        case unquantizedMoE(Int)
        case invalidConfiguration(String)

        public var errorDescription: String? {
            switch self {
            case .missingFile(let url):
                return "Missing LingBot-Video file: \(url.path)"
            case .unquantizedMoE(let experts):
                return "LingBot-Video MoE declares \(experts) experts and must be converted with `mere.run model quantize` before native inference."
            case .invalidConfiguration(let message):
                return "Invalid LingBot-Video configuration: \(message)"
            }
        }
    }

    public let rootURL: URL
    public let processorURL: URL
    public let textEncoderURL: URL
    public let transformerURL: URL
    public let refinerURL: URL?
    public let vaeURL: URL
    public let textConfig: LingBotVideoTextConfig
    public let transformerConfig: LingBotVideoTransformerConfig
    public let refinerConfig: LingBotVideoTransformerConfig?
    public let vaeConfig: LingBotVideoVAEConfig
    public let quantizationConfig: LingBotVideoQuantizationConfig?

    public init(rootURL: URL, fileManager: FileManager = .default) throws {
        let root = rootURL.resolvingSymlinksInPath()
        let modelIndexURL = root.appendingPathComponent("model_index.json")
        let processorURL = root.appendingPathComponent("processor", isDirectory: true)
        let textEncoderURL = root.appendingPathComponent("text_encoder", isDirectory: true)
        let transformerURL = root.appendingPathComponent("transformer", isDirectory: true)
        let refinerURL = root.appendingPathComponent("refiner", isDirectory: true)
        let vaeURL = root.appendingPathComponent("vae", isDirectory: true)

        let transformerWeights = transformerURL.appendingPathComponent("diffusion_pytorch_model.safetensors")
        let transformerIndex = transformerURL.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
        let required = [
            modelIndexURL,
            processorURL.appendingPathComponent("tokenizer.json"),
            processorURL.appendingPathComponent("tokenizer_config.json"),
            textEncoderURL.appendingPathComponent("config.json"),
            textEncoderURL.appendingPathComponent("model.safetensors.index.json"),
            transformerURL.appendingPathComponent("config.json"),
            vaeURL.appendingPathComponent("config.json"),
            vaeURL.appendingPathComponent("diffusion_pytorch_model.safetensors"),
        ]
        if let missing = required.first(where: { !fileManager.fileExists(atPath: $0.path) }) {
            throw ResourceError.missingFile(missing)
        }
        guard fileManager.fileExists(atPath: transformerWeights.path)
                || fileManager.fileExists(atPath: transformerIndex.path) else {
            throw ResourceError.missingFile(transformerWeights)
        }

        let decoder = JSONDecoder()
        let textConfig = try decoder.decode(
            LingBotVideoTextConfig.self,
            from: Data(contentsOf: textEncoderURL.appendingPathComponent("config.json"))
        )
        let transformerConfig = try decoder.decode(
            LingBotVideoTransformerConfig.self,
            from: Data(contentsOf: transformerURL.appendingPathComponent("config.json"))
        )
        let refinerConfigURL = refinerURL.appendingPathComponent("config.json")
        let hasRefiner = fileManager.fileExists(atPath: refinerConfigURL.path)
        let refinerConfig = hasRefiner
            ? try decoder.decode(
                LingBotVideoTransformerConfig.self,
                from: Data(contentsOf: refinerConfigURL)
            )
            : nil
        let vaeConfig = try decoder.decode(
            LingBotVideoVAEConfig.self,
            from: Data(contentsOf: vaeURL.appendingPathComponent("config.json"))
        )
        let quantizationURL = root.appendingPathComponent(LingBotVideoQuantizationConfig.filename)
        let quantizationConfig = fileManager.fileExists(atPath: quantizationURL.path)
            ? try decoder.decode(
                LingBotVideoQuantizationConfig.self,
                from: Data(contentsOf: quantizationURL)
            )
            : nil

        guard transformerConfig.patchSize.count == 3 else {
            throw ResourceError.invalidConfiguration("patch_size must contain temporal, height, and width dimensions")
        }
        guard transformerConfig.axesDims.count == 3,
              transformerConfig.axesDims.reduce(0, +) == transformerConfig.headDim
        else {
            throw ResourceError.invalidConfiguration("axes_dims must contain three values summing to the attention head dimension")
        }
        guard vaeConfig.latentsMean.count == transformerConfig.inChannels,
              vaeConfig.latentsStd.count == transformerConfig.inChannels
        else {
            throw ResourceError.invalidConfiguration("VAE latent statistics must match transformer channels")
        }
        if let quantizationConfig {
            guard quantizationConfig.schemaVersion == 1,
                  quantizationConfig.bits == 4,
                  quantizationConfig.groupSize > 0,
                  quantizationConfig.mode == "affine",
                  quantizationConfig.expertsOnly
            else {
                throw ResourceError.invalidConfiguration("unsupported LingBot MoE quantization metadata")
            }
        }

        self.rootURL = root
        self.processorURL = processorURL
        self.textEncoderURL = textEncoderURL
        self.transformerURL = transformerURL
        self.refinerURL = hasRefiner ? refinerURL : nil
        self.vaeURL = vaeURL
        self.textConfig = textConfig
        self.transformerConfig = transformerConfig
        self.refinerConfig = refinerConfig
        self.vaeConfig = vaeConfig
        self.quantizationConfig = quantizationConfig
    }

    public func validateForInference() throws {
        if transformerConfig.numExperts > 0, quantizationConfig == nil {
            throw ResourceError.unquantizedMoE(transformerConfig.numExperts)
        }
    }

    public func validateForRefiner(fileManager: FileManager = .default) throws {
        guard let refinerURL, let refinerConfig else {
            throw ResourceError.missingFile(rootURL.appendingPathComponent("refiner/config.json"))
        }
        guard transformerConfig.numExperts > 0 else {
            throw ResourceError.invalidConfiguration("the released refiner is available only for LingBot MoE")
        }
        if quantizationConfig?.includesRefiner != true {
            throw ResourceError.invalidConfiguration("quantized LingBot MoE artifact does not include refiner weights")
        }

        let weights = refinerURL.appendingPathComponent("diffusion_pytorch_model.safetensors")
        let index = refinerURL.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
        guard fileManager.fileExists(atPath: weights.path) || fileManager.fileExists(atPath: index.path) else {
            throw ResourceError.missingFile(weights)
        }
        guard refinerConfig.inChannels == transformerConfig.inChannels,
              refinerConfig.outChannels == transformerConfig.outChannels,
              refinerConfig.textDim == transformerConfig.textDim,
              refinerConfig.patchSize == transformerConfig.patchSize,
              refinerConfig.numExperts == transformerConfig.numExperts
        else {
            throw ResourceError.invalidConfiguration("refiner architecture does not match the base transformer contract")
        }
    }
}
