import Foundation
@preconcurrency import Hub
import MLX
import MLXFast
import MLXNN
import Tokenizers

public struct LTXGemmaTextEncoding: @unchecked Sendable {
    public let lastHiddenState: MLXArray
    public let attentionMask: MLXArray
    public let normalizedStackedHiddenStates: MLXArray
    public let features: MLXArray
    public let videoEmbeddings: MLXArray
    public let audioEmbeddings: MLXArray?

    public init(
        lastHiddenState: MLXArray,
        attentionMask: MLXArray,
        normalizedStackedHiddenStates: MLXArray,
        features: MLXArray,
        videoEmbeddings: MLXArray,
        audioEmbeddings: MLXArray?
    ) {
        self.lastHiddenState = lastHiddenState
        self.attentionMask = attentionMask
        self.normalizedStackedHiddenStates = normalizedStackedHiddenStates
        self.features = features
        self.videoEmbeddings = videoEmbeddings
        self.audioEmbeddings = audioEmbeddings
    }
}

public enum LTXGemmaTextEncoderError: LocalizedError {
    case missingConfig(URL)
    case missingWeights(URL)
    case missingTokenizer(URL)
    case missingTransformerWeights(URL)
    case missingConnectorProjection(URL)
    case missingConnectorWeights(URL)
    case invalidPackedTextEncoder(URL, String)
    case tokenizerUnavailable
    case modelNotLoaded
    case connectorNotLoaded
    case emptyPrompt
    case invalidMaxLength(Int)

    public var errorDescription: String? {
        switch self {
        case .missingConfig(let url):
            return "Gemma text config not found at \(url.path)"
        case .missingWeights(let url):
            return "Gemma text weights not found at \(url.path)"
        case .missingTokenizer(let url):
            return "Tokenizer directory not found at \(url.path)"
        case .missingTransformerWeights(let url):
            return "LTX transformer safetensors not found at \(url.path)"
        case .missingConnectorProjection(let url):
            return "Missing text embedding projection weight in \(url.path)"
        case .missingConnectorWeights(let url):
            return "Missing LTX connector weights in \(url.path)"
        case .invalidPackedTextEncoder(let url, let reason):
            return "Invalid packed LTX text encoder at \(url.path): \(reason)"
        case .tokenizerUnavailable:
            return "Tokenizer is not loaded."
        case .modelNotLoaded:
            return "Gemma text model is not loaded."
        case .connectorNotLoaded:
            return "LTX connector is not loaded."
        case .emptyPrompt:
            return "Prompt cannot be empty."
        case .invalidMaxLength(let value):
            return "maxLength must be >= 1 (got \(value))."
        }
    }
}

public actor LTXGemmaTextEncoder {
    private var tokenizer: (any Tokenizer)?
    private var model: LTXGemmaLanguageModel?
    private var gemma4Model: Gemma4LanguageModel?
    private var featureExtractor: LTXGemmaFeaturesExtractor?
    private var featureExtractorV2: LTXGemmaFeaturesExtractorV2?
    private var videoConnector: LTXEmbeddings1DConnector?
    private var audioConnector: LTXEmbeddings1DConnector?
    private var loadedRoot: URL?

    public init() {}

    public func load(
        modelRoot: URL,
        textEncoderRoot overrideTextEncoderRoot: URL? = nil,
        dtype: DType = .bfloat16,
        loadConnectorWeights: Bool = true
    ) async throws {
        let fm = FileManager.default
        let root = modelRoot.standardizedFileURL
        if isLTX25ModelRoot(root) {
            try await loadLTX25(modelRoot: root, dtype: dtype, loadConnectorWeights: loadConnectorWeights)
            return
        }
        let usesLTX23SplitConnector = isLTX23SplitModelRoot(root)
        let textRoot = (overrideTextEncoderRoot ?? root.appendingPathComponent("text_encoder", isDirectory: true))
            .standardizedFileURL

        let configURL = textRoot.appendingPathComponent("config.json", isDirectory: false)
        guard fm.fileExists(atPath: configURL.path) else {
            throw LTXGemmaTextEncoderError.missingConfig(configURL)
        }
        let indexURL = textRoot.appendingPathComponent("model.safetensors.index.json", isDirectory: false)
        guard fm.fileExists(atPath: indexURL.path) else {
            throw LTXGemmaTextEncoderError.missingWeights(indexURL)
        }

        let tokenizerRootCandidate = root.appendingPathComponent("tokenizer", isDirectory: true)
        let tokenizerRoot = fm.fileExists(atPath: tokenizerRootCandidate.path)
            ? tokenizerRootCandidate
            : textRoot
        guard fm.fileExists(atPath: tokenizerRoot.path) else {
            throw LTXGemmaTextEncoderError.missingTokenizer(tokenizerRoot)
        }

        let topConfigData = try Data(contentsOf: configURL)
        let topConfig = try JSONDecoder().decode(LTXGemmaTopConfig.self, from: topConfigData)
        let textConfig = topConfig.textConfig ?? topConfig.asTextConfig
        let modelConfig = LTXGemmaModelConfig(textConfig: textConfig)
        let languageModel = LTXGemmaLanguageModel(config: modelConfig)
        let hiddenStateCount = modelConfig.numHiddenLayers + 1
        let textFeatures = LTXGemmaFeaturesExtractor(
            inputDim: modelConfig.hiddenSize * hiddenStateCount,
            outputDim: modelConfig.hiddenSize
        )
        let textFeaturesV2 = LTXGemmaFeaturesExtractorV2(
            captionChannels: modelConfig.hiddenSize,
            numGemmaLayers: hiddenStateCount,
            videoDim: 4_096,
            audioDim: 2_048,
            numHeads: 32,
            videoHeadDim: 128,
            audioHeadDim: 64,
            numConnectorLayers: 8,
            numRegisters: 128
        )
        let connector = LTXEmbeddings1DConnector(
            dim: modelConfig.hiddenSize,
            numHeads: 30,
            headDim: 128,
            numLayers: 2,
            numLearnableRegisters: 128,
            positionalEmbeddingTheta: 10_000.0,
            positionalEmbeddingMaxPos: [4_096]
        )
        let audioConnector = LTXEmbeddings1DConnector(
            dim: modelConfig.hiddenSize,
            numHeads: 30,
            headDim: 128,
            numLayers: 2,
            numLearnableRegisters: 128,
            positionalEmbeddingTheta: 10_000.0,
            positionalEmbeddingMaxPos: [4_096]
        )

        if gemmaIndexContainsQuantizedWeights(indexURL: indexURL) {
            try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                indexURL: indexURL,
                to: languageModel,
                groupSize: 64,
                bits: 4,
                keyMapper: mapGemmaLanguageWeightKey,
                mapper: { key, value in
                    mapGemmaLanguageWeight(key: key, value: value, dtype: dtype)
                }
            )
        } else {
            try HFSafetensorsWeightsLoader.applyShardedWeights(
                indexURL: indexURL,
                to: languageModel,
                dtype: dtype,
                verify: .none,
                mapper: { key, value in
                    mapGemmaLanguageWeight(key: key, value: value, dtype: dtype)
                }
            )
        }

        let loadedTokenizer = try await AutoTokenizer.from(modelFolder: tokenizerRoot)

        if loadConnectorWeights {
            if usesLTX23SplitConnector {
                let connectorURL = root.appendingPathComponent("connector.safetensors", isDirectory: false)
                guard fm.fileExists(atPath: connectorURL.path) else {
                    throw LTXGemmaTextEncoderError.missingConnectorWeights(connectorURL)
                }
                let connectorMetadata = try SafetensorsStreamingLoader.metadata(url: connectorURL)
                guard connectorMetadata.keys.contains("connector.text_embedding_projection.video_aggregate_embed.weight"),
                      connectorMetadata.keys.contains("connector.text_embedding_projection.audio_aggregate_embed.weight")
                else {
                    throw LTXGemmaTextEncoderError.missingConnectorProjection(connectorURL)
                }
                guard connectorMetadata.keys.contains("connector.video_embeddings_connector.learnable_registers"),
                      connectorMetadata.keys.contains("connector.audio_embeddings_connector.learnable_registers")
                else {
                    throw LTXGemmaTextEncoderError.missingConnectorWeights(connectorURL)
                }

                try SafetensorsStreamingLoader.applyWeightsStreaming(
                    url: connectorURL,
                    to: textFeaturesV2,
                    dtype: dtype,
                    verify: .none,
                    include: { key in
                        key.hasPrefix("connector.")
                    },
                    mapper: { key, value in
                        mapLTX23TextConnectorWeight(key: key, value: value, dtype: dtype)
                    },
                    batchSize: 24
                )
            } else {
                let transformerURL = try findLTXTransformerWeights(modelRoot: root)
                let connectorSourceWeights = try SafetensorsStreamingLoader.loadArrays(
                    url: transformerURL,
                    where: { key in
                        key == "text_embedding_projection.aggregate_embed.weight"
                            || key == "model.diffusion_model.video_embeddings_connector.learnable_registers"
                            || key.hasPrefix("model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.")
                            || key == "model.diffusion_model.audio_embeddings_connector.learnable_registers"
                            || key.hasPrefix("model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.")
                    },
                    dtype: dtype
                )

                guard let projection = connectorSourceWeights["text_embedding_projection.aggregate_embed.weight"] else {
                    throw LTXGemmaTextEncoderError.missingConnectorProjection(transformerURL)
                }
                try textFeatures.update(
                    parameters: ModuleParameters.unflattened([("aggregate_embed.weight", projection)]),
                    verify: .none
                )

                let connectorWeights = mapConnectorWeights(
                    arrays: connectorSourceWeights,
                    prefix: "model.diffusion_model.video_embeddings_connector.",
                    dtype: dtype
                )
                guard !connectorWeights.isEmpty else {
                    throw LTXGemmaTextEncoderError.missingConnectorWeights(transformerURL)
                }
                try connector.update(
                    parameters: ModuleParameters.unflattened(connectorWeights),
                    verify: .none
                )

                let audioConnectorWeights = mapConnectorWeights(
                    arrays: connectorSourceWeights,
                    prefix: "model.diffusion_model.audio_embeddings_connector.",
                    dtype: dtype
                )
                if !audioConnectorWeights.isEmpty {
                    try audioConnector.update(
                        parameters: ModuleParameters.unflattened(audioConnectorWeights),
                        verify: .none
                    )
                }
            }
        }

        self.model = languageModel
        self.gemma4Model = nil
        self.featureExtractor = usesLTX23SplitConnector ? nil : textFeatures
        self.featureExtractorV2 = usesLTX23SplitConnector ? textFeaturesV2 : nil
        self.videoConnector = usesLTX23SplitConnector ? nil : connector
        self.audioConnector = usesLTX23SplitConnector ? nil : audioConnector
        self.tokenizer = loadedTokenizer
        self.loadedRoot = root
    }

    private func loadLTX25(
        modelRoot root: URL,
        dtype: DType,
        loadConnectorWeights: Bool
    ) async throws {
        let resources = LTX25Resources(rootURL: root)
        let textEncoderURL = resources.textEncoderURL
        let connectorURL = LTX25NativeModelPack.optimizedURLIfValid(
            resources: resources,
            kind: .connector
        ) ?? resources.transformerURL
        let metadata = try SafetensorsStreamingLoader.fileMetadata(url: textEncoderURL)
        guard let rawConfig = metadata["gemma_config"],
              let configData = rawConfig.data(using: .utf8) else {
            throw LTXGemmaTextEncoderError.invalidPackedTextEncoder(
                textEncoderURL,
                "missing gemma_config metadata"
            )
        }
        let config = try JSONDecoder().decode(Gemma4Config.self, from: configData)
        guard config.modelType == "gemma4_unified" else {
            throw LTXGemmaTextEncoderError.invalidPackedTextEncoder(
                textEncoderURL,
                "expected gemma4_unified, got \(config.modelType)"
            )
        }

        let languageModel = Gemma4LanguageModel(config: config.textConfig)
        try SafetensorsStreamingLoader.applyWeightsLazyMaterialized(
            url: textEncoderURL,
            to: languageModel,
            verify: .none,
            include: { key in
                key == "model.embed_tokens.weight"
                    || key == "model.norm.weight"
                    || key.hasPrefix("model.layers.")
            },
            mapper: { key, value in
                guard key.hasPrefix("model.") else { return [] }
                return [(
                    String(key.dropFirst("model.".count)),
                    HFSafetensorsWeightsLoader.castIfNeeded(value, dtype: dtype)
                )]
            },
            batchSize: 24
        )

        let textFeatures = LTXGemmaFeaturesExtractorV2(
            captionChannels: config.textConfig.hiddenSize,
            numGemmaLayers: config.textConfig.numHiddenLayers + 1,
            videoDim: 4_096,
            audioDim: 2_048,
            numHeads: 32,
            videoHeadDim: 128,
            audioHeadDim: 64,
            numConnectorLayers: 8,
            numRegisters: 128
        )
        if loadConnectorWeights {
            try SafetensorsStreamingLoader.applyWeightsLazyMaterialized(
                url: textEncoderURL,
                to: textFeatures,
                verify: .none,
                include: { $0.hasPrefix("text_embedding_projection.") },
                mapper: { key, value in
                    [(key, HFSafetensorsWeightsLoader.castIfNeeded(value, dtype: dtype))]
                },
                batchSize: 4
            )
            try SafetensorsStreamingLoader.applyWeightsLazyMaterialized(
                url: connectorURL,
                to: textFeatures,
                verify: .none,
                include: { key in
                    key.hasPrefix("model.diffusion_model.video_embeddings_connector.")
                        || key.hasPrefix("model.diffusion_model.audio_embeddings_connector.")
                },
                mapper: { key, value in
                    mapLTX25TextConnectorWeight(key: key, value: value, dtype: dtype)
                },
                batchSize: 24
            )
        }

        let assetArrays = try SafetensorsStreamingLoader.loadArrays(
            url: textEncoderURL,
            where: { key in
                key == "tokenizer_json" || key == "hf_asset__tokenizer_config.json"
            }
        )
        guard let tokenizerJSON = assetArrays["tokenizer_json"],
              let tokenizerConfigJSON = assetArrays["hf_asset__tokenizer_config.json"] else {
            throw LTXGemmaTextEncoderError.invalidPackedTextEncoder(
                textEncoderURL,
                "missing embedded tokenizer assets"
            )
        }
        MLX.eval(tokenizerJSON, tokenizerConfigJSON)
        let tokenizerData = try JSONDecoder().decode(
            Config.self,
            from: Data(tokenizerJSON.asArray(UInt8.self))
        )
        let tokenizerConfig = try JSONDecoder().decode(
            Config.self,
            from: Data(tokenizerConfigJSON.asArray(UInt8.self))
        )
        let loadedTokenizer = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData
        )

        self.model = nil
        self.gemma4Model = languageModel
        self.featureExtractor = nil
        self.featureExtractorV2 = textFeatures
        self.videoConnector = nil
        self.audioConnector = nil
        self.tokenizer = loadedTokenizer
        self.loadedRoot = root
    }

    public func encode(
        prompt: String,
        maxLength: Int = 1024
    ) throws -> LTXGemmaTextEncoding {
        guard maxLength >= 1 else {
            throw LTXGemmaTextEncoderError.invalidMaxLength(maxLength)
        }
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw LTXGemmaTextEncoderError.emptyPrompt
        }
        guard model != nil || gemma4Model != nil else {
            throw LTXGemmaTextEncoderError.modelNotLoaded
        }
        guard let tokenizer else {
            throw LTXGemmaTextEncoderError.tokenizerUnavailable
        }
        guard featureExtractor != nil || featureExtractorV2 != nil else {
            throw LTXGemmaTextEncoderError.connectorNotLoaded
        }

        var tokenIDs = tokenizer.encode(text: trimmed, addSpecialTokens: true)
        if gemma4Model != nil, let bosTokenID = tokenizer.bosTokenId,
           tokenIDs.first != bosTokenID {
            tokenIDs.insert(bosTokenID, at: 0)
        }
        if tokenIDs.count > maxLength {
            tokenIDs = Array(tokenIDs.prefix(maxLength))
        }

        let padID = tokenizer.convertTokenToId("<pad>") ?? tokenizer.eosTokenId ?? 0
        let padCount = max(0, maxLength - tokenIDs.count)
        let inputIDs = Array(repeating: padID, count: padCount) + tokenIDs
        let attention = Array(repeating: 0, count: padCount) + Array(repeating: 1, count: tokenIDs.count)

        let inputArray = MLXArray(inputIDs.map(Int32.init)).reshaped(1, maxLength)
        let maskArray = MLXArray(attention.map(Int32.init)).reshaped(1, maxLength)

        let lastHiddenState: MLXArray
        let hiddenStates: [MLXArray]
        if let gemma4Model {
            let result = gemma4Model.forwardHiddenStates(
                inputIds: inputArray,
                attentionMask: maskArray
            )
            lastHiddenState = result.lastHiddenState
            hiddenStates = result.hiddenStates
        } else if let model {
            let result = model.forward(
                inputIds: inputArray,
                attentionMask: maskArray,
                outputHiddenStates: true
            )
            lastHiddenState = result.lastHiddenState
            hiddenStates = result.hiddenStates ?? [result.lastHiddenState]
        } else {
            throw LTXGemmaTextEncoderError.modelNotLoaded
        }
        if let featureExtractorV2 {
            let normalized = normalizeAndConcatHiddenStatesV2(
                hiddenStates: hiddenStates,
                attentionMask: maskArray
            )
            let extraction = featureExtractorV2(normalized, attentionMask: maskArray)
            return LTXGemmaTextEncoding(
                lastHiddenState: lastHiddenState,
                attentionMask: maskArray,
                normalizedStackedHiddenStates: normalized,
                features: extraction.projectedVideo,
                videoEmbeddings: extraction.videoEmbeddings,
                audioEmbeddings: extraction.audioEmbeddings
            )
        }

        guard let featureExtractor, let videoConnector else {
            throw LTXGemmaTextEncoderError.connectorNotLoaded
        }
        let normalized = normalizeAndConcatHiddenStates(
            hiddenStates: hiddenStates,
            attentionMask: maskArray
        )
        let features = featureExtractor(normalized)
        let additiveMask = (maskArray.asType(features.dtype) - MLXArray(1.0).asType(features.dtype))
            .reshaped(maskArray.dim(0), 1, 1, maskArray.dim(1)) * MLXArray(1e9).asType(features.dtype)
        let (videoEmbeddings, _) = videoConnector(features, attentionMask: additiveMask)
        let audioEmbeddings = audioConnector.map { connector in
            connector(features, attentionMask: additiveMask).0
        }

        return LTXGemmaTextEncoding(
            lastHiddenState: lastHiddenState,
            attentionMask: maskArray,
            normalizedStackedHiddenStates: normalized,
            features: features,
            videoEmbeddings: videoEmbeddings,
            audioEmbeddings: audioEmbeddings
        )
    }

    public func unload() {
        tokenizer = nil
        model = nil
        gemma4Model = nil
        featureExtractor = nil
        featureExtractorV2 = nil
        videoConnector = nil
        audioConnector = nil
        loadedRoot = nil
        Memory.clearCache()
    }

    public func loadedModelRoot() -> URL? {
        loadedRoot
    }
}

// MARK: - Config

struct LTXGemmaTopConfig: Decodable {
    let textConfig: LTXGemmaTextConfig?
    let modelType: String?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case modelType = "model_type"
    }

    var asTextConfig: LTXGemmaTextConfig {
        LTXGemmaTextConfig(
            vocabSize: 262_208,
            hiddenSize: 3840,
            numHiddenLayers: 48,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            intermediateSize: 15_360,
            maxPositionEmbeddings: 131_072,
            rmsNormEps: 1e-6,
            headDim: 256,
            ropeTheta: 1_000_000,
            ropeLocalBaseFreq: 10_000,
            ropeGlobalBaseFreq: 1_000_000,
            ropeTraditional: false,
            queryPreAttnScalar: 256,
            slidingWindowPattern: 6
        )
    }
}

struct LTXGemmaTextConfig: Decodable {
    let vocabSize: Int?
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let intermediateSize: Int
    let maxPositionEmbeddings: Int?
    let rmsNormEps: Float?
    let headDim: Int?
    let ropeTheta: Float?
    let ropeLocalBaseFreq: Float?
    let ropeGlobalBaseFreq: Float?
    let ropeTraditional: Bool?
    let queryPreAttnScalar: Int?
    let slidingWindowPattern: Int?

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case headDim = "head_dim"
        case ropeTheta = "rope_theta"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeGlobalBaseFreq = "rope_global_base_freq"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindowPattern = "sliding_window_pattern"
    }
}

struct LTXGemmaModelConfig {
    let vocabSize: Int
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let intermediateSize: Int
    let maxPositionEmbeddings: Int
    let rmsNormEps: Float
    let headDim: Int
    let ropeTheta: Float
    let ropeLocalBaseFreq: Float
    let ropeGlobalBaseFreq: Float
    let ropeTraditional: Bool
    let queryPreAttnScalar: Float
    let slidingWindowPattern: Int

    init(textConfig: LTXGemmaTextConfig) {
        self.vocabSize = textConfig.vocabSize
            ?? 262_208
        self.hiddenSize = textConfig.hiddenSize
        self.numHiddenLayers = textConfig.numHiddenLayers
        self.numAttentionHeads = textConfig.numAttentionHeads
        self.numKeyValueHeads = textConfig.numKeyValueHeads
        self.intermediateSize = textConfig.intermediateSize
        self.maxPositionEmbeddings = textConfig.maxPositionEmbeddings
            ?? 131_072
        self.rmsNormEps = textConfig.rmsNormEps
            ?? 1e-6
        self.headDim = textConfig.headDim
            ?? 256
        self.ropeTheta = textConfig.ropeTheta
            ?? 1_000_000
        self.ropeLocalBaseFreq = textConfig.ropeLocalBaseFreq ?? 10_000
        self.ropeGlobalBaseFreq = textConfig.ropeGlobalBaseFreq ?? textConfig.ropeTheta
            ?? 1_000_000
        self.ropeTraditional = textConfig.ropeTraditional ?? false
        self.queryPreAttnScalar = Float(textConfig.queryPreAttnScalar ?? textConfig.headDim ?? 256)
        self.slidingWindowPattern = max(1, textConfig.slidingWindowPattern ?? 6)
    }
}

// MARK: - Model

private final class LTXGemmaLanguageModel: Module {
    @ModuleInfo(key: "model") var model: LTXGemmaModel

    init(config: LTXGemmaModelConfig) {
        self._model.wrappedValue = LTXGemmaModel(config: config)
    }

    func forward(
        inputIds: MLXArray,
        attentionMask: MLXArray?,
        outputHiddenStates: Bool
    ) -> (lastHiddenState: MLXArray, hiddenStates: [MLXArray]?) {
        model.forward(inputIds: inputIds, attentionMask: attentionMask, outputHiddenStates: outputHiddenStates)
    }
}

private final class LTXGemmaModel: Module {
    let config: LTXGemmaModelConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [LTXGemmaDecoderLayer]
    @ModuleInfo(key: "norm") var norm: LTXGemmaRMSNorm

    init(config: LTXGemmaModelConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { layerIndex in
            LTXGemmaDecoderLayer(config: config, layerIndex: layerIndex)
        }
        self._norm.wrappedValue = LTXGemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func forward(
        inputIds: MLXArray,
        attentionMask: MLXArray?,
        outputHiddenStates: Bool
    ) -> (lastHiddenState: MLXArray, hiddenStates: [MLXArray]?) {
        var ids = inputIds
        if ids.dtype != .int32 {
            ids = ids.asType(.int32)
        }

        var h = embedTokens(ids)
        h = h * MLXArray(Float(config.hiddenSize).squareRoot()).asType(h.dtype)

        let mask = makeCausalMask(sequenceLength: h.dim(1), attentionMask: attentionMask, dtype: .float32)
        var allHiddenStates: [MLXArray]? = outputHiddenStates ? [h] : nil

        let evalEvery = Int(ProcessInfo.processInfo.environment["LTX2_GEMMA_EVAL_EVERY"] ?? "1") ?? 1
        for (idx, layer) in layers.enumerated() {
            h = layer(h, mask: mask)
            if outputHiddenStates {
                allHiddenStates?.append(h)
            }
            if evalEvery > 0 && (idx + 1).isMultiple(of: evalEvery) {
                MLX.eval(h)
            }
        }

        h = norm(h)
        return (h, allHiddenStates)
    }
}

private final class LTXGemmaDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: LTXGemmaAttention
    @ModuleInfo(key: "mlp") var mlp: LTXGemmaMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LTXGemmaRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LTXGemmaRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardLayerNorm: LTXGemmaRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardLayerNorm: LTXGemmaRMSNorm

    init(config: LTXGemmaModelConfig, layerIndex: Int) {
        self._selfAttention.wrappedValue = LTXGemmaAttention(config: config, layerIndex: layerIndex)
        self._mlp.wrappedValue = LTXGemmaMLP(config: config)
        self._inputLayerNorm.wrappedValue = LTXGemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = LTXGemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedForwardLayerNorm.wrappedValue = LTXGemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedForwardLayerNorm.wrappedValue = LTXGemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let residualAttn = x
        var h = inputLayerNorm(x)
        h = selfAttention(h, mask: mask)
        h = postAttentionLayerNorm(h)
        h = residualAttn + h

        let residualFF = h
        h = preFeedForwardLayerNorm(h)
        h = mlp(h)
        h = postFeedForwardLayerNorm(h)
        h = residualFF + h
        return h
    }
}

private final class LTXGemmaAttention: Module {
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let numKeyValueGroups: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: LTXGemmaRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: LTXGemmaRMSNorm

    init(config: LTXGemmaModelConfig, layerIndex: Int) {
        self.numAttentionHeads = config.numAttentionHeads
        self.numKeyValueHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.numKeyValueGroups = config.numAttentionHeads / config.numKeyValueHeads
        self.scale = pow(config.queryPreAttnScalar, -0.5)
        let isSliding = ((layerIndex + 1) % config.slidingWindowPattern) != 0
        let ropeBase = isSliding ? config.ropeLocalBaseFreq : config.ropeGlobalBaseFreq
        self.rope = RoPE(
            dimensions: config.headDim,
            traditional: config.ropeTraditional,
            base: ropeBase,
            scale: 1.0
        )

        let hidden = config.hiddenSize
        self._qProj.wrappedValue = Linear(hidden, numAttentionHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hidden, numKeyValueHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hidden, numKeyValueHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numAttentionHeads * headDim, hidden, bias: false)
        self._qNorm.wrappedValue = LTXGemmaRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = LTXGemmaRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        var q = qProj(x).reshaped(batch, seqLen, numAttentionHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, seqLen, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(batch, seqLen, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

        q = qNorm(q)
        k = kNorm(k)
        q = rope(q)
        k = rope(k)

        if numKeyValueHeads != numAttentionHeads {
            k = expandKeyValue(k, repeats: numKeyValueGroups)
            v = expandKeyValue(v, repeats: numKeyValueGroups)
        }

        let attnOut = MLXFast.scaledDotProductAttention(
            queries: q.asType(.float32),
            keys: k.asType(.float32),
            values: v.asType(.float32),
            scale: scale,
            mask: mask
        ).asType(q.dtype)

        let output = attnOut.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numAttentionHeads * headDim)
        return oProj(output)
    }

    private func expandKeyValue(_ x: MLXArray, repeats: Int) -> MLXArray {
        guard repeats > 1 else { return x }
        let batch = x.dim(0)
        let heads = x.dim(1)
        let seq = x.dim(2)
        let dim = x.dim(3)
        let expanded = x.reshaped(batch, heads, 1, seq, dim)
        let tiled = broadcast(expanded, to: [batch, heads, repeats, seq, dim])
        return tiled.reshaped(batch, heads * repeats, seq, dim)
    }
}

private final class LTXGemmaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: LTXGemmaModelConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = geluApproximate(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}

private final class LTXGemmaRMSNorm: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.zeros([dimensions], dtype: .float32)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let x32 = x.asType(.float32)
        let variance = MLX.mean(x32 * x32, axis: -1, keepDims: true)
        let normed = x32 * rsqrt(variance + MLXArray(eps))
        let scaled = normed * (1 + weight.asType(.float32))
        return scaled.asType(dtype)
    }
}

private final class LTXGemmaFeaturesExtractor: Module {
    @ModuleInfo(key: "aggregate_embed") var aggregateEmbed: Linear

    init(inputDim: Int, outputDim: Int) {
        self._aggregateEmbed.wrappedValue = Linear(inputDim, outputDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        aggregateEmbed(x)
    }
}

private final class LTXTextEmbeddingProjectionV2: Module {
    @ModuleInfo(key: "video_aggregate_embed") var videoAggregateEmbed: Linear
    @ModuleInfo(key: "audio_aggregate_embed") var audioAggregateEmbed: Linear

    let embeddingDim: Int
    let videoDim: Int
    let audioDim: Int

    init(inputDim: Int, videoDim: Int, audioDim: Int, embeddingDim: Int) {
        self.embeddingDim = embeddingDim
        self.videoDim = videoDim
        self.audioDim = audioDim
        self._videoAggregateEmbed.wrappedValue = Linear(inputDim, videoDim, bias: true)
        self._audioAggregateEmbed.wrappedValue = Linear(inputDim, audioDim, bias: true)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> (video: MLXArray, audio: MLXArray) {
        let videoScale = Float(sqrt(Double(videoDim) / Double(embeddingDim)))
        let video = videoAggregateEmbed(hiddenStates * MLXArray(videoScale).asType(hiddenStates.dtype))
        MLX.eval(video)

        let audioScale = Float(sqrt(Double(audioDim) / Double(embeddingDim)))
        let audio = audioAggregateEmbed(hiddenStates * MLXArray(audioScale).asType(hiddenStates.dtype))
        MLX.eval(audio)
        return (video, audio)
    }
}

private final class LTXGemmaFeaturesExtractorV2: Module {
    @ModuleInfo(key: "text_embedding_projection") var textEmbeddingProjection: LTXTextEmbeddingProjectionV2
    @ModuleInfo(key: "video_embeddings_connector") var videoConnector: LTXEmbeddings1DConnector
    @ModuleInfo(key: "audio_embeddings_connector") var audioConnector: LTXEmbeddings1DConnector

    init(
        captionChannels: Int,
        numGemmaLayers: Int,
        videoDim: Int,
        audioDim: Int,
        numHeads: Int,
        videoHeadDim: Int,
        audioHeadDim: Int,
        numConnectorLayers: Int,
        numRegisters: Int
    ) {
        let inputDim = captionChannels * numGemmaLayers
        self._textEmbeddingProjection.wrappedValue = LTXTextEmbeddingProjectionV2(
            inputDim: inputDim,
            videoDim: videoDim,
            audioDim: audioDim,
            embeddingDim: captionChannels
        )
        self._videoConnector.wrappedValue = LTXEmbeddings1DConnector(
            dim: videoDim,
            numHeads: numHeads,
            headDim: videoHeadDim,
            numLayers: numConnectorLayers,
            numLearnableRegisters: numRegisters,
            positionalEmbeddingTheta: 10_000.0,
            positionalEmbeddingMaxPos: [4_096],
            applyGatedAttention: true
        )
        self._audioConnector.wrappedValue = LTXEmbeddings1DConnector(
            dim: audioDim,
            numHeads: numHeads,
            headDim: audioHeadDim,
            numLayers: numConnectorLayers,
            numLearnableRegisters: numRegisters,
            positionalEmbeddingTheta: 10_000.0,
            positionalEmbeddingMaxPos: [4_096],
            applyGatedAttention: true
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray?
    ) -> (projectedVideo: MLXArray, videoEmbeddings: MLXArray, audioEmbeddings: MLXArray) {
        let projected = textEmbeddingProjection(hiddenStates)
        let videoEmbeddings = videoConnector(projected.video, attentionMask: attentionMask).0
        let audioEmbeddings = audioConnector(projected.audio, attentionMask: attentionMask).0
        return (projected.video, videoEmbeddings, audioEmbeddings)
    }
}

private final class LTXEmbeddings1DConnector: Module {
    let dim: Int
    let numHeads: Int
    let headDim: Int
    let numLearnableRegisters: Int
    let positionalEmbeddingTheta: Float
    let positionalEmbeddingMaxPos: [Int]
    let eps: Float

    @ModuleInfo(key: "transformer_1d_blocks") var transformer1DBlocks: [LTXConnectorTransformerBlock]
    @ModuleInfo(key: "learnable_registers") var learnableRegisters: MLXArray

    init(
        dim: Int,
        numHeads: Int,
        headDim: Int,
        numLayers: Int,
        numLearnableRegisters: Int,
        positionalEmbeddingTheta: Float,
        positionalEmbeddingMaxPos: [Int],
        eps: Float = 1e-6,
        applyGatedAttention: Bool = false
    ) {
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = headDim
        self.numLearnableRegisters = numLearnableRegisters
        self.positionalEmbeddingTheta = positionalEmbeddingTheta
        self.positionalEmbeddingMaxPos = positionalEmbeddingMaxPos
        self.eps = eps

        self._transformer1DBlocks.wrappedValue = (0..<numLayers).map { _ in
            LTXConnectorTransformerBlock(
                dim: dim,
                numHeads: numHeads,
                headDim: headDim,
                eps: eps,
                applyGatedAttention: applyGatedAttention
            )
        }
        self._learnableRegisters.wrappedValue = MLX.zeros([max(0, numLearnableRegisters), dim], dtype: .float32)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        var states = hiddenStates
        var mask = attentionMask

        if numLearnableRegisters > 0, let attentionMask {
            let adjusted = replacePaddedWithRegisters(hiddenStates: states, attentionMask: attentionMask)
            states = adjusted.hiddenStates
            mask = nil
        }

        let pe = precomputeFreqsCis(seqLen: states.dim(1), dtype: states.dtype)
        for block in transformer1DBlocks {
            states = block(states, attentionMask: mask, pe: pe)
        }
        states = rmsNormNoWeight(states, eps: eps)
        return (states, mask)
    }

    private func precomputeFreqsCis(seqLen: Int, dtype: DType) -> (cos: MLXArray, sin: MLXArray) {
        let halfHeadDim = max(1, headDim / 2)
        let maxPos = Float(positionalEmbeddingMaxPos.first ?? 4_096)
        let maxPosCount = max(1, positionalEmbeddingMaxPos.count)
        let indexCount = max(1, (numHeads * headDim) / (2 * maxPosCount))

        var indices = [Double](repeating: 0.0, count: indexCount)
        if indexCount == 1 {
            indices[0] = Double.pi / 2.0
        } else {
            for i in 0..<indexCount {
                let fraction = Double(i) / Double(indexCount - 1)
                indices[i] = pow(Double(positionalEmbeddingTheta), fraction) * (Double.pi / 2.0)
            }
        }

        let total = numHeads * seqLen * halfHeadDim
        var cosValues = [Float](repeating: 0.0, count: total)
        var sinValues = [Float](repeating: 0.0, count: total)

        for head in 0..<numHeads {
            for tokenIndex in 0..<seqLen {
                let fractionalPosition = (Float(tokenIndex) / maxPos) * 2.0 - 1.0
                for i in 0..<halfHeadDim {
                    let featureIndex = min(indices.count - 1, head * halfHeadDim + i)
                    let phase = Double(fractionalPosition) * indices[featureIndex]
                    let flatIndex = (head * seqLen + tokenIndex) * halfHeadDim + i
                    cosValues[flatIndex] = Float(cos(phase))
                    sinValues[flatIndex] = Float(sin(phase))
                }
            }
        }

        let cosArray = MLXArray(cosValues).reshaped(1, numHeads, seqLen, halfHeadDim).asType(dtype)
        let sinArray = MLXArray(sinValues).reshaped(1, numHeads, seqLen, halfHeadDim).asType(dtype)
        return (cosArray, sinArray)
    }

    private func replacePaddedWithRegisters(
        hiddenStates: MLXArray,
        attentionMask: MLXArray
    ) -> (hiddenStates: MLXArray, attentionMask: MLXArray) {
        let batch = hiddenStates.dim(0)
        let seqLen = hiddenStates.dim(1)
        let dtype = hiddenStates.dtype

        let reshapedMask = attentionMask.reshaped(batch, seqLen)
        let binaryMask: MLXArray
        if attentionMask.ndim == 2 {
            binaryMask = (reshapedMask .> MLXArray(0).asType(reshapedMask.dtype)).asType(.int32)
        } else {
            binaryMask = (reshapedMask .>= MLXArray(-9_000.0).asType(reshapedMask.dtype)).asType(.int32)
        }
        let binaryValues = binaryMask.asArray(Int32.self)

        var rows: [MLXArray] = []
        rows.reserveCapacity(batch)

        for b in 0..<batch {
            let offset = b * seqLen
            let validCount = binaryValues[offset..<(offset + seqLen)].reduce(0) { partial, value in
                partial + (value != 0 ? 1 : 0)
            }
            let padCount = max(0, seqLen - validCount)

            let hiddenRow = hiddenStates[b, 0..., 0...]
            let validTokens: MLXArray
            if validCount > 0 {
                validTokens = hiddenRow[(seqLen - validCount)..<seqLen, 0...]
            } else {
                validTokens = MLX.zeros([0, dim], dtype: dtype)
            }

            let adjustedTokens: MLXArray
            if padCount > 0 {
                let rightPadding = MLX.zeros([padCount, dim], dtype: dtype)
                adjustedTokens = MLX.concatenated([validTokens, rightPadding], axis: 0)
            } else {
                adjustedTokens = validTokens
            }

            let validMaskValues = Array(repeating: Int32(1), count: validCount) + Array(repeating: Int32(0), count: padCount)
            let validMask = MLXArray(validMaskValues).reshaped(seqLen, 1).asType(dtype)
            let registers = repeatedRegisters(length: seqLen, dtype: dtype)
            let combined = validMask * adjustedTokens + (MLXArray(1.0).asType(dtype) - validMask) * registers
            rows.append(combined)
        }

        let replaced = MLX.stacked(rows, axis: 0)
        let clearedMask = MLX.zeros(attentionMask.shape, dtype: attentionMask.dtype)
        return (replaced, clearedMask)
    }

    private func repeatedRegisters(length: Int, dtype: DType) -> MLXArray {
        guard numLearnableRegisters > 0, length > 0 else {
            return MLX.zeros([max(0, length), dim], dtype: dtype)
        }

        let repeats = Int(ceil(Double(length) / Double(numLearnableRegisters)))
        var chunks: [MLXArray] = []
        chunks.reserveCapacity(max(1, repeats))
        for _ in 0..<max(1, repeats) {
            chunks.append(learnableRegisters.asType(dtype))
        }

        let tiled = chunks.count == 1 ? chunks[0] : MLX.concatenated(chunks, axis: 0)
        if tiled.dim(0) == length {
            return tiled
        }
        return tiled[0..<length, 0...]
    }
}

private final class LTXConnectorTransformerBlock: Module {
    @ModuleInfo(key: "attn1") var attn1: LTXConnectorAttention
    @ModuleInfo(key: "ff") var ff: LTXConnectorFeedForward
    let eps: Float

    init(
        dim: Int,
        numHeads: Int,
        headDim: Int,
        eps: Float = 1e-6,
        applyGatedAttention: Bool = false
    ) {
        self.eps = eps
        self._attn1.wrappedValue = LTXConnectorAttention(
            dim: dim,
            numHeads: numHeads,
            headDim: headDim,
            eps: eps,
            applyGatedAttention: applyGatedAttention
        )
        self._ff.wrappedValue = LTXConnectorFeedForward(dim: dim, mult: 4)
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXArray?,
        pe: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        var h = x
        let norm1 = rmsNormNoWeight(h, eps: eps)
        h = h + attn1(norm1, attentionMask: attentionMask, pe: pe)

        let norm2 = rmsNormNoWeight(h, eps: eps)
        h = h + ff(norm2)
        return h
    }
}

private final class LTXConnectorAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "to_gate_logits") var toGateLogits: Linear?
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(dim: Int, numHeads: Int, headDim: Int, eps: Float = 1e-6, applyGatedAttention: Bool = false) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.scale = 1.0 / Float(headDim).squareRoot()

        let innerDim = numHeads * headDim
        self._toQ.wrappedValue = Linear(dim, innerDim, bias: true)
        self._toK.wrappedValue = Linear(dim, innerDim, bias: true)
        self._toV.wrappedValue = Linear(dim, innerDim, bias: true)
        self._toOut.wrappedValue = Linear(innerDim, dim, bias: true)
        self._toGateLogits.wrappedValue = applyGatedAttention ? Linear(dim, numHeads, bias: true) : nil
        self._qNorm.wrappedValue = RMSNorm(dimensions: innerDim, eps: eps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: innerDim, eps: eps)
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXArray?,
        pe: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)
        let innerDim = numHeads * headDim

        var q = qNorm(toQ(x)).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kNorm(toK(x)).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = toV(x).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

        q = applySplitRoPE(q, cosFreq: pe.cos, sinFreq: pe.sin)
        k = applySplitRoPE(k, cosFreq: pe.cos, sinFreq: pe.sin)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = attentionMask
            .map { .array($0.asType(.float32)) } ?? .none
        var out = MLXFast.scaledDotProductAttention(
            queries: q.asType(.float32),
            keys: k.asType(.float32),
            values: v.asType(.float32),
            scale: scale,
            mask: maskMode
        ).asType(q.dtype)

        if let toGateLogits {
            let gate = MLXArray(2.0).asType(out.dtype) * MLX.sigmoid(toGateLogits(x).asType(out.dtype))
            out = out * gate.transposed(0, 2, 1).expandedDimensions(axis: 3)
        }

        let merged = out.transposed(0, 2, 1, 3).reshaped(batch, seqLen, innerDim)
        return toOut(merged)
    }

    private func applySplitRoPE(
        _ x: MLXArray,
        cosFreq: MLXArray,
        sinFreq: MLXArray
    ) -> MLXArray {
        let dtype = x.dtype
        let x32 = x.asType(.float32)
        let cos32 = cosFreq.asType(.float32)
        let sin32 = sinFreq.asType(.float32)
        let halfDim = x32.dim(3) / 2

        let x1 = x32[0..., 0..., 0..., 0..<halfDim]
        let x2 = x32[0..., 0..., 0..., halfDim...]
        let out1 = x1 * cos32 - x2 * sin32
        let out2 = x2 * cos32 + x1 * sin32
        return MLX.concatenated([out1, out2], axis: 3).asType(dtype)
    }
}

private final class LTXConnectorFeedForward: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(dim: Int, mult: Int = 4) {
        let inner = dim * mult
        self._projIn.wrappedValue = Linear(dim, inner, bias: true)
        self._projOut.wrappedValue = Linear(inner, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = geluApproximate(projIn(x))
        return projOut(hidden)
    }
}

// MARK: - Utilities

private func findLTXTransformerWeights(modelRoot: URL) throws -> URL {
    let fm = FileManager.default
    let distilled = modelRoot.appendingPathComponent("ltx-2-19b-distilled.safetensors", isDirectory: false)
    if fm.fileExists(atPath: distilled.path) {
        return distilled
    }

    let model19B = modelRoot.appendingPathComponent("ltx-2-19b.safetensors", isDirectory: false)
    if fm.fileExists(atPath: model19B.path) {
        return model19B
    }

    let entries = try fm.contentsOfDirectoryResolvingSymlinks(
        at: modelRoot,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
    )
    let matches = entries
        .filter { $0.pathExtension == "safetensors" && $0.lastPathComponent.hasPrefix("ltx-2-19") }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    if let first = matches.first {
        return first
    }
    throw LTXGemmaTextEncoderError.missingTransformerWeights(modelRoot)
}

private func mapConnectorWeights(
    arrays: [String: MLXArray],
    prefix: String,
    dtype: DType
) -> [(String, MLXArray)] {
    var updates: [(String, MLXArray)] = []
    updates.reserveCapacity(arrays.count)

    for (key, value) in arrays {
        guard key.hasPrefix(prefix) else { continue }

        var mapped = String(key.dropFirst(prefix.count))
        mapped = mapped.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
        mapped = mapped.replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
        mapped = mapped.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")

        let casted: MLXArray
        if value.dtype.isFloatingPoint && value.dtype != dtype {
            casted = value.asType(dtype)
        } else {
            casted = value
        }
        updates.append((mapped, casted))
    }

    updates.sort { lhs, rhs in lhs.0 < rhs.0 }
    return updates
}

func mapLTX23TextConnectorWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    guard key.hasPrefix("connector.") else {
        return []
    }

    var mapped = String(key.dropFirst("connector.".count))
    mapped = mapped.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
    mapped = mapped.replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
    mapped = mapped.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")

    let casted: MLXArray
    if value.dtype.isFloatingPoint && value.dtype != dtype {
        casted = value.asType(dtype)
    } else {
        casted = value
    }
    return [(mapped, casted)]
}

func mapLTX25TextConnectorWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    let prefix = "model.diffusion_model."
    guard key.hasPrefix(prefix) else {
        return []
    }

    var mapped = String(key.dropFirst(prefix.count))
    guard mapped.hasPrefix("video_embeddings_connector.")
            || mapped.hasPrefix("audio_embeddings_connector.") else {
        return []
    }
    mapped = mapped.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
    mapped = mapped.replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
    mapped = mapped.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")

    let casted = value.dtype.isFloatingPoint && value.dtype != dtype
        ? value.asType(dtype)
        : value
    return [(mapped, casted)]
}

func mapGemmaLanguageWeightKey(_ key: String) -> String {
    guard key.hasPrefix("language_model.") else {
        return key
    }
    return String(key.dropFirst("language_model.".count))
}

func gemmaIndexContainsQuantizedWeights(indexURL: URL) -> Bool {
    guard let data = try? Data(contentsOf: indexURL),
          let index = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: data) else {
        return false
    }
    return index.weightMap.keys.contains { $0.hasSuffix(".scales") }
}

private func rmsNormNoWeight(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let dtype = x.dtype
    let x32 = x.asType(.float32)
    let variance = MLX.mean(x32 * x32, axis: -1, keepDims: true)
    let normalized = x32 * rsqrt(variance + MLXArray(eps))
    return normalized.asType(dtype)
}

private func makeCausalMask(
    sequenceLength: Int,
    attentionMask: MLXArray?,
    dtype: DType
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    guard let attentionMask else {
        return .causal
    }

    let batch = attentionMask.dim(0)
    let idx = MLXArray(0..<sequenceLength).asType(.int32)
    let rows = idx.reshaped(sequenceLength, 1)
    let cols = idx.reshaped(1, sequenceLength)
    var causal = (cols .<= rows).asType(dtype).reshaped(1, 1, sequenceLength, sequenceLength)
    causal = broadcast(causal, to: [batch, 1, sequenceLength, sequenceLength])

    var keyMask = attentionMask
    if keyMask.dtype != .int32 {
        keyMask = keyMask.asType(.int32)
    }
    keyMask = keyMask.asType(dtype).reshaped(batch, 1, 1, sequenceLength)
    let combined = causal * keyMask

    let zeros = MLX.zeros(combined.shape, dtype: dtype)
    let negInf = MLXArray(-1e9).asType(dtype)
    let additive = MLX.where(combined .> MLXArray(0).asType(dtype), zeros, zeros + negInf)
    return .array(additive)
}

private func normalizeAndConcatHiddenStates(
    hiddenStates: [MLXArray],
    attentionMask: MLXArray
) -> MLXArray {
    guard !hiddenStates.isEmpty else {
        return MLXArray.zeros([attentionMask.dim(0), attentionMask.dim(1), 0], dtype: .bfloat16)
    }

    let batch = hiddenStates[0].dim(0)
    let seqLen = hiddenStates[0].dim(1)
    let hidden = hiddenStates[0].dim(2)
    let layerCount = hiddenStates.count

    let maskArray = attentionMask.asType(.int32).asArray(Int32.self)
    var rows: [MLXArray] = []
    rows.reserveCapacity(batch)

    for b in 0..<batch {
        let offset = b * seqLen
        let valid = maskArray[offset..<(offset + seqLen)].reduce(0) { $0 + (Int($1) != 0 ? 1 : 0) }
        let start = max(0, seqLen - valid)

        var perLayer: [MLXArray] = []
        perLayer.reserveCapacity(layerCount)
        for state in hiddenStates {
            perLayer.append(state[b, 0..., 0...].asType(.float32))
        }

        let stacked = MLX.stacked(perLayer, axis: -1)
        let validSlice = valid > 0 ? stacked[start..<seqLen, 0..., 0...] : MLX.zeros([0, hidden, layerCount], dtype: .float32)

        let normalizedValid: MLXArray
        if valid > 0 {
            let mean = MLX.mean(validSlice, axes: [0, 1], keepDims: true)
            let minValue = MLX.min(validSlice, axes: [0, 1], keepDims: true)
            let maxValue = MLX.max(validSlice, axes: [0, 1], keepDims: true)
            let range = maxValue - minValue
            normalizedValid = (MLXArray(8.0) * (validSlice - mean)) / (range + MLXArray(1e-6))
        } else {
            normalizedValid = MLX.zeros([0, hidden, layerCount], dtype: .float32)
        }

        let flattenedValid = normalizedValid.reshaped(max(0, valid), hidden * layerCount)
        let padded: MLXArray
        if valid < seqLen {
            let leftPad = MLX.zeros([seqLen - valid, hidden * layerCount], dtype: .float32)
            padded = MLX.concatenated([leftPad, flattenedValid], axis: 0)
        } else {
            padded = flattenedValid
        }
        rows.append(padded)
    }

    return MLX.stacked(rows, axis: 0).asType(.bfloat16)
}

func normalizeAndConcatHiddenStatesV2(
    hiddenStates: [MLXArray],
    attentionMask: MLXArray,
    eps: Float = 1e-6,
    dtype: DType = .bfloat16
) -> MLXArray {
    guard !hiddenStates.isEmpty else {
        return MLXArray.zeros([attentionMask.dim(0), attentionMask.dim(1), 0], dtype: dtype)
    }

    let batch = hiddenStates[0].dim(0)
    let seqLen = hiddenStates[0].dim(1)
    let hidden = hiddenStates[0].dim(2)
    let layerCount = hiddenStates.count
    let stacked = MLX.stacked(hiddenStates.map { $0.asType(.float32) }, axis: -1)
    let variance = MLX.mean(stacked * stacked, axis: 2, keepDims: true)
    let normalized = stacked * rsqrt(variance + MLXArray(eps))
    var flattened = normalized.reshaped(batch, seqLen, hidden * layerCount)
    let mask = attentionMask.asType(flattened.dtype).reshaped(batch, seqLen, 1)
    flattened = flattened * mask
    return flattened.asType(dtype)
}

func mapGemmaLanguageWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    guard key.hasPrefix("language_model.") || key.hasPrefix("model.") || key.hasPrefix("lm_head.") else {
        return []
    }

    let stripped = mapGemmaLanguageWeightKey(key)

    // Ignore logits head and other non-backbone keys.
    if stripped.hasPrefix("lm_head.") {
        return []
    }
    if stripped.contains(".rotary_emb.") {
        return []
    }

    let isSupported: Bool = {
        if stripped == "model.embed_tokens.weight" || stripped == "model.norm.weight" {
            return true
        }
        guard stripped.hasPrefix("model.layers.") else { return false }
        if stripped.contains(".self_attn.q_proj.weight")
            || stripped.contains(".self_attn.k_proj.weight")
            || stripped.contains(".self_attn.v_proj.weight")
            || stripped.contains(".self_attn.o_proj.weight")
            || stripped.contains(".self_attn.q_norm.weight")
            || stripped.contains(".self_attn.k_norm.weight")
            || stripped.contains(".mlp.gate_proj.weight")
            || stripped.contains(".mlp.up_proj.weight")
            || stripped.contains(".mlp.down_proj.weight")
            || stripped.contains(".input_layernorm.weight")
            || stripped.contains(".post_attention_layernorm.weight")
            || stripped.contains(".pre_feedforward_layernorm.weight")
            || stripped.contains(".post_feedforward_layernorm.weight")
        {
            return true
        }
        return false
    }()

    guard isSupported else {
        return []
    }

    let casted: MLXArray
    if value.dtype.isFloatingPoint && value.dtype != dtype {
        casted = value.asType(dtype)
    } else {
        casted = value
    }
    return [(stripped, casted)]
}
