import MLX
import MLXNN

public final class MiniMaxMusic3LanguageModel: Module {
    public let configuration: MiniMaxMusic3LanguageConfiguration

    @ModuleInfo(key: "model") var model: QwenEncoder
    @ModuleInfo(key: "lm_head") var languageModelHead: Linear

    public private(set) var usesCompactSemanticHead = false

    public init(configuration: MiniMaxMusic3LanguageConfiguration) {
        self.configuration = configuration
        let qwenConfiguration = QwenTextEncoderConfiguration(
            vocabSize: configuration.vocabSize,
            hiddenSize: configuration.hiddenSize,
            numHiddenLayers: configuration.numHiddenLayers,
            numAttentionHeads: configuration.numAttentionHeads,
            numKeyValueHeads: configuration.numKeyValueHeads,
            intermediateSize: configuration.intermediateSize,
            ropeTheta: configuration.ropeParameters.ropeTheta,
            maxPositionEmbeddings: configuration.maxPositionEmbeddings,
            rmsNormEps: configuration.rmsNormEps,
            headDim: configuration.headDim,
            cachedDecodeAttentionMode: .native
        )
        self._model.wrappedValue = QwenEncoder(configuration: qwenConfiguration)
        self._languageModelHead.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.vocabSize,
            bias: false
        )
    }

    public func makeCache(capacity: Int? = nil) -> [KVCache] {
        (0..<configuration.numHiddenLayers).map { _ in
            if let capacity {
                return KVCacheStatic(capacity: capacity)
            }
            return KVCacheSimple(step: 256)
        }
    }

    public func embed(tokenIDs: MLXArray) -> MLXArray {
        model.embed(inputIds: tokenIDs)
    }

    public func hidden(
        embeddings: MLXArray,
        cache: [KVCache],
        lastPositionOnly: Bool = true
    ) -> MLXArray {
        model.forwardCausalHidden(
            embeddings: embeddings,
            cache: cache,
            lastPositionOnly: lastPositionOnly
        )
    }

    public func logits(_ hidden: MLXArray) -> MLXArray {
        languageModelHead(hidden)
    }

    public func prepareFusedProjections() {
        model.prepareFusedProjections()
        MLX.Memory.clearCache()
    }

    /// The autoregressive sampler can emit only EOS or one of the checkpoint's
    /// semantic audio tokens. Projecting all 200,000 vocabulary rows streams a
    /// 1.64 GB BF16 head on every 25 Hz frame, so the optimized runtime retains
    /// only the 16,385 reachable rows after checkpoint loading.
    public func prepareCompactSemanticHead() {
        guard !usesCompactSemanticHead else { return }
        let end = languageModelHead.weight[
            MiniMaxMusic3Prompt.audioEndTokenID..<(MiniMaxMusic3Prompt.audioEndTokenID + 1),
            0...
        ]
        let semanticEnd = MiniMaxMusic3Prompt.audioCodeOffset
            + MiniMaxMusic3Prompt.semanticVocabularySize
        let semantic = languageModelHead.weight[
            MiniMaxMusic3Prompt.audioCodeOffset..<semanticEnd,
            0...
        ]
        let compact = MLX.concatenated([end, semantic], axis: 0)
        MLX.eval(compact)
        update(modules: ModuleChildren.unflattened([
            ("lm_head", Linear(weight: compact, bias: nil)),
        ]))
        usesCompactSemanticHead = true
        MLX.Memory.clearCache()
    }
}
