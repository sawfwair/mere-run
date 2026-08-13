import MLX
import MLXNN

public final class MiniMaxMusic3LanguageModel: Module {
    public let configuration: MiniMaxMusic3LanguageConfiguration

    @ModuleInfo(key: "model") var model: QwenEncoder
    @ModuleInfo(key: "lm_head") var languageModelHead: Linear

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

    public func makeCache() -> [KVCache] {
        (0..<configuration.numHiddenLayers).map { _ in KVCacheSimple(step: 256) }
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
}
