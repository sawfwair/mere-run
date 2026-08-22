import Foundation
import MLX
import MLXFast
import MLXNN

final class LFM2DSparkAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

    private let headCount: Int
    private let keyValueHeadCount: Int
    private let headDimensions: Int
    private let scale: Float
    private let rope: RoPE

    init(config: LFM2DSparkConfig) {
        headCount = config.attentionHeadCount
        keyValueHeadCount = config.keyValueHeadCount
        headDimensions = config.headDimensions
        scale = 1 / Foundation.sqrt(Float(max(1, config.headDimensions)))
        rope = RoPE(
            dimensions: config.headDimensions,
            traditional: !config.ropeUsesNeoXLayout,
            base: config.ropeTheta
        )
        self._queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.attentionHeadCount * config.headDimensions,
            bias: false
        )
        self._keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.keyValueHeadCount * config.headDimensions,
            bias: false
        )
        self._valueProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.keyValueHeadCount * config.headDimensions,
            bias: false
        )
        self._outputProjection.wrappedValue = Linear(
            config.attentionHeadCount * config.headDimensions,
            config.hiddenSize,
            bias: false
        )
        self._queryNorm.wrappedValue = RMSNorm(
            dimensions: config.headDimensions,
            eps: config.normEpsilon
        )
        self._keyNorm.wrappedValue = RMSNorm(
            dimensions: config.headDimensions,
            eps: config.normEpsilon
        )
        super.init()
    }

    func appendContext(_ context: MLXArray, cache: Gemma4AttentionCache) {
        let batch = context.dim(0)
        let length = context.dim(1)
        let keys = rope(
            keyNorm(
                keyProjection(context).reshaped(
                    batch,
                    length,
                    keyValueHeadCount,
                    headDimensions
                )
            ).transposed(0, 2, 1, 3),
            offset: cache.offset
        )
        let values = valueProjection(context)
            .reshaped(batch, length, keyValueHeadCount, headDimensions)
            .transposed(0, 2, 1, 3)
        cache.append(keys: keys, values: values)
    }

    func callAsFunction(_ hidden: MLXArray, cache: Gemma4AttentionCache) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let offset = cache.offset
        let queries = rope(
            queryNorm(
                queryProjection(hidden).reshaped(batch, length, headCount, headDimensions)
            ).transposed(0, 2, 1, 3),
            offset: offset
        )
        let keys = rope(
            keyNorm(
                keyProjection(hidden).reshaped(
                    batch,
                    length,
                    keyValueHeadCount,
                    headDimensions
                )
            ).transposed(0, 2, 1, 3),
            offset: offset
        )
        let values = valueProjection(hidden)
            .reshaped(batch, length, keyValueHeadCount, headDimensions)
            .transposed(0, 2, 1, 3)
        guard let state = cache.attentionState(appending: keys, values: values) else {
            preconditionFailure("LFM2 DSpark attention cache did not retain appended state.")
        }

        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: state.0,
            values: state.1,
            scale: scale,
            mask: .none
        )
        return outputProjection(
            attended.transposed(0, 2, 1, 3)
                .reshaped(batch, length, headCount * headDimensions)
        )
    }
}

final class LFM2DSparkMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProjection: Linear
    @ModuleInfo(key: "up_proj") var upProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(config: LFM2DSparkConfig) {
        self._gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        self._upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        self._downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        downProjection(MLXNN.silu(gateProjection(hidden)) * upProjection(hidden))
    }
}

final class LFM2DSparkLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LFM2DSparkAttention
    @ModuleInfo(key: "mlp") var mlp: LFM2DSparkMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    init(config: LFM2DSparkConfig) {
        self._attention.wrappedValue = LFM2DSparkAttention(config: config)
        self._mlp.wrappedValue = LFM2DSparkMLP(config: config)
        self._inputNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.normEpsilon
        )
        self._postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.normEpsilon
        )
        super.init()
    }

    func appendContext(_ context: MLXArray, cache: Gemma4AttentionCache) {
        attention.appendContext(context, cache: cache)
    }

    func callAsFunction(_ hidden: MLXArray, cache: Gemma4AttentionCache) -> MLXArray {
        let attended = hidden + attention(inputNorm(hidden), cache: cache)
        return attended + mlp(postAttentionNorm(attended))
    }
}

final class LFM2DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var tokenEmbedding: Embedding
    @ModuleInfo(key: "markov_w2") var outputProjection: Linear

    init(config: LFM2DSparkConfig) {
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.markovRank
        )
        self._outputProjection.wrappedValue = Linear(
            config.markovRank,
            config.vocabularySize,
            bias: false
        )
        super.init()
    }

    func bias(previousToken: Int) -> MLXArray {
        let input = MLXArray([Int32(previousToken)]).reshaped(1, 1)
        return outputProjection(tokenEmbedding(input))[0, 0, 0...]
    }

    func bias(previousToken: MLXArray) -> MLXArray {
        let input = previousToken.asType(.int32).reshaped(1, 1)
        return outputProjection(tokenEmbedding(input))[0, 0, 0...]
    }
}

final class LFM2DSparkModel: Module {
    @ModuleInfo(key: "fc") var featureProjection: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [LFM2DSparkLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "markov_head") var markovHead: LFM2DSparkMarkovHead

    let config: LFM2DSparkConfig

    init(config: LFM2DSparkConfig) {
        self.config = config
        self._featureProjection.wrappedValue = Linear(
            config.hiddenSize * config.features.targetLayerIDs.count,
            config.hiddenSize,
            bias: false
        )
        self._hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.normEpsilon
        )
        self._layers.wrappedValue = (0..<config.hiddenLayerCount).map { _ in
            LFM2DSparkLayer(config: config)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.normEpsilon
        )
        self._markovHead.wrappedValue = LFM2DSparkMarkovHead(config: config)
        super.init()
    }

    func combineTargetHiddenStates(_ states: [Int: MLXArray]) -> MLXArray {
        let ordered = config.features.targetLayerIDs.map { layerIndex -> MLXArray in
            guard let hidden = states[layerIndex] else {
                preconditionFailure("LFM2 DSpark target capture is missing layer \(layerIndex).")
            }
            return hidden
        }
        return hiddenNorm(featureProjection(MLX.concatenated(ordered, axis: -1)))
    }

    func appendTargetContext(
        _ combinedContext: MLXArray,
        cache: [Gemma4AttentionCache]
    ) {
        precondition(cache.count == layers.count)
        for (index, layer) in layers.enumerated() {
            layer.appendContext(combinedContext, cache: cache[index])
        }
    }

    func draftBaseHidden(
        anchorToken: Int,
        proposalCount: Int,
        target: LFM2Model,
        cache: [Gemma4AttentionCache]
    ) -> MLXArray {
        precondition(proposalCount > 0 && proposalCount <= config.blockSize)
        let inputTokens = [Int32(anchorToken)]
            + Array(repeating: Int32(config.features.maskTokenID), count: proposalCount - 1)
        let input = MLXArray(inputTokens).reshaped(1, proposalCount)
        var hidden = target.embeddings(for: input)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: cache[index])
        }
        return norm(hidden)
    }

    func draftBaseHidden(
        anchorToken: MLXArray,
        proposalCount: Int,
        target: LFM2Model,
        cache: [Gemma4AttentionCache]
    ) -> MLXArray {
        precondition(proposalCount > 0 && proposalCount <= config.blockSize)
        let masks = MLXArray(
            Array(repeating: Int32(config.features.maskTokenID), count: proposalCount - 1)
        )
        let input = MLX.concatenated(
            [anchorToken.asType(.int32).reshaped(1), masks],
            axis: 0
        ).reshaped(1, proposalCount)
        var hidden = target.embeddings(for: input)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: cache[index])
        }
        return norm(hidden)
    }

    func proposalTokenArray(
        baseHidden: MLXArray,
        anchorToken: MLXArray,
        target: LFM2Model,
        select: (MLXArray, Int) -> MLXArray
    ) -> MLXArray {
        let baseLogits = target.logits(from: baseHidden)
        var previous = anchorToken.asType(.int32).reshaped(1)
        var tokens: [MLXArray] = []
        tokens.reserveCapacity(baseHidden.dim(1))
        for index in 0..<baseHidden.dim(1) {
            let logits = baseLogits[0, index, 0...] + markovHead.bias(previousToken: previous)
            let token = select(logits, index).asType(.int32).reshaped(1)
            tokens.append(token)
            previous = token
        }
        return MLX.concatenated(tokens, axis: 0).reshaped(1, tokens.count)
    }

    func proposalTokens(
        baseHidden: MLXArray,
        anchorToken: Int,
        target: LFM2Model,
        sample: (MLXArray, Int) -> Int
    ) -> [Int] {
        let baseLogits = target.logits(from: baseHidden)
        var previous = anchorToken
        var tokens: [Int] = []
        for index in 0..<baseHidden.dim(1) {
            let logits = baseLogits[0, index, 0...] + markovHead.bias(previousToken: previous)
            let token = sample(logits, index)
            tokens.append(token)
            previous = token
        }
        return tokens
    }

    func makeCache() -> [Gemma4AttentionCache] {
        layers.map { _ in Gemma4FullKVCache() }
    }
}
