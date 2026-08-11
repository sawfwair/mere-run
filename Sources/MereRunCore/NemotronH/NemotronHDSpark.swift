import Foundation
import MLX
import MLXFast
import MLXNN

final class NemotronHDSparkAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "attention_sink_bias") var attentionSinkBias: MLXArray

    private let headCount: Int
    private let keyValueHeadCount: Int
    private let headDimensions: Int
    private let slidingWindow: Int
    private let scale: Float
    private let rope: RoPE

    init(config: NemotronHDSparkConfig) {
        headCount = config.numAttentionHeads
        keyValueHeadCount = config.numKeyValueHeads
        headDimensions = config.headDim
        slidingWindow = config.speculation.slidingWindow
        scale = 1 / Foundation.sqrt(Float(config.headDim))
        rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeTheta
        )
        self._queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.numAttentionHeads * config.headDim,
            bias: false
        )
        self._keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.numKeyValueHeads * config.headDim,
            bias: false
        )
        self._valueProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.numKeyValueHeads * config.headDim,
            bias: false
        )
        self._outputProjection.wrappedValue = Linear(
            config.numAttentionHeads * config.headDim,
            config.hiddenSize,
            bias: false
        )
        self._queryNorm.wrappedValue = RMSNorm(
            dimensions: config.headDim,
            eps: config.rmsNormEps
        )
        self._keyNorm.wrappedValue = RMSNorm(
            dimensions: config.headDim,
            eps: config.rmsNormEps
        )
        self._attentionSinkBias.wrappedValue = MLXArray.ones([config.numAttentionHeads])
        super.init()
    }

    func appendContext(_ context: MLXArray, cache: Gemma4AttentionCache) {
        let batch = context.dim(0)
        let length = context.dim(1)
        let offset = cache.offset
        let keys = rope(
            keyNorm(
                keyProjection(context).reshaped(
                    batch,
                    length,
                    keyValueHeadCount,
                    headDimensions
                )
            ).transposed(0, 2, 1, 3),
            offset: offset
        )
        let values = valueProjection(context)
            .reshaped(batch, length, keyValueHeadCount, headDimensions)
            .transposed(0, 2, 1, 3)
        cache.append(keys: keys, values: values)
    }

    func callAsFunction(_ x: MLXArray, cache: Gemma4AttentionCache) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let offset = cache.offset
        let queries = rope(
            queryNorm(
                queryProjection(x).reshaped(batch, length, headCount, headDimensions)
            ).transposed(0, 2, 1, 3),
            offset: offset
        )
        let keys = rope(
            keyNorm(
                keyProjection(x).reshaped(
                    batch,
                    length,
                    keyValueHeadCount,
                    headDimensions
                )
            ).transposed(0, 2, 1, 3),
            offset: offset
        )
        let values = valueProjection(x)
            .reshaped(batch, length, keyValueHeadCount, headDimensions)
            .transposed(0, 2, 1, 3)
        let state = cache.attentionState(appending: keys, values: values)!
        let keyLength = state.0.dim(2)
        let keyStart = max(0, offset + length - keyLength)
        let queryPositions = MLXArray(
            Int32(offset)..<Int32(offset + length)
        ).reshaped(length, 1)
        let keyIndices = MLXArray(0..<Int32(keyLength)).reshaped(1, keyLength)
        let keyPositions = keyIndices + Int32(keyStart)
        let allowed = (keyPositions .<= queryPositions)
            .&& (keyPositions .> (queryPositions - Int32(slidingWindow)))
        let zeros = MLXArray.zeros([length, keyLength], dtype: x.dtype)
        let mask = MLX.where(allowed, zeros, zeros - 1e9)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: state.0,
            values: state.1,
            scale: scale,
            mask: .array(mask),
            sinks: attentionSinkBias.asType(queries.dtype)
        )
        return outputProjection(
            attended.transposed(0, 2, 1, 3)
                .reshaped(batch, length, headCount * headDimensions)
        )
    }
}

final class NemotronHDSparkMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProjection: NemotronHNVFP4Linear
    @ModuleInfo(key: "up_proj") var upProjection: NemotronHNVFP4Linear
    @ModuleInfo(key: "down_proj") var downProjection: NemotronHNVFP4Linear

    init(config: NemotronHDSparkConfig) {
        self._gateProjection.wrappedValue = NemotronHNVFP4Linear(
            inputDimensions: config.hiddenSize,
            outputDimensions: config.intermediateSize
        )
        self._upProjection.wrappedValue = NemotronHNVFP4Linear(
            inputDimensions: config.hiddenSize,
            outputDimensions: config.intermediateSize
        )
        self._downProjection.wrappedValue = NemotronHNVFP4Linear(
            inputDimensions: config.intermediateSize,
            outputDimensions: config.hiddenSize
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProjection(MLXNN.silu(gateProjection(x)) * upProjection(x))
    }
}

final class NemotronHDSparkLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: NemotronHDSparkAttention
    @ModuleInfo(key: "mlp") var mlp: NemotronHDSparkMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    init(config: NemotronHDSparkConfig) {
        self._attention.wrappedValue = NemotronHDSparkAttention(config: config)
        self._mlp.wrappedValue = NemotronHDSparkMLP(config: config)
        self._inputNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }

    func appendContext(_ context: MLXArray, cache: Gemma4AttentionCache) {
        attention.appendContext(context, cache: cache)
    }

    func callAsFunction(_ x: MLXArray, cache: Gemma4AttentionCache) -> MLXArray {
        let attended = x + attention(inputNorm(x), cache: cache)
        return attended + mlp(postAttentionNorm(attended))
    }
}

final class NemotronHDSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var tokenEmbedding: Embedding
    @ModuleInfo(key: "markov_w2") var outputProjection: NemotronHNVFP4Linear

    init(config: NemotronHDSparkConfig) {
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.markovHeadDim
        )
        self._outputProjection.wrappedValue = NemotronHNVFP4Linear(
            inputDimensions: config.markovHeadDim,
            outputDimensions: config.vocabSize
        )
        super.init()
    }

    func bias(previousToken: MLXArray) -> MLXArray {
        outputProjection(tokenEmbedding(previousToken.asType(.int32)))
    }
}

final class NemotronHDSparkModel: Module {
    @ModuleInfo(key: "embed_tokens") var tokenEmbedding: Embedding
    @ModuleInfo(key: "fc") var featureProjection: NemotronHNVFP4Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [NemotronHDSparkLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "markov_head") var markovHead: NemotronHDSparkMarkovHead

    let config: NemotronHDSparkConfig

    init(config: NemotronHDSparkConfig) {
        self.config = config
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._featureProjection.wrappedValue = NemotronHNVFP4Linear(
            inputDimensions: config.hiddenSize * config.speculation.targetLayerIDs.count,
            outputDimensions: config.hiddenSize
        )
        self._hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            NemotronHDSparkLayer(config: config)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._markovHead.wrappedValue = NemotronHDSparkMarkovHead(config: config)
        super.init()
    }

    func combineTargetHiddenStates(_ states: [Int: MLXArray]) -> MLXArray {
        let ordered = config.speculation.targetLayerIDs.map { states[$0]! }
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
        speculativeTokenCount: Int,
        cache: [Gemma4AttentionCache]
    ) -> MLXArray {
        precondition(speculativeTokenCount > 0 && speculativeTokenCount < config.blockSize)
        let input = MLXArray(
            [Int32(anchorToken)]
                + Array(repeating: Int32(config.speculation.maskTokenID), count: speculativeTokenCount)
        ).reshaped(1, speculativeTokenCount + 1)
        var hidden = tokenEmbedding(input)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: cache[index])
        }
        return norm(hidden)[0..., 1..., 0...]
    }

    func proposalTokens(
        baseHidden: MLXArray,
        anchorToken: Int,
        target: NemotronHCausalLM,
        sample: (MLXArray, Int) -> Int
    ) -> [Int] {
        let baseLogits = target.logits(from: baseHidden)
        var previous = anchorToken
        var tokens: [Int] = []
        for index in 0..<baseHidden.dim(1) {
            let previousArray = MLXArray([Int32(previous)]).reshaped(1, 1)
            let logits = baseLogits[0, index, 0...]
                + markovHead.bias(previousToken: previousArray)[0, 0, 0...]
            let token = sample(logits, index)
            tokens.append(token)
            previous = token
        }
        return tokens
    }

    func makeCache(initialOffset: Int = 0) -> [Gemma4AttentionCache] {
        layers.map { _ in
            Gemma4SlidingKVCache(
                maxSize: config.speculation.slidingWindow,
                initialOffset: initialOffset
            )
        }
    }
}
