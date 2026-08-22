import Foundation
import MLX
import MLXFast
import MLXNN

final class SenseNovaU15RMSNorm: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    private let epsilon: Float

    init(dimensions: Int, epsilon: Float) {
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        self.epsilon = epsilon
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(input, weight: weight, eps: epsilon)
    }
}

final class SenseNovaU15MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProjection: Linear
    @ModuleInfo(key: "up_proj") var upProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(config: SenseNovaU15Config.LLMConfig) {
        self._gateProjection.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProjection.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProjection.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(MLXNN.silu(gateProjection(input)) * upProjection(input))
    }
}

final class SenseNovaU15KVCache: @unchecked Sendable {
    var keys: MLXArray?
    var values: MLXArray?

    func store(keys: MLXArray, values: MLXArray) {
        self.keys = keys
        self.values = values
    }
}

enum SenseNovaU15Expert {
    case understanding
    case generation
}

final class SenseNovaU15Attention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_proj_mot_gen") var generationQueryProjection: Linear
    @ModuleInfo(key: "k_proj_mot_gen") var generationKeyProjection: Linear
    @ModuleInfo(key: "v_proj_mot_gen") var generationValueProjection: Linear
    @ModuleInfo(key: "o_proj_mot_gen") var generationOutputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: SenseNovaU15RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: SenseNovaU15RMSNorm
    @ModuleInfo(key: "q_norm_hw") var queryNormHW: SenseNovaU15RMSNorm
    @ModuleInfo(key: "k_norm_hw") var keyNormHW: SenseNovaU15RMSNorm
    @ModuleInfo(key: "q_norm_mot_gen") var generationQueryNorm: SenseNovaU15RMSNorm
    @ModuleInfo(key: "k_norm_mot_gen") var generationKeyNorm: SenseNovaU15RMSNorm
    @ModuleInfo(key: "q_norm_hw_mot_gen") var generationQueryNormHW: SenseNovaU15RMSNorm
    @ModuleInfo(key: "k_norm_hw_mot_gen") var generationKeyNormHW: SenseNovaU15RMSNorm

    private let numberOfHeads: Int
    private let numberOfKeyValueHeads: Int
    private let headDimension: Int
    private let scale: Float
    private let timeBase: Float
    private let spatialBase: Float

    init(config: SenseNovaU15Config.LLMConfig) {
        let queryOutput = config.numberOfAttentionHeads * config.headDimension
        let keyValueOutput = config.numberOfKeyValueHeads * config.headDimension
        self._queryProjection.wrappedValue = Linear(config.hiddenSize, queryOutput, bias: config.attentionBias)
        self._keyProjection.wrappedValue = Linear(config.hiddenSize, keyValueOutput, bias: config.attentionBias)
        self._valueProjection.wrappedValue = Linear(config.hiddenSize, keyValueOutput, bias: config.attentionBias)
        self._outputProjection.wrappedValue = Linear(queryOutput, config.hiddenSize, bias: config.attentionBias)
        self._generationQueryProjection.wrappedValue = Linear(config.hiddenSize, queryOutput, bias: config.attentionBias)
        self._generationKeyProjection.wrappedValue = Linear(config.hiddenSize, keyValueOutput, bias: config.attentionBias)
        self._generationValueProjection.wrappedValue = Linear(config.hiddenSize, keyValueOutput, bias: config.attentionBias)
        self._generationOutputProjection.wrappedValue = Linear(queryOutput, config.hiddenSize, bias: config.attentionBias)
        let half = config.headDimension / 2
        self._queryNorm.wrappedValue = SenseNovaU15RMSNorm(dimensions: half, epsilon: config.rmsNormEpsilon)
        self._keyNorm.wrappedValue = SenseNovaU15RMSNorm(dimensions: half, epsilon: config.rmsNormEpsilon)
        self._queryNormHW.wrappedValue = SenseNovaU15RMSNorm(dimensions: half, epsilon: config.rmsNormEpsilon)
        self._keyNormHW.wrappedValue = SenseNovaU15RMSNorm(dimensions: half, epsilon: config.rmsNormEpsilon)
        self._generationQueryNorm.wrappedValue = SenseNovaU15RMSNorm(dimensions: half, epsilon: config.rmsNormEpsilon)
        self._generationKeyNorm.wrappedValue = SenseNovaU15RMSNorm(dimensions: half, epsilon: config.rmsNormEpsilon)
        self._generationQueryNormHW.wrappedValue = SenseNovaU15RMSNorm(dimensions: half, epsilon: config.rmsNormEpsilon)
        self._generationKeyNormHW.wrappedValue = SenseNovaU15RMSNorm(dimensions: half, epsilon: config.rmsNormEpsilon)
        self.numberOfHeads = config.numberOfAttentionHeads
        self.numberOfKeyValueHeads = config.numberOfKeyValueHeads
        self.headDimension = config.headDimension
        self.scale = 1 / sqrt(Float(config.headDimension))
        self.timeBase = config.ropeTheta
        self.spatialBase = config.ropeThetaHW
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        indexes: MLXArray,
        expert: SenseNovaU15Expert,
        cache: SenseNovaU15KVCache,
        updateCache: Bool
    ) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let queryLinear = expert == .understanding ? queryProjection(input) : generationQueryProjection(input)
        let keyLinear = expert == .understanding ? keyProjection(input) : generationKeyProjection(input)
        let valueLinear = expert == .understanding ? valueProjection(input) : generationValueProjection(input)
        var queries = queryLinear.reshaped(batch, sequence, numberOfHeads, headDimension)
        var keys = keyLinear.reshaped(batch, sequence, numberOfKeyValueHeads, headDimension)
        let values = valueLinear.reshaped(batch, sequence, numberOfKeyValueHeads, headDimension)

        let half = headDimension / 2
        let qTime = queries[0..., 0..., 0..., 0..<half]
        let qHW = queries[0..., 0..., 0..., half...]
        let kTime = keys[0..., 0..., 0..., 0..<half]
        let kHW = keys[0..., 0..., 0..., half...]
        let normalizedQTime = expert == .understanding ? queryNorm(qTime) : generationQueryNorm(qTime)
        let normalizedQHW = expert == .understanding ? queryNormHW(qHW) : generationQueryNormHW(qHW)
        let normalizedKTime = expert == .understanding ? keyNorm(kTime) : generationKeyNorm(kTime)
        let normalizedKHW = expert == .understanding ? keyNormHW(kHW) : generationKeyNormHW(kHW)
        let quarter = headDimension / 4
        queries = MLX.concatenated([
            applyRoPE(normalizedQTime, positions: indexes[0, 0...], base: timeBase),
            applyRoPE(normalizedQHW[0..., 0..., 0..., 0..<quarter], positions: indexes[1, 0...], base: spatialBase),
            applyRoPE(normalizedQHW[0..., 0..., 0..., quarter...], positions: indexes[2, 0...], base: spatialBase)
        ], axis: -1).transposed(0, 2, 1, 3)
        keys = MLX.concatenated([
            applyRoPE(normalizedKTime, positions: indexes[0, 0...], base: timeBase),
            applyRoPE(normalizedKHW[0..., 0..., 0..., 0..<quarter], positions: indexes[1, 0...], base: spatialBase),
            applyRoPE(normalizedKHW[0..., 0..., 0..., quarter...], positions: indexes[2, 0...], base: spatialBase)
        ], axis: -1).transposed(0, 2, 1, 3)
        var allValues = values.transposed(0, 2, 1, 3)

        if updateCache {
            cache.store(keys: keys, values: allValues)
        } else if let prefixKeys = cache.keys, let prefixValues = cache.values {
            keys = MLX.concatenated([prefixKeys, keys], axis: 2)
            allValues = MLX.concatenated([prefixValues, allValues], axis: 2)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: allValues,
            scale: scale,
            mask: updateCache ? .causal : .none
        ).transposed(0, 2, 1, 3).reshaped(batch, sequence, numberOfHeads * headDimension)
        return expert == .understanding ? outputProjection(attended) : generationOutputProjection(attended)
    }

    private func applyRoPE(_ input: MLXArray, positions: MLXArray, base: Float) -> MLXArray {
        let dimension = input.dim(-1)
        let doubledDimension = dimension * 2
        let full = MLXArray(stride(from: Float(0), to: Float(doubledDimension), by: 2))
        let fullInverseFrequency = 1 / MLX.pow(MLXArray(base), full / Float(doubledDimension))
        let selector = MLXArray(Array(stride(from: Int32(0), to: Int32(dimension), by: 2)))
        let inverseFrequency = fullInverseFrequency[selector]
        let frequencies = positions.asType(.float32)[0..., .newAxis] * inverseFrequency[.newAxis, 0...]
        let embedding = MLX.concatenated([frequencies, frequencies], axis: -1)
            .asType(input.dtype)[.newAxis, 0..., .newAxis, 0...]
        let half = dimension / 2
        let rotated = MLX.concatenated([
            -input[0..., 0..., 0..., half...],
            input[0..., 0..., 0..., 0..<half]
        ], axis: -1)
        return input * MLX.cos(embedding) + rotated * MLX.sin(embedding)
    }
}

final class SenseNovaU15DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SenseNovaU15Attention
    @ModuleInfo(key: "mlp") var mlp: SenseNovaU15MLP
    @ModuleInfo(key: "mlp_mot_gen") var generationMLP: SenseNovaU15MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: SenseNovaU15RMSNorm
    @ModuleInfo(key: "input_layernorm_mot_gen") var generationInputNorm: SenseNovaU15RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: SenseNovaU15RMSNorm
    @ModuleInfo(key: "post_attention_layernorm_mot_gen") var generationPostAttentionNorm: SenseNovaU15RMSNorm

    init(config: SenseNovaU15Config.LLMConfig) {
        self._attention.wrappedValue = SenseNovaU15Attention(config: config)
        self._mlp.wrappedValue = SenseNovaU15MLP(config: config)
        self._generationMLP.wrappedValue = SenseNovaU15MLP(config: config)
        self._inputNorm.wrappedValue = SenseNovaU15RMSNorm(dimensions: config.hiddenSize, epsilon: config.rmsNormEpsilon)
        self._generationInputNorm.wrappedValue = SenseNovaU15RMSNorm(dimensions: config.hiddenSize, epsilon: config.rmsNormEpsilon)
        self._postAttentionNorm.wrappedValue = SenseNovaU15RMSNorm(dimensions: config.hiddenSize, epsilon: config.rmsNormEpsilon)
        self._generationPostAttentionNorm.wrappedValue = SenseNovaU15RMSNorm(dimensions: config.hiddenSize, epsilon: config.rmsNormEpsilon)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        indexes: MLXArray,
        expert: SenseNovaU15Expert,
        cache: SenseNovaU15KVCache,
        updateCache: Bool
    ) -> MLXArray {
        let normalized = expert == .understanding ? inputNorm(input) : generationInputNorm(input)
        var hidden = input + attention(
            normalized,
            indexes: indexes,
            expert: expert,
            cache: cache,
            updateCache: updateCache
        )
        let postAttention = expert == .understanding ? postAttentionNorm(hidden) : generationPostAttentionNorm(hidden)
        hidden = hidden + (expert == .understanding ? mlp(postAttention) : generationMLP(postAttention))
        return hidden
    }
}

final class SenseNovaU15Transformer: Module {
    @ModuleInfo(key: "embed_tokens") var tokenEmbedding: Embedding
    @ModuleInfo(key: "layers") var layers: [SenseNovaU15DecoderLayer]
    @ModuleInfo(key: "norm") var norm: SenseNovaU15RMSNorm
    @ModuleInfo(key: "norm_mot_gen") var generationNorm: SenseNovaU15RMSNorm

    init(config: SenseNovaU15Config.LLMConfig) {
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0..<config.numberOfHiddenLayers).map { _ in
            SenseNovaU15DecoderLayer(config: config)
        }
        self._norm.wrappedValue = SenseNovaU15RMSNorm(dimensions: config.hiddenSize, epsilon: config.rmsNormEpsilon)
        self._generationNorm.wrappedValue = SenseNovaU15RMSNorm(dimensions: config.hiddenSize, epsilon: config.rmsNormEpsilon)
        super.init()
    }

    func embed(_ tokenIDs: MLXArray) -> MLXArray { tokenEmbedding(tokenIDs) }

    func forward(
        embeddings: MLXArray,
        indexes: MLXArray,
        expert: SenseNovaU15Expert,
        caches: [SenseNovaU15KVCache],
        updateCache: Bool
    ) -> MLXArray {
        precondition(caches.count == layers.count)
        var hidden = embeddings
        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                indexes: indexes,
                expert: expert,
                cache: caches[index],
                updateCache: updateCache
            )
        }
        return expert == .understanding ? norm(hidden) : generationNorm(hidden)
    }
}

final class SenseNovaU15CausalLM: Module {
    @ModuleInfo(key: "model") var model: SenseNovaU15Transformer

    init(config: SenseNovaU15Config.LLMConfig) {
        self._model.wrappedValue = SenseNovaU15Transformer(config: config)
        super.init()
    }
}
