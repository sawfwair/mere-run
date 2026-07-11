import Foundation
import MLX
import MLXNN
import MLXFast
import MereRunCore

// MARK: - Qwen3 TTS Model

/// Qwen3 language model configured for TTS audio token generation
public final class Qwen3TTSModel: Module {
    public let configuration: Qwen3TTSModelConfiguration
    @ModuleInfo(key: "model") var model: Qwen3TTSTransformer
    @ModuleInfo(key: "codec_head") var codecHead: Linear

    public init(configuration: Qwen3TTSModelConfiguration) {
        self.configuration = configuration
        self._model.wrappedValue = Qwen3TTSTransformer(configuration: configuration)
        // Codec head projects hidden states to codec vocabulary
        self._codecHead.wrappedValue = Linear(configuration.hiddenSize, configuration.vocabSize, bias: false)
    }

    /// Forward text tokens through the model and return codec logits
    public func forwardText(inputIds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let hidden = model.forwardText(inputIds: inputIds, cache: cache)
        return codecHead(hidden)
    }

    /// Forward codec tokens through the model and return next codec logits
    public func forwardCodec(inputIds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let hidden = model.forwardCodec(inputIds: inputIds, cache: cache)
        return codecHead(hidden)
    }

    /// Default forward (text)
    public func callAsFunction(inputIds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        forwardText(inputIds: inputIds, cache: cache)
    }

    /// Create a fresh KV cache for generation
    public func createCache() -> [KVCache] {
        (0..<configuration.numHiddenLayers).map { _ in KVCacheSimple(step: 256) }
    }
}

// MARK: - Configuration

public struct Qwen3TTSModelConfiguration: Sendable, Hashable {
    public var vocabSize: Int          // Codec vocabulary (3072)
    public var textVocabSize: Int      // Text vocabulary (151936)
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var intermediateSize: Int
    public var maxPositionEmbeddings: Int
    public var ropeTheta: Float
    public var rmsNormEps: Float
    public var headDim: Int

    public init(
        vocabSize: Int = 3072,
        textVocabSize: Int = 151936,
        hiddenSize: Int = 2048,
        numHiddenLayers: Int = 28,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 4,
        intermediateSize: Int = 6144,
        maxPositionEmbeddings: Int = 32768,
        ropeTheta: Float = 1_000_000.0,
        rmsNormEps: Float = 1e-6,
        headDim: Int = 128
    ) {
        self.vocabSize = vocabSize
        self.textVocabSize = textVocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
        self.headDim = headDim
    }

    public static func fromConfig(_ config: Qwen3TTSConfig) -> Qwen3TTSModelConfiguration {
        Qwen3TTSModelConfiguration(
            vocabSize: config.vocabSize,
            textVocabSize: config.textVocabSize,
            hiddenSize: config.hiddenSize,
            numHiddenLayers: config.numHiddenLayers,
            numAttentionHeads: config.numAttentionHeads,
            numKeyValueHeads: config.numKeyValueHeads,
            intermediateSize: config.intermediateSize,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            ropeTheta: config.ropeTheta,
            rmsNormEps: config.rmsNormEps,
            headDim: config.computedHeadDim
        )
    }
}

// MARK: - Transformer

final class Qwen3TTSTransformer: Module {
    let configuration: Qwen3TTSModelConfiguration

    // Text embedding for input tokens
    @ModuleInfo(key: "text_embedding") var textEmbedding: Embedding
    // Codec embedding for generated audio tokens
    @ModuleInfo(key: "codec_embedding") var codecEmbedding: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen3TTSDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let rope: RoPE

    init(configuration: Qwen3TTSModelConfiguration) {
        self.configuration = configuration

        // Text embedding for input text tokens (vocab 151936)
        self._textEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.textVocabSize,
            dimensions: configuration.hiddenSize
        )

        // Codec embedding for audio tokens (vocab 3072)
        self._codecEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.vocabSize,
            dimensions: configuration.hiddenSize
        )

        self._layers.wrappedValue = (0..<configuration.numHiddenLayers).map { _ in
            Qwen3TTSDecoderLayer(configuration: configuration)
        }

        self._norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )

        self.rope = RoPE(
            dimensions: configuration.headDim,
            traditional: false,
            base: configuration.ropeTheta,
            scale: 1.0
        )
    }

    /// Forward pass with text tokens (for prefill)
    func forwardText(inputIds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }

        var h = textEmbedding(tokenIds).asType(.bfloat16)
        let mask = createMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], rope: rope)
        }

        h = norm(h)
        return h
    }

    /// Forward pass with codec tokens (for generation)
    func forwardCodec(inputIds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }

        var h = codecEmbedding(tokenIds).asType(.bfloat16)
        let mask = createMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], rope: rope)
        }

        h = norm(h)
        return h
    }

    func callAsFunction(inputIds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        // Default to text forward
        forwardText(inputIds: inputIds, cache: cache)
    }

    private func createMask(h: MLXArray, cache: KVCache?) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let n = h.dim(1)
        if let cache = cache {
            return cache.makeMask(n: n)
        }
        if n == 1 {
            return .none
        }
        return .causal
    }
}

// MARK: - Decoder Layer

final class Qwen3TTSDecoderLayer: Module {
    let configuration: Qwen3TTSModelConfiguration

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3TTSAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen3TTSMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(configuration: Qwen3TTSModelConfiguration) {
        self.configuration = configuration

        self._selfAttn.wrappedValue = Qwen3TTSAttention(configuration: configuration)
        self._mlp.wrappedValue = Qwen3TTSMLP(configuration: configuration)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        rope: RoPE
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let attnOut = selfAttn(normed, mask: mask, cache: cache, rope: rope)
        let h = x + attnOut

        let postNormed = postAttentionLayerNorm(h)
        let mlpOut = mlp(postNormed)

        return h + mlpOut
    }
}

// MARK: - Attention

final class Qwen3TTSAttention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(configuration: Qwen3TTSModelConfiguration) {
        self.hiddenSize = configuration.hiddenSize
        self.numHeads = configuration.numAttentionHeads
        self.numKVHeads = configuration.numKeyValueHeads
        self.headDim = configuration.headDim
        self.scale = pow(Float(configuration.headDim), -0.5)

        self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: configuration.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: configuration.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        rope: RoPE
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // Reshape and normalize
        queries = qNorm(queries.reshaped(B, L, numHeads, headDim)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        // Apply RoPE
        let offset = cache?.offset ?? 0
        queries = rope(queries.asType(.bfloat16), offset: offset)
        keys = rope(keys.asType(.bfloat16), offset: offset)

        // Update cache
        if let cache = cache {
            let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
            keys = cachedKeys
            values = cachedValues
        }

        // MLX performs the SDPA softmax in float32 while retaining the native
        // projection/cache dtype for Q/K/V and its optimized decode kernel.
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        let flattened = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return oProj(flattened)
    }

}

// MARK: - MLP

final class Qwen3TTSMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(configuration: Qwen3TTSModelConfiguration) {
        self._gateProj.wrappedValue = Linear(configuration.hiddenSize, configuration.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(configuration.hiddenSize, configuration.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(configuration.intermediateSize, configuration.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Token Generation

extension Qwen3TTSModel {
    /// Generate tokens autoregressively
    /// - Parameters:
    ///   - inputIds: Initial token IDs
    ///   - maxTokens: Maximum number of tokens to generate
    ///   - temperature: Sampling temperature
    ///   - stopTokens: Token IDs that signal end of generation
    ///   - progressHandler: Called after each token is generated
    /// - Returns: Generated token sequence (including input)
    public func generate(
        inputIds: MLXArray,
        maxTokens: Int,
        temperature: Float = 0.6,
        stopTokens: Set<Int> = [],
        progressHandler: ((Int, Int) -> Void)? = nil
    ) -> [Int] {
        let cache = createCache()

        // Prefill with input sequence
        var logits = self(inputIds: inputIds, cache: cache)
        MLX.eval(logits)

        // Collect generated tokens
        let inputIdsList = inputIds.asArray(Int32.self).map { Int($0) }
        var generatedTokens: [Int] = []

        for i in 0..<maxTokens {
            // Sample from last position
            let lastLogits = logits[0, -1, 0...]

            let nextToken = sampleToken(logits: lastLogits, temperature: temperature)

            // Check for stop tokens
            if stopTokens.contains(nextToken) {
                generatedTokens.append(nextToken)
                break
            }

            generatedTokens.append(nextToken)
            progressHandler?(i + 1, maxTokens)

            // Generate next token
            let nextInput = MLXArray([Int32(nextToken)]).reshaped(1, 1)
            logits = self(inputIds: nextInput, cache: cache)
            MLX.eval(logits)
        }

        return inputIdsList + generatedTokens
    }

    private func sampleToken(logits: MLXArray, temperature: Float) -> Int {
        var adjustedLogits = logits

        if temperature > 0 {
            adjustedLogits = adjustedLogits / temperature
        }

        let probs = softmax(adjustedLogits)
        MLX.eval(probs)

        // Greedy for low temperature
        if temperature <= 0.1 {
            let tokenId = MLX.argMax(probs).item(Int32.self)
            return Int(tokenId)
        }

        // Sample from distribution
        let probsArray = probs.asArray(Float.self)
        let cumProbs = probsArray.reduce(into: [Float]()) { result, prob in
            result.append((result.last ?? 0) + prob)
        }

        let rand = Float.random(in: 0..<1)
        for (i, cumProb) in cumProbs.enumerated() {
            if rand < cumProb {
                return i
            }
        }

        return probsArray.count - 1
    }
}
