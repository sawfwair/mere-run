// Native implementation of Salesforce's DFlash drafter (Apache-2.0).
import MLX
import MLXFast
import MLXNN
import MereRunGemmaModel

package final class SOVDFlashAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let window: Int?
    let rope: RoPE

    init(config: SOVDFlashConfig, layer: Int) {
        heads = config.numAttentionHeads
        kvHeads = config.numKeyValueHeads
        headDim = config.headDim
        window = config.layerTypes[layer] == "sliding_attention" ? config.slidingWindow : nil
        rope = RoPE(dimensions: headDim, traditional: false, base: config.rope.theta)
        _query.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _key.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _value.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _output.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    func keysAndValues(_ x: MLXArray, offset: Int) -> (MLXArray, MLXArray) {
        let shape = [x.dim(0), x.dim(1), kvHeads, headDim]
        return (rope(keyNorm(key(x).reshaped(shape).transposed(0, 2, 1, 3)), offset: offset),
                value(x).reshaped(shape).transposed(0, 2, 1, 3))
    }

    func appendContext(_ context: MLXArray, cache: Gemma4AttentionCache) {
        let (keys, values) = keysAndValues(context, offset: cache.offset)
        cache.append(keys: keys, values: values)
    }

    func callAsFunction(_ x: MLXArray, cache: Gemma4AttentionCache) -> MLXArray {
        let q = rope(queryNorm(query(x).reshaped(x.dim(0), x.dim(1), heads, headDim).transposed(0, 2, 1, 3)), offset: cache.offset)
        let (blockKeys, blockValues) = keysAndValues(x, offset: cache.offset)
        let context = cache.currentState()
        let keys = context.map { concatenated([$0.0, blockKeys], axis: 2) } ?? blockKeys
        let values = context.map { concatenated([$0.1, blockValues], axis: 2) } ?? blockValues
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let window {
            mask = .array(Self.slidingMask(offset: cache.offset, contextLength: context?.0.dim(2) ?? 0,
                                           length: x.dim(1), window: window, dtype: x.dtype))
        } else { mask = .none }
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: values,
                                                        scale: 1 / Float(headDim).squareRoot(), mask: mask)
        return output(attended.transposed(0, 2, 1, 3).reshaped(x.dim(0), x.dim(1), heads * headDim))
    }

    package static func slidingMask(offset: Int, contextLength: Int, length: Int, window: Int, dtype: DType) -> MLXArray {
        let q = MLXArray(Int32(offset)..<Int32(offset + length)).reshaped(length, 1)
        let k = MLXArray(Int32(offset - contextLength)..<Int32(offset + length)).reshaped(1, -1)
        let allowed = (k .<= q) .&& (k .> q - Int32(window))
        return MLX.where(allowed, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(dtype)
    }
}

final class SOVDFlashMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(config: SOVDFlashConfig) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class SOVDFlashLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SOVDFlashAttention
    @ModuleInfo(key: "mlp") var mlp: SOVDFlashMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    init(config: SOVDFlashConfig, layer: Int) {
        _attention.wrappedValue = SOVDFlashAttention(config: config, layer: layer)
        _mlp.wrappedValue = SOVDFlashMLP(config: config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, cache: Gemma4AttentionCache) -> MLXArray {
        let attended = x + attention(inputNorm(x), cache: cache)
        return attended + mlp(postNorm(attended))
    }
}

package final class SOVDFlashModel: Module {
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [SOVDFlashLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    package let config: SOVDFlashConfig

    package init(config: SOVDFlashConfig) {
        self.config = config
        _fc.wrappedValue = Linear(config.hiddenSize * config.dflash.targetLayerIDs.count, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { SOVDFlashLayer(config: config, layer: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    package func makeCache() -> [Gemma4AttentionCache] {
        config.layerTypes.map { type -> Gemma4AttentionCache in
            if type == "sliding_attention" { return Gemma4SlidingKVCache(maxSize: config.slidingWindow) }
            return Gemma4FullKVCache()
        }
    }

    package func appendContext(_ states: [Int: MLXArray], cache: [Gemma4AttentionCache], evaluate: Bool = true) {
        let context = hiddenNorm(fc(concatenated(config.dflash.targetLayerIDs.map { states[$0]! }, axis: -1)))
        for (layer, entry) in zip(layers, cache) { layer.attention.appendContext(context, cache: entry) }
        if evaluate { evaluateGemma4CacheStorage(cache) }
    }

    /// The draft block is temporary. Only target context is retained in these caches.
    package func propose(anchor: Int, count: Int, target: Gemma4TextCausalLM, cache: [Gemma4AttentionCache]) -> [Int] {
        proposalIDs(anchor: anchor, count: count, target: target, cache: cache).asArray(Int.self)
    }

    /// Keep proposals on the device so verification can be built before the host reads them.
    package func proposalIDs(anchor: Int, count: Int, target: Gemma4TextCausalLM, cache: [Gemma4AttentionCache],
                             outputHead: QuantizedLinear? = nil) -> MLXArray {
        proposal(anchor: anchor, count: count, target: target, cache: cache, outputHead: outputHead).ids
    }

    package func proposal(anchor: Int, count: Int, target: Gemma4TextCausalLM, cache: [Gemma4AttentionCache],
                          outputHead: Linear? = nil, tokenMapping: MLXArray? = nil,
                          confidence: Bool = false) -> (ids: MLXArray, confidence: MLXArray?) {
        let ids = [anchor] + Array(repeating: config.dflash.maskTokenID, count: count)
        let hidden = denoise(target.rawInputEmbeddings(MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)), cache: cache)
        let proposed = hidden[0..., 1..., 0...]
        let logits = outputHead.map { target.softcapOutputLogits($0(proposed)) } ?? target.outputLogits(proposed)
        let localIDs = argMax(logits, axis: -1).asType(.int32).reshaped(count)
        return (tokenMapping.map { take($0, localIDs, axis: 0) } ?? localIDs,
                confidence ? softmax(logits.asType(.float32), axis: -1).max(axis: -1).reshaped(count) : nil)
    }

    package func denoise(_ embeddings: MLXArray, cache: [Gemma4AttentionCache]) -> MLXArray {
        var hidden = embeddings
        for (layer, entry) in zip(layers, cache) { hidden = layer(hidden, cache: entry) }
        return norm(hidden)
    }
}
