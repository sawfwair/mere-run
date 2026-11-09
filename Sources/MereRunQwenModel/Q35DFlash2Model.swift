import Foundation
import MereRunTensor
import MLX
import MLXFast
import MLXNN

package final class Q35DFlash2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    private let heads: Int
    private let kvHeads: Int
    private let headDim: Int
    private let scale: Float
    private let rope: RoPE
    private var fusedQKV: FusedQuantizedProjection?
    private var fusedKV: FusedQuantizedProjection?

    package init(config: Q35DFlash2Config) {
        heads = config.numAttentionHeads
        kvHeads = config.numKeyValueHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        rope = RoPE(dimensions: headDim, traditional: false, base: config.rope.theta)
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    package func prepareProjections() {
        fusedQKV = FusedQuantizedProjection.fuse([qProj, kProj, vProj])
        fusedKV = FusedQuantizedProjection.fuse([kProj, vProj])
    }

    private func keyValues(_ k: MLXArray, _ v: MLXArray, offset: Int) -> (MLXArray, MLXArray) {
        let shape = [k.dim(0), k.dim(1), kvHeads, headDim]
        let keys = rope(kNorm(k.reshaped(shape).transposed(0, 2, 1, 3)), offset: offset)
        let values = v.reshaped(shape).transposed(0, 2, 1, 3)
        return (keys, values)
    }

    package func appendContext(_ context: MLXArray, cache: Q35DFlash2Cache) {
        // The shared context projection is already normalized. No decoder
        // input norm or dynamic convolution is applied to target context KV.
        let k: MLXArray
        let v: MLXArray
        if let fusedKV, fusedKV.matches([kProj, vProj]) {
            let parts = fusedKV.callSplit(context)
            k = parts[0]
            v = parts[1]
        } else {
            k = kProj(context)
            v = vProj(context)
        }
        let (keys, values) = keyValues(k, v, offset: cache.offset)
        cache.append(keys: keys, values: values)
    }

    package func callAsFunction(_ x: MLXArray, cache: Q35DFlash2Cache, mask: MLXArray? = nil) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let q: MLXArray
        let k: MLXArray
        let v: MLXArray
        if let fusedQKV, fusedQKV.matches([qProj, kProj, vProj]) {
            let parts = fusedQKV.callSplit(x)
            q = parts[0]
            k = parts[1]
            v = parts[2]
        } else {
            q = qProj(x)
            k = kProj(x)
            v = vProj(x)
        }
        let queries = rope(
            qNorm(q.reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)),
            offset: cache.offset
        )
        let (blockKeys, blockValues) = keyValues(k, v, offset: cache.offset)
        let keys = cache.keys.map { concatenated([$0, blockKeys], axis: 2) } ?? blockKeys
        let values = cache.values.map { concatenated([$0, blockValues], axis: 2) } ?? blockValues
        let mask = mask ?? Self.blockMask(
            offset: cache.offset, length: length,
            contextLength: cache.keys?.dim(2) ?? 0, window: cache.window, dtype: x.dtype
        )
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .array(mask)
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, heads * headDim))
    }

    package static func blockMask(
        offset: Int, length: Int, contextLength: Int, window: Int, dtype: DType
    ) -> MLXArray {
        let queries = MLXArray(Int32(offset)..<Int32(offset + length)).reshaped(length, 1)
        let keys = MLXArray(Int32(offset - contextLength)..<Int32(offset + length))
            .reshaped(1, contextLength + length)
        let context = (keys .< Int32(offset)) .&& (keys .> queries - Int32(window))
        let allowed = context .|| (keys .>= Int32(offset))
        let zeros = MLXArray.zeros([1, 1, length, contextLength + length], dtype: dtype)
        return MLX.where(allowed.reshaped(zeros.shape), zeros, zeros + MLXArray(-1e9).asType(dtype))
    }
}

final class Q35DFlash2MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(config: Q35DFlash2Config) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(Q35CompiledOperations.current.swiglu(gate(x), up(x)))
    }
}

final class Q35DFlash2Layer: Module {
    @ModuleInfo(key: "self_attn") var attention: Q35DFlash2Attention
    @ModuleInfo(key: "mlp") var mlp: Q35DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlash2GroupedDynamicCausalConv
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlash2GroupedDynamicCausalConv

    init(config: Q35DFlash2Config) {
        _attention.wrappedValue = Q35DFlash2Attention(config: config)
        _mlp.wrappedValue = Q35DFlash2MLP(config: config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attentionConv.wrappedValue = DFlash2GroupedDynamicCausalConv(
            hiddenSize: config.hiddenSize, kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize
        )
        _mlpConv.wrappedValue = DFlash2GroupedDynamicCausalConv(
            hiddenSize: config.hiddenSize, kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Q35DFlash2Cache, mask: MLXArray) -> MLXArray {
        let attentionInput = attentionConv.prepare(inputNorm(x))
        let attended = x + attentionConv.finish(
            attention(attentionInput.hidden, cache: cache, mask: mask), dynamic: attentionInput.dynamic
        )
        let mlpInput = mlpConv.prepare(postNorm(attended))
        return attended + mlpConv.finish(mlp(mlpInput.hidden), dynamic: mlpInput.dynamic)
    }
}

package final class Q35DFlash2Model: Module {
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [Q35DFlash2Layer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var selector: DFlash2CandidateSelector

    package let config: Q35DFlash2Config

    package init(config: Q35DFlash2Config) {
        self.config = config
        _fc.wrappedValue = Linear(
            config.hiddenSize * config.dflash.targetLayerIDs.count, config.hiddenSize, bias: false
        )
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in Q35DFlash2Layer(config: config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _selector.wrappedValue = DFlash2CandidateSelector(
            vocabularySize: config.vocabSize, hiddenSize: config.hiddenSize,
            rank: config.dflash.selectorRank, topK: config.dflash.selectorTopK
        )
        super.init()
    }

    package func makeCache(offset: Int = 0) -> [Q35DFlash2Cache] {
        layers.map { _ in Q35DFlash2Cache(window: config.slidingWindow, offset: offset) }
    }

    package func quantizeProjections(groupSize: Int = 64) {
        var updates: [(String, Module)] = []
        for (path, module) in leafModules().flattened() {
            guard let linear = module as? Linear else { continue }
            let (weight, scales, biases) = MLX.quantized(linear.weight, groupSize: groupSize, bits: 4)
            MLX.eval(weight, scales)
            if let biases { MLX.eval(biases) }
            updates.append((path, PortableQuantizedLinear(
                weight: weight, bias: nil, scales: scales, biases: biases,
                groupSize: groupSize, bits: 4, mode: .affine
            )))
        }
        // A single nested update supplies every array index. Sparse updates
        // to layers.1, layers.2, ... cannot reconstruct MLX's typed layer array.
        update(modules: ModuleChildren.unflattened(updates))
        MLX.eval(parameters().flattened().map(\.1))
    }

    /// Prepare copies at load time, outside the generation loop. Keep the
    /// original parameter tree so checkpoint keys and replacement semantics stay intact.
    package func prepareDrafting() {
        for layer in layers { layer.attention.prepareProjections() }
    }

    package func appendTargetContext(_ states: [Int: MLXArray], cache: [Q35DFlash2Cache]) {
        let context = hiddenNorm(fc(concatenated(config.dflash.targetLayerIDs.map { states[$0]! }, axis: -1)))
        for (layer, entry) in zip(layers, cache) {
            layer.attention.appendContext(context, cache: entry)
        }
    }

    package func draft(
        anchor: Int, count: Int, cache: [Q35DFlash2Cache], target: Q35Model
    ) -> MLXArray {
        selection(anchor: anchor, count: count, cache: cache, target: target, temperature: 0).tokens
    }

    package func selection(
        anchor: Int, count: Int, cache: [Q35DFlash2Cache], target: Q35Model, temperature: Float
    ) -> DFlash2Selection {
        precondition(count > 0 && count < config.dflash.blockSize && cache.count == layers.count)
        let tokens = [anchor] + Array(repeating: config.dflash.maskTokenID, count: count)
        var hidden = target.embeddings(for: MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count))
        let first = cache[0]
        let mask = Q35DFlash2Attention.blockMask(
            offset: first.offset, length: tokens.count, contextLength: first.keys?.dim(2) ?? 0,
            window: first.window, dtype: hidden.dtype
        )
        for (layer, entry) in zip(layers, cache) {
            hidden = layer(hidden, cache: entry, mask: mask)
        }
        hidden = norm(hidden)[0..., 1..., 0...]
        return selector.select(
            hidden: hidden, logits: target.logits(from: hidden),
            anchorTokenIds: MLXArray([Int32(anchor)]), temperature: temperature
        )
    }

    package static func weightKey(_ key: String) -> String {
        switch key {
        case "candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook":
            return key + ".weight"
        default:
            return key
        }
    }
}
