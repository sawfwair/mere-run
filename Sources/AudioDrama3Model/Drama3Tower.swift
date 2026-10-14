#if !os(iOS)
import Foundation
// Architecture adapted from SGLang-Omni, Apache-2.0; see THIRD_PARTY_NOTICES.md.
import MLX
import MLXFast
import MLXNN
import MereRunKVCache

final class Drama3Norm: Module {
    @ParameterInfo var weight: MLXArray
    let eps: Float
    init(_ dim: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.zeros([dim])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)
    }
}

final class Drama3TowerLayer: Module {
    @ModuleInfo(key: "token_embedding") var embedding: Embedding
    @ModuleInfo var enorm: Drama3Norm
    @ModuleInfo var hnorm: Drama3Norm
    @ModuleInfo(key: "eh_proj") var projection: Linear
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "out") var output: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Drama3Norm
    @ModuleInfo(key: "k_norm") var kNorm: Drama3Norm
    @ModuleInfo(key: "input_norm") var inputNorm: Drama3Norm
    @ModuleInfo(key: "post_norm") var postNorm: Drama3Norm
    @ModuleInfo(key: "final_layernorm") var finalNorm: Drama3Norm
    @ModuleInfo var gate: Linear
    @ModuleInfo var up: Linear
    @ModuleInfo var down: Linear
    let config: Drama3Config.Audio
    let rope: RoPE

    init(_ c: Drama3Config.Audio) {
        config = c
        _embedding.wrappedValue = Embedding(embeddingCount: c.vocab, dimensions: c.dim)
        _enorm.wrappedValue = Drama3Norm(c.dim, eps: c.eps)
        _hnorm.wrappedValue = Drama3Norm(c.dim, eps: c.eps)
        _projection.wrappedValue = Linear(c.dim * 2, c.dim, bias: false)
        _qkv.wrappedValue = Linear(c.dim, 2 * c.heads * c.headDim + 2 * c.kvHeads * c.headDim, bias: false)
        _output.wrappedValue = Linear(c.heads * c.headDim, c.dim, bias: false)
        _qNorm.wrappedValue = Drama3Norm(c.headDim, eps: c.eps)
        _kNorm.wrappedValue = Drama3Norm(c.headDim, eps: c.eps)
        _inputNorm.wrappedValue = Drama3Norm(c.dim, eps: c.eps)
        _postNorm.wrappedValue = Drama3Norm(c.dim, eps: c.eps)
        _finalNorm.wrappedValue = Drama3Norm(c.dim, eps: c.eps)
        _gate.wrappedValue = Linear(c.dim, c.intermediate, bias: false)
        _up.wrappedValue = Linear(c.dim, c.intermediate, bias: false)
        _down.wrappedValue = Linear(c.intermediate, c.dim, bias: false)
        rope = RoPE(dimensions: Int(Float(c.headDim) * c.rotaryPercent), traditional: true, base: c.ropeBase)
    }

    func callAsFunction(_ hidden: MLXArray, code: MLXArray, semantic: MLXArray,
                        extra: MLXArray? = nil, cache: KVCacheSimple) -> MLXArray {
        let c = config, count = hidden.dim(1)
        var embedded = embedding(MLX.clip(code, min: 0, max: c.vocab - 1)) * semantic.expandedDimensions(axis: -1)
        if let extra { embedded = embedded + extra }
        let mixed = projection(MLX.concatenated([enorm(embedded), hnorm(hidden)], axis: -1))
        let projected = qkv(inputNorm(mixed))
        let qWidth = c.heads * c.headDim, kvWidth = c.kvHeads * c.headDim
        // Checkpoint rows are [all queries | all gates | all keys | all values].
        let query = projected[.ellipsis, 0..<qWidth].reshaped(1, count, c.heads, c.headDim)
        let gates = projected[.ellipsis, qWidth..<(2 * qWidth)]
        let key = projected[.ellipsis, (2 * qWidth)..<(2 * qWidth + kvWidth)]
            .reshaped(1, count, c.kvHeads, c.headDim)
        let value = projected[.ellipsis, (2 * qWidth + kvWidth)...]
            .reshaped(1, count, c.kvHeads, c.headDim).transposed(0, 2, 1, 3)
        let q = rope(qNorm(query).transposed(0, 2, 1, 3), offset: cache.offset)
        let k = rope(kNorm(key).transposed(0, 2, 1, 3), offset: cache.offset)
        let mask = createAttentionMask(h: mixed, cache: cache)
        let (keys, values) = cache.update(keys: k, values: value)
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: values,
            scale: 1 / sqrt(Float(c.headDim)), mask: mask).transposed(0, 2, 1, 3).reshaped(1, count, qWidth)
        let residual = mixed + output(attended * MLX.sigmoid(gates))
        let normalized = postNorm(residual)
        return finalNorm(residual + down(MLXNN.silu(gate(normalized)) * up(normalized)))
    }
}

final class Drama3Tower: Module {
    @ModuleInfo(key: "codebook_embeddings") var embeddings: Embedding
    @ModuleInfo var shared: [Drama3TowerLayer]
    @ModuleInfo var residual: [Drama3TowerLayer]
    @ModuleInfo var heads: [Linear]
    let config: Drama3Config

    init(_ c: Drama3Config) {
        config = c
        _embeddings.wrappedValue = Embedding(embeddingCount: c.audio.vocab * c.audio.codebooks, dimensions: c.audio.dim)
        _shared.wrappedValue = (0..<c.audio.sharedLayers).map { _ in Drama3TowerLayer(c.audio) }
        _residual.wrappedValue = (1..<c.audio.codebooks).map { _ in Drama3TowerLayer(c.audio) }
        _heads.wrappedValue = (1..<c.audio.codebooks).map { _ in Linear(c.audio.dim, c.audio.vocab, bias: false) }
    }

    func semantic(_ frames: MLXArray) -> MLXArray {
        let ids = frames[.ellipsis, 0]
        return MLX.logicalAnd(ids .>= config.semanticStart, ids .<= config.semanticEnd).asType(.bfloat16)
    }

    func backboneInput(_ frames: MLXArray, text: MLXArray) -> MLXArray {
        let code = frames[.ellipsis, 2] + config.audio.vocab
        let mixed = (text + embeddings(code)) / sqrt(Float(2))
        return MLX.where(semantic(frames).expandedDimensions(axis: -1) .> 0, mixed, text)
    }

    func extraInput(_ frames: MLXArray) -> MLXArray {
        let rows = (2..<config.audio.codebooks).map {
            embeddings(frames[.ellipsis, $0 + 1] + $0 * config.audio.vocab)
        }
        return MLX.sum(MLX.stacked(rows), axis: 0) * semantic(frames).expandedDimensions(axis: -1)
    }

    func makeCache() -> [KVCacheSimple] { (0..<(shared.count + residual.count)).map { _ in KVCacheSimple() } }

    func prefill(hidden: MLXArray, input: MLXArray, next: MLXArray, cache: [KVCacheSimple]) {
        var h = hidden
        let mask = semantic(next)
        for (i, layer) in shared.enumerated() {
            h = layer(h, code: next[.ellipsis, 1], semantic: mask, extra: i == 0 ? extraInput(input) : nil, cache: cache[i])
            MLX.eval(h)
        }
        for (i, layer) in residual.enumerated() {
            h = layer(h, code: next[.ellipsis, i + 1], semantic: mask, cache: cache[shared.count + i])
            MLX.eval(h)
        }
    }

    func decode(hidden: MLXArray, input: MLXArray, main: Int, sample: (MLXArray) -> Int,
                cache: [KVCacheSimple]) -> [Int32] {
        let isSemantic = (config.semanticStart...config.semanticEnd).contains(main)
        let mask = MLXArray(Float(isSemantic ? 1 : 0)).reshaped(1, 1)
        var codes = [Int32(max(0, min(config.audio.vocab - 1, main - config.semanticStart)))]
        var h = hidden
        for (i, layer) in shared.enumerated() {
            h = layer(h, code: MLXArray([codes[0]]).reshaped(1, 1), semantic: mask,
                extra: i == 0 ? extraInput(input) : nil, cache: cache[i])
            MLX.eval(h)
        }
        for (i, layer) in residual.enumerated() {
            h = layer(h, code: MLXArray([codes.last!]).reshaped(1, 1), semantic: mask, cache: cache[shared.count + i])
            MLX.eval(h)
            codes.append(Int32(sample(heads[i](h).reshaped(-1))))
        }
        return [Int32(main)] + codes
    }
}

#endif
