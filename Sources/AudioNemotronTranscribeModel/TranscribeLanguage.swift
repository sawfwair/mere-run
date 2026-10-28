import Foundation
import MLX
import MLXFast
import MLXNN

public final class NemotronTranscribeCache {
    var convolution: [MLXArray?]
    var recurrent: [MLXArray?]
    var keys: [MLXArray?]
    var values: [MLXArray?]
    public private(set) var offset = 0

    init(count: Int) {
        convolution = Array(repeating: nil, count: count)
        recurrent = Array(repeating: nil, count: count)
        keys = Array(repeating: nil, count: count)
        values = Array(repeating: nil, count: count)
    }

    func advance(_ count: Int) { offset += count }
}

struct TranscribeLanguage {
    let weights: TranscribeWeights
    let config: NemotronTranscribeConfig.Language

    func inputEmbeddings(_ tokens: MLXArray) -> MLXArray { weights["llm.model.embed_tokens.weight"][tokens] }

    func callAsFunction(embeddings: MLXArray, cache: NemotronTranscribeCache) throws -> MLXArray {
        var hidden = embeddings
        for (index, type) in config.layersBlockType.enumerated() {
            try Task.checkCancellation()
            let prefix = "llm.model.layers.\(index)"
            let norm = weights.rmsNorm(hidden, prefix + ".norm.weight", eps: config.normEps)
            let mixed: MLXArray
            switch type {
            case "mamba": mixed = mamba(norm, prefix: prefix + ".mixer", index: index, cache: cache)
            case "attention": mixed = attention(norm, prefix: prefix + ".mixer", index: index, cache: cache)
            case "moe": mixed = try experts(norm, prefix: prefix + ".mixer")
            default: preconditionFailure("Validated Transcribe block type")
            }
            hidden = hidden + mixed
            // Bound the Metal graph and retain FP32 recurrence across calls.
            if embeddings.dim(1) > 1 { eval(hidden) }
        }
        cache.advance(embeddings.dim(1))
        let last = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        return weights.linear(weights.rmsNorm(last, "llm.model.norm.weight", eps: config.normEps), "llm.lm_head")
    }

    private func mamba(_ x: MLXArray, prefix: String, index: Int, cache: NemotronTranscribeCache) -> MLXArray {
        let c = config
        let inner = c.mambaNumHeads * c.mambaHeadDim
        let convSize = inner + 2 * c.nGroups * c.ssmStateSize
        let projected = weights.linear(x, prefix + ".in_proj")
        let gate = projected[.ellipsis, ..<inner]
        let convInput = projected[.ellipsis, inner..<(inner + convSize)]
        let dt = projected[.ellipsis, (inner + convSize)...]
        let previous = cache.convolution[index] ?? MLXArray.zeros([x.dim(0), c.convKernel - 1, convSize], dtype: x.dtype)
        let padded = concatenated([previous, convInput], axis: 1)
        cache.convolution[index] = padded[0..., (padded.dim(1) - c.convKernel + 1)..., 0...]
        let convolved = silu(MLX.conv1d(padded, weights[prefix + ".conv1d.weight"].transposed(0, 2, 1), groups: convSize)
                             + weights[prefix + ".conv1d.bias"])
        let bEnd = inner + c.nGroups * c.ssmStateSize
        let result = transcribeSSMUpdate(
            x: convolved[.ellipsis, ..<inner].reshaped(x.dim(0), x.dim(1), c.mambaNumHeads, c.mambaHeadDim),
            b: convolved[.ellipsis, inner..<bEnd].reshaped(x.dim(0), x.dim(1), c.nGroups, c.ssmStateSize),
            c: convolved[.ellipsis, bEnd...].reshaped(x.dim(0), x.dim(1), c.nGroups, c.ssmStateSize),
            dt: dt, aLog: weights[prefix + "._fp32_params.A_log"], d: weights[prefix + "._fp32_params.D"],
            dtBias: weights[prefix + "._fp32_params.dt_bias"], state: cache.recurrent[index],
            minimumTimeStep: 0, maximumTimeStep: Float.greatestFiniteMagnitude)
        cache.recurrent[index] = result.1
        let gated = silu(gate.asType(.float32)) * result.0.reshaped(x.dim(0), x.dim(1), inner).asType(.float32)
        let grouped = gated.reshaped(x.dim(0), x.dim(1), c.nGroups, inner / c.nGroups)
        let normalized = grouped * rsqrt(square(grouped).mean(axis: -1, keepDims: true) + c.normEps)
        let output = normalized.reshaped(gated.shape) * weights[prefix + ".norm.weight"].asType(.float32)
        return weights.linear(output.asType(x.dtype), prefix + ".out_proj")
    }

    private func attention(_ x: MLXArray, prefix: String, index: Int, cache: NemotronTranscribeCache) -> MLXArray {
        let c = config
        func heads(_ name: String, _ count: Int) -> MLXArray {
            weights.linear(x, prefix + name).reshaped(x.dim(0), x.dim(1), count, c.headDim).transposed(0, 2, 1, 3)
        }
        let query = heads(".q_proj", c.numAttentionHeads)
        var key = heads(".k_proj", c.numKeyValueHeads)
        var value = heads(".v_proj", c.numKeyValueHeads)
        if let previous = cache.keys[index], let previousValue = cache.values[index] {
            key = concatenated([previous, key], axis: 2)
            value = concatenated([previousValue, value], axis: 2)
        }
        cache.keys[index] = key
        cache.values[index] = value
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if x.dim(1) == 1 { mask = .none }
        else {
            let queries = MLXArray(cache.offset..<(cache.offset + x.dim(1))).reshaped(x.dim(1), 1)
            let keys = MLXArray(0..<key.dim(2)).reshaped(1, key.dim(2))
            mask = .array(MLX.where(keys .<= queries, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(x.dtype))
        }
        let output = MLXFast.scaledDotProductAttention(queries: query, keys: key, values: value,
                                                     scale: 1 / sqrt(Float(c.headDim)), mask: mask)
        return weights.linear(output.transposed(0, 2, 1, 3).reshaped(x.dim(0), x.dim(1), c.numAttentionHeads * c.headDim), prefix + ".o_proj")
    }

    private func experts(_ x: MLXArray, prefix: String) throws -> MLXArray {
        // BF16 gathers materialize selected matrices. Packed QMM keeps weights shared across positions.
        var pieces: [MLXArray] = []
        let chunkSize = weights.quantization == nil ? 4 : 128
        for start in stride(from: 0, to: x.dim(1), by: chunkSize) {
            try Task.checkCancellation()
            let input = x[0..., start..<min(start + chunkSize, x.dim(1)), 0...]
            let route = Self.route(input, weight: weights[prefix + ".gate.weight"],
                                  bias: weights[prefix + ".gate.e_score_correction_bias"],
                                  topK: config.numExpertsPerTok, scale: config.routedScalingFactor)
            let up = weights.routedLinear(input.expandedDimensions(axis: -2).expandedDimensions(axis: -2),
                indices: route.0, key: prefix + ".experts.gate_and_up_projs").squeezed(axis: -2)
            let down = weights.routedLinear(square(relu(up)).expandedDimensions(axis: -2),
                indices: route.0, key: prefix + ".experts.down_projs").squeezed(axis: -2)
            let routed = (down * route.1.expandedDimensions(axis: -1).asType(down.dtype)).sum(axis: -2)
            let shared = weights.linear(square(relu(weights.linear(input, prefix + ".shared_experts.up_proj"))), prefix + ".shared_experts.down_proj")
            let output = routed + shared
            eval(output)
            pieces.append(output)
        }
        return concatenated(pieces, axis: 1)
    }

    static func route(_ input: MLXArray, weight: MLXArray, bias: MLXArray, topK: Int, scale: Float) -> (MLXArray, MLXArray) {
        let scores = sigmoid(input.asType(.float32).matmul(weight.asType(.float32).T))
        let indices = stopGradient(argPartition(-(scores + bias), kth: topK - 1, axis: -1)[.ellipsis, ..<topK])
        let selected = takeAlong(scores, indices, axis: -1)
        return (indices, selected / (selected.sum(axis: -1, keepDims: true) + Float(1e-20)) * scale)
    }
}
