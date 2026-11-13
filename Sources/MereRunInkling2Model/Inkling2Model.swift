import Foundation
import MLX

public final class Inkling2Model {
    public let config: Inkling2Configuration
    let weights: Inkling2Weights
    let prefix = "model.llm."

    public init(config: Inkling2Configuration, weights: Inkling2Weights) throws {
        try config.validate()
        self.config = config
        self.weights = weights
        try validateWeights()
    }

    func linear(_ x: MLXArray, _ key: String, bias: Bool = false) throws -> MLXArray {
        let weight = try weights.read(key + ".weight")
        let output = matmul(x.asType(weight.dtype), weight.T)
        return bias ? try output + weights.read(key + ".bias") : output
    }

    func norm(_ x: MLXArray, _ key: String) throws -> MLXArray {
        try (Inkling2Math.rms(x, epsilon: config.textConfig.rmsNormEps)
             * weights.read(key + ".weight").asType(.float32)).asType(x.dtype)
    }

    public func embeddings(_ tokens: MLXArray) throws -> MLXArray {
        try norm(weights.read(prefix + "embed.weight")[tokens], prefix + "embed_norm")
    }

    public func callAsFunction(_ tokens: MLXArray, cache: Inkling2Cache, lastPositionOnly: Bool = true) throws -> MLXArray {
        try forward(embeddings: embeddings(tokens), cache: cache, lastPositionOnly: lastPositionOnly)
    }

    public func forward(embeddings input: MLXArray, cache: Inkling2Cache, lastPositionOnly: Bool = true) throws -> MLXArray {
        let c = config.textConfig
        guard cache.layers.count == c.numHiddenLayers else { throw Inkling2Error.invalid("cache layer count mismatch") }
        var hidden = input
        var blocks: [MLXArray] = []
        for i in 0..<c.numHiddenLayers {
            try Task.checkCancellation()
            let p = prefix + "layers.\(i)."
            let state = cache.layers[i]
            let residual = hidden
            let attnInputs: [MLXArray]
            if i == 0 { attnInputs = [hidden, hidden, hidden] }
            else {
                let query = try weights.read(prefix + "attn_res.attn_queries.\(i - 1)")
                let mixed = Inkling2Math.residual(blocks, partial: hidden, query: query)
                attnInputs = c.isLocal(i) ? [mixed[0], mixed[0], mixed[0]] : mixed
            }
            let normalized: [MLXArray]
            if c.isLocal(i) {
                let x = try norm(attnInputs[0], p + "attn_norm").asType(input.dtype)
                normalized = [x, x, x]
            } else {
                normalized = try zip(attnInputs, ["q", "k", "v"]).map { try norm($0.0, p + $0.1 + "_input_norm").asType(input.dtype) }
            }
            var attended = try attention(normalized, layer: i, cache: state)
            attended = try convolution(attended, key: p + "attn_sconv", cache: state, index: 2)
            let boundary = i.isMultiple(of: c.nLayersPerAttnResBlock)
            let partial = boundary ? attended.asType(.float32) : residual.asType(.float32) + attended.asType(.float32)
            if boundary { blocks.append(residual) }
            let mixed = Inkling2Math.residual(blocks, partial: partial,
                                            query: try weights.read(prefix + "attn_res.mlp_queries.\(i)"))[0]
            let mlpInput = try norm(mixed, p + "mlp_norm").asType(input.dtype)
            let output = try convolution(moe(mlpInput, layer: i), key: p + "mlp_sconv", cache: state, index: 3)
            hidden = (partial + output.asType(.float32)).asType(input.dtype)
            eval([hidden] + blocks + Array(state.convolutions.values) + [state.keys, state.values].compactMap { $0 })
        }
        hidden = Inkling2Math.residual(blocks, partial: hidden,
                                      query: try weights.read(prefix + "attn_res.output_query"))[0]
        hidden = try norm(hidden, prefix + "norm").asType(input.dtype)
        if lastPositionOnly { hidden = hidden[0..., (hidden.dim(1) - 1)..<hidden.dim(1), 0...] }
        let logits = try linear(hidden, prefix + "unembed").asType(.float32)
        return Inkling2Math.softcap(logits, c.finalLogitSoftcapping)[.ellipsis, ..<c.unpaddedVocabSize]
    }

    func convolution(_ x: MLXArray, key: String, cache: Inkling2LayerCache, index: Int,
                     dynamicInput: MLXArray? = nil, heads: Int = 1) throws -> MLXArray {
        let kernel = config.textConfig.sconvKernelSize
        let context = cache.context(x, index: index, kernel: kernel)
        var output = x.asType(.float32)
        if let dynamicInput {
            let staticWeight = try weights.read(key + ".static_w").asType(.float32)
            let projection = try weights.read(key + ".weight_proj")
            let dynamic = matmul(dynamicInput.asType(projection.dtype), projection.T).asType(.float32)
                .reshaped(x.dim(0), x.dim(1), heads, 1, kernel)
            for lag in 0..<kernel {
                let tap = staticWeight[lag].reshaped(1, 1, heads, x.dim(2) / heads)
                    + dynamic[0..., 0..., 0..., 0..., lag]
                let start = kernel - 1 - lag
                output = output + (context[0..., start..<(start + x.dim(1)), 0...]
                    .reshaped(x.dim(0), x.dim(1), heads, x.dim(2) / heads) * tap).reshaped(x.shape)
            }
        } else {
            let taps = try weights.read(key + ".weight").asType(.float32).reshaped(x.dim(2), kernel)
            for tap in 0..<kernel {
                output = output + context[0..., tap..<(tap + x.dim(1)), 0...] * taps[0..., tap]
            }
        }
        return (output / sqrt(Float((dynamicInput == nil ? 1 : 2) * kernel + 1))).asType(x.dtype)
    }

    func attention(_ inputs: [MLXArray], layer: Int, cache: Inkling2LayerCache) throws -> MLXArray {
        let c = config.textConfig, local = c.isLocal(layer)
        let p = prefix + "layers.\(layer).attn."
        let x = inputs[0], b = x.dim(0), n = x.dim(1)
        let h = local ? c.swaNumAttentionHeads : c.numAttentionHeads
        let kv = local ? c.swaNumKeyValueHeads : c.numKeyValueHeads
        let d = local ? c.swaHeadDim : c.headDim
        let extent = local ? c.slidingWindowSize : c.relExtent
        let offset = cache.offset
        var q = try linear(x, p + "wq_du").reshaped(b, n, h, d)
        var k = try convolution(linear(inputs[1], p + "wk_dv"), key: p + "k_sconv", cache: cache, index: 0,
                                dynamicInput: local ? nil : inputs[1], heads: kv).reshaped(b, n, kv, d)
        var v = try convolution(linear(inputs[2], p + "wv_dv"), key: p + "v_sconv", cache: cache, index: 1,
                                dynamicInput: local ? nil : inputs[2], heads: kv).reshaped(b, n, kv, d)
        var r = try linear(x, p + "wr_du").reshaped(b, n, h, c.dRel)
        q = Inkling2Math.clip(q, scale: try weights.read(p + "q_l2_clip.scale"), threshold: c.qkL2ClipThreshold)
        k = Inkling2Math.clip(k, threshold: c.qkL2ClipThreshold)
        r = Inkling2Math.clip(r, scale: try weights.read(p + "rel_l2_clip.scale"), threshold: c.relL2ClipThreshold)
        v = Inkling2Math.rms(v, epsilon: c.rmsNormEps).asType(x.dtype)
        if !local {
            let positions = MLXArray(Int32(offset + 1)..<Int32(offset + n + 1)).asType(.float32)
            let tau = (1 + c.logScalingAlpha * log(maximum(positions / Float(c.logScalingNFloor), MLXArray(Float(1)))))
                .reshaped(1, n, 1, 1)
            q = (q.asType(.float32) * tau).asType(x.dtype)
            r = (r.asType(.float32) * tau).asType(x.dtype)
        }
        q = q.transposed(0, 2, 1, 3)
        let stored = cache.update(keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
                                  window: local ? c.slidingWindowSize : nil)
        let positions = MLXArray(Int32(offset)..<Int32(offset + n)).reshaped(n, 1)
        let keyPositions = MLXArray(Int32(stored.2)..<Int32(stored.2 + stored.0.dim(2))).reshaped(1, -1)
        let distance = positions - keyPositions
        let rel = try matmul(r.asType(.float32), weights.read(p + "rel_logits_proj.proj").asType(.float32)).transposed(0, 2, 1, 3)
        let indices = broadcast(clip(distance, min: 0, max: extent - 1).asType(.int32), to: [b, h, n, stored.0.dim(2)])
        let bias = MLX.where((distance .>= 0) .&& (distance .< extent), takeAlong(rel, indices, axis: -1), MLXArray(Float(0)))
        let expandedK = repeated(stored.0, count: h / kv, axis: 1).asType(.float32)
        let expandedV = repeated(stored.1, count: h / kv, axis: 1).asType(.float32)
        var scores = matmul(q.asType(.float32), expandedK.swappedAxes(-1, -2)) / sqrt(Float(d)) + bias
        let allowed = local ? (distance .>= 0) .&& (distance .< c.slidingWindowSize) : distance .>= 0
        scores = MLX.where(allowed, scores, MLXArray(-Float.infinity))
        let sink = try linear(x, p + "attn_sink_proj").transposed(0, 2, 1).expandedDimensions(axis: -1)
        let probabilities = softmax(concatenated([scores, sink.asType(.float32)], axis: -1), axis: -1)
        let attended = matmul(probabilities[.ellipsis, ..<stored.0.dim(2)].asType(x.dtype), expandedV.asType(x.dtype))
            .transposed(0, 2, 1, 3).reshaped(b, n, h * d)
        let gate = try sigmoid(linear(x, p + "gate_attn").asType(.float32))
        return try linear((attended.asType(.float32) * gate).asType(x.dtype), p + "wo_ud")
    }

    func moe(_ x: MLXArray, layer: Int) throws -> MLXArray {
        let c = config.textConfig, p = prefix + "layers.\(layer).mlp."
        let flat = x.reshaped(-1, c.hiddenSize)
        let router = try weights.read(p + "gate.weight").asType(.float32)
        let scores = sigmoid(matmul(flat.asType(.float32), router.T))
        let corrected = try scores + weights.read(p + "gate.bias")
        let selected = argSort(-corrected, axis: -1)[0..., ..<c.numExpertsPerTok]
        var routing = takeAlong(scores, selected, axis: -1)
        routing = routing / (routing.sum(axis: -1, keepDims: true) + 1e-20) * c.routeScale
        let indices = selected.asArray(Int32.self)
        var outputs: [MLXArray] = []
        for token in 0..<flat.dim(0) {
            var result = MLXArray.zeros([1, c.hiddenSize], dtype: .float32)
            for choice in 0..<c.numExpertsPerTok {
                try Task.checkCancellation()
                let expert = Int(indices[token * c.numExpertsPerTok + choice])
                let w13 = try weights.expert(p + "experts.w13_weight", index: expert).asType(x.dtype)
                let projected = matmul(flat[token..<(token + 1)], w13.T).reshaped(1, c.intermediateSize, 2)
                let activated = Inkling2Math.activation(gate: projected[0..., 0..., 0], up: projected[0..., 0..., 1],
                                                       gateCap: c.mlpGateSoftcap, upCap: c.mlpUpSoftcap)
                let w2 = try weights.expert(p + "experts.w2_weight", index: expert).asType(x.dtype)
                result = result + matmul(activated, w2.T).asType(.float32) * routing[token, choice]
                // Bound the live graph to one expert; no full-model dequantization.
                eval(result)
            }
            outputs.append(result)
        }
        return concatenated(outputs, axis: 0).reshaped(x.shape).asType(x.dtype)
    }
}
