// Native Swift/MLX implementation of the Apache-2.0 Diffusers Qwen Image 2.1 transformer.
// Reference: huggingface/diffusers, 8d3c30bfda9b511c00992f40cff4170a5502814d.
import Foundation
import MLX
import MLXFast
import MLXNN

/// A cache belongs to one prompt/layout and one denoising trajectory, never to the model.
public final class QwenImage21PrefixCache {
    var layers: [(MLXArray, MLXArray)?]
    public init(layerCount: Int) { layers = Array(repeating: nil, count: layerCount) }
    public var isPopulated: Bool { layers.allSatisfy { $0 != nil } }
}

public final class QwenImage21Transformer {
    public let config: QwenImage21TransformerConfig
    private let weights: QwenImage21Weights

    public init(config: QwenImage21TransformerConfig, arrays: [String: MLXArray]) throws {
        guard config.patchSize == 1, config.numLayers > 0, config.numAttentionHeads > 0,
              config.inChannels > 0, config.outChannels > 0, config.contextInDim > 0, config.mlpRatio > 0,
              config.axesDimsRope.count == 3, config.axesDimsRope.allSatisfy({ $0 > 0 && $0.isMultiple(of: 2) }),
              config.axesDimsRope.reduce(0, +) == config.attentionHeadDim, config.eps > 0 else {
            throw QwenImage21Error.invalidConfiguration("Unsupported transformer dimensions.")
        }
        self.config = config
        weights = try QwenImage21Weights(arrays, shapes: Self.weightShapes(config))
    }

    public static func weightShapes(_ c: QwenImage21TransformerConfig) -> [String: [Int]] {
        let dim = c.hiddenSize
        var result: [String: [Int]] = [
            "img_in.weight": [dim, c.inChannels], "txt_in.text_norm.weight": [c.contextInDim],
            "txt_in.in_layer.weight": [dim, c.contextInDim], "txt_in.out_layer.weight": [dim, dim],
            "time_text_embed.timestep_embedder.linear_1.weight": [dim, 256],
            "time_text_embed.timestep_embedder.linear_2.weight": [dim, dim],
            "modulation.1.weight": [4 * dim, dim], "norm_out.linear.weight": [dim, dim],
            "proj_out.weight": [c.outChannels, dim]
        ]
        for index in 0..<c.numLayers {
            let prefix = "transformer_blocks.\(index)"
            for name in ["to_q", "to_k", "to_v", "to_out.0"] { result[prefix + ".attn.\(name).weight"] = [dim, dim] }
            for name in ["norm_q", "norm_k"] { result[prefix + ".attn.\(name).weight"] = [c.attentionHeadDim] }
            for name in ["proj", "gate_layer"] { result[prefix + ".img_mlp.\(name).weight"] = [dim * c.mlpRatio, dim] }
            result[prefix + ".img_mlp.out.weight"] = [dim, dim * c.mlpRatio]
        }
        return result
    }

    public func callAsFunction(
        latents: MLXArray, text: MLXArray, timestep: Float,
        layout: QwenImage21Layout, cache: QwenImage21PrefixCache? = nil
    ) throws -> MLXArray {
        guard latents.shape == [1, layout.imageIndices.filter({ $0 >= 0 }).count, config.inChannels],
              text.ndim == 3, text.dim(0) == 1, text.dim(1) > 0, text.dim(2) == config.contextInDim,
              layout.textIndices.max()! < text.dim(1), timestep.isFinite,
              cache == nil || (config.causalCondition && cache?.layers.count == config.numLayers) else {
            throw QwenImage21Error.invalidLayout("Transformer input or prefix cache dimensions do not match.")
        }
        let cached = cache?.isPopulated == true
        let start = cached ? layout.prefixCount : 0
        let image = weights.linear(latents, "img_in")
        var hidden: MLXArray
        if cached {
            hidden = image[0..., (latents.dim(1) - layout.targetCount)..., 0...]
        } else {
            let projected = weights.linear(geluApproximate(weights.linear(
                weights.rms(text, "txt_in.text_norm", epsilon: config.eps, zeroCentered: true), "txt_in.in_layer"
            )), "txt_in.out_layer")
            let textIDs = MLXArray(layout.textIndices.map { Int32(max(0, $0)) })
            let imageIDs = MLXArray(layout.imageIndices.map { Int32(max(0, $0)) })
            let mask = MLXArray(layout.imageIndices.map { $0 >= 0 }).reshaped(1, layout.count, 1)
            hidden = which(mask, take(image, imageIDs, axis: 1), take(projected, textIDs, axis: 1))
        }
        let time = timeEmbedding(timestep, dtype: hidden.dtype)
        let mod = weights.linear(silu(time), "modulation.1")
        let params = split(mod, parts: 4, axis: -1).map {
            modulation($0, count: hidden.dim(1), prefix: cached ? 0 : layout.prefixCount)
        }
        let rotary = rope(layout.positions, start: start)
        for index in 0..<config.numLayers {
            try Task.checkCancellation()
            let name = "transformer_blocks.\(index)"
            let normalized = QwenImage21Weights.layerNorm(hidden, epsilon: config.eps) * (1 + params[0])
            let attn = attention(normalized, name: name + ".attn", rotary: rotary,
                                 layout: layout, cached: cached, cache: cache, index: index)
            hidden = hidden + tanh(params[1]) * attn
            let input = QwenImage21Weights.layerNorm(hidden, epsilon: config.eps) * (1 + params[2])
            let mlp = weights.linear(silu(weights.linear(input, name + ".img_mlp.gate_layer"))
                                     * weights.linear(input, name + ".img_mlp.proj"), name + ".img_mlp.out")
            hidden = hidden + tanh(params[3]) * mlp
            if hidden.dtype == .float16 { hidden = clip(hidden, min: -65504, max: 65504) }
            eval(hidden)
        }
        let scale = modulation(weights.linear(silu(time), "norm_out.linear"), count: hidden.dim(1),
                               prefix: cached ? 0 : layout.prefixCount)
        let output = weights.linear(QwenImage21Weights.layerNorm(hidden, epsilon: config.eps) * (1 + scale), "proj_out")
        return output[0..., (output.dim(1) - layout.targetCount)..., 0...]
    }

    private func timeEmbedding(_ timestep: Float, dtype: DType) -> MLXArray {
        let frequencies = exp(MLXArray((0..<128).map { -log(Float(10000)) * Float($0) / 128 }))
        let time = MLXArray(config.causalCondition ? [timestep, 0] : [timestep])
            .asType(dtype).asType(.float32).expandedDimensions(axis: 1)
        let phases = time * 1000 * frequencies
        let embedding = concatenated([cos(phases), sin(phases)], axis: -1).asType(dtype)
        let prefix = "time_text_embed.timestep_embedder"
        return weights.linear(silu(weights.linear(embedding, prefix + ".linear_1")), prefix + ".linear_2")
    }

    private func modulation(_ x: MLXArray, count: Int, prefix: Int) -> MLXArray {
        if !config.causalCondition { return x.expandedDimensions(axis: 1) }
        let mask = MLXArray((0..<count).map { $0 >= prefix }).reshaped(1, count, 1)
        return which(mask, x[0].reshaped(1, 1, -1), x[1].reshaped(1, 1, -1))
    }

    private func rope(_ positions: [[Int]], start: Int) -> (MLXArray, MLXArray) {
        var phases: [Float] = []
        for position in positions.dropFirst(start) {
            for axis in 0..<3 {
                let dim = config.axesDimsRope[axis]
                phases += stride(from: 0, to: dim, by: 2).map {
                    Float(position[axis]) * pow(10000, -Float($0) / Float(dim))
                }
            }
        }
        let phase = MLXArray(phases).reshaped(1, positions.count - start, 1, config.attentionHeadDim / 2)
        return (cos(phase), sin(phase))
    }

    private func rotate(_ x: MLXArray, _ rotary: (MLXArray, MLXArray)) -> MLXArray {
        let pairs = x.asType(.float32).reshaped(x.dim(0), x.dim(1), x.dim(2), -1, 2)
        let even = pairs[0..., 0..., 0..., 0..., 0], odd = pairs[0..., 0..., 0..., 0..., 1]
        return stacked([even * rotary.0 - odd * rotary.1, even * rotary.1 + odd * rotary.0], axis: -1)
            .reshaped(x.shape).asType(x.dtype)
    }

    private func attention(
        _ x: MLXArray, name: String, rotary: (MLXArray, MLXArray), layout: QwenImage21Layout,
        cached: Bool, cache: QwenImage21PrefixCache?, index: Int
    ) -> MLXArray {
        func projection(_ key: String) -> MLXArray {
            weights.linear(x, name + "." + key).reshaped(1, x.dim(1), config.numAttentionHeads, config.attentionHeadDim)
        }
        let query = rotate(weights.rms(projection("to_q"), name + ".norm_q", epsilon: config.eps), rotary).transposed(0, 2, 1, 3)
        var key = rotate(weights.rms(projection("to_k"), name + ".norm_k", epsilon: config.eps), rotary).transposed(0, 2, 1, 3)
        var value = projection("to_v").transposed(0, 2, 1, 3)
        if cached, let (oldKey, oldValue) = cache?.layers[index] {
            key = concatenated([oldKey, key], axis: 2)
            value = concatenated([oldValue, value], axis: 2)
        } else if let cache {
            cache.layers[index] = (contiguous(key[0..., 0..., 0..<layout.prefixCount, 0...]),
                                   contiguous(value[0..., 0..., 0..<layout.prefixCount, 0...]))
            eval(cache.layers[index]!.0, cache.layers[index]!.1)
        }
        let scale = 1 / sqrt(Float(config.attentionHeadDim))
        let result: MLXArray
        if cached {
            result = MLXFast.scaledDotProductAttention(queries: query, keys: key, values: value, scale: scale, mask: .none)
        } else {
            result = concatenated(layout.segments.map { segment in
                let range = segment.range
                let mask: MLXFast.ScaledDotProductAttentionMaskMode
                if segment.image { mask = .none } else {
                    let q = MLXArray(range.map(Int32.init)).expandedDimensions(axis: 1)
                    let k = MLXArray((0..<range.upperBound).map(Int32.init)).expandedDimensions(axis: 0)
                    mask = .array(q .>= k)
                }
                return MLXFast.scaledDotProductAttention(
                    queries: query[0..., 0..., range, 0...], keys: key[0..., 0..., 0..<range.upperBound, 0...],
                    values: value[0..., 0..., 0..<range.upperBound, 0...], scale: scale, mask: mask)
            }, axis: 2)
        }
        return weights.linear(result.transposed(0, 2, 1, 3).reshaped(1, x.dim(1), config.hiddenSize), name + ".to_out.0")
    }
}
