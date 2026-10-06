import Foundation
import MLX
import MLXFast
import MLXNN

/// Gemma 4 audio conformer: stride-four subsampling, blocked relative attention,
/// clipped projections, two feed-forward residuals, and causal depthwise convolution.
public final class EmbeddingGemma2AudioModel {
    public let config: EmbeddingGemma2AudioConfig
    private let weights: EmbeddingGemma2MediaTensorStore

    public init(config: EmbeddingGemma2AudioConfig, textHiddenSize: Int,
                tensors: [String: MLXArray], dtype: DType = .bfloat16) throws {
        try config.validate()
        self.config = config
        weights = try EmbeddingGemma2MediaTensorStore(tensors: tensors,
            shapes: Self.tensorShapes(config: config, textHiddenSize: textHiddenSize), dtype: dtype)
    }

    public static func tensorShapes(config: EmbeddingGemma2AudioConfig, textHiddenSize: Int) -> [String: [Int]] {
        let h = config.hiddenSize, base = "audio_tower.subsample_conv_projection"
        var shapes = [
            base + ".layer0.conv.weight": [config.subsamplingConvChannels[0], 1, 3, 3],
            base + ".layer1.conv.weight": [config.subsamplingConvChannels[1], config.subsamplingConvChannels[0], 3, 3],
            base + ".layer0.norm.weight": [config.subsamplingConvChannels[0]],
            base + ".layer1.norm.weight": [config.subsamplingConvChannels[1]],
            base + ".input_proj_linear.weight": [h, 32 * config.subsamplingConvChannels[1]],
            "audio_tower.output_proj.weight": [config.outputProjDims, h],
            "audio_tower.output_proj.bias": [config.outputProjDims],
            "embed_audio.embedding_projection.weight": [textHiddenSize, config.outputProjDims]
        ]
        for index in 0..<config.numHiddenLayers {
            let key = "audio_tower.layers.\(index)"
            for name in ["norm_pre_attn", "norm_post_attn", "norm_out"] { shapes[key + "." + name + ".weight"] = [h] }
            for name in ["feed_forward1", "feed_forward2"] {
                for norm in ["pre_layer_norm", "post_layer_norm"] { shapes[key + "." + name + "." + norm + ".weight"] = [h] }
                shapes.merge(EmbeddingGemma2MediaTensorStore.linearShapes(key + "." + name + ".ffw_layer_1", input: h, output: h * 4, clipped: true)) { _, new in new }
                shapes.merge(EmbeddingGemma2MediaTensorStore.linearShapes(key + "." + name + ".ffw_layer_2", input: h * 4, output: h, clipped: true)) { _, new in new }
            }
            for name in ["q_proj", "k_proj", "v_proj", "post"] {
                shapes.merge(EmbeddingGemma2MediaTensorStore.linearShapes(key + ".self_attn." + name, input: h, output: h, clipped: true)) { _, new in new }
            }
            shapes[key + ".self_attn.relative_k_proj.weight"] = [h, h]
            shapes[key + ".self_attn.per_dim_scale"] = [h / config.numAttentionHeads]
            for name in ["pre_layer_norm", "conv_norm"] { shapes[key + ".lconv1d." + name + ".weight"] = [h] }
            shapes[key + ".lconv1d.depthwise_conv1d.weight"] = [h, 1, config.convKernelSize]
            shapes.merge(EmbeddingGemma2MediaTensorStore.linearShapes(key + ".lconv1d.linear_start", input: h, output: 2 * h, clipped: true)) { _, new in new }
            shapes.merge(EmbeddingGemma2MediaTensorStore.linearShapes(key + ".lconv1d.linear_end", input: h, output: h, clipped: true)) { _, new in new }
        }
        return shapes
    }

    public func callAsFunction(features: MLXArray, validFrames: [Bool]) throws -> MLXArray {
        guard features.ndim == 3, features.dim(0) == 1, features.dim(2) == 128,
              features.dim(1) == validFrames.count, validFrames.contains(true) else {
            throw EmbeddingGemma2Error.invalidInput("Invalid audio spectrogram or frame mask.")
        }
        var mask = validFrames
        var hidden = features.asType(weights.dtype).expandedDimensions(axis: -1)
        let base = "audio_tower.subsample_conv_projection"
        for index in 0..<2 {
            let key = base + ".layer\(index)"
            hidden = hidden * MLXArray(mask.map { Float($0 ? 1 : 0) }, [1, mask.count, 1, 1]).asType(hidden.dtype)
            hidden = MLX.conv2d(hidden, weights[key + ".conv.weight"].transposed(0, 2, 3, 1), stride: 2, padding: 1)
            hidden = maximum(MLXFast.layerNorm(hidden, weight: weights[key + ".norm.weight"], bias: nil, eps: config.rmsNormEps), MLXArray(Float(0)))
            mask = stride(from: 0, to: mask.count, by: 2).map { mask[$0] }
        }
        hidden = weights.linear(hidden.reshaped(1, hidden.dim(1), -1), base + ".input_proj_linear")
        let positions = relativePositions(dtype: hidden.dtype)
        for index in 0..<config.numHiddenLayers {
            let key = "audio_tower.layers.\(index)"
            hidden = feedForward(hidden, key: key + ".feed_forward1")
            let residual = hidden
            let x = weights.norm(clamp(hidden), key + ".norm_pre_attn", eps: 1e-6)
            hidden = residual + weights.norm(clamp(attention(x, key: key + ".self_attn", positions: positions, valid: mask)), key + ".norm_post_attn", eps: 1e-6)
            hidden = lightConv(hidden, key: key + ".lconv1d")
            hidden = weights.norm(clamp(feedForward(hidden, key: key + ".feed_forward2")), key + ".norm_out", eps: 1e-6)
        }
        let projected = weights.linear(hidden, "audio_tower.output_proj", bias: true)
        let normalized = EmbeddingGemma2RMSNorm.normalize(projected, eps: config.rmsNormEps).asType(hidden.dtype)
        let embedded = weights.linear(normalized, "embed_audio.embedding_projection")
        let real = mask.enumerated().compactMap { $0.element ? Int32($0.offset) : nil }
        return take(embedded[0], MLXArray(real), axis: 0)
    }

    private func clamp(_ x: MLXArray) -> MLXArray {
        clip(x, min: -config.gradientClipping, max: config.gradientClipping)
    }

    private func feedForward(_ x: MLXArray, key: String) -> MLXArray {
        var h = weights.norm(clamp(x), key + ".pre_layer_norm", eps: 1e-6)
        h = silu(weights.clippedLinear(h, key + ".ffw_layer_1"))
        h = weights.norm(clamp(weights.clippedLinear(h, key + ".ffw_layer_2")), key + ".post_layer_norm", eps: 1e-6)
        return x + h * MLXArray(config.residualWeight).asType(h.dtype)
    }

    private func lightConv(_ x: MLXArray, key: String) -> MLXArray {
        var h = weights.clippedLinear(weights.norm(x, key + ".pre_layer_norm", eps: config.rmsNormEps), key + ".linear_start")
        let size = config.hiddenSize
        h = h[0..., 0..., 0..<size] * sigmoid(h[0..., 0..., size...])
        h = concatenated([MLXArray.zeros([1, config.convKernelSize - 1, size], dtype: h.dtype), h], axis: 1)
        h = MLX.conv1d(h, weights[key + ".depthwise_conv1d.weight"].transposed(0, 2, 1), groups: size)
        h = silu(weights.norm(clamp(h), key + ".conv_norm", eps: config.rmsNormEps))
        return x + weights.clippedLinear(h, key + ".linear_end")
    }

    private func relativePositions(dtype: DType) -> MLXArray {
        let half = config.hiddenSize / 2
        let context = config.attentionChunkSize + config.attentionContextLeft - 1
        let inverse = MLXArray((0..<half).map { Float(exp(-Double($0) * log(10_000) / Double(max(half - 1, 1)))) })
        let positions = MLXArray((0...(context / 2)).reversed().map(Float.init)).expandedDimensions(axis: -1) * inverse
        return concatenated([sin(positions), cos(positions)], axis: -1).expandedDimensions(axis: 0).asType(dtype)
    }

    private func attention(_ x: MLXArray, key: String, positions: MLXArray, valid: [Bool]) -> MLXArray {
        let length = x.dim(1), chunk = config.attentionChunkSize, left = config.attentionContextLeft - 1
        let context = chunk + left, blocks = (length + chunk - 1) / chunk
        let heads = config.numAttentionHeads, dim = config.hiddenSize / heads
        let qScale = Float(pow(Double(dim), -0.5) / log(Double(2)))
        let kScale = Float(log(1 + exp(Double(1))) / log(Double(2)))
        var q = weights.clippedLinear(x, key + ".q_proj").asType(.float32).reshaped(1, length, heads, dim)
        q = q * MLXArray(qScale) * log(1 + exp(weights[key + ".per_dim_scale"].asType(.float32)))
        q = concatenated([q, MLXArray.zeros([1, blocks * chunk - length, heads, dim])], axis: 1)
            .reshaped(1, blocks, chunk, heads, dim).transposed(0, 3, 1, 2, 4)
        let indices = (0..<blocks).flatMap { block in (0..<context).map { Int32(block * chunk + $0) } }
        func contexts(_ name: String, scale: Float) -> MLXArray {
            let values = weights.clippedLinear(x, key + "." + name).asType(.float32).reshaped(1, length, heads, dim) * MLXArray(scale)
            let padded = concatenated([MLXArray.zeros([1, left, heads, dim]), values,
                                       MLXArray.zeros([1, chunk - 1, heads, dim])], axis: 1)
            return take(padded, MLXArray(indices), axis: 1).reshaped(1, blocks, context, heads, dim).transposed(0, 3, 1, 2, 4)
        }
        let keys = contexts("k_proj", scale: kScale), values = contexts("v_proj", scale: 1)
        let relative = weights.linear(positions, key + ".relative_k_proj").asType(.float32)
            .reshaped(-1, heads, dim).transposed(1, 2, 0)
        var bias = matmul(q.reshaped(1, heads, blocks * chunk, dim), relative).reshaped(1, heads, blocks, chunk, -1)
        bias = concatenated([bias, MLXArray.zeros([1, heads, blocks, chunk, context + 1 - bias.dim(-1)])], axis: -1)
            .reshaped(1, heads, blocks, chunk * (context + 1))[0..., 0..., 0..., 0..<(chunk * context)]
            .reshaped(1, heads, blocks, chunk, context)
        var allowed: [Bool] = []
        for block in 0..<blocks {
            for query in 0..<chunk {
                let qi = block * chunk + query
                for offset in 0..<context {
                    let ki = block * chunk - left + offset
                    allowed.append(qi < length && ki >= 0 && ki < length && qi - ki >= 0 && qi - ki < left && valid[ki])
                }
            }
        }
        var logits = matmul(q, keys.transposed(0, 1, 2, 4, 3)) + bias
        logits = tanh(logits / MLXArray(config.attentionLogitCap)) * MLXArray(config.attentionLogitCap)
        logits = MLX.where(MLXArray(allowed, [1, 1, blocks, chunk, context]), logits, MLXArray(config.attentionInvalidLogitsValue))
        let output = matmul(softmax(logits, axis: -1), values).transposed(0, 2, 3, 1, 4).reshaped(1, blocks * chunk, -1)
        return weights.clippedLinear(output[0..., 0..<length, 0...].asType(x.dtype), key + ".post")
    }
}
