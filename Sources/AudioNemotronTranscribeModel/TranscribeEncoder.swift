import Foundation
import MLX
import MLXFast
import MLXNN

/// NeMo Transformer encoder. Input and output are [batch, time, channels].
struct TranscribeEncoder {
    let weights: TranscribeWeights
    let prefix: String
    let width: Int
    let heads: Int
    let layers: Int
    let rotaryDimensions: Int
    let qkNorm: Bool

    func callAsFunction(_ features: MLXArray) throws -> MLXArray {
        var frames = features
        let padding = (8 - frames.dim(1) % 8) % 8
        if padding > 0 { frames = MLX.padded(frames, widths: [.init((0, 0)), .init((0, padding)), .init((0, 0))]) }
        var hidden = weights.linear(frames.reshaped(frames.dim(0), frames.dim(1) / 8, frames.dim(2) * 8), prefix + ".pre_encode.proj")
        hidden = weights.layerNorm(hidden, prefix + ".embed_norm")
        for index in 0..<layers {
            try Task.checkCancellation()
            hidden = block(hidden, index: index)
            eval(hidden)
        }
        return weights.layerNorm(hidden, prefix + ".final_norm")
    }

    static func rotate(_ x: MLXArray, dimensions: Int) -> MLXArray {
        let positions = MLXArray(0..<x.dim(2)).asType(.float32).reshaped(x.dim(2), 1)
        let frequencies = exp(MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                              * Float(-log(10_000.0) / Double(dimensions)))
        let angles = positions * frequencies
        // NeMo caches cosine/sine in model dtype before doing Q/K arithmetic.
        let cosine = concatenated([cos(angles), cos(angles)], axis: -1).asType(x.dtype)
        let sine = concatenated([sin(angles), sin(angles)], axis: -1).asType(x.dtype)
        let first = x[.ellipsis, ..<(dimensions / 2)]
        let second = x[.ellipsis, (dimensions / 2)..<dimensions]
        let rotated = x[.ellipsis, ..<dimensions] * cosine + concatenated([-second, first], axis: -1) * sine
        return dimensions == x.dim(-1) ? rotated : concatenated([rotated, x[.ellipsis, dimensions...]], axis: -1)
    }

    func block(_ input: MLXArray, index: Int) -> MLXArray {
        let p = prefix + ".layers.\(index)"
        let normalized = weights.layerNorm(input, p + ".norm1")
        let projections = MLX.split(weights.linear(normalized, p + ".attn.w_qkv"), parts: 3, axis: -1)
        let depth = width / heads
        func head(_ x: MLXArray) -> MLXArray {
            x.reshaped(x.dim(0), x.dim(1), heads, depth).transposed(0, 2, 1, 3).contiguous()
        }
        var query = head(projections[0])
        var key = head(projections[1])
        let value = head(projections[2])
        if qkNorm {
            query = weights.layerNorm(query, p + ".attn.q_norm")
            key = weights.layerNorm(key, p + ".attn.k_norm")
        }
        query = Self.rotate(query, dimensions: rotaryDimensions)
        key = Self.rotate(key, dimensions: rotaryDimensions)
        let attention = MLXFast.scaledDotProductAttention(queries: query, keys: key, values: value,
                                                        scale: 1 / sqrt(Float(depth)), mask: .none)
        let residual = input + weights.linear(attention.transposed(0, 2, 1, 3).reshaped(input.shape), p + ".attn.out_proj", bias: true)
        let ff = weights.linear(weights.layerNorm(residual, p + ".norm2"), p + ".ffn.net.0", bias: true)
        return residual + weights.linear(gelu(ff), p + ".ffn.net.3", bias: true)
    }
}

public enum NemotronTranscribeFusion {
    /// The learned normalization is applied after thresholding mean pooled 10 ms speaker activity.
    public static func speakerKernel(speakers: Int, dimensions: Int) -> MLXArray {
        let positions = MLXArray(0..<speakers).asType(.float32).reshaped(speakers, 1)
        let frequencies = exp(MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                              * Float(-log(10_000.0) / Double(dimensions)))
        let angles = positions * frequencies
        return stacked([sin(angles), cos(angles)], axis: -1).reshaped(speakers, dimensions)
    }

    public static func normalizeFeatures(_ features: MLXArray) -> MLXArray {
        let mean = features.mean(axis: 1, keepDims: true)
        let centered = features - mean
        let variance = square(centered).sum(axis: 1, keepDims: true) / Float(features.dim(1) - 1)
        return centered / (sqrt(variance) + Float(1e-5))
    }
}
