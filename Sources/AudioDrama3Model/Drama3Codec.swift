#if !os(iOS)
// Native DAC graph from the pinned SGLang-Omni codec configuration.
// See THIRD_PARTY_NOTICES.md for the reference implementation attribution.
import Foundation
import MLX
import MLXFast
import MLXNN

/// Channels-last causal DAC. Weight normalization is folded once at ingestion.
/// No pickle reducer or external runtime participates in execution.
public final class Drama3Codec {
    public static let sampleRate = 44_100
    public static let samplesPerFrame = 2_048
    private let weights: [String: MLXArray]

    public init(weights source: [String: MLXArray]) throws {
        var result = source
        for (name, v) in source {
            let suffix: String
            let gainName: String
            if name.hasSuffix(".parametrizations.weight.original1") {
                suffix = ".parametrizations.weight.original1"
                gainName = String(name.dropLast(suffix.count)) + ".parametrizations.weight.original0"
            } else if name.hasSuffix(".weight_v") {
                suffix = ".weight_v"
                gainName = String(name.dropLast(suffix.count)) + ".weight_g"
            } else { continue }
            guard let gain = source[gainName], v.ndim == 3 else {
                throw Drama3Error.invalid("invalid codec weight normalization: \(name)")
            }
            let norm = MLX.sqrt(MLX.sum(v * v, axes: [1, 2], keepDims: true))
            let folded = v * (gain / norm)
            MLX.eval(folded)
            result[String(name.dropLast(suffix.count)) + ".weight"] = folded
            result.removeValue(forKey: name)
            result.removeValue(forKey: gainName)
        }
        weights = result
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let value = weights[name] else { throw Drama3Error.invalid("missing codec tensor \(name)") }
        return value
    }

    private func linear(_ x: MLXArray, _ path: String) throws -> MLXArray {
        let y = MLX.matmul(x, try weight(path + ".weight").transposed())
        return y + (weights[path + ".bias"] ?? MLXArray(Float(0)))
    }

    package func conv(_ x: MLXArray, _ path: String, stride: Int = 1, dilation: Int = 1,
                      transpose: Bool = false, groups: Int = 1) throws -> MLXArray {
        let w = try weight(path + ".weight")
        guard w.ndim == 3 else { throw Drama3Error.invalid("codec convolution rank: \(path)") }
        let kernel = w.dim(2)
        let y: MLXArray
        if transpose {
            // Torch transpose kernels [in, out, k] -> MLX [out, k, in].
            let full = MLX.convTransposed1d(x, w.transposed(1, 2, 0), stride: stride)
            y = full[0..., 0..<(full.dim(1) - (kernel - stride)), 0...]
        } else {
            let left = (kernel - 1) * dilation + 1 - stride
            let extra = (stride - x.dim(1) % stride) % stride
            let padded = MLX.padded(x, widths: [.init(0), .init((left, extra)), .init(0)])
            y = MLX.conv1d(padded, w.transposed(0, 2, 1), stride: stride, dilation: dilation, groups: groups)
        }
        return y + (weights[path + ".bias"] ?? MLXArray(Float(0)))
    }

    package func snake(_ x: MLXArray, _ path: String) throws -> MLXArray {
        let alpha = try weight(path + ".alpha").reshaped(1, 1, x.dim(-1))
        let wave = MLX.sin(alpha * x)
        return x + wave * wave / (alpha + MLXArray(Float(1e-9)))
    }

    private func residual(_ x: MLXArray, _ path: String, dilation: Int) throws -> MLXArray {
        let first = try conv(snake(x, path + ".block.0"), path + ".block.1.conv", dilation: dilation)
        return x + (try conv(snake(first, path + ".block.2"), path + ".block.3.conv"))
    }

    private func convNext(_ x: MLXArray, _ path: String) throws -> MLXArray {
        let depth = try conv(x, path + ".dwconv.conv", groups: x.dim(-1))
        let normed = MLXFast.layerNorm(depth, weight: try weight(path + ".norm.weight"),
                                       bias: try weight(path + ".norm.bias"), eps: 1e-6)
        let output = try linear(MLXNN.gelu(linear(normed, path + ".pwconv1")), path + ".pwconv2")
        return x + output * (try weight(path + ".gamma"))
    }

    private func norm(_ x: MLXArray, _ path: String) throws -> MLXArray {
        MLXFast.rmsNorm(x, weight: try weight(path + ".weight"), eps: 1e-5)
    }

    private func transformer(_ x: MLXArray, _ path: String, layers: Int, window: Int?) throws -> MLXArray {
        var h = x
        let count = x.dim(1), dim = x.dim(2), heads = dim / 64
        let rope = RoPE(dimensions: 64, traditional: true, base: 10_000)
        let positions = MLXArray(0..<count)
        let delta = positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0)
        let allowed = MLX.logicalAnd(delta .>= 0, delta .< (window ?? count))
        let mask = MLX.where(allowed, MLXArray(Float(0)), MLXArray(-Float.infinity))
        for i in 0..<layers {
            let p = path + ".layers.\(i)"
            let projected = try linear(norm(h, p + ".attention_norm"), p + ".attention.wqkv")
            let q = rope(projected[.ellipsis, 0..<dim].reshaped(1, count, heads, 64).transposed(0, 2, 1, 3))
            let k = rope(projected[.ellipsis, dim..<(2 * dim)].reshaped(1, count, heads, 64).transposed(0, 2, 1, 3))
            let v = projected[.ellipsis, (2 * dim)...].reshaped(1, count, heads, 64).transposed(0, 2, 1, 3)
            let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.125,
                mask: .array(mask)).transposed(0, 2, 1, 3).reshaped(1, count, dim)
            h = h + (try linear(attended, p + ".attention.wo")) * (try weight(p + ".attention_layer_scale.gamma"))
            let n = try norm(h, p + ".ffn_norm")
            let mlp = MLXNN.silu(try linear(n, p + ".feed_forward.w1")) * (try linear(n, p + ".feed_forward.w3"))
            h = h + (try linear(mlp, p + ".feed_forward.w2")) * (try weight(p + ".ffn_layer_scale.gamma"))
            MLX.eval(h)
        }
        return try norm(h, path + ".norm")
    }

    private func quantizerPath(_ index: Int) -> String {
        index == 0 ? "quantizer.semantic_quantizer.quantizers.0" : "quantizer.quantizer.quantizers.\(index - 1)"
    }

    public func encode(_ samples: [Float]) throws -> [[Int32]] {
        guard !samples.isEmpty else { throw Drama3Error.invalid("reference audio is empty") }
        // Released DAC.encode pads to the complete acoustic frame before the
        // encoder and its two RVQ downsamplers, including the final partial frame.
        let padded = samples + Array(repeating: 0,
            count: (Self.samplesPerFrame - samples.count % Self.samplesPerFrame) % Self.samplesPerFrame)
        var h = try conv(MLXArray(padded).reshaped(1, padded.count, 1), "encoder.block.0.conv")
        for (i, stride) in [2, 4, 8, 8].enumerated() {
            let p = "encoder.block.\(i + 1).block"
            for (j, d) in [1, 3, 9].enumerated() { h = try residual(h, p + ".\(j)", dilation: d) }
            h = try conv(snake(h, p + ".3"), p + ".4.conv", stride: stride)
            if i == 3 { h = try transformer(h, p + ".5", layers: 4, window: 512) }
            MLX.eval(h)
        }
        h = try conv(snake(h, "encoder.block.5"), "encoder.block.6.conv")
        for i in 0..<2 {
            let p = "quantizer.downsample.\(i)"
            h = try convNext(conv(h, p + ".0.conv", stride: 2), p + ".1")
        }
        h = try transformer(h, "quantizer.pre_module", layers: 8, window: 128)
        var codes: [[Int32]] = []
        for i in 0..<10 {
            let p = quantizerPath(i)
            let projected = try conv(h, p + ".in_proj")
            let book = try weight(p + ".codebook.weight")
            let normalized = projected / MLX.maximum(MLX.sqrt(MLX.sum(projected * projected, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
            let normalizedBook = book / MLX.maximum(MLX.sqrt(MLX.sum(book * book, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
            let distances = MLX.sum(normalized * normalized, axis: -1, keepDims: true)
                - 2 * MLX.matmul(normalized, normalizedBook.transposed())
                + MLX.sum(normalizedBook * normalizedBook, axis: -1).expandedDimensions(axis: 0)
            let ids = MLX.argMin(distances, axis: -1).asType(.int32)
            codes.append(ids.reshaped(-1).asArray(Int32.self))
            h = h - (try conv(book[ids], p + ".out_proj"))
            MLX.eval(h)
        }
        return codes
    }

    public func decode(_ codes: [[Int32]]) throws -> MLXArray {
        guard codes.count == 10, let count = codes.first?.count, count > 0,
              codes.allSatisfy({ $0.count == count && $0.allSatisfy({ $0 >= 0 && $0 < 4096 }) }) else {
            throw Drama3Error.invalid("codec needs ten equally sized nonempty codebook rows")
        }
        var latents: [MLXArray] = []
        for i in 0..<10 {
            let p = quantizerPath(i)
            let book = try weight(p + ".codebook.weight")
            let ids = MLX.minimum(MLXArray(codes[i]), MLXArray(Int32(book.dim(0) - 1)))
            latents.append(try conv(book[ids].expandedDimensions(axis: 0), p + ".out_proj"))
        }
        var h = try transformer(MLX.sum(MLX.stacked(latents), axis: 0), "quantizer.post_module", layers: 8, window: 128)
        for i in 0..<2 {
            let p = "quantizer.upsample.\(i)"
            h = try convNext(conv(h, p + ".0.conv", stride: 2, transpose: true), p + ".1")
        }
        h = try conv(h, "decoder.model.0.conv")
        for (i, stride) in [8, 8, 4, 2].enumerated() {
            let p = "decoder.model.\(i + 1).block"
            h = try conv(snake(h, p + ".0"), p + ".1.conv", stride: stride, transpose: true)
            for (j, d) in [1, 3, 9].enumerated() { h = try residual(h, p + ".\(j + 2)", dilation: d) }
            MLX.eval(h)
        }
        return MLX.tanh(try conv(snake(h, "decoder.model.5"), "decoder.model.6.conv")).reshaped(-1)
    }
}

#endif
