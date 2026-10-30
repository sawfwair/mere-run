import Foundation
import MLX
import MLXNN

package enum UnlimitedOCRMath {
    /// Upstream combines Python scalars in double precision before the tensor
    /// multiplication promotes the scalar to FP32. Separate tensor multiply
    /// and divide operations introduce an extra rounding boundary.
    package static func scaleRouterScores(_ scores: MLXArray, dimension: Int, factor: Float = 1) -> MLXArray {
        scores * Float(Double(factor) / sqrt(Double(dimension)))
    }

    /// PyTorch's intrinsic GELU and SiLU evaluate in FP32 before returning BF16.
    /// Evaluating the MLXNN expressions directly in BF16 rounds intermediates.
    package static func gelu(_ x: MLXArray) -> MLXArray {
        MLXNN.gelu(x.asType(.float32)).asType(x.dtype)
    }

    package static func silu(_ x: MLXArray) -> MLXArray {
        MLXNN.silu(x.asType(.float32)).asType(x.dtype)
    }

    /// PyTorch scalar multiplication keeps 1.702 in the FP32 operation dtype,
    /// then rounds the product before sigmoid and the final BF16 multiplication.
    package static func quickGelu(_ x: MLXArray) -> MLXArray {
        let scaled = (x.asType(.float32) * Float(1.702)).asType(x.dtype)
        return x * MLX.sigmoid(scaled.asType(.float32)).asType(x.dtype)
    }

    package static func rms(_ x: MLXArray, weight: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let value = x.asType(.float32)
        return (value * rsqrt(mean(square(value), axis: -1, keepDims: true) + eps)).asType(x.dtype) * weight
    }

    package static func norm(_ x: MLXArray, weight: MLXArray, bias: MLXArray, eps: Float) -> MLXArray {
        let value = x.asType(.float32)
        let centered = value - mean(value, axis: -1, keepDims: true)
        let normalized = centered * rsqrt(mean(square(centered), axis: -1, keepDims: true) + eps)
        return (normalized * weight.asType(.float32) + bias.asType(.float32)).asType(x.dtype)
    }

    /// DASH index projections use Gemma's 1 + weight convention and keep the
    /// normalization and affine multiplication in FP32 until the final cast.
    package static func routerRMS(_ x: MLXArray, weight: MLXArray) -> MLXArray {
        let value = x.asType(.float32)
        let normalized = clip(value * rsqrt(mean(square(value), axis: -1, keepDims: true) + Float(1e-6)), min: -10000, max: 10000)
        return clip(normalized * (1 + weight.asType(.float32)), min: -10000, max: 10000).asType(x.dtype)
    }

    package static func mixExperts(_ values: MLXArray, probabilities: MLXArray) -> MLXArray {
        (values.asType(.float32) * probabilities.expandedDimensions(axis: -1))
            .sum(axis: -2).asType(values.dtype)
    }

    /// Non-interleaved two-axis RoPE: 63 token pairs and one page parity pair.
    package static func rope(_ x: MLXArray, tokens: MLXArray, pages: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        // Upstream constructs its inverse-frequency buffer on CPU as reciprocal
        // powers in FP32. exp(log(base) * exponent) changes BF16 rotary rounding.
        let frequencies = MLXArray((0..<(half - 1)).map { 1 / pow(Float(10000), Float($0) / Float(half - 1)) })
        let angles = concatenated([
            tokens.asType(.float32).expandedDimensions(axis: -1) * frequencies,
            (pages % 2).asType(.float32).expandedDimensions(axis: -1)
        ], axis: -1)
        let phase = concatenated([angles, angles], axis: -1).expandedDimensions(axis: 0)
        let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
        return x * cos(phase).asType(x.dtype) + rotated * sin(phase).asType(x.dtype)
    }

    /// Exact entmax-1.5 threshold over the final axis, whose inputs are already sorted.
    package static func entmax(_ sorted: MLXArray) -> MLXArray {
        let x = (sorted - sorted[.ellipsis, 0..<1]) / 2
        let rho = MLXArray(1...x.dim(-1)).asType(.float32)
        let mu = cumsum(x, axis: -1) / rho
        let mu2 = cumsum(square(x), axis: -1) / rho
        let delta = maximum((1 - rho * (mu2 - square(mu))) / rho, 0)
        let tau = mu - sqrt(delta)
        let support = (tau .<= x).sum(axis: -1, keepDims: true).asType(.int32) - 1
        return square(maximum(x - takeAlong(tau, support, axis: -1), 0))
    }

    /// Resampling used by PyTorch interpolate for learned positional embeddings.
    /// This separable bicubic path uses align_corners=false and antialias=true.
    package static func resizeGrid(_ grid: MLXArray, height: Int, width: Int) -> MLXArray {
        func coefficients(_ input: Int, _ output: Int) -> MLXArray {
            let scale = Float(input) / Float(output)
            let filterScale = max(1, scale)
            let inverseScale = 1 / filterScale
            var rows = [Float](repeating: 0, count: output * input)
            for dst in 0..<output {
                let center = (Float(dst) + 0.5) * scale
                var total: Float = 0
                for src in 0..<input {
                    let t = abs((Float(src) - center + 0.5) * inverseScale)
                    let value: Float
                    if t < 1 { value = ((1.5 * t - 2.5) * t) * t + 1 }
                    else if t < 2 { value = (((-0.5 * t + 2.5) * t - 4) * t) + 2 }
                    else { value = 0 }
                    rows[dst * input + src] = value
                    total += value
                }
                for src in 0..<input { rows[dst * input + src] /= total }
            }
            return MLXArray(rows).reshaped(output, input)
        }
        if grid.dim(1) == height && grid.dim(2) == width { return grid }
        let vertical = coefficients(grid.dim(1), height)
        let horizontal = coefficients(grid.dim(2), width)
        // PyTorch's antialiased path rounds the horizontal FP32 pass before
        // applying the vertical pass. A three-input contraction can reorder it.
        let resizedWidth = einsum("bhwc,xw->bhxc", grid.asType(.float32), horizontal)
        return einsum("yh,bhxc->byxc", vertical, resizedWidth).asType(grid.dtype)
    }
}

/// Original checkpoint names stay intact. Boundary validation checks shapes before execution.
package struct UnlimitedOCRWeights {
    package let tensors: [String: MLXArray]

    package init(tensors: [String: MLXArray]) { self.tensors = tensors }

    package func tensor(_ name: String) -> MLXArray {
        guard let value = tensors[name] else { preconditionFailure("Validated tensor missing: \(name)") }
        return value
    }

    package func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let weight = tensor(name + ".weight").asType(x.dtype).T
        if let bias = tensors[name + ".bias"] {
            // Metal's BF16 addMM can round the product before adding bias.
            // Preserve PyTorch's FP32 affine accumulation on every device.
            return addMM(bias.asType(.float32), x.asType(.float32), weight.asType(.float32)).asType(x.dtype)
        }
        return matmul(x, weight)
    }

    package func norm(_ x: MLXArray, _ name: String, eps: Float = 1e-5) -> MLXArray {
        UnlimitedOCRMath.norm(x, weight: tensor(name + ".weight"), bias: tensor(name + ".bias"), eps: eps)
    }

    package func rms(_ x: MLXArray, _ name: String) -> MLXArray {
        UnlimitedOCRMath.rms(x, weight: tensor(name + ".weight"))
    }

    /// SAM's custom LayerNorm2d keeps its mean, variance, and affine arithmetic in
    /// the input dtype. Unlike nn.LayerNorm, it rounds each BF16 intermediate.
    package func channelNorm(_ x: MLXArray, _ name: String) -> MLXArray {
        let centered = x - mean(x, axis: -1, keepDims: true)
        let variance = mean(square(centered), axis: -1, keepDims: true)
        let stabilized = (variance.asType(.float32) + Float(1e-6)).asType(x.dtype)
        return centered / sqrt(stabilized) * tensor(name + ".weight") + tensor(name + ".bias")
    }

    package func conv(_ x: MLXArray, _ name: String, stride: Int = 1, padding: Int = 0) -> MLXArray {
        let weight = tensor(name + ".weight").transposed(0, 2, 3, 1).asType(x.dtype)
        if let bias = tensors[name + ".bias"] {
            let output = conv2d(x.asType(.float32), weight.asType(.float32), stride: .init(stride), padding: .init(padding))
            return (output + bias.asType(.float32)).asType(x.dtype)
        }
        let output = conv2d(x, weight, stride: .init(stride), padding: .init(padding))
        return output
    }
}
