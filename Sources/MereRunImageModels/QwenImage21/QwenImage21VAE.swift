// Native image specialization of the Apache-2.0 Diffusers Qwen Image 2.1 VAE.
// Temporal convolutions are checkpoint parameters but are inactive for the single first frame.
import Foundation
import MLX
import MLXFast
import MLXNN

public final class QwenImage21VAE {
    public let config: QwenImage21VAEConfig
    private let weights: QwenImage21Weights

    public init(config: QwenImage21VAEConfig, arrays: [String: MLXArray]) throws {
        guard config.isResidual, config.patchSize == nil, config.attnScales.isEmpty,
              config.dimMult.count == 5, config.temporalDownsample.count == 4,
              config.scaleFactorSpatial == 16, config.inChannels == 4, config.outChannels == 4,
              config.baseDim > 0, config.decoderBaseDim > 0, config.zDim > 0,
              config.numResBlocks > 0, config.dimMult.allSatisfy({ $0 > 0 }),
              config.latentsMean.count == config.zDim, config.latentsStd.count == config.zDim,
              config.latentsMean.allSatisfy(\.isFinite), config.latentsStd.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw QwenImage21Error.invalidConfiguration("Unsupported RGBA VAE configuration.")
        }
        self.config = config
        weights = try QwenImage21Weights(arrays, shapes: Self.weightShapes(config))
    }

    public static func weightShapes(_ c: QwenImage21VAEConfig) -> [String: [Int]] {
        var shapes: [String: [Int]] = [:]
        func conv(_ name: String, _ input: Int, _ output: Int, _ kernel: Int = 3) {
            shapes[name + ".weight"] = [output, input, kernel, kernel]
            shapes[name + ".bias"] = [output]
        }
        func norm(_ name: String, _ dim: Int, image: Bool = false) {
            shapes[name + ".gamma"] = image ? [dim, 1, 1] : [dim, 1, 1, 1]
        }
        func residual(_ name: String, _ input: Int, _ output: Int) {
            norm(name + ".norm1", input); norm(name + ".norm2", output)
            conv(name + ".conv1", input, output); conv(name + ".conv2", output, output)
            if input != output { conv(name + ".conv_shortcut", input, output, 1) }
        }
        func middle(_ name: String, _ dim: Int) {
            residual(name + ".resnets.0", dim, dim); residual(name + ".resnets.1", dim, dim)
            norm(name + ".attentions.0.norm", dim, image: true)
            conv(name + ".attentions.0.to_qkv", dim, 3 * dim, 1)
            conv(name + ".attentions.0.proj", dim, dim, 1)
        }
        conv("quant_conv", 2 * c.zDim, 2 * c.zDim, 1)
        conv("post_quant_conv", c.zDim, c.zDim, 1)
        for decoder in [false, true] {
            let name = decoder ? "decoder" : "encoder"
            let base = decoder ? c.decoderBaseDim : c.baseDim
            let multipliers = decoder ? [c.dimMult.last!] + c.dimMult.reversed() : [1] + c.dimMult
            let dims = multipliers.map { base * $0 }
            conv(name + ".conv_in", decoder ? c.zDim : c.inChannels, dims[0])
            middle(name + ".mid_block", decoder ? dims[0] : dims.last!)
            norm(name + ".norm_out", dims.last!)
            conv(name + ".conv_out", dims.last!, decoder ? c.outChannels : 2 * c.zDim)
            for index in 0..<c.dimMult.count {
                let block = name + (decoder ? ".up_blocks." : ".down_blocks.") + String(index)
                let input = dims[index], output = dims[index + 1]
                for layer in 0..<(c.numResBlocks + (decoder ? 1 : 0)) {
                    residual(block + ".resnets.\(layer)", layer == 0 ? input : output, output)
                }
                if index < c.dimMult.count - 1 {
                    let sampler = block + (decoder ? ".upsampler" : ".downsampler")
                    conv(sampler + ".resample.1", output, output)
                    let temporal = decoder ? Array(c.temporalDownsample.reversed())[index] : c.temporalDownsample[index]
                    if temporal { conv(sampler + ".time_conv", output, decoder ? 2 * output : output, 1) }
                }
            }
        }
        return shapes
    }

    /// Input and output use NHWC. Returns the normalized posterior mean, never a sampled posterior.
    public func encode(_ rgba: MLXArray) throws -> MLXArray {
        guard rgba.ndim == 4, rgba.dim(3) == 4, rgba.dim(1).isMultiple(of: 16), rgba.dim(2).isMultiple(of: 16) else {
            throw QwenImage21Error.invalidLayout("VAE input must be NHWC RGBA with dimensions divisible by 16.")
        }
        var x = conv(rgba, "encoder.conv_in")
        for index in 0..<config.dimMult.count {
            try Task.checkCancellation()
            let name = "encoder.down_blocks.\(index)"
            let original = x
            for layer in 0..<config.numResBlocks { x = residual(x, name + ".resnets.\(layer)") }
            let spatial = index < config.dimMult.count - 1 ? 2 : 1
            let temporal = index < config.temporalDownsample.count && config.temporalDownsample[index] ? 2 : 1
            if spatial == 2 {
                let padded = padded(x, widths: [.init(0), .init((0, 1)), .init((0, 1)), .init(0)])
                x = conv(padded, name + ".downsampler.resample.1", stride: 2, padding: 0)
            }
            x = x + Self.averageShortcut(original, outputChannels: x.dim(3), temporal: temporal, spatial: spatial)
            eval(x)
        }
        x = middle(x, "encoder.mid_block")
        x = conv(silu(norm(x, "encoder.norm_out")), "encoder.conv_out")
        let posterior = conv(x, "quant_conv", padding: 0)
        let latent = posterior[0..., 0..., 0..., 0..<config.zDim]
        return (latent - MLXArray(config.latentsMean).asType(latent.dtype)) / MLXArray(config.latentsStd).asType(latent.dtype)
    }

    /// Decodes normalized latents to straight RGBA in [0, 1].
    public func decode(_ latents: MLXArray) throws -> MLXArray {
        guard latents.ndim == 4, latents.dim(3) == config.zDim else {
            throw QwenImage21Error.invalidLayout("VAE latent channel count does not match.")
        }
        var x = latents * MLXArray(config.latentsStd).asType(latents.dtype) + MLXArray(config.latentsMean).asType(latents.dtype)
        x = conv(x, "post_quant_conv", padding: 0)
        x = middle(conv(x, "decoder.conv_in"), "decoder.mid_block")
        for index in 0..<config.dimMult.count {
            try Task.checkCancellation()
            let name = "decoder.up_blocks.\(index)"
            let original = x
            for layer in 0...config.numResBlocks { x = residual(x, name + ".resnets.\(layer)") }
            if index < config.dimMult.count - 1 {
                x = repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
                x = conv(x, name + ".upsampler.resample.1")
                let temporal = Array(config.temporalDownsample.reversed())[index] ? 2 : 1
                x = x + Self.duplicateShortcut(original, outputChannels: x.dim(3), temporal: temporal)
            }
            eval(x)
        }
        return (clip(conv(silu(norm(x, "decoder.norm_out")), "decoder.conv_out"), min: -1, max: 1) + 1) / 2
    }

    /// Mirrors Wan's channel/time/space fold, including the zero temporal frame before the first image.
    public static func averageShortcut(_ x: MLXArray, outputChannels: Int, temporal: Int, spatial: Int) -> MLXArray {
        let batch = x.dim(0), height = x.dim(1) / spatial, width = x.dim(2) / spatial, channels = x.dim(3)
        var value = x.transposed(0, 3, 1, 2).reshaped(batch, channels, 1, height, spatial, width, spatial)
        if temporal == 2 {
            value = concatenated([zeros(value.shape, dtype: value.dtype), value], axis: 2)
        }
        value = value.transposed(0, 1, 2, 4, 6, 3, 5)
        value = value.reshaped(batch, outputChannels, channels * temporal * spatial * spatial / outputChannels, height, width)
        return mean(value, axis: 2).transposed(0, 2, 3, 1)
    }

    public static func duplicateShortcut(_ x: MLXArray, outputChannels: Int, temporal: Int) -> MLXArray {
        let batch = x.dim(0), height = x.dim(1), width = x.dim(2)
        let repeats = outputChannels * temporal * 4 / x.dim(3)
        let value = repeated(x.transposed(0, 3, 1, 2), count: repeats, axis: 1)
            .reshaped(batch, outputChannels, temporal, 2, 2, height, width)
        // First-frame decoding discards the leading temporal copies.
        return value[0..., 0..., temporal - 1, 0..., 0..., 0..., 0...]
            .transposed(0, 4, 2, 5, 3, 1).reshaped(batch, height * 2, width * 2, outputChannels)
    }

    private func conv(_ x: MLXArray, _ name: String, stride: Int = 1, padding: Int = 1) -> MLXArray {
        conv2d(x, weights[name + ".weight"].transposed(0, 2, 3, 1), stride: .init(stride), padding: .init(padding))
            + weights[name + ".bias"]
    }

    private func norm(_ x: MLXArray, _ name: String) -> MLXArray {
        let value = x.asType(.float32)
        let denominator = maximum(sqrt(sum(value * value, axis: -1, keepDims: true)), 1e-12)
        return (value / denominator).asType(x.dtype) * sqrt(Float(x.dim(3))) * weights[name + ".gamma"].reshaped(-1)
    }

    private func residual(_ x: MLXArray, _ name: String) -> MLXArray {
        let shortcut = weights.arrays[name + ".conv_shortcut.weight"] == nil ? x : conv(x, name + ".conv_shortcut", padding: 0)
        let y = conv(silu(norm(x, name + ".norm1")), name + ".conv1")
        return shortcut + conv(silu(norm(y, name + ".norm2")), name + ".conv2")
    }

    private func middle(_ x: MLXArray, _ name: String) -> MLXArray {
        let hidden = residual(x, name + ".resnets.0")
        let prefix = name + ".attentions.0"
        let qkv = conv(norm(hidden, prefix + ".norm"), prefix + ".to_qkv", padding: 0)
        let parts = split(qkv.reshaped(x.dim(0), 1, x.dim(1) * x.dim(2), 3 * x.dim(3)), parts: 3, axis: -1)
        let attention = MLXFast.scaledDotProductAttention(queries: parts[0], keys: parts[1], values: parts[2],
                                                         scale: 1 / sqrt(Float(x.dim(3))), mask: .none)
        let output = hidden + conv(attention.reshaped(x.shape), prefix + ".proj", padding: 0)
        return residual(output, name + ".resnets.1")
    }
}
