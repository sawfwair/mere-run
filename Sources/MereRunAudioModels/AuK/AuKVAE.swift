import Foundation
import MLX

/// AuK's inference VAE: deterministic posterior mean and causal BigVGAN decode.
public struct AuKVAE {
    let weights: AuKTensorStore
    public init(weights: AuKTensorStore) { self.weights = weights }

    public func encode(_ waveform: MLXArray) throws -> MLXArray {
        func leaky(_ x: MLXArray, _ slope: Float) -> MLXArray { maximum(x, x * slope) }
        var x = try leaky(weights.conv(waveform, "audio_encoder.pre"), 0.2)
        for (stage, rate) in [2, 2, 2, 3, 4, 5].enumerated() {
            let key = "audio_encoder.stages.\(stage)"
            x = try weights.conv(x, key + ".down", stride: rate, padding: (2 * rate - 1) / 2)
            for layer in 0..<6 {
                let prefix = key + ".stack.layers.\(layer)"
                var h = try weights.conv(leaky(x, 0.01), prefix + ".0", dilation: 1 << layer)
                h = try weights.conv(leaky(h, 0.01), prefix + ".1")
                x = x + h
            }
            x = leaky(x, 0.2)
        }
        let stats = try weights.conv(x, "audio_encoder.post")
        return try (stats[.ellipsis, ..<64] - weights.tensor("global_mean")) / sqrt(weights.tensor("global_log_std"))
    }

    public func decode(_ latent: MLXArray) throws -> MLXArray {
        var x = try latent * sqrt(weights.tensor("global_log_std")) + weights.tensor("global_mean")
        // The first decoder convolution is centered, unlike all residual convolutions.
        x = try weights.conv(x, "decoder.conv_pre")
        for (stage, rate) in [5, 4, 3, 2, 2, 2].enumerated() {
            let key = "decoder.ups.\(stage)"
            var up = MLX.convTransposed1d(x, try weights.tensor(key + ".weight"), stride: rate)
            up = up + (try weights.tensor(key + ".bias"))
            x = up[0..., ..<(up.dim(1) - rate)]
            var results = [MLXArray]()
            for kernel in 0..<3 {
                let prefix = "decoder.resblocks.\(stage * 3 + kernel)"
                var h = x
                for (index, dilation) in [1, 3, 5].enumerated() {
                    var y = try activation(h, prefix + ".activations.\(2 * index)")
                    y = try weights.conv(y, prefix + ".convs1.\(index)", dilation: dilation, causal: true)
                    y = try activation(y, prefix + ".activations.\(2 * index + 1)")
                    h = h + (try weights.conv(y, prefix + ".convs2.\(index)", causal: true))
                }
                results.append(h)
            }
            x = (results[0] + results[1] + results[2]) / 3
            eval(x)
        }
        x = try weights.conv(activation(x, "decoder.activation_post"), "decoder.conv_post", causal: true)
        return clip(x, min: -1, max: 1)
    }

    func activation(_ x: MLXArray, _ key: String) throws -> MLXArray {
        let channels = x.dim(-1)
        let filter = broadcast(AuKVAE.sincFilter.reshaped(1, 12, 1), to: [channels, 12, 1])
        // Noncausal upsampler: five edge samples on each side, then crop 15 samples.
        let left = repeated(x[0..., 0..<1], count: 5, axis: 1)
        let right = repeated(x[0..., (x.dim(1) - 1)...], count: 5, axis: 1)
        let paddedInput = concatenated([left, x, right], axis: 1)
        var y = 2 * MLX.convTransposed1d(paddedInput, filter, stride: 2, groups: channels)
        y = y[0..., 15..<(y.dim(1) - 15)]
        let alpha = exp(try weights.tensor(key + ".act.alpha"))
        let beta = exp(try weights.tensor(key + ".act.beta"))
        let wave = sin(y * alpha)
        y = y + wave * wave / (beta + 1e-9)
        // Causal downsampler uses replicated history, not zero padding.
        y = concatenated([repeated(y[0..., 0..<1], count: 11, axis: 1), y], axis: 1)
        return MLX.conv1d(y, filter, stride: 2, groups: channels)
    }

    public static var sincFilter: MLXArray {
        func bessel(_ x: Double) -> Double {
            var sum = 1.0, term = 1.0
            for k in 1..<40 { term *= x * x / (4 * Double(k * k)); sum += term }
            return sum
        }
        let attenuation = 2.285 * 5 * Double.pi * 1.2 + 7.95
        let beta = 0.1102 * (attenuation - 8.7)
        let values = (0..<12).map { index -> Double in
            let r = (Double(index) - 5.5) / 5.5
            let window = bessel(beta * sqrt(max(0, 1 - r * r))) / bessel(beta)
            let angle = Double.pi * 0.5 * (Double(index) - 5.5)
            return 0.5 * window * sin(angle) / angle
        }
        let total = values.reduce(0, +)
        return MLXArray(values.map { Float($0 / total) })
    }
}
