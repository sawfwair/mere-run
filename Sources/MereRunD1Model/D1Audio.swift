#if !os(iOS)
// Native Swift/MLX reimplementation of the pinned LiquidAI D1 references.
// Modified for mere.run; see THIRD_PARTY_NOTICES.md and licenses/d1-LFM-OPEN-LICENSE.txt.
import Foundation
import MLX
import MLXNN

/// FastConformer arithmetic over the checkpoint's named tensors. Inference is single-clip FP32.
public struct D1Audio {
    let config: D1AudioConfiguration
    let weights: [String: MLXArray]
    public init(config: D1AudioConfiguration, outputWidth: Int, weights: [String: MLXArray]) throws {
        try config.validate()
        self.config = config; self.weights = weights
        var expected: [String: [Int]] = [:]
        func linear(_ name: String, _ input: Int, _ output: Int, bias: Bool = true) {
            expected[name + ".weight"] = [output, input]
            if bias { expected[name + ".bias"] = [output] }
        }
        func norm(_ name: String, _ width: Int) { expected[name + ".weight"] = [width]; expected[name + ".bias"] = [width] }
        let channels = config.subsampling_conv_channels, d = config.d_model
        for (index, input, kernel) in [(0, 1, 3), (2, 1, 3), (3, channels, 1), (5, 1, 3), (6, channels, 1)] {
            let name = "encoder.pre_encode.conv.\(index)"
            expected[name + ".weight"] = [channels, input, kernel, kernel]; expected[name + ".bias"] = [channels]
        }
        linear("encoder.pre_encode.out", channels * ((config.feat_in + 7) / 8), d)
        for index in 0..<config.n_layers {
            let name = "encoder.layers.\(index)"
            for suffix in ["norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"] { norm(name + "." + suffix, d) }
            for suffix in ["feed_forward1", "feed_forward2"] {
                linear(name + "." + suffix + ".linear1", d, d * config.ff_expansion_factor)
                linear(name + "." + suffix + ".linear2", d * config.ff_expansion_factor, d)
            }
            for suffix in ["linear_q", "linear_k", "linear_v", "linear_out", "linear_pos"] {
                linear(name + ".self_attn." + suffix, d, d, bias: suffix != "linear_pos")
            }
            expected[name + ".self_attn.pos_bias_u"] = [config.n_heads, d / config.n_heads]
            expected[name + ".self_attn.pos_bias_v"] = [config.n_heads, d / config.n_heads]
            for (suffix, input, output, kernel) in [("pointwise_conv1", d, 2 * d, 1), ("depthwise_conv", 1, d, config.conv_kernel_size), ("pointwise_conv2", d, d, 1)] {
                expected[name + ".conv." + suffix + ".weight"] = [output, input, kernel]
                expected[name + ".conv." + suffix + ".bias"] = [output]
            }
            norm(name + ".conv.batch_norm", d)
            expected[name + ".conv.batch_norm.running_mean"] = [d]; expected[name + ".conv.batch_norm.running_var"] = [d]
        }
        norm("adapter.norm", d); linear("adapter.linear_1", d, outputWidth); linear("adapter.linear_2", outputWidth, outputWidth)
        norm("residual.ln", outputWidth); linear("residual.down", outputWidth, config.residual_width); linear("residual.up", config.residual_width, outputWidth)
        for (name, shape) in expected {
            guard let array = weights["audio." + name], array.shape == shape, array.dtype == .float32 else {
                throw D1Error.invalid("Missing or malformed D1 FP32 audio weight: \(name)")
            }
        }
        for name in weights.keys where name.hasPrefix("audio.") {
            let local = String(name.dropFirst(6))
            guard expected[local] != nil || local.hasSuffix(".batch_norm.num_batches_tracked") else {
                throw D1Error.invalid("Unexpected D1 audio weight: \(name)")
            }
        }
    }
    func parameter(_ key: String) throws -> MLXArray {
        guard let value = weights["audio." + key] else { throw D1Error.invalid("Missing D1 audio weight: \(key)") }
        return value
    }
    func linear(_ x: MLXArray, _ key: String, bias: Bool = true) throws -> MLXArray {
        let weight = try parameter(key + ".weight")
        guard weight.ndim == 2, weight.dim(1) == x.dim(-1) else { throw D1Error.invalid("Invalid D1 audio linear weight: \(key)") }
        let y = matmul(x, weight.T)
        return try bias ? y + parameter(key + ".bias") : y
    }
    func norm(_ x: MLXArray, _ key: String) throws -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = ((x - mean) * (x - mean)).mean(axis: -1, keepDims: true)
        return try (x - mean) * rsqrt(variance + 1e-5) * parameter(key + ".weight") + parameter(key + ".bias")
    }
    func conv1(_ x: MLXArray, _ key: String, groups: Int = 1, padding: Int = 0) throws -> MLXArray {
        let w = try parameter(key + ".weight")
        return try MLX.conv1d(x, w.transposed(0, 2, 1), padding: padding, groups: groups) + parameter(key + ".bias")
    }
    func feedForward(_ x: MLXArray, _ key: String) throws -> MLXArray {
        try linear(silu(linear(x, key + ".linear1")), key + ".linear2")
    }
    public func callAsFunction(samples: [Float]) throws -> MLXArray {
        var x = try Self.mel(samples: samples).transposed(0, 2, 1).expandedDimensions(axis: -1)
        // Mel's extra STFT frame is zero, masked before each subsampling operation.
        var validLength = max(8_000, min(480_000, samples.count)) / 160
        for index in 0..<8 {
            let valid = MLXArray((0..<x.dim(1)).map { Float($0 < validLength ? 1 : 0) }).reshaped(1, x.dim(1), 1, 1)
            x = x * valid
            if [1, 4, 7].contains(index) { x = relu(x); continue }
            let key = "encoder.pre_encode.conv.\(index)"
            let w = try parameter(key + ".weight")
            let depthwise = index == 2 || index == 5
            let stride = index == 0 || depthwise ? 2 : 1
            x = try MLX.conv2d(x, w.transposed(0, 2, 3, 1), stride: IntOrPair(stride),
                padding: IntOrPair(stride == 2 ? 1 : 0), groups: depthwise ? config.subsampling_conv_channels : 1)
                + parameter(key + ".bias")
            if stride == 2 { validLength = (validLength - 1) / 2 + 1 }
        }
        x = x[0..., 0..<validLength, 0..., 0...].transposed(0, 1, 3, 2).reshaped(1, validLength, -1)
        x = try linear(x, "encoder.pre_encode.out")
        let t = x.dim(1), d = config.d_model, heads = config.n_heads, headDim = d / heads
        let posValues = (0..<(2 * t - 1)).flatMap { index in
            (0..<d).map { channel -> Float in
                let angle = Float(t - 1 - index) * exp(Float(channel / 2 * 2) * -log(10_000.0) / Float(d))
                return channel % 2 == 0 ? sin(angle) : cos(angle)
            }
        }
        let position = MLXArray(posValues).reshaped(1, 2 * t - 1, d)
        for index in 0..<config.n_layers {
            try Task.checkCancellation()
            let key = "encoder.layers.\(index)"
            x = try x + feedForward(norm(x, key + ".norm_feed_forward1"), key + ".feed_forward1") * 0.5
            let input = try norm(x, key + ".norm_self_att"), attention = key + ".self_attn"
            let q = try linear(input, attention + ".linear_q").reshaped(1, t, heads, headDim)
            let k = try linear(input, attention + ".linear_k").reshaped(1, t, heads, headDim).transposed(0, 2, 1, 3)
            let v = try linear(input, attention + ".linear_v").reshaped(1, t, heads, headDim).transposed(0, 2, 1, 3)
            let p = try linear(position, attention + ".linear_pos", bias: false)
                .reshaped(1, 2 * t - 1, heads, headDim).transposed(0, 2, 1, 3)
            let ac = try matmul((q + parameter(attention + ".pos_bias_u")).transposed(0, 2, 1, 3), k.transposed(0, 1, 3, 2))
            let bd = try matmul((q + parameter(attention + ".pos_bias_v")).transposed(0, 2, 1, 3), p.transposed(0, 1, 3, 2))
            let shifted = padded(bd, widths: [[0, 0], [0, 0], [0, 0], [1, 0]])
                .reshaped(1, heads, 2 * t, t)[0..., 0..., 1..., 0...].reshaped(1, heads, t, 2 * t - 1)
            let scores = (ac + shifted[0..., 0..., 0..., 0..<t]) / sqrt(Float(headDim))
            let result = matmul(softmax(scores, axis: -1), v).transposed(0, 2, 1, 3).reshaped(1, t, d)
            x = try x + linear(result, attention + ".linear_out")
            var conv = try conv1(norm(x, key + ".norm_conv"), key + ".conv.pointwise_conv1")
            conv = conv[.ellipsis, 0..<d] * sigmoid(conv[.ellipsis, d...])
            conv = try conv1(conv, key + ".conv.depthwise_conv", groups: d, padding: (config.conv_kernel_size - 1) / 2)
            let bn = key + ".conv.batch_norm"
            conv = try (conv - parameter(bn + ".running_mean")) * rsqrt(parameter(bn + ".running_var") + 1e-5)
                * parameter(bn + ".weight") + parameter(bn + ".bias")
            x = try x + conv1(silu(conv), key + ".conv.pointwise_conv2")
            x = try x + feedForward(norm(x, key + ".norm_feed_forward2"), key + ".feed_forward2") * 0.5
            x = try norm(x, key + ".norm_out")
        }
        x = try linear(gelu(linear(norm(x, "adapter.norm"), "adapter.linear_1")), "adapter.linear_2")
        return try x + linear(gelu(linear(norm(x, "residual.ln"), "residual.down")), "residual.up")
    }

    public static func mel(samples: [Float]) throws -> MLXArray {
        guard !samples.isEmpty, samples.allSatisfy(\.isFinite) else { throw D1Error.invalid("D1 audio requires finite mono samples.") }
        var samples = Array(samples.prefix(480_000))
        if samples.count < 8_000 { samples += Array(repeating: 0, count: 8_000 - samples.count) }
        let emphasized = samples.enumerated().map { index, value in index == 0 ? value : value - 0.97 * samples[index - 1] }
        let frames = samples.count / 160
        let windows = (0...frames).flatMap { frame in (0..<512).map { offset -> Float in
            let position = frame * 160 + offset - 256
            guard (0..<samples.count).contains(position), (56..<456).contains(offset) else { return 0 }
            return emphasized[position] * (0.5 - 0.5 * cos(2 * Float.pi * Float(offset - 56) / 399))
        } }
        let fft = MLX.rfft(MLXArray(windows).reshaped(frames + 1, 512), axis: -1)
        let power = abs(fft) * abs(fft)
        let logMel = log(matmul(power, MLXArray(slaney()).reshaped(128, 257).T) + Float(pow(2.0, -24)))
        let valid = logMel[0..<frames, 0...]
        let mean = valid.mean(axis: 0, keepDims: true)
        let std = sqrt(((valid - mean) * (valid - mean)).sum(axis: 0, keepDims: true) / Float(frames - 1))
        let normalized = (valid - mean) / (std + 1e-5)
        return concatenated([normalized, MLXArray.zeros([1, 128])], axis: 0).T.expandedDimensions(axis: 0)
    }
    static func slaney() -> [Float] {
        let step = log(6.4) / 27, minMel = 15.0
        let maxMel = minMel + log(8.0) / step
        let frequencies = (0..<130).map { index -> Double in
            let mel = Double(index) * maxMel / 129
            return mel >= minMel ? 1_000 * exp(step * (mel - minMel)) : mel * 200 / 3
        }
        return (0..<128).flatMap { band in (0..<257).map { bin -> Float in
            let hz = Double(bin) * 16_000 / 512
            let rising = (hz - frequencies[band]) / (frequencies[band + 1] - frequencies[band])
            let falling = (frequencies[band + 2] - hz) / (frequencies[band + 2] - frequencies[band + 1])
            // librosa rounds the triangle to FP32 before applying the Slaney normalization.
            return Float(Double(Float(max(0, min(rising, falling)))) * 2 / (frequencies[band + 2] - frequencies[band]))
        } }
    }
}
#endif
