import MLX
import MLXFast
import MLXNN

public struct MERT2Encoder {
    let config: MERT2Configuration
    var weights: SheetSage2Weights

    public init(configuration: MERT2Configuration, arrays: [String: MLXArray]) throws {
        config = configuration
        weights = SheetSage2Weights(values: arrays)
        var consumed = Set<String>()
        try weights.require("feature_extractor.mel_mean", [config.melBins], consumed: &consumed)
        try weights.require("feature_extractor.mel_std", [config.melBins], consumed: &consumed)
        try weights.require("feature_extractor.mel_scale.fb", [1025, config.melBins], consumed: &consumed)
        try weights.require("feature_extractor.spectrogram.window", [2048], consumed: &consumed)
        for index in config.channels.indices {
            let prefix = "subsampling_module.\(index)"
            let input = index == 0 ? config.melBins : config.channels[index - 1]
            let width = config.channels[index]
            if index > 0 || input != width {
                try weights.requireNorm(prefix + ".resampling_layer.0", size: input, consumed: &consumed)
                try weights.require(prefix + ".resampling_layer.2.weight", [width, input, 2], consumed: &consumed)
                try weights.require(prefix + ".resampling_layer.2.bias", [width], consumed: &consumed)
            }
            for layer in 0..<config.depths[index] {
                let name = prefix + ".convnext_layers.\(layer)"
                try weights.require(name + ".depthwise_block.1.weight", [width, 1, 7], consumed: &consumed)
                try weights.require(name + ".depthwise_block.1.bias", [width], consumed: &consumed)
                try weights.requireNorm(name + ".pointwise_block.0", size: width, consumed: &consumed)
                try weights.requireLinear(name + ".pointwise_block.1", input: width, output: width * 4, consumed: &consumed)
                try weights.require(name + ".pointwise_block.3.weight", [1, 1, width * 4], consumed: &consumed)
                try weights.require(name + ".pointwise_block.3.bias", [1, 1, width * 4], consumed: &consumed)
                try weights.requireLinear(name + ".pointwise_block.4", input: width * 4, output: width, consumed: &consumed)
            }
        }
        for index in 0..<config.layers {
            let prefix = "layers.\(index)"
            let width = config.hiddenSize
            for norm in ["ffn1_layer_norm", "ffn2_layer_norm", "attn_layer_norm", "final_layer_norm", "conv_module.layer_norm", "conv_module.conv_block.4.1"] {
                try weights.requireNorm(prefix + "." + norm, size: width, consumed: &consumed)
            }
            for ffn in ["ffn1", "ffn2"] {
                try weights.requireLinear(prefix + ".\(ffn).w_1", input: width, output: config.intermediateSize, consumed: &consumed)
                try weights.requireLinear(prefix + ".\(ffn).w_2", input: config.intermediateSize, output: width, consumed: &consumed)
            }
            for projection in ["query_proj", "key_proj", "value_proj", "out_proj"] {
                try weights.requireLinear(prefix + ".attn." + projection, input: width, output: width, consumed: &consumed)
            }
            for (name, shape) in [("1", [2 * width, width, 1]), ("3", [width, 1, config.kernel]), ("6", [width, width, 1])] {
                try weights.require(prefix + ".conv_module.conv_block.\(name).weight", shape, consumed: &consumed)
            }
        }
        guard Set(arrays.keys) == consumed else { throw SheetSage2Error.invalid("Unexpected MERT2 tensors: \(Set(arrays.keys).subtracting(consumed).sorted()).") }
    }

    /// FP32 HTK power log-mel, reflect padding, periodic Hann and the checkpoint's fixed normalization.
    public func mel(_ waveform: MLXArray) throws -> MLXArray {
        guard waveform.ndim == 1, waveform.size > 1024, all(isFinite(waveform)).item(Bool.self) else {
            throw SheetSage2Error.invalid("Expected finite mono audio with at least 1025 samples.")
        }
        let x = waveform.asType(.float32)
        let left = x[MLXArray(Array(stride(from: 1024, through: 1, by: -1)))]
        let right = x[MLXArray(Array(stride(from: x.size - 2, through: x.size - 1025, by: -1)))]
        let padded = concatenated([left, x, right])
        let count = x.size / 240
        let window = try weights.get("feature_extractor.spectrogram.window").asType(.float32)
        let filters = try weights.get("feature_extractor.mel_scale.fb").asType(.float32)
        var chunks: [MLXArray] = []
        for start in stride(from: 0, to: count, by: 512) {
            try Task.checkCancellation()
            let frames = MLX.asStrided(padded[(start * 240)...], [min(512, count - start), 2048], strides: [240, 1])
            let power = square(abs(MLX.rfft(frames * window, axis: -1)))
            let features = try (10 * log10(maximum(matmul(power, filters), 1e-10)) - weights.get("feature_extractor.mel_mean"))
                / maximum(weights.get("feature_extractor.mel_std"), 1e-5)
            MLX.eval(features)
            chunks.append(features)
        }
        return concatenated(chunks).expandedDimensions(axis: 0)
    }

    public func subsample(_ mel: MLXArray) throws -> MLXArray {
        var x = mel
        for index in config.channels.indices {
            let prefix = "subsampling_module.\(index)"
            let input = index == 0 ? config.melBins : config.channels[index - 1]
            let width = config.channels[index]
            if index > 0 || input != width {
                x = try weights.conv(weights.norm(x, prefix + ".resampling_layer.0", epsilon: config.subsamplingEpsilon),
                    prefix + ".resampling_layer.2", stride: index == 0 ? 1 : 2)
            }
            for layer in 0..<config.depths[index] {
                try Task.checkCancellation()
                let name = prefix + ".convnext_layers.\(layer)"
                var residual = try weights.conv(x, name + ".depthwise_block.1", padding: 3, groups: width)
                residual = try gelu(weights.linear(weights.norm(residual, name + ".pointwise_block.0", epsilon: config.subsamplingEpsilon),
                                                    name + ".pointwise_block.1"))
                let magnitude = sqrt(sum(square(residual), axis: 1, keepDims: true))
                let normalized = magnitude / (mean(magnitude, axis: -1, keepDims: true) + 1e-6)
                residual = try residual + weights.get(name + ".pointwise_block.3.weight") * residual * normalized
                    + weights.get(name + ".pointwise_block.3.bias")
                x = try x + weights.linear(residual, name + ".pointwise_block.4")
                MLX.eval(x)
            }
        }
        return x
    }

    public func block(_ input: MLXArray, index: Int) throws -> MLXArray {
        let prefix = "layers.\(index)"
        var x = try input + 0.5 * feedForward(weights.norm(input, prefix + ".ffn1_layer_norm", epsilon: config.epsilon), prefix + ".ffn1")
        let normalized = try weights.norm(x, prefix + ".attn_layer_norm", epsilon: config.epsilon)
        let width = config.hiddenSize / config.heads
        func project(_ name: String) throws -> MLXArray {
            try weights.linear(normalized, prefix + ".attn." + name).reshaped(1, -1, config.heads, width).transposed(0, 2, 1, 3)
        }
        func rotary(_ value: MLXArray) -> MLXArray {
            let frequencies = 1 / pow(config.theta, MLXArray(stride(from: 0, to: width, by: 2)).asType(.float32) / Float(width))
            let angles = MLXArray(0..<input.dim(1)).asType(.float32).reshaped(-1, 1) * frequencies
            let cosine = cos(angles).asType(value.dtype)
            let sine = sin(angles).asType(value.dtype)
            let a = value[.ellipsis, ..<(width / 2)]
            let b = value[.ellipsis, (width / 2)...]
            return concatenated([a * cosine - b * sine, b * cosine + a * sine], axis: -1)
        }
        let attended = try MLXFast.scaledDotProductAttention(queries: rotary(project("query_proj")),
            keys: rotary(project("key_proj")), values: project("value_proj"), scale: 1 / Float(width).squareRoot(), mask: .none)
        x = try x + weights.linear(attended.transposed(0, 2, 1, 3).reshaped(1, -1, config.hiddenSize), prefix + ".attn.out_proj")
        var convolution = try weights.conv(weights.norm(x, prefix + ".conv_module.layer_norm", epsilon: config.epsilon),
            prefix + ".conv_module.conv_block.1", bias: false)
        let halves = split(convolution, parts: 2, axis: -1)
        convolution = try weights.conv(halves[0] * sigmoid(halves[1]), prefix + ".conv_module.conv_block.3",
                                       padding: (config.kernel - 1) / 2, groups: config.hiddenSize, bias: false)
        convolution = try weights.norm(convolution, prefix + ".conv_module.conv_block.4.1", epsilon: config.epsilon)
        x = try x + weights.conv(gelu(convolution), prefix + ".conv_module.conv_block.6", bias: false)
        x = try x + 0.5 * feedForward(weights.norm(x, prefix + ".ffn2_layer_norm", epsilon: config.epsilon), prefix + ".ffn2")
        return try weights.norm(x, prefix + ".final_layer_norm", epsilon: config.epsilon)
    }

    private func feedForward(_ input: MLXArray, _ prefix: String) throws -> MLXArray {
        try weights.linear(gelu(weights.linear(input, prefix + ".w_1")), prefix + ".w_2")
    }
}
