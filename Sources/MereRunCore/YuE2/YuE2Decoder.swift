import AudioCore
import Foundation
import MLX

// Oobleck/SnakeBeta translated from YuE2 and stable-audio-tools; see THIRD_PARTY_NOTICES.md.
struct YuE2Convolution {
    let weight: MLXArray
    let bias: MLXArray?
    let stride: Int
    let padding: Int
    let dilation: Int
    let transposed: Bool

    init(
        _ weights: inout YuE2Weights, _ prefix: String, input: Int, output: Int,
        kernel: Int, stride: Int = 1, padding: Int, dilation: Int = 1,
        transposed: Bool = false, bias: Bool = true
    ) throws {
        let shape = transposed ? [input, output, kernel] : [output, input, kernel]
        let direction = try weights.take(prefix + ".weight_v", shape).asType(.float32)
        let magnitude = try weights.take(prefix + ".weight_g", [shape[0], 1, 1]).asType(.float32)
        let normalized = direction * (magnitude / sqrt(sum(square(direction), axes: [1, 2], keepDims: true)))
        weight = transposed ? normalized.transposed(1, 2, 0) : normalized.transposed(0, 2, 1)
        self.bias = try bias ? weights.take(prefix + ".bias", [output]).asType(.float32) : nil
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        MLX.eval(weight)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let result = transposed
            ? MLX.convTransposed1d(x, weight, stride: stride, padding: padding)
            : MLX.conv1d(x, weight, stride: stride, padding: padding, dilation: dilation)
        return bias.map { result + $0 } ?? result
    }
}

struct YuE2Snake {
    let alpha: MLXArray
    let beta: MLXArray

    init(_ weights: inout YuE2Weights, _ prefix: String, channels: Int) throws {
        alpha = exp(try weights.take(prefix + ".alpha", [channels]).asType(.float32))
        beta = exp(try weights.take(prefix + ".beta", [channels]).asType(.float32))
        MLX.eval(alpha, beta)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x + square(sin(x * alpha)) / (beta + 1e-9) }
}

struct YuE2ResidualUnit {
    let firstActivation: YuE2Snake
    let first: YuE2Convolution
    let secondActivation: YuE2Snake
    let second: YuE2Convolution

    init(_ weights: inout YuE2Weights, _ prefix: String, channels: Int, dilation: Int) throws {
        firstActivation = try YuE2Snake(&weights, prefix + ".layers.0", channels: channels)
        first = try YuE2Convolution(&weights, prefix + ".layers.1", input: channels, output: channels,
                                   kernel: 7, padding: dilation * 3, dilation: dilation)
        secondActivation = try YuE2Snake(&weights, prefix + ".layers.2", channels: channels)
        second = try YuE2Convolution(&weights, prefix + ".layers.3", input: channels, output: channels,
                                    kernel: 1, padding: 0)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x + second(secondActivation(first(firstActivation(x)))) }
}

struct YuE2DecoderBlock {
    let activation: YuE2Snake
    let upsample: YuE2Convolution
    let residual: [YuE2ResidualUnit]

    init(_ weights: inout YuE2Weights, _ prefix: String, input: Int, output: Int, stride: Int) throws {
        activation = try YuE2Snake(&weights, prefix + ".layers.0", channels: input)
        upsample = try YuE2Convolution(&weights, prefix + ".layers.1", input: input, output: output,
                                      kernel: 2 * stride, stride: stride, padding: (stride + 1) / 2, transposed: true)
        residual = try [1, 3, 9].enumerated().map {
            try YuE2ResidualUnit(&weights, prefix + ".layers.\($0.offset + 2)", channels: output, dilation: $0.element)
        }
    }

    func callAsFunction(_ x: MLXArray) throws -> MLXArray {
        var hidden = upsample(activation(x))
        for unit in residual {
            try Task.checkCancellation()
            hidden = unit(hidden)
            MLX.eval(hidden)
        }
        return hidden
    }
}

struct YuE2Decoder {
    let configuration: YuE2VAEConfiguration
    let input: YuE2Convolution
    let blocks: [YuE2DecoderBlock]
    let activation: YuE2Snake
    let output: YuE2Convolution

    init(configuration: YuE2VAEConfiguration, arrays: [String: MLXArray]) throws {
        self.configuration = configuration
        let config = configuration.decoderConfig
        let channels = ([1] + config.channelMultipliers).map { $0 * config.channels }
        var weights = YuE2Weights(arrays)
        input = try YuE2Convolution(&weights, "decoder.layers.0", input: config.latentDim,
                                   output: channels[channels.count - 1], kernel: 7, padding: 3)
        blocks = try config.strides.indices.reversed().enumerated().map { position, index in
            try YuE2DecoderBlock(&weights, "decoder.layers.\(position + 1)", input: channels[index + 1],
                                 output: channels[index], stride: config.strides[index])
        }
        activation = try YuE2Snake(&weights, "decoder.layers.\(blocks.count + 1)", channels: channels[0])
        output = try YuE2Convolution(&weights, "decoder.layers.\(blocks.count + 2)", input: channels[0],
                                    output: config.outChannels, kernel: 7, padding: 3, bias: false)
        try weights.finish()
        guard configuration.decodeHaloFrames >= requiredHalo(coreFrames: configuration.decodeCoreFrames) else {
            throw YuE2Error.invalidConfiguration("VAE halo is smaller than the decoder's receptive field.")
        }
    }

    /// Input [frames,64], output [1,samples,2]. All decoder arithmetic is FP32.
    func decode(_ latents: MLXArray) throws -> MLXArray {
        var hidden = input(latents.asType(.float32).expandedDimensions(axis: 0))
        for block in blocks { hidden = try block(hidden) }
        return output(activation(hidden))
    }

    func outputLength(frames: Int) -> Int {
        configuration.decoderConfig.strides.reversed().reduce(frames) { length, stride in
            (length - 1) * stride - 2 * ((stride + 1) / 2) + 2 * stride
        }
    }

    /// Propagate an output interval backwards through each convolution. Retained
    /// cores are exact; overlapping decoder context is discarded without blending.
    func requiredHalo(coreFrames: Int) -> Int {
        var low = -3
        var high = coreFrames * configuration.downsamplingRatio - 1 + 3
        for stride in configuration.decoderConfig.strides {
            low -= 3 * (1 + 3 + 9)
            high += 3 * (1 + 3 + 9)
            let padding = (stride + 1) / 2
            low = -Self.floorDivide(-(low + padding - (2 * stride - 1)), stride)
            high = Self.floorDivide(high + padding, stride)
        }
        low -= 3
        high += 3
        return max(0, -low, high - coreFrames + 1)
    }

    private static func floorDivide(_ value: Int, _ divisor: Int) -> Int {
        let quotient = value / divisor
        return value < 0 && value % divisor != 0 ? quotient - 1 : quotient
    }

    func decodeTiled(_ latents: MLXArray, progress: (Int, Int) -> Void) throws -> AudioWaveform {
        let frames = latents.dim(0)
        let core = configuration.decodeCoreFrames
        let halo = configuration.decodeHaloFrames
        let ratio = configuration.downsamplingRatio
        let samples = outputLength(frames: frames)
        var interleaved: [Float] = []
        interleaved.reserveCapacity(samples * configuration.audioChannels)
        for start in stride(from: 0, to: frames, by: core) {
            try Task.checkCancellation()
            let end = min(start + core, frames)
            let left = max(0, start - halo)
            let right = min(frames, end + halo)
            let tile = try decode(latents[left..<right])
            let first = (start - left) * ratio
            let count = min(end * ratio, samples) - start * ratio
            guard first + count <= tile.dim(1) else {
                throw YuE2Error.invalidAudio("Decoded tile does not cover its output core.")
            }
            let crop = tile[0, first..<(first + count), 0...].reshaped(-1)
            MLX.eval(crop)
            interleaved.append(contentsOf: crop.asArray(Float.self))
            progress(min(end, frames), frames)
        }
        guard interleaved.allSatisfy(\.isFinite) else {
            throw YuE2Error.invalidAudio("Decoder produced non-finite samples.")
        }
        return try AudioWaveform(interleaved: interleaved, channels: configuration.audioChannels,
                                 sampleRate: configuration.sampleRate)
    }
}
