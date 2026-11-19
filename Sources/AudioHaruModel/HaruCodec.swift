import MLX
import MLXNN
import MLXRandom

private final class HaruEncoder: Module {
    let stem: HaruCausalConvolution
    let down: [HaruCausalConvolution]
    let stages: [[HaruCodecBlock]]
    let head: HaruCausalConvolution

    init(_ config: HaruConfiguration.Codec) {
        let base = config.baseChannels
        stem = HaruCausalConvolution(input: 1, output: base, kernel: 7)
        down = config.ratios.reversed().enumerated().map { index, hop in
            HaruCausalConvolution(input: base << index, output: base << (index + 1), kernel: 2 * hop, hop: hop)
        }
        stages = config.encoderDepths.enumerated().map { index, depth in
            (0..<depth).map { _ in HaruCodecBlock(channels: base << index, epsilon: config.normEps) }
        }
        head = HaruCausalConvolution(input: base << config.ratios.count, output: config.latentDim, kernel: 7)
    }

    var layers: [HaruCodecLayer] {
        stages.enumerated().flatMap { index, stage -> [HaruCodecLayer] in
            [index == 0 ? stem : down[index - 1]] + stage
        } + [head]
    }
}

private final class HaruDecoder: Module {
    let stem: HaruCausalConvolution
    let up: [HaruCausalUpsampling]
    let stages: [[HaruCodecBlock]]
    let head: HaruCausalConvolution

    init(_ config: HaruConfiguration.Codec) {
        let base = config.baseChannels
        let count = config.ratios.count
        stem = HaruCausalConvolution(input: config.latentDim, output: base << count, kernel: 7)
        up = config.ratios.enumerated().map { index, hop in
            HaruCausalUpsampling(input: base << (count - index), output: base << (count - index - 1), stride: hop)
        }
        stages = config.decoderDepths.enumerated().map { index, depth in
            (0..<depth).map { _ in HaruCodecBlock(channels: base << (count - index), epsilon: config.normEps) }
        }
        head = HaruCausalConvolution(input: base, output: 1, kernel: 7)
    }

    var layers: [HaruCodecLayer] {
        stages.enumerated().flatMap { index, stage -> [HaruCodecLayer] in
            let transition: HaruCodecLayer = index == 0 ? stem : up[index - 1]
            return [transition] + stage
        } + [head]
    }
}

final class HaruCodec: Module {
    private let encoder: HaruEncoder
    private let decoder: HaruDecoder
    private let config: HaruConfiguration.Codec

    init(config: HaruConfiguration.Codec) {
        self.config = config
        encoder = HaruEncoder(config)
        decoder = HaruDecoder(config)
    }

    var encoderState: [MLXArray] { state(for: encoder.layers, dtype: encoder.head.weight.dtype) }
    var decoderState: [MLXArray] { state(for: decoder.layers, dtype: decoder.head.weight.dtype) }

    private func state(for layers: [HaruCodecLayer], dtype: DType) -> [MLXArray] {
        layers.map { MLXArray.zeros($0.stateShape, dtype: dtype) }
    }

    func encode(_ audio: MLXArray, state: inout [MLXArray]) -> MLXArray {
        run(encoder.layers, input: audio.reshaped(-1, 1).asType(encoder.head.weight.dtype), state: &state)
            .asType(.float32)
    }

    func decode(_ latents: MLXArray, state: inout [MLXArray]) -> MLXArray {
        run(decoder.layers, input: latents.asType(decoder.head.weight.dtype), state: &state)
            .reshaped(-1).asType(.float32)
    }

    func sample(_ mean: MLXArray, key: MLXArray) -> MLXArray {
        let (scaleKey, noiseKey) = MLXRandom.split(key: key)
        let scale = MLXRandom.normal([1], key: scaleKey)
        return mean + scale * (config.latentStd / 0.8) * MLXRandom.normal(mean.shape, key: noiseKey)
    }

    private func run(_ layers: [HaruCodecLayer], input: MLXArray, state: inout [MLXArray]) -> MLXArray {
        var hidden = input
        for (index, layer) in layers.enumerated() {
            let (output, next) = layer(hidden, state: state[index])
            hidden = output
            state[index] = next
        }
        return hidden
    }
}
