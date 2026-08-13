import MLX
import MLXNN

final class MiniMaxMusic3Snake: Module {
    @ModuleInfo(key: "alpha") var alpha: MLXArray

    init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.ones([1, 1, channels])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        input + MLX.square(MLX.sin(alpha * input)) / (alpha + MLXArray(1e-9).asType(input.dtype))
    }
}

final class MiniMaxMusic3VocoderResidualUnit: Module {
    @ModuleInfo(key: "snake1") var snake1: MiniMaxMusic3Snake
    @ModuleInfo(key: "conv1") var convolution1: WNConv1d
    @ModuleInfo(key: "snake2") var snake2: MiniMaxMusic3Snake
    @ModuleInfo(key: "conv2") var convolution2: WNConv1d

    init(dimensions: Int, dilation: Int) {
        self._snake1.wrappedValue = MiniMaxMusic3Snake(channels: dimensions)
        self._convolution1.wrappedValue = WNConv1d(
            inputChannels: dimensions,
            outputChannels: dimensions,
            kernelSize: 7,
            padding: 3 * dilation,
            dilation: dilation
        )
        self._snake2.wrappedValue = MiniMaxMusic3Snake(channels: dimensions)
        self._convolution2.wrappedValue = WNConv1d(
            inputChannels: dimensions,
            outputChannels: dimensions,
            kernelSize: 1
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        input + convolution2(snake2(convolution1(snake1(input))))
    }
}

final class MiniMaxMusic3VocoderBlock: Module {
    @ModuleInfo(key: "snake1") var snake1: MiniMaxMusic3Snake
    @ModuleInfo(key: "conv_t1") var transposedConvolution: OobleckWNConvTranspose1d
    @ModuleInfo(key: "res_unit1") var residualUnit1: MiniMaxMusic3VocoderResidualUnit
    @ModuleInfo(key: "res_unit2") var residualUnit2: MiniMaxMusic3VocoderResidualUnit
    @ModuleInfo(key: "res_unit3") var residualUnit3: MiniMaxMusic3VocoderResidualUnit

    init(inputDimensions: Int, outputDimensions: Int, stride: Int) {
        self._snake1.wrappedValue = MiniMaxMusic3Snake(channels: inputDimensions)
        self._transposedConvolution.wrappedValue = OobleckWNConvTranspose1d(
            inputChannels: inputDimensions,
            outputChannels: outputDimensions,
            kernelSize: 2 * stride,
            stride: stride,
            padding: (stride + 1) / 2
        )
        self._residualUnit1.wrappedValue = MiniMaxMusic3VocoderResidualUnit(
            dimensions: outputDimensions,
            dilation: 1
        )
        self._residualUnit2.wrappedValue = MiniMaxMusic3VocoderResidualUnit(
            dimensions: outputDimensions,
            dilation: 3
        )
        self._residualUnit3.wrappedValue = MiniMaxMusic3VocoderResidualUnit(
            dimensions: outputDimensions,
            dilation: 9
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = transposedConvolution(snake1(input))
        hidden = residualUnit1(hidden)
        hidden = residualUnit2(hidden)
        return residualUnit3(hidden)
    }
}

public final class MiniMaxMusic3Vocoder: Module {
    public let configuration: MiniMaxMusic3VocoderConfiguration

    @ModuleInfo(key: "dec_in_proj") var decoderInputProjection: Conv1d
    @ModuleInfo(key: "conv_in") var inputConvolution: WNConv1d
    @ModuleInfo(key: "blocks") var blocks: [MiniMaxMusic3VocoderBlock]
    @ModuleInfo(key: "snake_out") var outputSnake: MiniMaxMusic3Snake
    @ModuleInfo(key: "conv_out") var outputConvolution: WNConv1d

    public init(configuration: MiniMaxMusic3VocoderConfiguration) {
        self.configuration = configuration
        self._decoderInputProjection.wrappedValue = Conv1d(
            inputChannels: configuration.latentChannels / 2,
            outputChannels: configuration.decoderInputDim,
            kernelSize: 1
        )
        self._inputConvolution.wrappedValue = WNConv1d(
            inputChannels: configuration.decoderInputDim,
            outputChannels: configuration.decoderHiddenDim,
            kernelSize: 7,
            padding: 3
        )
        self._blocks.wrappedValue = configuration.upsamplingRatios.enumerated().map { index, stride in
            MiniMaxMusic3VocoderBlock(
                inputDimensions: configuration.decoderHiddenDim / (1 << index),
                outputDimensions: configuration.decoderHiddenDim / (1 << (index + 1)),
                stride: stride
            )
        }
        let outputDimensions = configuration.decoderHiddenDim / (1 << configuration.upsamplingRatios.count)
        self._outputSnake.wrappedValue = MiniMaxMusic3Snake(channels: outputDimensions)
        self._outputConvolution.wrappedValue = WNConv1d(
            inputChannels: outputDimensions,
            outputChannels: 1,
            kernelSize: 7,
            padding: 3
        )
    }

    public func callAsFunction(_ latents: MLXArray) -> MLXArray {
        let batch = latents.dim(0)
        let length = latents.dim(2)
        var hidden = latents.reshaped(batch * 2, configuration.latentChannels / 2, length)
            .transposed(0, 2, 1)
        hidden = inputConvolution(decoderInputProjection(hidden))
        for block in blocks {
            hidden = block(hidden)
        }
        let waveform = MLX.tanh(outputConvolution(outputSnake(hidden)))
        return waveform.reshaped(batch, 2, waveform.dim(1))
    }

    static func mapWeight(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key.hasSuffix(".alpha") {
            return [(key, value.transposed(0, 2, 1))]
        }
        if key.hasSuffix("conv_t1.weight_v") {
            return [(key, value.transposed(1, 2, 0))]
        }
        if key == "dec_in_proj.weight" || key.hasSuffix(".weight_v") {
            return [(key, value.transposed(0, 2, 1))]
        }
        return [(key, value)]
    }
}
