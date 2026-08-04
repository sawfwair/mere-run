import Foundation
import MLX
import MLXNN

private final class MiniMaxH3AudioSnake: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray

    init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.ones([1, channels, 1])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let frequency = alpha.transposed(0, 2, 1)
        let periodic = MLX.sin(input * frequency)
        return input + periodic * periodic / (frequency + 1e-9)
    }
}

private final class MiniMaxH3AudioResidualUnit: Module {
    @ModuleInfo(key: "block") private var layers: [Module]
    private let firstActivation: MiniMaxH3AudioSnake
    private let firstConvolution: Conv1d
    private let secondActivation: MiniMaxH3AudioSnake
    private let secondConvolution: Conv1d

    init(channels: Int, dilation: Int) {
        let firstActivation = MiniMaxH3AudioSnake(channels: channels)
        let firstConvolution = Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 7,
            padding: 3 * dilation,
            dilation: dilation
        )
        let secondActivation = MiniMaxH3AudioSnake(channels: channels)
        let secondConvolution = Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 1
        )
        self.firstActivation = firstActivation
        self.firstConvolution = firstConvolution
        self.secondActivation = secondActivation
        self.secondConvolution = secondConvolution
        self._layers.wrappedValue = [
            firstActivation,
            firstConvolution,
            secondActivation,
            secondConvolution,
        ]
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let update = secondConvolution(secondActivation(firstConvolution(firstActivation(input))))
        let difference = input.dim(1) - update.dim(1)
        guard difference > 0 else { return input + update }
        let left = difference / 2
        return input[0..., left..<(input.dim(1) - difference + left), 0...] + update
    }
}

private final class MiniMaxH3AudioEncoderBlock: Module {
    @ModuleInfo(key: "block") private var layers: [Module]
    private let firstResidual: MiniMaxH3AudioResidualUnit
    private let secondResidual: MiniMaxH3AudioResidualUnit
    private let thirdResidual: MiniMaxH3AudioResidualUnit
    private let activation: MiniMaxH3AudioSnake
    private let downsample: Conv1d

    init(inputChannels: Int, outputChannels: Int, stride: Int) {
        let firstResidual = MiniMaxH3AudioResidualUnit(channels: inputChannels, dilation: 1)
        let secondResidual = MiniMaxH3AudioResidualUnit(channels: inputChannels, dilation: 3)
        let thirdResidual = MiniMaxH3AudioResidualUnit(channels: inputChannels, dilation: 9)
        let activation = MiniMaxH3AudioSnake(channels: inputChannels)
        let downsample = Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: 2 * stride,
            stride: stride,
            padding: (stride + 1) / 2
        )
        self.firstResidual = firstResidual
        self.secondResidual = secondResidual
        self.thirdResidual = thirdResidual
        self.activation = activation
        self.downsample = downsample
        self._layers.wrappedValue = [firstResidual, secondResidual, thirdResidual, activation, downsample]
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downsample(activation(thirdResidual(secondResidual(firstResidual(input)))))
    }
}

private final class MiniMaxH3AudioEncoder: Module {
    @ModuleInfo(key: "block") private var layers: [Module]
    private let input: Conv1d
    private let firstBlock: MiniMaxH3AudioEncoderBlock
    private let secondBlock: MiniMaxH3AudioEncoderBlock
    private let thirdBlock: MiniMaxH3AudioEncoderBlock
    private let fourthBlock: MiniMaxH3AudioEncoderBlock
    private let fifthBlock: MiniMaxH3AudioEncoderBlock
    private let outputActivation: MiniMaxH3AudioSnake
    private let output: Conv1d

    override init() {
        let input = Conv1d(inputChannels: 1, outputChannels: 64, kernelSize: 7, padding: 3)
        let firstBlock = MiniMaxH3AudioEncoderBlock(
            inputChannels: 64, outputChannels: 128, stride: 2
        )
        let secondBlock = MiniMaxH3AudioEncoderBlock(
            inputChannels: 128, outputChannels: 256, stride: 4
        )
        let thirdBlock = MiniMaxH3AudioEncoderBlock(
            inputChannels: 256, outputChannels: 512, stride: 4
        )
        let fourthBlock = MiniMaxH3AudioEncoderBlock(
            inputChannels: 512, outputChannels: 1_024, stride: 5
        )
        let fifthBlock = MiniMaxH3AudioEncoderBlock(
            inputChannels: 1_024, outputChannels: 2_048, stride: 5
        )
        let outputActivation = MiniMaxH3AudioSnake(channels: 2_048)
        let output = Conv1d(
            inputChannels: 2_048, outputChannels: 2_048, kernelSize: 3, padding: 1
        )
        self.input = input
        self.firstBlock = firstBlock
        self.secondBlock = secondBlock
        self.thirdBlock = thirdBlock
        self.fourthBlock = fourthBlock
        self.fifthBlock = fifthBlock
        self.outputActivation = outputActivation
        self.output = output
        self._layers.wrappedValue = [
            input,
            firstBlock,
            secondBlock,
            thirdBlock,
            fourthBlock,
            fifthBlock,
            outputActivation,
            output,
        ]
        super.init()
    }

    func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        output(outputActivation(fifthBlock(fourthBlock(thirdBlock(secondBlock(firstBlock(input(waveform))))))))
    }
}

private final class MiniMaxH3AudioCausalAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ParameterInfo(key: "q_bias") var queryBias: MLXArray
    @ParameterInfo(key: "v_bias") var valueBias: MLXArray
    @ParameterInfo(key: "zero_k_bias") var keyBias: MLXArray
    @ModuleInfo(key: "proj") var output: Linear

    private let headCount = 8
    private let headDimension = 256

    override init() {
        self._qkv.wrappedValue = Linear(2_048, 6_144, bias: false)
        self._queryBias.wrappedValue = MLXArray.zeros([2_048])
        self._valueBias.wrappedValue = MLXArray.zeros([2_048])
        self._keyBias.wrappedValue = MLXArray.zeros([2_048])
        self._output.wrappedValue = Linear(32, 32)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let bias = MLX.concatenated([queryBias, keyBias, valueBias])
        let projected = (qkv(input) + bias)
            .reshaped(input.dim(0), input.dim(1), 3, headCount, headDimension)
            .transposed(2, 0, 3, 1, 4)
        let queries = projected[0]
        let keys = projected[1]
        let values = projected[2]
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / sqrt(Float(headDimension)),
            mask: .causal
        ).transposed(0, 2, 1, 3)
        let pooledHeads = MLX.mean(attended, axis: 2)
        let pooledFeatures = MLX.mean(
            pooledHeads.reshaped(input.dim(0), input.dim(1), 32, headDimension / 32),
            axis: 3
        )
        return output(pooledFeatures)
    }
}

private final class MiniMaxH3AudioGeGLU: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "w0") var gate: Linear
    @ModuleInfo(key: "w1") var value: Linear
    @ModuleInfo(key: "w2") var output: Linear

    override init() {
        self._norm.wrappedValue = LayerNorm(dimensions: 32)
        self._gate.wrappedValue = Linear(32, 64)
        self._value.wrappedValue = Linear(32, 64)
        self._output.wrappedValue = Linear(64, 32)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let normalized = norm(input)
        return output(geluApproximate(gate(normalized)) * value(normalized))
    }

    private func geluApproximate(_ input: MLXArray) -> MLXArray {
        0.5 * input * (1 + MLX.tanh(sqrt(2 / Float.pi) * (input + 0.044715 * input * input * input)))
    }
}

private final class MiniMaxH3AudioAttentionProjection: Module {
    @ModuleInfo(key: "norm1") var attentionNorm: LayerNorm
    @ModuleInfo(key: "attn") var attention: MiniMaxH3AudioCausalAttention
    @ModuleInfo(key: "proj") var projection: Linear
    @ModuleInfo(key: "norm3") var projectionNorm: LayerNorm
    @ModuleInfo(key: "norm2") var feedForwardNorm: LayerNorm
    @ModuleInfo(key: "mlp") var feedForward: MiniMaxH3AudioGeGLU

    override init() {
        self._attentionNorm.wrappedValue = LayerNorm(dimensions: 2_048)
        self._attention.wrappedValue = MiniMaxH3AudioCausalAttention()
        self._projection.wrappedValue = Linear(2_048, 32)
        self._projectionNorm.wrappedValue = LayerNorm(dimensions: 2_048)
        self._feedForwardNorm.wrappedValue = LayerNorm(dimensions: 32)
        self._feedForward.wrappedValue = MiniMaxH3AudioGeGLU()
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let hidden = projection(projectionNorm(input)) + attention(attentionNorm(input))
        return hidden + feedForward(feedForwardNorm(hidden))
    }
}

/// H3's waveform autoencoder. Stereo is represented as two batch items
/// throughout the model and only interleaved at the media boundary.
public final class MiniMaxH3AudioVAE: Module {
    public static let samplingRate = 32_000
    public static let hopLength = 800

    @ModuleInfo(key: "encoder") private var encoder: MiniMaxH3AudioEncoder
    @ModuleInfo(key: "pre_block") private var preBlock: MiniMaxH3AudioAttentionProjection
    @ModuleInfo(key: "mean_proj") var meanProjection: Conv1d
    @ModuleInfo(key: "logs_proj") var logStandardDeviationProjection: Conv1d
    @ModuleInfo(key: "dec_in_proj") var inputProjection: Conv1d
    @ModuleInfo(key: "decoder") var decoder: MMAudioBigVGAN
    @ParameterInfo(key: "latents_mean") var latentMean: MLXArray
    @ParameterInfo(key: "latents_std") var latentStandardDeviation: MLXArray

    public override init() {
        _encoder.wrappedValue = MiniMaxH3AudioEncoder()
        _preBlock.wrappedValue = MiniMaxH3AudioAttentionProjection()
        _meanProjection.wrappedValue = Conv1d(inputChannels: 32, outputChannels: 32, kernelSize: 1)
        _logStandardDeviationProjection.wrappedValue = Conv1d(
            inputChannels: 32, outputChannels: 32, kernelSize: 1
        )
        _inputProjection.wrappedValue = Conv1d(
            inputChannels: 32,
            outputChannels: 2_048,
            kernelSize: 1,
            bias: true
        )
        _decoder.wrappedValue = MMAudioBigVGAN(
            inputChannels: 2_048,
            initialChannels: 1_024,
            upsampleRates: [5, 5, 2, 2, 2, 2, 2],
            upsampleKernelSizes: [9, 9, 4, 4, 4, 4, 4],
            useFloat32: true
        )
        _latentMean.wrappedValue = MLXArray.zeros([32])
        _latentStandardDeviation.wrappedValue = MLXArray.ones([32])
        super.init()
    }

    /// Encodes `[1, samples, 2]` stereo to normalized `[1, 32, 2, T]`
    /// latents. H3 conditions on the posterior mean, never a posterior sample.
    public func encode(_ waveform: MLXArray) -> MLXArray {
        precondition(waveform.ndim == 3 && waveform.dim(0) == 1 && waveform.dim(2) == 2)
        let sampleCount = waveform.dim(1)
        let paddedCount = ((sampleCount + Self.hopLength - 1) / Self.hopLength) * Self.hopLength
        var stereoBatch = waveform[0].transposed(1, 0).expandedDimensions(axis: 2)
        if paddedCount > sampleCount {
            stereoBatch = MLX.padded(
                stereoBatch,
                widths: [[0, 0], [0, paddedCount - sampleCount], [0, 0]]
            )
        }
        let trunk = encoder(stereoBatch.asType(.float32))
        let mean = meanProjection(preBlock(trunk))
        let channelsFirst = mean.transposed(0, 2, 1)
        let normalized = (
            channelsFirst - latentMean.reshaped(1, 32, 1)
        ) / latentStandardDeviation.reshaped(1, 32, 1)
        return normalized.transposed(1, 0, 2).expandedDimensions(axis: 0)
    }

    /// Decodes normalized `[1, 32, 2, T]` latents to `[1, samples, 2]` stereo.
    public func decode(_ normalizedLatents: MLXArray) -> MLXArray {
        precondition(
            normalizedLatents.ndim == 4
                && normalizedLatents.dim(0) == 1
                && normalizedLatents.dim(1) == 32
                && normalizedLatents.dim(2) == 2
        )
        let mean = latentMean.reshaped(1, 32, 1, 1)
        let standardDeviation = latentStandardDeviation.reshaped(1, 32, 1, 1)
        let denormalized = normalizedLatents * standardDeviation + mean
        let stereoBatch = denormalized[0].transposed(1, 2, 0)
        let projected = inputProjection(stereoBatch.asType(.float32))
        let monoBatch = decoder(projected)
        return monoBatch[0..., 0..., 0].transposed(1, 0).expandedDimensions(axis: 0)
    }

    static func mapConvertedWeight(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key == "latents_mean" || key == "latents_std" {
            return [(key, value)]
        }
        let mappedKey = key
        let plainConvolutionPrefixes = ["encoder.", "mean_proj.", "logs_proj.", "dec_in_proj."]
        if plainConvolutionPrefixes.contains(where: mappedKey.hasPrefix),
           mappedKey.hasSuffix(".weight"), value.ndim == 3 {
            return [(mappedKey, value.transposed(0, 2, 1))]
        }
        if mappedKey.hasPrefix("encoder.") || mappedKey.hasPrefix("pre_block.") {
            return [(mappedKey, value)]
        }
        guard mappedKey.hasPrefix("decoder.") else { return [] }
        let suffix = String(mappedKey.dropFirst("decoder.".count))
        return MMAudioBigVGAN.mapSafetensorsWeights(key: suffix, value: value).map {
            ("decoder.\($0.0)", $0.1)
        }
    }
}
