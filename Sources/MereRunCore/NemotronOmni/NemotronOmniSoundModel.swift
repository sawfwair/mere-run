import Foundation
import MLX
import MLXFast
import MLXNN

final class NemotronOmniSoundFeedForward: Module {
    @ModuleInfo(key: "linear1") var first: Linear
    @ModuleInfo(key: "linear2") var second: Linear

    init(config: NemotronOmniSoundConfig) {
        self._first.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._second.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        second(silu(first(input)))
    }
}

final class NemotronOmniSoundAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "relative_k_proj") var relativeKeyProjection: Linear
    @ParameterInfo(key: "bias_u") var contentBias: MLXArray
    @ParameterInfo(key: "bias_v") var positionBias: MLXArray

    private let headCount: Int
    private let headDimension: Int
    private let scale: Float

    init(config: NemotronOmniSoundConfig) {
        self.headCount = config.numAttentionHeads
        self.headDimension = config.hiddenSize / config.numAttentionHeads
        self.scale = 1 / Float(Double(headDimension).squareRoot())
        self._queryProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        self._keyProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        self._valueProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        self._outputProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        self._relativeKeyProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        self._contentBias.wrappedValue = MLX.zeros(
            [config.numAttentionHeads, headDimension],
            dtype: .bfloat16
        )
        self._positionBias.wrappedValue = MLX.zeros(
            [config.numAttentionHeads, headDimension],
            dtype: .bfloat16
        )
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        positionEmbeddings: MLXArray,
        validLength: Int
    ) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let queries = queryProjection(input)
            .reshaped(batch, sequence, headCount, headDimension)
        let keys = keyProjection(input)
            .reshaped(batch, sequence, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        let values = valueProjection(input)
            .reshaped(batch, sequence, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        let queryContent = (
            queries + contentBias.reshaped(1, 1, headCount, headDimension)
        ).transposed(0, 2, 1, 3)
        let queryPosition = (
            queries + positionBias.reshaped(1, 1, headCount, headDimension)
        ).transposed(0, 2, 1, 3)
        let relativeKeys = relativeKeyProjection(positionEmbeddings)
            .reshaped(batch, positionEmbeddings.dim(1), headCount, headDimension)
            .transposed(0, 2, 1, 3)
        var relativeBias = MLX.matmul(
            queryPosition,
            relativeKeys.transposed(0, 1, 3, 2)
        )
        relativeBias = relativeShift(relativeBias)[0..., 0..., 0..., 0..<sequence] * scale
        if validLength < sequence {
            let valid = MLXArray((0..<sequence).map { $0 < validLength })
            let allowed = valid.reshaped(1, 1, sequence, 1)
                .&& valid.reshaped(1, 1, 1, sequence)
            relativeBias = MLX.where(
                allowed,
                relativeBias,
                MLXArray(-1e9).asType(relativeBias.dtype)
            )
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: queryContent,
            keys: keys,
            values: values,
            scale: scale,
            mask: .array(relativeBias)
        )
        return outputProjection(
            attended.transposed(0, 2, 1, 3)
                .reshaped(batch, sequence, headCount * headDimension)
        )
    }

    private func relativeShift(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let heads = input.dim(1)
        let queryLength = input.dim(2)
        let positionLength = input.dim(3)
        let padding = MLX.zeros([batch, heads, queryLength, 1], dtype: input.dtype)
        return MLX.concatenated([padding, input], axis: 3)
            .reshaped(batch, heads, positionLength + 1, queryLength)[
                0..., 0..., 1..<(positionLength + 1), 0...
            ]
            .reshaped(batch, heads, queryLength, positionLength)
    }
}

final class NemotronOmniSoundConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var firstPointwise: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwise: Conv1d
    @ModuleInfo(key: "norm") var norm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var secondPointwise: Conv1d

    init(config: NemotronOmniSoundConfig) {
        self._firstPointwise.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize * 2,
            kernelSize: 1,
            bias: false
        )
        self._depthwise.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.convKernelSize,
            padding: (config.convKernelSize - 1) / 2,
            groups: config.hiddenSize,
            bias: false
        )
        self._norm.wrappedValue = BatchNorm(featureCount: config.hiddenSize)
        self._secondPointwise.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: 1,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, validLength: Int) -> MLXArray {
        var hidden = firstPointwise(input)
        hidden = glu(hidden, axis: -1)
        hidden = maskPadding(hidden, validLength: validLength)
        hidden = depthwise(hidden)
        hidden = norm(hidden)
        hidden = silu(hidden)
        return secondPointwise(hidden)
    }

    private func maskPadding(_ input: MLXArray, validLength: Int) -> MLXArray {
        guard validLength < input.dim(1) else { return input }
        let mask = MLXArray((0..<input.dim(1)).map { $0 < validLength ? Float(1) : 0 })
            .reshaped(1, input.dim(1), 1)
            .asType(input.dtype)
        return input * mask
    }
}

final class NemotronOmniSoundBlock: Module {
    @ModuleInfo(key: "feed_forward1") var firstFeedForward: NemotronOmniSoundFeedForward
    @ModuleInfo(key: "self_attn") var attention: NemotronOmniSoundAttention
    @ModuleInfo(key: "conv") var convolution: NemotronOmniSoundConvolution
    @ModuleInfo(key: "feed_forward2") var secondFeedForward: NemotronOmniSoundFeedForward
    @ModuleInfo(key: "norm_feed_forward1") var firstFeedForwardNorm: LayerNorm
    @ModuleInfo(key: "norm_self_att") var attentionNorm: LayerNorm
    @ModuleInfo(key: "norm_conv") var convolutionNorm: LayerNorm
    @ModuleInfo(key: "norm_feed_forward2") var secondFeedForwardNorm: LayerNorm
    @ModuleInfo(key: "norm_out") var outputNorm: LayerNorm

    init(config: NemotronOmniSoundConfig) {
        self._firstFeedForward.wrappedValue = NemotronOmniSoundFeedForward(config: config)
        self._attention.wrappedValue = NemotronOmniSoundAttention(config: config)
        self._convolution.wrappedValue = NemotronOmniSoundConvolution(config: config)
        self._secondFeedForward.wrappedValue = NemotronOmniSoundFeedForward(config: config)
        self._firstFeedForwardNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        self._attentionNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        self._convolutionNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        self._secondFeedForwardNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        self._outputNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        positionEmbeddings: MLXArray,
        validLength: Int
    ) -> MLXArray {
        var hidden = input + 0.5 * firstFeedForward(firstFeedForwardNorm(input))
        hidden = hidden + attention(
            attentionNorm(hidden),
            positionEmbeddings: positionEmbeddings,
            validLength: validLength
        )
        hidden = hidden + convolution(convolutionNorm(hidden), validLength: validLength)
        hidden = hidden + 0.5 * secondFeedForward(secondFeedForwardNorm(hidden))
        return outputNorm(hidden)
    }
}

final class NemotronOmniSoundSubsampling: Module {
    @ModuleInfo(key: "layers") var layers: [Module]
    @ModuleInfo(key: "linear") var linear: Linear

    private let layerCount = 3

    init(config: NemotronOmniSoundConfig) {
        let channels = config.subsamplingConvChannels
        self._layers.wrappedValue = [
            Conv2d(
                inputChannels: 1,
                outputChannels: channels,
                kernelSize: 3,
                stride: 2,
                padding: 1,
                bias: true
            ),
            Identity(),
            Conv2d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: 3,
                stride: 2,
                padding: 1,
                groups: channels,
                bias: true
            ),
            Conv2d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: 1,
                bias: true
            ),
            Identity(),
            Conv2d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: 3,
                stride: 2,
                padding: 1,
                groups: channels,
                bias: true
            ),
            Conv2d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: 1,
                bias: true
            ),
        ]
        self._linear.wrappedValue = Linear(channels * (config.numMelBins / 8), config.hiddenSize)
        super.init()
    }

    func callAsFunction(_ input: MLXArray, validLength: Int) -> (MLXArray, Int) {
        var hidden = input.expandedDimensions(axis: -1)
        var currentValidLength = validLength
        for (index, module) in layers.enumerated() {
            if let convolution = module as? Conv2d {
                hidden = convolution(hidden)
                if index == 0 || index == 2 || index == 5 {
                    currentValidLength = (currentValidLength + 1) / 2
                }
                hidden = maskPadding(hidden, validLength: currentValidLength)
            }
            if index == 0 || index == 3 || index == 6 {
                hidden = relu(hidden)
            }
        }
        let batch = hidden.dim(0)
        let time = hidden.dim(1)
        // PyTorch's reference path is NCTF -> NT(CF) before the
        // subsampling projection. MLX Conv2d is NHWC, so transpose the
        // channel axis ahead of frequency before flattening.
        hidden = hidden.transposed(0, 1, 3, 2)
        return (
            linear(hidden.reshaped(batch, time, hidden.dim(2) * hidden.dim(3))),
            currentValidLength
        )
    }

    private func maskPadding(_ input: MLXArray, validLength: Int) -> MLXArray {
        guard validLength < input.dim(1) else { return input }
        let mask = MLXArray((0..<input.dim(1)).map { $0 < validLength ? Float(1) : 0 })
            .reshaped(1, input.dim(1), 1, 1)
            .asType(input.dtype)
        return input * mask
    }
}

final class NemotronOmniSoundEncoder: Module {
    @ModuleInfo(key: "subsampling") var subsampling: NemotronOmniSoundSubsampling
    @ModuleInfo(key: "layers") var layers: [NemotronOmniSoundBlock]

    private let hiddenSize: Int

    init(config: NemotronOmniSoundConfig) {
        self.hiddenSize = config.hiddenSize
        self._subsampling.wrappedValue = NemotronOmniSoundSubsampling(config: config)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            NemotronOmniSoundBlock(config: config)
        }
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        validFrameCount: Int,
        layerProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) -> MLXArray {
        var (hidden, validLength) = subsampling(input, validLength: validFrameCount)
        MLX.eval(hidden)
        layerProgress?(0, layers.count)
        let positions = relativePositions(length: hidden.dim(1), dtype: hidden.dtype)
        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                positionEmbeddings: positions,
                validLength: validLength
            )
            // Keep the 24-block conformer from becoming a single oversized
            // lazy Metal graph. Layer boundaries are also a useful memory
            // reclamation point for long recordings.
            MLX.eval(hidden)
            layerProgress?(index + 1, layers.count)
        }
        return hidden
    }

    private func relativePositions(length: Int, dtype: DType) -> MLXArray {
        let half = hiddenSize / 2
        let positions = MLXArray(
            (-(length - 1)...(length - 1)).reversed().map(Float.init)
        ).reshaped(1, 2 * length - 1, 1)
        let logIncrement = logf(10_000) / Float(max(1, half))
        let inverseValues: [Float] = (0..<half).map { index in
            expf(-Float(index) * logIncrement)
        }
        let inverse = MLXArray(inverseValues).reshaped(1, 1, half)
        let angles = positions * inverse
        return MLX.stacked([sin(angles), cos(angles)], axis: -1)
            .reshaped(1, 2 * length - 1, hiddenSize)
            .asType(dtype)
    }
}

final class NemotronOmniSoundProjection: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "linear1") var first: Linear
    @ModuleInfo(key: "linear2") var second: Linear

    init(config: NemotronOmniConfig) {
        self._norm.wrappedValue = RMSNorm(dimensions: config.sound.hiddenSize, eps: 1e-5)
        self._first.wrappedValue = Linear(
            config.sound.hiddenSize,
            config.sound.projectionHiddenSize,
            bias: false
        )
        self._second.wrappedValue = Linear(
            config.sound.projectionHiddenSize,
            config.language.hiddenSize,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let normalized = norm(input)
        MLX.eval(normalized)
        let hidden = first(normalized)
        MLX.eval(hidden)
        let activated = MLX.square(MLX.maximum(hidden, MLXArray(0).asType(hidden.dtype)))
        MLX.eval(activated)
        let output = second(activated)
        MLX.eval(output)
        return output
    }
}

final class NemotronOmniSoundTower: Module {
    @ModuleInfo(key: "encoder") var encoder: NemotronOmniSoundEncoder
    @ModuleInfo(key: "projection") var projection: NemotronOmniSoundProjection

    init(config: NemotronOmniConfig) {
        self._encoder.wrappedValue = NemotronOmniSoundEncoder(config: config.sound)
        self._projection.wrappedValue = NemotronOmniSoundProjection(config: config)
        super.init()
    }

    func callAsFunction(
        _ melFeatures: MLXArray,
        validFrameCount: Int,
        layerProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) -> MLXArray {
        projection(encoder(
            melFeatures,
            validFrameCount: validFrameCount,
            layerProgress: layerProgress
        ))
    }
}
