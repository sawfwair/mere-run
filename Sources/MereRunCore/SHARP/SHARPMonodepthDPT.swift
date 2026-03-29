import MLX
import MLXNN

public protocol SharpMonodepthEncoder: Sendable {
    func callAsFunction(_ image: MLXArray) -> [MLXArray]
    func featureDims() -> [Int]
    func internalResolution() -> Int
}

public final class SharpMonodepthDensePredictionTransformer: Module, SharpMonodepthPredictorCore, @unchecked Sendable {
    public let normalizer: SharpAffineRangeNormalizer
    public let encoder: any SharpMonodepthEncoder

    @ModuleInfo(key: "decoder") var decoder: SharpMultiresConvDecoder
    @ModuleInfo(key: "head_conv_0") var headConv0: SharpConv2dNCHW
    @ModuleInfo(key: "head_deconv") var headDeconv: SharpConvTransposed2dNCHW
    @ModuleInfo(key: "head_conv_1") var headConv1: SharpConv2dNCHW
    @ModuleInfo(key: "head_conv_2") var headConv2: SharpConv2dNCHW

    public init(
        encoder: any SharpMonodepthEncoder,
        decoder: SharpMultiresConvDecoder,
        lastDims: (Int, Int) = (32, 1)
    ) {
        self.encoder = encoder
        self.normalizer = SharpAffineRangeNormalizer(inputRange: (0, 1), outputRange: (-1, 1))
        self._decoder.wrappedValue = decoder

        let dimDecoder = decoder.dimOut
        self._headConv0.wrappedValue = SharpConv2dNCHW(
            inputChannels: dimDecoder,
            outputChannels: dimDecoder / 2,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            bias: true
        )
        self._headDeconv.wrappedValue = SharpConvTransposed2dNCHW(
            inputChannels: dimDecoder / 2,
            outputChannels: dimDecoder / 2,
            kernelSize: 2,
            stride: 2,
            padding: 0,
            bias: true
        )
        self._headConv1.wrappedValue = SharpConv2dNCHW(
            inputChannels: dimDecoder / 2,
            outputChannels: lastDims.0,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            bias: true
        )
        self._headConv2.wrappedValue = SharpConv2dNCHW(
            inputChannels: lastDims.0,
            outputChannels: lastDims.1,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: true
        )
        super.init()
    }

    public func callAsFunction(_ image: MLXArray) -> SharpMonodepthPredictorOutput {
        let normalized = normalizer(image)
        let encoderOutput = encoder(normalized)
        let numEncoderFeatures = encoder.featureDims().count

        let encoderFeatures = Array(encoderOutput.prefix(numEncoderFeatures))
        let intermediateFeatures = encoderOutput.count > numEncoderFeatures
            ? Array(encoderOutput.dropFirst(numEncoderFeatures))
            : []

        let decoderFeatures = decoder(encoderFeatures)
        var disparity = headConv0(decoderFeatures)
        disparity = headDeconv(disparity)
        disparity = headConv1(disparity)
        disparity = relu(disparity)
        disparity = headConv2(disparity)
        disparity = relu(disparity)

        return SharpMonodepthPredictorOutput(
            disparity: disparity,
            encoderFeatures: encoderFeatures,
            decoderFeatures: decoderFeatures,
            intermediateFeatures: intermediateFeatures
        )
    }

    public func internalResolution() -> Int {
        encoder.internalResolution()
    }

    public func encoderFeatureDims() -> [Int] {
        encoder.featureDims()
    }

    public func decoderFeatureDim() -> Int {
        decoder.dimOut
    }
}
