import MLX
import MLXNN

public func createSharpGaussianDecoder(
    params: SharpGaussianDecoderParameters,
    dimsDepthFeatures: [Int]
) -> SharpGaussianDensePredictionTransformer {
    let decoder = SharpMultiresConvDecoder(
        dimsEncoder: dimsDepthFeatures,
        dimsDecoder: params.dimsDecoder,
        upsamplingMode: params.upsamplingMode
    )
    return SharpGaussianDensePredictionTransformer(
        decoder: decoder,
        dimIn: params.dimIn,
        dimOut: params.dimOut,
        strideOut: params.stride,
        imageEncoderType: params.imageEncoderType,
        normType: params.normType,
        normNumGroups: params.normNumGroups,
        useDepthInput: params.useDepthInput
    )
}

final class SharpProjectUpsampleBlock: Module, @unchecked Sendable {
    @ModuleInfo(key: "projection") var projection: SharpConv2dNCHW
    @ModuleInfo(key: "upsamplers") var upsamplers: [SharpConvTransposed2dNCHW]

    init(dimIn: Int, dimOut: Int, upsampleLayers: Int, dimIntermediate: Int? = nil) {
        let intermediate = dimIntermediate ?? dimOut
        self._projection.wrappedValue = SharpConv2dNCHW(
            inputChannels: dimIn,
            outputChannels: intermediate,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: false
        )

        var layers: [SharpConvTransposed2dNCHW] = []
        for i in 0..<upsampleLayers {
            layers.append(SharpConvTransposed2dNCHW(
                inputChannels: i == 0 ? intermediate : dimOut,
                outputChannels: dimOut,
                kernelSize: 2,
                stride: 2,
                padding: 0,
                bias: false
            ))
        }
        self._upsamplers.wrappedValue = layers
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = projection(x)
        for layer in upsamplers {
            y = layer(y)
        }
        return y
    }
}

public final class SharpSkipConvBackbone: Module, @unchecked Sendable {
    @ModuleInfo(key: "conv") var conv: SharpConv2dNCHW
    public let strideOut: Int

    public init(dimIn: Int, dimOut: Int, kernelSize: Int, strideOut: Int) {
        precondition(!(strideOut == 1 && kernelSize != 1), "kernel_size must be 1 when stride_out is 1.")
        self.strideOut = strideOut
        let padding = (kernelSize - 1) / 2
        self._conv.wrappedValue = SharpConv2dNCHW(
            inputChannels: dimIn,
            outputChannels: dimOut,
            kernelSize: kernelSize,
            stride: strideOut,
            padding: padding,
            bias: true
        )
        super.init()
    }

    public func callAsFunction(_ inputFeatures: MLXArray) -> SharpImageFeatures {
        let output = conv(inputFeatures)
        return SharpImageFeatures(textureFeatures: output, geometryFeatures: output)
    }

    public var stride: Int { strideOut }
}

final class SharpGaussianDecoderHead: Module, @unchecked Sendable {
    @ModuleInfo(key: "residual_1") var residual1: SharpResidualBlock2D
    @ModuleInfo(key: "residual_2") var residual2: SharpResidualBlock2D
    @ModuleInfo(key: "conv_out") var convOut: SharpConv2dNCHW

    init(dimDecoder: Int, dimOut: Int, normType: SharpNormLayerName, normNumGroups: Int) {
        self._residual1.wrappedValue = SharpResidualBlock2D(
            dimIn: dimDecoder,
            dimOut: dimDecoder,
            dimHidden: dimDecoder / 2,
            normType: normType,
            normNumGroups: normNumGroups
        )
        self._residual2.wrappedValue = SharpResidualBlock2D(
            dimIn: dimDecoder,
            dimOut: dimDecoder,
            dimHidden: dimDecoder / 2,
            normType: normType,
            normNumGroups: normNumGroups
        )
        self._convOut.wrappedValue = SharpConv2dNCHW(
            inputChannels: dimDecoder,
            outputChannels: dimOut,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = residual1(x)
        y = residual2(y)
        y = relu(y)
        y = convOut(y)
        y = relu(y)
        return y
    }
}

public final class SharpGaussianDensePredictionTransformer: Module, SharpFeatureModel, @unchecked Sendable {
    @ModuleInfo(key: "decoder") var decoder: SharpMultiresConvDecoder
    @ModuleInfo(key: "image_encoder") var imageEncoder: SharpSkipConvBackbone
    @ModuleInfo(key: "fusion") var fusion: SharpFeatureFusionBlock2D
    @ModuleInfo(key: "upsample") var upsample: SharpProjectUpsampleBlock?
    @ModuleInfo(key: "texture_head") var textureHead: SharpGaussianDecoderHead
    @ModuleInfo(key: "geometry_head") var geometryHead: SharpGaussianDecoderHead

    public let dimIn: Int
    public let dimOut: Int
    public let strideOut: Int
    public let imageEncoderType: SharpDPTImageEncoderType
    public let normType: SharpNormLayerName
    public let normNumGroups: Int
    public let useDepthInput: Bool

    public init(
        decoder: SharpMultiresConvDecoder,
        dimIn: Int,
        dimOut: Int,
        strideOut: Int,
        imageEncoderType: SharpDPTImageEncoderType = .skipConv,
        normType: SharpNormLayerName = .groupNorm,
        normNumGroups: Int = 8,
        useDepthInput: Bool = true
    ) {
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.strideOut = strideOut
        self.imageEncoderType = imageEncoderType
        self.normType = normType
        self.normNumGroups = normNumGroups
        self.useDepthInput = useDepthInput

        self._decoder.wrappedValue = decoder

        let imageEncoderDimIn = useDepthInput ? dimIn : (dimIn - 1)
        let imageEncoderDimOut = decoder.dimOut

        switch imageEncoderType {
        case .skipConv:
            let kernel = strideOut == 1 ? 1 : 3
            self._imageEncoder.wrappedValue = SharpSkipConvBackbone(
                dimIn: imageEncoderDimIn,
                dimOut: imageEncoderDimOut,
                kernelSize: kernel,
                strideOut: strideOut
            )
        case .skipConvKernel2:
            self._imageEncoder.wrappedValue = SharpSkipConvBackbone(
                dimIn: imageEncoderDimIn,
                dimOut: imageEncoderDimOut,
                kernelSize: strideOut,
                strideOut: strideOut
            )
        }

        self._fusion.wrappedValue = SharpFeatureFusionBlock2D(dimIn: decoder.dimOut)

        if strideOut == 1 {
            self._upsample.wrappedValue = SharpProjectUpsampleBlock(
                dimIn: decoder.dimOut,
                dimOut: decoder.dimOut,
                upsampleLayers: 1
            )
        } else if strideOut == 2 {
            self._upsample.wrappedValue = nil
        } else {
            fatalError("Only stride_out == 1 or 2 is supported.")
        }

        self._textureHead.wrappedValue = SharpGaussianDecoderHead(
            dimDecoder: decoder.dimOut,
            dimOut: dimOut,
            normType: normType,
            normNumGroups: normNumGroups
        )
        self._geometryHead.wrappedValue = SharpGaussianDecoderHead(
            dimDecoder: decoder.dimOut,
            dimOut: dimOut,
            normType: normType,
            normNumGroups: normNumGroups
        )
        super.init()
    }

    public func callAsFunction(_ inputFeatures: MLXArray, encodings: [MLXArray]) -> SharpImageFeatures {
        var features = decoder(encodings)
        if let upsample {
            features = upsample(features)
        }

        let skipFeatures: MLXArray
        if useDepthInput {
            skipFeatures = imageEncoder(inputFeatures).textureFeatures
        } else {
            skipFeatures = imageEncoder(inputFeatures[0..., 0..<3, 0..., 0...]).textureFeatures
        }

        features = fusion(features, skipFeatures)
        let textureFeatures = textureHead(features)
        let geometryFeatures = geometryHead(features)

        return SharpImageFeatures(
            textureFeatures: textureFeatures,
            geometryFeatures: geometryFeatures
        )
    }

    public var stride: Int { strideOut }
}
