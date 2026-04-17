import MLX
import MLXNN

final class SharpConvProjector: Module, @unchecked Sendable {
    @ModuleInfo(key: "conv") var conv: SharpConv2dNCHW?

    init(conv: SharpConv2dNCHW?) {
        self._conv.wrappedValue = conv
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv?(x) ?? x
    }
}

public final class SharpMultiresConvDecoder: Module, @unchecked Sendable {
    public let dimsEncoder: [Int]
    public let dimsDecoder: [Int]
    public let dimOut: Int
    public let upsamplingMode: SharpUpsamplingMode

    @ModuleInfo(key: "convs") var convs: [SharpConvProjector]
    @ModuleInfo(key: "fusions") var fusions: [SharpFeatureFusionBlock2D]

    public init(
        dimsEncoder: [Int],
        dimsDecoder: [Int],
        upsamplingMode: SharpUpsamplingMode = .transposedConv
    ) {
        precondition(dimsEncoder.count == dimsDecoder.count, "dimsEncoder and dimsDecoder must match in size.")
        self.dimsEncoder = dimsEncoder
        self.dimsDecoder = dimsDecoder
        self.dimOut = dimsDecoder[0]
        self.upsamplingMode = upsamplingMode

        let numEncoders = dimsEncoder.count

        var convList: [SharpConvProjector] = []
        if dimsEncoder[0] != dimsDecoder[0] {
            convList.append(SharpConvProjector(conv: SharpConv2dNCHW(
                inputChannels: dimsEncoder[0],
                outputChannels: dimsDecoder[0],
                kernelSize: 1,
                stride: 1,
                padding: 0,
                bias: false
            )))
        } else {
            convList.append(SharpConvProjector(conv: nil))
        }

        if numEncoders > 1 {
            for i in 1..<numEncoders {
                convList.append(SharpConvProjector(conv: SharpConv2dNCHW(
                    inputChannels: dimsEncoder[i],
                    outputChannels: dimsDecoder[i],
                    kernelSize: 3,
                    stride: 1,
                    padding: 1,
                    bias: false
                )))
            }
        }
        self._convs.wrappedValue = convList

        var fusionList: [SharpFeatureFusionBlock2D] = []
        for i in 0..<numEncoders {
            let outDim = (i != 0) ? dimsDecoder[i - 1] : dimOut
            let mode: SharpUpsamplingMode? = (i != 0) ? upsamplingMode : nil
            fusionList.append(SharpFeatureFusionBlock2D(
                dimIn: dimsDecoder[i],
                dimOut: outDim,
                upsamplingMode: mode,
                batchNorm: false
            ))
        }
        self._fusions.wrappedValue = fusionList
        super.init()
    }

    public convenience init(
        dimsEncoder: [Int],
        decoderDim: Int,
        upsamplingMode: SharpUpsamplingMode = .transposedConv
    ) {
        self.init(
            dimsEncoder: dimsEncoder,
            dimsDecoder: Array(repeating: decoderDim, count: dimsEncoder.count),
            upsamplingMode: upsamplingMode
        )
    }

    public func callAsFunction(_ encodings: [MLXArray]) -> MLXArray {
        precondition(encodings.count == dimsEncoder.count, "Unexpected number of encoder levels.")
        let numLevels = encodings.count

        var features = convs[numLevels - 1](encodings[numLevels - 1])
        features = fusions[numLevels - 1](features, nil)

        if numLevels > 1 {
            for i in stride(from: numLevels - 2, through: 0, by: -1) {
                let featuresI = convs[i](encodings[i])
                features = fusions[i](features, featuresI)
            }
        }
        return features
    }
}
