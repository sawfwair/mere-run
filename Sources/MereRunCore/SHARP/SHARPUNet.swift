import MLX
import MLXNN

public final class SharpAvgPool2DNCHW: Module, @unchecked Sendable {
    @ModuleInfo(key: "pool") var pool: AvgPool2d

    public init(kernelSize: Int = 2, stride: Int = 2) {
        self._pool.wrappedValue = AvgPool2d(
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride)
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let nhwc = x.transposed(0, 2, 3, 1)
        let out = pool(nhwc)
        return out.transposed(0, 3, 1, 2)
    }
}

final class SharpUNetEncoderLayer: Module, @unchecked Sendable {
    @ModuleInfo(key: "pool") var pool: SharpAvgPool2DNCHW
    @ModuleInfo(key: "blocks") var blocks: [SharpResidualBlock2D]

    init(
        inputWidth: Int,
        currentWidth: Int,
        normType: SharpNormLayerName,
        normNumGroups: Int,
        blocksPerLayer: Int
    ) {
        self._pool.wrappedValue = SharpAvgPool2DNCHW(kernelSize: 2, stride: 2)

        var residualBlocks: [SharpResidualBlock2D] = []
        residualBlocks.append(SharpResidualBlock2D(
            dimIn: inputWidth,
            dimOut: currentWidth,
            normType: normType,
            normNumGroups: normNumGroups
        ))
        if blocksPerLayer > 1 {
            for _ in 0..<(blocksPerLayer - 1) {
                residualBlocks.append(SharpResidualBlock2D(
                    dimIn: currentWidth,
                    dimOut: currentWidth,
                    normType: normType,
                    normNumGroups: normNumGroups
                ))
            }
        }
        self._blocks.wrappedValue = residualBlocks
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = pool(x)
        for block in blocks {
            y = block(y)
        }
        return y
    }
}

public final class SharpUNetEncoder: Module, @unchecked Sendable {
    public let dimIn: Int
    public let outputDims: [Int]
    public let numSteps: Int

    @ModuleInfo(key: "conv_in") var convIn: SharpConv2dNCHW
    @ModuleInfo(key: "conv_in_norm") var convInNorm: SharpNorm2D
    @ModuleInfo(key: "convs_down") var convsDown: [SharpUNetEncoderLayer]

    public init(
        dimIn: Int,
        width: [Int],
        steps: Int = 6,
        normType: SharpNormLayerName = .groupNorm,
        normNumGroups: Int = 8,
        blocksPerLayer: Int = 2
    ) {
        precondition(blocksPerLayer >= 1, "blocksPerLayer must be greater or equal to one.")
        precondition(width.count == (steps + 1), "width must have steps + 1 entries.")

        self.dimIn = dimIn
        self.outputDims = width
        self.numSteps = steps

        self._convIn.wrappedValue = SharpConv2dNCHW(
            inputChannels: dimIn,
            outputChannels: width[0],
            kernelSize: 3,
            stride: 1,
            padding: 1,
            bias: true
        )
        self._convInNorm.wrappedValue = SharpNorm2D(
            numFeatures: width[0],
            normType: normType,
            numGroups: normNumGroups
        )

        var downLayers: [SharpUNetEncoderLayer] = []
        for i in 0..<steps {
            downLayers.append(SharpUNetEncoderLayer(
                inputWidth: width[i],
                currentWidth: width[i + 1],
                normType: normType,
                normNumGroups: normNumGroups,
                blocksPerLayer: blocksPerLayer
            ))
        }
        self._convsDown.wrappedValue = downLayers
        super.init()
    }

    public convenience init(
        dimIn: Int,
        width: Int,
        steps: Int = 6,
        normType: SharpNormLayerName = .groupNorm,
        normNumGroups: Int = 8,
        blocksPerLayer: Int = 2
    ) {
        let outputDims = (0...steps).map { width << $0 }
        self.init(
            dimIn: dimIn,
            width: outputDims,
            steps: steps,
            normType: normType,
            normNumGroups: normNumGroups,
            blocksPerLayer: blocksPerLayer
        )
    }

    public func callAsFunction(_ input: MLXArray) -> [MLXArray] {
        var features: [MLXArray] = []

        var feat = convIn(input)
        feat = convInNorm(feat)
        feat = relu(feat)
        features.append(feat)

        for convDown in convsDown {
            feat = convDown(feat)
            features.append(feat)
        }
        return features
    }

    public var outWidth: Int {
        outputDims.last ?? 0
    }
}

final class SharpUNetDecoderLayer: Module, @unchecked Sendable {
    @ModuleInfo(key: "upsample") var upsample: SharpUpsample2DNCHW
    @ModuleInfo(key: "blocks") var blocks: [SharpResidualBlock2D]

    init(
        inputWidth: Int,
        currentWidth: Int,
        isFirstLayer: Bool,
        normType: SharpNormLayerName,
        normNumGroups: Int,
        blocksPerLayer: Int
    ) {
        self._upsample.wrappedValue = SharpUpsample2DNCHW(scaleFactor: 2.0, mode: .nearest)

        var residualBlocks: [SharpResidualBlock2D] = []
        residualBlocks.append(SharpResidualBlock2D(
            dimIn: inputWidth * (isFirstLayer ? 1 : 2),
            dimOut: currentWidth,
            normType: normType,
            normNumGroups: normNumGroups
        ))
        if blocksPerLayer > 1 {
            for _ in 0..<(blocksPerLayer - 1) {
                residualBlocks.append(SharpResidualBlock2D(
                    dimIn: currentWidth,
                    dimOut: currentWidth,
                    normType: normType,
                    normNumGroups: normNumGroups
                ))
            }
        }
        self._blocks.wrappedValue = residualBlocks
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = upsample(x)
        for block in blocks {
            y = block(y)
        }
        return y
    }
}

public final class SharpUNetDecoder: Module, @unchecked Sendable {
    public let dimOut: Int
    public let inputDims: [Int]

    @ModuleInfo(key: "convs_up") var convsUp: [SharpUNetDecoderLayer]
    @ModuleInfo(key: "conv_out_norm_1") var convOutNorm1: SharpNorm2D
    @ModuleInfo(key: "conv_out") var convOut: SharpConv2dNCHW
    @ModuleInfo(key: "conv_out_norm_2") var convOutNorm2: SharpNorm2D

    public init(
        dimOut: Int,
        width: [Int],
        steps: Int = 5,
        normType: SharpNormLayerName = .groupNorm,
        normNumGroups: Int = 8,
        blocksPerLayer: Int = 2
    ) {
        precondition(blocksPerLayer >= 1, "blocksPerLayer must be greater or equal to one.")
        precondition(width.count >= (steps + 1), "width must have at least steps + 1 entries.")

        self.dimOut = dimOut
        self.inputDims = Array(width.reversed().prefix(steps + 1))

        var upLayers: [SharpUNetDecoderLayer] = []
        for i in 0..<steps {
            upLayers.append(SharpUNetDecoderLayer(
                inputWidth: inputDims[i],
                currentWidth: inputDims[i + 1],
                isFirstLayer: i == 0,
                normType: normType,
                normNumGroups: normNumGroups,
                blocksPerLayer: blocksPerLayer
            ))
        }
        self._convsUp.wrappedValue = upLayers

        let lastWidth = inputDims.last ?? dimOut
        self._convOutNorm1.wrappedValue = SharpNorm2D(
            numFeatures: lastWidth * 2,
            normType: normType,
            numGroups: normNumGroups
        )
        self._convOut.wrappedValue = SharpConv2dNCHW(
            inputChannels: lastWidth * 2,
            outputChannels: dimOut,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: true
        )
        self._convOutNorm2.wrappedValue = SharpNorm2D(
            numFeatures: dimOut,
            normType: normType,
            numGroups: normNumGroups
        )
        super.init()
    }

    public convenience init(
        dimOut: Int,
        width: Int,
        steps: Int = 5,
        normType: SharpNormLayerName = .groupNorm,
        normNumGroups: Int = 8,
        blocksPerLayer: Int = 2
    ) {
        let inputDims = (0...steps).map { width >> $0 }
        self.init(
            dimOut: dimOut,
            width: inputDims,
            steps: steps,
            normType: normType,
            normNumGroups: normNumGroups,
            blocksPerLayer: blocksPerLayer
        )
    }

    public func callAsFunction(_ features: [MLXArray]) -> MLXArray {
        precondition(!convsUp.isEmpty, "SharpUNetDecoder requires at least one upsampling step.")
        precondition(features.count >= (convsUp.count + 1), "Insufficient encoder features for decoder.")

        var featureIndex = features.count - 1
        var out = convsUp[0](features[featureIndex])
        featureIndex -= 1

        if convsUp.count > 1 {
            for convUp in convsUp.dropFirst() {
                out = MLX.concatenated([out, features[featureIndex]], axis: 1)
                out = convUp(out)
                featureIndex -= 1
            }
        }

        out = MLX.concatenated([out, features[featureIndex]], axis: 1)
        out = convOutNorm1(out)
        out = relu(out)
        out = convOut(out)
        out = convOutNorm2(out)
        out = relu(out)
        return out
    }
}
