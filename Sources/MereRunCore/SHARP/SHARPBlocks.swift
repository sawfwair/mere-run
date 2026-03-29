import MLX
import MLXNN

public final class SharpConv2dNCHW: Module, @unchecked Sendable {
    @ModuleInfo(key: "conv") var conv: Conv2d

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(padding),
            dilation: IntOrPair(dilation),
            bias: bias
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let nhwc = x.transposed(0, 2, 3, 1)
        let out = conv(nhwc)
        return out.transposed(0, 3, 1, 2)
    }
}

public final class SharpConvTransposed2dNCHW: Module, @unchecked Sendable {
    @ModuleInfo(key: "conv_transposed") var convTransposed: ConvTransposed2d

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self._convTransposed.wrappedValue = ConvTransposed2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(padding),
            dilation: IntOrPair(dilation),
            bias: bias
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let nhwc = x.transposed(0, 2, 3, 1)
        let out = convTransposed(nhwc)
        return out.transposed(0, 3, 1, 2)
    }
}

public final class SharpUpsample2DNCHW: Module, @unchecked Sendable {
    @ModuleInfo(key: "upsample") var upsample: Upsample

    public init(scaleFactor: Float = 2.0, mode: SharpUpsamplingMode) {
        let upsampleMode: Upsample.Mode
        switch mode {
        case .nearest:
            upsampleMode = .nearest
        case .bilinear:
            upsampleMode = .linear(alignCorners: false)
        case .transposedConv:
            fatalError("SharpUpsample2DNCHW only supports nearest/bilinear.")
        }
        self._upsample.wrappedValue = Upsample(scaleFactor: FloatOrArray(scaleFactor), mode: upsampleMode)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let nhwc = x.transposed(0, 2, 3, 1)
        let out = upsample(nhwc)
        return out.transposed(0, 3, 1, 2)
    }
}

public final class SharpNorm2D: Module, @unchecked Sendable {
    public let normType: SharpNormLayerName

    @ModuleInfo(key: "batch_norm") var batchNorm: BatchNorm?
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm?
    @ModuleInfo(key: "instance_norm") var instanceNorm: InstanceNorm?

    public init(numFeatures: Int, normType: SharpNormLayerName, numGroups: Int = 8) {
        self.normType = normType
        switch normType {
        case .noop:
            self._batchNorm.wrappedValue = nil
            self._groupNorm.wrappedValue = nil
            self._instanceNorm.wrappedValue = nil
        case .batchNorm:
            self._batchNorm.wrappedValue = BatchNorm(featureCount: numFeatures, affine: true)
            self._groupNorm.wrappedValue = nil
            self._instanceNorm.wrappedValue = nil
        case .groupNorm:
            self._batchNorm.wrappedValue = nil
            self._groupNorm.wrappedValue = GroupNorm(
                groupCount: numGroups,
                dimensions: numFeatures,
                affine: true,
                pytorchCompatible: true
            )
            self._instanceNorm.wrappedValue = nil
        case .instanceNorm:
            self._batchNorm.wrappedValue = nil
            self._groupNorm.wrappedValue = nil
            self._instanceNorm.wrappedValue = InstanceNorm(dimensions: numFeatures, affine: false)
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch normType {
        case .noop:
            return x
        case .batchNorm:
            guard let batchNorm else { return x }
            let nhwc = x.transposed(0, 2, 3, 1)
            let out = batchNorm(nhwc)
            return out.transposed(0, 3, 1, 2)
        case .groupNorm:
            guard let groupNorm else { return x }
            let nhwc = x.transposed(0, 2, 3, 1)
            let out = groupNorm(nhwc)
            return out.transposed(0, 3, 1, 2)
        case .instanceNorm:
            guard let instanceNorm else { return x }
            let nhwc = x.transposed(0, 2, 3, 1)
            let out = instanceNorm(nhwc)
            return out.transposed(0, 3, 1, 2)
        }
    }
}

public final class SharpResidualBlock2D: Module, @unchecked Sendable {
    @ModuleInfo(key: "norm_1") var norm1: SharpNorm2D
    @ModuleInfo(key: "conv_1") var conv1: SharpConv2dNCHW
    @ModuleInfo(key: "norm_2") var norm2: SharpNorm2D
    @ModuleInfo(key: "conv_2") var conv2: SharpConv2dNCHW
    @ModuleInfo(key: "shortcut") var shortcut: SharpConv2dNCHW?

    public init(
        dimIn: Int,
        dimOut: Int,
        dimHidden: Int? = nil,
        normType: SharpNormLayerName = .noop,
        normNumGroups: Int = 8,
        dilation: Int = 1,
        kernelSize: Int = 3,
        convBias: Bool = true
    ) {
        let hidden = dimHidden ?? (dimOut / 2)
        let padding = (dilation * (kernelSize - 1)) / 2

        self._norm1.wrappedValue = SharpNorm2D(numFeatures: dimIn, normType: normType, numGroups: normNumGroups)
        self._conv1.wrappedValue = SharpConv2dNCHW(
            inputChannels: dimIn,
            outputChannels: hidden,
            kernelSize: kernelSize,
            stride: 1,
            padding: padding,
            dilation: dilation,
            bias: convBias
        )
        self._norm2.wrappedValue = SharpNorm2D(numFeatures: hidden, normType: normType, numGroups: normNumGroups)
        self._conv2.wrappedValue = SharpConv2dNCHW(
            inputChannels: hidden,
            outputChannels: dimOut,
            kernelSize: kernelSize,
            stride: 1,
            padding: padding,
            dilation: dilation,
            bias: convBias
        )

        if dimIn != dimOut {
            self._shortcut.wrappedValue = SharpConv2dNCHW(
                inputChannels: dimIn,
                outputChannels: dimOut,
                kernelSize: 1,
                stride: 1,
                padding: 0,
                bias: true
            )
        } else {
            self._shortcut.wrappedValue = nil
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = norm1(x)
        y = relu(y)
        y = conv1(y)
        y = norm2(y)
        y = relu(y)
        y = conv2(y)
        let shortcutX = shortcut?(x) ?? x
        return shortcutX + y
    }
}

final class SharpFeatureFusionResidual: Module, @unchecked Sendable {
    @ModuleInfo(key: "conv_1") var conv1: SharpConv2dNCHW
    @ModuleInfo(key: "bn_1") var bn1: BatchNorm?
    @ModuleInfo(key: "conv_2") var conv2: SharpConv2dNCHW
    @ModuleInfo(key: "bn_2") var bn2: BatchNorm?

    init(numFeatures: Int, batchNorm: Bool) {
        self._conv1.wrappedValue = SharpConv2dNCHW(
            inputChannels: numFeatures,
            outputChannels: numFeatures,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            bias: !batchNorm
        )
        self._conv2.wrappedValue = SharpConv2dNCHW(
            inputChannels: numFeatures,
            outputChannels: numFeatures,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            bias: !batchNorm
        )

        if batchNorm {
            self._bn1.wrappedValue = BatchNorm(featureCount: numFeatures, affine: true)
            self._bn2.wrappedValue = BatchNorm(featureCount: numFeatures, affine: true)
        } else {
            self._bn1.wrappedValue = nil
            self._bn2.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = relu(x)
        y = conv1(y)
        if let bn1 {
            y = applyBatchNormNCHW(bn1, y)
        }
        y = relu(y)
        y = conv2(y)
        if let bn2 {
            y = applyBatchNormNCHW(bn2, y)
        }
        return x + y
    }

    private func applyBatchNormNCHW(_ bn: BatchNorm, _ x: MLXArray) -> MLXArray {
        let nhwc = x.transposed(0, 2, 3, 1)
        let out = bn(nhwc)
        return out.transposed(0, 3, 1, 2)
    }
}

final class SharpFeatureFusionDeconv: Module, @unchecked Sendable {
    @ModuleInfo(key: "transposed") var transposed: SharpConvTransposed2dNCHW?
    @ModuleInfo(key: "upsample") var upsample: SharpUpsample2DNCHW?

    init(dimIn: Int, mode: SharpUpsamplingMode?) {
        switch mode {
        case .none:
            self._transposed.wrappedValue = nil
            self._upsample.wrappedValue = nil
        case .transposedConv:
            self._transposed.wrappedValue = SharpConvTransposed2dNCHW(
                inputChannels: dimIn,
                outputChannels: dimIn,
                kernelSize: 2,
                stride: 2,
                padding: 0,
                bias: false
            )
            self._upsample.wrappedValue = nil
        case .nearest, .bilinear:
            self._transposed.wrappedValue = nil
            self._upsample.wrappedValue = SharpUpsample2DNCHW(scaleFactor: 2.0, mode: mode!)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let transposed {
            return transposed(x)
        }
        if let upsample {
            return upsample(x)
        }
        return x
    }
}

public final class SharpFeatureFusionBlock2D: Module, @unchecked Sendable {
    @ModuleInfo(key: "resnet1") var resnet1: SharpFeatureFusionResidual
    @ModuleInfo(key: "resnet2") var resnet2: SharpFeatureFusionResidual
    @ModuleInfo(key: "deconv") var deconv: SharpFeatureFusionDeconv
    @ModuleInfo(key: "out_conv") var outConv: SharpConv2dNCHW

    public init(
        dimIn: Int,
        dimOut: Int? = nil,
        upsamplingMode: SharpUpsamplingMode? = nil,
        batchNorm: Bool = false
    ) {
        let outputDim = dimOut ?? dimIn
        self._resnet1.wrappedValue = SharpFeatureFusionResidual(numFeatures: dimIn, batchNorm: batchNorm)
        self._resnet2.wrappedValue = SharpFeatureFusionResidual(numFeatures: dimIn, batchNorm: batchNorm)
        self._deconv.wrappedValue = SharpFeatureFusionDeconv(dimIn: dimIn, mode: upsamplingMode)
        self._outConv.wrappedValue = SharpConv2dNCHW(
            inputChannels: dimIn,
            outputChannels: outputDim,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: true
        )
        super.init()
    }

    public func callAsFunction(_ x0: MLXArray, _ x1: MLXArray? = nil) -> MLXArray {
        var x = x0
        if let x1 {
            let res = resnet1(x1)
            x = x + res
        }
        x = resnet2(x)
        x = deconv(x)
        x = outConv(x)
        return x
    }
}
