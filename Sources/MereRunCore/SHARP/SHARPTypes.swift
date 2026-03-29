import MLX

public enum SharpActivationType: String, Sendable {
    case linear
    case exp
    case sigmoid
    case softplus
    case reluWithPushback = "relu_with_pushback"
    case hardSigmoidWithPushback = "hard_sigmoid_with_pushback"
}

public enum SharpColorSpace: String, Sendable {
    case sRGB
    case linearRGB
}

public enum SharpColorInitOption: String, Sendable {
    case none
    case firstLayer = "first_layer"
    case allLayers = "all_layers"
}

public enum SharpDepthInitOption: String, Sendable {
    case surfaceMin = "surface_min"
    case surfaceMax = "surface_max"
    case baseDepth = "base_depth"
    case linearDisparity = "linear_disparity"
}

public enum SharpNormLayerName: String, Sendable {
    case noop
    case batchNorm = "batch_norm"
    case groupNorm = "group_norm"
    case instanceNorm = "instance_norm"
}

public enum SharpUpsamplingMode: String, Sendable {
    case transposedConv = "transposed_conv"
    case nearest
    case bilinear
}

public enum SharpDPTImageEncoderType: String, Sendable {
    case skipConv = "skip_conv"
    case skipConvKernel2 = "skip_conv_kernel2"
}

public struct SharpDeltaFactor: Sendable {
    public var xy: Float = 0.001
    public var z: Float = 0.001
    public var color: Float = 0.1
    public var opacity: Float = 1.0
    public var scale: Float = 1.0
    public var quaternion: Float = 1.0

    public init(
        xy: Float = 0.001,
        z: Float = 0.001,
        color: Float = 0.1,
        opacity: Float = 1.0,
        scale: Float = 1.0,
        quaternion: Float = 1.0
    ) {
        self.xy = xy
        self.z = z
        self.color = color
        self.opacity = opacity
        self.scale = scale
        self.quaternion = quaternion
    }
}

public struct SharpInitializerParameters: Sendable {
    public var scaleFactor: Float = 1.0
    public var disparityFactor: Float = 1.0
    public var stride: Int = 2
    public var numLayers: Int = 2
    public var firstLayerDepthOption: SharpDepthInitOption = .surfaceMin
    public var restLayerDepthOption: SharpDepthInitOption = .surfaceMin
    public var colorOption: SharpColorInitOption = .allLayers
    public var baseDepth: Float = 10.0
    public var normalizeDepth: Bool = true
    public var featureInputStopGrad: Bool = false

    public init(
        scaleFactor: Float = 1.0,
        disparityFactor: Float = 1.0,
        stride: Int = 2,
        numLayers: Int = 2,
        firstLayerDepthOption: SharpDepthInitOption = .surfaceMin,
        restLayerDepthOption: SharpDepthInitOption = .surfaceMin,
        colorOption: SharpColorInitOption = .allLayers,
        baseDepth: Float = 10.0,
        normalizeDepth: Bool = true,
        featureInputStopGrad: Bool = false
    ) {
        self.scaleFactor = scaleFactor
        self.disparityFactor = disparityFactor
        self.stride = stride
        self.numLayers = numLayers
        self.firstLayerDepthOption = firstLayerDepthOption
        self.restLayerDepthOption = restLayerDepthOption
        self.colorOption = colorOption
        self.baseDepth = baseDepth
        self.normalizeDepth = normalizeDepth
        self.featureInputStopGrad = featureInputStopGrad
    }
}

public struct SharpComposerParameters: Sendable {
    public var deltaFactor: SharpDeltaFactor = SharpDeltaFactor()
    public var minScale: Float = 0.0
    public var maxScale: Float = 10.0
    public var colorActivationType: SharpActivationType = .sigmoid
    public var opacityActivationType: SharpActivationType = .sigmoid
    public var colorSpace: SharpColorSpace = .linearRGB
    public var scaleFactor: Int = 1
    public var baseScaleOnPredictedMean: Bool = true

    public init(
        deltaFactor: SharpDeltaFactor = SharpDeltaFactor(),
        minScale: Float = 0.0,
        maxScale: Float = 10.0,
        colorActivationType: SharpActivationType = .sigmoid,
        opacityActivationType: SharpActivationType = .sigmoid,
        colorSpace: SharpColorSpace = .linearRGB,
        scaleFactor: Int = 1,
        baseScaleOnPredictedMean: Bool = true
    ) {
        self.deltaFactor = deltaFactor
        self.minScale = minScale
        self.maxScale = maxScale
        self.colorActivationType = colorActivationType
        self.opacityActivationType = opacityActivationType
        self.colorSpace = colorSpace
        self.scaleFactor = scaleFactor
        self.baseScaleOnPredictedMean = baseScaleOnPredictedMean
    }
}

public struct SharpGaussianDecoderParameters: Sendable {
    public var dimIn: Int = 5
    public var dimOut: Int = 32
    public var normType: SharpNormLayerName = .groupNorm
    public var normNumGroups: Int = 8
    public var stride: Int = 2
    public var dimsDecoder: [Int] = [128, 128, 128, 128, 128]
    public var useDepthInput: Bool = true
    public var upsamplingMode: SharpUpsamplingMode = .transposedConv
    public var imageEncoderType: SharpDPTImageEncoderType = .skipConvKernel2

    public init(
        dimIn: Int = 5,
        dimOut: Int = 32,
        normType: SharpNormLayerName = .groupNorm,
        normNumGroups: Int = 8,
        stride: Int = 2,
        dimsDecoder: [Int] = [128, 128, 128, 128, 128],
        useDepthInput: Bool = true,
        upsamplingMode: SharpUpsamplingMode = .transposedConv,
        imageEncoderType: SharpDPTImageEncoderType = .skipConvKernel2
    ) {
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.normType = normType
        self.normNumGroups = normNumGroups
        self.stride = stride
        self.dimsDecoder = dimsDecoder
        self.useDepthInput = useDepthInput
        self.upsamplingMode = upsamplingMode
        self.imageEncoderType = imageEncoderType
    }
}

public struct SharpAlignmentParameters: Sendable {
    public var stride: Int = 1
    public var frozen: Bool = false
    public var steps: Int = 4
    public var activationType: SharpActivationType = .exp
    public var depthDecoderFeatures: Bool = false
    public var baseWidth: Int = 16

    public init(
        stride: Int = 1,
        frozen: Bool = false,
        steps: Int = 4,
        activationType: SharpActivationType = .exp,
        depthDecoderFeatures: Bool = false,
        baseWidth: Int = 16
    ) {
        self.stride = stride
        self.frozen = frozen
        self.steps = steps
        self.activationType = activationType
        self.depthDecoderFeatures = depthDecoderFeatures
        self.baseWidth = baseWidth
    }
}

public struct SharpMonodepthAdaptorParameters: Sendable {
    public var encoderFeatures: Bool = true
    public var decoderFeatures: Bool = false

    public init(encoderFeatures: Bool = true, decoderFeatures: Bool = false) {
        self.encoderFeatures = encoderFeatures
        self.decoderFeatures = decoderFeatures
    }
}

public struct SharpMonodepthParameters: Sendable {
    public var patchEncoderPreset: String = "dinov2l16_384"
    public var imageEncoderPreset: String = "dinov2l16_384"

    public var checkpointURI: String? = nil
    public var unfreezePatchEncoder: Bool = false
    public var unfreezeImageEncoder: Bool = false
    public var unfreezeDecoder: Bool = false
    public var unfreezeHead: Bool = false
    public var unfreezeNormLayers: Bool = false
    public var gradCheckpointing: Bool = false
    public var usePatchOverlap: Bool = true
    public var dimsDecoder: [Int] = [256, 256, 256, 256, 256]

    public init(
        patchEncoderPreset: String = "dinov2l16_384",
        imageEncoderPreset: String = "dinov2l16_384",
        checkpointURI: String? = nil,
        unfreezePatchEncoder: Bool = false,
        unfreezeImageEncoder: Bool = false,
        unfreezeDecoder: Bool = false,
        unfreezeHead: Bool = false,
        unfreezeNormLayers: Bool = false,
        gradCheckpointing: Bool = false,
        usePatchOverlap: Bool = true,
        dimsDecoder: [Int] = [256, 256, 256, 256, 256]
    ) {
        self.patchEncoderPreset = patchEncoderPreset
        self.imageEncoderPreset = imageEncoderPreset
        self.checkpointURI = checkpointURI
        self.unfreezePatchEncoder = unfreezePatchEncoder
        self.unfreezeImageEncoder = unfreezeImageEncoder
        self.unfreezeDecoder = unfreezeDecoder
        self.unfreezeHead = unfreezeHead
        self.unfreezeNormLayers = unfreezeNormLayers
        self.gradCheckpointing = gradCheckpointing
        self.usePatchOverlap = usePatchOverlap
        self.dimsDecoder = dimsDecoder
    }
}

public struct SharpPredictorParameters: Sendable {
    public var initializer: SharpInitializerParameters = SharpInitializerParameters()
    public var monodepth: SharpMonodepthParameters = SharpMonodepthParameters()
    public var monodepthAdaptor: SharpMonodepthAdaptorParameters = SharpMonodepthAdaptorParameters()
    public var gaussianDecoder: SharpGaussianDecoderParameters = SharpGaussianDecoderParameters()
    public var depthAlignment: SharpAlignmentParameters = SharpAlignmentParameters()

    public var deltaFactor: SharpDeltaFactor = SharpDeltaFactor()
    public var maxScale: Float = 10.0
    public var minScale: Float = 0.0
    public var normType: SharpNormLayerName = .groupNorm
    public var normNumGroups: Int = 8
    public var usePredictedMean: Bool = false
    public var colorActivationType: SharpActivationType = .sigmoid
    public var opacityActivationType: SharpActivationType = .sigmoid
    public var colorSpace: SharpColorSpace = .linearRGB
    public var lowPassFilterEps: Float = 1e-2
    public var numMonodepthLayers: Int = 2
    public var sortingMonodepth: Bool = false
    public var baseScaleOnPredictedMean: Bool = true

    public init(
        initializer: SharpInitializerParameters = SharpInitializerParameters(),
        monodepth: SharpMonodepthParameters = SharpMonodepthParameters(),
        monodepthAdaptor: SharpMonodepthAdaptorParameters = SharpMonodepthAdaptorParameters(),
        gaussianDecoder: SharpGaussianDecoderParameters = SharpGaussianDecoderParameters(),
        depthAlignment: SharpAlignmentParameters = SharpAlignmentParameters(),
        deltaFactor: SharpDeltaFactor = SharpDeltaFactor(),
        maxScale: Float = 10.0,
        minScale: Float = 0.0,
        normType: SharpNormLayerName = .groupNorm,
        normNumGroups: Int = 8,
        usePredictedMean: Bool = false,
        colorActivationType: SharpActivationType = .sigmoid,
        opacityActivationType: SharpActivationType = .sigmoid,
        colorSpace: SharpColorSpace = .linearRGB,
        lowPassFilterEps: Float = 1e-2,
        numMonodepthLayers: Int = 2,
        sortingMonodepth: Bool = false,
        baseScaleOnPredictedMean: Bool = true
    ) {
        self.initializer = initializer
        self.monodepth = monodepth
        self.monodepthAdaptor = monodepthAdaptor
        self.gaussianDecoder = gaussianDecoder
        self.depthAlignment = depthAlignment
        self.deltaFactor = deltaFactor
        self.maxScale = maxScale
        self.minScale = minScale
        self.normType = normType
        self.normNumGroups = normNumGroups
        self.usePredictedMean = usePredictedMean
        self.colorActivationType = colorActivationType
        self.opacityActivationType = opacityActivationType
        self.colorSpace = colorSpace
        self.lowPassFilterEps = lowPassFilterEps
        self.numMonodepthLayers = numMonodepthLayers
        self.sortingMonodepth = sortingMonodepth
        self.baseScaleOnPredictedMean = baseScaleOnPredictedMean
    }
}

public struct SharpGaussianBaseValues: @unchecked Sendable {
    public let meanXNDC: MLXArray
    public let meanYNDC: MLXArray
    public let meanInverseZNDC: MLXArray
    public let scales: MLXArray
    public let quaternions: MLXArray
    public let colors: MLXArray
    public let opacities: MLXArray

    public init(
        meanXNDC: MLXArray,
        meanYNDC: MLXArray,
        meanInverseZNDC: MLXArray,
        scales: MLXArray,
        quaternions: MLXArray,
        colors: MLXArray,
        opacities: MLXArray
    ) {
        self.meanXNDC = meanXNDC
        self.meanYNDC = meanYNDC
        self.meanInverseZNDC = meanInverseZNDC
        self.scales = scales
        self.quaternions = quaternions
        self.colors = colors
        self.opacities = opacities
    }
}

public struct SharpInitializerOutput: @unchecked Sendable {
    public let gaussianBaseValues: SharpGaussianBaseValues
    public let featureInput: MLXArray
    public let globalScale: MLXArray?

    public init(gaussianBaseValues: SharpGaussianBaseValues, featureInput: MLXArray, globalScale: MLXArray?) {
        self.gaussianBaseValues = gaussianBaseValues
        self.featureInput = featureInput
        self.globalScale = globalScale
    }
}

public struct SharpGaussians3D: @unchecked Sendable {
    public let meanVectors: MLXArray
    public let singularValues: MLXArray
    public let quaternions: MLXArray
    public let colors: MLXArray
    public let opacities: MLXArray

    public init(
        meanVectors: MLXArray,
        singularValues: MLXArray,
        quaternions: MLXArray,
        colors: MLXArray,
        opacities: MLXArray
    ) {
        self.meanVectors = meanVectors
        self.singularValues = singularValues
        self.quaternions = quaternions
        self.colors = colors
        self.opacities = opacities
    }
}
