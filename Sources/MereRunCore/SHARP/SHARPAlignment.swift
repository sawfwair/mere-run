import Foundation
import MLX
import MLXNN

public func createSharpAlignment(
    params: SharpAlignmentParameters,
    depthDecoderDim: Int
) -> SharpLearnedAlignment {
    return SharpLearnedAlignment(
        steps: params.steps,
        stride: params.stride,
        baseWidth: params.baseWidth,
        depthDecoderFeatures: params.depthDecoderFeatures,
        depthDecoderDim: depthDecoderDim,
        activationType: params.activationType
    )
}

public final class SharpLearnedAlignment: Module, SharpScaleMapEstimator, @unchecked Sendable {
    @ModuleInfo(key: "encoder") var encoder: SharpUNetEncoder
    @ModuleInfo(key: "decoder") var decoder: SharpUNetDecoder
    @ModuleInfo(key: "conv_out") var convOut: SharpConv2dNCHW

    public let steps: Int
    public let stride: Int
    public let baseWidth: Int
    public let depthDecoderFeatures: Bool
    public let depthDecoderDim: Int
    public let activationType: SharpActivationType

    public init(
        steps: Int = 4,
        stride: Int = 8,
        baseWidth: Int = 16,
        depthDecoderFeatures: Bool = false,
        depthDecoderDim: Int = 256,
        activationType: SharpActivationType = .exp
    ) {
        precondition(SharpLearnedAlignment.isPowerOfTwo(stride), "stride must be a power of two.")

        let stepsDecoder = steps - Int(log2(Double(stride)))
        precondition(stepsDecoder >= 1, "steps - log2(stride) must be >= 1.")

        self.steps = steps
        self.stride = stride
        self.baseWidth = baseWidth
        self.depthDecoderFeatures = depthDecoderFeatures
        self.depthDecoderDim = depthDecoderDim
        self.activationType = activationType

        let dimIn = depthDecoderFeatures ? (2 + depthDecoderDim) : 2
        let widths = (0...steps).map { min(baseWidth << $0, 1024) }

        self._encoder.wrappedValue = SharpUNetEncoder(
            dimIn: dimIn,
            width: widths,
            steps: steps,
            normType: .groupNorm,
            normNumGroups: 4,
            blocksPerLayer: 2
        )
        self._decoder.wrappedValue = SharpUNetDecoder(
            dimOut: widths[0],
            width: widths,
            steps: stepsDecoder,
            normType: .groupNorm,
            normNumGroups: 4,
            blocksPerLayer: 2
        )
        self._convOut.wrappedValue = SharpConv2dNCHW(
            inputChannels: widths[0],
            outputChannels: 1,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: true
        )
        super.init()
    }

    public func callAsFunction(
        monodepth tensorSrc: MLXArray,
        depth tensorTgt: MLXArray,
        depthDecoderFeatures: MLXArray?
    ) -> MLXArray {
        let src = 1.0 / MLX.clip(tensorSrc, min: 1e-4, max: 1e4)
        let tgt = 1.0 / MLX.clip(tensorTgt, min: 1e-4, max: 1e4)

        var tensorInput = MLX.concatenated([src, tgt], axis: 1)
        if self.depthDecoderFeatures {
            guard let depthDecoderFeatures else {
                fatalError("depthDecoderFeatures are required when depthDecoderFeatures=true.")
            }
            let resizedFeatures = Self.resizeBilinearNCHW(
                depthDecoderFeatures,
                height: src.dim(2),
                width: src.dim(3)
            )
            tensorInput = MLX.concatenated([tensorInput, resizedFeatures], axis: 1)
        }

        let features = encoder(tensorInput)
        var output = decoder(features)
        output = convOut(output)
        let lowresMap = SharpMath.activationForward(activationType, output)

        if lowresMap.dim(2) == src.dim(2) && lowresMap.dim(3) == src.dim(3) {
            return lowresMap
        }

        return Self.resizeBilinearNCHW(lowresMap, height: src.dim(2), width: src.dim(3))
    }

    private static func isPowerOfTwo(_ n: Int) -> Bool {
        n > 0 && (n & (n - 1)) == 0
    }

    private static func resizeBilinearNCHW(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let currentHeight = x.dim(2)
        let currentWidth = x.dim(3)
        if currentHeight == height && currentWidth == width {
            return x
        }

        let hScale = Float(height) / Float(currentHeight)
        let wScale = Float(width) / Float(currentWidth)
        let nhwc = x.transposed(0, 2, 3, 1)
        let resized = Upsample(
            scaleFactor: .array([hScale, wScale]),
            mode: .linear(alignCorners: false)
        )(nhwc)
        return resized.transposed(0, 3, 1, 2)
    }
}
