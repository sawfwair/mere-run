import MLX

public struct SharpImageFeatures: @unchecked Sendable {
    public let textureFeatures: MLXArray
    public let geometryFeatures: MLXArray

    public init(textureFeatures: MLXArray, geometryFeatures: MLXArray) {
        self.textureFeatures = textureFeatures
        self.geometryFeatures = geometryFeatures
    }
}

public struct SharpMonodepthOutput: @unchecked Sendable {
    public let disparity: MLXArray
    public let outputFeatures: [MLXArray]
    public let decoderFeatures: MLXArray?

    public init(disparity: MLXArray, outputFeatures: [MLXArray], decoderFeatures: MLXArray? = nil) {
        self.disparity = disparity
        self.outputFeatures = outputFeatures
        self.decoderFeatures = decoderFeatures
    }
}

public protocol SharpMonodepthModel: Sendable {
    func callAsFunction(_ image: MLXArray) -> SharpMonodepthOutput
    func internalResolution() -> Int
}

public protocol SharpFeatureModel: Sendable {
    func callAsFunction(_ featureInput: MLXArray, encodings: [MLXArray]) -> SharpImageFeatures
}

public protocol SharpPredictionHead: Sendable {
    func callAsFunction(_ imageFeatures: SharpImageFeatures) -> MLXArray
}

public protocol SharpScaleMapEstimator: Sendable {
    func callAsFunction(monodepth: MLXArray, depth: MLXArray, depthDecoderFeatures: MLXArray?) -> MLXArray
}

public final class SharpDepthAlignment {
    private let scaleMapEstimator: (any SharpScaleMapEstimator)?

    public init(scaleMapEstimator: (any SharpScaleMapEstimator)? = nil) {
        self.scaleMapEstimator = scaleMapEstimator
    }

    public func callAsFunction(
        monodepth: MLXArray,
        depth: MLXArray?,
        depthDecoderFeatures: MLXArray? = nil
    ) -> (alignedMonodepth: MLXArray, alignmentMap: MLXArray) {
        if let depth, let estimator = scaleMapEstimator {
            let map = estimator(
                monodepth: monodepth[0..., 0..<1, 0..., 0...],
                depth: depth,
                depthDecoderFeatures: depthDecoderFeatures
            )
            return (map * monodepth, map)
        }

        let ones = MLX.full(monodepth.shape, values: MLXArray(1.0).asType(monodepth.dtype), dtype: monodepth.dtype)
        return (monodepth, ones)
    }
}

public final class SharpRGBGaussianPredictor {
    public let initializer: SharpInitializer
    public let monodepthModel: any SharpMonodepthModel
    public let featureModel: any SharpFeatureModel
    public let predictionHead: any SharpPredictionHead
    public let gaussianComposer: SharpGaussianComposer

    private let depthAlignment: SharpDepthAlignment

    public init(
        initializer: SharpInitializer,
        monodepthModel: any SharpMonodepthModel,
        featureModel: any SharpFeatureModel,
        predictionHead: any SharpPredictionHead,
        gaussianComposer: SharpGaussianComposer,
        scaleMapEstimator: (any SharpScaleMapEstimator)? = nil
    ) {
        self.initializer = initializer
        self.monodepthModel = monodepthModel
        self.featureModel = featureModel
        self.predictionHead = predictionHead
        self.gaussianComposer = gaussianComposer
        self.depthAlignment = SharpDepthAlignment(scaleMapEstimator: scaleMapEstimator)
    }

    public func callAsFunction(
        image: MLXArray,
        disparityFactor: MLXArray,
        depth: MLXArray? = nil
    ) -> SharpGaussians3D {
        let monodepthOutput = monodepthModel(image)
        let monodepthDisparity = monodepthOutput.disparity

        let disparityFactorExpanded = disparityFactor.reshaped(disparityFactor.dim(0), 1, 1, 1)
        let safeDisparity = MLX.clip(monodepthDisparity, min: 1e-4, max: 1e4)
        let monodepth = disparityFactorExpanded / safeDisparity

        let aligned = depthAlignment(
            monodepth: monodepth,
            depth: depth,
            depthDecoderFeatures: monodepthOutput.decoderFeatures
        )

        let initOutput = initializer(image: image, depth: aligned.alignedMonodepth)
        let imageFeatures = featureModel(initOutput.featureInput, encodings: monodepthOutput.outputFeatures)
        let deltaValues = predictionHead(imageFeatures)

        return gaussianComposer(
            delta: deltaValues,
            baseValues: initOutput.gaussianBaseValues,
            globalScale: initOutput.globalScale
        )
    }

    public func internalResolution() -> Int {
        monodepthModel.internalResolution()
    }
}
