import MLX

public struct SharpMonodepthPredictorOutput: @unchecked Sendable {
    public let disparity: MLXArray
    public let encoderFeatures: [MLXArray]
    public let decoderFeatures: MLXArray
    public let intermediateFeatures: [MLXArray]

    public init(
        disparity: MLXArray,
        encoderFeatures: [MLXArray],
        decoderFeatures: MLXArray,
        intermediateFeatures: [MLXArray] = []
    ) {
        self.disparity = disparity
        self.encoderFeatures = encoderFeatures
        self.decoderFeatures = decoderFeatures
        self.intermediateFeatures = intermediateFeatures
    }
}

public protocol SharpMonodepthPredictorCore: Sendable {
    func callAsFunction(_ image: MLXArray) -> SharpMonodepthPredictorOutput
    func internalResolution() -> Int
    func encoderFeatureDims() -> [Int]
    func decoderFeatureDim() -> Int
}

public final class SharpMonodepthWithEncodingAdaptor: SharpMonodepthModel {
    public let monodepthPredictor: any SharpMonodepthPredictorCore
    public let returnEncoderFeatures: Bool
    public let returnDecoderFeatures: Bool
    public let numMonodepthLayers: Int
    public let sortingMonodepth: Bool

    public init(
        monodepthPredictor: any SharpMonodepthPredictorCore,
        returnEncoderFeatures: Bool,
        returnDecoderFeatures: Bool,
        numMonodepthLayers: Int,
        sortingMonodepth: Bool
    ) {
        self.monodepthPredictor = monodepthPredictor
        self.returnEncoderFeatures = returnEncoderFeatures
        self.returnDecoderFeatures = returnDecoderFeatures
        self.numMonodepthLayers = numMonodepthLayers
        self.sortingMonodepth = sortingMonodepth
    }

    public func callAsFunction(_ image: MLXArray) -> SharpMonodepthOutput {
        let predictorOutput = monodepthPredictor(image)
        var disparity = predictorOutput.disparity

        if numMonodepthLayers == 2 && sortingMonodepth {
            let firstLayerDisparity = disparity.max(axis: 1, keepDims: true)
            let secondLayerDisparity = disparity.min(axis: 1, keepDims: true)
            disparity = MLX.concatenated([firstLayerDisparity, secondLayerDisparity], axis: 1)
        }

        var outputFeatures: [MLXArray] = []
        if returnEncoderFeatures {
            outputFeatures.append(contentsOf: predictorOutput.encoderFeatures)
        }
        if returnDecoderFeatures {
            outputFeatures.append(predictorOutput.decoderFeatures)
        }

        return SharpMonodepthOutput(
            disparity: disparity,
            outputFeatures: outputFeatures,
            decoderFeatures: predictorOutput.decoderFeatures
        )
    }

    public func getFeatureDims() -> [Int] {
        var dims: [Int] = []
        if returnEncoderFeatures {
            dims.append(contentsOf: monodepthPredictor.encoderFeatureDims())
        }
        if returnDecoderFeatures {
            dims.append(monodepthPredictor.decoderFeatureDim())
        }
        return dims
    }

    public func internalResolution() -> Int {
        monodepthPredictor.internalResolution()
    }
}

public func createSharpMonodepthAdaptor(
    monodepthPredictor: any SharpMonodepthPredictorCore,
    params: SharpMonodepthAdaptorParameters,
    numMonodepthLayers: Int,
    sortingMonodepth: Bool
) -> SharpMonodepthWithEncodingAdaptor {
    SharpMonodepthWithEncodingAdaptor(
        monodepthPredictor: monodepthPredictor,
        returnEncoderFeatures: params.encoderFeatures,
        returnDecoderFeatures: params.decoderFeatures,
        numMonodepthLayers: numMonodepthLayers,
        sortingMonodepth: sortingMonodepth
    )
}
