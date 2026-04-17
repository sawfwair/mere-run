import MLX
import MLXNN

public final class SharpDirectPredictionHead: Module, SharpPredictionHead, @unchecked Sendable {
    @ModuleInfo(key: "geometry_prediction_head") var geometryPredictionHead: Conv2d
    @ModuleInfo(key: "texture_prediction_head") var texturePredictionHead: Conv2d

    public let numLayers: Int

    public init(featureDim: Int, numLayers: Int) {
        self.numLayers = numLayers
        self._geometryPredictionHead.wrappedValue = Conv2d(
            inputChannels: featureDim,
            outputChannels: 3 * numLayers,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
        self._texturePredictionHead.wrappedValue = Conv2d(
            inputChannels: featureDim,
            outputChannels: (14 - 3) * numLayers,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
        super.init()
    }

    public func callAsFunction(_ imageFeatures: SharpImageFeatures) -> MLXArray {
        let deltaGeometry = applyConvNCHW(geometryPredictionHead, to: imageFeatures.geometryFeatures)
        let deltaTexture = applyConvNCHW(texturePredictionHead, to: imageFeatures.textureFeatures)

        let batch = deltaGeometry.dim(0)
        let height = deltaGeometry.dim(2)
        let width = deltaGeometry.dim(3)

        let deltaValuesGeometry = deltaGeometry.reshaped(batch, 3, numLayers, height, width)
        let deltaValuesTexture = deltaTexture.reshaped(batch, 14 - 3, numLayers, height, width)
        return MLX.concatenated([deltaValuesGeometry, deltaValuesTexture], axis: 1)
    }

    private func applyConvNCHW(_ conv: Conv2d, to x: MLXArray) -> MLXArray {
        let nhwc = x.transposed(0, 2, 3, 1)
        let out = conv(nhwc)
        return out.transposed(0, 3, 1, 2)
    }
}
