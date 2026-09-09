import Foundation
import MLX
import MLXFast
import MLXNN

final class Gemma4UnifiedVisionEmbedder: Module {
    @ModuleInfo(key: "patch_ln1") var patchLN1: LayerNorm
    @ModuleInfo(key: "patch_dense") var patchDense: Linear
    @ModuleInfo(key: "patch_ln2") var patchLN2: LayerNorm
    @ParameterInfo(key: "pos_embedding") var positionEmbedding: MLXArray
    @ModuleInfo(key: "pos_norm") var positionNorm: LayerNorm

    private let patchDim: Int

    init(config: Gemma4UnifiedVisionConfig) {
        self.patchDim = config.modelPatchSize * config.modelPatchSize * 3
        self._patchLN1.wrappedValue = LayerNorm(dimensions: patchDim, eps: config.rmsNormEps)
        self._patchDense.wrappedValue = Linear(patchDim, config.mmEmbedDim, bias: true)
        self._patchLN2.wrappedValue = LayerNorm(dimensions: config.mmEmbedDim, eps: config.rmsNormEps)
        self._positionEmbedding.wrappedValue = MLXArray.zeros([config.mmPosembSize, 2, config.mmEmbedDim])
        self._positionNorm.wrappedValue = LayerNorm(dimensions: config.mmEmbedDim, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        pixelValues: MLXArray,
        imagePositionIds: MLXArray?
    ) -> MLXArray {
        var hidden = pixelValues
        if hidden.ndim == 4 && hidden.dim(-1) == patchDim {
            hidden = hidden.reshaped(hidden.dim(0), -1, patchDim)
        }
        hidden = patchLN1(hidden)
        hidden = patchDense(hidden)
        hidden = patchLN2(hidden)

        if let imagePositionIds {
            let xIds = MLX.maximum(imagePositionIds[0..., 0..., 0], MLXArray(Int32(0))).asType(.int32)
            let yIds = MLX.maximum(imagePositionIds[0..., 0..., 1], MLXArray(Int32(0))).asType(.int32)
            let xValid = (imagePositionIds[0..., 0..., 0] .>= MLXArray(Int32(0)))
                .asType(hidden.dtype)
                .expandedDimensions(axis: -1)
            let yValid = (imagePositionIds[0..., 0..., 1] .>= MLXArray(Int32(0)))
                .asType(hidden.dtype)
                .expandedDimensions(axis: -1)
            let xTable = positionEmbedding[0..., 0, 0...]
            let yTable = positionEmbedding[0..., 1, 0...]
            let xPos = take(xTable, xIds, axis: 0)
            let yPos = take(yTable, yIds, axis: 0)
            hidden = hidden + (xPos * xValid + yPos * yValid).asType(hidden.dtype)
        }

        return positionNorm(hidden)
    }
}

final class Gemma4UnifiedMultimodalEmbedder: Module {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear

    private let eps: Float
    private let normWeight: MLXArray

    init(embeddingDim: Int, textHiddenSize: Int, eps: Float) {
        self.eps = eps
        self.normWeight = MLXArray.ones([embeddingDim])
        self._embeddingProjection.wrappedValue = Linear(embeddingDim, textHiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ inputsEmbeds: MLXArray) -> MLXArray {
        embeddingProjection(gemma4RMSNormNoScale(inputsEmbeds, weight: normWeight, eps: eps))
    }
}
