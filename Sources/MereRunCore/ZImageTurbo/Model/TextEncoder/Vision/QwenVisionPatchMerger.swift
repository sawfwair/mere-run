import Foundation
import MLX
import MLXNN

final class QwenVisionPatchMerger: Module {
  let spatialMergeSize: Int
  let contextDim: Int
  let hiddenDim: Int
  let outputDim: Int
  /// Qwen3-VL deepstack: norm after reshaping to hiddenDim instead of before
  let usePostshuffleNorm: Bool

  private let mergeUnit: Int

  @ModuleInfo(key: "ln_q") private var norm: LayerNorm
  @ModuleInfo(key: "mlp_0") private var mlpInput: Linear
  @ModuleInfo(key: "mlp_2") private var mlpOutput: Linear

  init(contextDim: Int, outputDim: Int, spatialMergeSize: Int, usePostshuffleNorm: Bool = false) {
    self.contextDim = contextDim
    self.outputDim = outputDim
    self.spatialMergeSize = spatialMergeSize
    self.mergeUnit = spatialMergeSize * spatialMergeSize
    self.hiddenDim = contextDim * mergeUnit
    self.usePostshuffleNorm = usePostshuffleNorm

    // For deepstack (postshuffle norm), the norm operates on hiddenDim instead of contextDim
    let normDim = usePostshuffleNorm ? contextDim * mergeUnit : contextDim
    self._norm.wrappedValue = LayerNorm(dimensions: normDim, eps: 1e-6, affine: true)
    self._mlpInput.wrappedValue = Linear(hiddenDim, hiddenDim)
    self._mlpOutput.wrappedValue = Linear(hiddenDim, outputDim)
  }

  func callAsFunction(_ hiddenStates: MLXArray) throws -> MLXArray {
    guard hiddenStates.ndim == 2 else {
      throw VisionTowerError.invalidPatchInputShape(hiddenStates.shape)
    }
    let targetType = hiddenStates.dtype
    let features = hiddenStates.dim(1)

    if usePostshuffleNorm {
      // Deepstack path: reshape first, then norm on hiddenDim
      let tokens = hiddenStates.dim(0)
      guard tokens % mergeUnit == 0 else {
        throw VisionTowerError.invalidMergedTokenCount(tokens: tokens, mergeUnit: mergeUnit)
      }
      var reshaped = hiddenStates.reshaped(tokens / mergeUnit, hiddenDim)
      reshaped = norm(reshaped)
      var merged = mlpInput(reshaped)
      merged = MLXNN.gelu(merged)
      merged = mlpOutput(merged)
      return merged.asType(targetType)
    } else if features == contextDim {
      let tokens = hiddenStates.dim(0)
      guard tokens % mergeUnit == 0 else {
        throw VisionTowerError.invalidMergedTokenCount(tokens: tokens, mergeUnit: mergeUnit)
      }
      var normed = norm(hiddenStates)
      normed = normed.reshaped(tokens / mergeUnit, hiddenDim)
      var merged = mlpInput(normed)
      merged = MLXNN.gelu(merged)
      merged = mlpOutput(merged)
      return merged.asType(targetType)
    } else if features == hiddenDim {
      let windows = hiddenStates.dim(0)
      var reshaped = hiddenStates.reshaped(windows * mergeUnit, contextDim)
      reshaped = norm(reshaped)
      reshaped = reshaped.reshaped(windows, hiddenDim)
      var merged = mlpInput(reshaped)
      merged = MLXNN.gelu(merged)
      merged = mlpOutput(merged)
      return merged.asType(targetType)
    } else {
      throw VisionTowerError.unexpectedMergedFeatureDimension(actual: features, expected: [contextDim, hiddenDim])
    }
  }
}
