import Foundation
import MLX
import MLXNN

final class QwenVisionPatchEmbed: Module {
  let patchSize: Int
  let temporalPatchSize: Int
  let inChannels: Int
  let embedDim: Int

  @ModuleInfo(key: "proj") private var projection: Conv3d

  init(
    patchSize: Int = 14,
    temporalPatchSize: Int = 2,
    inChannels: Int = 3,
    embedDim: Int = 1_280,
    bias: Bool = false
  ) {
    self.patchSize = patchSize
    self.temporalPatchSize = temporalPatchSize
    self.inChannels = inChannels
    self.embedDim = embedDim

    let kernel = IntOrTriple([temporalPatchSize, patchSize, patchSize])
    self._projection.wrappedValue = Conv3d(
      inputChannels: inChannels,
      outputChannels: embedDim,
      kernelSize: kernel,
      stride: kernel,
      padding: .init(0),
      bias: bias
    )

    super.init()
  }

  func callAsFunction(_ patches: MLXArray) -> MLXArray {
    precondition(patches.ndim == 2, "Expected flattened patches of shape [N, patchVolume]")
    let patchVolume = inChannels * temporalPatchSize * patchSize * patchSize
    precondition(
      patches.dim(1) == patchVolume,
      "Patch volume mismatch: expected \(patchVolume), got \(patches.dim(1))"
    )

    let targetType = patches.dtype
    let reshapedWeight = projection.weight.transposed(0, 4, 1, 2, 3)
    let kernel = reshapedWeight.reshaped(embedDim, patchVolume)
    let transposed = kernel.transposed(1, 0)
    var projected = MLX.matmul(patches, transposed)

    // Add bias if present
    if let bias = projection.bias {
      projected = projected + bias
    }

    return projected.asType(targetType)
  }
}
