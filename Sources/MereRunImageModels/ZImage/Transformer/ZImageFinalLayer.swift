import Foundation
import MLX
import MLXNN

final class ZImageFinalLayer: Module {
  let hiddenSize: Int
  let outChannels: Int

  @ModuleInfo(key: "norm_final") var normFinal: LayerNorm
  @ModuleInfo(key: "linear") var linear: Linear
  @ModuleInfo(key: "adaLN_modulation") var adaLN: (SiLUModule, Linear)

  init(hiddenSize: Int, outChannels: Int) {
    self.hiddenSize = hiddenSize
    self.outChannels = outChannels

    self._normFinal.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6, affine: false)
    self._linear.wrappedValue = Linear(hiddenSize, outChannels, bias: true)
    self._adaLN.wrappedValue = (SiLUModule(), Linear(min(hiddenSize, 256), hiddenSize, bias: true))
    super.init()
  }

  func callAsFunction(_ x: MLXArray, conditioning c: MLXArray) -> MLXArray {
    let scale = (1 + adaLN.1(adaLN.0(c))).expandedDimensions(axis: 1)
    return linear(normFinal(x) * scale)
  }
}
