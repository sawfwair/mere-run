import Foundation
import MLX
import MLXNN

final class ZImageFeedForward: Module {
  @ModuleInfo(key: "w1") var w1: Linear
  @ModuleInfo(key: "w2") var w2: Linear
  @ModuleInfo(key: "w3") var w3: Linear

  let dim: Int
  let hiddenDim: Int

  init(dim: Int, hiddenDim: Int) {
    self.dim = dim
    self.hiddenDim = hiddenDim
    self._w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
    self._w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
    self._w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    w2(MLXNN.silu(w1(x)) * w3(x))
  }
}
