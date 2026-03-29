import Foundation
import MLX
import MLXNN

final class QwenVisionMLP: Module {
  @ModuleInfo(key: "fc1") private var fc1: Linear
  @ModuleInfo(key: "fc2") private var fc2: Linear

  private let activation: QwenVisionConfiguration.Activation

  init(dim: Int, hiddenDim: Int, activation: QwenVisionConfiguration.Activation) {
    self.activation = activation
    self._fc1.wrappedValue = Linear(dim, hiddenDim)
    self._fc2.wrappedValue = Linear(hiddenDim, dim)
  }

  func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    var hidden = fc1(hiddenStates)
    switch activation {
    case .geluApproximate:
      // Use tanh-approximated GELU to match mlx-vlm's nn.GELU(approx="tanh")
      hidden = MLXNN.geluApproximate(hidden)
    case .silu:
      hidden = MLXNN.silu(hidden)
    }
    return fc2(hidden)
  }

}
