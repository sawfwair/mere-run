import Foundation
import MLX
import MLXNN

final class QwenVisionMLP: Module {
  @ModuleInfo(key: "fc1") private var fc1: Linear?
  @ModuleInfo(key: "fc2") private var fc2: Linear?
  @ModuleInfo(key: "gate_proj") private var gateProj: Linear?
  @ModuleInfo(key: "up_proj") private var upProj: Linear?
  @ModuleInfo(key: "down_proj") private var downProj: Linear?

  private let activation: QwenVisionConfiguration.Activation
  private let style: QwenVisionConfiguration.MLPStyle

  init(
    dim: Int,
    hiddenDim: Int,
    activation: QwenVisionConfiguration.Activation,
    style: QwenVisionConfiguration.MLPStyle
  ) {
    self.activation = activation
    self.style = style
    switch style {
    case .twoLayer:
      self._fc1.wrappedValue = Linear(dim, hiddenDim)
      self._fc2.wrappedValue = Linear(hiddenDim, dim)
    case .gated:
      self._gateProj.wrappedValue = Linear(dim, hiddenDim)
      self._upProj.wrappedValue = Linear(dim, hiddenDim)
      self._downProj.wrappedValue = Linear(hiddenDim, dim)
    }
  }

  func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    switch style {
    case .twoLayer:
      guard let fc1, let fc2 else {
        preconditionFailure("Qwen vision two-layer MLP is incomplete")
      }
      var hidden = fc1(hiddenStates)
      switch activation {
      case .geluApproximate:
        hidden = MLXNN.geluApproximate(hidden)
      case .silu:
        hidden = MLXNN.silu(hidden)
      }
      return fc2(hidden)
    case .gated:
      guard let gateProj, let upProj, let downProj else {
        preconditionFailure("Qwen vision gated MLP is incomplete")
      }
      return downProj(MLXNN.silu(gateProj(hiddenStates)) * upProj(hiddenStates))
    }
  }

}
