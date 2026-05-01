import Foundation
import MLX
import MLXNN

final class QwenVisionBlock: Module {
  let blockIndex: Int
  @ModuleInfo(key: "norm1") private var norm1: LayerNorm
  @ModuleInfo(key: "norm2") private var norm2: LayerNorm
  @ModuleInfo(key: "attn") private var attention: QwenVisionAttention
  @ModuleInfo(key: "mlp") private var mlp: QwenVisionMLP

  init(configuration: QwenVisionConfiguration, blockIndex: Int) {
    self.blockIndex = blockIndex
    self._norm1.wrappedValue = LayerNorm(dimensions: configuration.embedDim, eps: configuration.eps, affine: true)
    self._norm2.wrappedValue = LayerNorm(dimensions: configuration.embedDim, eps: configuration.eps, affine: true)
    self._attention.wrappedValue = QwenVisionAttention(embedDim: configuration.embedDim, numHeads: configuration.numHeads)
    let hiddenDim = configuration.mlpHiddenDim
    self._mlp.wrappedValue = QwenVisionMLP(
      dim: configuration.embedDim,
      hiddenDim: hiddenDim,
      activation: configuration.hiddenAct
    )
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    rotaryEmbedding: (cos: MLXArray, sin: MLXArray)? = nil,
    attentionMask: MLXArray? = nil,
    cuSeqlens: MLXArray? = nil
  ) -> MLXArray {
    let norm1Output = norm1(hiddenStates)
    let attentionInput = norm1Output
    let attentionOutput = attention(
      attentionInput,
      rotaryEmbedding: rotaryEmbedding,
      attentionMask: attentionMask,
      cuSeqlens: cuSeqlens
    )
    let postAttention = hiddenStates + attentionOutput

    let norm2Input = postAttention
    let norm2Output = norm2(norm2Input)
    let mlpInput = norm2Output
    let mlpOutput = mlp(mlpInput)
    let blockOutput = postAttention + mlpOutput
    return blockOutput
  }
}
