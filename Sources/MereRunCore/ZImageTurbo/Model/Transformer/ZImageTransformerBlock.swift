import Foundation
import MLX
import MLXNN

public final class ZImageTransformerBlock: Module {
  let dim: Int
  let nHeads: Int
  let nKvHeads: Int
  let modulation: Bool

  @ModuleInfo(key: "attention") var attention: ZImageSelfAttention
  @ModuleInfo(key: "adaLN_modulation") var adaLN: [Linear]?
  @ModuleInfo(key: "attention_norm1") var attentionNorm1: RMSNorm
  @ModuleInfo(key: "ffn_norm1") var ffnNorm1: RMSNorm
  @ModuleInfo(key: "attention_norm2") var attentionNorm2: RMSNorm
  @ModuleInfo(key: "ffn_norm2") var ffnNorm2: RMSNorm
  @ModuleInfo(key: "feed_forward") var feedForward: ZImageFeedForward

  init(
    layerId: Int,
    dim: Int,
    nHeads: Int,
    nKvHeads: Int,
    normEps: Float,
    qkNorm: Bool,
    modulation: Bool
  ) {
    self.dim = dim
    self.nHeads = nHeads
    self.nKvHeads = nKvHeads
    self.modulation = modulation

    self._attention.wrappedValue = ZImageSelfAttention(dim: dim, heads: nHeads, normEps: normEps, qkNorm: qkNorm)
    if modulation {
      self._adaLN.wrappedValue = [Linear(min(dim, 256), 4 * dim, bias: true)]
    }
    self._attentionNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    self._ffnNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    self._attentionNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    self._ffnNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)

    let hiddenDim = Int(Float(dim) / 3.0 * 8.0)
    self._feedForward.wrappedValue = ZImageFeedForward(dim: dim, hiddenDim: hiddenDim)

    super.init()
  }

  func callAsFunction(
    _ x: MLXArray,
    attnMask: MLXArray? = nil,
    freqsCis: MLXArray? = nil,
    adalnInput: MLXArray? = nil
  ) -> MLXArray {
    var out = x

    if modulation, let c = adalnInput, let adaLNModule = adaLN {
      let mod = adaLNModule[0](c)
      let chunkSize = dim

      let attnScale = (1 + mod[0..., 0..<chunkSize])[.ellipsis, .newAxis, 0...]
      let attnGate = MLX.tanh(mod[0..., chunkSize..<(2*chunkSize)])[.ellipsis, .newAxis, 0...]
      let mlpScale = (1 + mod[0..., (2*chunkSize)..<(3*chunkSize)])[.ellipsis, .newAxis, 0...]
      let mlpGate = MLX.tanh(mod[0..., (3*chunkSize)..<(4*chunkSize)])[.ellipsis, .newAxis, 0...]

      let xNormed = attentionNorm1(out)
      let xScaled = xNormed * attnScale
      let attnOut = attention(xScaled, attnMask: attnMask, freqsCis: freqsCis)
      let attnNormed = attentionNorm2(attnOut)
      out = out + attnGate * attnNormed

      let ffnOut = feedForward(ffnNorm1(out) * mlpScale)
      out = out + mlpGate * ffnNorm2(ffnOut)
    } else {
      let attnOut = attention(attentionNorm1(out), attnMask: attnMask, freqsCis: freqsCis)
      out = out + attentionNorm2(attnOut)

      let ffnOut = feedForward(ffnNorm1(out))
      out = out + ffnNorm2(ffnOut)
    }

    return out
  }
}
