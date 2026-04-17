import Foundation
import MLX
import MLXFast
import MLXNN

final class ZImageSelfAttention: Module {
  let dim: Int
  let heads: Int
  let headDim: Int
  let useQKNorm: Bool
  let scale: Float

  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "to_out") var toOut: [Linear]
  @ModuleInfo(key: "norm_q") var normQ: RMSNorm?
  @ModuleInfo(key: "norm_k") var normK: RMSNorm?

  init(dim: Int, heads: Int, normEps: Float, qkNorm: Bool) {
    self.dim = dim
    self.heads = heads
    self.headDim = dim / heads
    self.useQKNorm = qkNorm
    self.scale = 1.0 / sqrt(Float(dim / heads))

    self._toQ.wrappedValue = Linear(dim, dim, bias: false)
    self._toK.wrappedValue = Linear(dim, dim, bias: false)
    self._toV.wrappedValue = Linear(dim, dim, bias: false)
    self._toOut.wrappedValue = [Linear(dim, dim, bias: false)]
    if qkNorm {
      self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: normEps)
      self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: normEps)
    }
    super.init()
  }

  func callAsFunction(
    _ x: MLXArray,
    attnMask: MLXArray? = nil,
    freqsCis: MLXArray? = nil
  ) -> MLXArray {
    let batch = x.dim(0)
    let seqLen = x.dim(1)

    var q = toQ(x).reshaped(batch, seqLen, heads, headDim)
    var k = toK(x).reshaped(batch, seqLen, heads, headDim)
    let v = toV(x).reshaped(batch, seqLen, heads, headDim).transposed(0, 2, 1, 3)

    if useQKNorm {
      if let normQ { q = normQ(q) }
      if let normK { k = normK(k) }
    }

    if let freqsCis {
      (q, k) = ZImageAttentionUtils.applyComplexRoPEBLHD(query: q, key: k, freqsCis: freqsCis)
    }

    q = q.transposed(0, 2, 1, 3)
    k = k.transposed(0, 2, 1, 3)

    let attn = MLXFast.scaledDotProductAttention(
      queries: q,
      keys: k,
      values: v,
      scale: scale,
      mask: attnMask
    ).transposed(0, 2, 1, 3).reshaped(batch, seqLen, dim)

    return toOut[0](attn)
  }
}
