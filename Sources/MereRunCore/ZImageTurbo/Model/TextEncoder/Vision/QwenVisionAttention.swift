import Foundation
import MLX
import MLXNN
import MLXFast

final class QwenVisionAttention: Module {
  let embedDim: Int
  let numHeads: Int
  let headDim: Int
  let scale: Float

  @ModuleInfo(key: "qkv") private var qkv: Linear
  @ModuleInfo(key: "proj") private var proj: Linear

  init(embedDim: Int, numHeads: Int) {
    precondition(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")
    self.embedDim = embedDim
    self.numHeads = numHeads
    self.headDim = embedDim / numHeads
    self.scale = 1.0 / Float(sqrt(Double(headDim)))

    self._qkv.wrappedValue = Linear(embedDim, embedDim * 3)
    self._proj.wrappedValue = Linear(embedDim, embedDim)
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    rotaryEmbedding: (cos: MLXArray, sin: MLXArray)? = nil,
    attentionMask: MLXArray? = nil,
    cuSeqlens: MLXArray? = nil
  ) -> MLXArray {
    var states = hiddenStates

    let batch = states.dim(0)
    let sequence = states.dim(1)

    states = qkv(states)
    states = states.reshaped(batch, sequence, 3, numHeads, headDim)
    // Linear outputs before rotary, flattened over heads
    let qPre = states[0..., 0..., 0..<1, 0..., 0...].squeezed(axis: 2) // [B,S,H,D]
    let kPre = states[0..., 0..., 1..<2, 0..., 0...].squeezed(axis: 2) // [B,S,H,D]
    let vPre = states[0..., 0..., 2..<3, 0..., 0...].squeezed(axis: 2) // [B,S,H,D]

    var q = qPre.transposed(0, 2, 1, 3)  // [B, H, S, D]
    var k = kPre.transposed(0, 2, 1, 3)  // [B, H, S, D]
    var v = vPre.transposed(0, 2, 1, 3)  // [B, H, S, D]

    if let rotaryEmbedding {
      q = applyRotary(q, cos: rotaryEmbedding.cos, sin: rotaryEmbedding.sin)
      k = applyRotary(k, cos: rotaryEmbedding.cos, sin: rotaryEmbedding.sin)
    }

    var context: MLXArray

    // Use sequence splitting for cu_seqlens (like mlx-vlm) instead of mask-based attention
    if let cuSeqlens = cuSeqlens {
      MLX.eval(cuSeqlens)
      let seqlensArray = cuSeqlens.asType(.int32).asArray(Int32.self).map { Int($0) }

      // Split q, k, v by cu_seqlens boundaries and process each segment
      var attnOutputs: [MLXArray] = []
      for i in 0..<(seqlensArray.count - 1) {
        let start = seqlensArray[i]
        let end = seqlensArray[i + 1]
        guard end > start else { continue }

        // Slice along sequence dimension (axis 2 for [B, H, S, D])
        let qSlice = q[0..., 0..., start..<end, 0...]
        let kSlice = k[0..., 0..., start..<end, 0...]
        let vSlice = v[0..., 0..., start..<end, 0...]

        let segmentOutput = MLXFast.scaledDotProductAttention(
          queries: qSlice,
          keys: kSlice,
          values: vSlice,
          scale: scale,
          mask: .none
        )
        attnOutputs.append(segmentOutput)
      }

      // Concatenate all segment outputs
      context = attnOutputs.count == 1 ? attnOutputs[0] : MLX.concatenated(attnOutputs, axis: 2)
    } else if let attentionMask = attentionMask {
      // Fallback to mask-based attention for Qwen2-VL (windowed attention)
      let baseMask = prepareAttentionMask(attentionMask, batch: batch)
      context = MLXFast.scaledDotProductAttention(
        queries: q,
        keys: k,
        values: v,
        scale: scale,
        mask: .array(baseMask)
      )
    } else {
      // No mask, no cu_seqlens - standard attention
      context = MLXFast.scaledDotProductAttention(
        queries: q,
        keys: k,
        values: v,
        scale: scale,
        mask: .none
      )
    }

    context = context.transposed(0, 2, 1, 3)  // [B, S, H, D]
    context = context.reshaped(batch, sequence, embedDim)
    context = proj(context)
    return context
  }

  private func prepareAttentionMask(
    _ mask: MLXArray,
    batch: Int
  ) -> MLXArray {
    var prepared = mask
    if prepared.ndim == 2 {
      prepared = prepared[.newAxis, 0..., 0...]
    }
    precondition(
      prepared.ndim == 3,
      "Vision attention mask must have shape [batch, sequence, sequence]"
    )
    precondition(
      prepared.dim(0) == batch || prepared.dim(0) == 1,
      "Vision attention mask batch dimension mismatch"
    )
    if prepared.dim(0) == 1 && batch > 1 {
      let broadcastShape = [batch, prepared.dim(1), prepared.dim(2)]
      let floatMask = prepared.asType(prepared.dtype)
      prepared = MLX.broadcast(floatMask, to: broadcastShape)
    }
    return prepared.asType(.bool)
  }

  private func applyRotary(_ tensor: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let cosPrepared = cos[.newAxis, .newAxis, 0..., 0...]
    let sinPrepared = sin[.newAxis, .newAxis, 0..., 0...]
    return (tensor * cosPrepared) + (rotateHalf(tensor) * sinPrepared)
  }

  private func rotateHalf(_ tensor: MLXArray) -> MLXArray {
    let half = tensor.dim(-1) / 2
    let firstHalf = tensor[0..., 0..., 0..., 0..<half]
    let secondHalf = tensor[0..., 0..., 0..., half...]
    return MLX.concatenated([-secondHalf, firstHalf], axis: -1)
  }
}
