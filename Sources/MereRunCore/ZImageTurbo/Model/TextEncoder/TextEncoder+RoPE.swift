import Foundation
import MLX

// Owns rotary-position helpers for the Qwen text encoder stack.
// This file intentionally stays limited to RoPE construction and application;
// attention blocks and top-level encoder orchestration live elsewhere.

/// Qwen2-VL uses 3D rotary position ids for vision tokens and 1D for text tokens.
final class Qwen2VLRotaryEmbedding {
  let dim: Int
  let base: Float
  let invFreq: MLXArray

  init(dim: Int, base: Float = 1_000_000.0) {
    self.dim = dim
    self.base = base

    let indices = MLXArray(Array(stride(from: 0, to: dim, by: 2)).map { Float($0) })
    self.invFreq = 1.0 / MLX.pow(MLXArray(base), indices / Float(dim))
  }

  /// Compute cos and sin for rotary embeddings
  /// - Parameters:
  ///   - positionIds: Shape (3, batch, seq_len) for multimodal, or (batch, seq_len) for text-only
  ///   - dtype: Output dtype
  /// - Returns: (cos, sin) each with shape (3, batch, seq_len, head_dim)
  func callAsFunction(positionIds: MLXArray, dtype: DType = .bfloat16) -> (cos: MLXArray, sin: MLXArray) {
    var posIds = positionIds

    if posIds.ndim == 2 {
      posIds = broadcast(posIds[.newAxis, 0..., 0...], to: [3, posIds.dim(0), posIds.dim(1)])
    }

    let invFreqExpanded = broadcast(
      invFreq[.newAxis, .newAxis, 0..., .newAxis].asType(.float32),
      to: [3, posIds.dim(1), invFreq.count, 1]
    )

    let posIdsExpanded = posIds[0..., 0..., .newAxis, 0...].asType(.float32)

    let freqs = MLX.matmul(invFreqExpanded, posIdsExpanded).transposed(0, 1, 3, 2)
    let emb = MLX.concatenated([freqs, freqs], axis: -1)

    return (MLX.cos(emb).asType(dtype), MLX.sin(emb).asType(dtype))
  }
}

/// Qwen3-VL uses interleaved MRoPE for multimodal position ids.
final class Qwen3VLRotaryEmbedding {
  let dim: Int
  let base: Float
  let invFreq: MLXArray
  let mropeSection: [Int]
  let interleavedSelector: MLXArray

  init(dim: Int, base: Float = 1_000_000.0, mropeSection: [Int]) {
    self.dim = dim
    self.base = base
    self.mropeSection = mropeSection

    let indices = MLXArray(Array(stride(from: 0, to: dim, by: 2)).map { Float($0) })
    self.invFreq = 1.0 / MLX.pow(MLXArray(base), indices / Float(dim))
    self.interleavedSelector = Qwen3VLRotaryEmbedding.buildInterleavedSelector(
      dim: max(1, dim / 2),
      mropeSection: mropeSection
    )
  }

  private static func buildInterleavedSelector(dim: Int, mropeSection: [Int]) -> MLXArray {
    var selector = [Int32](repeating: 0, count: dim)
    guard mropeSection.count >= 3 else {
      return MLXArray(selector).reshaped(1, 1, 1, dim)
    }

    let limits = [(1, 1), (2, 2)]
    for (dimIdx, offset) in limits {
      let length = min(dim, mropeSection[dimIdx] * 3)
      guard offset < length else { continue }
      for i in stride(from: offset, to: length, by: 3) {
        selector[i] = Int32(dimIdx)
      }
    }

    return MLXArray(selector).reshaped(1, 1, 1, dim)
  }

  /// Apply interleaved MRoPE to 3D rotary embeddings.
  /// Reorganizes frequency layout from chunked [TTT...HHH...WWW] to interleaved [THTHWHTHW...TT].
  private func applyInterleavedMRoPE(_ freqs: MLXArray) -> MLXArray {
    let selector = broadcast(
      interleavedSelector,
      to: [1, freqs.dim(1), freqs.dim(2), freqs.dim(3)]
    )
    return takeAlong(freqs, selector, axis: 0).squeezed(axis: 0)
  }

  /// Compute cos and sin for rotary embeddings
  /// - Parameters:
  ///   - positionIds: Shape (3, batch, seq_len) for multimodal, or (batch, seq_len) for text-only
  ///   - dtype: Output dtype
  /// - Returns: (cos, sin) each with shape (batch, seq_len, head_dim)
  func callAsFunction(positionIds: MLXArray, dtype: DType = .bfloat16) -> (cos: MLXArray, sin: MLXArray) {
    var posIds = positionIds

    if posIds.ndim == 2 {
      posIds = broadcast(posIds[.newAxis, 0..., 0...], to: [3, posIds.dim(0), posIds.dim(1)])
    }

    let invFreqExpanded = broadcast(
      invFreq[.newAxis, .newAxis, 0..., .newAxis].asType(.float32),
      to: [3, posIds.dim(1), invFreq.count, 1]
    )

    let posIdsExpanded = posIds[0..., 0..., .newAxis, 0...].asType(.float32)

    let freqs = MLX.matmul(invFreqExpanded, posIdsExpanded).transposed(0, 1, 3, 2)
    let freqsInterleaved = applyInterleavedMRoPE(freqs)
    let emb = MLX.concatenated([freqsInterleaved, freqsInterleaved], axis: -1)

    return (MLX.cos(emb).asType(dtype), MLX.sin(emb).asType(dtype))
  }
}

func rotateHalf(_ x: MLXArray) -> MLXArray {
  let halfDim = x.dim(-1) / 2
  let x1 = x[0..., 0..., 0..., 0..<halfDim]
  let x2 = x[0..., 0..., 0..., halfDim...]
  return MLX.concatenated([-x2, x1], axis: -1)
}

func applyRotaryPosEmb(
  _ q: MLXArray, _ k: MLXArray,
  cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
  let cosExp = cos[0..., .newAxis, 0..., 0...]
  let sinExp = sin[0..., .newAxis, 0..., 0...]

  let qEmbed = (q * cosExp) + (rotateHalf(q) * sinExp)
  let kEmbed = (k * cosExp) + (rotateHalf(k) * sinExp)

  return (qEmbed, kEmbed)
}

func applyMultimodalRoPE(
  _ q: MLXArray, _ k: MLXArray,
  cos: MLXArray, sin: MLXArray,
  mropeSection: [Int]
) -> (MLXArray, MLXArray) {
  let sectionSizes = mropeSection.map { $0 * 2 }
  var cosSlices: [MLXArray] = []
  var sinSlices: [MLXArray] = []
  cosSlices.reserveCapacity(sectionSizes.count)
  sinSlices.reserveCapacity(sectionSizes.count)

  var start = 0
  for (index, size) in sectionSizes.enumerated() {
    let end = start + size
    let cosChunk = cos[0..., 0..., 0..., start..<end]
    let sinChunk = sin[0..., 0..., 0..., start..<end]
    cosSlices.append(cosChunk[index % 3])
    sinSlices.append(sinChunk[index % 3])
    start = end
  }

  let cosMerged = MLX.concatenated(cosSlices, axis: -1)
  let sinMerged = MLX.concatenated(sinSlices, axis: -1)

  let cosExp = cosMerged[0..., .newAxis, 0..., 0...]
  let sinExp = sinMerged[0..., .newAxis, 0..., 0...]

  let qEmbed = (q * cosExp) + (rotateHalf(q) * sinExp)
  let kEmbed = (k * cosExp) + (rotateHalf(k) * sinExp)

  return (qEmbed, kEmbed)
}
