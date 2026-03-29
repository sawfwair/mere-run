import Foundation
import MLX

public final class ZImageRopeEmbedder {
  let theta: Float
  let axesDims: [Int]
  let axesLens: [Int]
  private var freqsCis: [MLXArray]?

  init(theta: Float, axesDims: [Int], axesLens: [Int]) {
    self.theta = theta
    self.axesDims = axesDims
    self.axesLens = axesLens
    precondition(axesDims.count == axesLens.count, "axesDims and axesLens must have same length")
  }

  private func precomputeFreqsIfNeeded() {
    if freqsCis != nil { return }
    var tables: [MLXArray] = []
    tables.reserveCapacity(axesDims.count)

    for (dim, end) in zip(axesDims, axesLens) {
      let halfDim = dim / 2
      let idx = MLXArray(0..<halfDim).asType(.float32) * 2.0
      var exponent = idx / MLXArray(Float(dim))
      exponent = -exponent
      let base = MLXArray(theta)
      let freqs = MLX.pow(base, exponent)

      let timesteps = MLXArray(0..<end).asType(.float32)
      let angles = timesteps[.ellipsis, .newAxis] * freqs[.newAxis]

      let cosVals = MLX.cos(angles)
      let sinVals = MLX.sin(angles)

      let stacked = MLX.stacked([cosVals, sinVals], axis: -1)
      tables.append(stacked)
    }
    freqsCis = tables
  }

  func callAsFunction(ids: MLXArray) -> MLXArray {
    precomputeFreqsIfNeeded()
    guard let freqsCis else {
      return MLX.zeros([ids.dim(0), axesDims.reduce(0, +) / 2, 2])
    }
    precondition(ids.ndim == 2, "ids must be [N, numAxes]")
    precondition(ids.dim(1) == axesDims.count, "ids last dimension must equal axes count")

    let batch = ids.dim(0)
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(freqsCis.count)

    for (axisIndex, table) in freqsCis.enumerated() {
      let index = ids[0..., axisIndex]
      let selected = table[index, 0..., 0...]
      outputs.append(selected)
    }

    let totalHalfDim = axesDims.reduce(0) { $0 + $1 / 2 }
    if outputs.isEmpty {
      return MLX.zeros([batch, totalHalfDim, 2])
    }
    return MLX.concatenated(outputs, axis: 1)
  }
}
