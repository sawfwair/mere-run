import Foundation
import MLX

enum ZImageAttentionUtils {

  static func applyComplexRoPEBLHD(
    query: MLXArray,
    key: MLXArray,
    freqsCis: MLXArray
  ) -> (MLXArray, MLXArray) {
    let freqsCos = freqsCis[0..., 0..., 0][.newAxis, 0..., .newAxis, 0...]
    let freqsSin = freqsCis[0..., 0..., 1][.newAxis, 0..., .newAxis, 0...]

    return (
      applyRotary(query, freqsCos, freqsSin),
      applyRotary(key, freqsCos, freqsSin)
    )
  }

  @inline(__always)
  private static func applyRotary(
    _ x: MLXArray,
    _ freqsCos: MLXArray,
    _ freqsSin: MLXArray
  ) -> MLXArray {
    let shape = x.shape
    let newShape = Array(shape.dropLast()) + [shape.last! / 2, 2]
    let xReshaped = x.reshaped(newShape)

    let xReal = xReshaped[0..., 0..., 0..., 0..., 0]
    let xImag = xReshaped[0..., 0..., 0..., 0..., 1]

    let outReal = xReal * freqsCos - xImag * freqsSin
    let outImag = xReal * freqsSin + xImag * freqsCos

    return MLX.stacked([outReal, outImag], axis: -1).reshaped(shape)
  }
}
