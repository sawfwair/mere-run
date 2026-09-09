import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunTensor
import MereRunGemmaModel
import MereRunDecode

package struct LagunaNativeAffineWeight {
    package let packedCodes: MLXArray
    package let scales: MLXArray
    package let biases: MLXArray
    package let originalShape: [Int]

    package var arrays: [MLXArray] { [packedCodes, scales, biases] }
}

package func lagunaNativeAffineWeight(_ weight: MLXArray) -> LagunaNativeAffineWeight? {
    guard weight.dtype == .bfloat16,
          weight.ndim == 2,
          weight.dim(1).isMultiple(of: 32) else {
        return nil
    }
    let quantizedWeight = MLX.quantized(
        weight,
        groupSize: 32,
        bits: 8,
        mode: .affine
    )
    guard let biases = quantizedWeight.biases else { return nil }
    return LagunaNativeAffineWeight(
        packedCodes: quantizedWeight.wq,
        scales: quantizedWeight.scales,
        biases: biases,
        originalShape: weight.shape
    )
}
