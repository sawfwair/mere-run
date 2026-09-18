import Foundation
import MLX
import MLXNN

/// Prism's folded weights require a signed, normalized transform at each projection.
package enum Q35PrismTransform {
    package static func apply(_ input: MLXArray, block: Int, signs: MLXArray, inverse: Bool = false) -> MLXArray {
        if let output = Q35PrismHadamard.apply(input, block: block, signs: signs, inverse: inverse) {
            return output
        }
        return reference(input, block: block, signs: signs, inverse: inverse)
    }

    package static func reference(_ input: MLXArray, block: Int, signs: MLXArray, inverse: Bool = false) -> MLXArray {
        let value = input.asType(.float32)
        let signed = inverse ? value : value * signs
        let transformed = MLX.hadamardTransform(
            signed.reshaped(-1, block), scale: 1 / Float(block).squareRoot()
        ).reshaped(input.shape)
        return (inverse ? transformed * signs : transformed).asType(input.dtype)
    }
}

package final class Q35PrismLinear: Linear {
    let scales: MLXArray
    let biases: MLXArray
    let signs: MLXArray?
    let block: Int

    package init(weight: MLXArray, scales: MLXArray, biases: MLXArray, signs: MLXArray?, block: Int) {
        self.scales = scales
        self.biases = biases
        self.signs = signs
        self.block = block
        super.init(weight: weight, bias: nil)
        freeze()
    }

    package override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let value = signs.map { Q35PrismTransform.apply(input, block: block, signs: $0) } ?? input
        return projectTransformed(value)
    }

    func projectTransformed(_ value: MLXArray) -> MLXArray {
        MLX.quantizedMM(value, weight, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2)
    }
}

package final class Q35PrismEmbedding: Embedding {
    let scales: MLXArray
    let biases: MLXArray
    let signs: MLXArray?
    let block: Int

    package init(weight: MLXArray, scales: MLXArray, biases: MLXArray, signs: MLXArray?, block: Int) {
        self.scales = scales
        self.biases = biases
        self.signs = signs
        self.block = block
        super.init(weight: weight)
        freeze()
    }

    package override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let indices = input.flattened()
        let value = MLX.dequantized(
            weight[indices], scales: scales[indices], biases: biases[indices], groupSize: 128, bits: 2
        ).reshaped(input.shape + [-1]).asType(.float16)
        return signs.map { Q35PrismTransform.apply(value, block: block, signs: $0, inverse: true) } ?? value
    }
}
