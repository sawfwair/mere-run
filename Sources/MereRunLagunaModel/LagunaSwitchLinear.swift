import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunTensor
import MereRunGemmaModel
import MereRunDecode

package final class LagunaSwitchLinear: Module {
    @ModuleInfo(key: "weight") package var weight: MLXArray
    @ModuleInfo(key: "scales") package var scales: MLXArray?
    @ModuleInfo(key: "biases") package var biases: MLXArray?

    package let groupSize: Int
    package let bits: Int
    package let mode: QuantizationMode

    package init(
        inputDimensions: Int,
        outputDimensions: Int,
        expertCount: Int,
        quantization: LagunaQuantizationConfig?
    ) {
        self.groupSize = quantization?.groupSize ?? 16
        self.bits = quantization?.bits ?? 4
        self.mode = quantization.flatMap { QuantizationMode(rawValue: $0.mode) } ?? .affine

        if quantization != nil {
            let packedInputDimensions = (inputDimensions * bits + 31) / 32
            self._weight.wrappedValue = MLXArray.zeros(
                [expertCount, outputDimensions, packedInputDimensions],
                dtype: .uint32
            )
            self._scales.wrappedValue = MLXArray.zeros(
                [expertCount, outputDimensions, max(1, inputDimensions / groupSize)]
            )
        } else {
            let scale = sqrt(1 / Float(max(1, inputDimensions)))
            self._weight.wrappedValue = MLXRandom.uniform(
                low: -scale,
                high: scale,
                [expertCount, outputDimensions, inputDimensions]
            )
            self._scales.wrappedValue = nil
        }
        self._biases.wrappedValue = nil
        super.init()
    }

    package func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let topK = indices.dim(2)
        let inputDimensions = x.dim(-1)
        let tokenCount = batch * sequenceLength

        let flatInput: MLXArray
        if x.ndim == 4 && x.dim(2) == topK {
            flatInput = x.reshaped([tokenCount * topK, 1, inputDimensions])
        } else {
            let expanded = MLX.repeated(
                MLX.expandedDimensions(x.reshaped([tokenCount, 1, inputDimensions]), axis: 1),
                count: topK,
                axis: 1
            )
            flatInput = expanded.reshaped([tokenCount * topK, 1, inputDimensions])
        }

        let output = applyFlat(
            flatInput,
            indices: indices.reshaped([tokenCount * topK]),
            sortedIndices: false
        )
        return output.reshaped([batch, sequenceLength, topK, output.dim(-1)])
    }

    package func applyFlat(
        _ x: MLXArray,
        indices: MLXArray,
        sortedIndices: Bool
    ) -> MLXArray {
        let output: MLXArray
        if let scales {
            output = portableGatherQuantizedMM(
                x,
                weight,
                scales: scales,
                biases: biases,
                rhsIndices: indices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                sortedIndices: sortedIndices
            )
        } else {
            output = gatherMM(
                x,
                weight.swappedAxes(-1, -2),
                rhsIndices: indices,
                sortedIndices: sortedIndices
            )
        }
        return output
    }
}
