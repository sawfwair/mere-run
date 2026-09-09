import Foundation
import MLX
import MLXFast

final class Gemma4QuantizedTensorState {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?
    let groupSize: Int
    let bits: Int
    let dtype: DType
    let tokenCount: Int

    var tokenCapacity: Int {
        weight.dim(2)
    }

    var packedWidth: Int {
        weight.dim(3)
    }

    var groupCount: Int {
        scales.dim(3)
    }

    init(source: MLXArray, groupSize: Int, bits: Int) {
        let quantized = MLX.quantized(
            source,
            groupSize: groupSize,
            bits: bits,
            mode: .affine
        )
        self.weight = quantized.wq
        self.scales = quantized.scales
        self.biases = quantized.biases
        self.groupSize = groupSize
        self.bits = bits
        self.dtype = source.dtype
        self.tokenCount = source.dim(2)
    }

    init(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        dtype: DType,
        tokenCount: Int
    ) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
        self.dtype = dtype
        self.tokenCount = tokenCount
    }

    func appending(_ source: MLXArray) -> Gemma4QuantizedTensorState {
        let next = Gemma4QuantizedTensorState(source: source, groupSize: groupSize, bits: bits)
        var appendedBiases: MLXArray?
        if let biases, let nextBiases = next.biases {
            appendedBiases = Gemma4KVTokenStorage.appended(biases, rows: nextBiases, validCount: tokenCount)
        }
        return Gemma4QuantizedTensorState(
            weight: Gemma4KVTokenStorage.appended(weight, rows: next.weight, validCount: tokenCount),
            scales: Gemma4KVTokenStorage.appended(scales, rows: next.scales, validCount: tokenCount),
            biases: appendedBiases,
            groupSize: groupSize,
            bits: bits,
            dtype: dtype,
            tokenCount: tokenCount + next.tokenCount
        )
    }

    func dequantized() -> MLXArray {
        dequantized(tokenRange: 0..<tokenCount)
    }

    func dequantized(tokenRange: Range<Int>) -> MLXArray {
        let slicedWeight = weight[0..., 0..., tokenRange, 0...]
        let slicedScales = scales[0..., 0..., tokenRange, 0...]
        let slicedBiases = biases.map { $0[0..., 0..., tokenRange, 0...] }
        return MLX.dequantized(
            slicedWeight,
            scales: slicedScales,
            biases: slicedBiases,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            dtype: dtype
        )
    }
}
