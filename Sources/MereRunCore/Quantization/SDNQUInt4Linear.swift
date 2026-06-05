import Foundation
import MLX
import MLXNN

/// SDNQ asymmetric uint4 linear weights.
///
/// SDNQ stores two unsigned 4-bit values per byte, low nibble first, with
/// per-output/per-group affine parameters: `dequantized = zero_point + q * scale`.
public final class SDNQUInt4Linear: QuantizedLinear {
    private let inputDimensions: Int
    private let outputDimensions: Int
    private let zeroPoints: MLXArray
    private var cachedDequantizedWeight: MLXArray?
    private var cachedDequantizedWeightDType: DType?

    public init(
        weight: MLXArray,
        bias: MLXArray?,
        scales: MLXArray,
        zeroPoints: MLXArray,
        inputDimensions: Int,
        outputDimensions: Int,
        groupSize: Int = 64
    ) {
        self.inputDimensions = inputDimensions
        self.outputDimensions = outputDimensions
        self.zeroPoints = zeroPoints
        super.init(
            weight: weight,
            bias: bias,
            scales: scales,
            biases: nil,
            groupSize: groupSize,
            bits: 4,
            mode: .affine
        )
    }

    public override var shape: (Int, Int) {
        (outputDimensions, inputDimensions)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let computeDType: DType = x.dtype.isFloatingPoint ? x.dtype : .bfloat16
        let input = x.dtype == computeDType ? x : x.asType(computeDType)
        var result = MLX.matmul(input, dequantizedWeight(dtype: computeDType).T)
        if let bias {
            result = result + bias
        }
        return result
    }

    private func dequantizedWeight(dtype: DType) -> MLXArray {
        if let cachedDequantizedWeight, cachedDequantizedWeightDType == dtype {
            return cachedDequantizedWeight
        }

        precondition(inputDimensions > 0, "SDNQUInt4Linear requires a positive input dimension.")
        precondition(outputDimensions > 0, "SDNQUInt4Linear requires a positive output dimension.")
        precondition(inputDimensions % 2 == 0, "SDNQ uint4 input dimension must be even.")
        precondition(inputDimensions % groupSize == 0, "SDNQ uint4 input dimension must be divisible by group size.")

        let groupCount = inputDimensions / groupSize
        let unpacked = SDNQUInt4Dequantizer.unpack(
            weight,
            quantizedShape: [outputDimensions, groupCount, groupSize]
        ).asType(dtype)

        let normalizedScales = SDNQUInt4Dequantizer.squeezedGroupParameter(scales)
            .asType(dtype)
            .reshaped(outputDimensions, groupCount, 1)
        let normalizedZeroPoints = SDNQUInt4Dequantizer.squeezedGroupParameter(zeroPoints)
            .asType(dtype)
            .reshaped(outputDimensions, groupCount, 1)
        let dequantized = (normalizedZeroPoints + unpacked * normalizedScales)
            .reshaped(outputDimensions, inputDimensions)

        eval(dequantized)
        cachedDequantizedWeight = dequantized
        cachedDequantizedWeightDType = dtype
        return dequantized
    }
}
