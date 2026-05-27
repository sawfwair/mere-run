import Foundation
import MLX
import MLXFast
import MLXNN

public final class PrismBinaryQuantizedLinear: QuantizedLinear {
    private var cachedDequantizedWeight: MLXArray?

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let packedResult = packedBinaryMatmul(x) {
            return packedResult
        }

        let fullWeight = dequantizedBinaryWeight()
        var result = matmul(x, fullWeight.T)
        if let bias {
            result = result + bias
        }
        return result
    }

    private func dequantizedBinaryWeight() -> MLXArray {
        if let cachedDequantizedWeight {
            return cachedDequantizedWeight
        }

        precondition(bits == 1, "PrismBinaryQuantizedLinear only supports 1-bit weights.")
        precondition(groupSize > 0, "PrismBinaryQuantizedLinear requires a positive group size.")

        let outputDimensions = weight.dim(0)
        let inputDimensions = weight.dim(1) * 32
        precondition(inputDimensions % groupSize == 0, "Input dimension must be divisible by group size.")
        let groupCount = inputDimensions / groupSize

        let bitOffsets = MLXArray((0..<32).map { UInt32($0) }, [1, 1, 32])
        let packed = weight.expandedDimensions(axis: -1)
        let unpacked = bitwiseAnd(rightShift(packed, bitOffsets), UInt32(1))
            .reshaped([outputDimensions, groupCount, groupSize])
            .asType(scales.dtype)

        var fullWeight = unpacked * scales.expandedDimensions(axis: -1)
        if let biases {
            fullWeight = fullWeight + biases.expandedDimensions(axis: -1).asType(fullWeight.dtype)
        }

        let dequantized = fullWeight.reshaped([outputDimensions, inputDimensions])
        eval(dequantized)
        cachedDequantizedWeight = dequantized
        return dequantized
    }

    private func packedBinaryMatmul(_ x: MLXArray) -> MLXArray? {
        guard Device.defaultDevice().deviceType == .gpu,
              let biases else {
            return nil
        }

        precondition(bits == 1, "PrismBinaryQuantizedLinear only supports 1-bit weights.")
        precondition(groupSize > 0, "PrismBinaryQuantizedLinear requires a positive group size.")
        guard let inputDimensions = x.shape.last else {
            return nil
        }

        let outputDimensions = weight.dim(0)
        let packedWidth = weight.dim(1)
        guard packedWidth * 32 == inputDimensions else {
            return nil
        }
        precondition(inputDimensions % groupSize == 0, "Input dimension must be divisible by group size.")
        let groupCount = inputDimensions / groupSize
        let rowCount = x.size / inputDimensions
        let flatInput = x.reshaped([rowCount, inputDimensions])

        let kernel = PrismBinaryMatmulKernels.kernel(
            inputDimensions: inputDimensions,
            outputDimensions: outputDimensions,
            packedWidth: packedWidth,
            groupSize: groupSize,
            groupCount: groupCount
        )
        var output = kernel(
            [flatInput, weight, scales, biases],
            template: [
                ("InputDimensions", inputDimensions),
                ("OutputDimensions", outputDimensions),
                ("PackedWidth", packedWidth),
                ("GroupSize", groupSize),
                ("GroupCount", groupCount),
            ],
            grid: (32, outputDimensions, rowCount),
            threadGroup: (32, 1, 1),
            outputShapes: [[rowCount, outputDimensions]],
            outputDTypes: [x.dtype]
        )[0]

        var outputShape = Array(x.shape.dropLast())
        outputShape.append(outputDimensions)
        output = output.reshaped(outputShape)
        if let bias {
            output = output + bias
        }
        return output
    }
}

private struct PrismBinaryMatmulKernelKey: Hashable {
    let inputDimensions: Int
    let outputDimensions: Int
    let packedWidth: Int
    let groupSize: Int
    let groupCount: Int
}

private enum PrismBinaryMatmulKernels {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var kernels: [PrismBinaryMatmulKernelKey: MLXFast.MLXFastKernel] = [:]

    static func kernel(
        inputDimensions: Int,
        outputDimensions: Int,
        packedWidth: Int,
        groupSize: Int,
        groupCount: Int
    ) -> MLXFast.MLXFastKernel {
        let key = PrismBinaryMatmulKernelKey(
            inputDimensions: inputDimensions,
            outputDimensions: outputDimensions,
            packedWidth: packedWidth,
            groupSize: groupSize,
            groupCount: groupCount
        )

        lock.lock()
        defer { lock.unlock() }

        if let existing = kernels[key] {
            return existing
        }

        let kernel = MLXFast.metalKernel(
            name: "prism_binary_affine_matmul_i\(inputDimensions)_o\(outputDimensions)_pw\(packedWidth)_g\(groupSize)_gc\(groupCount)",
            inputNames: ["x", "packed", "scales", "biases"],
            outputNames: ["out"],
            source: """
                auto lane = thread_position_in_grid.x;
                auto output_idx = thread_position_in_grid.y;
                auto row_idx = thread_position_in_grid.z;
                if (output_idx >= OutputDimensions || row_idx >= x_shape[0]) {
                    return;
                }

                auto x_ptr = x + row_idx * InputDimensions;
                auto packed_ptr = packed + output_idx * PackedWidth;
                auto scale_ptr = scales + output_idx * GroupCount;
                auto bias_ptr = biases + output_idx * GroupCount;

                float acc = 0.0f;
                for (int input_idx = lane; input_idx < InputDimensions; input_idx += 32) {
                    uint packed_word = packed_ptr[input_idx / 32];
                    uint decoded_bit = (packed_word >> (input_idx % 32)) & 1u;
                    int group_idx = input_idx / GroupSize;
                    float decoded = static_cast<float>(decoded_bit) * static_cast<float>(scale_ptr[group_idx])
                        + static_cast<float>(bias_ptr[group_idx]);
                    acc += static_cast<float>(x_ptr[input_idx]) * decoded;
                }

                acc = simd_sum(acc);
                if (thread_index_in_simdgroup == 0) {
                    out[row_idx * OutputDimensions + output_idx] = acc;
                }
                """
        )
        kernels[key] = kernel
        return kernel
    }
}
