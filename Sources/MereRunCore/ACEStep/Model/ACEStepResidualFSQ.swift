import Foundation
import MLX
import MLXNN

/// Finite Scalar Quantization (FSQ) used by ACE-Step for 5Hz audio codes.
///
/// The turbo checkpoints use `levels = [8, 8, 8, 5, 5, 5]` which yields 64,000 codes.
final class ACEStepResidualFSQ: Module {
    @ModuleInfo(key: "project_in") var projectIn: Linear
    @ModuleInfo(key: "project_out") var projectOut: Linear

    let levels: [Int]
    private let levelsMinus1: MLXArray
    private let softClampValues: MLXArray
    private let basis: MLXArray

    init(config: ACEStepConfig) {
        precondition(config.fsqInputNumQuantizers == 1, "Only fsq_input_num_quantizers=1 is supported right now.")
        precondition(!config.fsqInputLevels.isEmpty, "fsq_input_levels must not be empty.")

        self.levels = config.fsqInputLevels

        let d = config.fsqInputLevels.count
        self._projectIn.wrappedValue = Linear(config.hiddenSize, d, bias: true)
        self._projectOut.wrappedValue = Linear(d, config.hiddenSize, bias: true)

        let minus1 = config.fsqInputLevels.map { Float32(max(1, $0 - 1)) }
        self.levelsMinus1 = MLXArray(minus1, [1, 1, d]).asType(.float32)
        self.softClampValues = MLXArray(
            minus1.map { Float32(1.0) + (Float32(1.0) / $0) },
            [1, 1, d]
        ).asType(.float32)

        var basisVals: [Int32] = []
        basisVals.reserveCapacity(d)
        var running: Int32 = 1
        for (i, level) in config.fsqInputLevels.enumerated() {
            if i == 0 {
                basisVals.append(1)
                running = Int32(level)
            } else {
                basisVals.append(running)
                running = running * Int32(level)
            }
        }
        self.basis = MLXArray(basisVals.map(Float32.init), [1, 1, d]).asType(.int32)
    }

    func callAsFunction(_ x: MLXArray) -> (quantized: MLXArray, indices: MLXArray) {
        // vector-quantize-pytorch ResidualFSQ soft-clamps before the
        // symmetry-preserving one-layer FSQ used by ACE-Step.
        var y = projectIn(x).asType(.float32)
        y = MLX.tanh(y / softClampValues) * softClampValues
        y = minimum(maximum(y, MLXArray(Float32(-1))), MLXArray(Float32(1)))

        let qFloat = floor(levelsMinus1 * ((y + 1) * 0.5) + MLXArray(Float32(0.5)))
        let qClamped = minimum(maximum(qFloat, MLXArray(Float32(0))), levelsMinus1)
        let qInt = qClamped.asType(.int32)

        // Mixed-radix flattening: idx in [0, prod(levels)-1].
        let idx = (qInt * basis).sum(axis: -1).asType(.int32) // [B, T]
        let indices = idx.expandedDimensions(axis: -1) // [B, T, 1]

        let quantizedScalars = (qClamped * (2.0 / levelsMinus1) - 1.0).asType(x.dtype)
        let quantized = projectOut(quantizedScalars).asType(x.dtype)
        return (quantized, indices)
    }

    func getOutputFromIndices(_ indices: MLXArray, dtype: DType = .bfloat16) -> MLXArray {
        var idx = indices
        if idx.ndim == 3 {
            idx = idx.squeezed(axis: -1)
        }
        idx = idx.asType(.int32)

        var remaining = idx
        var digits: [MLXArray] = []
        digits.reserveCapacity(levels.count)

        for level in levels {
            let l = MLXArray(Int32(level))
            let q = remainder(remaining, l).asType(.int32)
            digits.append(q)
            remaining = floorDivide(remaining, l).asType(.int32)
        }

        let qStack = MLX.concatenated(digits.map { $0.expandedDimensions(axis: -1) }, axis: -1).asType(.float32)
        let scalars = (qStack * (2.0 / levelsMinus1) - 1.0).asType(dtype)
        return projectOut(scalars).asType(dtype)
    }
}
