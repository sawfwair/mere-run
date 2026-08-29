import MLX
import XCTest
@testable import MereRunCore

/// Fixed-code affine least squares with a diagonal input-second-moment metric.
/// This is not a full AWQ/GPTQ/DWQ implementation and does not recover BF16 weights.
enum Q38ActivationWeightedRefit {
    static func fit(
        teacher: MLXArray, codes: MLXArray, importance: MLXArray,
        scales: MLXArray, biases: MLXArray
    ) -> (scales: MLXArray, biases: MLXArray) {
        let weights = importance.asType(.float32)
        let mass = weights.sum(axis: -1)
        let codeSum = (weights * codes).sum(axis: -1)
        let codeSquared = (weights * codes * codes).sum(axis: -1)
        let targetSum = (weights * teacher).sum(axis: -1)
        let productSum = (weights * codes * teacher).sum(axis: -1)
        let determinant = mass * codeSquared - codeSum * codeSum
        let valid = (mass .> 0) .&& (determinant .> MLX.abs(mass * codeSquared) * 0.000001)
        let newScale = (mass * productSum - codeSum * targetSum) / MLX.where(valid, determinant, 1)
        let newBias = (targetSum - newScale * codeSum) / MLX.where(mass .> 0, mass, 1)
        // Score after casting to the actual checkpoint parameter dtype. An
        // FP32-only improvement must not sneak through a BF16 parameter export.
        let castScale = newScale.asType(scales.dtype)
        let castBias = newBias.asType(biases.dtype)
        let previous = teacher - codes * scales.asType(.float32).expandedDimensions(axis: -1)
            - biases.asType(.float32).expandedDimensions(axis: -1)
        let candidate = teacher - codes * castScale.asType(.float32).expandedDimensions(axis: -1)
            - castBias.asType(.float32).expandedDimensions(axis: -1)
        let improves = (weights * candidate * candidate).sum(axis: -1)
            .< (weights * previous * previous).sum(axis: -1)
        // MLX affine groups can legitimately use a negative scale.
        let accept = valid .&& improves
        return (MLX.where(accept, castScale, scales), MLX.where(accept, castBias, biases))
    }

    static func arrays(
        source: Q35SwitchLinear, candidate: (MLXArray, MLXArray, MLXArray), importance: MLXArray
    ) throws -> (MLXArray, MLXArray, MLXArray) {
        let sourceScales = try XCTUnwrap(source.scales)
        let sourceBiases = try XCTUnwrap(source.biases)
        let groupSize = 64
        var scales: [MLXArray] = []
        var biases: [MLXArray] = []
        // Only eight experts' dense weights and objectives are live at a time.
        for start in stride(from: 0, to: source.weight.dim(0), by: 8) {
            let end = min(start + 8, source.weight.dim(0))
            let count = end - start
            let original = MLX.dequantized(
                source.weight[start..<end], scales: sourceScales[start..<end], biases: sourceBiases[start..<end],
                groupSize: source.groupSize, bits: source.bits, dtype: .bfloat16
            ).asType(.float32).reshaped([count, source.weight.dim(1), -1, groupSize])
            let oldScale = candidate.1[start..<end]
            let oldBias = candidate.2[start..<end]
            // Affine dequantization computes in the parameter dtype before
            // applying its output dtype. Promote parameters to recover exact
            // integer codes without BF16 reconstruction rounding.
            let decoded = MLX.dequantized(candidate.0[start..<end], scales: oldScale.asType(.float32),
                                          biases: oldBias.asType(.float32),
                                          groupSize: groupSize, bits: 3, dtype: .float32)
                .reshaped(original.shape)
            let scale = oldScale.asType(.float32).expandedDimensions(axis: -1)
            let offset = oldBias.asType(.float32).expandedDimensions(axis: -1)
            let codes = MLX.clip(MLX.round((decoded - offset) / MLX.where(scale .!= 0, scale, 1)), min: 0, max: 7)
            let weights = importance[start..<end].reshaped([count, 1, -1, groupSize])
            let fit = fit(teacher: original, codes: codes, importance: weights,
                          scales: oldScale, biases: oldBias)
            func actualError(scales: MLXArray, biases: MLXArray) -> MLXArray {
                let reconstruction = MLX.dequantized(candidate.0[start..<end], scales: scales, biases: biases,
                                                     groupSize: groupSize, bits: 3, dtype: .bfloat16)
                    .asType(.float32).reshaped(original.shape)
                let difference = original - reconstruction
                return (weights * difference * difference).sum(axis: -1)
            }
            let improves = actualError(scales: fit.scales, biases: fit.biases)
                .< actualError(scales: oldScale, biases: oldBias)
            let acceptedScale = MLX.where(improves, fit.scales, oldScale)
            let acceptedBias = MLX.where(improves, fit.biases, oldBias)
            MLX.eval(acceptedScale, acceptedBias)
            scales.append(acceptedScale)
            biases.append(acceptedBias)
            Memory.clearCache()
        }
        let result = (candidate.0, MLX.concatenated(scales, axis: 0), MLX.concatenated(biases, axis: 0))
        MLX.eval(result.0, result.1, result.2)
        return result
    }
}
