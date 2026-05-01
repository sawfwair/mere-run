import Foundation
import MLX
import MLXNN

public enum LoRAApplicator {

    /// Merges LoRA weights into the given transformer in-place (static merge).
    /// - Returns: The number of layers modified.
    @discardableResult
    public static func mergeIntoTransformer(
        _ transformer: Flux2Transformer2DModel,
        loraWeights: LoRAWeights,
        scale: Float
    ) -> Int {
        mergeIntoModule(transformer, loraWeights: loraWeights, scale: scale)
    }

    /// Merges LoRA weights into any MLX `Module` in-place (static merge).
    /// - Returns: The number of layers modified.
    @discardableResult
    public static func mergeIntoModule(
        _ module: Module,
        loraWeights: LoRAWeights,
        scale: Float
    ) -> Int {
        let effectiveScale = scale * loraWeights.effectiveScale

        var appliedCount = 0
        var layerUpdates: [String: MLXArray] = [:]

        for (key, child) in module.namedModules() {
            guard let (down, up) = loraWeights.weights[key] ?? loraWeights.weights[key + ".weight"] else {
                continue
            }

            guard let delta = computeDelta(up: up, down: down) else {
                continue
            }

            if let quantizedLinear = child as? QuantizedLinear {
                let dequantizedWeight = MLX.dequantized(
                    quantizedLinear.weight,
                    scales: quantizedLinear.scales,
                    biases: quantizedLinear.biases,
                    groupSize: quantizedLinear.groupSize,
                    bits: quantizedLinear.bits
                )

                guard let alignedDelta = alignShape(delta, to: dequantizedWeight.shape) else {
                    continue
                }

                let fusedWeight = dequantizedWeight + (alignedDelta * effectiveScale).asType(dequantizedWeight.dtype)

                let (newQuantizedWeight, newScales, newBiases) = MLX.quantized(
                    fusedWeight,
                    groupSize: quantizedLinear.groupSize,
                    bits: quantizedLinear.bits
                )

                layerUpdates[key + ".weight"] = newQuantizedWeight
                layerUpdates[key + ".scales"] = newScales
                if let biases = newBiases {
                    layerUpdates[key + ".biases"] = biases
                }

                appliedCount += 1
                continue
            }

            if let linear = child as? Linear {
                let currentWeight = linear.weight
                guard let alignedDelta = alignShape(delta, to: currentWeight.shape) else {
                    continue
                }

                layerUpdates[key + ".weight"] = currentWeight + (alignedDelta * effectiveScale).asType(currentWeight.dtype)
                appliedCount += 1
            }
        }

        if !layerUpdates.isEmpty {
            module.update(parameters: ModuleParameters.unflattened(layerUpdates))
        }

        return appliedCount
    }

    private static func computeDelta(up: MLXArray, down: MLXArray) -> MLXArray? {
        guard up.ndim == 2, down.ndim == 2 else { return nil }

        if up.dim(1) == down.dim(0) {
            return MLX.matmul(up, down)
        }
        if up.dim(0) == down.dim(1) {
            return MLX.matmul(up.T, down.T)
        }
        if up.dim(1) == down.dim(1) {
            return MLX.matmul(up, down.T)
        }

        return nil
    }

    private static func alignShape(_ delta: MLXArray, to targetShape: [Int]) -> MLXArray? {
        if delta.shape == targetShape {
            return delta
        }
        if delta.T.shape == targetShape {
            return delta.T
        }
        return nil
    }
}
