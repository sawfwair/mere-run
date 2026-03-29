import Foundation
import MLX
import MLXNN

public enum QwenTextLoRAApplicator {

    /// Merges LoRA weights into the given text encoder in-place (static merge).
    /// - Returns: The number of layers modified.
    @discardableResult
    public static func mergeIntoTextEncoder(
        _ encoder: QwenTextEncoder,
        loraWeights: LoRAWeights,
        scale: Float
    ) -> Int {
        let debugLog = MereRunRuntimeDebug.logger(keys: ["MERERUN_LORA_DEBUG"], prefix: "[QwenTextLoRAApplicator]")
        let effectiveScale = scale * loraWeights.effectiveScale

        var appliedCount = 0
        var layerUpdates: [String: MLXArray] = [:]

        // Debug: print available keys
        let loraKeys = Set(loraWeights.weights.keys)
        let encoderKeys = Set(encoder.namedModules().map { $0.0 })
        let intersection = loraKeys.intersection(encoderKeys)
        debugLog?("LoRA keys count: \(loraKeys.count)")
        debugLog?("Encoder keys count: \(encoderKeys.count)")
        debugLog?("Matching keys: \(intersection.count)")
        if intersection.isEmpty {
            debugLog?("Sample LoRA key: '\(loraKeys.first ?? "none")'")
            debugLog?("Sample encoder key with 'q_proj': '\(encoderKeys.first { $0.contains("q_proj") } ?? "none")'")
        }

        var checkedCount = 0
        var foundCount = 0
        for (key, module) in encoder.namedModules() {
            checkedCount += 1
            guard let (down, up) = loraWeights.weights[key] ?? loraWeights.weights[key + ".weight"] else {
                continue
            }
            foundCount += 1

            guard let delta = computeDelta(up: up, down: down) else {
                debugLog?("computeDelta failed for '\(key)' - down:\(down.shape) up:\(up.shape)")
                continue
            }

            let moduleType = String(describing: type(of: module))
            if !(module is QuantizedLinear) && !(module is Linear) {
                debugLog?("Skipping '\(key)' - not Linear/QuantizedLinear, is \(moduleType)")
                continue
            }

            // Debug: show first few successful matches
            if foundCount <= 3 {
                debugLog?("Processing '\(key)' delta:\(delta.shape) moduleType:\(moduleType)")
            }

            if let quantizedLinear = module as? QuantizedLinear {
                let dequantizedWeight = MLX.dequantized(
                    quantizedLinear.weight,
                    scales: quantizedLinear.scales,
                    biases: quantizedLinear.biases,
                    groupSize: quantizedLinear.groupSize,
                    bits: quantizedLinear.bits
                )

                guard let alignedDelta = alignShape(delta, to: dequantizedWeight.shape) else {
                    if foundCount <= 3 {
                        debugLog?("alignShape failed for QuantizedLinear '\(key)' - delta:\(delta.shape) target:\(dequantizedWeight.shape)")
                    }
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

            if let linear = module as? Linear {
                let currentWeight = linear.weight
                guard let alignedDelta = alignShape(delta, to: currentWeight.shape) else {
                    if foundCount <= 3 {
                        debugLog?("alignShape failed for Linear '\(key)' - delta:\(delta.shape) target:\(currentWeight.shape)")
                    }
                    continue
                }

                layerUpdates[key + ".weight"] = currentWeight + (alignedDelta * effectiveScale).asType(currentWeight.dtype)
                appliedCount += 1
            }
        }

        debugLog?("Checked \(checkedCount) modules, found \(foundCount) with LoRA weights, applied \(appliedCount)")

        if !layerUpdates.isEmpty {
            encoder.update(parameters: ModuleParameters.unflattened(layerUpdates))
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
