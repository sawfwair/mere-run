import Foundation
import MLX
import MLXNN

enum LTXStreamingLoRAFuser {
    enum FusionError: LocalizedError {
        case noPairs(URL)
        case unexpectedPairCount(expected: Int, actual: Int)
        case unsupportedTarget(String)
        case missingTargetParameter(String)
        case invalidPairShape(String, a: [Int], b: [Int])
        case targetShapeMismatch(String, expected: [Int], actual: [Int])

        var errorDescription: String? {
            switch self {
            case .noPairs(let url):
                return "No LTX LoRA tensor pairs were found in \(url.path)."
            case .unexpectedPairCount(let expected, let actual):
                return "LTX LoRA tensor-pair count mismatch: expected \(expected), found \(actual)."
            case .unsupportedTarget(let key):
                return "Unsupported LTX LoRA target key: \(key)"
            case .missingTargetParameter(let key):
                return "LTX LoRA target parameter is missing from the transformer: \(key)"
            case .invalidPairShape(let key, let a, let b):
                return "Invalid LTX LoRA pair for \(key): A=\(a), B=\(b)."
            case .targetShapeMismatch(let key, let expected, let actual):
                return "LTX LoRA delta for \(key) has shape \(actual); expected \(expected)."
            }
        }
    }

    /// Fuses the official mixed-rank distilled LoRA one A/B pair at a time.
    /// Only a pair, its delta, and one replacement base weight are transiently
    /// resident; the 7.6 GB adapter is never loaded into an in-memory dictionary.
    @discardableResult
    static func fuse(
        url: URL,
        into transformer: Module,
        strength: Float = 1,
        expectedPairCount: Int? = 1_660,
        debugOutputPrefix: URL? = nil
    ) throws -> Int {
        var parameters = Dictionary(uniqueKeysWithValues: transformer.parameters().flattened())
        var appliedCount = 0

        let pairCount = try SafetensorsStreamingLoader.forEachTensorPair(
            url: url,
            firstSuffix: ".lora_A.weight",
            secondSuffix: ".lora_B.weight"
        ) { baseKey, a, b in
            let sourceWeightKey = baseKey + ".weight"
            guard let targetKey = mapLTX23UnifiedTransformerKey(sourceWeightKey) else {
                throw FusionError.unsupportedTarget(sourceWeightKey)
            }
            guard let currentWeight = parameters[targetKey] else {
                throw FusionError.missingTargetParameter(targetKey)
            }
            let fused = try fusedWeight(
                targetKey: targetKey,
                currentWeight: currentWeight,
                a: a,
                b: b,
                strength: strength
            )
            MLX.eval(fused)
            try transformer.update(
                parameters: ModuleParameters.unflattened([(targetKey, fused)]),
                verify: .none
            )
            parameters[targetKey] = fused
            if let debugOutputPrefix,
               let suffix = debugSampleSuffix(for: targetKey) {
                let parent = debugOutputPrefix.deletingLastPathComponent()
                try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
                let outputURL = parent.appendingPathComponent(
                    "\(debugOutputPrefix.lastPathComponent)_\(suffix).npy",
                    isDirectory: false
                )
                try MLX.save(array: fused, url: outputURL)
            }
            appliedCount += 1
            if appliedCount.isMultiple(of: 8) {
                Memory.clearCache()
            }
        }

        guard pairCount > 0 else {
            throw FusionError.noPairs(url)
        }
        if let expectedPairCount, pairCount != expectedPairCount {
            throw FusionError.unexpectedPairCount(expected: expectedPairCount, actual: pairCount)
        }
        Memory.clearCache()
        return appliedCount
    }

    static func fusedWeight(
        targetKey: String,
        currentWeight: MLXArray,
        a: MLXArray,
        b: MLXArray,
        strength: Float
    ) throws -> MLXArray {
        guard a.ndim == 2, b.ndim == 2, b.dim(1) == a.dim(0) else {
            throw FusionError.invalidPairShape(targetKey, a: a.shape, b: b.shape)
        }

        var delta = MLX.matmul(
            b.asType(.float32) * MLXArray(strength),
            a.asType(.float32)
        )
        if delta.shape != currentWeight.shape, delta.T.shape == currentWeight.shape {
            delta = delta.T
        }
        guard delta.shape == currentWeight.shape else {
            throw FusionError.targetShapeMismatch(
                targetKey,
                expected: currentWeight.shape,
                actual: delta.shape
            )
        }

        return (currentWeight.asType(.float32) + delta)
            .asType(currentWeight.dtype)
    }

    private static func debugSampleSuffix(for targetKey: String) -> String? {
        switch targetKey {
        case "transformer_blocks.0.audio_attn1.to_gate_logits.weight":
            return "a2vid_lora_rank32_gate"
        case "audio_patchify_proj.weight":
            return "a2vid_lora_rank128_audio_patchify"
        default:
            return nil
        }
    }
}
