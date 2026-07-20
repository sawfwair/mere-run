import Foundation
import MLX
import MLXNN

public enum SCAIL2DistilledAdapter {
    public static let format = "scail2-distilled-wan-i2v-lora-v1"

    public struct ApplicationResult: Hashable, Sendable {
        public let pairCount: Int
        public let differenceCount: Int
        public let skippedDifferenceCount: Int

        public init(
            pairCount: Int,
            differenceCount: Int,
            skippedDifferenceCount: Int = 0
        ) {
            self.pairCount = pairCount
            self.differenceCount = differenceCount
            self.skippedDifferenceCount = skippedDifferenceCount
        }
    }

    public enum AdapterError: LocalizedError {
        case noPairs(URL)
        case unexpectedPairCount(expected: Int, actual: Int)
        case unexpectedDifferenceCount(expected: Int, actual: Int)
        case unsupportedTarget(String)
        case missingTargetParameter(String)
        case invalidPairShape(String, down: [Int], up: [Int])
        case targetShapeMismatch(String, expected: [Int], actual: [Int])

        public var errorDescription: String? {
            switch self {
            case .noPairs(let url):
                return "No SCAIL-2 distilled adapter tensor pairs were found in \(url.path)."
            case .unexpectedPairCount(let expected, let actual):
                return "SCAIL-2 distilled adapter pair count mismatch: expected \(expected), found \(actual)."
            case .unexpectedDifferenceCount(let expected, let actual):
                return "SCAIL-2 distilled adapter difference count mismatch: expected \(expected), found \(actual)."
            case .unsupportedTarget(let key):
                return "Unsupported SCAIL-2 distilled adapter target: \(key)"
            case .missingTargetParameter(let key):
                return "SCAIL-2 distilled adapter target is missing from the transformer: \(key)"
            case .invalidPairShape(let key, let down, let up):
                return "Invalid SCAIL-2 distilled adapter pair for \(key): down=\(down), up=\(up)."
            case .targetShapeMismatch(let key, let expected, let actual):
                return "SCAIL-2 distilled adapter delta for \(key) has shape \(actual); expected \(expected)."
            }
        }
    }

    /// Applies a distilled Wan I2V adapter directly to the resident native
    /// transformer. The checkpoint remains a separately pulled model artifact;
    /// no adapter runtime or source code is imported.
    @discardableResult
    public static func apply(
        url: URL,
        to transformer: SCAIL2TransformerModel,
        strength: Float = 1,
        expectedPairCount: Int? = 487,
        expectedDifferenceCount: Int? = 775
    ) throws -> ApplicationResult {
        var parameters = Dictionary(uniqueKeysWithValues: transformer.parameters().flattened())
        let modulesByPath = Dictionary(uniqueKeysWithValues: transformer.leafModules().flattened())
        var appliedPairs = 0

        let pairCount = try SafetensorsStreamingLoader.forEachTensorPair(
            url: url,
            firstSuffix: ".lora_down.weight",
            secondSuffix: ".lora_up.weight"
        ) { sourceBaseKey, down, up in
            guard let targetBaseKey = targetBaseKey(for: sourceBaseKey) else {
                throw AdapterError.unsupportedTarget(sourceBaseKey)
            }
            let targetKey = targetBaseKey + ".weight"
            guard let current = parameters[targetKey] else {
                throw AdapterError.missingTargetParameter(targetKey)
            }
            let fused = try fusedWeight(
                targetKey: targetKey,
                currentWeight: current,
                down: down,
                up: up,
                strength: strength
            )
            MLX.eval(fused)
            try update(
                targetKey: targetKey,
                value: fused,
                modulesByPath: modulesByPath
            )
            parameters[targetKey] = fused
            appliedPairs += 1
            if appliedPairs.isMultiple(of: 8) {
                Memory.clearCache()
            }
        }

        guard pairCount > 0 else {
            throw AdapterError.noPairs(url)
        }
        if let expectedPairCount, pairCount != expectedPairCount {
            throw AdapterError.unexpectedPairCount(expected: expectedPairCount, actual: pairCount)
        }

        let differences = try SafetensorsStreamingLoader.loadArrays(url: url) {
            $0.hasSuffix(".diff") || $0.hasSuffix(".diff_b")
        }
        if let expectedDifferenceCount, differences.count != expectedDifferenceCount {
            throw AdapterError.unexpectedDifferenceCount(
                expected: expectedDifferenceCount,
                actual: differences.count
            )
        }

        var appliedDifferences = 0
        var skippedDifferences = 0
        for sourceKey in differences.keys.sorted() {
            guard let difference = differences[sourceKey],
                  let targetKey = differenceTargetKey(for: sourceKey) else {
                throw AdapterError.unsupportedTarget(sourceKey)
            }
            guard let current = parameters[targetKey] else {
                throw AdapterError.missingTargetParameter(targetKey)
            }
            if isTaskInputProjectionDifference(
                sourceKey: sourceKey,
                sourceShape: difference.shape,
                targetShape: current.shape
            ) {
                skippedDifferences += 1
                continue
            }
            guard current.shape == difference.shape else {
                throw AdapterError.targetShapeMismatch(
                    targetKey,
                    expected: current.shape,
                    actual: difference.shape
                )
            }
            let fused = (
                current.asType(.float32)
                    + difference.asType(.float32) * MLXArray(strength)
            ).asType(current.dtype)
            MLX.eval(fused)
            try update(
                targetKey: targetKey,
                value: fused,
                modulesByPath: modulesByPath
            )
            parameters[targetKey] = fused
            appliedDifferences += 1
        }
        Memory.clearCache()
        return ApplicationResult(
            pairCount: appliedPairs,
            differenceCount: appliedDifferences,
            skippedDifferenceCount: skippedDifferences
        )
    }

    static func isTaskInputProjectionDifference(
        sourceKey: String,
        sourceShape: [Int],
        targetShape: [Int]
    ) -> Bool {
        sourceKey == "diffusion_model.patch_embedding.diff"
            && sourceShape == [5_120, 36, 1, 2, 2]
            && targetShape == [5_120, 80]
    }

    static func fusedWeight(
        targetKey: String,
        currentWeight: MLXArray,
        down: MLXArray,
        up: MLXArray,
        strength: Float
    ) throws -> MLXArray {
        guard down.ndim == 2,
              up.ndim == 2,
              up.dim(1) == down.dim(0) else {
            throw AdapterError.invalidPairShape(
                targetKey,
                down: down.shape,
                up: up.shape
            )
        }
        var delta = MLX.matmul(
            up.asType(.float32) * MLXArray(strength),
            down.asType(.float32)
        )
        if delta.shape != currentWeight.shape,
           delta.T.shape == currentWeight.shape {
            delta = delta.T
        }
        guard delta.shape == currentWeight.shape else {
            throw AdapterError.targetShapeMismatch(
                targetKey,
                expected: currentWeight.shape,
                actual: delta.shape
            )
        }
        return (currentWeight.asType(.float32) + delta)
            .asType(currentWeight.dtype)
    }

    static func targetBaseKey(for sourceKey: String) -> String? {
        let prefix = "diffusion_model."
        guard sourceKey.hasPrefix(prefix) else { return nil }
        var components = sourceKey
            .dropFirst(prefix.count)
            .split(separator: ".")
            .map(String.init)
        guard let first = components.first else { return nil }

        switch first {
        case "blocks":
            guard components.count >= 3 else { return nil }
            if components.count >= 4, components[2] == "ffn" {
                switch components[3] {
                case "0": components[3] = "fc1"
                case "2": components[3] = "fc2"
                default: return nil
                }
            }
        case "img_emb":
            guard components.count == 3,
                  components[1] == "proj",
                  ["0", "1", "3", "4"].contains(components[2]) else {
                return nil
            }
            components = ["img_emb", "layer_\(components[2])"]
        case "text_embedding":
            guard components.count == 2 else { return nil }
            switch components[1] {
            case "0": components = ["text_embedding_0"]
            case "2": components = ["text_embedding_1"]
            default: return nil
            }
        case "time_embedding":
            guard components.count == 2 else { return nil }
            switch components[1] {
            case "0": components = ["time_embedding_0"]
            case "2": components = ["time_embedding_1"]
            default: return nil
            }
        case "time_projection":
            guard components == ["time_projection", "1"] else { return nil }
            components = ["time_projection"]
        case "patch_embedding":
            guard components.count == 1 else { return nil }
            components = ["patch_embedding_proj"]
        case "head":
            guard components == ["head", "head"] else { return nil }
        default:
            return nil
        }
        return components.joined(separator: ".")
    }

    static func differenceTargetKey(for sourceKey: String) -> String? {
        let suffix: String
        let parameter: String
        if sourceKey.hasSuffix(".diff_b") {
            suffix = ".diff_b"
            parameter = ".bias"
        } else if sourceKey.hasSuffix(".diff") {
            suffix = ".diff"
            parameter = ".weight"
        } else {
            return nil
        }
        let sourceBaseKey = String(sourceKey.dropLast(suffix.count))
        return targetBaseKey(for: sourceBaseKey).map { $0 + parameter }
    }

    private static func update(
        targetKey: String,
        value: MLXArray,
        modulesByPath: [String: Module]
    ) throws {
        let components = targetKey.split(separator: ".")
        guard components.count >= 2,
              let parameterName = components.last else {
            throw AdapterError.unsupportedTarget(targetKey)
        }
        let modulePath = components.dropLast().joined(separator: ".")
        guard let module = modulesByPath[modulePath] else {
            throw AdapterError.missingTargetParameter(targetKey)
        }
        try module.update(
            parameters: ModuleParameters.unflattened([
                (String(parameterName), value),
            ]),
            verify: .none
        )
    }
}
