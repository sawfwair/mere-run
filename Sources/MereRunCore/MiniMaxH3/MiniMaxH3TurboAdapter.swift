import Foundation
import MLX
import MLXNN

public enum MiniMaxH3TurboAdapter {
    public static let format = "minimax-h3-runtime-lora-v1"
    public static let expectedPairCount = 259
    public static let recommendedSchedulePointCount = 5

    enum AdapterError: LocalizedError {
        case noPairs(URL)
        case unexpectedPairCount(expected: Int, actual: Int)
        case missingTargetModule(String)
        case targetIsNotDenseLinear(String)
        case duplicateTarget(String)
        case invalidPairShape(String, a: [Int], b: [Int])
        case targetShapeMismatch(String, expected: [Int], actual: [Int])

        var errorDescription: String? {
            switch self {
            case .noPairs(let url):
                return "No MiniMax-H3 LoRA tensor pairs were found in \(url.path)."
            case .unexpectedPairCount(let expected, let actual):
                return "MiniMax-H3 LoRA tensor-pair count mismatch: expected \(expected), found \(actual)."
            case .missingTargetModule(let path):
                return "MiniMax-H3 LoRA target module is missing: \(path)"
            case .targetIsNotDenseLinear(let path):
                return "MiniMax-H3 Turbo currently requires the BF16 transformer; target \(path) is not a dense Linear layer."
            case .duplicateTarget(let path):
                return "MiniMax-H3 LoRA contains a duplicate target: \(path)"
            case .invalidPairShape(let path, let a, let b):
                return "Invalid MiniMax-H3 LoRA pair for \(path): A=\(a), B=\(b)."
            case .targetShapeMismatch(let path, let expected, let actual):
                return "MiniMax-H3 LoRA target \(path) has shape \(actual); expected \(expected)."
            }
        }
    }

    @discardableResult
    static func install(
        url: URL,
        into transformer: MiniMaxH3Transformer,
        strength: Float,
        expectedPairCount: Int? = expectedPairCount
    ) throws -> Int {
        let leafModules = transformer.leafModules().flattened()
        let modulesByPath = Dictionary(uniqueKeysWithValues: leafModules)
        var replacements: [String: Module] = [:]

        let pairCount = try SafetensorsStreamingLoader.forEachTensorPair(
            url: url,
            firstSuffix: ".lora_A.weight",
            secondSuffix: ".lora_B.weight"
        ) { modulePath, rawDown, rawUp in
            guard replacements[modulePath] == nil else {
                throw AdapterError.duplicateTarget(modulePath)
            }
            guard let module = modulesByPath[modulePath] else {
                throw AdapterError.missingTargetModule(modulePath)
            }
            guard let linear = module as? Linear,
                  !(linear is QuantizedLinear) else {
                throw AdapterError.targetIsNotDenseLinear(modulePath)
            }
            guard rawDown.ndim == 2,
                  rawUp.ndim == 2,
                  rawUp.dim(1) == rawDown.dim(0) else {
                throw AdapterError.invalidPairShape(
                    modulePath,
                    a: rawDown.shape,
                    b: rawUp.shape
                )
            }

            let up = modulePath.hasSuffix(".attn.qkv_proj")
                ? MiniMaxH3ModelLoader.deinterleavedQKVOutputRows(
                    rawUp,
                    headCount: transformer.configuration.attentionHeadCount,
                    headDimension: transformer.configuration.attentionHeadDimension
                )
                : rawUp
            guard rawDown.dim(1) == linear.shape.1,
                  up.dim(0) == linear.shape.0 else {
                throw AdapterError.targetShapeMismatch(
                    modulePath,
                    expected: [linear.shape.0, linear.shape.1],
                    actual: [up.dim(0), rawDown.dim(1)]
                )
            }

            let layer = MiniMaxH3RuntimeLoRALinear(
                base: linear,
                loraDown: rawDown,
                loraUp: up,
                strength: strength
            )
            MLX.eval(layer.loraDown, layer.loraUp)
            replacements[modulePath] = layer
        }

        guard pairCount > 0 else { throw AdapterError.noPairs(url) }
        if let expectedPairCount, pairCount != expectedPairCount {
            throw AdapterError.unexpectedPairCount(expected: expectedPairCount, actual: pairCount)
        }
        applyModuleReplacements(replacements, leafModules: leafModules, to: transformer)
        Memory.clearCache()
        return pairCount
    }

    private static func applyModuleReplacements(
        _ replacements: [String: Module],
        leafModules: [(String, Module)],
        to model: Module
    ) {
        var arrayUpdates: [String: [(Int, Module)]] = [:]
        var directUpdates: [(String, Module)] = []

        for (path, replacement) in replacements {
            let components = path.split(separator: ".")
            if let last = components.last, let index = Int(last) {
                let parentPath = components.dropLast().joined(separator: ".")
                arrayUpdates[parentPath, default: []].append((index, replacement))
            } else {
                directUpdates.append((path, replacement))
            }
        }

        var moduleUpdates = directUpdates
        for (parentPath, indexedReplacements) in arrayUpdates {
            let currentModules = leafModules.filter { path, _ in
                let parts = path.split(separator: ".")
                guard parts.count >= 2, Int(parts.last!) != nil else { return false }
                return parts.dropLast().joined(separator: ".") == parentPath
            }
            let replacementMap = Dictionary(uniqueKeysWithValues: indexedReplacements)
            for (modulePath, originalModule) in currentModules {
                let index = Int(modulePath.split(separator: ".").last!)!
                moduleUpdates.append((modulePath, replacementMap[index] ?? originalModule))
            }
        }
        model.update(modules: ModuleChildren.unflattened(moduleUpdates))
    }
}

final class MiniMaxH3RuntimeLoRALinear: Linear {
    @ParameterInfo(key: "lora_down") var loraDown: MLXArray
    @ParameterInfo(key: "lora_up") var loraUp: MLXArray
    let strength: Float

    init(base: Linear, loraDown: MLXArray, loraUp: MLXArray, strength: Float) {
        self._loraDown.wrappedValue = loraDown.asType(base.weight.dtype)
        self._loraUp.wrappedValue = loraUp.asType(base.weight.dtype)
        self.strength = strength
        super.init(weight: base.weight, bias: base.bias)
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let baseOutput = super.callAsFunction(input)
        let adapterOutput = MLX.matmul(
            MLX.matmul(input.asType(loraDown.dtype), loraDown.T),
            loraUp.T
        ) * MLXArray(strength).asType(loraDown.dtype)
        return baseOutput + adapterOutput.asType(baseOutput.dtype)
    }
}
