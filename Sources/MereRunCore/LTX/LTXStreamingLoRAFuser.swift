import Foundation
import MLX
import MLXNN

final class LTXRuntimeLoRALinear: Linear {
    let loraDown: MLXArray
    let loraUp: MLXArray
    let strength: Float
    var isActive = false

    init(
        base: Linear,
        loraDown: MLXArray,
        loraUp: MLXArray,
        strength: Float
    ) {
        self.loraDown = loraDown.asType(base.weight.dtype)
        self.loraUp = loraUp.asType(base.weight.dtype)
        self.strength = strength
        super.init(weight: base.weight, bias: base.bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let baseOutput = super.callAsFunction(x)
        guard isActive else { return baseOutput }

        let input = x.asType(loraDown.dtype)
        let adapterOutput = MLX.matmul(
            MLX.matmul(input, loraDown.T),
            loraUp.T
        ) * MLXArray(strength).asType(loraDown.dtype)
        return baseOutput + adapterOutput.asType(baseOutput.dtype)
    }
}

final class LTXRuntimeLoRAAdapter {
    enum AdapterError: LocalizedError {
        case noPairs(URL)
        case unexpectedPairCount(expected: Int, actual: Int)
        case unsupportedTarget(String)
        case missingTargetModule(String)
        case targetIsNotLinear(String)
        case duplicateTarget(String)

        var errorDescription: String? {
            switch self {
            case .noPairs(let url):
                return "No LTX LoRA tensor pairs were found in \(url.path)."
            case .unexpectedPairCount(let expected, let actual):
                return "LTX runtime LoRA tensor-pair count mismatch: expected \(expected), found \(actual)."
            case .unsupportedTarget(let key):
                return "Unsupported LTX runtime LoRA target key: \(key)"
            case .missingTargetModule(let path):
                return "LTX runtime LoRA target module is missing: \(path)"
            case .targetIsNotLinear(let path):
                return "LTX runtime LoRA target is not Linear: \(path)"
            case .duplicateTarget(let path):
                return "LTX runtime LoRA contains a duplicate target: \(path)"
            }
        }
    }

    let pairCount: Int
    private let layers: [LTXRuntimeLoRALinear]

    private init(pairCount: Int, layers: [LTXRuntimeLoRALinear]) {
        self.pairCount = pairCount
        self.layers = layers
    }

    static func install(
        url: URL,
        into transformer: Module,
        strength: Float = 1,
        expectedPairCount: Int? = 1_660
    ) throws -> LTXRuntimeLoRAAdapter {
        let leafModules = transformer.leafModules().flattened()
        let modulesByPath = Dictionary(uniqueKeysWithValues: leafModules)
        var replacements: [String: Module] = [:]
        var layers: [LTXRuntimeLoRALinear] = []

        let pairCount = try SafetensorsStreamingLoader.forEachTensorPair(
            url: url,
            firstSuffix: ".lora_A.weight",
            secondSuffix: ".lora_B.weight"
        ) { baseKey, a, b in
            let sourceWeightKey = baseKey + ".weight"
            guard let targetKey = mapLTX23UnifiedTransformerKey(sourceWeightKey) else {
                throw AdapterError.unsupportedTarget(sourceWeightKey)
            }
            guard targetKey.hasSuffix(".weight") else {
                throw AdapterError.unsupportedTarget(targetKey)
            }
            let modulePath = String(targetKey.dropLast(".weight".count))
            guard replacements[modulePath] == nil else {
                throw AdapterError.duplicateTarget(modulePath)
            }
            guard let module = modulesByPath[modulePath] else {
                throw AdapterError.missingTargetModule(modulePath)
            }
            guard let linear = module as? Linear else {
                throw AdapterError.targetIsNotLinear(modulePath)
            }
            _ = try LTXStreamingLoRAFuser.fusedWeight(
                targetKey: targetKey,
                currentWeight: linear.weight,
                a: a,
                b: b,
                strength: strength
            )
            let layer = LTXRuntimeLoRALinear(
                base: linear,
                loraDown: a,
                loraUp: b,
                strength: strength
            )
            MLX.eval(layer.loraDown, layer.loraUp)
            replacements[modulePath] = layer
            layers.append(layer)
        }

        guard pairCount > 0 else {
            throw AdapterError.noPairs(url)
        }
        if let expectedPairCount, pairCount != expectedPairCount {
            throw AdapterError.unexpectedPairCount(expected: expectedPairCount, actual: pairCount)
        }

        applyModuleReplacements(replacements, leafModules: leafModules, to: transformer)
        Memory.clearCache()
        return LTXRuntimeLoRAAdapter(pairCount: pairCount, layers: layers)
    }

    func setActive(_ isActive: Bool) {
        for layer in layers {
            layer.isActive = isActive
        }
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
