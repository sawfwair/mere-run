import Foundation
import MLX
import MLXNN

public enum Krea2LoRAInjectionError: Error, LocalizedError {
    case invalidRank(Int)
    case noMatchingLayers

    public var errorDescription: String? {
        switch self {
        case .invalidRank(let rank):
            return "LoRA rank must be >= 1 (got \(rank))."
        case .noMatchingLayers:
            return "No matching Krea 2 transformer Linear layers found for LoRA injection."
        }
    }
}

public enum Krea2LoRAInjector {
    public static let defaultTargetPrefixes: [String] = [
        "transformer_blocks.",
    ]

    public static let defaultTargetSuffixes: [String] = [
        ".attn.to_q",
        ".attn.to_k",
        ".attn.to_v",
        ".attn.to_out.0",
        ".ff.gate",
        ".ff.up",
        ".ff.down",
    ]

    public static let liteTargetSuffixes: [String] = [
        ".attn.to_q",
        ".attn.to_v",
    ]

    public static func inject(
        into transformer: Krea2Transformer,
        rank: Int,
        alpha: Float? = nil,
        targetPrefixes: [String]? = defaultTargetPrefixes,
        targetSuffixes: [String]? = defaultTargetSuffixes,
        targetRanks: [String: Int]? = nil,
        zeroInitUp: Bool = false
    ) throws -> [String: TrainableLoRALayer] {
        guard rank >= 1 else {
            throw Krea2LoRAInjectionError.invalidRank(rank)
        }
        if let targetRanks {
            for configuredRank in targetRanks.values where configuredRank < 1 {
                throw Krea2LoRAInjectionError.invalidRank(configuredRank)
            }
        }

        let shouldTarget: (String) -> Bool = { path in
            if let targetPrefixes {
                guard targetPrefixes.contains(where: { path.hasPrefix($0) }) else { return false }
            }
            guard let targetSuffixes else { return true }
            return targetSuffixes.contains { path.hasSuffix($0) }
        }

        let leafModules = transformer.leafModules().flattened()
        var replacements: [String: Module] = [:]
        var loraLayers: [String: TrainableLoRALayer] = [:]

        for (path, module) in leafModules {
            let resolvedRank: Int = {
                if let targetRanks {
                    guard let explicitRank = targetRanks[path] else { return 0 }
                    return explicitRank
                }
                guard shouldTarget(path) else { return 0 }
                return rank
            }()
            guard resolvedRank > 0 else { continue }
            let alphaValue = alpha ?? Float(resolvedRank)

            if let existingFused = module as? FusedLoRALinear {
                let newLoRA = LoRALinear(
                    base: Linear(weight: existingFused.weight, bias: existingFused.bias),
                    rank: resolvedRank,
                    alpha: alphaValue,
                    zeroInitUp: zeroInitUp
                )
                newLoRA.role = .train
                for lora in existingFused.loras where lora.role == .train {
                    lora.role = .assistant
                }
                existingFused.loras.append(newLoRA)
                loraLayers[path] = existingFused
                continue
            }

            if let loraQuantized = module as? LoRAQuantizedLinear {
                let fused = FusedLoRALinear(
                    existingLoRA: loraQuantized,
                    newRank: resolvedRank,
                    newAlpha: alphaValue,
                    zeroInitUp: zeroInitUp
                )
                replacements[path] = fused
                loraLayers[path] = fused
                continue
            }

            if let loraLinear = module as? LoRALinear {
                let fused = FusedLoRALinear(
                    existingLoRA: loraLinear,
                    newRank: resolvedRank,
                    newAlpha: alphaValue,
                    zeroInitUp: zeroInitUp
                )
                replacements[path] = fused
                loraLayers[path] = fused
                continue
            }

            if let quantized = module as? QuantizedLinear {
                let wrapped = LoRAQuantizedLinear(
                    base: quantized,
                    rank: resolvedRank,
                    alpha: alphaValue,
                    zeroInitUp: zeroInitUp
                )
                replacements[path] = wrapped
                loraLayers[path] = wrapped
                continue
            }

            if let linear = module as? Linear {
                let wrapped = LoRALinear(
                    base: linear,
                    rank: resolvedRank,
                    alpha: alphaValue,
                    zeroInitUp: zeroInitUp
                )
                replacements[path] = wrapped
                loraLayers[path] = wrapped
            }
        }

        guard !loraLayers.isEmpty else {
            throw Krea2LoRAInjectionError.noMatchingLayers
        }

        applyModuleReplacements(replacements, leafModules: leafModules, to: transformer)
        return loraLayers
    }

    @discardableResult
    public static func loadWeights(
        from url: URL,
        into loraLayers: [String: TrainableLoRALayer],
        optimizerStateURL: URL? = nil,
        debug: Bool = false
    ) throws -> Int {
        try ZImageLoRAInjector.loadWeights(
            from: url,
            into: loraLayers,
            optimizerStateURL: optimizerStateURL,
            debug: debug
        )
    }

    @discardableResult
    public static func applyWeights(
        _ loraWeights: LoRAWeights,
        to loraLayers: [String: TrainableLoRALayer],
        debug: Bool = false
    ) -> Int {
        ZImageLoRAInjector.applyWeights(loraWeights, to: loraLayers, debug: debug)
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

        var moduleUpdates: [(String, Module)] = directUpdates

        for (parentPath, indexedReplacements) in arrayUpdates {
            let currentModules = leafModules.filter { path, _ in
                let parts = path.split(separator: ".")
                guard parts.count >= 2 else { return false }
                guard Int(parts.last!) != nil else { return false }
                let parent = parts.dropLast().joined(separator: ".")
                return parent == parentPath
            }

            var replacementMap: [Int: Module] = [:]
            for (index, replacement) in indexedReplacements {
                replacementMap[index] = replacement
            }

            for (modulePath, originalModule) in currentModules {
                let index = Int(modulePath.split(separator: ".").last!)!
                let resolved = replacementMap[index] ?? originalModule
                moduleUpdates.append((modulePath, resolved))
            }
        }

        guard !moduleUpdates.isEmpty else { return }
        model.update(modules: ModuleChildren.unflattened(moduleUpdates))
    }
}
