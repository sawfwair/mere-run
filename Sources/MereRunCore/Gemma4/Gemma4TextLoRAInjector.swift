import Foundation
import MLX
import MLXNN

public enum Gemma4TextLoRAInjectionError: Error, LocalizedError {
    case invalidRank(Int)
    case noMatchingLayers([String])

    public var errorDescription: String? {
        switch self {
        case .invalidRank(let rank):
            return "LoRA rank must be >= 1 (got \(rank))."
        case .noMatchingLayers(let suffixes):
            return "No matching Gemma4 text Linear layers found for LoRA injection. Target suffixes: \(suffixes.joined(separator: ","))"
        }
    }
}
public enum Gemma4TextLoRAInjector {
    public static let defaultTargetSuffixes: [String] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]

    public static func inject(
        into model: Module,
        rank: Int,
        alpha: Float? = nil,
        targetSuffixes: [String] = defaultTargetSuffixes,
        zeroInitUp: Bool = true
    ) throws -> [String: TrainableLoRALayer] {
        guard rank >= 1 else {
            throw Gemma4TextLoRAInjectionError.invalidRank(rank)
        }
        let normalizedSuffixes = targetSuffixes
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        let leafModules = model.leafModules().flattened()
        var replacements: [String: Module] = [:]
        var loraLayers: [String: TrainableLoRALayer] = [:]

        for (path, module) in leafModules {
            guard shouldTarget(path: path, suffixes: normalizedSuffixes) else { continue }
            let alphaValue = alpha ?? Float(rank)

            if let existingFused = module as? FusedLoRALinear {
                let newLoRA = LoRALinear(
                    base: Linear(weight: existingFused.weight, bias: existingFused.bias),
                    rank: rank,
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
                    newRank: rank,
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
                    newRank: rank,
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
                    rank: rank,
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
                    rank: rank,
                    alpha: alphaValue,
                    zeroInitUp: zeroInitUp
                )
                replacements[path] = wrapped
                loraLayers[path] = wrapped
            }
        }

        guard !loraLayers.isEmpty else {
            throw Gemma4TextLoRAInjectionError.noMatchingLayers(normalizedSuffixes)
        }

        applyModuleReplacements(replacements, to: model)
        return loraLayers
    }

    static func shouldTarget(path: String, suffixes: [String]) -> Bool {
        suffixes.contains { suffix in
            path == suffix || path.hasSuffix(".\(suffix)")
        }
    }

    private static func applyModuleReplacements(_ replacements: [String: Module], to model: Module) {
        guard !replacements.isEmpty else { return }
        let moduleUpdates = replacements.map { (path: $0.key, module: $0.value) }
        model.update(modules: ModuleChildren.unflattened(moduleUpdates))
    }
}
