import Foundation
import MLXNN

public enum LagunaTextLoRAInjectionError: Error, LocalizedError, Sendable {
    case invalidRank(Int)
    case noMatchingLayers([String])

    public var errorDescription: String? {
        switch self {
        case .invalidRank(let rank):
            return "LoRA rank must be >= 1 (got \(rank))."
        case .noMatchingLayers(let suffixes):
            return "No matching Laguna text Linear layers found for LoRA injection. Target suffixes: \(suffixes.joined(separator: ","))"
        }
    }
}

public enum LagunaTextLoRAInjector {
    public static let defaultTargetSuffixes = [
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
            throw LagunaTextLoRAInjectionError.invalidRank(rank)
        }
        let suffixes = targetSuffixes
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let alphaValue = alpha ?? Float(rank)
        var replacements: [String: Module] = [:]
        var loraLayers: [String: TrainableLoRALayer] = [:]

        for (path, module) in model.leafModules().flattened() {
            guard shouldTarget(path: path, suffixes: suffixes) else { continue }

            if let fused = module as? FusedLoRALinear {
                let newLoRA = LoRALinear(
                    base: Linear(weight: fused.weight, bias: fused.bias),
                    rank: rank,
                    alpha: alphaValue,
                    zeroInitUp: zeroInitUp
                )
                newLoRA.role = .train
                for lora in fused.loras where lora.role == .train {
                    lora.role = .assistant
                }
                fused.loras.append(newLoRA)
                loraLayers[path] = fused
                continue
            }

            if let existing = module as? LoRAQuantizedLinear {
                let fused = FusedLoRALinear(
                    existingLoRA: existing,
                    newRank: rank,
                    newAlpha: alphaValue,
                    zeroInitUp: zeroInitUp
                )
                replacements[path] = fused
                loraLayers[path] = fused
                continue
            }

            if let existing = module as? LoRALinear {
                let fused = FusedLoRALinear(
                    existingLoRA: existing,
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
            throw LagunaTextLoRAInjectionError.noMatchingLayers(suffixes)
        }
        if !replacements.isEmpty {
            let updates = replacements.map { (path: $0.key, module: $0.value) }
            model.update(modules: ModuleChildren.unflattened(updates))
        }
        return loraLayers
    }

    static func shouldTarget(path: String, suffixes: [String]) -> Bool {
        suffixes.contains { suffix in
            path == suffix || path.hasSuffix(".\(suffix)")
        }
    }
}
