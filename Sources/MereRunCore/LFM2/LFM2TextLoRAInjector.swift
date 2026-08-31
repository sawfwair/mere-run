import Foundation
import MLXNN

public enum LFM2TextLoRAInjectionError: Error, LocalizedError, Sendable {
    case invalidRank(Int)
    case noMatchingLayers([String])

    public var errorDescription: String? {
        switch self {
        case .invalidRank(let rank):
            return "LoRA rank must be >= 1 (got \(rank))."
        case .noMatchingLayers(let suffixes):
            return "No matching LFM2 attention Linear layers found for LoRA injection. "
                + "Target suffixes: \(suffixes.joined(separator: ","))"
        }
    }
}

public enum LFM2TextLoRAInjector {
    public static let defaultTargetSuffixes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
    ]

    public static func inject(
        into model: Module,
        rank: Int,
        alpha: Float? = nil,
        targetSuffixes: [String] = defaultTargetSuffixes,
        zeroInitUp: Bool = true
    ) throws -> [String: TrainableLoRALayer] {
        guard rank >= 1 else {
            throw LFM2TextLoRAInjectionError.invalidRank(rank)
        }
        let suffixes = targetSuffixes
            .map(canonicalTargetSuffix)
            .filter { !$0.isEmpty }
        let alphaValue = alpha ?? Float(rank)

        if let lfm2Model = model as? LFM2Model {
            return try injectIntoNativeModel(
                lfm2Model,
                rank: rank,
                alpha: alphaValue,
                suffixes: suffixes,
                zeroInitUp: zeroInitUp
            )
        }

        var replacements: [String: Module] = [:]
        var loraLayers: [String: TrainableLoRALayer] = [:]

        for (path, module) in model.leafModules().flattened() {
            guard shouldTarget(path: path, suffixes: suffixes) else { continue }
            if let injected = makeLoRALayer(
                from: module,
                rank: rank,
                alpha: alphaValue,
                zeroInitUp: zeroInitUp
            ) {
                if let replacement = injected.replacement {
                    replacements[path] = replacement
                }
                loraLayers[path] = injected.layer
            }
        }

        guard !loraLayers.isEmpty else {
            throw LFM2TextLoRAInjectionError.noMatchingLayers(suffixes)
        }
        if !replacements.isEmpty {
            let updates = replacements.map { (path: $0.key, module: $0.value) }
            model.update(modules: ModuleChildren.unflattened(updates))
        }
        return loraLayers
    }

    private static func injectIntoNativeModel(
        _ model: LFM2Model,
        rank: Int,
        alpha: Float,
        suffixes: [String],
        zeroInitUp: Bool
    ) throws -> [String: TrainableLoRALayer] {
        var loraLayers: [String: TrainableLoRALayer] = [:]

        for (layerIndex, decoderLayer) in model.model.layers.enumerated() {
            guard let attention = decoderLayer.selfAttention else { continue }
            var replacements: [String: Module] = [:]
            for (localPath, module) in attention.leafModules().flattened() {
                let fullPath = "model.layers.\(layerIndex).self_attn.\(localPath)"
                guard shouldTarget(path: fullPath, suffixes: suffixes) else { continue }
                guard let injected = makeLoRALayer(
                    from: module,
                    rank: rank,
                    alpha: alpha,
                    zeroInitUp: zeroInitUp
                ) else { continue }
                if let replacement = injected.replacement {
                    replacements[localPath] = replacement
                }
                loraLayers[fullPath] = injected.layer
            }
            if !replacements.isEmpty {
                let updates = replacements.map { (path: $0.key, module: $0.value) }
                attention.update(modules: ModuleChildren.unflattened(updates))
            }
        }

        guard !loraLayers.isEmpty else {
            throw LFM2TextLoRAInjectionError.noMatchingLayers(suffixes)
        }
        return loraLayers
    }

    private static func makeLoRALayer(
        from module: Module,
        rank: Int,
        alpha: Float,
        zeroInitUp: Bool
    ) -> (replacement: Module?, layer: TrainableLoRALayer)? {
        if let fused = module as? FusedLoRALinear {
            let newLoRA = LoRALinear(
                base: Linear(weight: fused.weight, bias: fused.bias),
                rank: rank,
                alpha: alpha,
                zeroInitUp: zeroInitUp
            )
            newLoRA.role = .train
            for lora in fused.loras where lora.role == .train {
                lora.role = .assistant
            }
            fused.loras.append(newLoRA)
            return (nil, fused)
        }
        if let existing = module as? LoRAQuantizedLinear {
            let fused = FusedLoRALinear(
                existingLoRA: existing,
                newRank: rank,
                newAlpha: alpha,
                zeroInitUp: zeroInitUp
            )
            return (fused, fused)
        }
        if let existing = module as? LoRALinear {
            let fused = FusedLoRALinear(
                existingLoRA: existing,
                newRank: rank,
                newAlpha: alpha,
                zeroInitUp: zeroInitUp
            )
            return (fused, fused)
        }
        if let quantized = module as? QuantizedLinear {
            let wrapped = LoRAQuantizedLinear(
                base: quantized,
                rank: rank,
                alpha: alpha,
                zeroInitUp: zeroInitUp
            )
            return (wrapped, wrapped)
        }
        if let linear = module as? Linear {
            let wrapped = LoRALinear(
                base: linear,
                rank: rank,
                alpha: alpha,
                zeroInitUp: zeroInitUp
            )
            return (wrapped, wrapped)
        }
        return nil
    }

    static func shouldTarget(path: String, suffixes: [String]) -> Bool {
        guard path.hasPrefix("self_attn.") || path.contains(".self_attn.") else {
            return false
        }
        return suffixes.map(canonicalTargetSuffix).contains { suffix in
            path == suffix || path.hasSuffix(".\(suffix)")
        }
    }

    private static func canonicalTargetSuffix(_ value: String) -> String {
        let suffix = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if suffix == "o_proj" {
            return "out_proj"
        }
        if suffix.hasSuffix(".o_proj") {
            return String(suffix.dropLast("o_proj".count)) + "out_proj"
        }
        return suffix
    }
}
