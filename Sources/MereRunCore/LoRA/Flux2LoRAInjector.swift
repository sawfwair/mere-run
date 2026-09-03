import Foundation
import MLX
import MLXNN

public enum Flux2LoRAInjectionError: Error, LocalizedError {
    case invalidRank(Int)
    case noMatchingLayers
    case weightLoadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRank(let rank):
            return "LoRA rank must be >= 1 (got \(rank))."
        case .noMatchingLayers:
            return "No matching Linear layers found to inject LoRA into."
        case .weightLoadFailed(let path):
            return "Failed to load LoRA weights from: \(path)"
        }
    }
}

public enum Flux2LoRAInjector {
    public enum TargetMode: String, Sendable {
        case suffix
        case transformerLinearWalk = "transformer-linear-walk"
    }

    public static let defaultTargetSuffixes: [String] = [
        // Joint blocks - attention
        ".attn.to_q",
        ".attn.to_k",
        ".attn.to_v",
        ".attn.to_out.0",
        ".attn.add_q_proj",
        ".attn.add_k_proj",
        ".attn.add_v_proj",
        ".attn.to_add_out",

        // Joint blocks - MLPs
        ".ff.linear_in",
        ".ff.linear_out",
        ".ff_context.linear_in",
        ".ff_context.linear_out",

        // Single blocks - fused attention+MLP
        ".attn.to_qkv_mlp_proj",
        ".attn.to_out",
    ]

    public static func inject(
        into transformer: Module,
        rank: Int,
        alpha: Float? = nil,
        targetMode: TargetMode = .suffix,
        targetSuffixes: [String] = defaultTargetSuffixes,
        targetRanks: [String: Int]? = nil,
        zeroInitUp: Bool = false,
        allowReinjection: Bool = false
    ) throws -> [String: TrainableLoRALayer] {
        guard rank >= 1 else {
            throw Flux2LoRAInjectionError.invalidRank(rank)
        }
        if let targetRanks {
            for configuredRank in targetRanks.values where configuredRank < 1 {
                throw Flux2LoRAInjectionError.invalidRank(configuredRank)
            }
        }

        let shouldTarget: (String, Module) -> Bool = { path, module in
            guard module is Linear || module is QuantizedLinear else { return false }
            switch targetMode {
            case .suffix:
                return targetSuffixes.contains { path.hasSuffix($0) }
            case .transformerLinearWalk:
                return true
            }
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
                guard shouldTarget(path, module) else { return 0 }
                return rank
            }()
            guard resolvedRank > 0 else { continue }
            let alphaValue = alpha ?? Float(resolvedRank)

            // Handle stacking: if already a LoRA layer and allowReinjection is true, create FusedLoRALinear
            if let existingFused = module as? FusedLoRALinear {
                if allowReinjection {
                    // Add a new train LoRA to the existing fused layer
                    let newLoRA = LoRALinear(
                        base: Linear(weight: existingFused.weight, bias: existingFused.bias),
                        rank: resolvedRank,
                        alpha: alphaValue,
                        zeroInitUp: zeroInitUp
                    )
                    newLoRA.role = .train
                    // Mark existing train LoRAs as assistant
                    for lora in existingFused.loras where lora.role == .train {
                        lora.role = .assistant
                    }
                    existingFused.loras.append(newLoRA)
                    loraLayers[path] = existingFused
                }
                continue
            }

            if let loraQuantized = module as? LoRAQuantizedLinear {
                if allowReinjection {
                    // Stack: create FusedLoRALinear with existing as assistant
                    let fused = FusedLoRALinear(existingLoRA: loraQuantized, newRank: resolvedRank, newAlpha: alphaValue, zeroInitUp: zeroInitUp)
                    replacements[path] = fused
                    loraLayers[path] = fused
                }
                continue
            }

            if let loraLinear = module as? LoRALinear {
                if allowReinjection {
                    // Stack: create FusedLoRALinear with existing as assistant
                    let fused = FusedLoRALinear(existingLoRA: loraLinear, newRank: resolvedRank, newAlpha: alphaValue, zeroInitUp: zeroInitUp)
                    replacements[path] = fused
                    loraLayers[path] = fused
                }
                continue
            }

            if let quantized = module as? QuantizedLinear {
                let wrapped = LoRAQuantizedLinear(base: quantized, rank: resolvedRank, alpha: alphaValue, zeroInitUp: zeroInitUp)
                replacements[path] = wrapped
                loraLayers[path] = wrapped
                continue
            }

            if let linear = module as? Linear {
                let wrapped = LoRALinear(base: linear, rank: resolvedRank, alpha: alphaValue, zeroInitUp: zeroInitUp)
                replacements[path] = wrapped
                loraLayers[path] = wrapped
                continue
            }
        }

        guard !loraLayers.isEmpty else {
            throw Flux2LoRAInjectionError.noMatchingLayers
        }

        applyModuleReplacements(
            replacements,
            leafModules: leafModules,
            to: transformer
        )

        return loraLayers
    }

    public static func resolveTargetRanks(
        in transformer: Module,
        defaultRank: Int,
        targetSuffixes: [String] = defaultTargetSuffixes,
        targetRankSuffixes: [String: Int]
    ) throws -> [String: Int] {
        guard defaultRank >= 1 else {
            throw Flux2LoRAInjectionError.invalidRank(defaultRank)
        }
        for rank in targetRankSuffixes.values where rank < 1 {
            throw Flux2LoRAInjectionError.invalidRank(rank)
        }

        let rankedSuffixes = targetRankSuffixes.keys.sorted { lhs, rhs in lhs.count > rhs.count }
        let leafModules = transformer.leafModules().flattened()
        var resolved: [String: Int] = [:]

        for (path, module) in leafModules {
            guard module is Linear || module is QuantizedLinear else { continue }
            if let suffix = rankedSuffixes.first(where: { path.hasSuffix($0) }) {
                resolved[path] = targetRankSuffixes[suffix]
            } else if targetSuffixes.contains(where: { path.hasSuffix($0) }) {
                resolved[path] = defaultRank
            }
        }

        guard !resolved.isEmpty else {
            throw Flux2LoRAInjectionError.noMatchingLayers
        }
        return resolved
    }

    private static func applyModuleReplacements(
        _ replacements: [String: Module],
        leafModules: [(String, Module)],
        to model: Module
    ) {
        // MLX's `update(modules:)` requires complete array structures when updating array elements.
        // If we replace an element inside an array/tuple (e.g. `to_out.0`), we must provide all
        // elements for that array.
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
            let currentModules = leafModules.filter { (path, _) in
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

    /// Load saved LoRA weights into already-injected layers.
    /// - Parameters:
    ///   - url: Path to the safetensors file containing LoRA weights
    ///   - loraLayers: The injected LoRA layers to update (from `inject()`)
    /// - Returns: Number of layers successfully updated
    @discardableResult
    public static func loadWeights(
        from url: URL,
        into loraLayers: [String: TrainableLoRALayer],
        optimizerStateURL: URL? = nil
    ) throws -> Int {
        let weights = try MLX.loadArrays(url: url)
        let externalOptimizerWeights: [String: MLXArray]? = {
            guard let optimizerStateURL else { return nil }
            return try? MLX.loadArrays(url: optimizerStateURL)
        }()
        var updatedCount = 0

        for (path, layer) in loraLayers {
            guard let candidate = matchingWeightCandidate(for: path, in: weights),
                  let downWeight = weights[candidate.down],
                  let upWeight = weights[candidate.up] else {
                continue
            }

            // Update the layer's weights
            layer.loraDown = downWeight.asType(.float32)
            layer.loraUp = upWeight.asType(.float32)

            // Load optimizer state if present (for resumable training)
            let optimizerCandidates = optimizerKeyCandidates(for: path)
            if let downM = firstArray(
                for: optimizerCandidates.map(\.downM),
                primary: weights,
                secondary: externalOptimizerWeights
            ) {
                layer.loraDownM = downM.asType(.float32)
            }
            if let downV = firstArray(
                for: optimizerCandidates.map(\.downV),
                primary: weights,
                secondary: externalOptimizerWeights
            ) {
                layer.loraDownV = downV.asType(.float32)
            }
            if let upM = firstArray(
                for: optimizerCandidates.map(\.upM),
                primary: weights,
                secondary: externalOptimizerWeights
            ) {
                layer.loraUpM = upM.asType(.float32)
            }
            if let upV = firstArray(
                for: optimizerCandidates.map(\.upV),
                primary: weights,
                secondary: externalOptimizerWeights
            ) {
                layer.loraUpV = upV.asType(.float32)
            }

            updatedCount += 1
        }

        return updatedCount
    }

    private struct WeightKeyCandidate {
        let down: String
        let up: String
    }

    private struct OptimizerKeyCandidate {
        let downM: String
        let downV: String
        let upM: String
        let upV: String
    }

    private static let keyPrefixes: [String] = [
        "",
        "transformer.",
        "diffusion_model.",
        "model.",
        "base_model.model.",
    ]

    private static func matchingWeightCandidate(
        for path: String,
        in weights: [String: MLXArray]
    ) -> WeightKeyCandidate? {
        for candidate in weightKeyCandidates(for: path) {
            if weights[candidate.down] != nil, weights[candidate.up] != nil {
                return candidate
            }
        }
        return nil
    }

    private static func weightKeyCandidates(for path: String) -> [WeightKeyCandidate] {
        var candidates: [WeightKeyCandidate] = []
        candidates.reserveCapacity(keyPrefixes.count * 3)
        for prefix in keyPrefixes {
            let basePath = prefix + path
            candidates.append(
                WeightKeyCandidate(
                    down: "\(basePath).lora_down.weight",
                    up: "\(basePath).lora_up.weight"
                )
            )
            candidates.append(
                WeightKeyCandidate(
                    down: "\(basePath).lora_A.weight",
                    up: "\(basePath).lora_B.weight"
                )
            )
            candidates.append(
                WeightKeyCandidate(
                    down: "\(basePath).lora_A.default.weight",
                    up: "\(basePath).lora_B.default.weight"
                )
            )
        }
        return candidates
    }

    private static func optimizerKeyCandidates(for path: String) -> [OptimizerKeyCandidate] {
        var candidates: [OptimizerKeyCandidate] = []
        candidates.reserveCapacity(keyPrefixes.count * 3)
        for prefix in keyPrefixes {
            let basePath = prefix + path
            candidates.append(
                OptimizerKeyCandidate(
                    downM: "\(basePath).lora_down.m",
                    downV: "\(basePath).lora_down.v",
                    upM: "\(basePath).lora_up.m",
                    upV: "\(basePath).lora_up.v"
                )
            )
            candidates.append(
                OptimizerKeyCandidate(
                    downM: "\(basePath).lora_A.m",
                    downV: "\(basePath).lora_A.v",
                    upM: "\(basePath).lora_B.m",
                    upV: "\(basePath).lora_B.v"
                )
            )
            candidates.append(
                OptimizerKeyCandidate(
                    downM: "\(basePath).lora_A.default.m",
                    downV: "\(basePath).lora_A.default.v",
                    upM: "\(basePath).lora_B.default.m",
                    upV: "\(basePath).lora_B.default.v"
                )
            )
        }
        return candidates
    }

    private static func firstArray(
        for keys: [String],
        primary: [String: MLXArray],
        secondary: [String: MLXArray]?
    ) -> MLXArray? {
        for key in keys {
            if let value = primary[key] {
                return value
            }
            if let secondary, let value = secondary[key] {
                return value
            }
        }
        return nil
    }
}
