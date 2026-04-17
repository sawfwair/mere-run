import Foundation
import MLX
import MLXNN
import MLXRandom

public enum ZImageLoRAInjectionError: Error, LocalizedError {
    case invalidRank(Int)
    case noMatchingLayers

    public var errorDescription: String? {
        switch self {
        case .invalidRank(let rank):
            return "LoRA rank must be >= 1 (got \(rank))."
        case .noMatchingLayers:
            return "No matching Linear layers found to inject LoRA into."
        }
    }
}

public enum ZImageLoRAInjector {
    /// Default target suffixes for ZImage Turbo transformer layers.
    /// Covers attention (Q/K/V/O) and feedforward (w1/w2/w3) in all block types.
    public static let defaultTargetSuffixes: [String] = [
        // Attention projections
        ".attention.to_q",
        ".attention.to_k",
        ".attention.to_v",
        ".attention.to_out.0",

        // Feedforward (SwiGLU style)
        ".feed_forward.w1",
        ".feed_forward.w2",
        ".feed_forward.w3",
    ]

    /// Lite target suffixes - only Q and V projections in main layers.
    /// Uses significantly less memory while still being effective.
    public static let liteTargetSuffixes: [String] = [
        ".attention.to_q",
        ".attention.to_v",
    ]

    /// ai-toolkit compatible targets - only main `layers` block (not refiners).
    /// Matches what ai-toolkit trains for Z-Image models.
    public static let aiToolkitCompatiblePrefixes: [String] = [
        "layers.",
    ]

    /// ai-toolkit compatible suffixes including adaLN_modulation.
    public static let aiToolkitCompatibleSuffixes: [String] = [
        ".attention.to_q",
        ".attention.to_k",
        ".attention.to_v",
        ".attention.to_out.0",
        ".feed_forward.w1",
        ".feed_forward.w2",
        ".feed_forward.w3",
        ".adaLN_modulation.0",
    ]

    /// Inject LoRA layers into a ZImageTransformer2DModel.
    /// - Parameters:
    ///   - transformer: The transformer to inject LoRA into.
    ///   - rank: LoRA rank (r).
    ///   - alpha: LoRA alpha (defaults to rank).
    ///   - targetPrefixes: Optional prefixes to filter layers (e.g. ["layers."] for ai-toolkit compatibility).
    ///   - targetSuffixes: Which layer paths to target.
    /// - Returns: Dictionary mapping layer paths to injected LoRA layers.
    public static func inject(
        into transformer: ZImageTransformer2DModel,
        rank: Int,
        alpha: Float? = nil,
        targetPrefixes: [String]? = nil,
        targetSuffixes: [String]? = defaultTargetSuffixes,
        targetRanks: [String: Int]? = nil,
        zeroInitUp: Bool = false
    ) throws -> [String: TrainableLoRALayer] {
        guard rank >= 1 else {
            throw ZImageLoRAInjectionError.invalidRank(rank)
        }
        if let targetRanks {
            for configuredRank in targetRanks.values where configuredRank < 1 {
                throw ZImageLoRAInjectionError.invalidRank(configuredRank)
            }
        }

        let shouldTarget: (String) -> Bool = { path in
            // Check prefix if specified
            if let prefixes = targetPrefixes {
                guard prefixes.contains(where: { path.hasPrefix($0) }) else { return false }
            }
            // Check suffix if specified
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

            // Handle stacking: if already a LoRA layer, create FusedLoRALinear
            if let existingFused = module as? FusedLoRALinear {
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
                continue
            }

            if let loraQuantized = module as? LoRAQuantizedLinear {
                // Stack: create FusedLoRALinear with existing as assistant
                let fused = FusedLoRALinear(existingLoRA: loraQuantized, newRank: resolvedRank, newAlpha: alphaValue, zeroInitUp: zeroInitUp)
                replacements[path] = fused
                loraLayers[path] = fused
                continue
            }

            if let loraLinear = module as? LoRALinear {
                // Stack: create FusedLoRALinear with existing as assistant
                let fused = FusedLoRALinear(existingLoRA: loraLinear, newRank: resolvedRank, newAlpha: alphaValue, zeroInitUp: zeroInitUp)
                replacements[path] = fused
                loraLayers[path] = fused
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
            throw ZImageLoRAInjectionError.noMatchingLayers
        }

        applyModuleReplacements(
            replacements,
            leafModules: leafModules,
            to: transformer
        )

        return loraLayers
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
    @discardableResult
    public static func loadWeights(
        from url: URL,
        into loraLayers: [String: TrainableLoRALayer],
        optimizerStateURL: URL? = nil,
        debug: Bool = false
    ) throws -> Int {
        let weights = try MLX.loadArrays(url: url)
        let externalOptimizerWeights: [String: MLXArray]? = {
            guard let optimizerStateURL else { return nil }
            return try? MLX.loadArrays(url: optimizerStateURL)
        }()
        var updatedCount = 0

        if debug {
            let checkpointKeys = Set(weights.keys.compactMap { key in
                for candidate in [".lora_down.weight", ".lora_A.default.weight", ".lora_A.weight"] {
                    if key.hasSuffix(candidate) {
                        return key.replacingOccurrences(of: candidate, with: "")
                    }
                }
                return nil
            })
            let modelKeys = Set(loraLayers.keys)
            let matched = checkpointKeys.intersection(modelKeys)
            let unmatchedCheckpoint = checkpointKeys.subtracting(modelKeys)
            let unmatchedModel = modelKeys.subtracting(checkpointKeys)
            FileHandle.standardError.write(
                Data("[LoRA Debug] Checkpoint keys: \(checkpointKeys.count), Model keys: \(modelKeys.count), Matched: \(matched.count)\n".utf8)
            )
            if !unmatchedCheckpoint.isEmpty {
                FileHandle.standardError.write(
                    Data("[LoRA Debug] Unmatched checkpoint keys (first 5): \(Array(unmatchedCheckpoint.sorted().prefix(5)))\n".utf8)
                )
            }
            if !unmatchedModel.isEmpty {
                FileHandle.standardError.write(
                    Data("[LoRA Debug] Unmatched model keys (first 5): \(Array(unmatchedModel.sorted().prefix(5)))\n".utf8)
                )
            }
        }

        for (path, layer) in loraLayers {
            var downWeight: MLXArray?
            var upWeight: MLXArray?
            for candidate in weightKeyCandidates(for: path) {
                if let d = weights[candidate.down], let u = weights[candidate.up] {
                    downWeight = d
                    upWeight = u
                    break
                }
            }

            guard let downWeight, let upWeight else { continue }

            if debug {
                let expectedDownShape = layer.loraDown.shape
                let expectedUpShape = layer.loraUp.shape
                let actualDownShape = downWeight.shape
                let actualUpShape = upWeight.shape
                if expectedDownShape != actualDownShape || expectedUpShape != actualUpShape {
                    FileHandle.standardError.write(Data("[LoRA Debug] Shape mismatch at \(path):\n".utf8))
                    FileHandle.standardError.write(Data("  down: expected \(expectedDownShape), got \(actualDownShape)\n".utf8))
                    FileHandle.standardError.write(Data("  up: expected \(expectedUpShape), got \(actualUpShape)\n".utf8))
                }
            }

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
        "diffusion_model.",
        "transformer.",
        "model.",
        "base_model.model.",
    ]

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
                    down: "\(basePath).lora_A.default.weight",
                    up: "\(basePath).lora_B.default.weight"
                )
            )
            candidates.append(
                WeightKeyCandidate(
                    down: "\(basePath).lora_A.weight",
                    up: "\(basePath).lora_B.weight"
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
                    downM: "\(basePath).lora_A.default.m",
                    downV: "\(basePath).lora_A.default.v",
                    upM: "\(basePath).lora_B.default.m",
                    upV: "\(basePath).lora_B.default.v"
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

    /// Apply pre-mapped LoRA weights to already-injected layers.
    /// Use this with weights from LoRAWeightLoader which handles key format conversion.
    @discardableResult
    public static func applyWeights(
        _ loraWeights: LoRAWeights,
        to loraLayers: [String: TrainableLoRALayer],
        debug: Bool = false
    ) -> Int {
        var updatedCount = 0

        if debug {
            let checkpointKeys = Set(loraWeights.weights.keys)
            let modelKeys = Set(loraLayers.keys)
            let matched = checkpointKeys.intersection(modelKeys)
            FileHandle.standardError.write(
                Data("[LoRA Debug] Mapped weights: \(checkpointKeys.count), Model keys: \(modelKeys.count), Matched: \(matched.count)\n".utf8)
            )
            if matched.count < checkpointKeys.count {
                let unmatched = checkpointKeys.subtracting(modelKeys)
                FileHandle.standardError.write(
                    Data("[LoRA Debug] Unmatched weight keys (first 5): \(Array(unmatched.sorted().prefix(5)))\n".utf8)
                )
            }
        }

        for (path, layer) in loraLayers {
            guard let weights = loraWeights.weights[path] else { continue }

            layer.loraDown = weights.down.asType(.float32)
            layer.loraUp = weights.up.asType(.float32)

            if debug && updatedCount == 0 {
                // Log first layer's weight stats
                let downNorm = MLX.sqrt(MLX.sum(layer.loraDown * layer.loraDown)).item(Float.self)
                let upNorm = MLX.sqrt(MLX.sum(layer.loraUp * layer.loraUp)).item(Float.self)
                FileHandle.standardError.write(
                    Data("[LoRA Debug] First layer '\(path)' - down norm: \(downNorm), up norm: \(upNorm)\n".utf8)
                )
            }

            updatedCount += 1
        }

        return updatedCount
    }
}
