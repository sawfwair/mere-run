import Foundation
import MLX
import MLXNN

public struct LTXLoRAConfiguration: Sendable, Equatable {
    public let url: URL
    public let strength: Float

    public init(url: URL, strength: Float = 1) {
        precondition(strength.isFinite, "LoRA strength must be finite")
        self.url = url.standardizedFileURL
        self.strength = strength
    }
}

public struct LTXHDRLoRAConfiguration: Sendable, Hashable {
    public let hdrTransform: LTXHDRTransfer
    public let referenceDownscaleFactor: Int

    public init(
        hdrTransform: LTXHDRTransfer = .logC3,
        referenceDownscaleFactor: Int = 1
    ) {
        precondition(referenceDownscaleFactor > 0)
        self.hdrTransform = hdrTransform
        self.referenceDownscaleFactor = referenceDownscaleFactor
    }
}

public enum LTXHDRLoRAMetadataError: LocalizedError {
    case invalidTransform(String)
    case invalidReferenceDownscaleFactor(String)

    public var errorDescription: String? {
        switch self {
        case .invalidTransform(let value):
            return "Unsupported LTX HDR LoRA transform metadata: \(value)."
        case .invalidReferenceDownscaleFactor(let value):
            return "Invalid LTX HDR LoRA reference_downscale_factor metadata: \(value)."
        }
    }
}

public struct LTXLoRAReferenceScaleConfiguration: Sendable, Hashable {
    public let downscaleFactor: Int
    public let temporalScaleFactor: Int

    public init(downscaleFactor: Int = 1, temporalScaleFactor: Int = 1) {
        precondition(downscaleFactor > 0 && temporalScaleFactor > 0)
        self.downscaleFactor = downscaleFactor
        self.temporalScaleFactor = temporalScaleFactor
    }
}

public enum LTXLoRAReferenceScaleMetadataError: LocalizedError {
    case invalidValue(key: String, value: String, url: URL)
    case conflictingValues(key: String, first: Int, second: Int, url: URL)

    public var errorDescription: String? {
        switch self {
        case .invalidValue(let key, let value, let url):
            return "Invalid \(key) metadata \(value) in LTX LoRA \(url.path)."
        case .conflictingValues(let key, let first, let second, let url):
            return "Conflicting \(key) values in stacked LTX LoRAs: \(first) and \(second) in \(url.path)."
        }
    }
}

/// Resolves the reference geometry contract shared by stacked upstream IC-LoRAs.
/// A value of one is neutral; multiple non-neutral values must agree.
public func ltxLoRAReferenceScaleConfiguration(
    _ configurations: [LTXLoRAConfiguration]
) throws -> LTXLoRAReferenceScaleConfiguration {
    var downscaleFactor = 1
    var temporalScaleFactor = 1
    for configuration in configurations {
        let metadata = try SafetensorsStreamingLoader.fileMetadata(url: configuration.url)
        downscaleFactor = try mergeLTXLoRAReferenceScale(
            current: downscaleFactor,
            raw: metadata["reference_downscale_factor"],
            key: "reference_downscale_factor",
            url: configuration.url
        )
        temporalScaleFactor = try mergeLTXLoRAReferenceScale(
            current: temporalScaleFactor,
            raw: metadata["reference_temporal_scale_factor"],
            key: "reference_temporal_scale_factor",
            url: configuration.url
        )
    }
    return LTXLoRAReferenceScaleConfiguration(
        downscaleFactor: downscaleFactor,
        temporalScaleFactor: temporalScaleFactor
    )
}

private func mergeLTXLoRAReferenceScale(
    current: Int,
    raw: String?,
    key: String,
    url: URL
) throws -> Int {
    guard let raw else { return current }
    guard let value = Int(raw), value > 0 else {
        throw LTXLoRAReferenceScaleMetadataError.invalidValue(
            key: key,
            value: raw,
            url: url
        )
    }
    guard value != 1 else { return current }
    guard current == 1 || current == value else {
        throw LTXLoRAReferenceScaleMetadataError.conflictingValues(
            key: key,
            first: current,
            second: value,
            url: url
        )
    }
    return value
}

/// Reads the exact metadata contract emitted by the upstream HDR IC-LoRA trainer.
/// A non-empty `hdr_transform` or legacy `use_hdr_transform` enables HDR.
public func ltxHDRLoRAConfiguration(
    _ configuration: LTXLoRAConfiguration
) throws -> LTXHDRLoRAConfiguration? {
    let metadata = try SafetensorsStreamingLoader.fileMetadata(url: configuration.url)
    let rawTransform = metadata["hdr_transform"]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    let legacyEnable = metadata["use_hdr_transform"]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    guard !rawTransform.isEmpty || !legacyEnable.isEmpty else { return nil }

    let transform: LTXHDRTransfer
    if rawTransform.isEmpty || rawTransform.lowercased() == "true" {
        transform = .logC3
    } else if let parsed = LTXHDRTransfer(rawValue: rawTransform.lowercased()) {
        transform = parsed
    } else {
        throw LTXHDRLoRAMetadataError.invalidTransform(rawTransform)
    }

    let downscaleFactor: Int
    if let rawScale = metadata["reference_downscale_factor"] {
        guard let parsed = Int(rawScale), parsed > 0 else {
            throw LTXHDRLoRAMetadataError.invalidReferenceDownscaleFactor(rawScale)
        }
        downscaleFactor = parsed
    } else {
        downscaleFactor = 1
    }
    return LTXHDRLoRAConfiguration(
        hdrTransform: transform,
        referenceDownscaleFactor: downscaleFactor
    )
}

public func ltxLoRAReferenceDownscaleFactor(_ configuration: LTXLoRAConfiguration) -> Int {
    guard let raw = try? SafetensorsStreamingLoader.fileMetadata(url: configuration.url)[
        "reference_downscale_factor"
    ], let value = Int(raw), value > 0 else {
        return 1
    }
    return value
}

public func ltxLoRAReferenceTemporalScaleFactor(_ configuration: LTXLoRAConfiguration) -> Int {
    guard let raw = try? SafetensorsStreamingLoader.fileMetadata(url: configuration.url)[
        "reference_temporal_scale_factor"
    ], let value = Int(raw), value > 0 else {
        return 1
    }
    return value
}

private final class LTXRuntimeLoRAContribution {
    let loraDown: MLXArray
    let loraUp: MLXArray
    var strength: Float
    var isActive = false

    init(
        loraDown: MLXArray,
        loraUp: MLXArray,
        strength: Float,
        dtype: DType
    ) {
        self.loraDown = loraDown.asType(dtype)
        self.loraUp = loraUp.asType(dtype)
        self.strength = strength
    }
}

private final class LTXRuntimeLoRALinear: Linear {
    private var contributions: [LTXRuntimeLoRAContribution]

    init(
        base: Linear,
        contribution: LTXRuntimeLoRAContribution
    ) {
        self.contributions = [contribution]
        super.init(weight: base.weight, bias: base.bias)
    }

    func add(_ contribution: LTXRuntimeLoRAContribution) {
        contributions.append(contribution)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var output = super.callAsFunction(x)
        for contribution in contributions where contribution.isActive {
            let input = x.asType(contribution.loraDown.dtype)
            let adapterOutput = MLX.matmul(
                MLX.matmul(input, contribution.loraDown.T),
                contribution.loraUp.T
            ) * MLXArray(contribution.strength).asType(contribution.loraDown.dtype)
            output = output + adapterOutput.asType(output.dtype)
        }
        return output
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
    private let contributions: [LTXRuntimeLoRAContribution]

    private init(pairCount: Int, contributions: [LTXRuntimeLoRAContribution]) {
        self.pairCount = pairCount
        self.contributions = contributions
    }

    static func install(
        url: URL,
        into transformer: Module,
        strength: Float = 1,
        expectedPairCount: Int? = 1_660,
        ignoreMissingTargets: Bool = false
    ) throws -> LTXRuntimeLoRAAdapter {
        let leafModules = transformer.leafModules().flattened()
        let modulesByPath = Dictionary(uniqueKeysWithValues: leafModules)
        var replacements: [String: Module] = [:]
        var contributions: [LTXRuntimeLoRAContribution] = []

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
                if ignoreMissingTargets {
                    return
                }
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
            let contribution = LTXRuntimeLoRAContribution(
                loraDown: a,
                loraUp: b,
                strength: strength,
                dtype: linear.weight.dtype
            )
            MLX.eval(contribution.loraDown, contribution.loraUp)
            if let layer = linear as? LTXRuntimeLoRALinear {
                layer.add(contribution)
            } else {
                replacements[modulePath] = LTXRuntimeLoRALinear(
                    base: linear,
                    contribution: contribution
                )
            }
            contributions.append(contribution)
        }

        guard pairCount > 0, !contributions.isEmpty else {
            throw AdapterError.noPairs(url)
        }
        if let expectedPairCount, pairCount != expectedPairCount {
            throw AdapterError.unexpectedPairCount(expected: expectedPairCount, actual: pairCount)
        }

        applyModuleReplacements(replacements, leafModules: leafModules, to: transformer)
        Memory.clearCache()
        return LTXRuntimeLoRAAdapter(pairCount: contributions.count, contributions: contributions)
    }

    func setActive(_ isActive: Bool) {
        for contribution in contributions {
            contribution.isActive = isActive
        }
    }

    func setStrength(_ strength: Float) {
        precondition(strength.isFinite, "LoRA strength must be finite")
        for contribution in contributions {
            contribution.strength = strength
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
