import Foundation
import MLX
import MLXNN

public enum ACEStepAdapterKind: String, Codable, CaseIterable, Sendable {
    case auto
    case lora
    case lokr
}

public struct ACEStepAdapterDescriptor: Codable, Hashable, Sendable {
    public var kind: ACEStepAdapterKind
    public var filename: String
    public var sha256: String
    public var scale: Float
    public var matchedLayerCount: Int
    public var matchedLayers: [String]

    public init(
        kind: ACEStepAdapterKind,
        filename: String,
        sha256: String,
        scale: Float,
        matchedLayers: [String]
    ) {
        self.kind = kind
        self.filename = filename
        self.sha256 = sha256
        self.scale = scale
        self.matchedLayerCount = matchedLayers.count
        self.matchedLayers = matchedLayers
    }
}

public enum ACEStepAdapterError: LocalizedError {
    case fileNotFound(String)
    case unsupportedFormat(String)
    case noCompatibleLayers
    case ambiguousLayer(String, [String])
    case invalidTensor(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "ACE-Step adapter file not found: \(path)"
        case .unsupportedFormat(let reason):
            return "Unsupported ACE-Step adapter: \(reason)"
        case .noCompatibleLayers:
            return "The adapter contains no weights compatible with ACE-Step decoder Linear layers."
        case .ambiguousLayer(let key, let matches):
            return "Adapter layer '\(key)' ambiguously matches: \(matches.joined(separator: ", "))."
        case .invalidTensor(let reason):
            return "Invalid ACE-Step adapter tensor: \(reason)"
        }
    }
}

extension ACEStepPipeline {
    /// Loads a PEFT LoRA or LyCORIS LoKr adapter directly onto the resident
    /// decoder. Repeated calls stack adapters without rebuilding the pipeline.
    @discardableResult
    public func loadAdapter(
        from url: URL,
        kind requestedKind: ACEStepAdapterKind = .auto,
        scale: Float = 1
    ) throws -> ACEStepAdapterDescriptor {
        try ACEStepAdapterLoader.load(
            from: url,
            kind: requestedKind,
            scale: scale,
            into: decoder
        )
    }
}

private final class ACEStepAdapterLinear: Linear {
    struct LoRAContribution {
        var down: MLXArray
        var up: MLXArray
        var scale: Float
    }

    struct LoKRContribution {
        var w1: MLXArray
        var w2: MLXArray
        var scale: Float
    }

    private let base: Linear
    var loraContributions: [LoRAContribution] = []
    var lokrContributions: [LoKRContribution] = []

    init(base: Linear) {
        self.base = base
        super.init(weight: base.weight, bias: base.bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var output = base(x)
        let input = x.asType(.float32)
        for contribution in loraContributions {
            let projected = MLX.matmul(
                MLX.matmul(input, contribution.down.T),
                contribution.up.T
            ) * MLXArray(contribution.scale)
            output = output + projected.asType(output.dtype)
        }
        for contribution in lokrContributions {
            let prefixShape = Array(input.shape.dropLast())
            let grouped = input.reshaped(
                prefixShape + [contribution.w1.dim(1), contribution.w2.dim(1)]
            )
            let projectedW2 = MLX.matmul(grouped, contribution.w2.T)
            let swapped = projectedW2.swappedAxes(-1, -2)
            let projectedW1 = MLX.matmul(swapped, contribution.w1.T)
                .swappedAxes(-1, -2)
                .reshaped(prefixShape + [contribution.w1.dim(0) * contribution.w2.dim(0)])
            let projected = projectedW1 * MLXArray(contribution.scale)
            output = output + projected.asType(output.dtype)
        }
        return output
    }
}

enum ACEStepAdapterLoader {
    private struct LoRAPair {
        var key: String
        var down: MLXArray
        var up: MLXArray
        var alpha: Float
    }

    private struct LoKRGroup {
        var key: String
        var tensors: [String: MLXArray]
    }

    private static let targetSuffixes = [
        ".self_attn.q_proj",
        ".self_attn.k_proj",
        ".self_attn.v_proj",
        ".self_attn.o_proj",
        ".cross_attn.q_proj",
        ".cross_attn.k_proj",
        ".cross_attn.v_proj",
        ".cross_attn.o_proj",
    ]

    static func load(
        from url: URL,
        kind requestedKind: ACEStepAdapterKind,
        scale: Float,
        into decoder: Module
    ) throws -> ACEStepAdapterDescriptor {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ACEStepAdapterError.fileNotFound(url.path)
        }
        guard scale.isFinite else {
            throw ACEStepAdapterError.invalidTensor("adapter scale must be finite.")
        }

        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: url)
        let kind = try resolveKind(
            requestedKind,
            keys: Array(arrays.keys),
            metadata: metadata
        )
        let matchedLayers: [String]
        switch kind {
        case .lora:
            matchedLayers = try loadLoRA(
                arrays: arrays,
                metadata: metadata,
                scale: scale,
                into: decoder
            )
        case .lokr:
            matchedLayers = try loadLoKR(
                arrays: arrays,
                metadata: metadata,
                scale: scale,
                into: decoder
            )
        case .auto:
            preconditionFailure("Adapter kind must be resolved before loading.")
        }
        guard !matchedLayers.isEmpty else {
            throw ACEStepAdapterError.noCompatibleLayers
        }

        return try ACEStepAdapterDescriptor(
            kind: kind,
            filename: url.lastPathComponent,
            sha256: ModelArtifactPin.fileSHA256(url),
            scale: scale,
            matchedLayers: matchedLayers.sorted()
        )
    }

    private static func resolveKind(
        _ requested: ACEStepAdapterKind,
        keys: [String],
        metadata: [String: String]?
    ) throws -> ACEStepAdapterKind {
        if requested != .auto {
            return requested
        }
        if keys.contains(where: { $0.contains(".lokr_w") })
            || metadata?["algo"]?.lowercased() == "lokr"
        {
            return .lokr
        }
        if keys.contains(where: {
            $0.contains(".lora_A")
                || $0.contains(".lora_B")
                || $0.contains(".lora_down")
                || $0.contains(".lora_up")
        }) {
            return .lora
        }
        throw ACEStepAdapterError.unsupportedFormat(
            "could not detect PEFT LoRA or LyCORIS LoKr keys."
        )
    }

    private static func loadLoRA(
        arrays: [String: MLXArray],
        metadata: [String: String]?,
        scale: Float,
        into decoder: Module
    ) throws -> [String] {
        let pairs = try loraPairs(arrays: arrays, metadata: metadata)
        let leaves = targetLeaves(in: decoder)
        var replacements: [String: Module] = [:]
        var matched: [String] = []

        for pair in pairs {
            let path = try matchPath(pair.key, among: Array(leaves.keys))
            guard let path, let linear = leaves[path] else { continue }
            guard pair.down.ndim == 2, pair.up.ndim == 2 else {
                throw ACEStepAdapterError.invalidTensor(
                    "\(pair.key) LoRA tensors must both be rank 2."
                )
            }
            let rank = pair.down.dim(0)
            let shape = linear.shape
            guard pair.down.shape == [rank, shape.1],
                  pair.up.shape == [shape.0, rank] else {
                throw ACEStepAdapterError.invalidTensor(
                    "\(pair.key) expects down [\(rank), \(shape.1)] and "
                        + "up [\(shape.0), \(rank)], got "
                        + "\(pair.down.shape) and \(pair.up.shape)."
                )
            }
            let wrapper = adapterWrapper(
                for: path,
                linear: linear,
                replacements: &replacements
            )
            wrapper.loraContributions.append(
                .init(
                    down: pair.down.asType(.float32),
                    up: pair.up.asType(.float32),
                    scale: (pair.alpha / Float(rank)) * scale
                )
            )
            matched.append(path)
        }

        apply(replacements: replacements, to: decoder)
        return matched
    }

    private static func loadLoKR(
        arrays: [String: MLXArray],
        metadata: [String: String]?,
        scale: Float,
        into decoder: Module
    ) throws -> [String] {
        let groups = lokrGroups(arrays: arrays)
        let leaves = targetLeaves(in: decoder)
        var replacements: [String: Module] = [:]
        var matched: [String] = []

        for group in groups {
            let path = try matchPath(group.key, among: Array(leaves.keys))
            guard let path, let linear = leaves[path] else { continue }
            let (w1, w2, nativeScale) = try resolveLoKR(
                group,
                metadata: metadata
            )
            let deltaShape = [
                w1.dim(0) * w2.dim(0),
                w1.dim(1) * w2.dim(1),
            ]
            guard deltaShape == [linear.shape.0, linear.shape.1] else {
                throw ACEStepAdapterError.invalidTensor(
                    "\(group.key) LoKr delta has shape \(deltaShape); "
                        + "expected [\(linear.shape.0), \(linear.shape.1)]."
                )
            }
            let wrapper = adapterWrapper(
                for: path,
                linear: linear,
                replacements: &replacements
            )
            wrapper.lokrContributions.append(
                .init(
                    w1: w1.asType(.float32),
                    w2: w2.asType(.float32),
                    scale: nativeScale * scale
                )
            )
            matched.append(path)
        }

        apply(replacements: replacements, to: decoder)
        return matched
    }

    private static func targetLeaves(in decoder: Module) -> [String: Linear] {
        var result: [String: Linear] = [:]
        for (path, module) in decoder.namedModules() {
            guard targetSuffixes.contains(where: { path.hasSuffix($0) }),
                  let linear = module as? Linear else {
                continue
            }
            result[path] = linear
        }
        return result
    }

    private static func adapterWrapper(
        for path: String,
        linear: Linear,
        replacements: inout [String: Module]
    ) -> ACEStepAdapterLinear {
        if let pending = replacements[path] as? ACEStepAdapterLinear {
            return pending
        }
        if let existing = linear as? ACEStepAdapterLinear {
            return existing
        }
        let wrapper = ACEStepAdapterLinear(base: linear)
        replacements[path] = wrapper
        return wrapper
    }

    private static func apply(
        replacements: [String: Module],
        to decoder: Module
    ) {
        guard !replacements.isEmpty else { return }
        decoder.update(
            modules: ModuleChildren.unflattened(
                replacements.map { ($0.key, $0.value) }
            )
        )
    }

    private static func loraPairs(
        arrays: [String: MLXArray],
        metadata: [String: String]?
    ) throws -> [LoRAPair] {
        let patterns = [
            (down: ".lora_A.default.weight", up: ".lora_B.default.weight"),
            (down: ".lora_A.weight", up: ".lora_B.weight"),
            (down: ".lora_down.weight", up: ".lora_up.weight"),
            (down: ".lora_down", up: ".lora_up"),
        ]
        let fallbackAlpha = metadata.flatMap { values in
            ["lora_alpha", "alpha", "network_alpha"]
                .compactMap { key in values[key].flatMap(Float.init) }
                .first
        }
        var pairs: [LoRAPair] = []
        var seen = Set<String>()

        for key in arrays.keys.sorted() {
            for pattern in patterns where key.hasSuffix(pattern.down) {
                let base = String(key.dropLast(pattern.down.count))
                guard seen.insert(base).inserted else { continue }
                let upKey = base + pattern.up
                guard let down = arrays[key], let up = arrays[upKey] else {
                    throw ACEStepAdapterError.invalidTensor(
                        "missing pair for \(key)."
                    )
                }
                let rank = down.ndim == 2 ? down.dim(0) : 0
                let alpha = scalar(
                    arrays[base + ".alpha"]
                        ?? arrays[base + ".lora_alpha"]
                ) ?? fallbackAlpha ?? Float(rank)
                pairs.append(
                    .init(key: base, down: down, up: up, alpha: alpha)
                )
            }
        }
        if pairs.isEmpty {
            throw ACEStepAdapterError.unsupportedFormat(
                "no complete LoRA A/B or down/up tensor pairs were found."
            )
        }
        return pairs
    }

    private static func lokrGroups(
        arrays: [String: MLXArray]
    ) -> [LoKRGroup] {
        let suffixes = [
            ".lokr_w1",
            ".lokr_w1_a",
            ".lokr_w1_b",
            ".lokr_w2",
            ".lokr_w2_a",
            ".lokr_w2_b",
            ".lokr_t2",
            ".alpha",
        ]
        var groups: [String: [String: MLXArray]] = [:]
        for (key, array) in arrays {
            guard let suffix = suffixes.first(where: { key.hasSuffix($0) }) else {
                continue
            }
            let base = String(key.dropLast(suffix.count))
            groups[base, default: [:]][String(suffix.dropFirst())] = array
        }
        return groups.keys.sorted().map {
            LoKRGroup(key: $0, tensors: groups[$0]!)
        }
    }

    private static func resolveLoKR(
        _ group: LoKRGroup,
        metadata: [String: String]?
    ) throws -> (MLXArray, MLXArray, Float) {
        if group.tensors["lokr_t2"] != nil {
            throw ACEStepAdapterError.unsupportedFormat(
                "\(group.key) uses Tucker-convolution LoKr weights; "
                    + "ACE-Step attention projections are Linear and must not contain lokr_t2."
            )
        }
        let w1 = try factor(
            direct: group.tensors["lokr_w1"],
            left: group.tensors["lokr_w1_a"],
            right: group.tensors["lokr_w1_b"],
            name: "\(group.key).lokr_w1"
        )
        let w2 = try factor(
            direct: group.tensors["lokr_w2"],
            left: group.tensors["lokr_w2_a"],
            right: group.tensors["lokr_w2_b"],
            name: "\(group.key).lokr_w2"
        )
        guard w1.ndim == 2, w2.ndim == 2 else {
            throw ACEStepAdapterError.invalidTensor(
                "\(group.key) LoKr factors must materialize to rank-2 matrices."
            )
        }

        let rank = group.tensors["lokr_w1_a"]?.dim(1)
            ?? group.tensors["lokr_w2_a"]?.dim(1)
        let metadataAlpha = metadata.flatMap { values in
            ["linear_alpha", "alpha"]
                .compactMap { key in values[key].flatMap(Float.init) }
                .first
        }
        let alpha = scalar(group.tensors["alpha"]) ?? metadataAlpha
        let nativeScale: Float
        if group.tensors["lokr_w1"] != nil,
           group.tensors["lokr_w2"] != nil {
            nativeScale = 1
        } else if let rank, rank > 0 {
            nativeScale = (alpha ?? Float(rank)) / Float(rank)
        } else {
            nativeScale = 1
        }
        return (w1, w2, nativeScale)
    }

    private static func factor(
        direct: MLXArray?,
        left: MLXArray?,
        right: MLXArray?,
        name: String
    ) throws -> MLXArray {
        if let direct {
            return direct
        }
        guard let left, let right,
              left.ndim == 2, right.ndim == 2,
              left.dim(1) == right.dim(0) else {
            throw ACEStepAdapterError.invalidTensor(
                "\(name) requires a direct tensor or compatible _a/_b factors."
            )
        }
        return MLX.matmul(left, right)
    }

    private static func scalar(_ array: MLXArray?) -> Float? {
        guard let array, array.size == 1 else { return nil }
        MLX.eval(array)
        return array.item(Float.self)
    }

    private static func matchPath(
        _ externalKey: String,
        among paths: [String]
    ) throws -> String? {
        let normalized = normalize(externalKey)
        let canonical = canonicalize(normalized)
        let matches = paths.filter { path in
            normalized == path
                || normalized.hasSuffix("." + path)
                || canonical == canonicalize(path)
                || canonical.hasSuffix("_" + canonicalize(path))
        }
        if matches.count > 1 {
            throw ACEStepAdapterError.ambiguousLayer(externalKey, matches)
        }
        return matches.first
    }

    private static func normalize(_ key: String) -> String {
        var result = key
        let prefixes = [
            "base_model.model.",
            "base_model.",
            "model.model.",
            "model.decoder.",
            "diffusion_model.",
            "transformer.",
            "decoder.",
            "module.",
            "lycoris_",
            "lora_unet_",
        ]
        var removed = true
        while removed {
            removed = false
            for prefix in prefixes where result.hasPrefix(prefix) {
                result.removeFirst(prefix.count)
                removed = true
            }
        }
        return result
    }

    private static func canonicalize(_ key: String) -> String {
        key.replacingOccurrences(of: ".", with: "_")
    }
}
