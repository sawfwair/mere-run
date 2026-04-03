import Foundation

public enum QuantizedModelManifestWriter {
    public static func writeQuantizedModelManifest(
        engine: MereRunModelManifest.Engine,
        inputModelRoot: URL,
        outputModelRoot: URL,
        bits: Int,
        groupSize: Int,
        scheme: String = "mlx-quantized-linear",
        svdResidualRank: Int? = nil,
        svdTargets: [String]? = nil,
        svdMaxLayers: Int? = nil,
        createdAt: Date = Date(),
        fileManager: FileManager = .default
    ) throws -> MereRunModelManifest {
        let baseManifest = try MereRunModelManifest.loadRequired(from: inputModelRoot, fileManager: fileManager)

        let id = outputModelRoot.lastPathComponent.isEmpty
            ? baseManifest.id
            : outputModelRoot.lastPathComponent

        func inferFamily(from id: String) -> MereRunModelManifest.Family? {
            if id.hasPrefix("image-klein-") { return .klein }
            if id.hasPrefix("image-zimage-") { return .zimage }
            if id.hasPrefix("vision-segment-") { return .sam }
            return nil
        }

        func inferTier(from id: String) -> MereRunModelManifest.Tier? {
            if id.hasSuffix("-nano") { return .nano }
            if id.hasSuffix("-max") { return .max }
            if id.hasSuffix("-base") { return .base }
            return nil
        }

        let family = baseManifest.family ?? inferFamily(from: id) ?? {
            switch engine {
            case .flux2Klein: return .klein
            case .zimageTurbo: return .zimage
            case .qwen35HybridMoE: return .qwen
            case .samSegmentation: return .sam
            }
        }()

        let tier = baseManifest.tier ?? inferTier(from: id)

        let variant = baseManifest.variant ?? {
            if family == .sam { return .standard }
            if tier == .base { return .base }
            return .distilled
        }()

        let precision: MereRunModelManifest.Precision = {
            switch bits {
            case 4: return .int4
            case 8: return .int8
            default: return .unknown
            }
        }()

        let supports: [MereRunModelManifest.Capability] = {
            let baseline: [MereRunModelManifest.Capability] = baseManifest.supports ?? {
                switch engine {
                case .flux2Klein:
                    return [.txt2img, .referenceEdit, .loraInference]
                case .zimageTurbo:
                    return [.txt2img, .img2img, .loraInference]
                case .qwen35HybridMoE:
                    return [.chat]
                case .samSegmentation:
                    return [.visionSegmentation, .visionTracking]
                }
            }()
            return baseline.filter { $0 != .loraTraining }
        }()

        let components = baseManifest.components ?? MereRunModelManifest.Components(
            tokenizer: .local(path: "tokenizer"),
            textEncoder: .local(path: "text_encoder"),
            transformer: .local(path: "transformer"),
            vae: .local(path: "vae"),
            scheduler: .local(path: "scheduler")
        )

        let residualRank: Int? = {
            guard let svdResidualRank, svdResidualRank > 0 else { return nil }
            return svdResidualRank
        }()

        let quantization = MereRunModelManifest.Quantization(
            bits: bits,
            groupSize: groupSize,
            scheme: scheme,
            svdResidualRank: residualRank,
            svdTargets: residualRank == nil ? nil : svdTargets,
            svdMaxLayers: residualRank == nil ? nil : svdMaxLayers
        )

        var manifest = MereRunModelManifest(
            schemaVersion: MereRunModelManifest.currentSchemaVersion,
            id: id,
            engine: baseManifest.engine ?? engine,
            family: family,
            tier: tier,
            variant: variant,
            precision: precision,
            quantization: quantization,
            defaults: baseManifest.defaults,
            supports: supports,
            components: components,
            upstreamRepoId: baseManifest.upstreamRepoId,
            createdAt: createdAt
        )

        // If we couldn't infer defaults from the base manifest, set reasonable engine defaults.
        if manifest.defaults == nil {
            switch engine {
            case .flux2Klein:
                manifest.defaults = MereRunModelManifest.Defaults(steps: 4, cfg: 1.0)
            case .zimageTurbo:
                manifest.defaults = MereRunModelManifest.Defaults(steps: 4, cfg: 1.0)
            case .qwen35HybridMoE:
                break
            case .samSegmentation:
                break
            }
        }

        try manifest.write(to: outputModelRoot)
        return manifest
    }
}
