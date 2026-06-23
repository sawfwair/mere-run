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
            if id.hasPrefix("image-hidream-") { return .hidream }
            if id.hasPrefix("image-krea2-") { return .krea }
            if id.hasPrefix("image-ideogram4-") { return .ideogram }
            if id.hasPrefix("vision-segment-") { return .sam }
            if id.hasPrefix("vision-ground-") { return .falcon }
            return nil
        }

        func inferTier(from id: String) -> MereRunModelManifest.Tier? {
            if id.hasSuffix("-nano") { return .nano }
            if id.hasSuffix("-small") { return .small }
            if id.hasSuffix("-max") { return .max }
            if id.hasSuffix("-base") { return .base }
            return nil
        }

        let family = baseManifest.family ?? inferFamily(from: id) ?? {
            switch engine {
            case .flux2Klein: return .klein
            case .zimageTurbo: return .zimage
            case .hidreamO1: return .hidream
            case .krea2: return .krea
            case .ideogram4: return .ideogram
            case .gemma4: return .gemma
            case .lfm2: return .liquid
            case .qwen35HybridMoE: return .qwen
            case .samSegmentation: return .sam
            case .falconPerception: return .falcon
            case .qwen3TTS: return .tts
            case .qwen3ASR, .parakeetASR: return .asr
            case .qwen3Embedding: return .embed
            case .openAIPrivacyFilter: return .privacy
            case .qwen3Coder: return .code
            case .lightOnOCR: return .ocr
            case .aceStep, .magentaRT2: return .music
            case .woosh: return .sfx
            case .ltxVideo: return .video
            case .psiChat: return .psi
            case .deepseekV4Flash: return .deepseek
            }
        }()

        let tier = baseManifest.tier ?? inferTier(from: id)

        let variant = baseManifest.variant ?? {
            if family == .sam || family == .falcon { return .standard }
            if tier == .base { return .base }
            return .distilled
        }()

        let precision: MereRunModelManifest.Precision = {
            switch bits {
            case 1: return .int1
            case 2: return .int2
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
                case .hidreamO1:
                    return [.txt2img, .referenceEdit, .subjectPersonalization]
                case .krea2:
                    return [.txt2img]
                case .ideogram4:
                    return [.txt2img]
                case .gemma4:
                    return [.chat]
                case .lfm2:
                    return [.chat]
                case .qwen35HybridMoE:
                    return [.chat]
                case .samSegmentation:
                    return [.visionSegmentation, .visionTracking]
                case .falconPerception:
                    return [.visionGrounding, .visionDetection, .visionSegmentation]
                case .qwen3TTS:
                    return [.speechSynthesis]
                case .qwen3ASR, .parakeetASR:
                    return [.speechRecognition]
                case .qwen3Embedding:
                    return [.textEmbedding]
                case .openAIPrivacyFilter:
                    return [.textAnonymization]
                case .qwen3Coder:
                    return [.chat, .codeGeneration]
                case .lightOnOCR:
                    return [.visionOCR]
                case .aceStep, .magentaRT2:
                    return [.musicGeneration]
                case .woosh:
                    return [.soundEffectGeneration]
                case .ltxVideo:
                    return [.videoGeneration]
                case .psiChat:
                    return [.chat]
                case .deepseekV4Flash:
                    return [.chat]
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
            case .hidreamO1:
                manifest.defaults = MereRunModelManifest.Defaults(steps: 28, cfg: 0.0)
            case .krea2:
                manifest.defaults = MereRunModelManifest.Defaults(
                    steps: 8,
                    cfg: 0.0,
                    sigmaShift: Double(Krea2SampleBuilder.defaultMu)
                )
            case .ideogram4:
                manifest.defaults = MereRunModelManifest.Defaults(steps: 20, cfg: 7.0)
            case .gemma4:
                break
            case .lfm2:
                break
            case .qwen35HybridMoE:
                break
            case .samSegmentation, .falconPerception:
                break
            case .qwen3TTS, .qwen3ASR, .parakeetASR, .qwen3Embedding, .openAIPrivacyFilter, .qwen3Coder, .lightOnOCR, .woosh, .psiChat, .deepseekV4Flash:
                break
            case .aceStep, .magentaRT2, .ltxVideo:
                manifest.defaults = MereRunModelManifest.Defaults(steps: 8, cfg: 1.0)
            }
        }

        try manifest.write(to: outputModelRoot)
        return manifest
    }
}
