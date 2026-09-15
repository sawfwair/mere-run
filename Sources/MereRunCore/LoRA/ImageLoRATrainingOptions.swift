import Foundation

/// Inputs shared by image training callers. Dashboard and console settings
/// remain owned by the caller.
public struct ImageLoRATrainingOptions: Sendable {
    public static let defaultManagedModelID: ModelResolver.ModelID = .krea2Raw
    public static let defaultWidth = 1024
    public static let defaultHeight = 1024
    public static let defaultTrainingSteps = 1000
    public static let defaultLearningRate: Float = 1e-4
    public static let defaultRank = 16
    public static let defaultCaptionDropout: Float = 0.05

    public var data: String? = nil
    public var output: String
    public var model: String? = nil
    public var widthOverride: Int? = nil
    public var heightOverride: Int? = nil
    public var trainingStepsOverride: Int? = nil
    public var batchSize: Int = 1
    public var learningRateOverride: Float? = nil
    public var rankOverride: Int? = nil
    public var alphaOverride: Float? = nil
    public var maxTextLength: Int = 512
    public var schedulerSteps: Int = 1000
    public var captionDropoutOverride: Float? = nil
    public var seed: UInt64 = 0
    public var lite: Bool = false
    public var baseQuantizationBits: Int? = nil
    public var excludePreviewImages: Bool = false
    public var checkpointInterval: Int? = nil
    public var resumeFrom: String? = nil
    public var maxResolution: Int? = nil
    public var progressive: Bool = false
    public var lowRam: Bool = false
    public var noCompile: Bool = false
    public var gradientCheckpointing: Bool = false
    public var recipe: String? = nil
    public var benchmarkSteps: Int? = nil
    public var benchmarkWarmupSteps: Int = 5
    public var sampleInterval: Int? = nil
    public var samplePrompt: String? = nil
    public var sampleModel: String? = nil
    public var sampleSteps: Int = 8
    public var sampleGuidanceScale: Double = 1.0
    public var sampleLoRAScale: Double = 1.0
    public var sampleSeed: UInt64? = nil
    public var loraTargetRanks: String? = nil
    public var loraRankPreset: String? = nil
    public var loraTargetPreset: String? = nil
    public var loraTargetMode: String? = nil
    public var timestepSampling: String? = nil
    public var timestepLossWeighting: String? = nil
    public var lossWeighting: String? = nil
    public var timestepLow: Int? = nil
    public var timestepHigh: Int? = nil
    public var lrWarmupSteps: Int? = nil
    public var noCosineScheduler: Bool = false
    public var lrMinFactor: Float? = nil
    public var adamWeightDecay: Float? = nil
    public var syntheticSamples: Int? = nil

    public init(data: String? = nil, output: String) {
        self.data = data
        self.output = output
    }

    var width: Int { widthOverride ?? Self.defaultWidth }
    var height: Int { heightOverride ?? Self.defaultHeight }
    var trainingSteps: Int { trainingStepsOverride ?? Self.defaultTrainingSteps }
    var learningRate: Float { learningRateOverride ?? Self.defaultLearningRate }
    var rank: Int { rankOverride ?? Self.defaultRank }
    var alpha: Float? { alphaOverride }
    var captionDropout: Float { captionDropoutOverride ?? Self.defaultCaptionDropout }

    public struct Resolved: Sendable {
        public let model: String?
        public let width: Int
        public let height: Int
        public let trainingSteps: Int
        public let learningRate: Float
        public let rank: Int
        public let alpha: Float?
        public let captionDropout: Float
        public let checkpointInterval: Int?
        public let maxResolution: Int?
        public let lowRam: Bool
        public let noCompile: Bool
        public let loraTargetPreset: String?
        public let lrWarmupSteps: Int?
        public let useCosineScheduler: Bool?
        public let lrMinFactor: Float?
    }

    public func validate(_ resolvedOptions: Resolved) throws {
        guard resolvedOptions.width > 0,
              resolvedOptions.height > 0,
              resolvedOptions.width % 16 == 0,
              resolvedOptions.height % 16 == 0 else {
            throw ImageLoRATrainingIssue("--width/--height must be > 0 and divisible by 16")
        }
        guard resolvedOptions.trainingSteps >= 1 else {
            throw ImageLoRATrainingIssue("--training-steps must be >= 1")
        }
        guard batchSize >= 1 else {
            throw ImageLoRATrainingIssue("--batch-size must be >= 1")
        }
        guard schedulerSteps >= 1 else {
            throw ImageLoRATrainingIssue("--scheduler-steps must be >= 1")
        }
        guard resolvedOptions.rank >= 1 else {
            throw ImageLoRATrainingIssue("--rank must be >= 1")
        }
        guard (0.0...1.0).contains(resolvedOptions.captionDropout) else {
            throw ImageLoRATrainingIssue("--caption-dropout must be between 0.0 and 1.0")
        }
        if let syntheticSamples, syntheticSamples < 1 {
            throw ImageLoRATrainingIssue("--synthetic-samples must be >= 1")
        }
        if let checkpointInterval = resolvedOptions.checkpointInterval, checkpointInterval < 1 {
            throw ImageLoRATrainingIssue("--checkpoint-interval must be >= 1")
        }
        if let resumeFrom {
            let resumeURL = URL(fileURLWithPath: resumeFrom).standardizedFileURL
            guard FileManager.default.fileExists(atPath: resumeURL.path) else {
                throw ImageLoRATrainingIssue("--resume-from checkpoint not found: \(resumeURL.path)")
            }
        }
        if let maxResolution = resolvedOptions.maxResolution, maxResolution < 1 {
            throw ImageLoRATrainingIssue("--max-resolution must be >= 1")
        }
        if let benchmarkSteps, benchmarkSteps < 1 {
            throw ImageLoRATrainingIssue("--benchmark-steps must be >= 1")
        }
        guard benchmarkWarmupSteps >= 0 else {
            throw ImageLoRATrainingIssue("--benchmark-warmup-steps must be >= 0")
        }
        if resolvedOptions.maxResolution != nil, progressive {
            throw ImageLoRATrainingIssue("--max-resolution cannot be combined with --progressive")
        }
        if let sampleInterval, sampleInterval < 1 {
            throw ImageLoRATrainingIssue("--sample-interval must be >= 1")
        }
        guard sampleSteps >= 1 else {
            throw ImageLoRATrainingIssue("--sample-steps must be >= 1")
        }
        guard sampleGuidanceScale >= 0 else {
            throw ImageLoRATrainingIssue("--sample-cfg must be >= 0")
        }
        guard sampleLoRAScale >= 0 else {
            throw ImageLoRATrainingIssue("--sample-lora-scale must be >= 0")
        }
        if loraTargetRanks != nil, loraRankPreset != nil {
            throw ImageLoRATrainingIssue("--lora-target-ranks cannot be combined with --lora-rank-preset")
        }
        if let loraTargetPreset = resolvedOptions.loraTargetPreset {
            if lite {
                throw ImageLoRATrainingIssue("--lite cannot be combined with --lora-target-preset")
            }
            if loraTargetRanks != nil || loraRankPreset != nil {
                throw ImageLoRATrainingIssue("--lora-target-preset cannot be combined with --lora-target-ranks or --lora-rank-preset")
            }
            _ = try Self.resolveKleinTargetPreset(loraTargetPreset, rank: resolvedOptions.rank)
        }
        let parsedLoRATargetMode = try Self.resolveKleinLoRATargetMode(loraTargetMode)
        if parsedLoRATargetMode == .transformerLinearWalk {
            if lite {
                throw ImageLoRATrainingIssue("--lite cannot be combined with --lora-target-mode transformer-linear-walk")
            }
            if loraTargetRanks != nil || loraRankPreset != nil || resolvedOptions.loraTargetPreset != nil {
                throw ImageLoRATrainingIssue("--lora-target-mode transformer-linear-walk cannot be combined with LoRA target/rank presets")
            }
        }
        if let timestepLow, timestepLow < 0 {
            throw ImageLoRATrainingIssue("--timestep-low must be >= 0")
        }
        if let timestepHigh, timestepHigh < 1 {
            throw ImageLoRATrainingIssue("--timestep-high must be >= 1")
        }
        if let timestepLow, let timestepHigh, timestepHigh <= timestepLow {
            throw ImageLoRATrainingIssue("--timestep-high must be greater than --timestep-low")
        }
        if let timestepHigh, timestepHigh > schedulerSteps {
            throw ImageLoRATrainingIssue("--timestep-high must be <= --scheduler-steps")
        }
        if let lrWarmupSteps, lrWarmupSteps < 0 {
            throw ImageLoRATrainingIssue("--lr-warmup-steps must be >= 0")
        }
        if let lrMinFactor, !(0.0...1.0).contains(lrMinFactor) {
            throw ImageLoRATrainingIssue("--lr-min-factor must be between 0.0 and 1.0")
        }
        if let adamWeightDecay, adamWeightDecay < 0 {
            throw ImageLoRATrainingIssue("--adam-weight-decay must be >= 0")
        }
        if let baseQuantizationBits, baseQuantizationBits != 4, baseQuantizationBits != 8 {
            throw ImageLoRATrainingIssue("--base-quantization-bits must be 4 or 8")
        }
    }

    private struct LoRATrainingRecipe {
        let model: String?
        let width: Int?
        let height: Int?
        let trainingSteps: Int?
        let learningRate: Float?
        let rank: Int?
        let alpha: Float?
        let captionDropout: Float?
        let checkpointInterval: Int?
        let maxResolution: Int?
        let lowRam: Bool
        let noCompile: Bool
        let loraTargetPreset: String?
        let lrWarmupSteps: Int?
        let useCosineScheduler: Bool?
        let lrMinFactor: Float?
    }

    public func resolve() throws -> Resolved {
        let recipe = try Self.resolveLoRATrainingRecipe(recipe)
        let explicitSchedulerRequested = lrWarmupSteps != nil || lrMinFactor != nil
        return Resolved(
            model: model ?? recipe?.model,
            width: widthOverride ?? recipe?.width ?? Self.defaultWidth,
            height: heightOverride ?? recipe?.height ?? Self.defaultHeight,
            trainingSteps: trainingStepsOverride ?? recipe?.trainingSteps ?? Self.defaultTrainingSteps,
            learningRate: learningRateOverride ?? recipe?.learningRate ?? Self.defaultLearningRate,
            rank: rankOverride ?? recipe?.rank ?? Self.defaultRank,
            alpha: alphaOverride ?? recipe?.alpha,
            captionDropout: captionDropoutOverride ?? recipe?.captionDropout ?? Self.defaultCaptionDropout,
            checkpointInterval: checkpointInterval ?? recipe?.checkpointInterval,
            maxResolution: maxResolution ?? recipe?.maxResolution,
            lowRam: lowRam || recipe?.lowRam == true,
            noCompile: noCompile || recipe?.noCompile == true,
            loraTargetPreset: loraTargetPreset ?? recipe?.loraTargetPreset,
            lrWarmupSteps: lrWarmupSteps ?? recipe?.lrWarmupSteps,
            useCosineScheduler: noCosineScheduler ? false : recipe?.useCosineScheduler ?? (explicitSchedulerRequested ? true : nil),
            lrMinFactor: lrMinFactor ?? recipe?.lrMinFactor
        )
    }

    private static func resolveLoRATrainingRecipe(_ raw: String?) throws -> LoRATrainingRecipe? {
        guard let raw else { return nil }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "krea-fast-style", "local-krea-style", "fal-krea-style", "krea2-fast-style":
            return LoRATrainingRecipe(
                model: ModelResolver.ModelID.krea2Raw.rawValue,
                width: 768,
                height: 768,
                trainingSteps: 100,
                learningRate: 0.0005,
                rank: 32,
                alpha: 32,
                captionDropout: Self.defaultCaptionDropout,
                checkpointInterval: nil,
                maxResolution: nil,
                lowRam: false,
                noCompile: false,
                loraTargetPreset: nil,
                lrWarmupSteps: 10,
                useCosineScheduler: true,
                lrMinFactor: 0
            )
        case "krea-cinematic-style", "krea-movie-style", "krea-wide-style":
            return LoRATrainingRecipe(
                model: ModelResolver.ModelID.krea2Raw.rawValue,
                width: 768,
                height: 416,
                trainingSteps: 200,
                learningRate: 0.0001,
                rank: 32,
                alpha: 32,
                captionDropout: Self.defaultCaptionDropout,
                checkpointInterval: nil,
                maxResolution: nil,
                lowRam: false,
                noCompile: true,
                loraTargetPreset: nil,
                lrWarmupSteps: 20,
                useCosineScheduler: true,
                lrMinFactor: 0
            )
        case "klein-fast-style", "local-klein-style", "flux2-klein-fast-style":
            return LoRATrainingRecipe(
                model: ModelResolver.ModelID.kleinBase9B.rawValue,
                width: nil,
                height: nil,
                trainingSteps: 1000,
                learningRate: 0.00005,
                rank: nil,
                alpha: nil,
                captionDropout: nil,
                checkpointInterval: 250,
                maxResolution: 512,
                lowRam: true,
                noCompile: true,
                loraTargetPreset: "fal-klein-fast",
                lrWarmupSteps: nil,
                useCosineScheduler: nil,
                lrMinFactor: nil
            )
        default:
            throw ImageLoRATrainingIssue(
                "Unsupported --recipe '\(raw)'. Supported recipes: krea-fast-style, krea-cinematic-style, klein-fast-style"
            )
        }
    }

    struct KleinRankPreset {
        let rank: Int
        let alpha: Float
        let targetRankSuffixes: [String: Int]
    }

    static func resolveKleinRankPreset(_ raw: String?) throws -> KleinRankPreset? {
        guard let raw else { return nil }
        switch raw.lowercased() {
        case "flux2-style-128", "style-128":
            return KleinRankPreset(
                rank: 128,
                alpha: 64,
                targetRankSuffixes: Dictionary(
                    uniqueKeysWithValues: Flux2LoRAInjector.defaultTargetSuffixes.map { ($0, 128) }
                )
            )
        default:
            throw ImageLoRATrainingIssue("Unsupported --lora-rank-preset '\(raw)'. Supported preset: flux2-style-128")
        }
    }

    static func resolveKleinLoRATargetMode(_ raw: String?) throws -> Flux2LoRAInjector.TargetMode {
        guard let raw else { return .suffix }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "suffix", "default":
            return .suffix
        case "transformer-linear-walk", "linear-walk", "all-linear", "walk":
            return .transformerLinearWalk
        default:
            throw ImageLoRATrainingIssue("Unsupported --lora-target-mode '\(raw)'. Supported modes: suffix, transformer-linear-walk")
        }
    }

    public static func resolveKleinTargetPreset(_ raw: String?, rank: Int) throws -> [String: Int]? {
        guard let raw else { return nil }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "fal-klein-fast", "fal-fast", "flux2-klein-fal-fast":
            return falKleinFastTargetRanks(rank: rank)
        default:
            throw ImageLoRATrainingIssue("Unsupported --lora-target-preset '\(raw)'. Supported preset: fal-klein-fast")
        }
    }

    static func falKleinFastTargetRanks(rank: Int) -> [String: Int] {
        var ranks: [String: Int] = [
            "x_embedder": rank,
            "context_embedder": rank,
            "time_guidance_embed.timestep_embedder.linear_1": rank,
            "time_guidance_embed.timestep_embedder.linear_2": rank,
            "double_stream_modulation_img.linear": rank,
            "double_stream_modulation_txt.linear": rank,
            "single_stream_modulation.linear": rank,
            "proj_out": rank,
        ]

        for block in 0..<8 {
            for suffix in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
            ] {
                ranks["transformer_blocks.\(block).\(suffix)"] = rank
            }
        }

        for block in 0..<24 {
            ranks["single_transformer_blocks.\(block).attn.to_qkv_mlp_proj"] = rank
            ranks["single_transformer_blocks.\(block).attn.to_out"] = rank
        }

        return ranks
    }

    static func parseKleinTargetRankSuffixes(_ raw: String) throws -> [String: Int] {
        let entries = raw
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !entries.isEmpty else {
            throw ImageLoRATrainingIssue("--lora-target-ranks cannot be empty")
        }

        var ranks: [String: Int] = [:]
        for entry in entries {
            let parts = entry
                .split(separator: "=", maxSplits: 1)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            guard parts.count == 2, !parts[0].isEmpty, let rank = Int(parts[1]) else {
                throw ImageLoRATrainingIssue("Invalid --lora-target-ranks entry '\(entry)'; use suffix=rank")
            }
            guard rank >= 1 else {
                throw ImageLoRATrainingIssue("--lora-target-ranks entry '\(entry)' must use rank >= 1")
            }
            ranks[parts[0]] = rank
        }
        return ranks
    }

    public func hasKleinOnlyTrainingOptions(options: Resolved) -> Bool {
        options.maxResolution != nil ||
            resumeFrom != nil ||
            progressive ||
            options.lowRam ||
            gradientCheckpointing ||
            benchmarkSteps != nil ||
            benchmarkWarmupSteps != 5 ||
            sampleInterval != nil ||
            samplePrompt != nil ||
            sampleModel != nil ||
            sampleSteps != 8 ||
            sampleGuidanceScale != 1.0 ||
            sampleLoRAScale != 1.0 ||
            sampleSeed != nil ||
            loraTargetRanks != nil ||
            loraRankPreset != nil ||
            options.loraTargetPreset != nil ||
            loraTargetMode != nil ||
            timestepSampling != nil ||
            timestepLossWeighting != nil ||
            lossWeighting != nil ||
            timestepLow != nil ||
            timestepHigh != nil ||
            adamWeightDecay != nil
    }

    static let kleinLiteTargetSuffixes: [String] = [
        ".attn.to_q",
        ".attn.to_v",
        ".attn.add_q_proj",
        ".attn.add_v_proj",
    ]

}
