import Foundation

/// A lightweight, human-readable manifest that describes a locally installed model directory.
///
/// This is intentionally independent of external hub cache metadata. For models installed by mere.run
/// (e.g. via canonical managed Hugging Face snapshots), we write this file into the model root so that:
/// - CLI + app can validate a model directory consistently
/// - inference/training code can branch on a single source of truth (engine, variant, defaults)
///
/// As of Phase 11 (strict mode), the manifest is required for all local model roots used by mere.run's
/// image pipelines. This eliminates silent guessing (variant/precision/quantization/components).
public struct MereRunModelManifest: Codable, Hashable, Sendable {
    public static let currentSchemaVersion: Int = 2
    public static let filename: String = "mererun_model.json"

    public enum Engine: String, Codable, CaseIterable, Hashable, Sendable {
        /// MereRun family (FLUX.2 Klein based).
        case flux2Klein = "flux2-klein"
        /// Zeta family (Z-Image Turbo based).
        case zimageTurbo = "zimage-turbo"
        /// HiDream O1 unified pixel transformer family.
        case hidreamO1 = "hidream-o1"
        /// Gemma 4 family via the native Swift runtime.
        case gemma4 = "gemma-4"
        /// Q35 family (Qwen3.5 hybrid MoE + hybrid attention).
        case qwen35HybridMoE = "qwen3.5-hybrid-moe"
        /// SAM image segmentation family.
        case samSegmentation = "sam-segmentation"
        /// Falcon Perception grounded detection and segmentation family.
        case falconPerception = "falcon-perception"
        /// Qwen3 TTS family.
        case qwen3TTS = "qwen3-tts"
        /// Qwen3 ASR family.
        case qwen3ASR = "qwen3-asr"
        /// Parakeet ASR family.
        case parakeetASR = "parakeet-asr"
        /// Qwen3 embeddings family.
        case qwen3Embedding = "qwen3-embedding"
        /// OpenAI Privacy Filter token-classification family.
        case openAIPrivacyFilter = "openai-privacy-filter"
        /// GGUF code generation family.
        case qwen3Coder = "qwen3-coder"
        /// LightOn OCR family.
        case lightOnOCR = "lighton-ocr"
        /// ACE-Step music family.
        case aceStep = "ace-step"
        /// LTX video family.
        case ltxVideo = "ltx-video"
        /// Psi agent chat family.
        case psiChat = "psi-chat"
        /// DeepSeek V4 Flash family, served by the bundled `ds4-server` subprocess.
        case deepseekV4Flash = "deepseek-v4-flash"
    }

    public enum Family: String, Codable, CaseIterable, Hashable, Sendable {
        case klein
        case zimage
        case hidream
        case gemma
        case qwen
        case sam
        case falcon
        case tts
        case asr
        case embed
        case privacy
        case code
        case ocr
        case music
        case video
        case psi
        case deepseek
    }

    public enum Tier: String, Codable, CaseIterable, Hashable, Sendable {
        case nano
        case base
        case max
        case latest
        case turbo
    }

    public enum Variant: String, Codable, CaseIterable, Hashable, Sendable {
        /// Distilled / turbo variants (fast, low step count).
        case distilled
        /// Base / undistilled variants (slower, higher diversity; used for training).
        case base
        /// Non-distilled runtime families that do not map onto the image-generation variants.
        case standard
    }

    public enum Precision: String, Codable, CaseIterable, Hashable, Sendable {
        case bf16
        case fp16
        case fp32
        case int8
        case int4
        case unknown
    }

    public enum Capability: String, Codable, CaseIterable, Hashable, Sendable {
        case txt2img = "txt2img"
        case img2img = "img2img"
        case referenceEdit = "reference_edit"
        case subjectPersonalization = "subject_personalization"
        case chat = "chat"
        case codeGeneration = "code_generation"
        case textEmbedding = "text_embedding"
        case textAnonymization = "text_anonymization"
        case speechSynthesis = "speech_synthesis"
        case speechRecognition = "speech_recognition"
        case visionOCR = "vision_ocr"
        case musicGeneration = "music_generation"
        case videoGeneration = "video_generation"
        case loraInference = "lora_inference"
        case loraTraining = "lora_training"
        case visionSegmentation = "vision_segmentation"
        case visionTracking = "vision_tracking"
        case visionGrounding = "vision_grounding"
        case visionDetection = "vision_detection"
    }

    public enum ComponentRef: Codable, Hashable, Sendable {
        case local(path: String)
        case absolute(path: String)
        case model(modelID: String, path: String)
        case remote(id: String, revision: String?, path: String)
        case anyOf([ComponentRef])

        private enum CodingKeys: String, CodingKey {
            case type
            case path
            case modelId
            case id
            case revision
            case candidates
        }

        private enum RefType: String, Codable {
            case local
            case absolute
            case model
            case remote
            case anyOf
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decode(RefType.self, forKey: .type)
            switch type {
            case .local:
                self = .local(path: try container.decode(String.self, forKey: .path))
            case .absolute:
                self = .absolute(path: try container.decode(String.self, forKey: .path))
            case .model:
                self = .model(
                    modelID: try container.decode(String.self, forKey: .modelId),
                    path: try container.decode(String.self, forKey: .path)
                )
            case .remote:
                self = .remote(
                    id: try container.decode(String.self, forKey: .id),
                    revision: try container.decodeIfPresent(String.self, forKey: .revision),
                    path: try container.decode(String.self, forKey: .path)
                )
            case .anyOf:
                self = .anyOf(try container.decode([ComponentRef].self, forKey: .candidates))
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .local(let path):
                try container.encode(RefType.local, forKey: .type)
                try container.encode(path, forKey: .path)
            case .absolute(let path):
                try container.encode(RefType.absolute, forKey: .type)
                try container.encode(path, forKey: .path)
            case .model(let modelID, let path):
                try container.encode(RefType.model, forKey: .type)
                try container.encode(modelID, forKey: .modelId)
                try container.encode(path, forKey: .path)
            case .remote(let id, let revision, let path):
                try container.encode(RefType.remote, forKey: .type)
                try container.encode(id, forKey: .id)
                try container.encodeIfPresent(revision, forKey: .revision)
                try container.encode(path, forKey: .path)
            case .anyOf(let candidates):
                try container.encode(RefType.anyOf, forKey: .type)
                try container.encode(candidates, forKey: .candidates)
            }
        }
    }

    public struct Components: Codable, Hashable, Sendable {
        public var tokenizer: ComponentRef?
        public var textEncoder: ComponentRef?
        public var transformer: ComponentRef?
        public var vae: ComponentRef?
        public var scheduler: ComponentRef?

        public init(
            tokenizer: ComponentRef? = nil,
            textEncoder: ComponentRef? = nil,
            transformer: ComponentRef? = nil,
            vae: ComponentRef? = nil,
            scheduler: ComponentRef? = nil
        ) {
            self.tokenizer = tokenizer
            self.textEncoder = textEncoder
            self.transformer = transformer
            self.vae = vae
            self.scheduler = scheduler
        }

        private enum CodingKeys: String, CodingKey {
            case tokenizer
            case textEncoder = "text_encoder"
            case transformer
            case vae
            case scheduler
        }
    }

    public struct Defaults: Codable, Hashable, Sendable {
        public var steps: Int?
        public var cfg: Double?

        public init(steps: Int? = nil, cfg: Double? = nil) {
            self.steps = steps
            self.cfg = cfg
        }
    }

    public struct Quantization: Codable, Hashable, Sendable {
        public var bits: Int?
        public var groupSize: Int?
        public var scheme: String?
        public var svdResidualRank: Int?
        public var svdTargets: [String]?
        public var svdMaxLayers: Int?

        public init(
            bits: Int? = nil,
            groupSize: Int? = nil,
            scheme: String? = nil,
            svdResidualRank: Int? = nil,
            svdTargets: [String]? = nil,
            svdMaxLayers: Int? = nil
        ) {
            self.bits = bits
            self.groupSize = groupSize
            self.scheme = scheme
            self.svdResidualRank = svdResidualRank
            self.svdTargets = svdTargets
            self.svdMaxLayers = svdMaxLayers
        }
    }

    public var schemaVersion: Int
    public var id: String

    public var engine: Engine?
    public var family: Family?
    public var tier: Tier?
    public var variant: Variant?

    public var precision: Precision?
    public var quantization: Quantization?

    public var defaults: Defaults?

    public var supports: [Capability]?
    public var components: Components?

    /// Upstream identifier for the source repository or registry record.
    public var upstreamRepoId: String?

    /// ISO8601 timestamp; informational only.
    public var createdAt: Date?

    public init(
        schemaVersion: Int = Self.currentSchemaVersion,
        id: String,
        engine: Engine? = nil,
        family: Family? = nil,
        tier: Tier? = nil,
        variant: Variant? = nil,
        precision: Precision? = nil,
        quantization: Quantization? = nil,
        defaults: Defaults? = nil,
        supports: [Capability]? = nil,
        components: Components? = nil,
        upstreamRepoId: String? = nil,
        createdAt: Date? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.id = id
        self.engine = engine
        self.family = family
        self.tier = tier
        self.variant = variant
        self.precision = precision
        self.quantization = quantization
        self.defaults = defaults
        self.supports = supports
        self.components = components
        self.upstreamRepoId = upstreamRepoId
        self.createdAt = createdAt
    }

    // MARK: - IO

    public enum ManifestError: LocalizedError, Sendable {
        case missing(URL)

        public var errorDescription: String? {
            switch self {
            case .missing(let modelRoot):
                return "Missing \(MereRunModelManifest.filename) in model directory: \(modelRoot.path)"
            }
        }
    }

    public static func url(in modelRoot: URL) -> URL {
        modelRoot.appendingPathComponent(Self.filename)
    }

    public static func loadIfPresent(from modelRoot: URL, fileManager: FileManager = .default) throws -> MereRunModelManifest? {
        let url = url(in: modelRoot)
        guard fileManager.fileExists(atPath: url.path) else {
            return nil
        }
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(MereRunModelManifest.self, from: data)
    }

    public static func loadRequired(from modelRoot: URL, fileManager: FileManager = .default) throws -> MereRunModelManifest {
        let url = url(in: modelRoot)
        guard fileManager.fileExists(atPath: url.path) else {
            throw ManifestError.missing(modelRoot)
        }
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(MereRunModelManifest.self, from: data)
    }

    public func write(to modelRoot: URL) throws {
        let url = Self.url(in: modelRoot)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(self)
        try data.write(to: url, options: [.atomic])
    }

    // MARK: - Templates (known models)

    /// If `modelId` is one of `ModelResolver.ModelID`, writes a manifest into `modelRoot` and returns it.
    public static func writeTemplateIfKnown(
        modelId: String,
        to modelRoot: URL,
        createdAt: Date = Date()
    ) throws -> MereRunModelManifest? {
        guard let known = ModelResolver.ModelID(rawValue: modelId) else {
            return nil
        }
        do {
            if let existing = try MereRunModelManifest.loadIfPresent(from: modelRoot) {
                return existing
            }
        } catch {
            // Ignore decode errors and overwrite with a fresh template below.
        }

        let manifest = template(for: known, createdAt: createdAt)
        try manifest.write(to: modelRoot)
        return manifest
    }

    /// Returns a manifest template for a known `ModelResolver.ModelID`.
    ///
    /// This is used by the app to write manifests after managed downloads, and by the CLI for validation.
    public static func template(for modelID: ModelResolver.ModelID, createdAt: Date = Date()) -> MereRunModelManifest {
        let defaultComponents = Components(
            tokenizer: .local(path: "tokenizer"),
            textEncoder: .local(path: "text_encoder"),
            transformer: .local(path: "transformer"),
            vae: .local(path: "vae"),
            scheduler: .local(path: "scheduler")
        )
        let kleinSharedComponents = Components(
            tokenizer: .local(path: "tokenizer"),
            textEncoder: .local(path: "text_encoder"),
            transformer: nil,
            vae: .local(path: "vae"),
            scheduler: .local(path: "scheduler")
        )
        let kleinHybridComponents = Components(
            tokenizer: .anyOf([
                .local(path: "tokenizer"),
                .model(modelID: ModelResolver.ModelID.kleinShared.rawValue, path: "tokenizer"),
            ]),
            textEncoder: .anyOf([
                .local(path: "text_encoder"),
                .model(modelID: ModelResolver.ModelID.kleinShared.rawValue, path: "text_encoder"),
            ]),
            transformer: .local(path: "transformer"),
            vae: .anyOf([
                .local(path: "vae"),
                .model(modelID: ModelResolver.ModelID.kleinShared.rawValue, path: "vae"),
            ]),
            scheduler: .anyOf([
                .local(path: "scheduler"),
                .model(modelID: ModelResolver.ModelID.kleinShared.rawValue, path: "scheduler"),
            ])
        )
        let mebotTextComponents = Components(
            tokenizer: .anyOf([
                .local(path: "tokenizer"),
                .model(modelID: ModelResolver.ModelID.kleinNano.rawValue, path: "tokenizer"),
                .model(modelID: ModelResolver.ModelID.kleinMax.rawValue, path: "tokenizer")
            ]),
            textEncoder: .anyOf([
                .local(path: "text_encoder"),
                .model(modelID: ModelResolver.ModelID.kleinNano.rawValue, path: "text_encoder"),
                .model(modelID: ModelResolver.ModelID.kleinMax.rawValue, path: "text_encoder")
            ]),
            transformer: nil,
            vae: nil,
            scheduler: nil
        )
        let q35TextComponents = Components(
            tokenizer: .local(path: "."),
            textEncoder: .local(path: "."),
            transformer: nil,
            vae: nil,
            scheduler: nil
        )
        let genericTextComponents = Components(
            tokenizer: .local(path: "."),
            textEncoder: .local(path: "."),
            transformer: nil,
            vae: nil,
            scheduler: nil
        )
        let gemma4TextComponents = Components(
            tokenizer: .local(path: "."),
            textEncoder: .local(path: "."),
            transformer: nil,
            vae: nil,
            scheduler: nil
        )
        let samComponents = Components(
            tokenizer: .local(path: "tokenizer"),
            textEncoder: nil,
            transformer: nil,
            vae: nil,
            scheduler: nil
        )
        let falconPerceptionComponents = Components(
            tokenizer: .anyOf([
                .local(path: "tokenizer"),
                .local(path: "."),
            ]),
            textEncoder: nil,
            transformer: nil,
            vae: nil,
            scheduler: nil
        )

        switch modelID {
        case .kleinNano:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .flux2Klein,
                family: .klein,
                tier: .nano,
                variant: .distilled,
                precision: .int4,
                quantization: Quantization(bits: 4, groupSize: 64, scheme: "mlx-quantized-linear"),
                defaults: Defaults(steps: 4, cfg: 1.0),
                supports: [.txt2img, .referenceEdit, .loraInference],
                components: defaultComponents,
                upstreamRepoId: "stereovoid/flux2-klein-4b-4bit",
                createdAt: createdAt
            )
        case .kleinMax:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .flux2Klein,
                family: .klein,
                tier: .max,
                variant: .distilled,
                precision: .bf16,
                defaults: Defaults(steps: 4, cfg: 1.0),
                supports: [.txt2img, .referenceEdit, .loraInference],
                components: kleinHybridComponents,
                upstreamRepoId: "black-forest-labs/FLUX.2-klein-4B",
                createdAt: createdAt
            )
        case .kleinShared:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .flux2Klein,
                family: .klein,
                precision: .unknown,
                supports: [.txt2img, .referenceEdit, .loraInference],
                components: kleinSharedComponents,
                createdAt: createdAt
            )
        case .mebot:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .flux2Klein,
                family: .klein,
                tier: .nano,
                variant: .distilled,
                precision: .unknown,
                supports: [.chat, .loraInference],
                components: mebotTextComponents,
                createdAt: createdAt
            )
        case .gemma4:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .gemma4,
                family: .gemma,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                defaults: nil,
                supports: [.chat],
                components: gemma4TextComponents,
                upstreamRepoId: Gemma4Resources.defaultUpstreamModelId,
                createdAt: createdAt
            )
        case .gemma4Nano:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .gemma4,
                family: .gemma,
                tier: .nano,
                variant: .standard,
                precision: .bf16,
                defaults: nil,
                supports: [.chat],
                components: gemma4TextComponents,
                upstreamRepoId: Gemma4Resources.nanoUpstreamModelId,
                createdAt: createdAt
            )
        case .gemma4Max:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .gemma4,
                family: .gemma,
                tier: .max,
                variant: .standard,
                precision: .bf16,
                defaults: nil,
                supports: [.chat],
                components: gemma4TextComponents,
                upstreamRepoId: Gemma4Resources.maxUpstreamModelId,
                createdAt: createdAt
            )
        case .gemma4Turbo:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .gemma4,
                family: .gemma,
                tier: .turbo,
                variant: .standard,
                precision: .int4,
                defaults: nil,
                supports: [.chat],
                components: gemma4TextComponents,
                upstreamRepoId: Gemma4Resources.turboUpstreamModelId,
                createdAt: createdAt
            )
        case .q35:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .max,
                variant: .base,
                precision: .int4,
                quantization: Quantization(bits: 4, groupSize: 64, scheme: "mlx-affine"),
                defaults: nil,
                supports: [.chat],
                components: q35TextComponents,
                upstreamRepoId: "\(Q35Resources.upstreamRepoId)@\(Q35Resources.upstreamRevision)",
                createdAt: createdAt
            )
        case .q35Nano:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .nano,
                variant: .base,
                precision: .int4,
                quantization: Quantization(bits: 4, groupSize: 64, scheme: "mlx-affine"),
                defaults: nil,
                supports: [.chat],
                components: q35TextComponents,
                upstreamRepoId: "\(Q35Resources.nanoUpstreamRepoId)@\(Q35Resources.nanoUpstreamRevision)",
                createdAt: createdAt
            )
        case .qwen35Agent9B:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen3Coder,
                family: .code,
                tier: .nano,
                variant: .standard,
                precision: .int4,
                defaults: nil,
                supports: [.chat, .codeGeneration],
                components: nil,
                upstreamRepoId: "\(AgentModelResources.qwen35NineBRepoId)@\(AgentModelResources.qwen35NineBRevision)",
                createdAt: createdAt
            )
        case .deepseekV4Flash:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .deepseekV4Flash,
                family: .deepseek,
                tier: .max,
                variant: .standard,
                precision: .int4,
                quantization: Quantization(bits: 2, groupSize: 32, scheme: "iq2-xxs-imatrix"),
                defaults: nil,
                supports: [.chat],
                components: nil,
                upstreamRepoId: "\(DeepseekV4FlashResources.defaultRepoId)@\(DeepseekV4FlashResources.defaultRevision)",
                createdAt: createdAt
            )
        case .kleinBase:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .flux2Klein,
                family: .klein,
                tier: .base,
                variant: .base,
                precision: .unknown,
                defaults: Defaults(steps: 50, cfg: 1.0),
                supports: [.txt2img, .referenceEdit, .loraInference, .loraTraining],
                components: kleinHybridComponents,
                upstreamRepoId: "black-forest-labs/FLUX.2-klein-base-4B",
                createdAt: createdAt
            )
        case .zetaNano:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .zimageTurbo,
                family: .zimage,
                tier: .nano,
                variant: .distilled,
                precision: .int4,
                quantization: Quantization(bits: 4, groupSize: 64, scheme: "mlx-quantized-linear"),
                defaults: Defaults(steps: 4, cfg: 1.0),
                supports: [.txt2img, .img2img, .loraInference],
                components: defaultComponents,
                upstreamRepoId: "filipstrand/Z-Image-Turbo-mflux-4bit@main",
                createdAt: createdAt
            )
        case .zetaMax:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .zimageTurbo,
                family: .zimage,
                tier: .max,
                variant: .distilled,
                precision: .unknown,
                defaults: Defaults(steps: 4, cfg: 1.0),
                supports: [.txt2img, .img2img, .loraInference],
                components: defaultComponents,
                upstreamRepoId: "Tongyi-MAI/Z-Image-Turbo",
                createdAt: createdAt
            )
        case .zetaBase:
            let hybridComponents = Components(
                tokenizer: .anyOf([
                    .local(path: "tokenizer"),
                    .model(modelID: ModelResolver.ModelID.zetaMax.rawValue, path: "tokenizer"),
                    .model(modelID: ModelResolver.ModelID.zetaNano.rawValue, path: "tokenizer"),
                ]),
                textEncoder: .anyOf([
                    .local(path: "text_encoder"),
                    .model(modelID: ModelResolver.ModelID.zetaMax.rawValue, path: "text_encoder"),
                    .model(modelID: ModelResolver.ModelID.zetaNano.rawValue, path: "text_encoder"),
                ]),
                transformer: .local(path: "transformer"),
                vae: .anyOf([
                    .local(path: "vae"),
                    .model(modelID: ModelResolver.ModelID.zetaMax.rawValue, path: "vae"),
                    .model(modelID: ModelResolver.ModelID.zetaNano.rawValue, path: "vae"),
                ]),
                scheduler: .anyOf([
                    .local(path: "scheduler"),
                    .model(modelID: ModelResolver.ModelID.zetaMax.rawValue, path: "scheduler"),
                    .model(modelID: ModelResolver.ModelID.zetaNano.rawValue, path: "scheduler"),
                ])
            )
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .zimageTurbo,
                family: .zimage,
                tier: .base,
                variant: .base,
                precision: .unknown,
                defaults: nil,
                supports: [.txt2img, .img2img, .loraInference, .loraTraining],
                components: hybridComponents,
                upstreamRepoId: "Tongyi-MAI/Z-Image@main",
                createdAt: createdAt
            )
        case .hidreamO1:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .hidreamO1,
                family: .hidream,
                tier: .max,
                variant: .base,
                precision: .bf16,
                defaults: Defaults(steps: 50, cfg: 5.0),
                supports: [.txt2img, .referenceEdit, .subjectPersonalization],
                components: Components(
                    tokenizer: .local(path: "."),
                    textEncoder: .local(path: "."),
                    transformer: .local(path: "."),
                    vae: nil,
                    scheduler: nil
                ),
                upstreamRepoId: "HiDream-ai/HiDream-O1-Image",
                createdAt: createdAt
            )
        case .hidreamO1Dev:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .hidreamO1,
                family: .hidream,
                tier: .max,
                variant: .distilled,
                precision: .bf16,
                defaults: Defaults(steps: 28, cfg: 0.0),
                supports: [.txt2img, .referenceEdit, .subjectPersonalization],
                components: Components(
                    tokenizer: .local(path: "."),
                    textEncoder: .local(path: "."),
                    transformer: .local(path: "."),
                    vae: nil,
                    scheduler: nil
                ),
                upstreamRepoId: "HiDream-ai/HiDream-O1-Image-Dev",
                createdAt: createdAt
            )
        case .visionSegmentSAM31:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .samSegmentation,
                family: .sam,
                tier: .latest,
                variant: .standard,
                precision: .unknown,
                defaults: nil,
                supports: [.visionSegmentation, .visionTracking],
                components: samComponents,
                upstreamRepoId: "facebook/sam3.1",
                createdAt: createdAt
            )
        case .visionGroundFalconPerception:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .falconPerception,
                family: .falcon,
                tier: .latest,
                variant: .standard,
                precision: .unknown,
                defaults: nil,
                supports: [.visionGrounding, .visionDetection, .visionSegmentation],
                components: falconPerceptionComponents,
                upstreamRepoId: "tiiuae/Falcon-Perception",
                createdAt: createdAt
            )
        case .psiAgent:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .psiChat,
                family: .psi,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                defaults: nil,
                supports: [.chat],
                components: genericTextComponents,
                createdAt: createdAt
            )
        case .qwen3TTSNano:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen3TTS,
                family: .tts,
                tier: .nano,
                variant: .standard,
                precision: .bf16,
                defaults: nil,
                supports: [.speechSynthesis],
                components: genericTextComponents,
                upstreamRepoId: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                createdAt: createdAt
            )
        case .qwen3TTSCustomVoice:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen3TTS,
                family: .tts,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                defaults: nil,
                supports: [.speechSynthesis],
                components: genericTextComponents,
                upstreamRepoId: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                createdAt: createdAt
            )
        case .qwen3ASR:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen3ASR,
                family: .asr,
                tier: .latest,
                variant: .standard,
                precision: .int8,
                defaults: nil,
                supports: [.speechRecognition],
                components: genericTextComponents,
                upstreamRepoId: "mlx-community/Qwen3-ASR-1.7B-8bit",
                createdAt: createdAt
            )
        case .parakeetASR:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .parakeetASR,
                family: .asr,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                defaults: nil,
                supports: [.speechRecognition],
                components: genericTextComponents,
                upstreamRepoId: "mlx-community/parakeet-tdt-0.6b-v3",
                createdAt: createdAt
            )
        case .qwen3Code:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen3Coder,
                family: .code,
                tier: .latest,
                variant: .standard,
                precision: .int4,
                defaults: nil,
                supports: [.chat, .codeGeneration],
                components: nil,
                upstreamRepoId: CodeGenResources.defaultRepoId,
                createdAt: createdAt
            )
        case .qwen3Embedding:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen3Embedding,
                family: .embed,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                defaults: nil,
                supports: [.textEmbedding],
                components: genericTextComponents,
                upstreamRepoId: Qwen3EmbeddingCatalog.defaultRepoId,
                createdAt: createdAt
            )
        case .privacyFilter:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .openAIPrivacyFilter,
                family: .privacy,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                defaults: nil,
                supports: [.textAnonymization],
                components: genericTextComponents,
                upstreamRepoId: OpenAIPrivacyFilterCatalog.defaultRepoId,
                createdAt: createdAt
            )
        case .lightOnOCR:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .lightOnOCR,
                family: .ocr,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                defaults: nil,
                supports: [.visionOCR],
                components: Components(
                    tokenizer: .anyOf([.local(path: "tokenizer"), .local(path: ".")]),
                    textEncoder: .local(path: "."),
                    transformer: nil,
                    vae: nil,
                    scheduler: nil
                ),
                upstreamRepoId: "lightonai/LightOnOCR-2-1B",
                createdAt: createdAt
            )
        case .aceStep:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .aceStep,
                family: .music,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                defaults: Defaults(steps: 8, cfg: 1.0),
                supports: [.musicGeneration],
                components: nil,
                upstreamRepoId: "ACE-Step/Ace-Step1.5",
                createdAt: createdAt
            )
        case .ltxVideoAV:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .ltxVideo,
                family: .video,
                tier: .latest,
                variant: .distilled,
                precision: .bf16,
                defaults: Defaults(steps: 8, cfg: 1.0),
                supports: [.videoGeneration],
                components: nil,
                upstreamRepoId: "mlx-community/LTX-2-distilled-bf16",
                createdAt: createdAt
            )
        }
    }
}
