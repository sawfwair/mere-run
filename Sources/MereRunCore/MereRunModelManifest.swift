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
    public static let currentSchemaVersion: Int = 3
    public static let filename: String = "mererun_model.json"

    public enum Engine: String, Codable, CaseIterable, Hashable, Sendable {
        /// MereRun family (FLUX.2 Klein based).
        case flux2Klein = "flux2-klein"
        /// Zeta family (Z-Image Turbo based).
        case zimageTurbo = "zimage-turbo"
        /// HiDream O1 unified pixel transformer family.
        case hidreamO1 = "hidream-o1"
        /// Krea 2 text-to-image family.
        case krea2 = "krea-2"
        /// Ideogram 4 text-to-image family.
        case ideogram4 = "ideogram-4"
        /// Gemma 4 family via the native Swift runtime.
        case gemma4 = "gemma-4"
        /// LiquidAI LFM2 family via the native Swift runtime.
        case lfm2 = "lfm2"
        /// Q35 family (Qwen3.5 hybrid MoE + hybrid attention).
        case qwen35HybridMoE = "qwen3.5-hybrid-moe"
        /// SAM image segmentation family.
        case samSegmentation = "sam-segmentation"
        /// Falcon Perception grounded detection and segmentation family.
        case falconPerception = "falcon-perception"
        /// InsightFace Buffalo-L face detection and identity-embedding family.
        case insightFace = "insightface"
        /// MoGe-2 metric monocular geometry family.
        case moge2 = "moge-2"
        /// Video Depth Anything temporal depth family.
        case videoDepthAnything = "video-depth-anything"
        /// Depth Anything 3 multi-view geometry family.
        case depthAnything3 = "depth-anything-3"
        /// TripoSR single-image object reconstruction family.
        case tripoSR = "triposr"
        /// InstantMesh multi-view object reconstruction family.
        case instantMesh = "instantmesh"
        /// Microsoft TRELLIS.2 image-to-PBR-mesh family.
        case trellis2 = "trellis2"
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
        /// North Mini Code GGUF family via the native llama.cpp runtime.
        case northMiniCode = "north-mini-code"
        /// LightOn OCR family.
        case lightOnOCR = "lighton-ocr"
        /// ACE-Step music family.
        case aceStep = "ace-step"
        /// Magenta RealTime 2 streaming music family.
        case magentaRT2 = "magenta-rt2"
        /// MuScriptor multi-instrument audio transcription family.
        case muScriptor = "muscriptor"
        /// Sony Research Woosh sound-effect generation family.
        case woosh = "woosh"
        /// MMAudio synchronized video-to-audio and text-to-audio family.
        case mmaudio = "mmaudio"
        /// LTX video family.
        case ltxVideo = "ltx-video"
        /// Wan2 native video family.
        case wanVideo = "wan-video"
        /// Psi agent chat family.
        case psiChat = "psi-chat"
        /// DeepSeek V4 Flash family, served by the bundled `ds4-server` subprocess.
        case deepseekV4Flash = "deepseek-v4-flash"
    }

    public enum Family: String, Codable, CaseIterable, Hashable, Sendable {
        case klein
        case zimage
        case hidream
        case krea
        case ideogram
        case gemma
        case liquid
        case qwen
        case sam
        case falcon
        case face
        case geometry
        case depth
        case threeD = "3d"
        case tts
        case asr
        case embed
        case privacy
        case code
        case ocr
        case music
        case sfx
        case video
        case psi
        case deepseek
    }

    public enum Tier: String, Codable, CaseIterable, Hashable, Sendable {
        case nano
        case small
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
        case int1
        case int2
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
        case visionChat = "vision_chat"
        case visionOCR = "vision_ocr"
        case musicGeneration = "music_generation"
        case musicTranscription = "music_transcription"
        case videoGeneration = "video_generation"
        case audioToVideoGeneration = "audio_to_video_generation"
        case loraInference = "lora_inference"
        case loraTraining = "lora_training"
        case visionSegmentation = "vision_segmentation"
        case visionTracking = "vision_tracking"
        case visionGrounding = "vision_grounding"
        case visionDetection = "vision_detection"
        case faceDetection = "face_detection"
        case faceLandmarks = "face_landmarks"
        case faceEmbedding = "face_embedding"
        case faceVerification = "face_verification"
        case soundEffectGeneration = "sound_effect_generation"
        case soundEffectEmbedding = "sound_effect_embedding"
        case videoToAudioGeneration = "video_to_audio_generation"
        case metricDepth = "metric_depth"
        case relativeDepth = "relative_depth"
        case temporalDepth = "temporal_depth"
        case surfaceNormals = "surface_normals"
        case pointMap = "point_map"
        case cameraIntrinsics = "camera_intrinsics"
        case cameraExtrinsics = "camera_extrinsics"
        case pointCloud = "point_cloud"
        case imageTo3D = "image_to_3d"
        case multiViewReconstruction = "multi_view_reconstruction"
        case meshGeneration = "mesh_generation"
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
        public var unconditionalTransformer: ComponentRef?
        public var vae: ComponentRef?
        public var scheduler: ComponentRef?

        public init(
            tokenizer: ComponentRef? = nil,
            textEncoder: ComponentRef? = nil,
            transformer: ComponentRef? = nil,
            unconditionalTransformer: ComponentRef? = nil,
            vae: ComponentRef? = nil,
            scheduler: ComponentRef? = nil
        ) {
            self.tokenizer = tokenizer
            self.textEncoder = textEncoder
            self.transformer = transformer
            self.unconditionalTransformer = unconditionalTransformer
            self.vae = vae
            self.scheduler = scheduler
        }

        private enum CodingKeys: String, CodingKey {
            case tokenizer
            case textEncoder = "text_encoder"
            case transformer
            case unconditionalTransformer = "unconditional_transformer"
            case vae
            case scheduler
        }
    }

    public struct Defaults: Codable, Hashable, Sendable {
        public var steps: Int?
        public var cfg: Double?
        public var sigmaShift: Double?

        public init(steps: Int? = nil, cfg: Double? = nil, sigmaShift: Double? = nil) {
            self.steps = steps
            self.cfg = cfg
            self.sigmaShift = sigmaShift
        }

        private enum CodingKeys: String, CodingKey {
            case steps
            case cfg
            case sigmaShift = "sigma_shift"
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

    public struct SourceProvenance: Codable, Hashable, Sendable {
        public var role: String
        public var repository: String
        public var revision: String
        public var destinationPath: String?

        public init(
            role: String,
            repository: String,
            revision: String,
            destinationPath: String? = nil
        ) {
            self.role = role
            self.repository = repository
            self.revision = revision
            self.destinationPath = destinationPath
        }

        private enum CodingKeys: String, CodingKey {
            case role
            case repository
            case revision
            case destinationPath = "destination_path"
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

    /// Exact repositories and requested revisions materialized into this managed install.
    public var sources: [SourceProvenance]?

    /// Third-party model/component terms that required explicit acknowledgement before download.
    public var usageTerms: [ManagedModelUsageTerm]?

    /// True only when the managed installer was invoked with explicit acknowledgement.
    public var usageTermsAcknowledged: Bool?

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
        sources: [SourceProvenance]? = nil,
        usageTerms: [ManagedModelUsageTerm]? = nil,
        usageTermsAcknowledged: Bool? = nil,
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
        self.sources = sources
        self.usageTerms = usageTerms
        self.usageTermsAcknowledged = usageTermsAcknowledged
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
        usageTermsAcknowledged: Bool = false,
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

        var manifest = template(for: known, createdAt: createdAt)
        if manifest.usageTerms?.isEmpty == false {
            manifest.usageTermsAcknowledged = usageTermsAcknowledged
        }
        try manifest.write(to: modelRoot)
        return manifest
    }

    /// Returns a manifest template for a known `ModelResolver.ModelID`.
    ///
    /// This is used by the app to write manifests after managed downloads, and by the CLI for validation.
    public static func template(for modelID: ModelResolver.ModelID, createdAt: Date = Date()) -> MereRunModelManifest {
        var manifest = baseTemplate(for: modelID, createdAt: createdAt)
        guard let spec = ManagedModelCatalog.spec(for: modelID.rawValue) else {
            return manifest
        }
        manifest.sources = sourceProvenance(for: spec)
        manifest.usageTerms = spec.usageRestriction?.terms
        return manifest
    }

    private static func sourceProvenance(for spec: ManagedModelSpec) -> [SourceProvenance]? {
        var sources: [SourceProvenance] = []
        if let primary = spec.hubFallback {
            sources.append(
                SourceProvenance(
                    role: "primary",
                    repository: primary.repoId,
                    revision: primary.revision
                )
            )
        }
        sources.append(contentsOf: spec.mountedHubFallbacks.map { mounted in
            SourceProvenance(
                role: "component",
                repository: mounted.hubFallback.repoId,
                revision: mounted.hubFallback.revision,
                destinationPath: mounted.destinationPath
            )
        })
        if let upstreamRepoId = spec.upstreamRepoId,
           !sources.contains(where: { $0.repository == upstreamRepoId }) {
            sources.append(
                SourceProvenance(
                    role: "upstream_reference",
                    repository: upstreamRepoId,
                    revision: spec.upstreamRevision ?? "unspecified"
                )
            )
        }
        return sources.isEmpty ? nil : sources
    }

    private static func baseTemplate(for modelID: ModelResolver.ModelID, createdAt: Date) -> MereRunModelManifest {
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
        let bonsaiComponents = Components(
            tokenizer: .local(path: "tokenizer"),
            textEncoder: .local(path: "text_encoder-mlx-4bit"),
            transformer: .local(path: "transformer-packed-mflux"),
            vae: .local(path: "vae"),
            scheduler: .local(path: "scheduler")
        )
        let ideogram4Components = Components(
            tokenizer: .local(path: "tokenizer"),
            textEncoder: .local(path: "text_encoder"),
            transformer: .local(path: "transformer"),
            unconditionalTransformer: .local(path: "unconditional_transformer"),
            vae: .local(path: "vae"),
            scheduler: .local(path: "scheduler")
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
        case .klein9B:
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
                upstreamRepoId: "black-forest-labs/FLUX.2-klein-9B",
                createdAt: createdAt
            )
        case .kleinBase9B:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .flux2Klein,
                family: .klein,
                tier: .max,
                variant: .base,
                precision: .bf16,
                defaults: Defaults(steps: 50, cfg: 1.0),
                supports: [.txt2img, .referenceEdit, .loraInference, .loraTraining],
                components: kleinHybridComponents,
                upstreamRepoId: "black-forest-labs/FLUX.2-klein-base-9B",
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
        case .bonsaiBinary:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .flux2Klein,
                family: .klein,
                tier: .nano,
                variant: .distilled,
                precision: .int1,
                quantization: Quantization(bits: 1, groupSize: 128, scheme: "prism-packed-affine-binary"),
                defaults: Defaults(steps: 4, cfg: 1.0, sigmaShift: 3.0),
                supports: [.txt2img],
                components: bonsaiComponents,
                upstreamRepoId: "prism-ml/bonsai-image-binary-4B-mlx-1bit@main",
                createdAt: createdAt
            )
        case .bonsaiTernary:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .flux2Klein,
                family: .klein,
                tier: .nano,
                variant: .distilled,
                precision: .int2,
                quantization: Quantization(bits: 2, groupSize: 128, scheme: "mlx-packed-affine-ternary"),
                defaults: Defaults(steps: 4, cfg: 1.0, sigmaShift: 3.0),
                supports: [.txt2img],
                components: bonsaiComponents,
                upstreamRepoId: "prism-ml/bonsai-image-ternary-4B-mlx-2bit@main",
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
        case .gemma4TwelveB:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .gemma4,
                family: .gemma,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                defaults: nil,
                supports: [.chat, .codeGeneration],
                components: gemma4TextComponents,
                upstreamRepoId: Gemma4Resources.twelveBUpstreamModelId,
                createdAt: createdAt
            )
        case .gemma4TwelveB4Bit:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .gemma4,
                family: .gemma,
                tier: .latest,
                variant: .standard,
                precision: .int4,
                quantization: Quantization(bits: 4, groupSize: 64, scheme: "mlx-affine"),
                defaults: nil,
                supports: [.chat, .codeGeneration],
                components: gemma4TextComponents,
                upstreamRepoId: Gemma4Resources.twelveB4BitUpstreamModelId
                    + "@\(Gemma4Resources.twelveB4BitUpstreamRevision)",
                createdAt: createdAt
            )
        case .ltxGemma3TwelveB4Bit:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .gemma4,
                family: .gemma,
                tier: .latest,
                variant: .standard,
                precision: .int4,
                quantization: Quantization(bits: 4, groupSize: 64, scheme: "mlx-affine"),
                defaults: nil,
                supports: [],
                components: gemma4TextComponents,
                upstreamRepoId: "mlx-community/gemma-3-12b-it-4bit",
                createdAt: createdAt
            )
        case .gemma4VisionTwelveB:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .gemma4,
                family: .gemma,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                defaults: nil,
                supports: [.chat, .codeGeneration, .visionChat],
                components: gemma4TextComponents,
                upstreamRepoId: Gemma4Resources.twelveBUpstreamModelId,
                createdAt: createdAt
            )
        case .gemma4TwelveBMTP:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .gemma4,
                family: .gemma,
                tier: .latest,
                variant: .standard,
                precision: .bf16,
                defaults: nil,
                supports: [],
                components: nil,
                upstreamRepoId: Gemma4MTPResources.upstreamModelId,
                createdAt: createdAt
            )
        case .q36Nano:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .nano,
                variant: .base,
                precision: .int4,
                quantization: Quantization(bits: 4, groupSize: 64, scheme: "mlx-optiq-mixed-affine"),
                defaults: nil,
                supports: [.chat],
                components: q35TextComponents,
                upstreamRepoId: "\(Q35Resources.q36NanoUpstreamRepoId)@\(Q35Resources.q36NanoUpstreamRevision)",
                createdAt: createdAt
            )
        case .bonsai27B1Bit:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .int1,
                quantization: Quantization(bits: 1, groupSize: 128, scheme: "prism-packed-affine-binary"),
                defaults: nil,
                supports: [.chat, .codeGeneration, .visionChat],
                components: q35TextComponents,
                upstreamRepoId: "\(Q35Resources.bonsai27B1BitUpstreamRepoId)"
                    + "@\(Q35Resources.bonsai27B1BitUpstreamRevision)",
                createdAt: createdAt
            )
        case .bonsai27B2Bit:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .latest,
                variant: .standard,
                precision: .int2,
                quantization: Quantization(bits: 2, groupSize: 128, scheme: "prism-packed-affine-ternary"),
                defaults: nil,
                supports: [.chat, .codeGeneration, .visionChat],
                components: q35TextComponents,
                upstreamRepoId: "\(Q35Resources.bonsai27B2BitUpstreamRepoId)"
                    + "@\(Q35Resources.bonsai27B2BitUpstreamRevision)",
                createdAt: createdAt
            )
        case .lfm25A1B8Bit:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .lfm2,
                family: .liquid,
                tier: .nano,
                variant: .standard,
                precision: .int8,
                quantization: Quantization(bits: 8, groupSize: 64, scheme: "mlx-affine"),
                defaults: nil,
                supports: [.chat],
                components: q35TextComponents,
                upstreamRepoId: "\(LFM2Resources.upstreamRepoId)@\(LFM2Resources.upstreamRevision)",
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
        case .ornith9B:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen35HybridMoE,
                family: .code,
                tier: .nano,
                variant: .standard,
                precision: .int4,
                quantization: Quantization(bits: 4, groupSize: 64, scheme: "mlx-optiq-mixed-affine"),
                defaults: nil,
                supports: [.chat, .codeGeneration],
                components: q35TextComponents,
                upstreamRepoId: "\(Q35Resources.ornith9BUpstreamRepoId)"
                    + "@\(Q35Resources.ornith9BUpstreamRevision)",
                createdAt: createdAt
            )
        case .ornith35BMLX:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen35HybridMoE,
                family: .code,
                tier: .small,
                variant: .standard,
                precision: .int4,
                quantization: Quantization(bits: 4, groupSize: 64, scheme: "mlx-affine-moe"),
                defaults: nil,
                supports: [.chat, .codeGeneration],
                components: q35TextComponents,
                upstreamRepoId: "\(Q35Resources.ornith35BMLXUpstreamRepoId)"
                    + "@\(Q35Resources.ornith35BMLXUpstreamRevision)",
                createdAt: createdAt
            )
        case .ornith35B:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen3Coder,
                family: .code,
                tier: .small,
                variant: .standard,
                precision: .int4,
                quantization: Quantization(bits: 4, groupSize: 64, scheme: "gguf-q4-k-m"),
                defaults: nil,
                supports: [.chat, .codeGeneration],
                components: nil,
                upstreamRepoId: "\(Ornith35BCodeResources.upstreamRepoId)"
                    + "@\(Ornith35BCodeResources.upstreamRevision)",
                createdAt: createdAt
            )
        case .northMiniCode:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .northMiniCode,
                family: .code,
                tier: .small,
                variant: .standard,
                precision: .int4,
                defaults: nil,
                supports: [.chat, .codeGeneration],
                components: nil,
                upstreamRepoId: "\(NorthMiniCodeResources.upstreamRepoId)"
                    + "@\(NorthMiniCodeResources.upstreamRevision)",
                createdAt: createdAt
            )
        case .q36NanoGGUF:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen35HybridMoE,
                family: .qwen,
                tier: .nano,
                variant: .base,
                precision: .int4,
                quantization: Quantization(bits: 4, groupSize: 64, scheme: "gguf-q4-k-m"),
                defaults: nil,
                supports: [.chat],
                components: q35TextComponents,
                upstreamRepoId: "unsloth/Qwen3.6-35B-A3B-GGUF@main",
                createdAt: createdAt
            )
        case .infinityParser2Flash, .infinityParser2Pro, .infinityParser2ProInt8:
            let isPro = modelID != .infinityParser2Flash
            let isProInt8 = modelID == .infinityParser2ProInt8
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .qwen35HybridMoE,
                family: .ocr,
                tier: isPro ? .max : .nano,
                variant: .standard,
                precision: isProInt8 ? .int8 : .bf16,
                quantization: isProInt8
                    ? Quantization(bits: 8, groupSize: 64, scheme: "mlx-quantized-linear")
                    : nil,
                defaults: nil,
                supports: [.chat, .visionChat, .visionOCR],
                components: q35TextComponents,
                upstreamRepoId: isProInt8
                    ? "\(Q35Resources.infinityParser2ProInt8UpstreamRepoId)@\(Q35Resources.infinityParser2ProInt8UpstreamRevision)"
                    : isPro
                    ? "\(Q35Resources.infinityParser2ProUpstreamRepoId)@\(Q35Resources.infinityParser2ProUpstreamRevision)"
                    : "\(Q35Resources.infinityParser2FlashUpstreamRepoId)@\(Q35Resources.infinityParser2FlashUpstreamRevision)",
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
                upstreamRepoId: "filipstrand/Z-Image-Turbo-mflux-4bit@b3a8f31115a11f2f9e2fa0bfbc8d78dcc3e6568b",
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
        case .krea2Raw:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .krea2,
                family: .krea,
                tier: .max,
                variant: .base,
                precision: .bf16,
                defaults: Defaults(steps: 52, cfg: 3.5, sigmaShift: Double(Krea2SampleBuilder.defaultMu)),
                supports: [.txt2img, .loraTraining],
                components: defaultComponents,
                upstreamRepoId: "\(Krea2RawResources.upstreamRepoId)@\(Krea2RawResources.upstreamRevision)",
                createdAt: createdAt
            )
        case .krea2Turbo:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .krea2,
                family: .krea,
                tier: .turbo,
                variant: .distilled,
                precision: .bf16,
                defaults: Defaults(steps: 8, cfg: 0.0, sigmaShift: Double(Krea2SampleBuilder.defaultMu)),
                supports: [.txt2img, .loraInference],
                components: defaultComponents,
                upstreamRepoId: "\(Krea2Resources.upstreamRepoId)@\(Krea2Resources.upstreamRevision)",
                createdAt: createdAt
            )
        case .ideogram4SDNQUInt4:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .ideogram4,
                family: .ideogram,
                tier: .latest,
                variant: .standard,
                precision: .int4,
                quantization: Quantization(bits: 4, groupSize: 64, scheme: "sdnq-uint4"),
                defaults: Defaults(steps: 20, cfg: 7.0),
                supports: [.txt2img],
                components: ideogram4Components,
                upstreamRepoId: "\(Ideogram4Resources.upstreamRepoId)@\(Ideogram4Resources.upstreamRevision)",
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
        case .visionFaceBuffaloL:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .insightFace,
                family: .face,
                tier: .latest,
                variant: .standard,
                precision: .fp32,
                defaults: nil,
                supports: [.faceDetection, .faceLandmarks, .faceEmbedding, .faceVerification],
                components: nil,
                upstreamRepoId: "deepghs/insightface@4e1f33d3fe0e50a0945f3a53ab94ae8977ae7ddb",
                createdAt: createdAt
            )
        case .visionGeometryMoGe2Small:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .moge2,
                family: .geometry,
                tier: .small,
                variant: .standard,
                precision: .fp32,
                supports: [.metricDepth, .surfaceNormals, .pointMap, .cameraIntrinsics, .pointCloud],
                components: nil,
                upstreamRepoId: "Ruicheng/moge-2-vits-normal-onnx@e50ffda41565591092adea54c6ac83d6212e1e23",
                createdAt: createdAt
            )
        case .visionDepthVDASmall, .visionDepthVDASmallMetric:
            let metric = modelID == .visionDepthVDASmallMetric
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .videoDepthAnything,
                family: .depth,
                tier: .small,
                variant: .standard,
                precision: .fp32,
                supports: metric ? [.metricDepth, .temporalDepth] : [.relativeDepth, .temporalDepth],
                components: nil,
                upstreamRepoId: metric
                    ? "depth-anything/Metric-Video-Depth-Anything-Small@273d090f2ce17df50c2872d82c8322c45da5b4dd"
                    : "depth-anything/Video-Depth-Anything-Small@256875362cff76724b920335dfb4b29dd611f66e",
                createdAt: createdAt
            )
        case .visionGeometryDA3Small:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .depthAnything3,
                family: .geometry,
                tier: .small,
                variant: .standard,
                precision: .fp32,
                supports: [.relativeDepth, .cameraIntrinsics, .cameraExtrinsics, .pointCloud],
                components: nil,
                upstreamRepoId: "depth-anything/DA3-SMALL@e08cab65ca0ec38e7826075418411ab90cab4da3",
                createdAt: createdAt
            )
        case .image3DTripoSR:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .tripoSR,
                family: .threeD,
                tier: .small,
                variant: .standard,
                precision: .fp32,
                supports: [.imageTo3D, .meshGeneration],
                components: nil,
                upstreamRepoId: "stabilityai/TripoSR@5b521936b01fbe1890f6f9baed0254ab6351c04a",
                createdAt: createdAt
            )
        case .image3DInstantMeshBase:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .instantMesh,
                family: .threeD,
                tier: .base,
                variant: .standard,
                precision: .fp32,
                supports: [.multiViewReconstruction, .meshGeneration],
                components: nil,
                upstreamRepoId: "TencentARC/InstantMesh@b785b4ecfb6636ef34a08c748f96f6a5686244d0",
                createdAt: createdAt
            )
        case .image3DTrellis2:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .trellis2,
                family: .threeD,
                tier: .max,
                variant: .standard,
                precision: .bf16,
                supports: [.imageTo3D, .meshGeneration],
                components: nil,
                upstreamRepoId: "microsoft/TRELLIS.2-4B@af44b45f2e35a493886929c6d786e563ec68364d",
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
        case .aceStep, .aceStepXLTurbo, .aceStepXLTurboLM4B:
            let isXL = modelID == .aceStepXLTurbo || modelID == .aceStepXLTurboLM4B
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .aceStep,
                family: .music,
                tier: isXL ? .turbo : .latest,
                variant: isXL ? .distilled : .standard,
                precision: .bf16,
                defaults: Defaults(steps: 8, cfg: 1.0),
                supports: [.musicGeneration],
                components: nil,
                upstreamRepoId: isXL ? "ACE-Step/acestep-v15-xl-turbo" : "ACE-Step/Ace-Step1.5",
                createdAt: createdAt
            )
        case .magentaRT2Small, .magentaRT2Base:
            let isBase = modelID == .magentaRT2Base
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .magentaRT2,
                family: .music,
                tier: isBase ? .base : .small,
                variant: .standard,
                precision: .unknown,
                defaults: Defaults(steps: isBase ? 100 : 100, cfg: 3.0),
                supports: [.musicGeneration],
                components: nil,
                upstreamRepoId: "google/magenta-realtime-2@010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc",
                createdAt: createdAt
            )
        case .muScriptorSmall, .muScriptorMedium, .muScriptorLarge:
            let muScriptorVariant = MuScriptorVariant.resolve(modelID: modelID.rawValue)!
            let muScriptorTier: Tier = switch muScriptorVariant {
            case .small: .small
            case .medium: .base
            case .large: .max
            }
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .muScriptor,
                family: .music,
                tier: muScriptorTier,
                variant: .standard,
                precision: .fp32,
                defaults: nil,
                supports: [.musicTranscription],
                components: nil,
                upstreamRepoId: "MuScriptor/muscriptor-\(muScriptorVariant.rawValue)",
                createdAt: createdAt
            )
        case .wooshDFlow, .wooshFlow, .wooshVFlow8s, .wooshDVFlow8s:
            let isDFlow = modelID == .wooshDFlow
            let isVideoFlow = modelID == .wooshVFlow8s || modelID == .wooshDVFlow8s
            let isDistilledVideoFlow = modelID == .wooshDVFlow8s
            let transformerName = if modelID == .wooshFlow {
                "Woosh-Flow"
            } else if modelID == .wooshVFlow8s {
                "Woosh-VFlow-8s"
            } else if modelID == .wooshDVFlow8s {
                "Woosh-DVFlow-8s"
            } else {
                "Woosh-DFlow"
            }
            let textConditionerName = isVideoFlow ? "TextConditionerV" : "TextConditionerA"
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .woosh,
                family: .sfx,
                tier: .latest,
                variant: (isDFlow || isDistilledVideoFlow) ? .distilled : .standard,
                precision: .fp32,
                defaults: Defaults(
                    steps: (isDFlow || isDistilledVideoFlow) ? 4 : 32,
                    cfg: isDistilledVideoFlow ? 3.0 : 4.5
                ),
                supports: isVideoFlow ? [.videoToAudioGeneration] : [.soundEffectGeneration],
                components: Components(
                    tokenizer: .local(path: "checkpoints/\(textConditionerName)/tokenizer"),
                    textEncoder: .local(path: "checkpoints/\(textConditionerName)"),
                    transformer: .local(path: "checkpoints/\(transformerName)"),
                    vae: .local(path: "checkpoints/Woosh-AE"),
                    scheduler: nil
                ),
                upstreamRepoId: "\(WooshResources.upstreamRepoId)@\(WooshResources.upstreamRelease)",
                createdAt: createdAt
            )
        case .wooshClap:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .woosh,
                family: .sfx,
                tier: .latest,
                variant: .standard,
                precision: .fp32,
                supports: [.soundEffectEmbedding],
                components: Components(
                    tokenizer: .local(path: "checkpoints/Woosh-CLAP/tokenizer"),
                    textEncoder: .local(path: "checkpoints/Woosh-CLAP"),
                    transformer: nil,
                    vae: nil,
                    scheduler: nil
                ),
                upstreamRepoId: "\(WooshResources.upstreamRepoId)@\(WooshResources.upstreamRelease)",
                createdAt: createdAt
            )
        case .wooshSynchformer:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .woosh,
                family: .sfx,
                tier: .latest,
                variant: .standard,
                precision: .fp16,
                supports: [.videoToAudioGeneration],
                components: nil,
                upstreamRepoId: WooshResources.synchformerRepoId,
                createdAt: createdAt
            )
        case .mmaudioLarge44kV2:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .mmaudio,
                family: .sfx,
                tier: .max,
                variant: .standard,
                precision: .fp16,
                defaults: Defaults(
                    steps: MMAudioResources.defaultSteps,
                    cfg: Double(MMAudioResources.defaultGuidanceScale)
                ),
                supports: [.soundEffectGeneration, .videoToAudioGeneration],
                components: nil,
                upstreamRepoId: "\(MMAudioResources.upstreamRepoID)@\(MMAudioResources.upstreamRevision)",
                createdAt: createdAt
            )
        case .ltxVideoAV, .ltxVideo23AVMLX:
            let isLTX23 = modelID == .ltxVideo23AVMLX
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
                upstreamRepoId: isLTX23 ? "dgrauet/ltx-2.3-mlx@main" : "mlx-community/LTX-2-distilled-bf16",
                createdAt: createdAt
            )
        case .ltxVideo23FullMLX, .ltxVideo23A2VMLX:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .ltxVideo,
                family: .video,
                tier: .latest,
                variant: .base,
                precision: .bf16,
                defaults: Defaults(steps: 30, cfg: 3),
                supports: [.videoGeneration, .audioToVideoGeneration],
                components: nil,
                upstreamRepoId: "dgrauet/ltx-2.3-mlx@baa5f235ea04fd9c95899d751295c4fd825ee4e2",
                createdAt: createdAt
            )
        case .wan22TI2V5BMLX:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .wanVideo,
                family: .video,
                tier: .base,
                variant: .base,
                precision: .bf16,
                defaults: Defaults(steps: 40, cfg: 5.0, sigmaShift: 5.0),
                supports: [.videoGeneration],
                components: Components(
                    tokenizer: .local(path: "."),
                    textEncoder: .local(path: "."),
                    transformer: .local(path: "."),
                    vae: .local(path: "."),
                    scheduler: nil
                ),
                upstreamRepoId: "\(Wan2Resources.managedRepoID)@\(Wan2Resources.managedRevision)",
                createdAt: createdAt
            )
        case .scail2Video14BMLX:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .wanVideo,
                family: .video,
                tier: .latest,
                variant: .base,
                precision: .bf16,
                defaults: Defaults(steps: 40, cfg: 5.0, sigmaShift: 3.0),
                supports: [.videoGeneration],
                components: Components(
                    tokenizer: .local(path: "."),
                    textEncoder: .local(path: "."),
                    transformer: .local(path: "."),
                    vae: .local(path: "."),
                    scheduler: nil
                ),
                upstreamRepoId: "\(SCAIL2Resources.upstreamRepoID)@\(SCAIL2Resources.upstreamRevision)",
                createdAt: createdAt
            )
        case .dreamXWorld5BARMLX:
            return MereRunModelManifest(
                id: modelID.rawValue,
                engine: .wanVideo,
                family: .video,
                tier: .latest,
                variant: .base,
                precision: .bf16,
                defaults: Defaults(steps: 4, cfg: 1.0, sigmaShift: 5.0),
                supports: [.videoGeneration],
                components: Components(
                    tokenizer: nil,
                    textEncoder: nil,
                    transformer: .local(path: Wan2DreamXCausalResources.weightsFilename),
                    vae: nil,
                    scheduler: nil
                ),
                upstreamRepoId: "\(Wan2DreamXCausalResources.upstreamRepoID)@\(Wan2DreamXCausalResources.upstreamRevision)",
                createdAt: createdAt
            )
        }
    }
}
