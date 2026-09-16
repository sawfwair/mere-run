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
        /// FLUX.1 Diffusers family.
        case flux1 = "flux1"
        /// MereRun family (FLUX.2 Klein based).
        case flux2Klein = "flux2-klein"
        /// Zeta family (Z-Image Turbo based).
        case zimageTurbo = "zimage-turbo"
        /// HiDream O1 unified pixel transformer family.
        case hidreamO1 = "hidream-o1"
        /// SenseNova U1.5 unified understanding and raw-pixel generation transformer.
        case senseNovaU15 = "sensenova-u1.5"
        /// Krea 2 text-to-image family.
        case krea2 = "krea-2"
        /// Qwen Image Edit native multi-reference editing family.
        case qwenImageEdit = "qwen-image-edit"
        /// Ideogram 4 text-to-image family.
        case ideogram4 = "ideogram-4"
        /// Gemma 4 family via the native Swift runtime.
        case gemma4 = "gemma-4"
        /// LiquidAI LFM2 family via the native Swift runtime.
        case lfm2 = "lfm2"
        /// Poolside Laguna family via the native Swift runtime.
        case laguna = "laguna"
        /// Qwen3.5-family dense or MoE models with hybrid attention.
        case qwen35HybridMoE = "qwen3.5-hybrid-moe"
        /// SAM image segmentation family.
        case samSegmentation = "sam-segmentation"
        /// Falcon Perception grounded detection and segmentation family.
        case falconPerception = "falcon-perception"
        /// IBM/ESA TerraMind temporal flood-segmentation family.
        case terramindFlood = "terramind-flood"
        /// IBM/ESA TerraMind temporal fire-segmentation family.
        case terramindFire = "terramind-fire"
        /// TESSERA v2 Sentinel-1/2 temporal embedding family.
        case tessera
        /// Ai2 OlmoEarth multisensor spatial embedding family.
        case olmoEarth = "olmoearth"
        /// InsightFace Buffalo-L face detection and identity-embedding family.
        case insightFace = "insightface"
        /// MoGe-2 metric monocular geometry family.
        case moge2 = "moge-2"
        /// Video Depth Anything temporal depth family.
        case videoDepthAnything = "video-depth-anything"
        /// Depth Anything 3 multi-view geometry family.
        case depthAnything3 = "depth-anything-3"
        /// Marigold V2 single-step dense prediction family.
        case marigoldV2 = "marigold-v2"
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
        /// NVIDIA Sortformer speaker diarization family.
        case sortformer = "sortformer"
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
        /// MiniMax Music 3 autoregressive and flow-matching music family.
        case miniMaxMusic3 = "minimax-music3"
        /// Magenta RealTime 2 streaming music family.
        case magentaRT2 = "magenta-rt2"
        /// MuScriptor multi-instrument audio transcription family.
        case muScriptor = "muscriptor"
        /// Band-split RoFormer music source-separation family.
        case roFormer = "bs-roformer"
        /// AP-BWE speech bandwidth-extension family.
        case apBWE = "ap-bwe"
        /// UniverSR flow-matching general-audio super-resolution family.
        case univerSR = "universr"
        /// Sony Research Woosh sound-effect generation family.
        case woosh = "woosh"
        /// MMAudio synchronized video-to-audio and text-to-audio family.
        case mmaudio = "mmaudio"
        /// LTX video family.
        case ltxVideo = "ltx-video"
        /// Wan2 native video family.
        case wanVideo = "wan-video"
        /// MiniMax-H3 joint video/audio family.
        case miniMaxH3 = "minimax-h3"
        /// NVIDIA Cosmos3-Edge native omnimodal world-model family.
        case cosmos3Edge = "cosmos3-edge"
        /// Psi agent chat family.
        case psiChat = "psi-chat"
        /// DeepSeek V4 Flash family, served by the bundled `ds4-server` subprocess.
        case deepseekV4Flash = "deepseek-v4-flash"
        /// Thinking Machines Lab Inkling family via the native Swift/MLX runtime.
        case inkling
        /// Meta Muse Glimmer multimodal agent family via native Swift/MLX.
        case museGlimmer = "muse-glimmer"
        /// NVIDIA Nemotron-H hybrid Mamba/MoE family via native Swift/MLX.
        case nemotronH = "nemotron-h"
        /// NVIDIA Nemotron 3 Nano Omni multimodal understanding family.
        case nemotronOmni = "nemotron-omni"
    }

    public enum Family: String, Codable, CaseIterable, Hashable, Sendable {
        case flux1
        case klein
        case zimage
        case hidream
        case senseNova = "sensenova"
        case krea
        case ideogram
        case gemma
        case liquid
        case laguna
        case qwen
        case sam
        case falcon
        case terramind
        case tessera
        case olmoEarth = "olmoearth"
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
        case audio
        case music
        case sfx
        case video
        case psi
        case deepseek
        case inkling
        case muse
        case nemotron
    }

    public enum Tier: String, Codable, CaseIterable, Hashable, Sendable {
        case nano
        case tiny
        case small
        case medium
        case large
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
        case int3
        case int8
        case int4
        case int6
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
        case multimodalEmbedding = "multimodal_embedding"
        case textAnonymization = "text_anonymization"
        case speechSynthesis = "speech_synthesis"
        case speechRecognition = "speech_recognition"
        case speakerDiarization = "speaker_diarization"
        case visionChat = "vision_chat"
        case visionOCR = "vision_ocr"
        case audioUnderstanding = "audio_understanding"
        case videoUnderstanding = "video_understanding"
        case documentUnderstanding = "document_understanding"
        case musicGeneration = "music_generation"
        case musicTranscription = "music_transcription"
        case musicSeparation = "music_separation"
        case audioEnhancement = "audio_enhancement"
        case videoGeneration = "video_generation"
        case actionGeneration = "action_generation"
        case worldSimulation = "world_simulation"
        case visionReasoning = "vision_reasoning"
        case audioToVideoGeneration = "audio_to_video_generation"
        case loraInference = "lora_inference"
        case loraTraining = "lora_training"
        case visionSegmentation = "vision_segmentation"
        case floodSegmentation = "flood_segmentation"
        case fireSegmentation = "fire_segmentation"
        case earthObservationEmbedding = "earth_observation_embedding"
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

    /// Third-party model/component terms that required explicit acceptance before download.
    public var usageTerms: [ManagedModelUsageTerm]?

    /// True only when the managed installer was invoked with explicit confirmation of acceptance.
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

}
