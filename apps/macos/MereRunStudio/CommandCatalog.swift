import Foundation
import MereRunContract
import UniformTypeIdentifiers

enum StudioProductBoundary {
    static let dioramaURL = URL(string: "https://diorama.mere.run")!
}

enum CommandCategory: String, CaseIterable, Identifiable {
    case setup = "Setup"
    case models = "Models"
    case image = "Image"
    case text = "Text"
    case speech = "Speech"
    case vision = "Vision"
    case geospatial = "Geospatial"
    case media = "Music & Video"
    case sfx = "Sound FX"
    case operations = "Operations"
    case server = "Server"
    case custom = "Custom"

    var id: String { rawValue }
}

enum CommandTemplateID: String, CaseIterable, Codable {
    case setup
    case agentOnboard
    case agentStatus
    case agentInstallPi
    case agentStart
    case modelList
    case modelCapabilities
    case modelPull
    case modelInfo
    case modelRemove
    case modelRepairManifests
    case modelOptimize
    case imageGenerate
    case imageTrainLoRA
    case imageValidate
    case imageDatasetDiscover
    case imageRunPlan
    case imageVisualizeRun
    case imageReconstruct3D
    case imageReconstruct3DTrellis2
    case imageReconstruct3DMultiview
    case textChat
    case textCode
    case textEmbed
    case textAnonymize
    case textTrainLoRA
    case speechSynthesize
    case speechTranscribe
    case speechDiarize
    case speechProfileList
    case speechProfileCreate
    case speechProfileDelete
    case visionInspect
    case visionEmbed
    case visionCaption
    case visionOCR
    case visionGround
    case visionSegment
    case visionTrack
    case visionTrackLive
    case visionFaceDetect
    case visionFaceEmbed
    case visionFaceCompare
    case visionFaceBatch
    case visionPose
    case visionFlow
    case visionDepthVideo
    case visionGeometry
    case visionGeometryMultiview
    case audioEnhance
    case audioGenerate
    case musicGenerate
    case musicAnalyze
    case musicTranscribe
    case musicSeparate
    case musicRealtime
    case musicTrainAdapter
    case musicServe
    case videoGenerate
    case videoRetake
    case videoDubIt
    case videoAnimate
    case videoCosmos3
    case videoPrepareMasks
    case videoExportLatents
    case videoSession
    case adapterList
    case adapterPull
    case runList
    case runInspect
    case runWatch
    case runFetch
    case runCancel
    case runRetry
    case evaluationPackValidate
    case evaluationRun
    case evaluationPromote
    case worldServe
    case statusSnapshot
    case qualityGate
    case modelStorage
    case modelGarbageCollect
    case modelRuntimeGet
    case modelRuntimeSet
    case graphStudio
    case nodeConsole
    case sfxGenerate
    case sfxVideo
    case sfxAEEncode
    case sfxAEDecode
    case sfxClapScore
    case sfxConditionText
    case modelBenchmark
    case modelBenchmarkLagunaDFlash
    case pluginList
    case pluginInstall
    case pluginDoctor
    case openWebui
    case apiServe
    case geoFlood
    case geoFire
    case geoTessera
    case geoOlmoEarth
    case modelLocationList
    case modelLocationAdd
    case modelLocationRemove
    case modelLocationBind
    case modelLocationUnbind
    case modelBenchmarkChat
    case modelBenchmarkCode
    case modelBenchmarkFused
    case modelBenchmarkFusedFixture
    case modelBenchmarkVLM
    case modelBenchmarkToolCalls
    case modelBenchmarkToolContinuations
    case modelBenchmarkGemma4KV
    case modelBenchmarkGemma4MTP
    case modelBenchmarkAPIWorkload
    case pluginInfo
    case pluginRun
    case pluginRollback
    case speechListen
    case visionServe
    case custom

    var capabilityID: String? {
        switch self {
        case .setup: return "setup"
        case .agentOnboard: return "agent.onboard"
        case .agentStatus: return "agent.status"
        case .agentInstallPi: return "agent.install-pi"
        case .agentStart: return "agent.start"
        case .modelList: return "model.list"
        case .modelCapabilities: return "model.capabilities"
        case .modelPull: return "model.pull"
        case .modelInfo: return "model.info"
        case .modelRemove: return "model.remove"
        case .modelRepairManifests: return "model.repair-manifests"
        case .modelOptimize: return "model.optimize"
        case .imageGenerate: return "image.generate"
        case .imageTrainLoRA: return "image.train-lora"
        case .imageValidate: return "image.validate"
        case .imageDatasetDiscover: return "image.dataset.discover"
        case .imageRunPlan: return "image.run-plan"
        case .imageVisualizeRun: return "image.visualize-run"
        case .imageReconstruct3D: return "image.reconstruct-3d"
        case .imageReconstruct3DTrellis2: return "image.reconstruct-3d-trellis2"
        case .imageReconstruct3DMultiview: return "image.reconstruct-3d-multiview"
        case .textChat: return "text.chat"
        case .textCode: return "text.code"
        case .textEmbed: return "text.embed"
        case .textAnonymize: return "text.anonymize"
        case .textTrainLoRA: return "text.train-lora"
        case .speechSynthesize: return "speech.synthesize"
        case .speechTranscribe: return "speech.transcribe"
        case .speechDiarize: return "speech.diarize"
        case .speechProfileList: return "speech.profile.list"
        case .speechProfileCreate: return "speech.profile.create"
        case .speechProfileDelete: return "speech.profile.delete"
        case .visionInspect: return "vision.inspect"
        case .visionEmbed: return "vision.embed"
        case .visionCaption: return "vision.caption"
        case .visionOCR: return "vision.ocr"
        case .visionGround: return "vision.ground"
        case .visionSegment: return "vision.segment"
        case .visionTrack: return "vision.track"
        case .visionTrackLive: return "vision.track-live"
        case .visionFaceDetect: return "vision.face.detect"
        case .visionFaceEmbed: return "vision.face.embed"
        case .visionFaceCompare: return "vision.face.compare"
        case .visionFaceBatch: return "vision.face.batch"
        case .visionPose: return "vision.pose"
        case .visionFlow: return "vision.flow"
        case .visionDepthVideo: return "vision.depth-video"
        case .visionGeometry: return "vision.geometry"
        case .visionGeometryMultiview: return "vision.geometry-multiview"
        case .audioEnhance: return "audio.enhance"
        case .audioGenerate: return "audio.generate"
        case .musicGenerate: return "music.generate"
        case .musicAnalyze: return "music.analyze"
        case .musicTranscribe: return "music.transcribe"
        case .musicSeparate: return "music.separate"
        case .musicRealtime: return "music.realtime"
        case .musicTrainAdapter: return "music.train-adapter"
        case .musicServe: return "music.serve"
        case .videoGenerate: return "video.generate"
        case .videoRetake: return "video.retake"
        case .videoDubIt: return "video.dub-it"
        case .videoAnimate: return "video.animate"
        case .videoCosmos3: return "video.cosmos3"
        case .videoPrepareMasks: return "video.prepare-masks"
        case .videoExportLatents: return "video.export-latents"
        case .videoSession: return "video.session"
        case .adapterList: return "adapter.list"
        case .adapterPull: return "adapter.pull"
        case .runList: return "run.list"
        case .runInspect: return "run.inspect"
        case .runWatch: return "run.watch"
        case .runFetch: return "run.fetch"
        case .runCancel: return "run.cancel"
        case .runRetry: return "run.retry"
        case .evaluationPackValidate: return "eval.pack.validate"
        case .evaluationRun: return "eval.run"
        case .evaluationPromote: return "eval.promote"
        case .worldServe: return "world.serve"
        case .statusSnapshot: return "status"
        case .qualityGate: return "gate"
        case .modelStorage: return "model.storage"
        case .modelGarbageCollect: return "model.gc"
        case .modelRuntimeGet: return "model.runtime.get"
        case .modelRuntimeSet: return "model.runtime.set"
        case .graphStudio, .nodeConsole, .custom: return nil
        case .sfxGenerate: return "sfx.generate"
        case .sfxVideo: return "sfx.video.generate"
        case .sfxAEEncode: return "sfx.ae.encode"
        case .sfxAEDecode: return "sfx.ae.decode"
        case .sfxClapScore: return "sfx.clap.score"
        case .sfxConditionText: return "sfx.condition.text"
        case .modelBenchmark: return "model.benchmark.q36-mtp"
        case .modelBenchmarkLagunaDFlash: return "model.benchmark.laguna-dflash"
        case .pluginList: return "plugin.list"
        case .pluginInstall: return "plugin.install"
        case .pluginDoctor: return "plugin.doctor"
        case .openWebui: return "open-webui.quickstart"
        case .apiServe: return "api.serve"
        case .geoFlood: return "geo.flood"
        case .geoFire: return "geo.fire"
        case .geoTessera: return "geo.tessera"
        case .geoOlmoEarth: return "geo.olmoearth"
        case .modelLocationList: return "model.location.list"
        case .modelLocationAdd: return "model.location.add"
        case .modelLocationRemove: return "model.location.remove"
        case .modelLocationBind: return "model.location.bind"
        case .modelLocationUnbind: return "model.location.unbind"
        case .modelBenchmarkChat: return "model.benchmark.chat"
        case .modelBenchmarkCode: return "model.benchmark.code"
        case .modelBenchmarkFused: return "model.benchmark.fused"
        case .modelBenchmarkFusedFixture: return "model.benchmark.fused-fixture"
        case .modelBenchmarkVLM: return "model.benchmark.vlm"
        case .modelBenchmarkToolCalls: return "model.benchmark.tool-calls"
        case .modelBenchmarkToolContinuations: return "model.benchmark.tool-continuations"
        case .modelBenchmarkGemma4KV: return "model.benchmark.gemma4-kv"
        case .modelBenchmarkGemma4MTP: return "model.benchmark.gemma4-mtp"
        case .modelBenchmarkAPIWorkload: return "model.benchmark.api-workload"
        case .pluginInfo: return "plugin.info"
        case .pluginRun: return "plugin.run"
        case .pluginRollback: return "plugin.rollback"
        case .speechListen: return "speech.listen"
        case .visionServe: return "vision.serve"
        }
    }

    /// The capability's contract entry, when the command is one the shared contract declares.
    var capability: MereRunCommandCapability? {
        capabilityID.flatMap { MereRunCapabilityCatalog.command(id: $0) }
    }

    /// True when the CLI prints the `{"event":"result"}` receipt line for this command under
    /// `--receipt`. The contract owns the list, so a capability that gains the flag needs no
    /// change here.
    var emitsRunReceipt: Bool {
        guard let capabilityID else { return false }
        return MereRunCapabilityCatalog.receiptCapabilityIDs.contains(capabilityID)
    }

    /// True when the CLI streams `--progress-json` events for this command.
    var emitsProgressJSON: Bool {
        guard let capabilityID else { return false }
        return MereRunCapabilityCatalog.progressJSONCapabilityIDs.contains(capabilityID)
    }
}

/// The flags the app adds to a run's argv so it can read the CLI's structured output: the
/// result receipt it resolves artifacts from, and the progress event stream it renders.
///
/// They are transport rather than user intent, so they are appended to the launched argv only —
/// never to the "Will run" preview, a library row's command, or the console's echo of the
/// command, all of which stay the command a person would type. `--preflight` runs get neither:
/// the CLI rejects `--receipt --preflight`, and a preflight report has no progress to stream.
enum StudioMachineOutputFlags {
    static let receipt = "--receipt"
    static let progressJSON = "--progress-json"

    /// The flags to append to `arguments` for one run, in a stable order. Empty when the
    /// capability declares neither, when the run is a preflight, or when the app already passes
    /// the flag from the draft.
    static func arguments(
        template: CommandTemplate,
        draft: CommandDraft,
        appendingTo arguments: [String]
    ) -> [String] {
        guard !draft.preflight, !arguments.contains("--preflight") else { return [] }
        var flags: [String] = []
        if template.id.emitsRunReceipt, !arguments.contains(receipt) {
            flags.append(receipt)
        }
        if template.id.emitsProgressJSON, !arguments.contains(progressJSON) {
            flags.append(progressJSON)
        }
        return flags
    }
}

enum CommandInputKind: Equatable {
    case none
    case file([UTType])
    case directory
    case image
    case audio
    case video

    var title: String {
        switch self {
        case .none: return "Input"
        case .file: return "File"
        case .directory: return "Directory"
        case .image: return "Image"
        case .audio: return "Audio"
        case .video: return "Video"
        }
    }

    var allowedTypes: [UTType] {
        switch self {
        case .none, .directory: return []
        case .file(let types): return types
        case .image: return [.image]
        case .audio: return [.audio]
        case .video: return [.movie, .video, .audiovisualContent]
        }
    }
}

enum CommandOutputKind: Equatable {
    case none
    case file(String)
    case directory

    var isFile: Bool {
        if case .file = self { return true }
        return false
    }
}

enum StudioChatDefaults {
    static let fallbackModelID = "text-chat-gemma4-12b-4bit"
    static let fallbackServingEngine = "text-chat-gemma4"

    static func servingEngine(for modelID: String) -> String {
        let normalized = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalized.contains("deepseek-v4-flash") {
            return "text-chat-deepseek-v4-flash"
        }
        if normalized.contains("q36") || normalized.contains("q35") {
            return "text-chat-q36"
        }
        if normalized.contains("laguna") {
            return "text-chat-laguna"
        }
        if normalized.contains("lfm") {
            return "text-chat-lfm2"
        }
        if normalized.contains("klein") {
            return "text-chat-klein"
        }
        if normalized.contains("gemma") {
            return "text-chat-gemma4"
        }
        return fallbackServingEngine
    }

    static func shouldReplaceModelDefault(_ modelID: String, oldRecommendation: String? = nil) -> Bool {
        let normalized = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalized.isEmpty { return true }
        var replaceable: Set<String> = [
            fallbackModelID,
            "text-chat-q36-nano",
            "text-chat-bonsai-27b-1bit",
            "text-chat-bonsai-27b-2bit",
            "text-chat-gemma4",
            "text-chat-gemma4-12b",
            "text-chat-gemma4-12b-4bit"
        ]
        if let oldRecommendation, !oldRecommendation.isBlank {
            replaceable.insert(oldRecommendation)
        }
        return replaceable.contains(normalized)
    }

    static func shouldReplaceServingEngineDefault(_ engine: String, oldRecommendation: String? = nil) -> Bool {
        let normalized = engine.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalized.isEmpty { return true }
        var replaceable: Set<String> = [
            fallbackServingEngine,
            "text-chat-q36",
            "text-chat-gemma4"
        ]
        if let oldRecommendation, !oldRecommendation.isBlank {
            replaceable.insert(servingEngine(for: oldRecommendation))
        }
        return replaceable.contains(normalized)
    }
}

enum StudioCodeDefaults {
    static let fallbackModelID = "text-code-north-mini"

    static func shouldReplaceModelDefault(_ modelID: String, oldRecommendation: String? = nil) -> Bool {
        let normalized = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalized.isEmpty { return true }
        var replaceable: Set<String> = [
            fallbackModelID,
            "text-agent-qwen35-9b",
            "text-code-qwen3",
        ]
        if let oldRecommendation, !oldRecommendation.isBlank {
            replaceable.insert(oldRecommendation)
        }
        return replaceable.contains(normalized)
    }
}

enum StudioVideoModelFamily: Equatable {
    case ltx
    case wan
    case miniMaxH3FL2VA
    case miniMaxH3Ref2VA

    init(model: String) {
        let normalized = model.lowercased()
        if normalized.contains("minimax-h3") || normalized.contains("minimax_h3") {
            self = normalized.contains("ref2va") ? .miniMaxH3Ref2VA : .miniMaxH3FL2VA
        } else if normalized.contains("wan") {
            self = .wan
        } else {
            self = .ltx
        }
    }

    var isMiniMaxH3: Bool {
        self == .miniMaxH3FL2VA || self == .miniMaxH3Ref2VA
    }

    static func alignedMiniMaxH3FrameCount(_ requested: Int) -> Int {
        let clamped = max(22, requested)
        return ((clamped - 5 + 16) / 17) * 17 + 5
    }
}

struct CommandDraft: Equatable, Codable {
    var prompt = ""
    var secondaryText = ""
    var model = ""
    var inputPath = ""
    /// Optional so Library rows written before mask editing still decode through synthesized Codable.
    var imageMaskPath: String?
    var imageOutpaint: String?
    var imageMaskFeather: Int?
    var outputPath = ""
    var width = 1024
    var height = 1024
    var steps = 4
    var seed = ""
    var cfgScale = 1.0
    var strength = 0.75
    var sigmaShift = 0.0
    var referenceImagePaths = ""
    var keepOriginalAspect = false
    var structuredPrompt = false
    var structuredPromptModel = "text-chat-gemma4-12b-4bit"
    var structuredPromptModelRoot = ""
    var structuredPromptMaxTokens = 2048
    var structuredPromptOutputPath = ""
    var kreaConditioningMultiplier = 0.0
    var kreaConditioningLayerWeights = ""
    var kreaBaseQuantizationBits = ""
    var progressJSON = false
    var maxTokens = 2048
    var contextSize = 0
    var temperature = 0.7
    var topP = 0.9
    var topK = 0
    var minP = 0.0
    var kvBits = 0
    var kvQuantScheme = ""
    var kvGroupSize = 0
    var quantizedKVStart = 0
    var responseFormat: TextResponseFormat = .text
    var thinkingMode: TextThinkingMode = .automatic
    /// Optional preserves Library rows written before Inkling reasoning effort was exposed.
    var reasoningEffort: Double?
    var loraPath = ""
    var loraScale = 1.0
    var replacement = "[{label}]"
    var evalPath = ""
    var adapterName = "local-assistant"
    var batchSize = 1
    var learningRate = 0.0001
    var rank = 16
    var alpha = 0.0
    var maxSequenceLength = 4096
    var targetModules = ""
    var dryRun = false
    var visualize = false
    var visualizePort = 8787
    var schedulerSteps = 1_000
    var captionDropout = 0.0
    var trainingLite = false
    var baseQuantizationBits = ""
    var excludePreviewImages = false
    var checkpointInterval = 0
    /// Optional preserves Library decoding for rows written before checkpoint resume was exposed.
    var trainingResumePath: String?
    var maxResolution = 0
    var progressive = false
    var lowRAM = false
    var disableCompile = false
    var gradientCheckpointing = false
    var trainingRecipe = ""
    var overrideTrainingRecipe = false
    var benchmarkSteps = 0
    var benchmarkWarmupSteps = 5
    var sampleInterval = 0
    var samplePrompt = ""
    var sampleModel = ""
    var sampleSteps = 8
    var sampleCFG = 1.0
    var sampleLoRAScale = 1.0
    var sampleSeed = ""
    var loraTargetRanks = ""
    var loraRankPreset = ""
    var loraTargetPreset = ""
    var loraTargetMode = ""
    var timestepSampling = ""
    var timestepLossWeighting = ""
    var lossWeighting = ""
    var timestepLow = 0
    var timestepHigh = 0
    var lrWarmupSteps = 0
    var disableCosineScheduler = false
    var lrMinFactor = 0.0
    var adamWeightDecay = 0.0
    var syntheticSamples = 0
    var maxDepth = 4
    var minUsablePairs = 1
    var trainingOutputRoot = ""
    var trainingModel = ""
    var materializePath = ""
    var referenceDirectoryPath = ""
    var reconstructionResolution = 256
    var densityThreshold = 25.0
    var foregroundRatio = 0.85
    var alreadyFramed = false
    var noVertexColors = false
    var camerasPath = ""
    /// Optional so Library rows written before the first-class TRELLIS.2 workspace keep decoding.
    var trellisTextureSeed: String?
    var trellisNoRemesh: Bool?
    var trellisRemeshBand: Double?
    var trellisSealRadius: Int?
    var durationSeconds = 10.0
    // Music production workspace. Empty strings intentionally mean "let the CLI quality
    // preset choose" for optional numeric overrides.
    var musicLyricsFile = ""
    var musicLRCFile = ""
    var musicLRCOutput = ""
    var musicExportFormat = "pcm24"
    var musicNormalization = "peak"
    var musicTargetPeakDB = -1.0
    var musicFadeInMS = 5.0
    var musicFadeOutMS = 20.0
    var musicNoDither = false
    var musicRecipeOutput = ""
    var musicNoRecipe = false
    var musicDAWBundle = ""
    var musicStems = ""
    var musicAdapterPaths = ""
    var musicAdapterKind = "auto"
    var musicAdapterScales = ""
    var musicCheckpointsRoot = ""
    var musicDecoderSubdirectory = "acestep-v15-turbo"
    var musicVAESubdirectory = "vae"
    var musicLMModel = ""
    var musicLMSubdirectory = ""
    var musicTextSubdirectory = ""
    var musicLMMode = "auto"
    var musicAnalyzeSourceAudio = false
    var musicQuality = "song"
    var musicOverrideSteps = false
    var musicShift = ""
    var musicInferMethod = ""
    var musicSampler = ""
    var musicGuidanceScale = ""
    var musicGuidanceMode = ""
    var musicCFGIntervalStart = ""
    var musicCFGIntervalEnd = ""
    var musicVelocityNormThreshold = ""
    var musicVelocityEMAFactor = ""
    var musicCandidates = 0
    var musicKeepCandidates = false
    var musicCoverStrength = 1.0
    var musicCoverNoiseStrength = 0.0
    var musicRetakeSeed = ""
    var musicRetakeVariance = 0.0
    var musicVocalLanguage = "en"
    var musicInstruction = "Fill the audio semantic mask based on the given conditions:"
    var musicTask = "text2music"
    var musicSourceAudio = ""
    var musicReferenceAudioPaths = ""
    var musicTrackName = ""
    var musicCompleteTrackClasses = ""
    var musicNonCover = false
    var musicRepaintStart = 0.0
    var musicRepaintEnd = -1.0
    var musicChunkMaskMode = "auto"
    var musicRepaintMode = "balanced"
    var musicRepaintStrength = 0.5
    var musicFlowEdit = false
    var musicSourceCaption = ""
    var musicSourceLyrics = ""
    var musicFlowEditNMin = 0.0
    var musicFlowEditNMax = 1.0
    var musicFlowEditNAverage = 1
    var musicBPM = ""
    var musicKey = ""
    var musicTimeSignature = ""
    var musicLMTemperature = 0.85
    var musicLMTopK = 0
    var musicLMTopP = 0.9
    var musicLMRepetitionPenalty = 1.0
    var musicLMCFGScale = 2.0
    var musicLMNegativePrompt = "NO USER INPUT"
    var musicInstrumental = false
    var musicMetadataDuration = ""
    var musicMetadataLanguage = ""
    var musicNoTiledVAE = false
    var musicVAEChunkSize = 512
    var musicVAEOverlap = 64
    var musicStyleConditioning = "streaming"
    var musicTemperature = 1.0
    var musicTopK = 100
    var musicCFGMusicCoCa = 3.0
    var musicCFGNotes = 5.0
    var musicCFGDrums = 1.0
    var musicDrumless = false
    var musicUnmaskWidth = 0
    var musicSeedRotation = 0
    var musicPrefillSilence = false
    var musicPrefillDuration = 1.64
    var musicIncludeRawLM = false
    var musicIncludeAudioCodes = false
    var musicAnalysisMaxTokens = 2048
    var musicAnalysisTemperature = 0.3
    var musicTranscribeModelPath = ""
    var musicTranscribeVariant = ""
    var musicTranscribeFormat = "midi"
    var musicInstruments = ""
    var musicListInstruments = false
    var musicSampling = false
    var musicMaxTokensPerChunk = 2_000
    var musicStrictEOS = false
    var musicBeamSize = 1
    var musicChunkBatchSize = 4
    var musicDType = "bfloat16"
    var musicNoMusicalContext = false
    var musicContextOutput = ""
    var musicPlay = true
    var musicInteractive = false
    var musicListMIDIInputs = false
    var musicMIDIMonitor = false
    var musicMIDILogEvents = false
    var musicMIDILogRaw = false
    var musicMIDIInput = ""
    var musicMIDIChannel = "all"
    var musicMIDINoteOffset = 0
    var musicMIDICCMappings = ""
    var musicTrainingKind = "lora"
    var musicTrainingFactor = -1
    var musicTrainingWeightDecay = 0.0001
    var musicTrainingMaxDuration = 30.0
    var musicTrainingLogEvery = 10
    /// Optional fields preserve older Library rows while owning the post-0.31 audio tool wave.
    var audioOverlap: Int?
    var audioInputRate: Int?
    var audioODEMethod: String?
    var audioODESteps: Int?
    var audioGuidanceScale: Double?
    var audioChunkSeconds: Int?
    var audioDType: String?
    // Vision and VFX analysis controls.
    var visionAdditionalInputs = ""
    var visionSecondInputPath = ""
    var visionPromptFile = ""
    var visionFocus = ""
    var visionTriggerToken = ""
    var visionJSONOutputPath = ""
    var visionMaskOutputDirectory = ""
    var visionBoxPrompts = ""
    var visionPointPrompts = ""
    var visionThreshold = 0.05
    var visionResolution = 1008
    var visionMultimask = false
    var visionInitFrame = 0
    var visionEndFrame = ""
    var visionShowLabels = false
    var visionCamera = 0
    var visionSeedSearchFrames = 30
    var visionGLMOCRCLI = "glmocr"
    var visionGLMConfig = ""
    var visionInfinityRuntime = "native"
    var visionInfinityParserCLI = "parser"
    var visionInfinityModel = "vision-ocr-infinity-pro-int8"
    var visionInfinityBackend = "vllm-server"
    var visionInfinityAPIURL = "http://localhost:8000/v1/chat/completions"
    var visionInfinityAPIKey = "EMPTY"
    var visionInfinityTask = "doc2json"
    var visionInfinityPrompt = ""
    var visionInfinityOutputFormat = "md"
    var visionInfinityBatchSize = 1
    var visionInfinityModelCacheDirectory = ""
    var visionInfinityMinPixels = 2_048
    var visionInfinityMaxPixels = 16_777_216
    var visionPoseBody = true
    var visionPoseHands = true
    var visionPoseFace = true
    var visionMaxHands = 2
    var visionMinimumConfidence = 0.1
    var visionFlowAccuracy = "high"
    var visionInputSize = 518
    var visionMaxFrames = 240
    var visionResolutionLevel = 9
    var visionTokenCount = 0
    var visionMaxPoints = 0
    var visionProcessResolution = 504
    var visionReferenceView = "saddle-balanced"
    var visionConfidencePercentile = 40.0
    var visionFaceScoreThreshold = 0.65
    var visionExecutionProvider = "auto"
    var visionMaxFaces = 0
    var visionIncludeEmbeddings = false
    var visionFaceIndex = ""
    var visionReferenceFaceIndex = ""
    var visionCandidateFaceIndex = ""
    var visionInputList = ""
    var visionJSONLOutput = ""
    var visionFailFast = false
    // Operations, durable workflow runs, runtime diagnostics, and world sessions.
    var operationsReference = ""
    var operationsRoot = ""
    var operationsExecutor = ""
    var operationsArtifacts = ""
    var operationsLimit = 50
    var operationsPollInterval = 2.0
    var operationsTimeoutSeconds = 1.0
    var operationsAllArtifacts = false
    var operationsJSONStream = false
    var operationsGateSuite = "all"
    var operationsUpdateBaselines = false
    var operationsStrictPerformance = false
    var operationsListOnly = false
    var operationsWorldBackend = "dreamx"
    var operationsBaseModel = "video-wan22-ti2v-5b-mlx"
    var operationsStateDirectory = ""
    var operationsPrepare = false
    var operationsRuntimeAlias = ""
    var operationsRuntimeTTL = ""
    var operationsRuntimeContext = ""
    var operationsRuntimeMaxTokens = ""
    var operationsRuntimeTemperature = ""
    var operationsRuntimeTopP = ""
    var operationsRuntimeMinP = ""
    var operationsRuntimeEngine = ""
    var operationsRuntimeKVCacheMode = ""
    var operationsClearAlias = false
    var operationsPinned = false
    var operationsUnpinned = false
    var operationsClearTTL = false
    var operationsClearContext = false
    var operationsClearMaxTokens = false
    var operationsClearTemperature = false
    var operationsClearTopP = false
    var operationsClearMinP = false
    var operationsClearEngine = false
    var operationsClearKVCacheMode = false
    var fps = 24
    var numFrames = 65
    var useDuration = false
    var host = "127.0.0.1"
    var port = 8080
    var apiKey = ""
    var engine = StudioChatDefaults.fallbackServingEngine
    var videoQuality: LTXVideoQuality = .final
    var videoOutputMode: LTXVideoOutputMode = .videoOnly
    var audioPath = ""
    var audioStartTime = 0.0
    /// Optional preserves saved Library rows from before source-audio max-duration parity.
    var audioMaxDuration: Double?
    var endImagePath = ""
    var endImageStrength = 1.0
    var scheduleShift = 5.0
    var a2vGuidanceScale = 3.0
    var videoCFGGuidanceScale = 3.0
    var audioCFGGuidanceScale = 7.0
    var v2aGuidanceScale = 3.0
    var a2vSteps = 30
    var retakeStartTime = 0.0
    var retakeEndTime = 4.0
    var retakePreserveVideo = false
    var retakePreserveAudio = false
    var preflight = false
    var timings = false
    var timingsOutputPath = ""
    /// Optional preserves older Library rows before MiniMax-H3 became a Studio-native workflow.
    var h3WeightMode: String?
    /// Quality is exact; balanced and maximum enable bounded approximate block reuse.
    var h3AccelerationMode: String?
    /// Nil keeps MiniMax-H3's geometry-aware 9/16/21-point adaptive schedule.
    var h3Steps: Int?
    /// Ordered `image:path`, `video:path`, and `audio:path` reference specifications.
    var h3ReferenceInputs: [String]?
    var modelRoot = ""
    var referenceMaskPath = ""
    var drivingVideoPath = ""
    var drivingMaskPath = ""
    /// One mask path per newline-delimited additional SCAIL reference image.
    /// Optional preserves synthesized Codable compatibility with older Library rows.
    var scailAdditionalReferenceMaskPaths: String?
    var videoTaskMode = "animation"
    var renderProfile = "fast"
    var sampler = "unipc"
    var segmentLength = 81
    var segmentOverlap = 5
    var tailPolicy = "drop"
    var audioSource = "none"
    var cosmosMode = "text-to-video"
    var cosmosImagePath = ""
    var cosmosVideoPath = ""
    var actionsOutputPath = ""
    var schedule = "nvidia"
    var previewFrame = ""
    var backend = "auto"
    var task = "transcribe"
    var language = "auto"
    var timestamps = true
    var voiceMode = "style"
    var voiceProfile = ""
    var refAudioPath = ""
    var refText = ""
    var saveProfileName = ""
    var speechStreamChunkTokens = 25
    var speechStreamChunkMS = 200
    var speechStreamDecodeMS = 2_000
    var speechInputFormat = ""
    var speechSampleRate = 16_000
    var speechJSONL = false
    /// Optional fields preserve decoding of Library rows created before native Sortformer controls.
    var speechDiarizationFormat: String?
    var speechDiarizationThreshold: Double?
    var speechDiarizationMinDuration: Double?
    var speechDiarizationMergeGap: Double?
    // Vision chat (vision-capable chat models) and the agentic tool loop for `text chat`.
    var imagePath = ""
    var tools = ""
    var toolLoop = false
    var allowShellExec = false
    var allowAbsoluteToolPaths = false
    var autoApproveTools = false
    var requireInstalled = false
    var sandboxDir = ""
    var setupMode = "agent"
    var agentModel = "tier"
    var piPath = ""
    var noBootstrap = false
    var modelKeepCache = false
    var modelRemovalJSON = false
    var benchmarkPromptFile = ""
    var benchmarkPromptRepeat = 150
    var benchmarkPromptRepeatValues = ""
    var benchmarkDecodeTokens = 32
    var benchmarkDecodeTokenValues = ""
    var benchmarkTemperatureValues = ""
    var benchmarkMTPBlockSize = ""
    var benchmarkForcedMTPMinPromptTokens = 1
    var benchmarkRepetitions = 3
    var benchmarkLagunaDFlashTokens = 12
    var benchmarkFixture = "deterministic-prose"
    var benchmarkConcurrencyValues = ""
    var benchmarkWarmupRepetitions = 1
    var benchmarkMixedFixtures = false
    var benchmarkIncludeAutomatic = false
    var benchmarkLogResponses = false
    var sfxRenoise = ""
    var sfxSynchformerModel = "sfx-woosh-synchformer"
    var sfxSyncBatchSize = 1
    var sfxClipBatchSize = 4
    var pluginCatalogURL = ""
    var pluginChannel = ""
    var geoDimensions = ""
    var geoPatchSize = 4
    var geoInputResolution = 10.0
    var geoIncludeTokens = false
    var benchmarkModels = ""
    var benchmarkSuite = ""
    var benchmarkDataset = ""
    var benchmarkCases = ""
    var benchmarkTrials = ""
    var benchmarkResume = false
    var benchmarkFixtureCheck = false
    var benchmarkSandbox = ""
    var benchmarkAllowCodeExecution = false
    var speechListenDevice = ""
    var speechListenListDevices = false
    var speechListenDecodeMS = 0
    var speechListenSilenceMS = 0
    var visionServeMaxFrameBytes = 0
    var visionServeMaxBatchSize = 8
    var visionServeMaxBatchBytes = 0
    var openWebUIHost = "127.0.0.1"
    var openWebUIPort = 3_000
    var openWebUIContainerName = "open-webui-mere-run"
    var openWebUIVolumeName = "open-webui-mere-run"
    var openWebUIImage = "ghcr.io/open-webui/open-webui:main"
    var openWebUIVisionModel = "vision-chat-gemma4-12b"
    var openWebUIEmbeddingModel = "text-embed-qwen3-0.6b"
    var openWebUIImageModel = "image-zimage-nano"
    var openWebUITTSModel = "speech-tts-qwen3-nano"
    var openWebUISTTModel = "speech-asr-parakeet"
    var openWebUITTSFormat = "wav"
    var openWebUIAdminEmail = "admin@localhost"
    var openWebUIAdminPassword = "admin"
    var openWebUIWaitSeconds = 180
    var openWebUIPull = false
    var openWebUISkipServer = false
    var openWebUISkipDocker = false
    var openWebUISkipConfigure = false
    var openWebUIReset = false
    var apiLoRA = ""
    var apiRateLimitPerMinute = 60
    var apiMaxActiveRequests = 1
    var apiMemoryGuard = "balanced"
    var apiMemoryGuardCustomCeilingGB = ""
    var variant = "zimage"
    var quiet = false
    var force = false
    var all = false
    var acceptModelLicense = false
    var json = false
    var stream = false
    var extraArguments = ""
}

struct CommandTemplate: Identifiable, Equatable {
    let id: CommandTemplateID
    let category: CommandCategory
    let title: String
    let subtitle: String
    let systemImage: String
    let promptLabel: String?
    let secondaryLabel: String?
    let inputKind: CommandInputKind
    let outputKind: CommandOutputKind
    let defaultPrompt: String
    let defaultSecondaryText: String
    let defaultModel: String
    let defaultExtraArguments: String
    let externalURL: URL?

    init(
        id: CommandTemplateID,
        category: CommandCategory,
        title: String,
        subtitle: String,
        systemImage: String,
        promptLabel: String? = nil,
        secondaryLabel: String? = nil,
        inputKind: CommandInputKind = .none,
        outputKind: CommandOutputKind = .none,
        defaultPrompt: String = "",
        defaultSecondaryText: String = "",
        defaultModel: String = "",
        defaultExtraArguments: String = "",
        externalURL: URL? = nil
    ) {
        self.id = id
        self.category = category
        self.title = title
        self.subtitle = subtitle
        self.systemImage = systemImage
        self.promptLabel = promptLabel
        self.secondaryLabel = secondaryLabel
        self.inputKind = inputKind
        self.outputKind = outputKind
        self.defaultPrompt = defaultPrompt
        self.defaultSecondaryText = defaultSecondaryText
        self.defaultModel = defaultModel
        self.defaultExtraArguments = defaultExtraArguments
        self.externalURL = externalURL
    }

    /// The output kind the shared contract declares for this command, or nil for the rows the
    /// contract does not cover (`.custom`, and launcher-only entries).
    var declaredOutputKind: MereRunCapabilityOutputKind? {
        id.capability?.output.kind
    }

    /// True when the command materializes a file or directory the app can adopt as the run's
    /// primary artifact.
    ///
    /// The contract's declared kind is read first, so a command the app models as producing
    /// nothing still resolves its `--output` path when the contract says it writes one. The
    /// app's own `outputKind` still counts, because a handful of commands write a file the
    /// contract summarizes as `text` (the benchmarks' `--output` report) or as a `service`
    /// (`music realtime`, which also records a WAV); see
    /// `ArtifactReceiptTests.testDeclaredOutputKindsDriftOnlyWhereRecorded`.
    var producesOutputFile: Bool {
        if declaredOutputKind == .file || declaredOutputKind == .directory {
            return true
        }
        return outputKind != .none
    }

    func defaultDraft() -> CommandDraft {
        var draft = CommandDraft()
        draft.prompt = defaultPrompt
        draft.secondaryText = defaultSecondaryText
        draft.model = defaultModel
        draft.extraArguments = defaultExtraArguments

        switch id {
        case .textChat:
            draft.stream = true
        case .textCode:
            draft.temperature = 1.0
            draft.topP = 0.95
            draft.stream = true
        case .textTrainLoRA:
            draft.steps = 600
            draft.seed = "42"
        case .audioEnhance:
            draft.audioODEMethod = "midpoint"
            draft.audioODESteps = 4
            draft.audioGuidanceScale = 1.5
            draft.audioChunkSeconds = 10
            draft.audioDType = "float32"
            draft.seed = "42"
        case .audioGenerate:
            draft.useDuration = true
            draft.durationSeconds = 10
            draft.steps = 30
            draft.seed = "42"
        case .musicSeparate:
            draft.audioDType = "float16"
        case .visionEmbed:
            draft.maxTokens = 8_192
        case .visionCaption, .visionOCR:
            draft.maxTokens = id == .visionCaption ? 96 : 4096
            draft.temperature = id == .visionCaption ? 0.2 : 0.2
            if id == .visionOCR { draft.backend = "lighton" }
        case .visionDepthVideo:
            draft.dryRun = true
        case .visionGeometry, .visionGeometryMultiview:
            draft.dryRun = true
        case .visionFaceDetect, .visionFaceEmbed, .visionFaceCompare, .visionFaceBatch:
            draft.json = true
        case .apiServe:
            draft.engine = StudioChatDefaults.fallbackServingEngine
            draft.port = 8080
            draft.contextSize = 32_768
        case .openWebui:
            draft.host = "0.0.0.0"
            draft.port = 8080
        case .setup:
            draft.setupMode = "agent"
            draft.agentModel = "tier"
        case .agentStart:
            draft.prompt = defaultPrompt
            draft.port = 8080
        case .imageValidate:
            draft.backend = "all"
            draft.variant = "zimage"
        case .imageGenerate:
            draft.maxSequenceLength = 512
        case .imageTrainLoRA:
            draft.steps = 1000
            draft.maxSequenceLength = 512
        case .imageDatasetDiscover:
            draft.json = true
        case .geoFlood, .geoFire, .geoTessera, .geoOlmoEarth:
            draft.json = true
        case .modelLocationList:
            draft.json = true
        case .modelBenchmarkFused:
            draft.benchmarkSuite = "lite"
            draft.json = true
        case .modelBenchmarkChat, .modelBenchmarkToolCalls, .modelBenchmarkToolContinuations:
            draft.json = true
        case .modelBenchmarkCode:
            draft.benchmarkSandbox = "auto"
            draft.json = true
        case .modelBenchmarkVLM:
            draft.benchmarkDataset = "synthetic-vqa-v1"
            draft.json = true
        case .modelBenchmarkGemma4KV, .modelBenchmarkGemma4MTP:
            draft.json = true
        case .modelBenchmarkAPIWorkload:
            draft.port = 8080
            draft.json = true
        case .pluginInfo:
            draft.json = true
        case .visionServe:
            draft.port = 8_091
        case .imageRunPlan:
            draft.preflight = true
            draft.json = true
        case .imageVisualizeRun:
            draft.port = 8787
        case .imageReconstruct3D:
            draft.reconstructionResolution = 256
        case .imageReconstruct3DTrellis2:
            draft.seed = "42"
            draft.trellisTextureSeed = "42"
            draft.trellisNoRemesh = false
            draft.trellisRemeshBand = 1
            draft.trellisSealRadius = 12
            draft.maxTokens = 2_097_152
        case .imageReconstruct3DMultiview:
            draft.reconstructionResolution = 128
        case .videoGenerate:
            draft.width = 768
            draft.height = 512
            draft.steps = 40
            draft.cfgScale = 5
            draft.videoQuality = .final
            draft.videoOutputMode = .videoOnly
        case .videoRetake:
            draft.steps = 30
            draft.seed = "42"
            draft.retakeStartTime = 0
            draft.retakeEndTime = 4
        case .videoDubIt:
            draft.width = 768
            draft.height = 512
            draft.seed = "42"
        case .videoAnimate:
            draft.width = 832
            draft.height = 480
            draft.steps = 40
            draft.cfgScale = 5
            draft.scheduleShift = 3
            draft.fps = 16
            draft.seed = "42"
        case .videoCosmos3:
            draft.width = 1280
            draft.height = 720
            draft.numFrames = 189
            draft.steps = 0
            draft.cfgScale = 0
            draft.scheduleShift = 0
            draft.fps = 0
            draft.seed = "0"
        case .videoExportLatents:
            draft.width = 768
            draft.height = 512
            draft.numFrames = 65
            draft.seed = "42"
        case .musicGenerate:
            draft.steps = 8
            draft.durationSeconds = 10
            draft.useDuration = false
            draft.musicOverrideSteps = false
        case .musicAnalyze:
            draft.useDuration = false
        case .musicTranscribe:
            draft.temperature = 1
        case .musicRealtime:
            draft.durationSeconds = 30
            draft.musicPlay = true
        case .musicTrainAdapter:
            draft.steps = 1_000
            draft.rank = 8
            draft.alpha = 16
            draft.learningRate = 0.0001
            draft.seed = "42"
        case .musicServe:
            draft.port = 8081
        case .adapterList:
            draft.json = true
        case .modelRepairManifests:
            draft.force = true
            draft.json = true
        case .modelOptimize:
            draft.json = true
        case .runList:
            draft.json = true
        case .runInspect, .runFetch, .runCancel, .runRetry:
            draft.json = true
        case .evaluationPackValidate, .evaluationRun, .evaluationPromote:
            draft.json = true
            if id == .evaluationRun { draft.dryRun = true }
        case .worldServe:
            draft.port = 8791
            draft.model = "video-dreamx-world-5b-ar-mlx"
        case .statusSnapshot, .modelStorage, .modelGarbageCollect,
             .modelRuntimeGet, .modelRuntimeSet:
            draft.json = true
        case .qualityGate:
            draft.operationsGateSuite = "all"
        case .sfxGenerate, .sfxVideo:
            draft.steps = 4
            draft.durationSeconds = 8
            draft.cfgScale = id == .sfxVideo ? 3 : 4.5
        case .speechTranscribe:
            draft.backend = "auto"
            draft.maxTokens = 448
            draft.task = "transcribe"
            draft.language = "auto"
            draft.timestamps = true
        case .speechDiarize:
            draft.speechDiarizationFormat = "json"
            draft.speechDiarizationThreshold = 0.5
            draft.speechDiarizationMinDuration = 0.25
            draft.speechDiarizationMergeGap = 0.25
            draft.quiet = true
        case .modelBenchmark:
            draft.temperature = 0
            draft.topP = 0.9
            draft.contextSize = 16_384
        case .modelBenchmarkLagunaDFlash:
            draft.benchmarkDecodeTokenValues = "8,12,16,24,32,48"
            draft.temperature = 0
            draft.topP = 1
            draft.topK = 0
            draft.minP = 0.02
            draft.contextSize = 4_096
            draft.json = true
        default:
            break
        }

        // Destinations are user-visible folders per domain (`StudioOutputLocation`); the composer
        // renames the file after the prompt, and specialist surfaces keep this stamped name.
        if let path = StudioOutputLocation.templateOutputPath(
            templateID: id, title: title, outputKind: outputKind
        ) {
            draft.outputPath = path
        }

        return draft
    }

    func validationMessage(for draft: CommandDraft) -> String? {
        if promptLabel != nil
            && id != .custom
            && id != .musicRealtime
            && id != .visionEmbed
            && id != .visionSegment
            && id != .visionTrack
            && draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "\(promptLabel ?? "Prompt") is required."
        }

        let optionalInputs: Set<CommandTemplateID> = [
            .imageGenerate,
            .imageTrainLoRA,
            .videoGenerate,
            .musicTranscribe,
            .visionEmbed,
            .visionFaceBatch
        ]
        if inputKind != .none
            && !optionalInputs.contains(id)
            && draft.inputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "\(inputKind.title) path is required."
        }

        switch id {
        case .visionEmbed:
            if draft.prompt.isBlank && draft.inputPath.isBlank {
                return "Text or image input is required."
            }
        case .visionSegment, .visionTrack:
            if draft.prompt.isBlank && draft.visionBoxPrompts.isBlank
                && draft.visionPointPrompts.isBlank {
                return "Add a text, box, or point prompt."
            }
        case .visionFaceCompare, .visionFlow:
            if draft.visionSecondInputPath.isBlank {
                return "A second image is required."
            }
        case .visionFaceBatch:
            if draft.inputPath.isBlank && draft.visionAdditionalInputs.isBlank
                && draft.visionInputList.isBlank {
                return "Choose images or an input-list file."
            }
        case .visionGeometryMultiview:
            let images = ([draft.inputPath] + pathList(draft.visionAdditionalInputs))
                .filter { !$0.isBlank }
            if images.count < 2 {
                return "Add at least two ordered views."
            }
        case .musicGenerate:
            if draft.musicInstrumental,
               !draft.secondaryText.isBlank || !draft.musicLyricsFile.isBlank
                    || !draft.musicLRCFile.isBlank
            {
                return "Instrumental cannot be combined with lyrics."
            }
            if !draft.musicLRCFile.isBlank
                && (!draft.secondaryText.isBlank || !draft.musicLyricsFile.isBlank) {
                return "Use synchronized LRC or plain lyrics, not both."
            }
            let sourceTasks = ["repaint", "cover", "cover-nofsq", "extract", "lego", "complete"]
            if (sourceTasks.contains(draft.musicTask) || draft.musicFlowEdit)
                && draft.musicSourceAudio.isBlank {
                return "Source audio is required for \(draft.musicTask) and flow-edit workflows."
            }
        case .musicTranscribe:
            if draft.inputPath.isBlank && !draft.musicListInstruments {
                return "Audio path is required unless listing instruments."
            }
        case .musicRealtime:
            if draft.prompt.isBlank && !draft.musicListMIDIInputs && !draft.musicMIDIMonitor {
                return "A prompt is required unless listing or monitoring MIDI inputs."
            }
            if !draft.musicPlay && draft.outputPath.isBlank && !draft.musicListMIDIInputs
                && !draft.musicMIDIMonitor {
                return "Enable playback or choose an output file."
            }
        case .musicTrainAdapter:
            if draft.inputPath.isBlank {
                return "Dataset manifest is required."
            }
            if draft.outputPath.isBlank {
                return "Adapter output is required."
            }
        case .musicServe:
            if draft.host != "127.0.0.1" && draft.host != "localhost"
                && draft.host != "::1" && draft.apiKey.isBlank {
                return "An API key is required for non-loopback music servers."
            }
        case .adapterPull:
            if draft.prompt.isBlank {
                return "Adapter id is required."
            }
        case .runList:
            if draft.operationsRoot.isBlank == draft.operationsExecutor.isBlank {
                return "Choose exactly one local root or remote executor."
            }
            if !draft.operationsExecutor.isBlank && !(1...500).contains(draft.operationsLimit) {
                return "Remote result limit must be between 1 and 500."
            }
            if draft.operationsExecutor.isBlank && draft.maxDepth < 0 {
                return "Scan depth must be zero or greater."
            }
        case .runInspect, .runWatch, .runCancel, .runRetry:
            if draft.operationsReference.isBlank {
                return "Run path or remote reference is required."
            }
            if id == .runWatch && draft.operationsPollInterval < 0.25 {
                return "Polling interval must be at least 0.25 seconds."
            }
            if id == .runWatch && draft.operationsJSONStream && draft.json {
                return "Choose streaming events or one final JSON object, not both."
            }
        case .runFetch:
            if draft.operationsReference.isBlank {
                return "Remote run reference is required."
            }
            if draft.outputPath.isBlank {
                return "Destination run directory is required."
            }
            if draft.operationsAllArtifacts && !lineList(draft.operationsArtifacts).isEmpty {
                return "Choose all artifacts or named artifacts, not both."
            }
        case .evaluationPackValidate:
            if draft.inputPath.isBlank {
                return "Evaluation pack directory is required."
            }
        case .evaluationRun:
            if draft.inputPath.isBlank {
                return "Evaluation pack directory is required."
            }
            if lineList(draft.prompt).isEmpty {
                return "At least one model binding is required."
            }
        case .evaluationPromote:
            if draft.inputPath.isBlank {
                return "Evaluation report is required."
            }
        case .worldServe:
            if draft.model.isBlank || draft.operationsBaseModel.isBlank {
                return "Base and world model ids are required."
            }
            if !(1...65_535).contains(draft.port) {
                return "Port must be between 1 and 65535."
            }
            if draft.host != "127.0.0.1" && draft.host != "localhost"
                && draft.host != "::1" && draft.apiKey.isBlank {
                return "An API key is required for non-loopback world servers."
            }
        case .statusSnapshot:
            if !(1...65_535).contains(draft.port) {
                return "Port must be between 1 and 65535."
            }
            if draft.operationsTimeoutSeconds <= 0 {
                return "Probe timeout must be greater than zero."
            }
        case .qualityGate:
            let validSuites = Set(["text", "speech", "vision", "image", "embed"])
            let selectedSuites = Set(
                draft.operationsGateSuite
                    .split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            )
            if draft.operationsGateSuite.lowercased() != "all"
                && (selectedSuites.isEmpty || !selectedSuites.isSubset(of: validSuites)) {
                return "Suites must be all or a comma-separated set of text, speech, vision, image, and embed."
            }
        case .modelRuntimeGet, .modelRuntimeSet:
            if draft.model.isBlank {
                return "Managed model id or alias is required."
            }
            if draft.operationsPinned && draft.operationsUnpinned {
                return "Choose pinned or unpinned, not both."
            }
            let conflictingRuntimeValues = [
                (!draft.operationsRuntimeAlias.isBlank, draft.operationsClearAlias, "alias"),
                (!draft.operationsRuntimeTTL.isBlank, draft.operationsClearTTL, "TTL"),
                (!draft.operationsRuntimeContext.isBlank, draft.operationsClearContext, "context"),
                (!draft.operationsRuntimeMaxTokens.isBlank, draft.operationsClearMaxTokens, "max tokens"),
                (!draft.operationsRuntimeTemperature.isBlank, draft.operationsClearTemperature, "temperature"),
                (!draft.operationsRuntimeTopP.isBlank, draft.operationsClearTopP, "top-p"),
                (!draft.operationsRuntimeMinP.isBlank, draft.operationsClearMinP, "min-p"),
                (!draft.operationsRuntimeEngine.isBlank, draft.operationsClearEngine, "engine"),
                (!draft.operationsRuntimeKVCacheMode.isBlank, draft.operationsClearKVCacheMode, "KV cache")
            ]
            if let conflict = conflictingRuntimeValues.first(where: { $0.0 && $0.1 }) {
                return "Set or clear \(conflict.2), not both."
            }
        case .modelBenchmarkLagunaDFlash:
            if draft.modelRoot.isBlank {
                return "Laguna model path is required."
            }
            if draft.secondaryText.isBlank {
                return "Laguna DFlash model path is required."
            }
        case .imageTrainLoRA:
            if draft.inputPath.isBlank && draft.syntheticSamples <= 0 {
                return "A dataset directory is required unless synthetic samples are enabled."
            }
        case .modelPull:
            if !draft.all && draft.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return "Choose a model id, or enable All."
            }
        case .setup, .agentOnboard, .agentStart, .apiServe, .openWebui:
            if !(1...65_535).contains(draft.port) {
                return "Port must be between 1 and 65535."
            }
            if id == .openWebui && !(1...65_535).contains(draft.openWebUIPort) {
                return "Open WebUI port must be between 1 and 65535."
            }
            if id == .apiServe && draft.apiMemoryGuard == "custom"
                && draft.apiMemoryGuardCustomCeilingGB.isBlank {
                return "A custom memory ceiling is required for the custom guard."
            }
        case .speechSynthesize:
            if draft.stream && draft.speechStreamChunkTokens < 1 {
                return "Streaming chunk tokens must be greater than zero."
            }
        case .speechTranscribe:
            if draft.stream && (draft.speechStreamChunkMS < 1 || draft.speechStreamDecodeMS < 1) {
                return "Streaming feed and decode intervals must be greater than zero."
            }
        case .speechDiarize:
            let format = draft.speechDiarizationFormat ?? "json"
            if !["json", "rttm"].contains(format) {
                return "Diarization format must be JSON or RTTM."
            }
            if !(0...1).contains(draft.speechDiarizationThreshold ?? 0.5) {
                return "Diarization threshold must be between zero and one."
            }
            if (draft.speechDiarizationMinDuration ?? 0.25) < 0 {
                return "Minimum speaker duration must be zero or greater."
            }
            if (draft.speechDiarizationMergeGap ?? 0.25) < 0 {
                return "Speaker merge gap must be zero or greater."
            }
        case .modelRemove:
            if draft.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return "Model id is required."
            }
        case .modelInfo:
            if draft.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return "Model id or local model path is required."
            }
        case .modelOptimize:
            if draft.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return "MiniMax-H3 model id or local model path is required."
            }
        case .audioEnhance:
            if let overlap = draft.audioOverlap, overlap <= 0 {
                return "Overlap must be positive."
            }
            if draft.model.localizedCaseInsensitiveContains("universr") {
                if let inputRate = draft.audioInputRate,
                   ![8_000, 12_000, 16_000, 24_000].contains(inputRate) {
                    return "UniverSR input bandwidth must be 8000, 12000, 16000, or 24000 Hz."
                }
                if (draft.audioODESteps ?? 4) <= 0 {
                    return "UniverSR ODE steps must be positive."
                }
                if (draft.audioChunkSeconds ?? 10) < 3 {
                    return "UniverSR chunks must be at least 3 seconds."
                }
            }
        case .musicSeparate:
            if let overlap = draft.audioOverlap, overlap <= 0 {
                return "Overlap must be positive."
            }
        case .videoGenerate:
            let family = StudioVideoModelFamily(model: draft.modelRoot.isBlank ? draft.model : draft.modelRoot)
            if family == .miniMaxH3Ref2VA {
                if !(draft.inputPath.isBlank && draft.endImagePath.isBlank) {
                    return "MiniMax-H3 Ref2VA uses ordered image, video, or audio references instead of keyframes."
                }
                if (draft.h3ReferenceInputs ?? []).isEmpty {
                    return "MiniMax-H3 Ref2VA requires at least one ordered reference."
                }
            } else if !draft.endImagePath.isBlank && draft.inputPath.isBlank {
                return "A start image is required when an end keyframe is selected."
            }
            if family == .miniMaxH3FL2VA && !(draft.h3ReferenceInputs ?? []).isEmpty {
                return "MiniMax-H3 FL2VA does not accept ordered references."
            }
        case .videoRetake:
            if draft.retakeStartTime < 0 || draft.retakeStartTime >= draft.retakeEndTime {
                return "Retake requires a nonnegative start before the end time."
            }
            if draft.retakePreserveVideo && draft.retakePreserveAudio {
                return "Retake must regenerate video, audio, or both."
            }
        case .videoDubIt:
            if draft.loraPath.isBlank {
                return "Dub-It requires an IC-LoRA file."
            }
        case .imageReconstruct3DMultiview:
            let views = pathList(draft.referenceImagePaths)
            if views.count != 4 && views.count != 6 {
                return "Add exactly 4 or 6 ordered source views."
            }
        case .videoAnimate:
            if draft.referenceMaskPath.isBlank {
                return "Reference mask path is required."
            }
            if draft.drivingVideoPath.isBlank {
                return "Driving video path is required."
            }
            if draft.drivingMaskPath.isBlank {
                return "Driving mask path is required."
            }
            let additionalReferences = pathList(draft.referenceImagePaths)
            let additionalMasks = pathList(draft.scailAdditionalReferenceMaskPaths ?? "")
            if additionalReferences.count != additionalMasks.count {
                return "Each additional SCAIL reference needs one matching reference mask."
            }
            if additionalReferences.count > 5 {
                return "SCAIL supports at most six subjects total."
            }
        case .videoPrepareMasks:
            if draft.outputPath.isBlank {
                return "Output directory is required."
            }
        case .custom:
            if ShellWords.split(draft.extraArguments).isEmpty {
                return "Enter mere.run arguments."
            }
        default:
            break
        }

        return nil
    }

    func arguments(from draft: CommandDraft) -> [String] {
        var args: [String]

        switch id {
        case .setup:
            args = ["setup", "--mode", draft.setupMode, "--agent-model", draft.agentModel]
            if draft.force { args.append("--install") }
            if draft.stream { args.append("--start") }
            if draft.dryRun { args.append("--dry-run") }
            args += ["--host", draft.host, "--port", String(draft.port)]
            if !draft.piPath.isBlank { args += ["--pi-path", draft.piPath] }
            if draft.quiet { args.append("--quiet") }

        case .agentOnboard:
            args = ["agent", "onboard"]
            if draft.force { args.append("--pull-recommended") }
            if draft.acceptModelLicense { args.append("--accept-model-license") }
            if draft.all { args.append("--install-pi") }
            if draft.stream { args.append("--configure-pi") }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--host", draft.host, "--port", String(draft.port)]
            if draft.quiet { args.append("--quiet") }

        case .agentStatus:
            args = ["agent", "status"]
            if !draft.piPath.isBlank { args += ["--pi-path", draft.piPath] }
            if draft.json { args.append("--json") }

        case .agentInstallPi:
            args = ["agent", "install-pi"]
            if draft.force { args.append("--force") }

        case .agentStart:
            args = ["agent", "start", "--host", draft.host, "--port", String(draft.port)]
            if !draft.piPath.isBlank { args += ["--pi-path", draft.piPath] }
            if !draft.prompt.isBlank { args += ["--prompt", draft.prompt] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.stream { args.append("--skip-server") }
            if draft.force { args.append("--allow-unsupported") }
            if draft.noBootstrap { args.append("--no-bootstrap") }
            if draft.quiet { args.append("--quiet") }

        case .modelList:
            args = ["model", "list"]

        case .modelCapabilities:
            args = ["model", "capabilities"]
            if draft.all { args.append("--all") }
            if draft.force { args.append("--recommended") }
            if draft.json { args.append("--json") }

        case .modelPull:
            args = ["model", "pull"]
            if draft.all {
                args.append("--all")
            } else {
                args.append(draft.model)
            }
            if draft.force { args.append("--force") }
            if draft.stream { args.append("--allow-unsupported") }
            if draft.quiet { args.append("--quiet") }
            if draft.acceptModelLicense { args.append("--accept-model-license") }
            if draft.preflight { args.append("--preflight") }
            if draft.preflight, draft.json { args.append("--json") }

        case .modelInfo:
            args = ["model", "info", draft.model]
            if draft.all { args.append("--json") }
            if draft.force { args.append("--components") }

        case .modelRemove:
            args = ["model", "remove", draft.model]
            if draft.force { args.append("--force") }
            if draft.modelKeepCache { args.append("--keep-cache") }
            if draft.modelRemovalJSON { args.append("--json") }

        case .modelRepairManifests:
            args = ["model", "repair-manifests"]
            if draft.force { args.append("--dry-run") }
            if draft.json { args.append("--json") }

        case .modelOptimize:
            args = ["model", "optimize", draft.model]
            if draft.force { args.append("--force") }
            if draft.json { args.append("--json") }

        case .imageGenerate:
            args = ["image", "generate", "--prompt", draft.prompt, "--output", draft.outputPath]
            args += ["--width", String(draft.width), "--height", String(draft.height)]
            args += ["--steps", String(draft.steps)]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.secondaryText.isBlank { args += ["--negative-prompt", draft.secondaryText] }
            if draft.cfgScale != 1.0 { args += ["--cfg", format(draft.cfgScale)] }
            if draft.sigmaShift > 0 { args += ["--sigma-shift", format(draft.sigmaShift)] }
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if !draft.inputPath.isBlank {
                args += ["--input", draft.inputPath, "--strength", format(draft.strength)]
            }
            if let mask = draft.imageMaskPath, !mask.isBlank { args += ["--mask", mask] }
            if let outpaint = draft.imageOutpaint, !outpaint.isBlank { args += ["--outpaint", outpaint] }
            if let feather = draft.imageMaskFeather, feather != 8 {
                args += ["--mask-feather", String(feather)]
            }
            for path in pathList(draft.referenceImagePaths) {
                args += ["--ref-image", path]
            }
            if draft.keepOriginalAspect { args.append("--keep-original-aspect") }
            if draft.inputPath.isBlank, !draft.referenceImagePaths.isBlank, draft.strength != 0 {
                args += ["--strength", format(draft.strength)]
            }
            if draft.maxSequenceLength != 512 {
                args += ["--max-sequence-length", String(draft.maxSequenceLength)]
            }
            if draft.structuredPrompt {
                args.append("--structured-prompt")
                if !draft.structuredPromptModel.isBlank {
                    args += ["--structured-prompt-model", draft.structuredPromptModel]
                }
                if !draft.structuredPromptModelRoot.isBlank {
                    args += ["--structured-prompt-model-root", draft.structuredPromptModelRoot]
                }
                args += ["--structured-prompt-max-tokens", String(draft.structuredPromptMaxTokens)]
                if !draft.structuredPromptOutputPath.isBlank {
                    args += ["--structured-prompt-output", draft.structuredPromptOutputPath]
                }
            }
            if !draft.loraPath.isBlank {
                args += ["--lora", draft.loraPath, "--lora-scale", format(draft.loraScale)]
            }
            if draft.kreaConditioningMultiplier > 0 {
                args += ["--krea-conditioning-multiplier", format(draft.kreaConditioningMultiplier)]
            }
            if !draft.kreaConditioningLayerWeights.isBlank {
                args += ["--krea-conditioning-layer-weights", draft.kreaConditioningLayerWeights]
            }
            if !draft.kreaBaseQuantizationBits.isBlank {
                args += ["--krea-base-quantization-bits", draft.kreaBaseQuantizationBits]
            }
            if draft.preflight { args.append("--preflight") }
            if draft.preflight, draft.json { args.append("--json") }
            if draft.progressJSON { args.append("--progress-json") }
            if draft.quiet { args.append("--quiet") }

        case .imageTrainLoRA:
            args = ["image", "train-lora", "--output", draft.outputPath]
            if !draft.inputPath.isBlank { args += ["--data", draft.inputPath] }
            if !draft.trainingRecipe.isBlank { args += ["--recipe", draft.trainingRecipe] }
            let emitsRecipeOverrides = draft.trainingRecipe.isBlank || draft.overrideTrainingRecipe
            if emitsRecipeOverrides {
                args += ["--width", String(draft.width), "--height", String(draft.height)]
                args += ["--training-steps", String(draft.steps)]
                if !draft.model.isBlank { args += ["--model", draft.model] }
                args += ["--learning-rate", format(draft.learningRate)]
                args += ["--rank", String(draft.rank)]
                if draft.alpha > 0 { args += ["--alpha", format(draft.alpha)] }
                if draft.captionDropout > 0 {
                    args += ["--caption-dropout", format(draft.captionDropout)]
                }
            }
            args += ["--batch-size", String(draft.batchSize)]
            args += ["--max-text-length", String(draft.maxSequenceLength)]
            args += ["--scheduler-steps", String(draft.schedulerSteps)]
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if draft.trainingLite { args.append("--lite") }
            if !draft.baseQuantizationBits.isBlank {
                args += ["--base-quantization-bits", draft.baseQuantizationBits]
            }
            if draft.excludePreviewImages { args.append("--exclude-preview-images") }
            if emitsRecipeOverrides, draft.checkpointInterval > 0 {
                args += ["--checkpoint-interval", String(draft.checkpointInterval)]
            }
            if let resumePath = draft.trainingResumePath, !resumePath.isBlank {
                args += ["--resume-from", resumePath]
            }
            if emitsRecipeOverrides, draft.maxResolution > 0 {
                args += ["--max-resolution", String(draft.maxResolution)]
            }
            if draft.progressive { args.append("--progressive") }
            if emitsRecipeOverrides, draft.lowRAM { args.append("--low-ram") }
            if emitsRecipeOverrides, draft.disableCompile { args.append("--no-compile") }
            if draft.gradientCheckpointing { args.append("--gradient-checkpointing") }
            if draft.benchmarkSteps > 0 { args += ["--benchmark-steps", String(draft.benchmarkSteps)] }
            if draft.benchmarkSteps > 0 {
                args += ["--benchmark-warmup-steps", String(draft.benchmarkWarmupSteps)]
            }
            if draft.sampleInterval > 0 {
                args += ["--sample-interval", String(draft.sampleInterval)]
                if !draft.samplePrompt.isBlank { args += ["--sample-prompt", draft.samplePrompt] }
                if !draft.sampleModel.isBlank { args += ["--sample-model", draft.sampleModel] }
                args += ["--sample-steps", String(draft.sampleSteps)]
                args += ["--sample-cfg", format(draft.sampleCFG)]
                args += ["--sample-lora-scale", format(draft.sampleLoRAScale)]
                if !draft.sampleSeed.isBlank { args += ["--sample-seed", draft.sampleSeed] }
            }
            if draft.visualize {
                args += ["--visualize", "--visualize-port", String(draft.visualizePort)]
            }
            if draft.preflight { args.append("--preflight") }
            if draft.preflight, draft.json { args.append("--json") }
            if !draft.loraTargetRanks.isBlank { args += ["--lora-target-ranks", draft.loraTargetRanks] }
            if !draft.loraRankPreset.isBlank { args += ["--lora-rank-preset", draft.loraRankPreset] }
            if emitsRecipeOverrides, !draft.loraTargetPreset.isBlank {
                args += ["--lora-target-preset", draft.loraTargetPreset]
            }
            if !draft.loraTargetMode.isBlank { args += ["--lora-target-mode", draft.loraTargetMode] }
            if !draft.timestepSampling.isBlank { args += ["--timestep-sampling", draft.timestepSampling] }
            if !draft.timestepLossWeighting.isBlank {
                args += ["--timestep-loss-weighting", draft.timestepLossWeighting]
            }
            if !draft.lossWeighting.isBlank { args += ["--loss-weighting", draft.lossWeighting] }
            if draft.timestepLow > 0 { args += ["--timestep-low", String(draft.timestepLow)] }
            if draft.timestepHigh > 0 { args += ["--timestep-high", String(draft.timestepHigh)] }
            if emitsRecipeOverrides, draft.lrWarmupSteps > 0 {
                args += ["--lr-warmup-steps", String(draft.lrWarmupSteps)]
            }
            if draft.disableCosineScheduler { args.append("--no-cosine-scheduler") }
            if emitsRecipeOverrides, draft.lrMinFactor > 0 {
                args += ["--lr-min-factor", format(draft.lrMinFactor)]
            }
            if draft.adamWeightDecay > 0 {
                args += ["--adam-weight-decay", format(draft.adamWeightDecay)]
            }
            if draft.syntheticSamples > 0 { args += ["--synthetic-samples", String(draft.syntheticSamples)] }
            if draft.quiet { args.append("--quiet") }

        case .imageValidate:
            args = ["image", "validate", "--test", draft.backend, "--family", draft.variant]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if draft.force { args.append("--save-reference") }
            if draft.all { args.append("--compare") }
            if !draft.referenceDirectoryPath.isBlank {
                args += ["--reference-dir", draft.referenceDirectoryPath]
            }

        case .imageDatasetDiscover:
            args = [
                "image", "dataset", "discover",
                "--root", draft.inputPath,
                "--max-depth", String(draft.maxDepth),
                "--min-usable-pairs", String(draft.minUsablePairs)
            ]
            if !draft.trainingOutputRoot.isBlank {
                args += ["--training-output-root", draft.trainingOutputRoot]
            }
            if !draft.trainingModel.isBlank { args += ["--training-model", draft.trainingModel] }
            if !draft.trainingRecipe.isBlank { args += ["--training-recipe", draft.trainingRecipe] }
            if draft.excludePreviewImages { args.append("--exclude-preview-images") }
            if draft.json { args.append("--json") }

        case .imageRunPlan:
            args = ["image", "run-plan", draft.inputPath]
            if draft.preflight { args.append("--preflight") }
            if !draft.materializePath.isBlank { args += ["--materialize", draft.materializePath] }
            if draft.json { args.append("--json") }

        case .imageVisualizeRun:
            args = ["image", "visualize-run", draft.inputPath, "--port", String(draft.port)]

        case .imageReconstruct3D:
            args = ["image", "reconstruct-3d", draft.inputPath]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--resolution", String(draft.reconstructionResolution)]
            args += ["--density-threshold", format(draft.densityThreshold)]
            args += ["--foreground-ratio", format(draft.foregroundRatio)]
            if draft.alreadyFramed { args.append("--already-framed") }
            if draft.noVertexColors { args.append("--no-vertex-colors") }
            if draft.dryRun { args.append("--dry-run") }
            if draft.json { args.append("--json") }

        case .imageReconstruct3DTrellis2:
            args = ["image", "reconstruct-3d-trellis2", draft.inputPath]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if let textureSeed = draft.trellisTextureSeed, !textureSeed.isBlank {
                args += ["--texture-seed", textureSeed]
            }
            args += ["--max-tokens", String(draft.maxTokens)]
            if draft.alreadyFramed { args.append("--already-framed") }
            if draft.trellisNoRemesh == true { args.append("--no-remesh") }
            if let remeshBand = draft.trellisRemeshBand {
                args += ["--remesh-band", format(remeshBand)]
            }
            if let sealRadius = draft.trellisSealRadius {
                args += ["--seal-radius", String(sealRadius)]
            }
            if draft.dryRun { args.append("--dry-run") }
            if draft.json { args.append("--json") }

        case .imageReconstruct3DMultiview:
            args = ["image", "reconstruct-3d-multiview"]
            for path in pathList(draft.referenceImagePaths) {
                args += ["--view", path]
            }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.camerasPath.isBlank { args += ["--cameras", draft.camerasPath] }
            args += ["--resolution", String(draft.reconstructionResolution)]
            if draft.noVertexColors { args.append("--no-vertex-colors") }
            if draft.dryRun { args.append("--dry-run") }
            if draft.json { args.append("--json") }

        case .textChat:
            args = ["text", "chat", "--prompt", draft.prompt]
            if !draft.secondaryText.isBlank { args += ["--system", draft.secondaryText] }
            if !draft.imagePath.isBlank { args += ["--image", draft.imagePath] }
            args += [
                "--max-tokens", String(draft.maxTokens),
                "--temperature", format(draft.temperature),
                "--top-p", format(draft.topP)
            ]
            if draft.contextSize > 0 { args += ["--context-size", String(draft.contextSize)] }
            if draft.topK > 0 { args += ["--top-k", String(draft.topK)] }
            if draft.minP > 0 { args += ["--min-p", format(draft.minP)] }
            if draft.kvBits > 0 { args += ["--kv-bits", String(draft.kvBits)] }
            if !draft.kvQuantScheme.isBlank { args += ["--kv-quant-scheme", draft.kvQuantScheme] }
            if draft.kvGroupSize > 0 { args += ["--kv-group-size", String(draft.kvGroupSize)] }
            if draft.quantizedKVStart > 0 {
                args += ["--quantized-kv-start", String(draft.quantizedKVStart)]
            }
            if !draft.modelRoot.isBlank { args += ["--model-root", draft.modelRoot] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.responseFormat != .text {
                args += ["--response-format", draft.responseFormat.rawValue]
            }
            if !draft.loraPath.isBlank {
                args += ["--lora", draft.loraPath, "--lora-scale", format(draft.loraScale)]
            }
            switch draft.thinkingMode {
            case .automatic:
                break
            case .show:
                args.append("--thinking")
            case .hide:
                args.append("--no-thinking")
            }
            if let reasoningEffort = draft.reasoningEffort {
                args += ["--reasoning-effort", format(reasoningEffort)]
            }
            if !draft.tools.isBlank { args += ["--tools", draft.tools] }
            if draft.toolLoop { args.append("--tool-loop") }
            if draft.allowShellExec { args.append("--allow-shell-exec") }
            if draft.allowAbsoluteToolPaths { args.append("--allow-absolute-tool-paths") }
            if draft.autoApproveTools { args.append("--auto-approve-tools") }
            if !draft.sandboxDir.isBlank { args += ["--sandbox-dir", draft.sandboxDir] }
            if draft.stream { args.append("--stream") }
            if draft.force { args.append("--stats") }
            if draft.quiet { args.append("--quiet") }
            if draft.preflight { args.append("--preflight") }
            if draft.preflight, draft.json { args.append("--json") }
            if draft.requireInstalled { args.append("--require-installed") }

        case .textCode:
            args = ["text", "code", "--prompt", draft.prompt]
            if !draft.secondaryText.isBlank { args += ["--system", draft.secondaryText] }
            args += ["--max-tokens", String(draft.maxTokens), "--temperature", format(draft.temperature), "--top-p", format(draft.topP)]
            if draft.minP > 0 { args += ["--min-p", format(draft.minP)] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.stream { args.append("--stream") }
            if draft.force { args.append("--stats") }
            if draft.quiet { args.append("--quiet") }

        case .textEmbed:
            let embeddingTexts = draft.prompt
                .components(separatedBy: .newlines)
                .filter { !$0.isBlank }
            args = ["text", "embed"] + (embeddingTexts.isEmpty ? [draft.prompt] : embeddingTexts)
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.maxTokens > 0 { args += ["--max-tokens", String(draft.maxTokens)] }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if draft.force { args.append("--pretty") }

        case .textAnonymize:
            args = ["text", "anonymize", draft.prompt]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.maxTokens > 0 { args += ["--max-tokens", String(draft.maxTokens)] }
            if draft.replacement != "[{label}]" {
                args += ["--replacement", draft.replacement]
            }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if draft.all { args.append("--json") }
            if draft.force { args.append("--pretty") }

        case .textTrainLoRA:
            args = [
                "text", "train-lora",
                "--data", draft.inputPath,
                "--output", draft.outputPath,
                "--model", draft.model,
                "--adapter-name", draft.adapterName,
                "--training-steps", String(draft.steps),
                "--batch-size", String(draft.batchSize),
                "--learning-rate", format(draft.learningRate),
                "--rank", String(draft.rank),
                "--max-sequence-length", String(draft.maxSequenceLength),
                "--seed", draft.seed
            ]
            if !draft.modelRoot.isBlank { args += ["--model-path", draft.modelRoot] }
            if !draft.evalPath.isBlank { args += ["--eval", draft.evalPath] }
            if draft.alpha > 0 { args += ["--alpha", format(draft.alpha)] }
            if let reasoningEffort = draft.reasoningEffort {
                args += ["--reasoning-effort", format(reasoningEffort)]
            }
            if !draft.targetModules.isBlank {
                args += ["--target-modules", draft.targetModules]
            }
            if draft.dryRun { args.append("--dry-run") }
            if draft.visualize {
                args += ["--visualize", "--visualize-port", String(draft.visualizePort)]
            }
            if draft.json { args.append("--json") }

        case .speechSynthesize:
            args = ["speech", "synthesize", draft.prompt, "--output", draft.outputPath]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.secondaryText.isBlank { args += ["--voice", draft.secondaryText] }
            if draft.voiceMode == "clone" { args += ["--mode", "clone"] }
            if !draft.voiceProfile.isBlank { args += ["--profile", draft.voiceProfile] }
            if !draft.refAudioPath.isBlank { args += ["--ref-audio", draft.refAudioPath] }
            if !draft.refText.isBlank { args += ["--ref-text", draft.refText] }
            if !draft.saveProfileName.isBlank { args += ["--save-profile", draft.saveProfileName] }
            if !draft.language.isBlank, draft.language != "auto" { args += ["--language", draft.language] }
            args += ["--temperature", format(draft.temperature)]
            if draft.stream {
                args += ["--stream", "--stream-chunk-tokens", String(draft.speechStreamChunkTokens)]
            }
            if draft.quiet { args.append("--quiet") }

        case .speechTranscribe:
            args = ["speech", "transcribe", draft.inputPath]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--backend", draft.backend, "--task", draft.task, "--max-tokens", String(draft.maxTokens)]
            if !draft.language.isBlank, draft.language != "auto" { args += ["--language", draft.language] }
            if draft.stream {
                args += [
                    "--stream",
                    "--stream-chunk-ms", String(draft.speechStreamChunkMS),
                    "--stream-decode-ms", String(draft.speechStreamDecodeMS)
                ]
            }
            if !draft.speechInputFormat.isBlank {
                args += ["--input-format", draft.speechInputFormat]
            }
            if draft.speechSampleRate != 16_000 {
                args += ["--sample-rate", String(draft.speechSampleRate)]
            }
            if draft.speechJSONL { args.append("--jsonl") }
            if !draft.timestamps { args.append("--no-timestamps") }
            if draft.quiet { args.append("--quiet") }

        case .speechDiarize:
            args = ["speech", "diarize", draft.inputPath]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if let outputFormat = draft.speechDiarizationFormat {
                args += ["--format", outputFormat]
            }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if let threshold = draft.speechDiarizationThreshold {
                args += ["--threshold", format(threshold)]
            }
            if let minimumDuration = draft.speechDiarizationMinDuration {
                args += ["--min-duration", format(minimumDuration)]
            }
            if let mergeGap = draft.speechDiarizationMergeGap {
                args += ["--merge-gap", format(mergeGap)]
            }
            if draft.quiet { args.append("--quiet") }

        case .speechProfileList:
            args = ["speech", "profile", "list"]

        case .speechProfileCreate:
            args = ["speech", "profile", "create", "--name", draft.prompt, "--audio", draft.inputPath]
            if !draft.secondaryText.isBlank { args += ["--text", draft.secondaryText] }
            if !draft.language.isBlank { args += ["--language", draft.language] }
            if draft.quiet { args.append("--quiet") }

        case .speechProfileDelete:
            args = ["speech", "profile", "delete", "--id", draft.prompt]

        case .visionInspect:
            args = ["vision", "inspect", draft.inputPath, "--prompt", draft.prompt]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--max-tokens", String(draft.maxTokens), "--temperature", format(draft.temperature), "--top-p", format(draft.topP)]

        case .visionEmbed:
            args = ["vision", "embed"]
            if !draft.prompt.isBlank { args += ["--text", draft.prompt] }
            if !draft.inputPath.isBlank { args += ["--image", draft.inputPath] }
            if !draft.secondaryText.isBlank { args += ["--instruction", draft.secondaryText] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--max-tokens", String(draft.maxTokens)]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }

        case .visionCaption:
            args = ["vision", "caption"]
            if !draft.inputPath.isBlank { args.append(draft.inputPath) }
            args.append(contentsOf: pathList(draft.visionAdditionalInputs))
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.outputPath.isBlank { args += ["--output-dir", draft.outputPath] }
            if !draft.prompt.isBlank { args += ["--prompt", draft.prompt] }
            if !draft.visionPromptFile.isBlank { args += ["--prompt-file", draft.visionPromptFile] }
            for focus in lineList(draft.visionFocus) { args += ["--focus", focus] }
            if !draft.visionTriggerToken.isBlank {
                args += ["--trigger-token", draft.visionTriggerToken]
            }
            args += [
                "--max-tokens", String(draft.maxTokens),
                "--temperature", format(draft.temperature),
                "--top-p", format(draft.topP)
            ]

        case .visionOCR:
            args = ["vision", "ocr"]
            if !draft.inputPath.isBlank { args.append(draft.inputPath) }
            args.append(contentsOf: pathList(draft.visionAdditionalInputs))
            args += ["--backend", draft.backend]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.outputPath.isBlank { args += ["--output-dir", draft.outputPath] }
            args += ["--max-tokens", String(draft.maxTokens), "--temperature", format(draft.temperature)]
            if draft.all { args.append("--compare") }
            if !draft.visionGLMOCRCLI.isBlank { args += ["--glmocr-cli", draft.visionGLMOCRCLI] }
            if !draft.visionGLMConfig.isBlank { args += ["--glm-config", draft.visionGLMConfig] }
            args += [
                "--infinity-runtime", draft.visionInfinityRuntime,
                "--infinity-parser-cli", draft.visionInfinityParserCLI,
                "--infinity-model", draft.visionInfinityModel,
                "--infinity-backend", draft.visionInfinityBackend,
                "--infinity-api-url", draft.visionInfinityAPIURL,
                "--infinity-api-key", draft.visionInfinityAPIKey,
                "--infinity-task", draft.visionInfinityTask,
                "--infinity-output-format", draft.visionInfinityOutputFormat,
                "--infinity-batch-size", String(draft.visionInfinityBatchSize),
                "--infinity-min-pixels", String(draft.visionInfinityMinPixels),
                "--infinity-max-pixels", String(draft.visionInfinityMaxPixels)
            ]
            if !draft.visionInfinityPrompt.isBlank {
                args += ["--infinity-prompt", draft.visionInfinityPrompt]
            }
            if !draft.visionInfinityModelCacheDirectory.isBlank {
                args += ["--infinity-model-cache-dir", draft.visionInfinityModelCacheDirectory]
            }
            if draft.quiet { args.append("--quiet") }

        case .visionGround:
            args = ["vision", "ground", draft.inputPath]
            for query in lineList(draft.prompt) { args += ["--query", query] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.visionJSONOutputPath.isBlank {
                args += ["--json-output", draft.visionJSONOutputPath]
            }
            if !draft.visionMaskOutputDirectory.isBlank {
                args += ["--mask-output-dir", draft.visionMaskOutputDirectory]
            }

        case .visionSegment:
            args = ["vision", "segment", draft.inputPath]
            for prompt in lineList(draft.prompt) { args += ["--prompt", prompt] }
            for box in lineList(draft.visionBoxPrompts) { args += ["--box", box] }
            for point in lineList(draft.visionPointPrompts) { args += ["--point", point] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.visionJSONOutputPath.isBlank {
                args += ["--json-output", draft.visionJSONOutputPath]
            }
            if !draft.visionMaskOutputDirectory.isBlank {
                args += ["--mask-output-dir", draft.visionMaskOutputDirectory]
            }
            args += [
                "--threshold", format(draft.visionThreshold),
                "--resolution", String(draft.visionResolution)
            ]
            if draft.force { args.append("--show-boxes") }
            if draft.visionMultimask { args.append("--multimask") }

        case .visionTrack:
            args = ["vision", "track", draft.inputPath]
            for prompt in lineList(draft.prompt) { args += ["--prompt", prompt] }
            for box in lineList(draft.visionBoxPrompts) { args += ["--box", box] }
            for point in lineList(draft.visionPointPrompts) { args += ["--point", point] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.visionJSONOutputPath.isBlank {
                args += ["--json-output", draft.visionJSONOutputPath]
            }
            if !draft.visionMaskOutputDirectory.isBlank {
                args += ["--mask-output-dir", draft.visionMaskOutputDirectory]
            }
            args += [
                "--init-frame", String(draft.visionInitFrame),
                "--threshold", format(draft.visionThreshold),
                "--resolution", String(draft.visionResolution)
            ]
            if !draft.visionEndFrame.isBlank { args += ["--end-frame", draft.visionEndFrame] }
            if draft.force { args.append("--show-boxes") }
            if draft.visionShowLabels { args.append("--show-labels") }
            if draft.preflight {
                args.append("--preflight")
                if draft.json { args.append("--json") }
            }

        case .visionTrackLive:
            args = ["vision", "track-live"]
            for prompt in lineList(draft.prompt) { args += ["--prompt", prompt] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.visionJSONOutputPath.isBlank {
                args += ["--json-output", draft.visionJSONOutputPath]
            }
            args += [
                "--camera", String(draft.visionCamera),
                "--duration-seconds", format(draft.durationSeconds),
                "--init-frame", String(draft.visionInitFrame),
                "--seed-search-frames", String(draft.visionSeedSearchFrames),
                "--threshold", format(draft.visionThreshold),
                "--resolution", String(draft.visionResolution)
            ]
            if draft.force { args.append("--show-boxes") }
            if draft.visionShowLabels { args.append("--show-labels") }

        case .visionFaceDetect:
            args = ["vision", "face", "detect", draft.inputPath]
            appendFaceOptions(to: &args, draft: draft)
            if draft.visionMaxFaces > 0 { args += ["--max-faces", String(draft.visionMaxFaces)] }
            if draft.visionIncludeEmbeddings { args.append("--include-embeddings") }

        case .visionFaceEmbed:
            args = ["vision", "face", "embed", draft.inputPath]
            appendFaceOptions(to: &args, draft: draft)
            if !draft.visionFaceIndex.isBlank { args += ["--face-index", draft.visionFaceIndex] }

        case .visionFaceCompare:
            args = ["vision", "face", "compare", draft.inputPath, draft.visionSecondInputPath]
            appendFaceOptions(to: &args, draft: draft)
            if !draft.visionReferenceFaceIndex.isBlank {
                args += ["--reference-face-index", draft.visionReferenceFaceIndex]
            }
            if !draft.visionCandidateFaceIndex.isBlank {
                args += ["--candidate-face-index", draft.visionCandidateFaceIndex]
            }

        case .visionFaceBatch:
            args = ["vision", "face", "batch"]
            if !draft.inputPath.isBlank { args.append(draft.inputPath) }
            args.append(contentsOf: pathList(draft.visionAdditionalInputs))
            if !draft.visionInputList.isBlank { args += ["--input-list", draft.visionInputList] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += [
                "--score-threshold", format(draft.visionFaceScoreThreshold),
                "--execution-provider", draft.visionExecutionProvider
            ]
            if draft.visionMaxFaces > 0 { args += ["--max-faces", String(draft.visionMaxFaces)] }
            if draft.visionIncludeEmbeddings { args.append("--include-embeddings") }
            if !draft.visionJSONLOutput.isBlank {
                args += ["--jsonl-output", draft.visionJSONLOutput]
            }
            if draft.visionFailFast { args.append("--fail-fast") }

        case .visionPose:
            args = ["vision", "pose", draft.inputPath]
            if !draft.visionJSONOutputPath.isBlank {
                args += ["--json-output", draft.visionJSONOutputPath]
            }
            if !draft.visionPoseBody { args.append("--no-body") }
            if !draft.visionPoseHands { args.append("--no-hands") }
            if !draft.visionPoseFace { args.append("--no-face") }
            args += [
                "--max-hands", String(draft.visionMaxHands),
                "--minimum-confidence", format(draft.visionMinimumConfidence)
            ]
            if draft.json { args.append("--json") }

        case .visionFlow:
            args = ["vision", "flow", draft.inputPath, draft.visionSecondInputPath]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.visionJSONOutputPath.isBlank {
                args += ["--json-output", draft.visionJSONOutputPath]
            }
            args += ["--accuracy", draft.visionFlowAccuracy]
            if draft.json { args.append("--json") }

        case .visionDepthVideo:
            args = ["vision", "depth-video", draft.inputPath]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += [
                "--input-size", String(draft.visionInputSize),
                "--max-frames", String(draft.visionMaxFrames)
            ]
            if draft.dryRun { args.append("--dry-run") }
            if draft.json { args.append("--json") }

        case .visionGeometry:
            args = ["vision", "geometry", draft.inputPath]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--resolution-level", String(draft.visionResolutionLevel)]
            if draft.visionTokenCount > 0 {
                args += ["--token-count", String(draft.visionTokenCount)]
            }
            if draft.visionMaxPoints > 0 {
                args += ["--max-points", String(draft.visionMaxPoints)]
            }
            if draft.dryRun { args.append("--dry-run") }
            if draft.json { args.append("--json") }

        case .visionGeometryMultiview:
            args = ["vision", "geometry-multiview"]
            if !draft.inputPath.isBlank { args.append(draft.inputPath) }
            args.append(contentsOf: pathList(draft.visionAdditionalInputs))
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.camerasPath.isBlank { args += ["--cameras", draft.camerasPath] }
            args += [
                "--process-resolution", String(draft.visionProcessResolution),
                "--reference-view", draft.visionReferenceView,
                "--confidence-percentile", format(draft.visionConfidencePercentile)
            ]
            if draft.visionMaxPoints > 0 {
                args += ["--max-points", String(draft.visionMaxPoints)]
            }
            if draft.dryRun { args.append("--dry-run") }
            if draft.json { args.append("--json") }

        case .musicGenerate:
            args = ["music", "generate", draft.prompt]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.musicInstrumental {
                if !draft.secondaryText.isBlank { args += ["--lyrics", draft.secondaryText] }
                if !draft.musicLyricsFile.isBlank { args += ["--lyrics-file", draft.musicLyricsFile] }
                if !draft.musicLRCFile.isBlank { args += ["--lrc-file", draft.musicLRCFile] }
            }
            if !draft.musicLRCOutput.isBlank { args += ["--lrc-output", draft.musicLRCOutput] }
            args += [
                "--export-format", draft.musicExportFormat,
                "--normalize", draft.musicNormalization,
                "--target-peak-db", format(draft.musicTargetPeakDB),
                "--fade-in-ms", format(draft.musicFadeInMS),
                "--fade-out-ms", format(draft.musicFadeOutMS)
            ]
            if draft.musicNoDither { args.append("--no-dither") }
            if !draft.musicRecipeOutput.isBlank { args += ["--recipe-output", draft.musicRecipeOutput] }
            if draft.musicNoRecipe { args.append("--no-recipe") }
            if !draft.musicDAWBundle.isBlank { args += ["--daw-bundle", draft.musicDAWBundle] }
            if !draft.musicStems.isBlank { args += ["--stems", draft.musicStems] }
            for path in pathList(draft.musicAdapterPaths) {
                args += ["--adapter", path]
            }
            if !draft.musicAdapterPaths.isBlank {
                args += ["--adapter-kind", draft.musicAdapterKind]
                for scale in pathList(draft.musicAdapterScales) {
                    args += ["--adapter-scale", scale]
                }
            }
            if !draft.musicCheckpointsRoot.isBlank {
                args += ["--checkpoints-root", draft.musicCheckpointsRoot]
            }
            if !draft.musicDecoderSubdirectory.isBlank {
                args += ["--decoder-subdirectory", draft.musicDecoderSubdirectory]
            }
            if !draft.musicVAESubdirectory.isBlank {
                args += ["--vae-subdirectory", draft.musicVAESubdirectory]
            }
            if !draft.musicLMSubdirectory.isBlank {
                args += ["--lm-subdirectory", draft.musicLMSubdirectory]
            }
            if !draft.musicLMModel.isBlank {
                args += ["--lm-model", draft.musicLMModel]
            }
            if !draft.musicTextSubdirectory.isBlank {
                args += ["--text-subdirectory", draft.musicTextSubdirectory]
            }
            if draft.musicLMMode == "use" { args.append("--use-lm") }
            if draft.musicLMMode == "disable" { args.append("--no-lm") }
            if draft.musicAnalyzeSourceAudio { args.append("--analyze-source-audio") }
            if draft.useDuration { args += ["--duration", format(draft.durationSeconds)] }
            args += ["--quality", draft.musicQuality]
            if draft.musicOverrideSteps { args += ["--steps", String(draft.steps)] }
            if !draft.musicShift.isBlank { args += ["--shift", draft.musicShift] }
            if !draft.musicInferMethod.isBlank { args += ["--infer-method", draft.musicInferMethod] }
            if !draft.musicSampler.isBlank { args += ["--sampler", draft.musicSampler] }
            if !draft.musicGuidanceScale.isBlank { args += ["--guidance-scale", draft.musicGuidanceScale] }
            if !draft.musicGuidanceMode.isBlank { args += ["--guidance-mode", draft.musicGuidanceMode] }
            if !draft.musicCFGIntervalStart.isBlank {
                args += ["--cfg-interval-start", draft.musicCFGIntervalStart]
            }
            if !draft.musicCFGIntervalEnd.isBlank {
                args += ["--cfg-interval-end", draft.musicCFGIntervalEnd]
            }
            if !draft.musicVelocityNormThreshold.isBlank {
                args += ["--velocity-norm-threshold", draft.musicVelocityNormThreshold]
            }
            if !draft.musicVelocityEMAFactor.isBlank {
                args += ["--velocity-ema-factor", draft.musicVelocityEMAFactor]
            }
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if draft.musicCandidates > 0 { args += ["--candidates", String(draft.musicCandidates)] }
            if draft.musicKeepCandidates { args.append("--keep-candidates") }
            args += [
                "--audio-cover-strength", format(draft.musicCoverStrength),
                "--cover-noise-strength", format(draft.musicCoverNoiseStrength),
                "--retake-variance", format(draft.musicRetakeVariance),
                "--vocal-language", draft.musicVocalLanguage,
                "--instruction", draft.musicInstruction,
                "--task-type", draft.musicTask
            ]
            if !draft.musicRetakeSeed.isBlank { args += ["--retake-seed", draft.musicRetakeSeed] }
            if !draft.musicSourceAudio.isBlank { args += ["--source-audio", draft.musicSourceAudio] }
            for path in pathList(draft.musicReferenceAudioPaths) {
                args += ["--reference-audio", path]
            }
            if !draft.musicTrackName.isBlank { args += ["--track-name", draft.musicTrackName] }
            if !draft.musicCompleteTrackClasses.isBlank {
                args += ["--complete-track-classes", draft.musicCompleteTrackClasses]
            }
            if draft.musicNonCover { args.append("--non-cover") }
            if ["repaint", "lego"].contains(draft.musicTask) {
                args += [
                    "--repaint-start", format(draft.musicRepaintStart),
                    "--repaint-end", format(draft.musicRepaintEnd),
                    "--chunk-mask-mode", draft.musicChunkMaskMode,
                    "--repaint-mode", draft.musicRepaintMode,
                    "--repaint-strength", format(draft.musicRepaintStrength)
                ]
            }
            if draft.musicFlowEdit {
                args += [
                    "--flow-edit",
                    "--flow-edit-n-min", format(draft.musicFlowEditNMin),
                    "--flow-edit-n-max", format(draft.musicFlowEditNMax),
                    "--flow-edit-n-average", String(draft.musicFlowEditNAverage)
                ]
                if !draft.musicSourceCaption.isBlank {
                    args += ["--source-caption", draft.musicSourceCaption]
                }
                if !draft.musicSourceLyrics.isBlank {
                    args += ["--source-lyrics", draft.musicSourceLyrics]
                }
            }
            if !draft.musicBPM.isBlank { args += ["--bpm", draft.musicBPM] }
            if !draft.musicKey.isBlank { args += ["--keyscale", draft.musicKey] }
            if !draft.musicTimeSignature.isBlank {
                args += ["--timesignature", draft.musicTimeSignature]
            }
            args += [
                "--lm-temperature", format(draft.musicLMTemperature),
                "--lm-top-k", String(draft.musicLMTopK),
                "--lm-top-p", format(draft.musicLMTopP),
                "--lm-repetition-penalty", format(draft.musicLMRepetitionPenalty),
                "--lm-cfg-scale", format(draft.musicLMCFGScale),
                "--lm-negative-prompt", draft.musicLMNegativePrompt
            ]
            if draft.musicInstrumental { args.append("--instrumental") }
            if !draft.musicMetadataDuration.isBlank {
                args += ["--metadata-duration", draft.musicMetadataDuration]
            }
            if !draft.musicMetadataLanguage.isBlank {
                args += ["--metadata-language", draft.musicMetadataLanguage]
            }
            if draft.musicNoTiledVAE { args.append("--no-tiled-vae") }
            args += [
                "--vae-chunk-size", String(draft.musicVAEChunkSize),
                "--vae-overlap", String(draft.musicVAEOverlap)
            ]
            if draft.model.localizedCaseInsensitiveContains("magenta") {
                args += [
                    "--temperature", format(draft.musicTemperature),
                    "--style-conditioning", draft.musicStyleConditioning,
                    "--top-k", String(draft.musicTopK),
                    "--cfg-musiccoca", format(draft.musicCFGMusicCoCa),
                    "--cfg-notes", format(draft.musicCFGNotes),
                    "--cfg-drums", format(draft.musicCFGDrums),
                    "--unmask-width", String(draft.musicUnmaskWidth),
                    "--seed-rotation", String(draft.musicSeedRotation),
                    "--prefill-duration", format(draft.musicPrefillDuration)
                ]
                if draft.musicDrumless { args.append("--drumless") }
                if draft.musicPrefillSilence { args.append("--prefill-silence") }
            }
            if draft.quiet { args.append("--quiet") }

        case .videoGenerate:
            args = ["video", "generate", draft.prompt]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.modelRoot.isBlank { args += ["--model-root", draft.modelRoot] }
            let family = StudioVideoModelFamily(model: draft.modelRoot.isBlank ? draft.model : draft.modelRoot)
            if family == .ltx {
                let quality = draft.audioPath.isBlank ? draft.videoQuality : .final
                let outputMode = draft.audioPath.isBlank ? draft.videoOutputMode : .audioVideo
                args += ["--quality", quality.rawValue, "--output-mode", outputMode.rawValue]
            }
            args += ["--width", String(draft.width), "--height", String(draft.height)]
            if draft.useDuration {
                args += ["--duration", format(draft.durationSeconds)]
            } else {
                args += ["--num-frames", String(draft.numFrames)]
            }
            if !family.isMiniMaxH3 { args += ["--fps", String(draft.fps)] }
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if !family.isMiniMaxH3, !draft.secondaryText.isBlank {
                args += ["--negative-prompt", draft.secondaryText]
            }
            if family == .wan {
                args += [
                    "--steps", String(draft.steps),
                    "--guidance-scale", format(draft.cfgScale),
                    "--shift", format(draft.scheduleShift)
                ]
            }
            if family.isMiniMaxH3 {
                if let h3Steps = draft.h3Steps { args += ["--steps", String(h3Steps)] }
                if let weightMode = draft.h3WeightMode, !weightMode.isBlank {
                    args += ["--h3-weight-mode", weightMode]
                }
                if let accelerationMode = draft.h3AccelerationMode,
                   !accelerationMode.isBlank {
                    args += ["--h3-acceleration", accelerationMode]
                }
                for reference in draft.h3ReferenceInputs ?? [] where !reference.isBlank {
                    args += ["--reference", reference]
                }
            } else if !draft.audioPath.isBlank {
                args += [
                    "--audio", draft.audioPath,
                    "--audio-start-time", format(draft.audioStartTime),
                    "--a2v-guidance-scale", format(draft.a2vGuidanceScale),
                    "--video-cfg-guidance-scale", format(draft.videoCFGGuidanceScale),
                    "--audio-cfg-guidance-scale", format(draft.audioCFGGuidanceScale),
                    "--v2a-guidance-scale", format(draft.v2aGuidanceScale),
                    "--a2v-steps", String(draft.a2vSteps)
                ]
                if let audioMaxDuration = draft.audioMaxDuration, audioMaxDuration > 0 {
                    args += ["--audio-max-duration", format(audioMaxDuration)]
                }
            }
            if !draft.inputPath.isBlank { args += ["--image", draft.inputPath, "--image-strength", format(draft.strength)] }
            if !draft.endImagePath.isBlank {
                args += [
                    "--end-image", draft.endImagePath,
                    "--end-image-strength", format(draft.endImageStrength)
                ]
            }
            if draft.preflight {
                args.append("--preflight")
                if draft.json { args.append("--json") }
            }
            if !family.isMiniMaxH3, draft.timings { args.append("--timings") }
            if !family.isMiniMaxH3, !draft.timingsOutputPath.isBlank {
                args += ["--timings-output", draft.timingsOutputPath]
            }
            if draft.quiet { args.append("--quiet") }

        case .videoRetake:
            args = [
                "video", "retake", draft.prompt,
                "--source", draft.inputPath,
                "--start-time", format(draft.retakeStartTime),
                "--end-time", format(draft.retakeEndTime)
            ]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.modelRoot.isBlank { args += ["--model-root", draft.modelRoot] }
            if !draft.secondaryText.isBlank {
                args += ["--negative-prompt", draft.secondaryText]
            }
            args += ["--steps", String(draft.steps)]
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if draft.retakePreserveVideo { args.append("--preserve-video") }
            if draft.retakePreserveAudio { args.append("--preserve-audio") }
            if draft.quiet { args.append("--quiet") }

        case .videoDubIt:
            args = [
                "video", "dub-it", draft.prompt,
                "--reference-video", draft.inputPath,
                "--ic-lora", draft.loraPath,
                "--ic-lora-strength", format(draft.loraScale),
                "--reference-strength", format(draft.strength),
                "--width", String(draft.width),
                "--height", String(draft.height)
            ]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.modelRoot.isBlank { args += ["--model-root", draft.modelRoot] }
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if draft.quiet { args.append("--quiet") }

        case .videoAnimate:
            args = [
                "video", "animate", draft.prompt,
                "--reference", draft.inputPath,
                "--reference-mask", draft.referenceMaskPath,
                "--driving-video", draft.drivingVideoPath,
                "--driving-mask", draft.drivingMaskPath,
                "--output", draft.outputPath,
                "--mode", draft.videoTaskMode,
                "--profile", draft.renderProfile,
                "--width", String(draft.width),
                "--height", String(draft.height),
                "--fps", String(draft.fps),
                "--segment-length", String(draft.segmentLength),
                "--segment-overlap", String(draft.segmentOverlap),
                "--tail-policy", draft.tailPolicy,
                "--audio-source", draft.audioSource
            ]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.modelRoot.isBlank { args += ["--model-root", draft.modelRoot] }
            let additionalReferences = pathList(draft.referenceImagePaths)
            let additionalMasks = pathList(draft.scailAdditionalReferenceMaskPaths ?? "")
            for (reference, mask) in zip(additionalReferences, additionalMasks) {
                args += ["--additional-reference", reference]
                args += ["--additional-reference-mask", mask]
            }
            if !draft.loraPath.isBlank {
                args += [
                    "--distilled-adapter", draft.loraPath,
                    "--distilled-adapter-strength", format(draft.loraScale)
                ]
            }
            if !draft.secondaryText.isBlank { args += ["--negative-prompt", draft.secondaryText] }
            if draft.renderProfile == "quality" {
                args += [
                    "--steps", String(draft.steps),
                    "--guidance-scale", format(draft.cfgScale),
                    "--shift", format(draft.scheduleShift),
                    "--sampler", draft.sampler
                ]
            }
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if draft.preflight {
                args.append("--preflight")
                if draft.json { args.append("--json") }
            }
            if draft.quiet { args.append("--quiet") }

        case .videoCosmos3:
            args = [
                "video", "cosmos3", draft.prompt,
                "--mode", draft.cosmosMode,
                "--output", draft.outputPath,
                "--width", String(draft.width),
                "--height", String(draft.height),
                "--num-frames", String(draft.numFrames),
                "--schedule", draft.schedule
            ]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.cosmosImagePath.isBlank { args += ["--image", draft.cosmosImagePath] }
            if !draft.cosmosVideoPath.isBlank { args += ["--video", draft.cosmosVideoPath] }
            if !draft.actionsOutputPath.isBlank { args += ["--actions-output", draft.actionsOutputPath] }
            if !draft.secondaryText.isBlank { args += ["--negative-prompt", draft.secondaryText] }
            if draft.steps > 0 { args += ["--steps", String(draft.steps)] }
            if draft.cfgScale > 0 { args += ["--guidance-scale", format(draft.cfgScale)] }
            if draft.scheduleShift > 0 { args += ["--shift", format(draft.scheduleShift)] }
            if draft.fps > 0 { args += ["--fps", String(draft.fps)] }
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if draft.quiet { args.append("--quiet") }

        case .videoPrepareMasks:
            args = [
                "video", "prepare-masks",
                "--plan", draft.inputPath,
                "--output-dir", draft.outputPath
            ]
            if !draft.previewFrame.isBlank { args += ["--preview-frame", draft.previewFrame] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.preflight {
                args.append("--preflight")
                if draft.json { args.append("--json") }
            }
            if draft.quiet { args.append("--quiet") }

        case .videoExportLatents:
            args = ["video", "export-latents", draft.prompt]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--width", String(draft.width), "--height", String(draft.height)]
            args += ["--num-frames", String(draft.numFrames)]
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if draft.quiet { args.append("--quiet") }

        case .videoSession:
            args = ["video", "session"]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.modelRoot.isBlank { args += ["--model-root", draft.modelRoot] }
            if draft.quiet { args.append("--quiet") }

        case .sfxGenerate:
            args = ["sfx", "generate", draft.prompt]
            if !draft.secondaryText.isBlank {
                args += ["--negative-prompt", draft.secondaryText]
            }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--duration", format(draft.durationSeconds), "--steps", String(draft.steps)]
            if draft.cfgScale != 1.0 { args += ["--cfg", format(draft.cfgScale)] }
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if !draft.sfxRenoise.isBlank { args += ["--renoise", draft.sfxRenoise] }
            if draft.quiet { args.append("--quiet") }

        case .sfxVideo:
            args = ["sfx", "video", "generate", draft.prompt, draft.inputPath]
            if !draft.secondaryText.isBlank {
                args += ["--negative-prompt", draft.secondaryText]
            }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--duration", format(draft.durationSeconds), "--steps", String(draft.steps)]
            if draft.cfgScale > 0 { args += ["--cfg", format(draft.cfgScale)] }
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if !draft.sfxRenoise.isBlank { args += ["--renoise", draft.sfxRenoise] }
            if !draft.sfxSynchformerModel.isBlank {
                args += ["--synchformer-model", draft.sfxSynchformerModel]
            }
            args += [
                "--sync-batch-size", String(draft.sfxSyncBatchSize),
                "--clip-batch-size", String(draft.sfxClipBatchSize)
            ]
            if draft.preflight { args.append("--preflight") }
            if draft.preflight, draft.json { args.append("--json") }
            if draft.quiet { args.append("--quiet") }

        case .audioEnhance:
            args = ["audio", "enhance", draft.inputPath]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.modelRoot.isBlank { args += ["--model-path", draft.modelRoot] }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if let overlap = draft.audioOverlap { args += ["--overlap", String(overlap)] }
            if let inputRate = draft.audioInputRate { args += ["--input-rate", String(inputRate)] }
            if let method = draft.audioODEMethod, !method.isBlank {
                args += ["--ode-method", method]
            }
            if let steps = draft.audioODESteps { args += ["--ode-steps", String(steps)] }
            if let guidance = draft.audioGuidanceScale {
                args += ["--guidance-scale", format(guidance)]
            }
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if let seconds = draft.audioChunkSeconds {
                args += ["--chunk-seconds", String(seconds)]
            }
            if let dtype = draft.audioDType, !dtype.isBlank { args += ["--dtype", dtype] }
            if draft.quiet { args.append("--quiet") }

        case .audioGenerate:
            args = ["audio", "generate", draft.prompt]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.modelRoot.isBlank { args += ["--model-root", draft.modelRoot] }
            if !draft.secondaryText.isBlank {
                args += ["--negative-prompt", draft.secondaryText]
            }
            if draft.useDuration {
                args += ["--duration", format(draft.durationSeconds)]
            } else {
                args += ["--num-frames", String(draft.numFrames), "--fps", String(draft.fps)]
            }
            args += ["--steps", String(draft.steps)]
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if draft.quiet { args.append("--quiet") }

        case .musicAnalyze:
            args = ["music", "analyze", draft.inputPath]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.musicCheckpointsRoot.isBlank {
                args += ["--checkpoints-root", draft.musicCheckpointsRoot]
            }
            if !draft.musicDecoderSubdirectory.isBlank {
                args += ["--decoder-subdirectory", draft.musicDecoderSubdirectory]
            }
            if !draft.musicVAESubdirectory.isBlank {
                args += ["--vae-subdirectory", draft.musicVAESubdirectory]
            }
            if !draft.musicLMSubdirectory.isBlank {
                args += ["--lm-subdirectory", draft.musicLMSubdirectory]
            }
            if !draft.musicLMModel.isBlank {
                args += ["--lm-model", draft.musicLMModel]
            }
            if draft.useDuration { args += ["--duration", format(draft.durationSeconds)] }
            args += [
                "--max-new-tokens", String(draft.musicAnalysisMaxTokens),
                "--lm-temperature", format(draft.musicAnalysisTemperature),
                "--lm-top-k", String(draft.musicLMTopK),
                "--lm-top-p", format(draft.musicLMTopP)
            ]
            if draft.musicIncludeRawLM { args.append("--include-raw-lm") }
            if draft.musicIncludeAudioCodes { args.append("--include-audio-codes") }
            if draft.quiet { args.append("--quiet") }

        case .musicTranscribe:
            args = ["music", "transcribe"]
            if !draft.inputPath.isBlank { args.append(draft.inputPath) }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.musicTranscribeModelPath.isBlank {
                args += ["--model-path", draft.musicTranscribeModelPath]
            }
            if !draft.musicTranscribeVariant.isBlank {
                args += ["--variant", draft.musicTranscribeVariant]
            }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            args += ["--format", draft.musicTranscribeFormat]
            if !draft.musicInstruments.isBlank {
                args += ["--instruments", draft.musicInstruments]
            }
            if draft.musicListInstruments { args.append("--list-instruments") }
            if draft.musicSampling { args.append("--sampling") }
            args += [
                "--temperature", format(draft.temperature),
                "--max-tokens-per-chunk", String(draft.musicMaxTokensPerChunk),
                "--beam-size", String(draft.musicBeamSize),
                "--chunk-batch-size", String(draft.musicChunkBatchSize),
                "--dtype", draft.musicDType
            ]
            if draft.musicStrictEOS { args.append("--strict-eos") }
            if draft.musicNoMusicalContext { args.append("--no-musical-context") }
            if !draft.musicContextOutput.isBlank {
                args += ["--context-output", draft.musicContextOutput]
            }
            if draft.quiet { args.append("--quiet") }

        case .musicSeparate:
            args = ["music", "separate", draft.inputPath]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.modelRoot.isBlank { args += ["--model-path", draft.modelRoot] }
            if !draft.outputPath.isBlank { args += ["--output-dir", draft.outputPath] }
            if let overlap = draft.audioOverlap { args += ["--overlap", String(overlap)] }
            if let dtype = draft.audioDType, !dtype.isBlank { args += ["--dtype", dtype] }
            if draft.quiet { args.append("--quiet") }

        case .musicRealtime:
            args = ["music", "realtime"]
            if !draft.prompt.isBlank { args.append(draft.prompt) }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--duration", format(draft.durationSeconds)]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.musicPlay { args.append("--no-play") }
            args += [
                "--style-conditioning", draft.musicStyleConditioning,
                "--temperature", format(draft.musicTemperature),
                "--top-k", String(draft.musicTopK),
                "--cfg-musiccoca", format(draft.musicCFGMusicCoCa),
                "--cfg-notes", format(draft.musicCFGNotes),
                "--cfg-drums", format(draft.musicCFGDrums),
                "--unmask-width", String(draft.musicUnmaskWidth),
                "--seed-rotation", String(draft.musicSeedRotation),
                "--prefill-duration", format(draft.musicPrefillDuration),
                "--midi-channel", draft.musicMIDIChannel,
                "--midi-note-offset", String(draft.musicMIDINoteOffset)
            ]
            if draft.musicDrumless { args.append("--drumless") }
            if draft.musicPrefillSilence { args.append("--prefill-silence") }
            if draft.musicInteractive { args.append("--interactive") }
            if draft.musicListMIDIInputs { args.append("--list-midi-inputs") }
            if draft.musicMIDIMonitor { args.append("--midi-monitor") }
            if draft.musicMIDILogEvents { args.append("--midi-log-events") }
            if draft.musicMIDILogRaw { args.append("--midi-log-raw") }
            if !draft.musicMIDIInput.isBlank { args += ["--midi-input", draft.musicMIDIInput] }
            for mapping in pathList(draft.musicMIDICCMappings) {
                args += ["--midi-cc", mapping]
            }
            if draft.quiet { args.append("--quiet") }

        case .musicTrainAdapter:
            args = [
                "music", "train-adapter",
                "--dataset", draft.inputPath,
                "--output", draft.outputPath,
                "--kind", draft.musicTrainingKind,
                "--rank", String(draft.rank),
                "--alpha", format(draft.alpha),
                "--factor", String(draft.musicTrainingFactor),
                "--steps", String(draft.steps),
                "--learning-rate", format(draft.learningRate),
                "--weight-decay", format(draft.musicTrainingWeightDecay),
                "--seed", draft.seed,
                "--max-duration", format(draft.musicTrainingMaxDuration),
                "--decoder-subdirectory", draft.musicDecoderSubdirectory,
                "--vae-subdirectory", draft.musicVAESubdirectory,
                "--log-every", String(draft.musicTrainingLogEvery)
            ]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.musicCheckpointsRoot.isBlank {
                args += ["--checkpoints-root", draft.musicCheckpointsRoot]
            }
            if !draft.musicTextSubdirectory.isBlank {
                args += ["--text-subdirectory", draft.musicTextSubdirectory]
            }

        case .musicServe:
            args = [
                "music", "serve",
                "--host", draft.host,
                "--port", String(draft.port)
            ]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.musicCheckpointsRoot.isBlank {
                args += ["--checkpoints-root", draft.musicCheckpointsRoot]
            }
            if !draft.musicDecoderSubdirectory.isBlank {
                args += ["--decoder-subdirectory", draft.musicDecoderSubdirectory]
            }
            if !draft.musicVAESubdirectory.isBlank {
                args += ["--vae-subdirectory", draft.musicVAESubdirectory]
            }
            if !draft.musicLMSubdirectory.isBlank {
                args += ["--lm-subdirectory", draft.musicLMSubdirectory]
            }
            if !draft.musicLMModel.isBlank {
                args += ["--lm-model", draft.musicLMModel]
            }
            if !draft.musicTextSubdirectory.isBlank {
                args += ["--text-subdirectory", draft.musicTextSubdirectory]
            }
            for path in pathList(draft.musicAdapterPaths) {
                args += ["--adapter", path]
            }
            if !draft.musicAdapterPaths.isBlank {
                args += ["--adapter-kind", draft.musicAdapterKind]
                for scale in pathList(draft.musicAdapterScales) {
                    args += ["--adapter-scale", scale]
                }
            }

        case .adapterList:
            args = ["adapter", "list"]
            if draft.json { args.append("--json") }

        case .adapterPull:
            args = ["adapter", "pull", draft.prompt]
            if draft.force { args.append("--force") }
            if draft.quiet { args.append("--quiet") }

        case .runList:
            args = ["run", "list"]
            if !draft.operationsRoot.isBlank { args += ["--root", draft.operationsRoot] }
            if !draft.operationsExecutor.isBlank {
                args += ["--executor", draft.operationsExecutor]
                args += ["--limit", String(draft.operationsLimit)]
            } else {
                args += ["--max-depth", String(draft.maxDepth)]
            }
            if draft.json { args.append("--json") }

        case .runInspect:
            args = ["run", "inspect", draft.operationsReference]
            if draft.json { args.append("--json") }

        case .runWatch:
            args = [
                "run", "watch", draft.operationsReference,
                "--poll-interval", format(draft.operationsPollInterval)
            ]
            if draft.operationsJSONStream { args.append("--json-stream") }
            if draft.json { args.append("--json") }

        case .runFetch:
            args = ["run", "fetch", draft.operationsReference, "--into", draft.outputPath]
            if draft.operationsAllArtifacts { args.append("--all-artifacts") }
            for artifact in lineList(draft.operationsArtifacts) {
                args += ["--artifact", artifact]
            }
            if draft.json { args.append("--json") }

        case .runCancel:
            args = ["run", "cancel", draft.operationsReference]
            if draft.json { args.append("--json") }

        case .runRetry:
            args = ["run", "retry", draft.operationsReference]
            if draft.json { args.append("--json") }

        case .evaluationPackValidate:
            args = ["eval", "pack", "validate", draft.inputPath]
            if draft.json { args.append("--json") }

        case .evaluationRun:
            args = ["eval", "run", draft.inputPath]
            for binding in lineList(draft.prompt) {
                args += ["--model", binding]
            }
            for binding in lineList(draft.secondaryText) {
                args += ["--adapter", binding]
            }
            if draft.dryRun { args.append("--dry-run") }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if draft.json { args.append("--json") }

        case .evaluationPromote:
            args = ["eval", "promote", draft.inputPath]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if draft.json { args.append("--json") }

        case .worldServe:
            args = [
                "world", "serve",
                "--host", draft.host,
                "--port", String(draft.port),
                "--backend", draft.operationsWorldBackend,
                "--base-model", draft.operationsBaseModel,
                "--model", draft.model
            ]
            if !draft.operationsStateDirectory.isBlank {
                args += ["--state-directory", draft.operationsStateDirectory]
            }
            if draft.operationsPrepare { args.append("--prepare") }

        case .statusSnapshot:
            args = [
                "status",
                "--host", draft.host,
                "--port", String(draft.port),
                "--timeout-seconds", format(draft.operationsTimeoutSeconds)
            ]
            if draft.json { args.append("--json") }

        case .qualityGate:
            args = ["gate", "--suite", draft.operationsGateSuite]
            if draft.operationsUpdateBaselines { args.append("--update-baselines") }
            if draft.operationsStrictPerformance { args.append("--strict-perf") }
            if !draft.outputPath.isBlank { args += ["--json-output", draft.outputPath] }
            if draft.operationsListOnly { args.append("--list") }

        case .modelStorage:
            args = ["model", "storage"]
            if draft.json { args.append("--json") }

        case .modelGarbageCollect:
            args = ["model", "gc"]
            if draft.force { args.append("--force") }
            if draft.json { args.append("--json") }

        case .modelRuntimeGet:
            args = ["model", "runtime", "get", draft.model]
            if draft.json { args.append("--json") }

        case .modelRuntimeSet:
            args = ["model", "runtime", "set", draft.model]
            if !draft.operationsRuntimeAlias.isBlank {
                args += ["--alias", draft.operationsRuntimeAlias]
            }
            if draft.operationsClearAlias { args.append("--clear-alias") }
            if draft.operationsPinned { args.append("--pinned") }
            if draft.operationsUnpinned { args.append("--unpinned") }
            if !draft.operationsRuntimeTTL.isBlank {
                args += ["--ttl-seconds", draft.operationsRuntimeTTL]
            }
            if draft.operationsClearTTL { args.append("--clear-ttl") }
            if !draft.operationsRuntimeContext.isBlank {
                args += ["--max-context-tokens", draft.operationsRuntimeContext]
            }
            if draft.operationsClearContext { args.append("--clear-max-context-tokens") }
            if !draft.operationsRuntimeMaxTokens.isBlank {
                args += ["--max-tokens", draft.operationsRuntimeMaxTokens]
            }
            if draft.operationsClearMaxTokens { args.append("--clear-max-tokens") }
            if !draft.operationsRuntimeTemperature.isBlank {
                args += ["--temperature", draft.operationsRuntimeTemperature]
            }
            if draft.operationsClearTemperature { args.append("--clear-temperature") }
            if !draft.operationsRuntimeTopP.isBlank {
                args += ["--top-p", draft.operationsRuntimeTopP]
            }
            if draft.operationsClearTopP { args.append("--clear-top-p") }
            if !draft.operationsRuntimeMinP.isBlank {
                args += ["--min-p", draft.operationsRuntimeMinP]
            }
            if draft.operationsClearMinP { args.append("--clear-min-p") }
            if !draft.operationsRuntimeEngine.isBlank {
                args += ["--engine", draft.operationsRuntimeEngine]
            }
            if draft.operationsClearEngine { args.append("--clear-engine") }
            if !draft.operationsRuntimeKVCacheMode.isBlank {
                args += ["--kv-cache-mode", draft.operationsRuntimeKVCacheMode]
            }
            if draft.operationsClearKVCacheMode { args.append("--clear-kv-cache-mode") }
            if draft.json { args.append("--json") }

        case .graphStudio, .nodeConsole:
            return []
        case .sfxAEEncode:
            args = ["sfx", "ae", "encode", draft.inputPath]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.quiet { args.append("--quiet") }

        case .sfxAEDecode:
            args = ["sfx", "ae", "decode", draft.inputPath]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.quiet { args.append("--quiet") }

        case .sfxClapScore:
            args = ["sfx", "clap", "score", draft.prompt, draft.inputPath]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.quiet { args.append("--quiet") }

        case .sfxConditionText:
            args = ["sfx", "condition", "text", draft.prompt]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.quiet { args.append("--quiet") }

        case .modelBenchmark:
            args = ["model", "benchmark", "q36-mtp"]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.modelRoot.isBlank { args += ["--model-root", draft.modelRoot] }
            if !draft.prompt.isBlank { args += ["--prompt", draft.prompt] }
            if !draft.benchmarkPromptFile.isBlank {
                args += ["--prompt-file", draft.benchmarkPromptFile]
            }
            args += [
                "--prompt-repeat", String(draft.benchmarkPromptRepeat),
                "--decode-tokens", String(draft.benchmarkDecodeTokens),
                "--temperature", format(draft.temperature),
                "--top-p", format(draft.topP),
                "--context-size", String(draft.contextSize),
                "--forced-mtp-min-prompt-tokens",
                String(draft.benchmarkForcedMTPMinPromptTokens)
            ]
            if !draft.benchmarkPromptRepeatValues.isBlank {
                args += ["--prompt-repeat-values", draft.benchmarkPromptRepeatValues]
            }
            if !draft.benchmarkDecodeTokenValues.isBlank {
                args += ["--decode-token-values", draft.benchmarkDecodeTokenValues]
            }
            if !draft.benchmarkTemperatureValues.isBlank {
                args += ["--temperature-values", draft.benchmarkTemperatureValues]
            }
            if !draft.benchmarkMTPBlockSize.isBlank {
                args += ["--mtp-block-size", draft.benchmarkMTPBlockSize]
            }
            if draft.json { args.append("--json") }

        case .modelBenchmarkLagunaDFlash:
            args = [
                "model", "benchmark", "laguna-dflash",
                "--laguna-path", draft.modelRoot,
                "--laguna-dflash-path", draft.secondaryText,
                "--decode-token-values", draft.benchmarkDecodeTokenValues,
                "--repetitions", String(draft.benchmarkRepetitions),
                "--laguna-dflash-tokens", String(draft.benchmarkLagunaDFlashTokens),
                "--temperature", format(draft.temperature),
                "--top-p", format(draft.topP),
                "--top-k", String(draft.topK),
                "--min-p", format(draft.minP),
                "--fixture", draft.benchmarkFixture,
                "--context-size", String(draft.contextSize),
                "--warmup-repetitions", String(draft.benchmarkWarmupRepetitions)
            ]
            if !draft.prompt.isBlank { args += ["--prompt", draft.prompt] }
            if !draft.benchmarkPromptFile.isBlank {
                args += ["--prompt-file", draft.benchmarkPromptFile]
            }
            if !draft.benchmarkConcurrencyValues.isBlank {
                args += ["--concurrency-values", draft.benchmarkConcurrencyValues]
            }
            if draft.benchmarkMixedFixtures { args.append("--mixed-fixtures") }
            if draft.benchmarkIncludeAutomatic { args.append("--include-automatic") }
            if draft.benchmarkLogResponses { args.append("--log-responses") }
            if draft.json { args.append("--json") }

        case .pluginList:
            args = ["plugin", "list"]
            if !draft.pluginCatalogURL.isBlank {
                args += ["--catalog-url", draft.pluginCatalogURL]
            }
            if draft.json { args.append("--json") }

        case .pluginInstall:
            args = ["plugin", "install", draft.prompt]
            if !draft.pluginCatalogURL.isBlank {
                args += ["--catalog-url", draft.pluginCatalogURL]
            }
            if !draft.pluginChannel.isBlank { args += ["--channel", draft.pluginChannel] }
            if draft.all { args.append("--yes") }
            if draft.force { args.append("--force") }

        case .pluginDoctor:
            args = ["plugin", "doctor", draft.prompt]
            if !draft.pluginCatalogURL.isBlank {
                args += ["--catalog-url", draft.pluginCatalogURL]
            }

        case .openWebui:
            args = ["open-webui", "quickstart", "--host", draft.host, "--port", String(draft.port)]
            if !draft.engine.isBlank { args += ["--engine", draft.engine] }
            args += [
                "--webui-host", draft.openWebUIHost,
                "--webui-port", String(draft.openWebUIPort),
                "--container-name", draft.openWebUIContainerName,
                "--volume-name", draft.openWebUIVolumeName,
                "--image", draft.openWebUIImage
            ]
            if !draft.model.isBlank { args += ["--text-model", draft.model] }
            args += [
                "--vision-model", draft.openWebUIVisionModel,
                "--embedding-model", draft.openWebUIEmbeddingModel,
                "--image-model", draft.openWebUIImageModel,
                "--tts-model", draft.openWebUITTSModel,
                "--stt-model", draft.openWebUISTTModel,
                "--tts-format", draft.openWebUITTSFormat,
                "--admin-email", draft.openWebUIAdminEmail,
                "--wait-seconds", String(draft.openWebUIWaitSeconds)
            ]
            if draft.openWebUIPull { args.append("--pull") }
            if draft.acceptModelLicense { args.append("--accept-model-license") }
            if draft.openWebUISkipServer { args.append("--skip-server") }
            if draft.openWebUISkipDocker { args.append("--skip-docker") }
            if draft.openWebUISkipConfigure { args.append("--skip-configure") }
            if draft.openWebUIReset { args.append("--reset") }
            if draft.dryRun { args.append("--dry-run") }
            if draft.quiet { args.append("--quiet") }

        case .apiServe:
            args = ["api", "serve", "--host", draft.host, "--port", String(draft.port), "--engine", draft.engine]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.apiLoRA.isBlank { args += ["--lora", draft.apiLoRA] }
            args += [
                "--rate-limit-per-minute", String(draft.apiRateLimitPerMinute),
                "--max-active-requests", String(draft.apiMaxActiveRequests),
                "--memory-guard", draft.apiMemoryGuard,
                "--context-size", String(draft.contextSize)
            ]
            if !draft.apiMemoryGuardCustomCeilingGB.isBlank {
                args += [
                    "--memory-guard-custom-ceiling-gb",
                    draft.apiMemoryGuardCustomCeilingGB
                ]
            }
            if draft.kvBits > 0 { args += ["--kv-bits", String(draft.kvBits)] }
            if !draft.kvQuantScheme.isBlank {
                args += ["--kv-quant-scheme", draft.kvQuantScheme]
            }
            if draft.kvGroupSize > 0 {
                args += ["--kv-group-size", String(draft.kvGroupSize)]
            }
            if draft.quantizedKVStart > 0 {
                args += ["--quantized-kv-start", String(draft.quantizedKVStart)]
            }
            if draft.preflight { args.append("--preflight") }
            if draft.preflight, draft.json { args.append("--json") }

        case .geoFlood, .geoFire, .geoTessera, .geoOlmoEarth:
            let verb: String
            switch id {
            case .geoFire: verb = "fire"
            case .geoTessera: verb = "tessera"
            case .geoOlmoEarth: verb = "olmoearth"
            default: verb = "flood"
            }
            args = ["geo", verb, draft.inputPath]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if id == .geoTessera, !draft.geoDimensions.isBlank {
                args += ["--dimensions", draft.geoDimensions]
            }
            if id == .geoOlmoEarth {
                args += ["--patch-size", String(draft.geoPatchSize)]
                args += ["--input-resolution", String(draft.geoInputResolution)]
                if draft.geoIncludeTokens { args.append("--include-tokens") }
            }
            if draft.preflight { args.append("--preflight") }
            if draft.json { args.append("--json") }

        case .modelLocationList:
            args = ["model", "location", "list"]
            if draft.json { args.append("--json") }

        case .modelLocationAdd:
            args = ["model", "location", "add", draft.inputPath]

        case .modelLocationRemove:
            args = ["model", "location", "remove", draft.inputPath]

        case .modelLocationBind:
            args = ["model", "location", "bind", draft.model, draft.inputPath]
            if draft.acceptModelLicense { args.append("--accept-model-license") }

        case .modelLocationUnbind:
            args = ["model", "location", "unbind", draft.model]
            if !draft.inputPath.isBlank { args.append(draft.inputPath) }

        case .modelBenchmarkChat, .modelBenchmarkToolCalls:
            args = ["model", "benchmark", id == .modelBenchmarkChat ? "chat" : "tool-calls"]
            if !draft.benchmarkModels.isBlank { args += ["--models", draft.benchmarkModels] }
            if id == .modelBenchmarkChat, !draft.benchmarkSuite.isBlank {
                args += ["--suite", draft.benchmarkSuite]
            }
            if !draft.benchmarkCases.isBlank { args += ["--cases", draft.benchmarkCases] }
            if draft.maxTokens > 0 { args += ["--max-tokens", String(draft.maxTokens)] }
            if draft.contextSize > 0 { args += ["--context-size", String(draft.contextSize)] }
            if draft.dryRun { args.append("--dry-run") }
            if draft.benchmarkLogResponses { args.append("--log-responses") }
            if draft.json { args.append("--json") }

        case .modelBenchmarkCode:
            args = ["model", "benchmark", "code"]
            if !draft.benchmarkModels.isBlank { args += ["--models", draft.benchmarkModels] }
            if !draft.benchmarkSuite.isBlank { args += ["--suite", draft.benchmarkSuite] }
            if draft.maxTokens > 0 { args += ["--max-tokens", String(draft.maxTokens)] }
            if !draft.benchmarkSandbox.isBlank { args += ["--sandbox", draft.benchmarkSandbox] }
            if draft.benchmarkAllowCodeExecution { args.append("--allow-code-execution") }
            if draft.dryRun { args.append("--dry-run") }
            if draft.json { args.append("--json") }

        case .modelBenchmarkFused:
            args = ["model", "benchmark", "fused"]
            if !draft.benchmarkSuite.isBlank { args += ["--suite", draft.benchmarkSuite] }
            if !draft.benchmarkModels.isBlank { args += ["--models", draft.benchmarkModels] }
            if !draft.benchmarkCases.isBlank { args += ["--cases", draft.benchmarkCases] }
            if !draft.benchmarkTrials.isBlank { args += ["--trials", draft.benchmarkTrials] }
            if draft.maxTokens > 0 { args += ["--max-tokens", String(draft.maxTokens)] }
            if draft.contextSize > 0 { args += ["--context-size", String(draft.contextSize)] }
            if !draft.benchmarkSandbox.isBlank { args += ["--sandbox", draft.benchmarkSandbox] }
            if draft.benchmarkAllowCodeExecution { args.append("--allow-code-execution") }
            if draft.benchmarkLogResponses { args.append("--log-responses") }
            if draft.benchmarkResume { args.append("--resume") }
            if draft.dryRun { args.append("--dry-run") }
            if draft.json { args.append("--json") }

        case .modelBenchmarkFusedFixture:
            args = ["model", "benchmark", "fused-fixture", draft.inputPath]
            if draft.benchmarkFixtureCheck { args.append("--check") }

        case .modelBenchmarkVLM:
            args = ["model", "benchmark", "vlm"]
            if !draft.benchmarkModels.isBlank { args += ["--models", draft.benchmarkModels] }
            if !draft.benchmarkDataset.isBlank { args += ["--dataset", draft.benchmarkDataset] }
            if !draft.outputPath.isBlank { args += ["--output-dir", draft.outputPath] }
            if draft.maxTokens > 0 { args += ["--max-tokens", String(draft.maxTokens)] }
            if draft.contextSize > 0 { args += ["--context-size", String(draft.contextSize)] }
            if draft.dryRun { args.append("--dry-run") }
            if draft.json { args.append("--json") }

        case .modelBenchmarkToolContinuations:
            args = ["model", "benchmark", "tool-continuations"]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.maxTokens > 0 { args += ["--max-tokens", String(draft.maxTokens)] }
            if draft.contextSize > 0 { args += ["--context-size", String(draft.contextSize)] }
            if draft.dryRun { args.append("--dry-run") }
            if draft.benchmarkLogResponses { args.append("--log-responses") }
            if draft.json { args.append("--json") }

        case .modelBenchmarkGemma4KV, .modelBenchmarkGemma4MTP:
            args = ["model", "benchmark", id == .modelBenchmarkGemma4KV ? "gemma4-kv" : "gemma4-mtp"]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.benchmarkPromptFile.isBlank {
                args += ["--prompt-file", draft.benchmarkPromptFile]
            }
            if draft.benchmarkPromptRepeat > 0 {
                args += ["--prompt-repeat", String(draft.benchmarkPromptRepeat)]
            }
            if draft.benchmarkDecodeTokens > 0 {
                args += ["--decode-tokens", String(draft.benchmarkDecodeTokens)]
            }
            if !draft.benchmarkDecodeTokenValues.isBlank {
                args += ["--decode-token-values", draft.benchmarkDecodeTokenValues]
            }
            if id == .modelBenchmarkGemma4MTP, !draft.benchmarkMTPBlockSize.isBlank {
                args += ["--mtp-block-size", draft.benchmarkMTPBlockSize]
            }
            if draft.json { args.append("--json") }

        case .modelBenchmarkAPIWorkload:
            args = ["model", "benchmark", "api-workload", "--host", draft.host]
            args += ["--port", String(draft.port)]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.maxTokens > 0 { args += ["--max-tokens", String(draft.maxTokens)] }
            if draft.dryRun { args.append("--dry-run") }
            if draft.json { args.append("--json") }

        case .pluginInfo:
            args = ["plugin", "info", draft.prompt]
            if !draft.pluginCatalogURL.isBlank {
                args += ["--catalog-url", draft.pluginCatalogURL]
            }
            if !draft.pluginChannel.isBlank { args += ["--channel", draft.pluginChannel] }
            if draft.json { args.append("--json") }

        case .pluginRun:
            args = ["plugin", "run", draft.prompt]

        case .pluginRollback:
            args = ["plugin", "rollback", draft.prompt]
            if draft.all { args.append("--yes") }

        case .speechListen:
            args = ["speech", "listen"]
            if draft.speechListenListDevices { args.append("--list-devices") }
            if !draft.speechListenDevice.isBlank { args += ["--device", draft.speechListenDevice] }
            if !draft.language.isBlank { args += ["--language", draft.language] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.speechListenDecodeMS > 0 {
                args += ["--decode-ms", String(draft.speechListenDecodeMS)]
            }
            if draft.speechListenSilenceMS > 0 {
                args += ["--silence-ms", String(draft.speechListenSilenceMS)]
            }
            if draft.quiet { args.append("--quiet") }
            if draft.speechJSONL { args.append("--jsonl") }

        case .visionServe:
            args = ["vision", "serve", "--host", draft.host, "--port", String(draft.port)]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.visionServeMaxFrameBytes > 0 {
                args += ["--max-frame-bytes", String(draft.visionServeMaxFrameBytes)]
            }
            if draft.visionServeMaxBatchSize > 0 {
                args += ["--max-batch-size", String(draft.visionServeMaxBatchSize)]
            }
            if draft.visionServeMaxBatchBytes > 0 {
                args += ["--max-batch-bytes", String(draft.visionServeMaxBatchBytes)]
            }
            if draft.preflight { args.append("--preflight") }
            if draft.json { args.append("--json") }

        case .custom:
            return ShellWords.split(draft.extraArguments)
        }

        let extra = ShellWords.split(draft.extraArguments)
        if !extra.isEmpty {
            args.append(contentsOf: extra)
        }
        return args
    }


    private func format(_ value: Double) -> String {
        String(format: "%.4g", value)
    }

    private func appendFaceOptions(to args: inout [String], draft: CommandDraft) {
        if !draft.model.isBlank { args += ["--model", draft.model] }
        args += [
            "--score-threshold", format(draft.visionFaceScoreThreshold),
            "--execution-provider", draft.visionExecutionProvider
        ]
        if !draft.visionJSONOutputPath.isBlank {
            args += ["--json-output", draft.visionJSONOutputPath]
        }
        if draft.json { args.append("--json") }
    }

    private func lineList(_ raw: String) -> [String] {
        raw.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func pathList(_ raw: String) -> [String] {
        raw.components(separatedBy: .newlines)
            .flatMap { $0.split(separator: ",", omittingEmptySubsequences: true) }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
}

extension CommandTemplate {
    /// The primary Studio workspace that owns this command's durable Library entry.
    ///
    /// Advanced exposes specialist commands without adding dozens of modes to the main sidebar.
    /// Mapping them into the nearest creative workspace lets every run use the same queue,
    /// progress, history, and result canvas while `StudioLibraryItem.templateID` preserves the
    /// precise command identity and title.
    var libraryMode: StudioMode {
        switch id {
        case .imageGenerate,
             .imageTrainLoRA,
             .imageValidate,
             .imageDatasetDiscover,
             .imageRunPlan,
             .imageVisualizeRun,
             .imageReconstruct3D,
             .imageReconstruct3DTrellis2,
             .imageReconstruct3DMultiview:
            return .createImage

        case .textChat, .textEmbed, .textAnonymize, .textTrainLoRA:
            return .chat
        case .textCode:
            return .code

        case .speechSynthesize,
             .speechProfileList,
             .speechProfileCreate,
             .speechProfileDelete:
            return .speak
        case .speechTranscribe, .speechDiarize:
            return .listen
        case .speechListen:
            return .listen
        case .audioEnhance, .audioGenerate:
            return .listen

        case .visionGround:
            return .findObjects
        case .visionServe:
            return .findObjects
        case .visionSegment:
            return .segment
        case .visionTrack, .visionTrackLive:
            return .track
        case .visionInspect,
             .visionEmbed,
             .visionCaption,
             .visionOCR,
             .visionFaceDetect,
             .visionFaceEmbed,
             .visionFaceCompare,
             .visionFaceBatch,
             .visionPose,
             .visionFlow,
             .visionDepthVideo,
             .visionGeometry,
             .visionGeometryMultiview,
             .geoFlood,
             .geoFire,
             .geoTessera,
             .geoOlmoEarth:
            return .readImage

        case .musicGenerate,
             .musicAnalyze,
             .musicTranscribe,
             .musicSeparate,
             .musicRealtime,
             .musicTrainAdapter,
             .musicServe:
            return .music

        case .videoGenerate,
             .videoRetake,
             .videoDubIt,
             .videoAnimate,
             .videoCosmos3,
             .videoPrepareMasks,
             .videoExportLatents,
             .videoSession,
             .worldServe:
            return .video

        case .sfxGenerate,
             .sfxVideo,
             .sfxAEEncode,
             .sfxAEDecode,
             .sfxClapScore,
             .sfxConditionText:
            return .sfx

        case .setup,
             .agentOnboard,
             .agentStatus,
             .agentInstallPi,
             .agentStart,
             .modelList,
             .modelCapabilities,
             .modelPull,
             .modelInfo,
             .modelRemove,
             .modelRepairManifests,
             .modelOptimize,
             .adapterList,
             .adapterPull,
             .runList,
             .runInspect,
             .runWatch,
             .runFetch,
             .runCancel,
             .runRetry,
             .evaluationPackValidate,
             .evaluationRun,
             .evaluationPromote,
             .statusSnapshot,
             .qualityGate,
             .modelStorage,
             .modelGarbageCollect,
             .modelRuntimeGet,
             .modelRuntimeSet,
             .modelLocationList,
             .modelLocationAdd,
             .modelLocationRemove,
             .modelLocationBind,
             .modelLocationUnbind,
             .graphStudio,
             .nodeConsole,
             .modelBenchmark,
             .modelBenchmarkLagunaDFlash,
             .modelBenchmarkChat,
             .modelBenchmarkCode,
             .modelBenchmarkFused,
             .modelBenchmarkFusedFixture,
             .modelBenchmarkVLM,
             .modelBenchmarkToolCalls,
             .modelBenchmarkToolContinuations,
             .modelBenchmarkGemma4KV,
             .modelBenchmarkGemma4MTP,
             .modelBenchmarkAPIWorkload,
             .pluginList,
             .pluginInstall,
             .pluginDoctor,
             .pluginInfo,
             .pluginRun,
             .pluginRollback,
             .openWebui,
             .apiServe,
             .custom:
            return .chat
        }
    }
}

enum CommandLaunchEnvironment {
    static let apiKeyEnvironmentKey = "MERERUN_API_KEY"
    static let openWebUIAdminPasswordEnvironmentKey = "MERERUN_OPEN_WEBUI_ADMIN_PASSWORD"

    static func overrides(templateID: CommandTemplateID, draft: CommandDraft) -> [String: String] {
        guard templateID == .apiServe
            || templateID == .openWebui
            || templateID == .musicServe
            || templateID == .worldServe
            || templateID == .visionServe
            || templateID == .statusSnapshot else { return [:] }
        var overrides: [String: String] = [:]
        let apiKey = draft.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if !apiKey.isEmpty {
            overrides[apiKeyEnvironmentKey] = apiKey
        }
        if templateID == .openWebui {
            let adminPassword = draft.openWebUIAdminPassword
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !adminPassword.isEmpty {
                overrides[openWebUIAdminPasswordEnvironmentKey] = adminPassword
            }
        }
        return overrides
    }
}

enum CommandCatalog {
    static let templates: [CommandTemplate] = [
        CommandTemplate(
            id: .setup,
            category: .setup,
            title: "Setup path",
            subtitle: "Plan or run guided, BYOA, or manual setup",
            systemImage: "wand.and.stars"
        ),
        CommandTemplate(
            id: .agentOnboard,
            category: .setup,
            title: "Agent onboarding",
            subtitle: "Check local readiness and prepare Pi integration",
            systemImage: "person.crop.circle.badge.gearshape",
            defaultModel: StudioCodeDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .agentStatus,
            category: .setup,
            title: "Agent status",
            subtitle: "Inspect Pi, provider, and local agent readiness",
            systemImage: "person.crop.circle.badge.checkmark"
        ),
        CommandTemplate(
            id: .agentInstallPi,
            category: .setup,
            title: "Install Pi",
            subtitle: "Install or replace the optional setup agent",
            systemImage: "square.and.arrow.down"
        ),
        CommandTemplate(
            id: .agentStart,
            category: .setup,
            title: "Start agent",
            subtitle: "Launch a guided setup session",
            systemImage: "terminal.fill",
            promptLabel: "Prompt",
            defaultPrompt: "Guide me through setting up mere.run on this machine. Start by summarizing what this Mac can run, then help me install only supported models."
        ),
        CommandTemplate(id: .modelList, category: .models, title: "List models", subtitle: "Installed and missing managed models", systemImage: "list.bullet.rectangle"),
        CommandTemplate(id: .modelCapabilities, category: .models, title: "Capabilities", subtitle: "Hardware support and recommended pulls", systemImage: "memorychip"),
        CommandTemplate(id: .modelPull, category: .models, title: "Pull model", subtitle: "Download a managed model", systemImage: "arrow.down.circle", defaultModel: "image-zimage-nano"),
        CommandTemplate(id: .modelInfo, category: .models, title: "Model info", subtitle: "Manifest and validation report", systemImage: "info.circle", defaultModel: "image-zimage-nano"),
        CommandTemplate(id: .modelRemove, category: .models, title: "Remove model", subtitle: "Delete a local model install", systemImage: "trash", defaultModel: "image-zimage-nano"),
        CommandTemplate(id: .modelRepairManifests, category: .models, title: "Repair manifests", subtitle: "Write missing known model manifests", systemImage: "wrench.and.screwdriver"),
        CommandTemplate(
            id: .modelOptimize,
            category: .models,
            title: "Optimize MiniMax-H3",
            subtitle: "Build or replace the inference-only AdaLN cache",
            systemImage: "bolt.badge.clock",
            defaultModel: "video-minimax-h3-fl2va-mlx"
        ),
        CommandTemplate(
            id: .imageGenerate,
            category: .image,
            title: "Generate or edit",
            subtitle: "Text, image, multi-reference, structured prompt, and LoRA",
            systemImage: "photo",
            promptLabel: "Prompt",
            secondaryLabel: "Negative prompt",
            inputKind: .image,
            outputKind: .file("png"),
            defaultPrompt: "a ceramic coffee mug in soft morning light",
            defaultModel: "image-zimage-nano"
        ),
        CommandTemplate(
            id: .imageTrainLoRA,
            category: .image,
            title: "Train LoRA",
            subtitle: "Krea 2 and FLUX.2 Klein recipes, previews, and dashboards",
            systemImage: "slider.horizontal.3",
            inputKind: .directory,
            outputKind: .file("safetensors"),
            defaultModel: "image-krea2-raw"
        ),
        CommandTemplate(
            id: .imageValidate,
            category: .image,
            title: "Validate image stack",
            subtitle: "Run deterministic image runtime checks",
            systemImage: "checkmark.seal",
            outputKind: .directory
        ),
        CommandTemplate(
            id: .imageDatasetDiscover,
            category: .image,
            title: "Discover datasets",
            subtitle: "Find trainable image-caption folders",
            systemImage: "folder.badge.questionmark",
            inputKind: .directory
        ),
        CommandTemplate(
            id: .imageRunPlan,
            category: .image,
            title: "Run workflow plan",
            subtitle: "Preflight, materialize, or execute a saved image plan",
            systemImage: "list.bullet.clipboard",
            inputKind: .file([.json])
        ),
        CommandTemplate(
            id: .imageVisualizeRun,
            category: .image,
            title: "Training dashboard",
            subtitle: "Open a durable LoRA run viewer",
            systemImage: "chart.xyaxis.line",
            inputKind: .directory
        ),
        CommandTemplate(
            id: .imageReconstruct3D,
            category: .image,
            title: "TripoSR 3D",
            subtitle: "Reconstruct a colored mesh from one image",
            systemImage: "cube.transparent",
            inputKind: .image,
            outputKind: .directory,
            defaultModel: "image-3d-triposr"
        ),
        CommandTemplate(
            id: .imageReconstruct3DTrellis2,
            category: .image,
            title: "TRELLIS.2 PBR 3D",
            subtitle: "Build a 512-resolution PBR O-Voxel asset",
            systemImage: "cube.fill",
            inputKind: .image,
            outputKind: .directory,
            defaultModel: "image-3d-trellis2-4b"
        ),
        CommandTemplate(
            id: .imageReconstruct3DMultiview,
            category: .image,
            title: "InstantMesh multiview",
            subtitle: "Reconstruct from four or six ordered views",
            systemImage: "square.3.layers.3d",
            outputKind: .directory,
            defaultModel: "image-3d-instantmesh-base"
        ),
        CommandTemplate(
            id: .textChat,
            category: .text,
            title: "Chat",
            subtitle: "Local chat models",
            systemImage: "bubble.left.and.bubble.right",
            promptLabel: "Prompt",
            secondaryLabel: "System",
            defaultPrompt: "Summarize diffusion models in one paragraph.",
            defaultModel: StudioChatDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .textCode,
            category: .text,
            title: "Code",
            subtitle: "Local code generation",
            systemImage: "chevron.left.forwardslash.chevron.right",
            promptLabel: "Prompt",
            secondaryLabel: "System",
            defaultPrompt: "Write a tiny Swift function that formats byte counts.",
            defaultModel: StudioCodeDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .textEmbed,
            category: .text,
            title: "Embeddings",
            subtitle: "Generate JSON embedding vectors",
            systemImage: "point.3.connected.trianglepath.dotted",
            promptLabel: "Text",
            outputKind: .file("json"),
            defaultPrompt: "semantic search query",
            defaultModel: "text-embed-qwen3-0.6b"
        ),
        CommandTemplate(
            id: .textAnonymize,
            category: .text,
            title: "Anonymize",
            subtitle: "Detect and redact PII",
            systemImage: "eye.slash",
            promptLabel: "Text",
            outputKind: .file("txt"),
            defaultPrompt: "My name is Alice Smith and my email is alice@example.com",
            defaultModel: "text-anonymize-privacy-filter"
        ),
        CommandTemplate(
            id: .textTrainLoRA,
            category: .text,
            title: "Train text LoRA",
            subtitle: "Fine-tune from chat SFT JSONL",
            systemImage: "text.badge.plus",
            inputKind: .file([.json, .plainText]),
            outputKind: .file("safetensors"),
            defaultModel: "text-chat-gemma4-12b-4bit"
        ),
        CommandTemplate(
            id: .speechSynthesize,
            category: .speech,
            title: "Synthesize",
            subtitle: "Text to speech",
            systemImage: "waveform",
            promptLabel: "Text",
            secondaryLabel: "Voice",
            outputKind: .file("wav"),
            defaultPrompt: "Hello from mere.run.",
            defaultSecondaryText: "A calm female voice with clear pronunciation",
            defaultModel: "speech-tts-qwen3-nano"
        ),
        CommandTemplate(
            id: .speechTranscribe,
            category: .speech,
            title: "Transcribe",
            subtitle: "Speech to text",
            systemImage: "captions.bubble",
            inputKind: .audio,
            outputKind: .file("txt")
        ),
        CommandTemplate(
            id: .speechDiarize,
            category: .speech,
            title: "Diarize speakers",
            subtitle: "Identify who spoke when",
            systemImage: "person.2.fill",
            inputKind: .audio,
            outputKind: .file("json"),
            defaultModel: "speech-diarization-sortformer"
        ),
        CommandTemplate(id: .speechProfileList, category: .speech, title: "Voice profiles", subtitle: "List saved clone profiles", systemImage: "person.wave.2"),
        CommandTemplate(
            id: .speechProfileCreate,
            category: .speech,
            title: "Create voice profile",
            subtitle: "Save reference audio for clone mode",
            systemImage: "person.badge.plus",
            promptLabel: "Profile name",
            secondaryLabel: "Transcript override",
            inputKind: .audio,
            defaultPrompt: "Narration profile"
        ),
        CommandTemplate(
            id: .speechProfileDelete,
            category: .speech,
            title: "Delete voice profile",
            subtitle: "Remove a saved clone profile",
            systemImage: "person.badge.minus",
            promptLabel: "Profile UUID"
        ),
        CommandTemplate(
            id: .visionInspect,
            category: .vision,
            title: "Inspect image",
            subtitle: "Ask a VLM about an image",
            systemImage: "eye",
            promptLabel: "Question",
            inputKind: .image,
            defaultPrompt: "Describe this image."
        ),
        CommandTemplate(
            id: .visionEmbed,
            category: .vision,
            title: "Embed text + image",
            subtitle: "Create shared retrieval vectors",
            systemImage: "point.3.connected.trianglepath.dotted",
            promptLabel: "Text (optional)",
            secondaryLabel: "Retrieval instruction",
            inputKind: .image,
            outputKind: .file("json"),
            defaultSecondaryText: "Represent the item for retrieval.",
            defaultModel: "vision-embed-qwen3-vl-2b"
        ),
        CommandTemplate(
            id: .visionCaption,
            category: .vision,
            title: "Caption",
            subtitle: "Write LoRA-friendly captions",
            systemImage: "text.bubble",
            promptLabel: "Instruction",
            inputKind: .image,
            outputKind: .directory,
            defaultPrompt: "Write a short, concrete caption describing the image for LoRA training. Avoid fluff."
        ),
        CommandTemplate(
            id: .visionOCR,
            category: .vision,
            title: "OCR",
            subtitle: "Extract text from images",
            systemImage: "doc.text.viewfinder",
            inputKind: .image,
            outputKind: .directory,
            defaultModel: "vision-ocr-lighton"
        ),
        CommandTemplate(
            id: .visionGround,
            category: .vision,
            title: "Ground objects",
            subtitle: "Find prompted objects in an image",
            systemImage: "scope",
            promptLabel: "Query",
            inputKind: .image,
            outputKind: .file("png"),
            defaultPrompt: "a person",
            defaultModel: "vision-ground-falcon-perception"
        ),
        CommandTemplate(
            id: .visionSegment,
            category: .vision,
            title: "Segment",
            subtitle: "Segment prompted objects",
            systemImage: "square.dashed",
            promptLabel: "Prompt",
            inputKind: .image,
            outputKind: .file("png"),
            defaultPrompt: "a person",
            defaultModel: "vision-segment-sam31"
        ),
        CommandTemplate(
            id: .visionTrack,
            category: .vision,
            title: "Track video",
            subtitle: "Track prompted objects through a clip",
            systemImage: "point.topleft.down.curvedto.point.bottomright.up",
            promptLabel: "Prompt",
            inputKind: .video,
            outputKind: .file("mp4"),
            defaultPrompt: "a person",
            defaultModel: "vision-segment-sam31"
        ),
        CommandTemplate(
            id: .visionTrackLive,
            category: .vision,
            title: "Track camera",
            subtitle: "Capture then track from a camera",
            systemImage: "video.badge.waveform",
            promptLabel: "Prompt",
            outputKind: .file("mp4"),
            defaultPrompt: "a person",
            defaultModel: "vision-segment-sam31"
        ),
        CommandTemplate(
            id: .visionFaceDetect,
            category: .vision,
            title: "Detect faces",
            subtitle: "Buffalo-L boxes, landmarks, and optional embeddings",
            systemImage: "face.dashed",
            inputKind: .image,
            defaultModel: "vision-face-buffalo-l"
        ),
        CommandTemplate(
            id: .visionFaceEmbed,
            category: .vision,
            title: "Embed face",
            subtitle: "Create a normalized ArcFace identity vector",
            systemImage: "person.crop.square",
            inputKind: .image,
            defaultModel: "vision-face-buffalo-l"
        ),
        CommandTemplate(
            id: .visionFaceCompare,
            category: .vision,
            title: "Compare faces",
            subtitle: "Cosine similarity between two selected faces",
            systemImage: "person.2",
            inputKind: .image,
            defaultModel: "vision-face-buffalo-l"
        ),
        CommandTemplate(
            id: .visionFaceBatch,
            category: .vision,
            title: "Batch face analysis",
            subtitle: "Warm-session detection and embeddings to JSONL",
            systemImage: "person.3.sequence",
            inputKind: .image,
            defaultModel: "vision-face-buffalo-l"
        ),
        CommandTemplate(
            id: .visionPose,
            category: .vision,
            title: "Pose landmarks",
            subtitle: "Native body, hand, and face landmarks",
            systemImage: "figure.stand",
            inputKind: .image
        ),
        CommandTemplate(
            id: .visionFlow,
            category: .vision,
            title: "Optical flow",
            subtitle: "Dense motion between two equal-size images",
            systemImage: "arrow.triangle.2.circlepath",
            inputKind: .image,
            outputKind: .file("flo")
        ),
        CommandTemplate(
            id: .visionDepthVideo,
            category: .vision,
            title: "Video depth",
            subtitle: "Temporally consistent native VDA-S depth",
            systemImage: "square.3.layers.3d",
            inputKind: .video,
            outputKind: .directory,
            defaultModel: "vision-depth-vda-small"
        ),
        CommandTemplate(
            id: .visionGeometry,
            category: .vision,
            title: "Metric geometry",
            subtitle: "MoGe-2 depth, normals, camera, and point cloud",
            systemImage: "rotate.3d",
            inputKind: .image,
            outputKind: .directory,
            defaultModel: "vision-geometry-moge2-small"
        ),
        CommandTemplate(
            id: .visionGeometryMultiview,
            category: .vision,
            title: "Multi-view geometry",
            subtitle: "DA3 cameras, confidence, and colored point cloud",
            systemImage: "view.3d",
            inputKind: .image,
            outputKind: .directory,
            defaultModel: "vision-geometry-da3-small"
        ),
        CommandTemplate(
            id: .audioEnhance,
            category: .media,
            title: "Enhance audio",
            subtitle: "AP-BWE speech extension or UniverSR restoration",
            systemImage: "waveform.badge.plus",
            inputKind: .audio,
            outputKind: .file("wav"),
            defaultModel: "audio-enhance-ap-bwe-16kto48k"
        ),
        CommandTemplate(
            id: .audioGenerate,
            category: .media,
            title: "Generate audio",
            subtitle: "Native LTX-2.5 text-to-audio generation",
            systemImage: "waveform.badge.sparkles",
            promptLabel: "Audio prompt",
            secondaryLabel: "Negative prompt",
            outputKind: .file("wav"),
            defaultPrompt: "a quiet forest at dawn with distant birds",
            defaultModel: "video-ltx25-full-bf16"
        ),
        CommandTemplate(
            id: .musicGenerate,
            category: .media,
            title: "Generate music",
            subtitle: "ACE-Step or Magenta RT2 music generation",
            systemImage: "music.note",
            promptLabel: "Caption",
            secondaryLabel: "Lyrics",
            outputKind: .file("wav"),
            defaultPrompt: "upbeat electronic groove",
            defaultModel: "music-acestep"
        ),
        CommandTemplate(
            id: .videoGenerate,
            category: .media,
            title: "Generate video",
            subtitle: "Full-power LTX-2.5, Wan, or synchronized MiniMax-H3 generation",
            systemImage: "film",
            promptLabel: "Prompt",
            secondaryLabel: "Negative prompt",
            inputKind: .image,
            outputKind: .file("mp4"),
            defaultPrompt: "a cinematic drone flythrough over snowy mountains",
            defaultModel: "video-ltx25-full-bf16"
        ),
        CommandTemplate(
            id: .videoRetake,
            category: .media,
            title: "Retake video",
            subtitle: "Regenerate a timed LTX-2.5 video or audio region",
            systemImage: "timeline.selection",
            promptLabel: "Replacement prompt",
            secondaryLabel: "Negative prompt",
            inputKind: .video,
            outputKind: .file("mp4"),
            defaultPrompt: "continue the performance with natural synchronized motion",
            defaultModel: "video-ltx25-distilled-bf16"
        ),
        CommandTemplate(
            id: .videoDubIt,
            category: .media,
            title: "Dub-It",
            subtitle: "Transfer synchronized video and audio identity with LTX-2.5 IC-LoRA",
            systemImage: "person.wave.2",
            promptLabel: "Scene prompt",
            inputKind: .video,
            outputKind: .file("mp4"),
            defaultPrompt: "the speaker performs on a rain-lit street",
            defaultModel: "video-ltx25-distilled-bf16"
        ),
        CommandTemplate(
            id: .videoAnimate,
            category: .media,
            title: "Animate subject",
            subtitle: "SCAIL-2 animation and replacement",
            systemImage: "figure.walk.motion",
            promptLabel: "Prompt",
            secondaryLabel: "Negative prompt",
            inputKind: .image,
            outputKind: .file("mp4"),
            defaultPrompt: "a dancer in a red silk dress",
            defaultModel: "video-scail2-14b-mlx"
        ),
        CommandTemplate(
            id: .videoCosmos3,
            category: .media,
            title: "Cosmos3",
            subtitle: "Generation, dynamics, policy, and reasoning",
            systemImage: "sparkles.tv",
            promptLabel: "Prompt or action task",
            secondaryLabel: "Negative prompt",
            outputKind: .file("mp4"),
            defaultPrompt: "a cinematic rover crossing a windswept alien plain",
            defaultModel: "video-cosmos3-edge-mlx"
        ),
        CommandTemplate(
            id: .videoPrepareMasks,
            category: .media,
            title: "Prepare SCAIL-2 masks",
            subtitle: "SAM 3.1 mask-plan preparation",
            systemImage: "square.stack.3d.up",
            inputKind: .file([.json]),
            outputKind: .directory,
            defaultModel: "vision-segment-sam31"
        ),
        CommandTemplate(
            id: .videoExportLatents,
            category: .media,
            title: "Export video latents",
            subtitle: "Write native LTX final latents",
            systemImage: "shippingbox",
            promptLabel: "Prompt",
            outputKind: .file("safetensors"),
            defaultPrompt: "a cinematic drone flythrough over snowy mountains",
            defaultModel: "video-ltx-av"
        ),
        CommandTemplate(
            id: .videoSession,
            category: .media,
            title: "Resident LTX session",
            subtitle: "Keep LTX 2.3 warm for JSONL requests",
            systemImage: "bolt.horizontal.circle",
            defaultModel: "video-ltx23-full-mlx"
        ),
        CommandTemplate(
            id: .adapterList,
            category: .operations,
            title: "Browse adapters",
            subtitle: "Verified LoRA catalog and install state",
            systemImage: "square.stack.3d.up"
        ),
        CommandTemplate(
            id: .adapterPull,
            category: .operations,
            title: "Pull adapter",
            subtitle: "Download and verify a cataloged LoRA",
            systemImage: "arrow.down.circle",
            promptLabel: "Adapter ID",
            defaultPrompt: "mere-platform-assistant"
        ),
        CommandTemplate(
            id: .runList,
            category: .operations,
            title: "Browse runs",
            subtitle: "Find local reports or remote jobs",
            systemImage: "clock.arrow.circlepath"
        ),
        CommandTemplate(
            id: .runInspect,
            category: .operations,
            title: "Inspect run",
            subtitle: "Read a durable run, report, plan, or remote job",
            systemImage: "doc.text.magnifyingglass"
        ),
        CommandTemplate(
            id: .runWatch,
            category: .operations,
            title: "Watch remote run",
            subtitle: "Stream SSH or Relay worker events",
            systemImage: "dot.radiowaves.left.and.right"
        ),
        CommandTemplate(
            id: .runFetch,
            category: .operations,
            title: "Fetch remote run",
            subtitle: "Verify and materialize remote artifacts locally",
            systemImage: "square.and.arrow.down",
            outputKind: .directory
        ),
        CommandTemplate(
            id: .runCancel,
            category: .operations,
            title: "Cancel run",
            subtitle: "Request local or remote cancellation",
            systemImage: "stop.circle"
        ),
        CommandTemplate(
            id: .runRetry,
            category: .operations,
            title: "Retry Relay run",
            subtitle: "Retry the same immutable job bundle",
            systemImage: "arrow.clockwise.circle"
        ),
        CommandTemplate(
            id: .evaluationPackValidate,
            category: .operations,
            title: "Validate evaluation pack",
            subtitle: "Verify and content-hash an external pack",
            systemImage: "checkmark.seal",
            inputKind: .directory
        ),
        CommandTemplate(
            id: .evaluationRun,
            category: .operations,
            title: "Run evaluation pack",
            subtitle: "Plan or run matched model, prompt, and adapter arms",
            systemImage: "chart.bar.doc.horizontal",
            promptLabel: "Model bindings (one slot=id per line)",
            secondaryLabel: "Adapter bindings (one slot=reference per line)",
            inputKind: .directory,
            outputKind: .file("json")
        ),
        CommandTemplate(
            id: .evaluationPromote,
            category: .operations,
            title: "Promote evaluation report",
            subtitle: "Issue a receipt for a complete gate-passing report",
            systemImage: "checkmark.shield",
            inputKind: .file([.json]),
            outputKind: .file("json")
        ),
        CommandTemplate(
            id: .worldServe,
            category: .operations,
            title: "World session",
            subtitle: "Serve a warm DreamX or Cosmos3 world",
            systemImage: "globe.americas.fill",
            defaultModel: "video-dreamx-world-5b-ar-mlx"
        ),
        CommandTemplate(
            id: .statusSnapshot,
            category: .operations,
            title: "Status snapshot",
            subtitle: "Server, loaded models, and local inventory",
            systemImage: "waveform.path.ecg"
        ),
        CommandTemplate(
            id: .qualityGate,
            category: .operations,
            title: "Quality gate",
            subtitle: "Installed-model correctness and performance",
            systemImage: "checkmark.shield",
            outputKind: .file("json")
        ),
        CommandTemplate(
            id: .modelStorage,
            category: .operations,
            title: "Model storage",
            subtitle: "Physical storage, sharing, and reclaimable bytes",
            systemImage: "internaldrive"
        ),
        CommandTemplate(
            id: .modelGarbageCollect,
            category: .operations,
            title: "Storage cleanup",
            subtitle: "Dry-run or execute safe garbage collection",
            systemImage: "trash.slash"
        ),
        CommandTemplate(
            id: .modelRuntimeGet,
            category: .operations,
            title: "Read runtime policy",
            subtitle: "Inspect API residency and generation defaults",
            systemImage: "gearshape.2",
            defaultModel: StudioChatDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .modelRuntimeSet,
            category: .operations,
            title: "Set runtime policy",
            subtitle: "Pin, expire, alias, and tune resident models",
            systemImage: "slider.horizontal.3",
            defaultModel: StudioChatDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .graphStudio,
            category: .operations,
            title: "Open Graph Studio",
            subtitle: "Author and execute portable Graph v2 workflows",
            systemImage: "point.3.connected.trianglepath.dotted",
            externalURL: URL(string: "https://studio.mere.run/app")
        ),
        CommandTemplate(
            id: .nodeConsole,
            category: .operations,
            title: "Manage Nodes & Relay",
            subtitle: "Pair GPUs, schedule fleet work, and inspect nodes",
            systemImage: "server.rack",
            externalURL: URL(string: "https://relay.mere.run")
        ),
        CommandTemplate(
            id: .sfxGenerate,
            category: .sfx,
            title: "Generate sound effect",
            subtitle: "Woosh or MMAudio text-to-audio",
            systemImage: "speaker.wave.2",
            promptLabel: "Prompt",
            secondaryLabel: "Negative prompt",
            outputKind: .file("wav"),
            defaultPrompt: "a heavy wooden door creaking open",
            defaultModel: "sfx-woosh-dflow"
        ),
        CommandTemplate(
            id: .sfxVideo,
            category: .sfx,
            title: "Video foley",
            subtitle: "Generate sound effects from a video",
            systemImage: "video.badge.waveform",
            promptLabel: "Prompt",
            secondaryLabel: "Negative prompt",
            inputKind: .video,
            outputKind: .file("wav"),
            defaultPrompt: "footsteps on gravel",
            defaultModel: "sfx-woosh-dvflow-8s"
        ),
        CommandTemplate(
            id: .musicAnalyze,
            category: .media,
            title: "Analyze music",
            subtitle: "ACE-Step audio understanding (JSON)",
            systemImage: "waveform.badge.magnifyingglass",
            inputKind: .audio,
            defaultModel: "music-acestep"
        ),
        CommandTemplate(
            id: .musicTranscribe,
            category: .media,
            title: "Transcribe to MIDI",
            subtitle: "MuScriptor full-mix audio to instrument tracks",
            systemImage: "pianokeys",
            inputKind: .audio,
            outputKind: .file("mid"),
            defaultModel: "music-muscriptor-medium"
        ),
        CommandTemplate(
            id: .musicSeparate,
            category: .media,
            title: "Separate or restore",
            subtitle: "RoFormer stems, dereverb, and denoise",
            systemImage: "slider.horizontal.3",
            inputKind: .audio,
            outputKind: .directory,
            defaultModel: "music-separate-bs-roformer-viperx-1297"
        ),
        CommandTemplate(
            id: .musicRealtime,
            category: .media,
            title: "Realtime music",
            subtitle: "Magenta RT2 playback, capture, and MIDI steering",
            systemImage: "dot.radiowaves.left.and.right",
            promptLabel: "Prompt",
            outputKind: .file("wav"),
            defaultPrompt: "warm ambient pads with a slow build",
            defaultModel: "music-magenta-rt2-small"
        ),
        CommandTemplate(
            id: .musicTrainAdapter,
            category: .media,
            title: "Train music adapter",
            subtitle: "Native ACE-Step LoRA or LoKr training",
            systemImage: "tuningfork",
            inputKind: .file([.json]),
            outputKind: .file("safetensors"),
            defaultModel: "music-acestep"
        ),
        CommandTemplate(
            id: .musicServe,
            category: .media,
            title: "Resident music API",
            subtitle: "Keep ACE-Step, LM, and adapters warm",
            systemImage: "server.rack",
            defaultModel: "music-acestep"
        ),
        CommandTemplate(
            id: .sfxAEEncode,
            category: .sfx,
            title: "Autoencoder · encode",
            subtitle: "Audio → Woosh latents (.npy)",
            systemImage: "arrow.down.doc",
            inputKind: .audio,
            outputKind: .file("npy"),
            defaultModel: "sfx-woosh-dflow"
        ),
        CommandTemplate(
            id: .sfxAEDecode,
            category: .sfx,
            title: "Autoencoder · decode",
            subtitle: "Woosh latents (.npy) → audio",
            systemImage: "arrow.up.doc",
            inputKind: .file([.data]),
            outputKind: .file("wav"),
            defaultModel: "sfx-woosh-dflow"
        ),
        CommandTemplate(
            id: .sfxClapScore,
            category: .sfx,
            title: "CLAP score",
            subtitle: "Score audio against a prompt (JSON)",
            systemImage: "checkmark.seal",
            promptLabel: "Prompt",
            inputKind: .audio,
            defaultPrompt: "a heavy wooden door creaking open",
            defaultModel: "sfx-woosh-clap"
        ),
        CommandTemplate(
            id: .sfxConditionText,
            category: .sfx,
            title: "Conditioning · text",
            subtitle: "Export Woosh conditioning tensors",
            systemImage: "function",
            promptLabel: "Prompt",
            outputKind: .file("safetensors"),
            defaultPrompt: "a heavy wooden door creaking open",
            defaultModel: "sfx-woosh-dflow"
        ),
        CommandTemplate(
            id: .modelBenchmark,
            category: .models,
            title: "Qwen3.6 benchmark",
            subtitle: "Focused Qwen3.6 MTP benchmark",
            systemImage: "speedometer",
            defaultModel: "text-chat-q36-nano"
        ),
        CommandTemplate(
            id: .modelBenchmarkLagunaDFlash,
            category: .models,
            title: "Laguna benchmark",
            subtitle: "Target-only, fixed DFlash, and adaptive routing",
            systemImage: "bolt.horizontal.circle"
        ),
        CommandTemplate(
            id: .pluginList,
            category: .server,
            title: "Plugins",
            subtitle: "List official companion plugins",
            systemImage: "puzzlepiece.extension"
        ),
        CommandTemplate(
            id: .pluginInstall,
            category: .server,
            title: "Install plugin",
            subtitle: "Install an official plugin by id",
            systemImage: "square.and.arrow.down",
            promptLabel: "Plugin id",
            defaultPrompt: "mere-runpod"
        ),
        CommandTemplate(
            id: .pluginDoctor,
            category: .server,
            title: "Plugin doctor",
            subtitle: "Run an installed plugin's doctor",
            systemImage: "stethoscope",
            promptLabel: "Plugin id",
            defaultPrompt: "mere-runpod"
        ),
        CommandTemplate(
            id: .openWebui,
            category: .server,
            title: "Open WebUI",
            subtitle: "Start the Open WebUI companion",
            systemImage: "globe",
            defaultModel: StudioChatDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .apiServe,
            category: .server,
            title: "API server",
            subtitle: "OpenAI-compatible local server",
            systemImage: "network",
            defaultModel: StudioChatDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .geoFlood,
            category: .geospatial,
            title: "Flood inference",
            subtitle: "TerraMind flood logits from a normalized tile batch",
            systemImage: "water.waves",
            inputKind: .file([.data]),
            outputKind: .file("safetensors")
        ),
        CommandTemplate(
            id: .geoFire,
            category: .geospatial,
            title: "Fire inference",
            subtitle: "TerraMind fire logits from a normalized tile batch",
            systemImage: "flame",
            inputKind: .file([.data]),
            outputKind: .file("safetensors")
        ),
        CommandTemplate(
            id: .geoTessera,
            category: .geospatial,
            title: "TESSERA embeddings",
            subtitle: "Encode Sentinel-1/2 time series with TESSERA v2",
            systemImage: "square.stack.3d.down.right",
            inputKind: .file([.data]),
            outputKind: .file("safetensors")
        ),
        CommandTemplate(
            id: .geoOlmoEarth,
            category: .geospatial,
            title: "OlmoEarth embeddings",
            subtitle: "Encode multisensor Earth observations with OlmoEarth v1.2",
            systemImage: "globe.europe.africa",
            inputKind: .file([.data]),
            outputKind: .file("safetensors")
        ),
        CommandTemplate(
            id: .modelLocationList,
            category: .models,
            title: "Model locations",
            subtitle: "Writable store, search roots, and explicit bindings",
            systemImage: "externaldrive.badge.checkmark"
        ),
        CommandTemplate(
            id: .modelLocationAdd,
            category: .models,
            title: "Add search root",
            subtitle: "Register a read-only root of canonical model directories",
            systemImage: "externaldrive.badge.plus",
            inputKind: .directory
        ),
        CommandTemplate(
            id: .modelLocationRemove,
            category: .models,
            title: "Remove search root",
            subtitle: "Unregister a search root without deleting files",
            systemImage: "externaldrive.badge.minus",
            inputKind: .directory
        ),
        CommandTemplate(
            id: .modelLocationBind,
            category: .models,
            title: "Bind model directory",
            subtitle: "Point a canonical model id at a read-only directory",
            systemImage: "link.badge.plus",
            inputKind: .directory
        ),
        CommandTemplate(
            id: .modelLocationUnbind,
            category: .models,
            title: "Unbind model directory",
            subtitle: "Remove explicit bindings without deleting files",
            systemImage: "link.circle"
        ),
        CommandTemplate(
            id: .modelBenchmarkFused,
            category: .models,
            title: "Fused quality suite",
            subtitle: "Mere Lite or Mere Comprehensive versioned suite",
            systemImage: "chart.bar.doc.horizontal",
            outputKind: .file("json")
        ),
        CommandTemplate(
            id: .modelBenchmarkChat,
            category: .models,
            title: "Chat benchmark",
            subtitle: "Grounded-chat evaluation slice",
            systemImage: "bubble.left.and.text.bubble.right",
            outputKind: .file("json")
        ),
        CommandTemplate(
            id: .modelBenchmarkCode,
            category: .models,
            title: "Code benchmark",
            subtitle: "Real coding-evaluation slice with sandboxed execution",
            systemImage: "chevron.left.forwardslash.chevron.right",
            outputKind: .file("json")
        ),
        CommandTemplate(
            id: .modelBenchmarkVLM,
            category: .models,
            title: "Vision-language benchmark",
            subtitle: "Synthetic or lmms-eval multimodal datasets",
            systemImage: "photo.badge.checkmark",
            outputKind: .file("json")
        ),
        CommandTemplate(
            id: .modelBenchmarkToolCalls,
            category: .models,
            title: "Tool-call benchmark",
            subtitle: "Tool selection accuracy across chat models",
            systemImage: "wrench.and.screwdriver",
            outputKind: .file("json")
        ),
        CommandTemplate(
            id: .modelBenchmarkToolContinuations,
            category: .models,
            title: "Tool continuation benchmark",
            subtitle: "Gemma 4 continuation after completed tool calls",
            systemImage: "arrow.turn.down.right",
            outputKind: .file("json")
        ),
        CommandTemplate(
            id: .modelBenchmarkGemma4KV,
            category: .models,
            title: "Gemma4 KV benchmark",
            subtitle: "Default KV cache decode against packed PolarKV",
            systemImage: "memorychip"
        ),
        CommandTemplate(
            id: .modelBenchmarkGemma4MTP,
            category: .models,
            title: "Gemma4 MTP benchmark",
            subtitle: "Serial decode against verified MTP speculative decode",
            systemImage: "bolt.badge.clock"
        ),
        CommandTemplate(
            id: .modelBenchmarkAPIWorkload,
            category: .models,
            title: "API workload benchmark",
            subtitle: "Replay a chat workload against a running API server",
            systemImage: "server.rack",
            outputKind: .file("json")
        ),
        CommandTemplate(
            id: .modelBenchmarkFusedFixture,
            category: .models,
            title: "Fused fixture hashes",
            subtitle: "Stamp or verify normalized fixture JSONL hashes",
            systemImage: "number.square",
            inputKind: .file([.json, .plainText])
        ),
        CommandTemplate(
            id: .pluginInfo,
            category: .server,
            title: "Plugin details",
            subtitle: "Catalog entry and install command for one plugin",
            systemImage: "info.circle",
            promptLabel: "Plugin id"
        ),
        CommandTemplate(
            id: .pluginRun,
            category: .server,
            title: "Run plugin",
            subtitle: "Run an installed plugin without changing PATH",
            systemImage: "play.rectangle",
            promptLabel: "Plugin entrypoint"
        ),
        CommandTemplate(
            id: .pluginRollback,
            category: .server,
            title: "Roll back plugin",
            subtitle: "Restore a retained signed plugin bundle",
            systemImage: "arrow.uturn.backward.circle",
            promptLabel: "Plugin id"
        ),
        CommandTemplate(
            id: .speechListen,
            category: .speech,
            title: "Live transcription",
            subtitle: "Stream microphone audio through live Qwen ASR",
            systemImage: "waveform.badge.mic"
        ),
        CommandTemplate(
            id: .visionServe,
            category: .server,
            title: "Vision grounding server",
            subtitle: "Resident binary-frame grounding over HTTP",
            systemImage: "viewfinder.circle"
        ),
        CommandTemplate(
            id: .custom,
            category: .custom,
            title: "Raw arguments",
            subtitle: "Run any mere.run command",
            systemImage: "terminal",
            defaultExtraArguments: "--help"
        ),
    ]

    static func templates(in category: CommandCategory) -> [CommandTemplate] {
        templates.filter { $0.category == category }
    }

    static func template(id: CommandTemplateID) -> CommandTemplate? {
        templates.first { $0.id == id }
    }
}

enum ShellWords {
    static func split(_ raw: String) -> [String] {
        var words: [String] = []
        var current = ""
        var quote: Character?
        var escaping = false

        for char in raw {
            if escaping {
                current.append(char)
                escaping = false
                continue
            }

            if char == "\\" {
                escaping = true
                continue
            }

            if let activeQuote = quote {
                if char == activeQuote {
                    quote = nil
                } else {
                    current.append(char)
                }
                continue
            }

            if char == "\"" || char == "'" {
                quote = char
                continue
            }

            if char.isWhitespace {
                if !current.isEmpty {
                    words.append(current)
                    current.removeAll(keepingCapacity: true)
                }
                continue
            }

            current.append(char)
        }

        if escaping {
            current.append("\\")
        }
        if !current.isEmpty {
            words.append(current)
        }
        return words
    }
}

extension String {
    var isBlank: Bool {
        trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

extension DateFormatter {
    static let mereRunTimestamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter
    }()
}
