import Foundation
import MereRunContract
import UniformTypeIdentifiers

package enum StudioProductBoundary {
    package static let dioramaURL = URL(string: "https://diorama.mere.run")!
}

package enum CommandCategory: String, CaseIterable, Identifiable {
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

    package var id: String { rawValue }
}

package enum CommandTemplateID: String, CaseIterable, Codable {
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
    case textDecide
    case textTrainLoRA
    case speechSynthesize
    case speechTranscribe
    case speechDiarize
    case speechDiarizeLive
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
    case visionDepth
    case visionDepthVideo
    case visionGeometry
    case visionGeometryMultiview
    case audioEnhance
    case audioEdit
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
    case modelBenchmarkParakeetCoreML
    case modelBenchmarkAPIWorkload
    case pluginInfo
    case pluginRun
    case pluginRollback
    case speechListen
    case visionServe
    case custom

    package var capabilityID: String? {
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
        case .textDecide: return "text.decide"
        case .textTrainLoRA: return "text.train-lora"
        case .speechSynthesize: return "speech.synthesize"
        case .speechTranscribe: return "speech.transcribe"
        case .speechDiarize: return "speech.diarize"
        case .speechDiarizeLive: return "speech.diarize-live"
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
        case .visionDepth: return "vision.depth"
        case .visionDepthVideo: return "vision.depth-video"
        case .visionGeometry: return "vision.geometry"
        case .visionGeometryMultiview: return "vision.geometry-multiview"
        case .audioEnhance: return "audio.enhance"
        case .audioEdit: return "audio.edit"
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
        case .modelBenchmarkParakeetCoreML: return "model.benchmark.parakeet-coreml"
        case .modelBenchmarkAPIWorkload: return "model.benchmark.api-workload"
        case .pluginInfo: return "plugin.info"
        case .pluginRun: return "plugin.run"
        case .pluginRollback: return "plugin.rollback"
        case .speechListen: return "speech.listen"
        case .visionServe: return "vision.serve"
        }
    }

    /// The capability's contract entry, when the command is one the shared contract declares.
    package var capability: MereRunCommandCapability? {
        capabilityID.flatMap { MereRunCapabilityCatalog.command(id: $0) }
    }

    /// True when the CLI prints the `{"event":"result"}` receipt line for this command under
    /// `--receipt`. The contract owns the list, so a capability that gains the flag needs no
    /// change here.
    package var emitsRunReceipt: Bool {
        guard let capabilityID else { return false }
        return MereRunCapabilityCatalog.receiptCapabilityIDs.contains(capabilityID)
    }

    /// True when the CLI streams `--progress-json` events for this command.
    package var emitsProgressJSON: Bool {
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
package enum StudioMachineOutputFlags {
    package static let receipt = "--receipt"
    package static let progressJSON = "--progress-json"

    /// The flags to append to `arguments` for one run, in a stable order. Empty when the
    /// capability declares neither, when the run is a preflight or a dry run (neither produces
    /// a result, and the CLI rejects `--receipt` with either), or when the app already passes
    /// the flag from the draft.
    package static func arguments(
        template: CommandTemplate,
        draft: CommandDraft,
        appendingTo arguments: [String]
    ) -> [String] {
        guard !draft.preflight, !arguments.contains("--preflight"), !arguments.contains("--dry-run") else {
            return []
        }
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

package enum CommandInputKind: Equatable {
    case none
    case file([UTType])
    case directory
    case image
    case audio
    case video

    package var title: String {
        switch self {
        case .none: return "Input"
        case .file: return "File"
        case .directory: return "Directory"
        case .image: return "Image"
        case .audio: return "Audio"
        case .video: return "Video"
        }
    }

    package var allowedTypes: [UTType] {
        switch self {
        case .none, .directory: return []
        case .file(let types): return types
        case .image: return [.image]
        case .audio: return [.audio]
        case .video: return [.movie, .video, .audiovisualContent]
        }
    }
}

package enum CommandOutputKind: Equatable {
    case none
    case file(String)
    case directory

    package var isFile: Bool {
        if case .file = self { return true }
        return false
    }
}

package enum StudioChatDefaults {
    package static let fallbackModelID = "text-chat-gemma4-12b-4bit"
    package static let fallbackServingEngine = "text-chat-gemma4"

    package static func servingEngine(for modelID: String) -> String {
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

    package static func shouldReplaceModelDefault(_ modelID: String, oldRecommendation: String? = nil) -> Bool {
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

    package static func shouldReplaceServingEngineDefault(_ engine: String, oldRecommendation: String? = nil) -> Bool {
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

package enum StudioCodeDefaults {
    package static let fallbackModelID = "text-code-north-mini"

    package static func shouldReplaceModelDefault(_ modelID: String, oldRecommendation: String? = nil) -> Bool {
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

package struct CommandDraft: Equatable, Codable {
    /// Every stored property carries a default, so this is the synthesized memberwise
    /// initializer's package-visible stand-in: the app builds drafts with `CommandDraft()`
    /// and assigns the fields a template needs.
    package init() {}

    package var prompt = ""
    package var secondaryText = ""
    package var model = ""
    package var inputPath = ""
    /// Optional so Library rows written before mask editing still decode through synthesized Codable.
    package var imageMaskPath: String?
    package var imageOutpaint: String?
    package var imageMaskFeather: Int?
    package var outputPath = ""
    package var width = 1024
    package var height = 1024
    package var steps = 4
    package var seed = ""
    package var cfgScale = 1.0
    package var strength = 0.75
    package var sigmaShift = 0.0
    package var referenceImagePaths = ""
    package var keepOriginalAspect = false
    package var structuredPrompt = false
    package var structuredPromptModel = "text-chat-gemma4-12b-4bit"
    package var structuredPromptModelRoot = ""
    package var structuredPromptMaxTokens = 2048
    package var structuredPromptOutputPath = ""
    package var kreaConditioningMultiplier = 0.0
    package var kreaConditioningLayerWeights = ""
    package var kreaBaseQuantizationBits = ""
    package var progressJSON = false
    package var maxTokens = 2048
    package var contextSize = 0
    package var temperature = 0.7
    package var topP = 0.9
    package var topK = 0
    package var minP = 0.0
    package var kvBits = 0
    package var kvQuantScheme = ""
    package var kvGroupSize = 0
    package var quantizedKVStart = 0
    package var responseFormat: TextResponseFormat = .text
    package var thinkingMode: TextThinkingMode = .automatic
    /// Optional preserves Library rows written before Inkling reasoning effort was exposed.
    package var reasoningEffort: Double?
    package var loraPath = ""
    package var loraScale = 1.0
    package var replacement = "[{label}]"
    package var evalPath = ""
    package var adapterName = "local-assistant"
    package var batchSize = 1
    package var learningRate = 0.0001
    package var rank = 16
    package var alpha = 0.0
    package var maxSequenceLength = 4096
    package var targetModules = ""
    package var dryRun = false
    package var visualize = false
    package var visualizePort = 8787
    package var schedulerSteps = 1_000
    package var captionDropout = 0.0
    package var trainingLite = false
    package var baseQuantizationBits = ""
    package var excludePreviewImages = false
    package var checkpointInterval = 0
    /// Optional preserves Library decoding for rows written before checkpoint resume was exposed.
    package var trainingResumePath: String?
    package var maxResolution = 0
    package var progressive = false
    package var lowRAM = false
    package var disableCompile = false
    package var gradientCheckpointing = false
    package var trainingRecipe = ""
    package var overrideTrainingRecipe = false
    package var benchmarkSteps = 0
    package var benchmarkWarmupSteps = 5
    package var sampleInterval = 0
    package var samplePrompt = ""
    package var sampleModel = ""
    package var sampleSteps = 8
    package var sampleCFG = 1.0
    package var sampleLoRAScale = 1.0
    package var sampleSeed = ""
    package var loraTargetRanks = ""
    package var loraRankPreset = ""
    package var loraTargetPreset = ""
    package var loraTargetMode = ""
    package var timestepSampling = ""
    package var timestepLossWeighting = ""
    package var lossWeighting = ""
    package var timestepLow = 0
    package var timestepHigh = 0
    package var lrWarmupSteps = 0
    package var disableCosineScheduler = false
    package var lrMinFactor = 0.0
    package var adamWeightDecay = 0.0
    package var syntheticSamples = 0
    package var maxDepth = 4
    package var minUsablePairs = 1
    package var trainingOutputRoot = ""
    package var trainingModel = ""
    package var materializePath = ""
    package var referenceDirectoryPath = ""
    package var reconstructionResolution = 256
    package var densityThreshold = 25.0
    package var foregroundRatio = 0.85
    package var alreadyFramed = false
    package var noVertexColors = false
    package var camerasPath = ""
    /// Optional so Library rows written before the first-class TRELLIS.2 workspace keep decoding.
    package var trellisTextureSeed: String?
    package var trellisNoRemesh: Bool?
    package var trellisRemeshBand: Double?
    package var trellisSealRadius: Int?
    package var durationSeconds = 10.0
    // Music production workspace. Empty strings intentionally mean "let the CLI quality
    // preset choose" for optional numeric overrides.
    package var musicLyricsFile = ""
    package var musicLRCFile = ""
    package var musicLRCOutput = ""
    package var musicExportFormat = "pcm24"
    package var musicNormalization = "peak"
    package var musicTargetPeakDB = -1.0
    package var musicFadeInMS = 5.0
    package var musicFadeOutMS = 20.0
    package var musicNoDither = false
    package var musicRecipeOutput = ""
    package var musicNoRecipe = false
    package var musicDAWBundle = ""
    package var musicStems = ""
    package var musicAdapterPaths = ""
    package var musicAdapterKind = "auto"
    package var musicAdapterScales = ""
    package var musicCheckpointsRoot = ""
    package var musicDecoderSubdirectory = "acestep-v15-turbo"
    package var musicVAESubdirectory = "vae"
    package var musicLMModel = ""
    package var musicLMSubdirectory = ""
    package var musicTextSubdirectory = ""
    package var musicLMMode = "auto"
    package var musicAnalyzeSourceAudio = false
    package var musicQuality = "song"
    package var musicOverrideSteps = false
    package var musicShift = ""
    package var musicInferMethod = ""
    package var musicSampler = ""
    package var musicGuidanceScale = ""
    package var musicGuidanceMode = ""
    package var musicCFGIntervalStart = ""
    package var musicCFGIntervalEnd = ""
    package var musicVelocityNormThreshold = ""
    package var musicVelocityEMAFactor = ""
    package var musicCandidates = 0
    package var musicKeepCandidates = false
    package var musicCoverStrength = 1.0
    package var musicCoverNoiseStrength = 0.0
    package var musicRetakeSeed = ""
    package var musicRetakeVariance = 0.0
    package var musicVocalLanguage = "en"
    package var musicInstruction = "Fill the audio semantic mask based on the given conditions:"
    package var musicTask = "text2music"
    package var musicSourceAudio = ""
    package var musicReferenceAudioPaths = ""
    package var musicTrackName = ""
    package var musicCompleteTrackClasses = ""
    package var musicNonCover = false
    package var musicRepaintStart = 0.0
    package var musicRepaintEnd = -1.0
    package var musicChunkMaskMode = "auto"
    package var musicRepaintMode = "balanced"
    package var musicRepaintStrength = 0.5
    package var musicFlowEdit = false
    package var musicSourceCaption = ""
    package var musicSourceLyrics = ""
    package var musicFlowEditNMin = 0.0
    package var musicFlowEditNMax = 1.0
    package var musicFlowEditNAverage = 1
    package var musicBPM = ""
    package var musicKey = ""
    package var musicTimeSignature = ""
    package var musicLMTemperature = 0.85
    package var musicLMTopK = 0
    package var musicLMTopP = 0.9
    package var musicLMRepetitionPenalty = 1.0
    package var musicLMCFGScale = 2.0
    package var musicLMNegativePrompt = "NO USER INPUT"
    package var musicInstrumental = false
    package var musicMetadataDuration = ""
    package var musicMetadataLanguage = ""
    package var musicNoTiledVAE = false
    package var musicVAEChunkSize = 512
    package var musicVAEOverlap = 64
    package var musicStyleConditioning = "streaming"
    package var musicTemperature = 1.0
    package var musicTopK = 100
    package var musicCFGMusicCoCa = 3.0
    package var musicCFGNotes = 5.0
    package var musicCFGDrums = 1.0
    package var musicDrumless = false
    package var musicUnmaskWidth = 0
    package var musicSeedRotation = 0
    package var musicPrefillSilence = false
    package var musicPrefillDuration = 1.64
    package var musicIncludeRawLM = false
    package var musicIncludeAudioCodes = false
    package var musicAnalysisMaxTokens = 2048
    package var musicAnalysisTemperature = 0.3
    package var musicTranscribeModelPath = ""
    package var musicTranscribeVariant = ""
    package var musicTranscribeFormat = "midi"
    package var musicInstruments = ""
    package var musicListInstruments = false
    package var musicSampling = false
    package var musicMaxTokensPerChunk = 2_000
    package var musicStrictEOS = false
    package var musicBeamSize = 1
    package var musicChunkBatchSize = 4
    package var musicDType = "bfloat16"
    package var musicNoMusicalContext = false
    package var musicContextOutput = ""
    package var musicPlay = true
    package var musicInteractive = false
    package var musicListMIDIInputs = false
    package var musicMIDIMonitor = false
    package var musicMIDILogEvents = false
    package var musicMIDILogRaw = false
    package var musicMIDIInput = ""
    package var musicMIDIChannel = "all"
    package var musicMIDINoteOffset = 0
    package var musicMIDICCMappings = ""
    package var musicTrainingKind = "lora"
    package var musicTrainingFactor = -1
    package var musicTrainingWeightDecay = 0.0001
    package var musicTrainingMaxDuration = 30.0
    package var musicTrainingLogEvery = 10
    /// Optional fields preserve older Library rows while owning the post-0.31 audio tool wave.
    package var audioOverlap: Int?
    package var audioInputRate: Int?
    package var audioODEMethod: String?
    package var audioODESteps: Int?
    package var audioGuidanceScale: Double?
    package var audioChunkSeconds: Int?
    package var audioDType: String?
    // Vision and VFX analysis controls.
    package var visionAdditionalInputs = ""
    package var visionSecondInputPath = ""
    package var visionPromptFile = ""
    package var visionFocus = ""
    package var visionTriggerToken = ""
    package var visionJSONOutputPath = ""
    package var visionMaskOutputDirectory = ""
    package var visionBoxPrompts = ""
    package var visionPointPrompts = ""
    package var visionThreshold = 0.05
    package var visionResolution = 1008
    package var visionMultimask = false
    package var visionInitFrame = 0
    package var visionEndFrame = ""
    package var visionShowLabels = false
    package var visionCamera = 0
    package var visionSeedSearchFrames = 30
    package var visionGLMOCRCLI = "glmocr"
    package var visionGLMConfig = ""
    package var visionInfinityRuntime = "native"
    package var visionInfinityParserCLI = "parser"
    package var visionInfinityModel = "vision-ocr-infinity-pro-int8"
    package var visionInfinityBackend = "vllm-server"
    package var visionInfinityAPIURL = "http://localhost:8000/v1/chat/completions"
    package var visionInfinityAPIKey = "EMPTY"
    package var visionInfinityTask = "doc2json"
    package var visionInfinityPrompt = ""
    package var visionInfinityOutputFormat = "md"
    package var visionInfinityBatchSize = 1
    package var visionInfinityModelCacheDirectory = ""
    package var visionInfinityMinPixels = 2_048
    package var visionInfinityMaxPixels = 16_777_216
    package var visionPoseBody = true
    package var visionPoseHands = true
    package var visionPoseFace = true
    package var visionMaxHands = 2
    package var visionMinimumConfidence = 0.1
    package var visionFlowAccuracy = "high"
    package var visionInputSize = 518
    package var visionMaxFrames = 240
    package var visionResolutionLevel = 9
    package var visionTokenCount = 0
    package var visionMaxPoints = 0
    // Image depth. Optional preserves Library rows written before the task exposed these;
    // nil, 0, and blank leave the CLI's own defaults in place.
    package var visionMaxEdge: Int?
    package var visionNative: Bool?
    package var visionCheckpoint: String?
    package var visionProcessResolution = 504
    package var visionReferenceView = "saddle-balanced"
    package var visionConfidencePercentile = 40.0
    package var visionFaceScoreThreshold = 0.65
    package var visionExecutionProvider = "auto"
    package var visionMaxFaces = 0
    package var visionIncludeEmbeddings = false
    package var visionFaceIndex = ""
    package var visionReferenceFaceIndex = ""
    package var visionCandidateFaceIndex = ""
    package var visionInputList = ""
    package var visionJSONLOutput = ""
    package var visionFailFast = false
    // Operations, durable workflow runs, runtime diagnostics, and world sessions.
    package var operationsReference = ""
    package var operationsRoot = ""
    package var operationsExecutor = ""
    package var operationsArtifacts = ""
    package var operationsLimit = 50
    package var operationsPollInterval = 2.0
    package var operationsTimeoutSeconds = 1.0
    package var operationsAllArtifacts = false
    package var operationsJSONStream = false
    package var operationsGateSuite = "all"
    package var operationsUpdateBaselines = false
    package var operationsStrictPerformance = false
    package var operationsListOnly = false
    package var operationsWorldBackend = "dreamx"
    package var operationsBaseModel = "video-wan22-ti2v-5b-mlx"
    package var operationsStateDirectory = ""
    package var operationsPrepare = false
    package var operationsRuntimeAlias = ""
    package var operationsRuntimeTTL = ""
    package var operationsRuntimeContext = ""
    package var operationsRuntimeMaxTokens = ""
    package var operationsRuntimeTemperature = ""
    package var operationsRuntimeTopP = ""
    package var operationsRuntimeMinP = ""
    package var operationsRuntimeEngine = ""
    package var operationsRuntimeKVCacheMode = ""
    package var operationsClearAlias = false
    package var operationsPinned = false
    package var operationsUnpinned = false
    package var operationsClearTTL = false
    package var operationsClearContext = false
    package var operationsClearMaxTokens = false
    package var operationsClearTemperature = false
    package var operationsClearTopP = false
    package var operationsClearMinP = false
    package var operationsClearEngine = false
    package var operationsClearKVCacheMode = false
    package var fps = 24
    package var numFrames = 65
    package var useDuration = false
    package var host = "127.0.0.1"
    package var port = 8080
    package var apiKey = ""
    package var engine = StudioChatDefaults.fallbackServingEngine
    package var videoQuality: LTXVideoQuality = .final
    package var videoOutputMode: LTXVideoOutputMode = .videoOnly
    package var audioPath = ""
    package var audioStartTime = 0.0
    /// Optional preserves saved Library rows from before source-audio max-duration parity.
    package var audioMaxDuration: Double?
    package var endImagePath = ""
    package var endImageStrength = 1.0
    package var scheduleShift = 5.0
    package var a2vGuidanceScale = 3.0
    package var videoCFGGuidanceScale = 3.0
    package var audioCFGGuidanceScale = 7.0
    package var v2aGuidanceScale = 3.0
    package var a2vSteps = 30
    package var retakeStartTime = 0.0
    package var retakeEndTime = 4.0
    package var retakePreserveVideo = false
    package var retakePreserveAudio = false
    package var preflight = false
    package var timings = false
    package var timingsOutputPath = ""
    /// Optional preserves older Library rows before MiniMax-H3 became a Studio-native workflow.
    package var h3WeightMode: String?
    /// Quality is exact; balanced and maximum enable bounded approximate block reuse.
    package var h3AccelerationMode: String?
    /// Nil keeps MiniMax-H3's geometry-aware 9/16/21-point adaptive schedule.
    package var h3Steps: Int?
    /// Ordered `image:path`, `video:path`, and `audio:path` reference specifications.
    package var h3ReferenceInputs: [String]?
    package var modelRoot = ""
    package var referenceMaskPath = ""
    package var drivingVideoPath = ""
    package var drivingMaskPath = ""
    /// One mask path per newline-delimited additional SCAIL reference image.
    /// Optional preserves synthesized Codable compatibility with older Library rows.
    package var scailAdditionalReferenceMaskPaths: String?
    package var videoTaskMode = "animation"
    package var renderProfile = "fast"
    package var sampler = "unipc"
    package var segmentLength = 81
    package var segmentOverlap = 5
    package var tailPolicy = "drop"
    package var audioSource = "none"
    package var cosmosMode = "text-to-video"
    package var cosmosImagePath = ""
    package var cosmosVideoPath = ""
    package var actionsOutputPath = ""
    package var schedule = "nvidia"
    package var previewFrame = ""
    package var backend = "auto"
    package var task = "transcribe"
    package var language = "auto"
    package var timestamps = true
    package var voiceMode = "style"
    package var voiceProfile = ""
    package var refAudioPath = ""
    package var refText = ""
    package var saveProfileName = ""
    package var speechStreamChunkTokens = 25
    package var speechStreamChunkMS = 200
    package var speechStreamDecodeMS = 2_000
    package var speechInputFormat = ""
    package var speechSampleRate = 16_000
    package var speechJSONL = false
    /// Optional fields preserve decoding of Library rows created before native Sortformer controls.
    package var speechDiarizationFormat: String?
    package var speechDiarizationLatency: String?
    package var speechDiarizationLiveStdin = false
    package var speechDiarizationThreshold: Double?
    package var speechDiarizationMinDuration: Double?
    package var speechDiarizationMergeGap: Double?
    // Vision chat (vision-capable chat models) and the agentic tool loop for `text chat`.
    package var imagePath = ""
    package var tools = ""
    package var toolLoop = false
    package var allowShellExec = false
    package var allowAbsoluteToolPaths = false
    package var autoApproveTools = false
    package var requireInstalled = false
    package var sandboxDir = ""
    package var setupMode = "agent"
    package var agentModel = "tier"
    package var piPath = ""
    package var noBootstrap = false
    package var modelKeepCache = false
    package var modelRemovalJSON = false
    package var benchmarkPromptFile = ""
    package var benchmarkPromptRepeat = 150
    package var benchmarkPromptRepeatValues = ""
    package var benchmarkDecodeTokens = 32
    package var benchmarkDecodeTokenValues = ""
    package var benchmarkTemperatureValues = ""
    package var benchmarkMTPBlockSize = ""
    package var benchmarkForcedMTPMinPromptTokens = 1
    package var benchmarkRepetitions = 3
    package var benchmarkLagunaDFlashTokens = 12
    package var benchmarkFixture = "deterministic-prose"
    package var benchmarkConcurrencyValues = ""
    package var benchmarkWarmupRepetitions = 1
    package var benchmarkMixedFixtures = false
    package var benchmarkIncludeAutomatic = false
    package var benchmarkLogResponses = false
    package var sfxRenoise = ""
    package var sfxSynchformerModel = "sfx-woosh-synchformer"
    package var sfxSyncBatchSize = 1
    package var sfxClipBatchSize = 4
    package var pluginCatalogURL = ""
    package var pluginChannel = ""
    package var geoDimensions = ""
    package var geoPatchSize = 4
    package var geoInputResolution = 10.0
    package var geoIncludeTokens = false
    package var benchmarkModels = ""
    package var benchmarkSuite = ""
    package var benchmarkDataset = ""
    package var benchmarkCases = ""
    package var benchmarkTrials = ""
    package var benchmarkResume = false
    package var benchmarkFixtureCheck = false
    package var benchmarkSandbox = ""
    package var benchmarkAllowCodeExecution = false
    package var speechListenDevice = ""
    package var speechListenListDevices = false
    package var speechListenDecodeMS = 0
    package var speechListenSilenceMS = 0
    package var visionServeMaxFrameBytes = 0
    package var visionServeMaxBatchSize = 8
    package var visionServeMaxBatchBytes = 0
    package var openWebUIHost = "127.0.0.1"
    package var openWebUIPort = 3_000
    package var openWebUIContainerName = "open-webui-mere-run"
    package var openWebUIVolumeName = "open-webui-mere-run"
    package var openWebUIImage = "ghcr.io/open-webui/open-webui:main"
    package var openWebUIVisionModel = "vision-chat-gemma4-12b"
    package var openWebUIEmbeddingModel = "text-embed-qwen3-0.6b"
    package var openWebUIImageModel = "image-zimage-nano"
    package var openWebUITTSModel = "speech-tts-qwen3-nano"
    package var openWebUISTTModel = "speech-asr-parakeet"
    package var openWebUITTSFormat = "wav"
    package var openWebUIAdminEmail = "admin@localhost"
    package var openWebUIAdminPassword = "admin"
    package var openWebUIWaitSeconds = 180
    package var openWebUIPull = false
    package var openWebUISkipServer = false
    package var openWebUISkipDocker = false
    package var openWebUISkipConfigure = false
    package var openWebUIReset = false
    package var apiLoRA = ""
    package var apiRateLimitPerMinute = 60
    package var apiMaxActiveRequests = 1
    package var apiMemoryGuard = "balanced"
    package var apiMemoryGuardCustomCeilingGB = ""
    package var variant = "zimage"
    package var quiet = false
    package var force = false
    package var all = false
    package var acceptModelLicense = false
    package var json = false
    package var stream = false
    package var extraArguments = ""
}

package struct CommandTemplate: Identifiable, Equatable {
    package let id: CommandTemplateID
    package let category: CommandCategory
    package let title: String
    package let subtitle: String
    package let systemImage: String
    package let promptLabel: String?
    package let secondaryLabel: String?
    package let inputKind: CommandInputKind
    package let outputKind: CommandOutputKind
    package let defaultPrompt: String
    package let defaultSecondaryText: String
    package let defaultModel: String
    package let defaultExtraArguments: String
    package let externalURL: URL?

    package init(
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
    package var declaredOutputKind: MereRunCapabilityOutputKind? {
        id.capability?.output.kind
    }

    /// True when the command can materialize a file or directory the app adopts as the run's
    /// primary artifact.
    ///
    /// The contract is the authority: a command writes one either because that is all it does
    /// (`kind` is `file` or `directory`) or because the caller can name a destination
    /// (`flag`) — a `text` or `service` command that prints to stdout until it is asked to
    /// write. Only the rows the contract does not cover (`.custom`, the launcher entries) fall
    /// back to the app's own `outputKind`.
    ///
    /// The caller still pairs this with a non-blank `outputPath`, so a run that was never asked
    /// to write never waits for a file that will not appear; see
    /// `ArtifactResolver.expectedOutput`.
    package var producesOutputFile: Bool {
        guard let output = id.capability?.output else { return outputKind != .none }
        return output.kind == .file || output.kind == .directory || output.flag != nil
    }

    package func defaultDraft() -> CommandDraft {
        var draft = CommandDraft()
        draft.prompt = defaultPrompt
        draft.secondaryText = defaultSecondaryText
        draft.model = defaultModel
        draft.extraArguments = defaultExtraArguments
        CommandDefaults.apply(to: &draft, id: id)

        // Destinations are user-visible folders per domain (`StudioOutputLocation`); the composer
        // renames the file after the prompt, and specialist surfaces keep this stamped name.
        if let path = StudioOutputLocation.templateOutputPath(
            templateID: id, title: title, outputKind: outputKind
        ) {
            draft.outputPath = path
        }

        return draft
    }

    package func validationMessage(for draft: CommandDraft, source: StudioScopeSource) -> String? {
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
            .audioEdit,
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

        if let message = categoryValidationMessage(for: draft, source: source) {
            return message
        }

        guard let capability = id.capability else { return nil }
        let arguments = arguments(from: draft, source: source)
        let launching = source.scope(capability: capability, commandLine: arguments)
        // Commands with a reply budget (Chat, Code, the vision prompts) validate the form's argv
        // against the contract, so a budget the runtime would reject never replaces a reply.
        if capability.options.contains(where: { $0.flag == "--max-tokens" }) {
            return StudioConsoleCommand.validationMessage(
                for: capability, draft: StudioConsoleCommand.seed(capability: capability, arguments: arguments),
                launching: launching
            )
        }
        // Every other command still hears the CLI gate's own objection before it launches.
        return launching.refusal
    }

    /// The template's own checks, which live beside its argv in `Catalog/<Category>.swift`.
    private func categoryValidationMessage(for draft: CommandDraft, source: StudioScopeSource) -> String? {
        switch category {
        case .setup: return CommandCatalog.setupValidationMessage(for: id, draft: draft)
        case .models: return CommandCatalog.modelsValidationMessage(for: id, draft: draft)
        case .image: return CommandCatalog.imageValidationMessage(for: id, draft: draft)
        case .text: return CommandCatalog.textValidationMessage(for: id, draft: draft)
        case .speech: return CommandCatalog.speechValidationMessage(for: id, draft: draft)
        case .vision: return CommandCatalog.visionValidationMessage(for: id, draft: draft)
        case .geospatial: return CommandCatalog.geospatialValidationMessage(for: id, draft: draft)
        case .media:
            return CommandCatalog.audioValidationMessage(for: id, draft: draft, source: source)
                ?? CommandCatalog.videoValidationMessage(for: id, draft: draft, source: source)
                ?? CommandCatalog.musicValidationMessage(for: id, draft: draft)
        case .sfx: return CommandCatalog.soundFXValidationMessage(for: id, draft: draft)
        case .operations: return CommandCatalog.operationsValidationMessage(for: id, draft: draft)
        case .server: return CommandCatalog.serverValidationMessage(for: id, draft: draft)
        case .custom: return CommandCatalog.customValidationMessage(for: id, draft: draft)
        }
    }

    /// The command `draft` runs: the template's argv scoped to the model it runs, then Extra
    /// arguments verbatim. A builder may emit a flag the selected model does not use (a default
    /// it always states, a value the model hides); the scope drops it here, while Extra
    /// arguments stay a raw escape hatch that the CLI's gate answers.
    package func arguments(from draft: CommandDraft, source: StudioScopeSource) -> [String] {
        let generated = CommandArguments.build(for: id, draft: draft, source: source)
        guard let capability = source.capability(for: id) else { return withExtraArguments(generated, draft) }
        let scope = source.scope(capability: capability, commandLine: withExtraArguments(generated, draft))
        return withExtraArguments(StudioOptionScopes.filtered(generated, scope: scope), draft)
    }

    /// The argv the template builds for `draft` before any scope: what a surface reads its scope
    /// from, and what a task draft is first seeded with, so a value only another model uses is
    /// there when the user switches to it.
    package func unscopedArguments(from draft: CommandDraft, source: StudioScopeSource) -> [String] {
        withExtraArguments(CommandArguments.build(for: id, draft: draft, source: source), draft)
    }

    private func withExtraArguments(_ generated: [String], _ draft: CommandDraft) -> [String] {
        switch id {
        // `custom` is already the raw command line a person typed, and the two launcher rows
        // hand off to another product, so neither takes the extra-arguments field.
        case .custom, .graphStudio, .nodeConsole:
            return generated
        default:
            return generated + ShellWords.split(draft.extraArguments)
        }
    }
}

extension CommandTemplate {
    /// Whether a repeatable prompt positional takes one text per line (`text embed`, where each
    /// line is a document to compare) or the whole passage as one text (`text anonymize`, whose
    /// page sent the paste as one argument so a paragraph keeps its lines together).
    package var promptSplitsLines: Bool {
        id != .textAnonymize
    }

    /// The primary Studio workspace that owns this command's durable Library entry.
    ///
    /// Advanced exposes specialist commands without adding dozens of modes to the main sidebar.
    /// Mapping them into the nearest creative workspace lets every run use the same queue,
    /// progress, history, and result canvas while `StudioLibraryItem.templateID` preserves the
    /// precise command identity and title.
    package var libraryMode: StudioMode {
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

        case .textChat, .textEmbed, .textAnonymize, .textDecide, .textTrainLoRA:
            return .chat
        case .textCode:
            return .code

        case .speechSynthesize,
             .speechProfileList,
             .speechProfileCreate,
             .speechProfileDelete:
            return .speak
        case .speechTranscribe, .speechDiarize, .speechDiarizeLive:
            return .listen
        case .speechListen:
            return .listen
        case .audioEnhance, .audioEdit, .audioGenerate:
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
             .visionDepth,
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
             .modelBenchmarkGemma4MTP, .modelBenchmarkParakeetCoreML,
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

package enum CommandLaunchEnvironment {
    package static let apiKeyEnvironmentKey = "MERERUN_API_KEY"
    package static let openWebUIAdminPasswordEnvironmentKey = "MERERUN_OPEN_WEBUI_ADMIN_PASSWORD"

    /// The options whose value the app carries in the environment instead of argv, and the
    /// `CommandDraft` field each one is read from. The Command Console leaves these flags out of
    /// the command it builds from the contract and fills the draft field instead, so a key typed
    /// there travels the same way one typed anywhere else in the app does.
    package static func secretFlags(for templateID: CommandTemplateID) -> [String: WritableKeyPath<CommandDraft, String>] {
        switch templateID {
        case .apiServe, .musicServe, .worldServe, .visionServe, .statusSnapshot:
            return ["--api-key": \.apiKey]
        case .openWebui:
            return ["--api-key": \.apiKey, "--admin-password": \.openWebUIAdminPassword]
        default:
            return [:]
        }
    }

    package static func overrides(templateID: CommandTemplateID, draft: CommandDraft) -> [String: String] {
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

package enum CommandCatalog {
    /// Every command Advanced can run, grouped the way the picker lists them. The records
    /// and the argv each one builds live together in `Catalog/<Category>.swift`.
    package static let templates: [CommandTemplate] =
        setupTemplates
        + modelsTemplates
        + imageTemplates
        + textTemplates
        + speechTemplates
        + visionTemplates
        + geospatialTemplates
        + audioTemplates
        + videoTemplates
        + musicTemplates
        + soundFXTemplates
        + operationsTemplates
        + serverTemplates
        + customTemplates

    package static func templates(in category: CommandCategory) -> [CommandTemplate] {
        templates.filter { $0.category == category }
    }

    package static func template(id: CommandTemplateID) -> CommandTemplate? {
        templates.first { $0.id == id }
    }
}

package enum ShellWords {
    package static func split(_ raw: String) -> [String] {
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
    package var isBlank: Bool {
        trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

extension DateFormatter {
    package static let mereRunTimestamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter
    }()
}
