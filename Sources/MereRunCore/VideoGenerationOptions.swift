import Foundation
import MereRunContract

/// User settings shared by video planning and execution. Core parses compound
/// conditioning inputs before constructing the native request.
public struct VideoGenerationOptions: Sendable {
    public let prompt: String
    public let outputURL: URL
    public let model: String
    public let quality: LTXVideoQuality?
    public let outputMode: LTXVideoOutputMode?
    public let legacyVariant: LTXVideoVariant?
    public let modelRoot: String?
    public let width: Int?
    public let height: Int?
    public let numFrames: Int?
    public let steps: Int?
    public let h3WeightMode: String
    public let h3AccelerationMode: String
    public let h3RenderWidth: Int?
    public let h3RenderHeight: Int?
    public let h3Adapter: String?
    public let h3AdapterStrength: Float
    public let h3FrameInputs: [String]
    public let h3WindowFrames: Int?
    public let h3WindowOverlap: Int
    public let duration: Double?
    public let autoDuration: [Double]
    public let videoDecoder: LTXVideoDecoderKind?
    public let hdrColorSpace: LTXHDRColorSpace?
    public let hdrTransfer: LTXHDRTransfer?
    public let highQualityHDR: Bool
    public let textEmbeddings: String?
    public let vaeSpatialTileSize: Int?
    public let vaeSpatialTileOverlap: Int
    public let skipHDRMP4: Bool
    public let fps: Double
    public let seed: Int?
    public let negativePrompt: String?
    public let enhancePrompt: Bool
    public let promptEnhancerModel: String?
    public let promptEnhancerModelRoot: String?
    public let audio: String?
    public let audioStartTime: Double
    public let audioMaxDuration: Double?
    public let a2vGuidanceScale: Float
    public let videoCFGGuidanceScale: Float
    public let audioCFGGuidanceScale: Float
    public let v2aGuidanceScale: Float
    public let a2vSteps: Int
    public let ltxPreset: LTXGenerationPreset
    public let ltxPipeline: LTXGenerationPipeline
    public let ltxSampler: LTXSamplerMode?
    public let ltxSigmas: [Float]
    public let ltxStage2Sigmas: [Float]
    public let distilledLoRAStrengthStage1: Float?
    public let distilledLoRAStrengthStage2: Float?
    public let ltxSamplerEta: Float
    public let videoSTGScale: Float
    public let videoGuidanceRescale: Float
    public let videoSTGBlocks: [Int]
    public let videoGuidanceSkipStep: Int
    public let audioSTGScale: Float
    public let audioGuidanceRescale: Float
    public let audioSTGBlocks: [Int]
    public let audioGuidanceSkipStep: Int
    public let noRes2sBongMath: Bool
    public let res2sBongMaxIterations: Int
    public let gradientEstimationGamma: Float
    public let image: String?
    public let imageStrength: Float
    public let endImage: String?
    public let endImageStrength: Float
    public let imageConditionings: [String]
    public let numGeneratedKeyframes: Int
    public let generatedKeyframeIndices: [Int]
    public let loras: [String]
    public let videoConditionings: [String]
    public let conditioningAttentionStrength: Float
    public let conditioningAttentionMask: String?
    public let skipStage2: Bool
    public let referenceDownscaleFactor: Int?
    public let referenceTemporalScaleFactor: Int?
    public let dfr: Bool
    public let temporalUpsampleRounds: Int
    public let detailingLoRAs: [String]
    public let detailingReferenceDownscaleFactor: Int?
    public let references: [String]
    public let timings: Bool
    public let timingsOutput: String?
    public let guidanceScale: Float
    public let shift: Float

    public let ltxTransformerExecution: LTXTransformerExecution

    public let ltxGuidanceProjectionCache: LTXGuidanceProjectionCacheMode

    public let ltxTeaCache: Bool

    public let ltxTeaCacheThreshold: Float?

    public let ltxTeaCacheCalibrationOutput: String?

    public init(
        prompt: String,
        outputURL: URL,
        model: String,
        quality: LTXVideoQuality?,
        outputMode: LTXVideoOutputMode?,
        legacyVariant: LTXVideoVariant?,
        modelRoot: String?,
        width: Int?,
        height: Int?,
        numFrames: Int?,
        steps: Int?,
        h3WeightMode: String,
        h3AccelerationMode: String,
        h3RenderWidth: Int?,
        h3RenderHeight: Int?,
        h3Adapter: String?,
        h3AdapterStrength: Float,
        h3FrameInputs: [String],
        h3WindowFrames: Int?,
        h3WindowOverlap: Int,
        duration: Double?,
        autoDuration: [Double],
        videoDecoder: LTXVideoDecoderKind?,
        hdrColorSpace: LTXHDRColorSpace?,
        hdrTransfer: LTXHDRTransfer?,
        highQualityHDR: Bool,
        textEmbeddings: String?,
        vaeSpatialTileSize: Int?,
        vaeSpatialTileOverlap: Int,
        skipHDRMP4: Bool,
        fps: Double,
        seed: Int?,
        negativePrompt: String?,
        enhancePrompt: Bool,
        promptEnhancerModel: String?,
        promptEnhancerModelRoot: String?,
        audio: String?,
        audioStartTime: Double,
        audioMaxDuration: Double?,
        a2vGuidanceScale: Float,
        videoCFGGuidanceScale: Float,
        audioCFGGuidanceScale: Float,
        v2aGuidanceScale: Float,
        a2vSteps: Int,
        ltxPreset: LTXGenerationPreset,
        ltxPipeline: LTXGenerationPipeline,
        ltxSampler: LTXSamplerMode?,
        ltxSigmas: [Float],
        ltxStage2Sigmas: [Float],
        distilledLoRAStrengthStage1: Float?,
        distilledLoRAStrengthStage2: Float?,
        ltxSamplerEta: Float,
        videoSTGScale: Float,
        videoGuidanceRescale: Float,
        videoSTGBlocks: [Int],
        videoGuidanceSkipStep: Int,
        audioSTGScale: Float,
        audioGuidanceRescale: Float,
        audioSTGBlocks: [Int],
        audioGuidanceSkipStep: Int,
        noRes2sBongMath: Bool,
        res2sBongMaxIterations: Int,
        gradientEstimationGamma: Float,
        image: String?,
        imageStrength: Float,
        endImage: String?,
        endImageStrength: Float,
        imageConditionings: [String],
        numGeneratedKeyframes: Int,
        generatedKeyframeIndices: [Int],
        loras: [String],
        videoConditionings: [String],
        conditioningAttentionStrength: Float,
        conditioningAttentionMask: String?,
        skipStage2: Bool,
        referenceDownscaleFactor: Int?,
        referenceTemporalScaleFactor: Int?,
        dfr: Bool,
        temporalUpsampleRounds: Int,
        detailingLoRAs: [String],
        detailingReferenceDownscaleFactor: Int?,
        references: [String],
        timings: Bool,
        timingsOutput: String?,
        guidanceScale: Float,
        shift: Float,
        ltxTransformerExecution: LTXTransformerExecution,
        ltxGuidanceProjectionCache: LTXGuidanceProjectionCacheMode,
        ltxTeaCache: Bool,
        ltxTeaCacheThreshold: Float?,
        ltxTeaCacheCalibrationOutput: String?
    ) {
        self.prompt = prompt
        self.outputURL = outputURL
        self.model = model
        self.quality = quality
        self.outputMode = outputMode
        self.legacyVariant = legacyVariant
        self.modelRoot = modelRoot
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.steps = steps
        self.h3WeightMode = h3WeightMode
        self.h3AccelerationMode = h3AccelerationMode
        self.h3RenderWidth = h3RenderWidth
        self.h3RenderHeight = h3RenderHeight
        self.h3Adapter = h3Adapter
        self.h3AdapterStrength = h3AdapterStrength
        self.h3FrameInputs = h3FrameInputs
        self.h3WindowFrames = h3WindowFrames
        self.h3WindowOverlap = h3WindowOverlap
        self.duration = duration
        self.autoDuration = autoDuration
        self.videoDecoder = videoDecoder
        self.hdrColorSpace = hdrColorSpace
        self.hdrTransfer = hdrTransfer
        self.highQualityHDR = highQualityHDR
        self.textEmbeddings = textEmbeddings
        self.vaeSpatialTileSize = vaeSpatialTileSize
        self.vaeSpatialTileOverlap = vaeSpatialTileOverlap
        self.skipHDRMP4 = skipHDRMP4
        self.fps = fps
        self.seed = seed
        self.negativePrompt = negativePrompt
        self.enhancePrompt = enhancePrompt
        self.promptEnhancerModel = promptEnhancerModel
        self.promptEnhancerModelRoot = promptEnhancerModelRoot
        self.audio = audio.flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }
        self.audioStartTime = audioStartTime
        self.audioMaxDuration = audioMaxDuration
        self.a2vGuidanceScale = a2vGuidanceScale
        self.videoCFGGuidanceScale = videoCFGGuidanceScale
        self.audioCFGGuidanceScale = audioCFGGuidanceScale
        self.v2aGuidanceScale = v2aGuidanceScale
        self.a2vSteps = a2vSteps
        self.ltxPreset = ltxPreset
        self.ltxPipeline = ltxPipeline
        self.ltxSampler = ltxSampler
        self.ltxSigmas = ltxSigmas
        self.ltxStage2Sigmas = ltxStage2Sigmas
        self.distilledLoRAStrengthStage1 = distilledLoRAStrengthStage1
        self.distilledLoRAStrengthStage2 = distilledLoRAStrengthStage2
        self.ltxSamplerEta = ltxSamplerEta
        self.videoSTGScale = videoSTGScale
        self.videoGuidanceRescale = videoGuidanceRescale
        self.videoSTGBlocks = videoSTGBlocks
        self.videoGuidanceSkipStep = videoGuidanceSkipStep
        self.audioSTGScale = audioSTGScale
        self.audioGuidanceRescale = audioGuidanceRescale
        self.audioSTGBlocks = audioSTGBlocks
        self.audioGuidanceSkipStep = audioGuidanceSkipStep
        self.noRes2sBongMath = noRes2sBongMath
        self.res2sBongMaxIterations = res2sBongMaxIterations
        self.gradientEstimationGamma = gradientEstimationGamma
        self.image = image.flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }
        self.imageStrength = imageStrength
        self.endImage = endImage.flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }
        self.endImageStrength = endImageStrength
        self.imageConditionings = imageConditionings
        self.numGeneratedKeyframes = numGeneratedKeyframes
        self.generatedKeyframeIndices = generatedKeyframeIndices
        self.loras = loras
        self.videoConditionings = videoConditionings
        self.conditioningAttentionStrength = conditioningAttentionStrength
        self.conditioningAttentionMask = conditioningAttentionMask
        self.skipStage2 = skipStage2
        self.referenceDownscaleFactor = referenceDownscaleFactor
        self.referenceTemporalScaleFactor = referenceTemporalScaleFactor
        self.dfr = dfr
        self.temporalUpsampleRounds = temporalUpsampleRounds
        self.detailingLoRAs = detailingLoRAs
        self.detailingReferenceDownscaleFactor = detailingReferenceDownscaleFactor
        self.references = references
        self.timings = timings
        self.timingsOutput = timingsOutput
        self.guidanceScale = guidanceScale
        self.shift = shift
        self.ltxTransformerExecution = ltxTransformerExecution
        self.ltxGuidanceProjectionCache = ltxGuidanceProjectionCache
        self.ltxTeaCache = ltxTeaCache
        self.ltxTeaCacheThreshold = ltxTeaCacheThreshold
        self.ltxTeaCacheCalibrationOutput = ltxTeaCacheCalibrationOutput
    }
}
