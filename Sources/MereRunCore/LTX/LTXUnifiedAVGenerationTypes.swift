import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

public struct LTXUnifiedAVGenerationOptions: Sendable {
    public static let defaultNegativePrompt = LTXAudioToVideoGenerationOptions.defaultNegativePrompt

    public let prompt: String
    public let negativePrompt: String
    public let width: Int
    public let height: Int
    public let numFrames: Int
    public let fps: Double
    public let seed: Int
    public let inferenceSteps: Int
    public let maxTextLength: Int
    public let videoGuidance: LTXMultiModalGuidance
    public let audioGuidance: LTXMultiModalGuidance
    public let sourceImageURL: URL?
    public let imageStrength: Float
    public let imageFrameIndex: Int
    public let endImageURL: URL?
    public let endImageStrength: Float
    public let imageConditionings: [LTXVideoConditioningInput]
    public let generatedKeyframeCount: Int
    public let generatedKeyframeIndices: [Int]
    public let referenceVideos: [LTXReferenceVideoConditioningInput]
    public let loras: [LTXLoRAConfiguration]
    public let dfr: LTX25DFROptions?
    public let sigmas: [Float]?
    public let stage2Sigmas: [Float]?
    public let sampler: LTXSamplerConfiguration
    public let pipeline: LTXGenerationPipeline
    public let distilledLoRAStrengthStage1: Float
    public let distilledLoRAStrengthStage2: Float
    public let hdrColorSpace: LTXHDRColorSpace?
    public let hdrTransfer: LTXHDRTransfer
    public let hdrICLoRA: LTXHDRICLoRAOptions?
    public let vaeSpatialTileSize: Int?
    public let vaeSpatialTileOverlap: Int
    public let skipStage2: Bool
    public let precomputedTextEmbeddingsURL: URL?
    public let retake: LTXRetakeOptions?
    public let dubIt: LTXDubItOptions?
    public let transformerExecution: LTXTransformerExecution
    public let guidanceProjectionCache: LTXGuidanceProjectionCacheMode
    public let teaCache: LTXTeaCacheConfiguration?

    public init(
        prompt: String,
        negativePrompt: String = Self.defaultNegativePrompt,
        width: Int,
        height: Int,
        numFrames: Int,
        fps: Double = 24,
        seed: Int,
        inferenceSteps: Int = 30,
        maxTextLength: Int = 1024,
        videoGuidance: LTXMultiModalGuidance = LTXMultiModalGuidance(classifierFreeScale: 3),
        audioGuidance: LTXMultiModalGuidance = LTXMultiModalGuidance(classifierFreeScale: 7),
        sourceImageURL: URL? = nil,
        imageStrength: Float = 1.0,
        imageFrameIndex: Int = 0,
        endImageURL: URL? = nil,
        endImageStrength: Float = 1.0,
        imageConditionings: [LTXVideoConditioningInput] = [],
        generatedKeyframeCount: Int = 0,
        generatedKeyframeIndices: [Int] = [],
        referenceVideos: [LTXReferenceVideoConditioningInput] = [],
        loras: [LTXLoRAConfiguration] = [],
        dfr: LTX25DFROptions? = nil,
        sigmas: [Float]? = nil,
        stage2Sigmas: [Float]? = nil,
        sampler: LTXSamplerConfiguration = LTXSamplerConfiguration(),
        pipeline: LTXGenerationPipeline = .twoStage,
        distilledLoRAStrengthStage1: Float = 0,
        distilledLoRAStrengthStage2: Float = 1,
        hdrColorSpace: LTXHDRColorSpace? = nil,
        hdrTransfer: LTXHDRTransfer = .acesCCT,
        hdrICLoRA: LTXHDRICLoRAOptions? = nil,
        vaeSpatialTileSize: Int? = nil,
        vaeSpatialTileOverlap: Int = 256,
        skipStage2: Bool = false,
        precomputedTextEmbeddingsURL: URL? = nil,
        retake: LTXRetakeOptions? = nil,
        dubIt: LTXDubItOptions? = nil,
        transformerExecution: LTXTransformerExecution = .eager,
        guidanceProjectionCache: LTXGuidanceProjectionCacheMode = .disabled,
        teaCache: LTXTeaCacheConfiguration? = nil
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.fps = fps
        self.seed = seed
        self.inferenceSteps = inferenceSteps
        self.maxTextLength = maxTextLength
        self.videoGuidance = videoGuidance
        self.audioGuidance = audioGuidance
        self.sourceImageURL = sourceImageURL
        self.imageStrength = imageStrength
        self.imageFrameIndex = imageFrameIndex
        self.endImageURL = endImageURL
        self.endImageStrength = endImageStrength
        self.imageConditionings = imageConditionings
        self.generatedKeyframeCount = generatedKeyframeCount
        self.generatedKeyframeIndices = generatedKeyframeIndices
        self.referenceVideos = referenceVideos
        self.loras = loras
        self.dfr = dfr
        self.sigmas = sigmas
        self.stage2Sigmas = stage2Sigmas
        self.sampler = sampler
        self.pipeline = pipeline
        self.distilledLoRAStrengthStage1 = distilledLoRAStrengthStage1
        self.distilledLoRAStrengthStage2 = distilledLoRAStrengthStage2
        self.hdrColorSpace = hdrColorSpace
        self.hdrTransfer = hdrTransfer
        self.hdrICLoRA = hdrICLoRA
        self.vaeSpatialTileSize = vaeSpatialTileSize
        self.vaeSpatialTileOverlap = vaeSpatialTileOverlap
        self.skipStage2 = skipStage2
        self.precomputedTextEmbeddingsURL = precomputedTextEmbeddingsURL?.standardizedFileURL
        self.retake = retake
        self.dubIt = dubIt
        self.transformerExecution = transformerExecution
        self.guidanceProjectionCache = guidanceProjectionCache
        self.teaCache = teaCache
    }
}

public struct LTXUnifiedAVGenerationResult: @unchecked Sendable {
    public let frames: MLXArray
    public let videoLatents: MLXArray
    public let audioLatents: MLXArray
    public let audioWaveform: MLXArray
    public let audioSampleRate: Int
    public let hdrOutput: LTXHDROutputFrames?
    public let generatedKeyframeLatents: MLXArray?
    public let generatedKeyframeIndices: [Int]
    public let playbackFPS: Double
    public let timings: LTXGenerationTimings

    public init(
        frames: MLXArray,
        videoLatents: MLXArray,
        audioLatents: MLXArray,
        audioWaveform: MLXArray,
        audioSampleRate: Int,
        hdrOutput: LTXHDROutputFrames? = nil,
        generatedKeyframeLatents: MLXArray? = nil,
        generatedKeyframeIndices: [Int] = [],
        playbackFPS: Double = 24,
        timings: LTXGenerationTimings = LTXGenerationTimings()
    ) {
        self.frames = frames
        self.videoLatents = videoLatents
        self.audioLatents = audioLatents
        self.audioWaveform = audioWaveform
        self.audioSampleRate = audioSampleRate
        self.hdrOutput = hdrOutput
        self.generatedKeyframeLatents = generatedKeyframeLatents
        self.generatedKeyframeIndices = generatedKeyframeIndices
        self.playbackFPS = playbackFPS
        self.timings = timings
    }
}

public struct LTXUnifiedVideoGenerationResult: @unchecked Sendable {
    public let frames: MLXArray
    public let hdrOutput: LTXHDROutputFrames?
    public let videoLatents: MLXArray
    public let generatedKeyframeLatents: MLXArray?
    public let generatedKeyframeIndices: [Int]
    public let playbackFPS: Double
    public let timings: LTXGenerationTimings

    public init(
        frames: MLXArray,
        hdrOutput: LTXHDROutputFrames? = nil,
        videoLatents: MLXArray,
        generatedKeyframeLatents: MLXArray? = nil,
        generatedKeyframeIndices: [Int] = [],
        playbackFPS: Double = 24,
        timings: LTXGenerationTimings = LTXGenerationTimings()
    ) {
        self.frames = frames
        self.hdrOutput = hdrOutput
        self.videoLatents = videoLatents
        self.generatedKeyframeLatents = generatedKeyframeLatents
        self.generatedKeyframeIndices = generatedKeyframeIndices
        self.playbackFPS = playbackFPS
        self.timings = timings
    }
}
