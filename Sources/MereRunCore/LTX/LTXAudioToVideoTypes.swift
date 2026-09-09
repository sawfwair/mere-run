import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

public struct LTXAudioToVideoGenerationOptions: Sendable {
    public static let defaultNegativePrompt = """
    blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, \
    grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, \
    deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, \
    wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of \
    field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent \
    lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny \
    valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, \
    mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, \
    off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward \
    pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, \
    inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.
    """

    public let prompt: String
    public let negativePrompt: String
    public let audioURL: URL
    public let audioStartTime: Double
    public let audioMaxDuration: Double?
    public let width: Int
    public let height: Int
    public let numFrames: Int
    public let fps: Double
    public let seed: Int
    public let inferenceSteps: Int
    public let maxTextLength: Int
    public let guidance: LTXAudioToVideoGuidance
    public let sourceImageURL: URL?
    public let imageStrength: Float
    public let imageFrameIndex: Int
    public let endImageURL: URL?
    public let endImageStrength: Float
    public let imageConditionings: [LTXVideoConditioningInput]
    public let generatedKeyframeCount: Int
    public let generatedKeyframeIndices: [Int]
    public let hdrColorSpace: LTXHDRColorSpace?
    public let hdrTransfer: LTXHDRTransfer
    public let transformerExecution: LTXTransformerExecution

    public init(
        prompt: String,
        negativePrompt: String = Self.defaultNegativePrompt,
        audioURL: URL,
        audioStartTime: Double = 0,
        audioMaxDuration: Double? = nil,
        width: Int,
        height: Int,
        numFrames: Int,
        fps: Double = 24,
        seed: Int,
        inferenceSteps: Int = 30,
        maxTextLength: Int = 1_024,
        guidance: LTXAudioToVideoGuidance = LTXAudioToVideoGuidance(),
        sourceImageURL: URL? = nil,
        imageStrength: Float = 1,
        imageFrameIndex: Int = 0,
        endImageURL: URL? = nil,
        endImageStrength: Float = 1,
        imageConditionings: [LTXVideoConditioningInput] = [],
        generatedKeyframeCount: Int = 0,
        generatedKeyframeIndices: [Int] = [],
        hdrColorSpace: LTXHDRColorSpace? = nil,
        hdrTransfer: LTXHDRTransfer = .acesCCT,
        transformerExecution: LTXTransformerExecution = .eager
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.audioURL = audioURL
        self.audioStartTime = audioStartTime
        self.audioMaxDuration = audioMaxDuration
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.fps = fps
        self.seed = seed
        self.inferenceSteps = inferenceSteps
        self.maxTextLength = maxTextLength
        self.guidance = guidance
        self.sourceImageURL = sourceImageURL
        self.imageStrength = imageStrength
        self.imageFrameIndex = imageFrameIndex
        self.endImageURL = endImageURL
        self.endImageStrength = endImageStrength
        self.imageConditionings = imageConditionings
        self.generatedKeyframeCount = generatedKeyframeCount
        self.generatedKeyframeIndices = generatedKeyframeIndices
        self.hdrColorSpace = hdrColorSpace
        self.hdrTransfer = hdrTransfer
        self.transformerExecution = transformerExecution
    }
}

public struct LTXAudioToVideoGenerationResult: @unchecked Sendable {
    public let frames: MLXArray
    public let hdrOutput: LTXHDROutputFrames?
    public let videoLatents: MLXArray
    public let audioLatents: MLXArray
    public let sourceAudio: MediaAudioBuffer
    public let generatedKeyframeLatents: MLXArray?
    public let generatedKeyframeIndices: [Int]
    public let timings: LTXGenerationTimings

    public init(
        frames: MLXArray,
        hdrOutput: LTXHDROutputFrames? = nil,
        videoLatents: MLXArray,
        audioLatents: MLXArray,
        sourceAudio: MediaAudioBuffer,
        generatedKeyframeLatents: MLXArray? = nil,
        generatedKeyframeIndices: [Int] = [],
        timings: LTXGenerationTimings = LTXGenerationTimings()
    ) {
        self.frames = frames
        self.hdrOutput = hdrOutput
        self.videoLatents = videoLatents
        self.audioLatents = audioLatents
        self.sourceAudio = sourceAudio
        self.generatedKeyframeLatents = generatedKeyframeLatents
        self.generatedKeyframeIndices = generatedKeyframeIndices
        self.timings = timings
    }
}

public enum LTXUnifiedAVGeneratorError: LocalizedError {
    case transformerWeightsMissing(URL)
    case upsamplerWeightsMissing(URL)
    case unsupportedLTX23SplitModel(URL)
    case ltx23TextEncoderMissing(String)
    case generatorNotLoaded
    case invalidResolution(width: Int, height: Int)
    case invalidFrameCount(Int)
    case invalidImageStrength(Float)
    case invalidImageFrameIndex(Int)
    case ltx25ConditioningRequiresLTX25
    case invalidGeneratedKeyframes([Int])
    case imageNotFound(URL)
    case imageDecodeFailed(URL)
    case referenceVideoNotFound(URL)
    case referenceVideoDecodeFailed(URL, String)
    case emptyPrompt
    case decoderNotLoaded
    case encoderNotLoaded
    case upsamplerNotLoaded
    case audioEmbeddingsMissing
    case audioDecoderNotLoaded
    case vocoderNotLoaded
    case bweVocoderConfigMissing(URL)
    case fullGenerationRequiresCompatibleModel(URL)
    case audioToVideoRequiresCompatibleModel(URL)
    case textToAudioRequiresLTX25Full(URL)
    case durationPredictionRequiresLTX25(URL?)
    case distilledLoRAMissing(URL)
    case loraMissing(URL)
    case audioVAEWeightsMissing(URL)
    case audioSourceNotFound(URL)
    case unsupportedAudioChannels(Int)
    case audioSegmentTooShort(required: Double, available: Double)
    case invalidAudioStartTime(Double)
    case invalidInferenceSteps(Int)
    case invalidSigmaSchedule([Float])
    case invalidFrameRate(Double)
    case audioDecodeReturnedTooFewSamples(required: Int, actual: Int)
    case audioLatentTooShort(required: Int, actual: Int)
    case audioToVideoGeneratorNotLoaded
    case textToAudioGeneratorNotLoaded
    case audioToVideoRequiresReload
    case fullGenerationRequiresReload
    case dfrRequiresLTX25Full(URL)
    case retakeRequiresLTX25(URL?)
    case invalidRetakeRange(start: Double, end: Double)
    case dubItRequiresLTX25(URL?)
    case dubItRequiresOneICLoRA(Int)
    case dubItReferenceAudioMissing(URL)
    case incompatibleLTX25Workflows(String)

    public var errorDescription: String? {
        switch self {
        case .transformerWeightsMissing(let url):
            return "Missing LTX transformer weights at \(url.path)"
        case .upsamplerWeightsMissing(let url):
            return "Missing LTX upsampler weights at \(url.path)"
        case .unsupportedLTX23SplitModel(let url):
            return """
            Detected an LTX 2.3 split MLX model at \(url.path). Use \
            `mere.run video generate --variant distilled` for video-only output, or \
            `--variant unified-av` for synchronized audio and video.
            """
        case .ltx23TextEncoderMissing(let id):
            return """
            LTX 2.3 requires the companion Gemma 3 text encoder `\(id)`. Install it with \
            `mere.run model pull video-ltx23-full-mlx --accept-model-license` for unified AV and A2Vid, or \
            `mere.run model pull video-ltx23-av-mlx --accept-model-license` for the distilled lane, or set \
            MERERUN_VIDEO_LTX_TEXT_ENCODER_ROOT to a local mlx-community/gemma-3-12b-it-4bit checkout.
            """
        case .generatorNotLoaded:
            return "LTX unified AV generator is not loaded."
        case .invalidResolution(let width, let height):
            return "Resolution must be divisible by 64 (got \(width)x\(height))."
        case .invalidFrameCount(let value):
            return "numFrames must satisfy 8n+1 and be >= 9 (got \(value))."
        case .invalidImageStrength(let value):
            return "imageStrength must be in [0, 1] (got \(value))."
        case .invalidImageFrameIndex(let value):
            return "imageFrameIndex must be >= 0 (got \(value))."
        case .ltx25ConditioningRequiresLTX25:
            return "Arbitrary timed image conditioning and generated keyframe slots require an LTX 2.5 checkpoint."
        case .invalidGeneratedKeyframes(let values):
            return "Generated keyframe indices must be strictly increasing pixel-frame positions inside the output (got \(values))."
        case .imageNotFound(let url):
            return "Source image not found: \(url.path)"
        case .imageDecodeFailed(let url):
            return "Could not decode source image: \(url.path)"
        case .referenceVideoNotFound(let url):
            return "Reference video not found: \(url.path)"
        case .referenceVideoDecodeFailed(let url, let details):
            return "Could not decode reference video \(url.path): \(details)"
        case .emptyPrompt:
            return "Prompt cannot be empty."
        case .decoderNotLoaded:
            return "LTX video decoder is not loaded."
        case .encoderNotLoaded:
            return "LTX video encoder is not loaded."
        case .upsamplerNotLoaded:
            return "LTX upsampler is not loaded."
        case .audioEmbeddingsMissing:
            return "Audio text embeddings are unavailable in this model."
        case .audioDecoderNotLoaded:
            return "Audio decoder is not loaded."
        case .vocoderNotLoaded:
            return "Audio vocoder is not loaded."
        case .bweVocoderConfigMissing(let url):
            return "LTX BWE vocoder weights require a vocoder BWE config under \(url.path)."
        case .fullGenerationRequiresCompatibleModel(let url):
            return "Native full LTX generation requires a compatible LTX 2.3 or 2.5 dev, distilled LoRA, VAE, and vocoder bundle at \(url.path)."
        case .audioToVideoRequiresCompatibleModel(let url):
            return "Native LTX audio-to-video requires a compatible full LTX 2.3 or 2.5 model at \(url.path)."
        case .textToAudioRequiresLTX25Full(let url):
            return "Native LTX text-to-audio requires the full LTX 2.5 checkpoint at \(url.path)."
        case .durationPredictionRequiresLTX25(let url):
            let location = url.map { " at \($0.path)" } ?? ""
            return "Automatic duration prediction requires an LTX 2.5 checkpoint\(location)."
        case .distilledLoRAMissing(let url):
            return "Missing the official LTX 2.3 distilled LoRA at \(url.path)."
        case .loraMissing(let url):
            return "Missing LTX LoRA at \(url.path)."
        case .audioVAEWeightsMissing(let url):
            return "Missing LTX audio VAE weights at \(url.path)."
        case .audioSourceNotFound(let url):
            return "Audio source not found: \(url.path)"
        case .unsupportedAudioChannels(let channels):
            return "LTX audio-to-video supports mono or stereo source audio, not \(channels) channels."
        case .audioSegmentTooShort(let required, let available):
            return "The selected audio segment is too short: generation requires \(required) seconds, but only \(available) seconds remain."
        case .invalidAudioStartTime(let value):
            return "audioStartTime must be finite and nonnegative (got \(value))."
        case .invalidInferenceSteps(let value):
            return "inferenceSteps must be positive (got \(value))."
        case .invalidSigmaSchedule(let values):
            return "LTX sigmas must be finite, nonincreasing values in [0, 1] ending at zero (got \(values))."
        case .invalidFrameRate(let value):
            return "fps must be positive (got \(value))."
        case .audioDecodeReturnedTooFewSamples(let required, let actual):
            return "Audio decoding returned \(actual) samples; the requested segment requires \(required) without padding."
        case .audioLatentTooShort(let required, let actual):
            return "The encoded audio contains \(actual) latent frames; generation requires \(required)."
        case .audioToVideoGeneratorNotLoaded:
            return "LTX audio-to-video generator is not loaded."
        case .textToAudioGeneratorNotLoaded:
            return "LTX text-to-audio generator is not loaded."
        case .audioToVideoRequiresReload:
            return "Reload the LTX audio-to-video generator before starting another generation."
        case .fullGenerationRequiresReload:
            return "Reload the full LTX generator before starting another two-stage generation."
        case .dfrRequiresLTX25Full(let url):
            return "LTX 2.5 DFR requires the full checkpoint, spatial/temporal upsamplers, and distilled LoRA at \(url.path)."
        case .retakeRequiresLTX25(let url):
            let location = url.map { " at \($0.path)" } ?? ""
            return "LTX Retake requires an official LTX 2.5 checkpoint\(location)."
        case .invalidRetakeRange(let start, let end):
            return "Retake requires 0 <= start-time < end-time <= output duration (got \(start)...\(end))."
        case .dubItRequiresLTX25(let url):
            let location = url.map { " at \($0.path)" } ?? ""
            return "LTX Dub-It requires an official LTX 2.5 checkpoint\(location)."
        case .dubItRequiresOneICLoRA(let count):
            return "LTX Dub-It requires exactly one IC-LoRA (got \(count))."
        case .dubItReferenceAudioMissing(let url):
            return "LTX Dub-It reference video has no audio track: \(url.path)"
        case .incompatibleLTX25Workflows(let details):
            return "Incompatible LTX 2.5 workflow options: \(details)"
        }
    }
}
