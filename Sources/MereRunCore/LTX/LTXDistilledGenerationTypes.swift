import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

public struct LTXDistilledLatentGenerationOptions: Sendable {
    public let prompt: String
    public let width: Int
    public let height: Int
    public let numFrames: Int
    public let fps: Double
    public let seed: Int
    public let maxTextLength: Int
    public let sourceImageURL: URL?
    public let imageStrength: Float
    public let imageFrameIndex: Int
    public let endImageURL: URL?
    public let endImageStrength: Float

    public init(
        prompt: String,
        width: Int,
        height: Int,
        numFrames: Int,
        fps: Double = 24,
        seed: Int,
        maxTextLength: Int = 1024,
        sourceImageURL: URL? = nil,
        imageStrength: Float = 1.0,
        imageFrameIndex: Int = 0,
        endImageURL: URL? = nil,
        endImageStrength: Float = 1.0
    ) {
        self.prompt = prompt
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.fps = fps
        self.seed = seed
        self.maxTextLength = maxTextLength
        self.sourceImageURL = sourceImageURL
        self.imageStrength = imageStrength
        self.imageFrameIndex = imageFrameIndex
        self.endImageURL = endImageURL
        self.endImageStrength = endImageStrength
    }
}

public struct LTXDistilledLatentGenerationResult: @unchecked Sendable {
    public let latents: MLXArray
    public let stage1Latents: MLXArray

    public init(latents: MLXArray, stage1Latents: MLXArray) {
        self.latents = latents
        self.stage1Latents = stage1Latents
    }
}

public struct LTXDistilledVideoGenerationResult: @unchecked Sendable {
    public let frames: MLXArray
    public let latents: MLXArray

    public init(frames: MLXArray, latents: MLXArray) {
        self.frames = frames
        self.latents = latents
    }
}

public enum LTXDistilledLatentGeneratorError: LocalizedError {
    case transformerWeightsMissing(URL)
    case upsamplerWeightsMissing(URL)
    case unsupportedLTX23SplitModel(URL)
    case generatorNotLoaded
    case invalidResolution(width: Int, height: Int)
    case invalidFrameCount(Int)
    case invalidImageStrength(Float)
    case invalidImageFrameIndex(Int)
    case ltx25ConditioningRequiresLTX25
    case invalidGeneratedKeyframes([Int])
    case imageNotFound(URL)
    case imageDecodeFailed(URL)
    case emptyPrompt
    case decoderNotLoaded
    case encoderNotLoaded
    case upsamplerNotLoaded

    public var errorDescription: String? {
        switch self {
        case .transformerWeightsMissing(let url):
            return "Missing LTX transformer weights at \(url.path)"
        case .upsamplerWeightsMissing(let url):
            return "Missing LTX upsampler weights at \(url.path)"
        case .unsupportedLTX23SplitModel(let url):
            return """
            Detected an LTX 2.3 split MLX model at \(url.path). This legacy loader only supports the older \
            merged LTX layout. Use `mere.run video generate --variant distilled` for video-only output from \
            split models, or `--variant unified-av` for synchronized audio and video.
            """
        case .generatorNotLoaded:
            return "LTX distilled latent generator is not loaded."
        case .invalidResolution(let width, let height):
            return "Resolution does not meet the selected LTX pipeline's 32- or 64-pixel alignment (got \(width)x\(height))."
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
        case .emptyPrompt:
            return "Prompt cannot be empty."
        case .decoderNotLoaded:
            return "LTX decoder is not loaded."
        case .encoderNotLoaded:
            return "LTX encoder is not loaded."
        case .upsamplerNotLoaded:
            return "LTX latent upsampler is not loaded."
        }
    }
}

public func isLTX23SplitModelRoot(_ rootURL: URL, fileManager: FileManager = .default) -> Bool {
    let root = rootURL.standardizedFileURL
    let splitModel = root.appendingPathComponent("split_model.json", isDirectory: false)
    let transformer = root.appendingPathComponent("transformer-distilled.safetensors", isDirectory: false)
    guard fileManager.fileExists(atPath: splitModel.path),
          fileManager.fileExists(atPath: transformer.path) else {
        return false
    }

    let config = root.appendingPathComponent("config.json", isDirectory: false)
    guard let data = try? Data(contentsOf: config),
          let text = String(data: data, encoding: .utf8) else {
        return true
    }
    return text.contains(#""model_version""#) && text.contains("2.3")
}

public func isLTX23AudioToVideoModelRoot(
    _ rootURL: URL,
    fileManager: FileManager = .default
) -> Bool {
    let root = rootURL.standardizedFileURL
    let required = [
        "split_model.json",
        "config.json",
        "connector.safetensors",
        "transformer-dev.safetensors",
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "vae_decoder.safetensors",
        "vae_encoder.safetensors",
        "audio_vae.safetensors",
        "spatial_upscaler_x2_v1_1.safetensors",
    ]
    return required.allSatisfy {
        fileManager.fileExists(atPath: root.appendingPathComponent($0).path)
    }
}

public func isLTX23FullModelRoot(
    _ rootURL: URL,
    fileManager: FileManager = .default
) -> Bool {
    isLTX23AudioToVideoModelRoot(rootURL, fileManager: fileManager)
        && fileManager.fileExists(
            atPath: rootURL.standardizedFileURL
                .appendingPathComponent("vocoder.safetensors", isDirectory: false).path
        )
}
