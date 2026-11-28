import Foundation

public enum VideoGenerationError: Error, LocalizedError, Equatable, Sendable {
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidInput(let message): message
        }
    }
}

/// Controls reuse of positive-prompt attention projections across full-model guidance passes.
public enum LTXGuidanceProjectionCacheMode: String, Codable, CaseIterable, Sendable {
    /// Reuse projections only when more than one positive pass is required and memory has a safe reserve.
    case automatic

    /// Recompute prompt projections for every guidance pass.
    case disabled

    /// Prefer projection reuse, while still falling back when the machine cannot retain the cache safely.
    case enabled
}

public enum LTXHDRColorSpace: String, Sendable, CaseIterable, Hashable {
    case srgbLinear = "srgb-linear"
    case acescg
    case acescct
}

public enum LTXHDRTransfer: String, Sendable, CaseIterable, Hashable {
    case logC3 = "logc3"
    case acesCCT = "acescct"
}

/// Guidance for one stream of the full LTX 2.3 joint audio/video denoiser.
public struct LTXMultiModalGuidance: Sendable, Hashable {
    public let classifierFreeScale: Float
    public let spatioTemporalScale: Float
    public let rescale: Float
    public let modalityScale: Float
    public let spatioTemporalBlocks: Set<Int>
    public let skipStep: Int

    public init(
        classifierFreeScale: Float,
        spatioTemporalScale: Float = 1,
        rescale: Float = 0.7,
        modalityScale: Float = 3,
        spatioTemporalBlocks: Set<Int> = [28],
        skipStep: Int = 0
    ) {
        self.classifierFreeScale = classifierFreeScale
        self.spatioTemporalScale = spatioTemporalScale
        self.rescale = rescale
        self.modalityScale = modalityScale
        self.spatioTemporalBlocks = spatioTemporalBlocks
        self.skipStep = skipStep
    }
}

public struct MiniMaxH3SlidingWindowOptions: Sendable, Hashable {
    public let totalFrameCount: Int
    public let windowFrameCount: Int
    public let overlapFrameCount: Int

    public init(
        totalFrameCount: Int,
        windowFrameCount: Int,
        overlapFrameCount: Int
    ) throws {
        guard totalFrameCount >= 22, totalFrameCount % 17 == 5 else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "sliding output frame count must be at least 22 and have the form 17*n+5"
            )
        }
        guard windowFrameCount >= 22, windowFrameCount % 17 == 5 else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "--h3-window-frames must be at least 22 and have the form 17*n+5"
            )
        }
        guard overlapFrameCount >= 1, overlapFrameCount % 17 == 1 else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "--h3-window-overlap must have the form 17*n+1"
            )
        }
        guard overlapFrameCount < windowFrameCount else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "--h3-window-overlap must be smaller than --h3-window-frames"
            )
        }
        let continuationTargetCount = windowFrameCount - overlapFrameCount + 1
        guard continuationTargetCount >= 22 else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "sliding windows must leave at least 22 target frames after overlap"
            )
        }
        self.totalFrameCount = totalFrameCount
        self.windowFrameCount = windowFrameCount
        self.overlapFrameCount = overlapFrameCount
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
