import Foundation

public struct LTX25Resources: Sendable, Hashable {
    public static let modelID = "video-ltx25-distilled-bf16"
    public static let sourceRepository = "Lightricks/LTX-2.5"
    public static let sourceRevision = "dd53cc2cd45bbeaa3563dfb575cba3f49cf44761"
    public static let upstreamCodeRepository = "Lightricks/LTX-2"
    public static let upstreamCodeRevision = "d151147788a9284cca791edc6ce898007e727fe6"
    public static let upstreamCodeRelease = "v1.2.0"
    public static let estimatedDownloadBytes: Int64 = 71_098_810_082

    public static let transformerRelativePath =
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    public static let textEncoderRelativePath =
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    public static let videoVAERelativePath =
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
    public static let audioVAERelativePath =
        "vae/ltx-2.5-audio-vae-bf16.safetensors"
    public static let spatialUpsamplerRelativePath =
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    public static let durationHeadRelativePath =
        "model_patches/ltx-2.5-duration-head-bf16.safetensors"

    public static let requiredRelativePaths = [
        transformerRelativePath,
        textEncoderRelativePath,
        videoVAERelativePath,
        audioVAERelativePath,
        spatialUpsamplerRelativePath,
    ]

    public static let snapshotPatterns = [
        "README.md",
        transformerRelativePath,
        textEncoderRelativePath,
        videoVAERelativePath,
        audioVAERelativePath,
        spatialUpsamplerRelativePath,
        durationHeadRelativePath,
    ]

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL.standardizedFileURL
    }

    public var transformerURL: URL { rootURL.appendingPathComponent(Self.transformerRelativePath) }
    public var textEncoderURL: URL { rootURL.appendingPathComponent(Self.textEncoderRelativePath) }
    public var videoVAEURL: URL { rootURL.appendingPathComponent(Self.videoVAERelativePath) }
    public var audioVAEURL: URL { rootURL.appendingPathComponent(Self.audioVAERelativePath) }
    public var spatialUpsamplerURL: URL { rootURL.appendingPathComponent(Self.spatialUpsamplerRelativePath) }
    public var durationHeadURL: URL { rootURL.appendingPathComponent(Self.durationHeadRelativePath) }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        Self.requiredRelativePaths.compactMap { relativePath in
            let url = rootURL.appendingPathComponent(relativePath)
            return fileManager.fileExists(atPath: url.path) ? nil : url
        }
    }
}

public func isLTX25ModelRoot(
    _ rootURL: URL,
    fileManager: FileManager = .default
) -> Bool {
    LTX25Resources(rootURL: rootURL).validate(fileManager: fileManager).isEmpty
}
