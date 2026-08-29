import Foundation

public struct LTX25Resources: Sendable, Hashable {
    public static let modelID = "video-ltx25-distilled-bf16"
    public static let fullModelID = "video-ltx25-full-bf16"
    public static let sourceRepository = "Lightricks/LTX-2.5"
    public static let sourceRevision = "dd53cc2cd45bbeaa3563dfb575cba3f49cf44761"
    public static let managedRepository =
        "Sawfwair/LTX-2.5-Distilled-BF16-MLX-Q4-Text"
    public static let managedRevision = "cf8a174746cd14796c81ca2b54e035dc32e69bd8"
    public static let fullManagedRepository = "Sawfwair/LTX-2.5-Full-BF16-MLX"
    public static let fullManagedRevision = "ac74d124f7211fc3cb8b32f418a08d8e71655c8d"
    public static let upstreamCodeRepository = "Lightricks/LTX-2"
    public static let upstreamCodeRevision = "d151147788a9284cca791edc6ce898007e727fe6"
    public static let upstreamCodeRelease = "v1.2.0"
    public static let textEncoderSourceBytes: Int64 = 26_263_860_594
    public static let estimatedDownloadBytes: Int64 = 53_878_517_792
    public static let fullEstimatedDownloadBytes: Int64 = 119_718_579_164

    public static let distilledTransformerRelativePath =
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    public static let transformerRelativePath = distilledTransformerRelativePath
    public static let devTransformerRelativePath =
        "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
    public static let textEncoderRelativePath =
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    public static let videoVAERelativePath =
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
    public static let diffusionVideoVAERelativePath =
        "vae/ltx-2.5-video-vae-bf16.safetensors"
    public static let audioVAERelativePath =
        "vae/ltx-2.5-audio-vae-bf16.safetensors"
    public static let spatialUpsamplerRelativePath =
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    public static let temporalUpsamplerRelativePath =
        "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
    public static let distilledLoRARelativePath =
        "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
    public static let durationHeadRelativePath =
        "model_patches/ltx-2.5-duration-head-bf16.safetensors"
    public static let nativeDistilledTransformerRelativePath =
        "\(LTX25NativeModelPack.relativeDirectory)/distilled-transformer-bf16.safetensors"
    public static let nativeDevTransformerRelativePath =
        "\(LTX25NativeModelPack.relativeDirectory)/dev-transformer-bf16.safetensors"
    public static let nativeConnectorRelativePath =
        "\(LTX25NativeModelPack.relativeDirectory)/connector-bf16.safetensors"

    public static let requiredRelativePaths = [
        distilledTransformerRelativePath,
        videoVAERelativePath,
        audioVAERelativePath,
        spatialUpsamplerRelativePath,
    ]

    private static let sharedRequiredRelativePaths = [
        videoVAERelativePath,
        audioVAERelativePath,
        spatialUpsamplerRelativePath,
    ]

    public static let fullRequiredRelativePaths = [
        devTransformerRelativePath,
        distilledTransformerRelativePath,
        textEncoderRelativePath,
        videoVAERelativePath,
        diffusionVideoVAERelativePath,
        audioVAERelativePath,
        spatialUpsamplerRelativePath,
        temporalUpsamplerRelativePath,
        distilledLoRARelativePath,
        durationHeadRelativePath,
    ]

    public static let snapshotPatterns = [
        "README.md",
        "LICENSE.md",
        "LTX-ACCEPTABLE-USE-POLICY.pdf",
        "GEMMA-4-LICENSE.txt",
        "NOTICE.md",
        nativeDistilledTransformerRelativePath,
        nativeConnectorRelativePath,
        videoVAERelativePath,
        audioVAERelativePath,
        spatialUpsamplerRelativePath,
        durationHeadRelativePath,
        "\(LTX25TextEncoderQuantizedPack.relativeDirectory)/pack.json",
        "\(LTX25TextEncoderQuantizedPack.relativeDirectory)/model.safetensors.index.json",
        "\(LTX25TextEncoderQuantizedPack.relativeDirectory)/*.safetensors",
    ]

    public static let fullSnapshotPatterns = [
        "README.md",
        "LICENSE.md",
        "LTX-ACCEPTABLE-USE-POLICY.pdf",
        "GEMMA-4-LICENSE.txt",
        "NOTICE.md",
        nativeDevTransformerRelativePath,
        nativeDistilledTransformerRelativePath,
        nativeConnectorRelativePath,
        textEncoderRelativePath,
        videoVAERelativePath,
        diffusionVideoVAERelativePath,
        audioVAERelativePath,
        spatialUpsamplerRelativePath,
        temporalUpsamplerRelativePath,
        distilledLoRARelativePath,
        durationHeadRelativePath,
    ]

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL.standardizedFileURL
    }

    public var transformerURL: URL { rootURL.appendingPathComponent(Self.transformerRelativePath) }
    public var distilledTransformerURL: URL {
        rootURL.appendingPathComponent(Self.distilledTransformerRelativePath)
    }
    public var devTransformerURL: URL { rootURL.appendingPathComponent(Self.devTransformerRelativePath) }
    public var textEncoderURL: URL { rootURL.appendingPathComponent(Self.textEncoderRelativePath) }
    public var videoVAEURL: URL { rootURL.appendingPathComponent(Self.videoVAERelativePath) }
    public var diffusionVideoVAEURL: URL {
        rootURL.appendingPathComponent(Self.diffusionVideoVAERelativePath)
    }
    public var audioVAEURL: URL { rootURL.appendingPathComponent(Self.audioVAERelativePath) }
    public var spatialUpsamplerURL: URL { rootURL.appendingPathComponent(Self.spatialUpsamplerRelativePath) }
    public var temporalUpsamplerURL: URL {
        rootURL.appendingPathComponent(Self.temporalUpsamplerRelativePath)
    }
    public var distilledLoRAURL: URL { rootURL.appendingPathComponent(Self.distilledLoRARelativePath) }
    public var durationHeadURL: URL { rootURL.appendingPathComponent(Self.durationHeadRelativePath) }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing = Self.sharedRequiredRelativePaths.compactMap { relativePath in
            let url = rootURL.appendingPathComponent(relativePath)
            return fileManager.fileExists(atPath: url.path) ? nil : url
        }
        missing.append(contentsOf: missingTransformerPaths(kinds: [.distilled], fileManager: fileManager))
        if !fileManager.fileExists(atPath: distilledTransformerURL.path),
           LTX25NativeModelPack.optimizedURLIfValid(
               resources: self,
               kind: .connector,
               fileManager: fileManager
           ) == nil
        {
            missing.append(LTX25NativeModelPack.outputURL(resources: self, kind: .connector))
        }
        let hasOfficialTextEncoder = fileManager.fileExists(atPath: textEncoderURL.path)
        let hasBundledQ4TextEncoder = LTX25TextEncoderQuantizedPack
            .optimizedIndexURLIfValid(resources: self, fileManager: fileManager) != nil
        if !hasOfficialTextEncoder, !hasBundledQ4TextEncoder {
            missing.append(textEncoderURL)
        }
        return missing
    }

    public func validateFull(fileManager: FileManager = .default) -> [URL] {
        let transformerPaths = Set([
            Self.devTransformerRelativePath,
            Self.distilledTransformerRelativePath,
        ])
        var missing = Self.fullRequiredRelativePaths.compactMap { relativePath -> URL? in
            guard !transformerPaths.contains(relativePath) else { return nil }
            let url = rootURL.appendingPathComponent(relativePath)
            return fileManager.fileExists(atPath: url.path) ? nil : url
        }
        missing.append(contentsOf: missingTransformerPaths(kinds: [.dev, .distilled], fileManager: fileManager))
        if !fileManager.fileExists(atPath: distilledTransformerURL.path),
           LTX25NativeModelPack.optimizedURLIfValid(
               resources: self,
               kind: .connector,
               fileManager: fileManager
           ) == nil
        {
            missing.append(LTX25NativeModelPack.outputURL(resources: self, kind: .connector))
        }
        return missing
    }

    private func missingTransformerPaths(
        kinds: [LTX25NativeModelPackKind],
        fileManager: FileManager
    ) -> [URL] {
        kinds.compactMap { kind in
            let sourceURL = kind == .dev ? devTransformerURL : distilledTransformerURL
            if fileManager.fileExists(atPath: sourceURL.path)
                || LTX25NativeModelPack.optimizedURLIfValid(
                    resources: self,
                    kind: kind,
                    fileManager: fileManager
                ) != nil
            {
                return nil
            }
            return LTX25NativeModelPack.outputURL(resources: self, kind: kind)
        }
    }
}

public func isLTX25ModelRoot(
    _ rootURL: URL,
    fileManager: FileManager = .default
) -> Bool {
    LTX25Resources(rootURL: rootURL).validate(fileManager: fileManager).isEmpty
}

public func isLTX25FullModelRoot(
    _ rootURL: URL,
    fileManager: FileManager = .default
) -> Bool {
    LTX25Resources(rootURL: rootURL).validateFull(fileManager: fileManager).isEmpty
}

func ltxStandaloneAudioVAEWeightsURL(
    modelRoot: URL,
    isLTX23: Bool,
    isLTX25: Bool,
    transformerURL: URL
) -> URL {
    if isLTX25 {
        return LTX25Resources(rootURL: modelRoot).audioVAEURL
    }
    if isLTX23 {
        return modelRoot.appendingPathComponent("audio_vae.safetensors", isDirectory: false)
    }
    return transformerURL
}
