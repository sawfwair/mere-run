import Foundation

public struct Ideogram4Resources: Sendable, Hashable {
    public static let modelId = "image-ideogram4-sdnq-uint4"
    public static let upstreamRepoId = "WaveCut/ideogram-4-sdnq-uint4"
    public static let upstreamRevision = "main"
    public static let estimatedDownloadBytes: Int64 = 16 * 1_073_741_824

    public static let snapshotPatterns = [
        "ideogram4_sdnq_pipeline.py",
        "model_index.json",
        "quantization_manifest.json",
        "scheduler/scheduler_config.json",
        "text_encoder/*",
        "tokenizer/*",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "transformer/quantization_config.json",
        "unconditional_transformer/config.json",
        "unconditional_transformer/diffusion_pytorch_model.safetensors",
        "unconditional_transformer/quantization_config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "vae/quantization_config.json",
    ]

    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var modelIndexURL: URL { rootURL.appending(path: "model_index.json") }
    public var quantizationManifestURL: URL { rootURL.appending(path: "quantization_manifest.json") }
    public var tokenizerURL: URL { rootURL.appending(path: "tokenizer") }
    public var textEncoderURL: URL { rootURL.appending(path: "text_encoder") }
    public var transformerURL: URL { rootURL.appending(path: "transformer") }
    public var unconditionalTransformerURL: URL { rootURL.appending(path: "unconditional_transformer") }
    public var vaeURL: URL { rootURL.appending(path: "vae") }
    public var schedulerURL: URL { rootURL.appending(path: "scheduler") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        for file in [
            modelIndexURL,
            quantizationManifestURL,
            textEncoderURL.appending(path: "config.json"),
            transformerURL.appending(path: "config.json"),
            transformerURL.appending(path: "diffusion_pytorch_model.safetensors"),
            unconditionalTransformerURL.appending(path: "config.json"),
            unconditionalTransformerURL.appending(path: "diffusion_pytorch_model.safetensors"),
            vaeURL.appending(path: "config.json"),
            vaeURL.appending(path: "diffusion_pytorch_model.safetensors"),
            schedulerURL.appending(path: "scheduler_config.json"),
        ] where !fileManager.fileExists(atPath: file.path) {
            missing.append(file)
        }

        let tokenizerJSON = tokenizerURL.appending(path: "tokenizer.json")
        let tokenizerConfig = tokenizerURL.appending(path: "tokenizer_config.json")
        if !fileManager.fileExists(atPath: tokenizerJSON.path)
            && !fileManager.fileExists(atPath: tokenizerConfig.path) {
            missing.append(tokenizerConfig)
        }

        return missing
    }
}
