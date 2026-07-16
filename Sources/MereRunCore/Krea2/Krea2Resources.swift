import Foundation

public struct Krea2Resources: Sendable, Hashable {
    public static let modelId = "image-krea2-turbo"
    public static let upstreamRepoId = "krea/Krea-2-Turbo"
    public static let upstreamRevision = "1161245028ef398cd0a951101b2bbf486464f841"
    public static let estimatedDownloadBytes: Int64 = 36 * 1_073_741_824

    /// Krea publishes both split Diffusers component weights and a root-level
    /// `turbo.safetensors` copy of the transformer. Pull only the component
    /// layout so managed installs avoid downloading the same 26 GB twice.
    public static let snapshotPatterns = [
        "LICENSE.pdf",
        "README.md",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "tokenizer/chat_template.jinja",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "transformer/diffusion_pytorch_model-*.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]

    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var modelIndexURL: URL { rootURL.appending(path: "model_index.json") }
    public var schedulerURL: URL { rootURL.appending(path: "scheduler") }
    public var textEncoderURL: URL { rootURL.appending(path: "text_encoder") }
    public var tokenizerURL: URL { rootURL.appending(path: "tokenizer") }
    public var transformerURL: URL { rootURL.appending(path: "transformer") }
    public var vaeURL: URL { rootURL.appending(path: "vae") }
    public var schedulerConfigURL: URL { schedulerURL.appending(path: "scheduler_config.json") }
    public var textEncoderConfigURL: URL { textEncoderURL.appending(path: "config.json") }
    public var textEncoderWeightsURL: URL { textEncoderURL.appending(path: "model.safetensors") }
    public var transformerConfigURL: URL { transformerURL.appending(path: "config.json") }
    public var transformerWeightsIndexURL: URL {
        transformerURL.appending(path: "diffusion_pytorch_model.safetensors.index.json")
    }
    public var vaeConfigURL: URL { vaeURL.appending(path: "config.json") }
    public var vaeWeightsURL: URL { vaeURL.appending(path: "diffusion_pytorch_model.safetensors") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        for file in [
            modelIndexURL,
            schedulerConfigURL,
            textEncoderConfigURL,
            textEncoderWeightsURL,
            tokenizerURL.appending(path: "tokenizer.json"),
            tokenizerURL.appending(path: "tokenizer_config.json"),
            transformerConfigURL,
            transformerWeightsIndexURL,
            vaeConfigURL,
            vaeWeightsURL,
        ] where !fileManager.fileExists(atPath: file.path) {
            missing.append(file)
        }

        if !hasTransformerShard(fileManager: fileManager) {
            missing.append(transformerURL.appending(path: "diffusion_pytorch_model-00001-of-00003.safetensors"))
        }

        return missing
    }

    private func hasTransformerShard(fileManager: FileManager) -> Bool {
        let transformerURL = transformerURL.resolvingSymlinksInPath()
        guard let children = try? fileManager.contentsOfDirectory(
            at: transformerURL,
            includingPropertiesForKeys: nil
        ) else {
            return false
        }
        return children.contains { url in
            let name = url.lastPathComponent
            return name.hasPrefix("diffusion_pytorch_model-")
                && name.hasSuffix(".safetensors")
        }
    }
}
