import Foundation

public enum ZImageTurboRepository {
    public static let id = "Tongyi-MAI/Z-Image-Turbo"
    public static let revision = "main"
    public static let defaultModelID: ModelResolver.ModelID = .zetaMax

    public static func hubFallbackConfig(for modelID: ModelResolver.ModelID) -> HubFallbackConfig? {
        guard modelID == .zetaMax else {
            return nil
        }

        return HubFallbackConfig(
            repoId: id,
            revision: revision,
            patterns: [
                "model_index.json",
                "tokenizer/*",
                "text_encoder/*",
                "transformer/*",
                "vae/*",
                "scheduler/*",
            ]
        )
    }

    public static func resolveModelID(from modelSpec: String) -> ModelResolver.ModelID? {
        let raw = modelSpec
            .split(separator: ":", maxSplits: 1)
            .first
            .map(String.init)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? ""
        guard !raw.isEmpty else {
            return nil
        }

        if let canonical = ModelResolver.ModelID(rawValue: raw),
           [.zetaNano, .zetaMax, .zetaBase].contains(canonical) {
            return canonical
        }

        switch raw {
        case id.lowercased(), "z-image-turbo", "z_image_turbo", "zimage-turbo", "zimage_turbo":
            return .zetaMax
        case "z-image-base", "z_image_base", "zimage-base", "zimage_base", "z-image":
            return .zetaBase
        case "z-image-nano", "z_image_nano", "zimage-nano", "zimage_nano":
            return .zetaNano
        default:
            return nil
        }
    }

    public static func resolveRemoteModelRoot(
        modelSpec: String,
        fileManager: FileManager = .default,
        progress: (@Sendable (PretrainedModelLoader.ProgressEvent) -> Void)? = nil
    ) async throws -> URL {
        guard let modelID = resolveModelID(from: modelSpec) else {
            throw PretrainedModelLoader.LoadError.unsupportedModelId(modelSpec)
        }
        guard let hubFallback = hubFallbackConfig(for: modelID) else {
            throw PretrainedModelLoader.LoadError.downloadFailed(
                ManagedModelCatalog.missingHubSourceMessage(for: modelID.rawValue)
            )
        }

        let root = try await PretrainedModelLoader.fromPretrainedSnapshot(
            modelPath: nil,
            modelId: modelID.rawValue,
            defaultModelIds: [modelID.rawValue],
            storageId: modelID.rawValue,
            hubFallback: hubFallback,
            fileManager: fileManager,
            validate: { root, manager in
                ZImageTurboResources.validateDownloadedRoot(
                    root,
                    modelID: modelID,
                    fileManager: manager
                )
            },
            progress: progress
        )

        _ = try MereRunModelManifest.writeTemplateIfKnown(modelId: modelID.rawValue, to: root)
        _ = try MereRunModelManifest.loadRequired(from: root)
        return root
    }
}

public struct ZImageTurboResources: Sendable, Hashable {
    /// Model root (where `mererun_model.json` may live).
    public var modelRootURL: URL

    public var tokenizerDirURL: URL
    public var textEncoderDirURL: URL
    public var transformerDirURL: URL
    public var vaeDirURL: URL
    public var schedulerDirURL: URL

    public init(rootURL: URL) {
        self.init(
            modelRootURL: rootURL,
            tokenizerDirURL: rootURL.appendingPathComponent("tokenizer", isDirectory: true),
            textEncoderDirURL: rootURL.appendingPathComponent("text_encoder", isDirectory: true),
            transformerDirURL: rootURL.appendingPathComponent("transformer", isDirectory: true),
            vaeDirURL: rootURL.appendingPathComponent("vae", isDirectory: true),
            schedulerDirURL: rootURL.appendingPathComponent("scheduler", isDirectory: true)
        )
    }

    public init(
        modelRootURL: URL,
        tokenizerDirURL: URL,
        textEncoderDirURL: URL,
        transformerDirURL: URL,
        vaeDirURL: URL,
        schedulerDirURL: URL
    ) {
        self.modelRootURL = modelRootURL
        self.tokenizerDirURL = tokenizerDirURL
        self.textEncoderDirURL = textEncoderDirURL
        self.transformerDirURL = transformerDirURL
        self.vaeDirURL = vaeDirURL
        self.schedulerDirURL = schedulerDirURL
    }

    public var modelIndexURL: URL {
        modelRootURL.appending(path: "model_index.json")
    }

    public var schedulerConfigURL: URL {
        schedulerDirURL.appending(path: "scheduler_config.json")
    }

    public var transformerConfigURL: URL {
        transformerDirURL.appending(path: "config.json")
    }

    public var transformerWeightsIndexURL: URL {
        transformerDirURL.appending(path: "diffusion_pytorch_model.safetensors.index.json")
    }

    public var transformerWeightsURL: URL {
        transformerDirURL.appending(path: "diffusion_pytorch_model.safetensors")
    }

    public var textEncoderConfigURL: URL {
        textEncoderDirURL.appending(path: "config.json")
    }

    public var textEncoderWeightsIndexURL: URL {
        textEncoderDirURL.appending(path: "model.safetensors.index.json")
    }

    public var textEncoderWeightsURL: URL {
        textEncoderDirURL.appending(path: "model.safetensors")
    }

    public var vaeConfigURL: URL {
        vaeDirURL.appending(path: "config.json")
    }

    public var vaeWeightsURL: URL {
        vaeDirURL.appending(path: "diffusion_pytorch_model.safetensors")
    }

    public var tokenizerJSONURL: URL {
        tokenizerDirURL.appending(path: "tokenizer.json")
    }

    public var tokenizerConfigURL: URL {
        tokenizerDirURL.appending(path: "tokenizer_config.json")
    }

    public var tokenizerMergesURL: URL {
        tokenizerDirURL.appending(path: "merges.txt")
    }

    public var tokenizerVocabURL: URL {
        tokenizerDirURL.appending(path: "vocab.json")
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var urls: [URL] = [
            schedulerConfigURL,
            transformerConfigURL,
            textEncoderConfigURL,
            vaeConfigURL,
            tokenizerJSONURL,
            tokenizerConfigURL,
            tokenizerMergesURL,
            tokenizerVocabURL,
        ]

        let transformerWeightsOK =
            fileManager.fileExists(atPath: transformerWeightsIndexURL.path)
            || fileManager.fileExists(atPath: transformerWeightsURL.path)
        if !transformerWeightsOK {
            urls.append(transformerWeightsIndexURL)
        }

        let textEncoderWeightsOK =
            fileManager.fileExists(atPath: textEncoderWeightsIndexURL.path)
            || fileManager.fileExists(atPath: textEncoderWeightsURL.path)
        if !textEncoderWeightsOK {
            urls.append(textEncoderWeightsIndexURL)
        }

        if !fileManager.fileExists(atPath: vaeWeightsURL.path) {
            urls.append(vaeWeightsURL)
        }

        return urls.filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public static func validateDownloadedRoot(
        _ rootURL: URL,
        modelID: ModelResolver.ModelID,
        fileManager: FileManager = .default
    ) -> [URL] {
        switch modelID {
        case .zetaBase:
            let transformerDir = rootURL.appendingPathComponent("transformer", isDirectory: true)
            let transformerConfigURL = transformerDir.appendingPathComponent("config.json")
            let transformerWeightsIndexURL = transformerDir
                .appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
            let transformerWeightsURL = transformerDir
                .appendingPathComponent("diffusion_pytorch_model.safetensors")

            var missing: [URL] = []
            if !fileManager.fileExists(atPath: transformerConfigURL.path) {
                missing.append(transformerConfigURL)
            }
            let hasTransformerWeights =
                fileManager.fileExists(atPath: transformerWeightsIndexURL.path)
                || fileManager.fileExists(atPath: transformerWeightsURL.path)
            if !hasTransformerWeights {
                missing.append(transformerWeightsIndexURL)
            }
            return missing
        case .zetaNano, .zetaMax:
            return ZImageTurboResources(rootURL: rootURL).validate(fileManager: fileManager)
        default:
            return ZImageTurboResources(rootURL: rootURL).validate(fileManager: fileManager)
        }
    }
}
