import Foundation

public enum ZImageTurboRepository {
    public static let id = "Tongyi-MAI/Z-Image-Turbo"
    public static let revision = "main"
    public static let baseRepoId = "Tongyi-MAI/Z-Image"
    public static let nanoRepoId = "filipstrand/Z-Image-Turbo-mflux-4bit"
    public static let defaultModelID: ModelResolver.ModelID = .zetaNano
    public static let defaultModelSpec = defaultModelID.rawValue
    public static let snapshotPatterns = [
        "model_index.json",
        "tokenizer/*",
        "text_encoder/*",
        "transformer/*",
        "vae/*",
        "scheduler/*",
    ]

    public static func hubFallbackConfig(for modelID: ModelResolver.ModelID) -> HubFallbackConfig? {
        switch modelID {
        case .zetaNano:
            return HubFallbackConfig(
                repoId: nanoRepoId,
                revision: revision,
                patterns: snapshotPatterns
            )
        case .zetaMax:
            return HubFallbackConfig(
                repoId: id,
                revision: revision,
                patterns: snapshotPatterns
            )
        case .zetaBase:
            return HubFallbackConfig(
                repoId: baseRepoId,
                revision: revision,
                patterns: snapshotPatterns
            )
        default:
            return nil
        }
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

    public var transformerMFluxWeightsIndexURL: URL {
        transformerDirURL.appending(path: "model.safetensors.index.json")
    }

    public var transformerWeightsURL: URL {
        transformerDirURL.appending(path: "diffusion_pytorch_model.safetensors")
    }

    public var transformerMFluxWeightsURL: URL {
        transformerDirURL.appending(path: "model.safetensors")
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

    public var vaeWeightsIndexURL: URL {
        vaeDirURL.appending(path: "model.safetensors.index.json")
    }

    public var vaeMFluxWeightsURL: URL {
        vaeDirURL.appending(path: "model.safetensors")
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
        let mfluxFormat = hasMFluxWeights(fileManager: fileManager)
        var urls: [URL] = [
            tokenizerJSONURL,
            tokenizerConfigURL,
            tokenizerMergesURL,
            tokenizerVocabURL,
        ]

        if !mfluxFormat {
            urls.append(contentsOf: [
                schedulerConfigURL,
                transformerConfigURL,
                textEncoderConfigURL,
                vaeConfigURL,
            ])
        }

        let transformerWeightsOK =
            fileManager.fileExists(atPath: transformerWeightsIndexURL.path)
            || fileManager.fileExists(atPath: transformerWeightsURL.path)
            || fileManager.fileExists(atPath: transformerMFluxWeightsIndexURL.path)
            || fileManager.fileExists(atPath: transformerMFluxWeightsURL.path)
            || Self.hasSafetensorsShards(in: transformerDirURL, fileManager: fileManager)
        if !transformerWeightsOK {
            urls.append(transformerWeightsIndexURL)
        }

        let textEncoderWeightsOK =
            fileManager.fileExists(atPath: textEncoderWeightsIndexURL.path)
            || fileManager.fileExists(atPath: textEncoderWeightsURL.path)
            || Self.hasSafetensorsShards(in: textEncoderDirURL, fileManager: fileManager)
        if !textEncoderWeightsOK {
            urls.append(textEncoderWeightsIndexURL)
        }

        let vaeWeightsOK =
            fileManager.fileExists(atPath: vaeWeightsURL.path)
            || fileManager.fileExists(atPath: vaeWeightsIndexURL.path)
            || fileManager.fileExists(atPath: vaeMFluxWeightsURL.path)
            || Self.hasSafetensorsShards(in: vaeDirURL, fileManager: fileManager)
        if !vaeWeightsOK {
            urls.append(vaeWeightsURL)
        }

        return urls.filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public func hasMFluxWeights(fileManager: FileManager = .default) -> Bool {
        (
            fileManager.fileExists(atPath: transformerMFluxWeightsIndexURL.path)
            || fileManager.fileExists(atPath: transformerMFluxWeightsURL.path)
        ) && (
            fileManager.fileExists(atPath: vaeWeightsIndexURL.path)
            || fileManager.fileExists(atPath: vaeMFluxWeightsURL.path)
        )
    }

    private static func hasSafetensorsShards(in directoryURL: URL, fileManager: FileManager) -> Bool {
        guard let urls = try? fileManager.contentsOfDirectoryResolvingSymlinks(
            at: directoryURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return false
        }
        return urls.contains { $0.pathExtension == "safetensors" }
    }

    private static func hasStaleManagedSource(
        in rootURL: URL,
        expectedRepoId: String,
        expectedRevision: String?,
        fileManager: FileManager
    ) -> Bool {
        guard let manifest = try? MereRunModelManifest.loadIfPresent(from: rootURL, fileManager: fileManager),
              let installed = manifest.upstreamRepoId else {
            return false
        }
        let expected = expectedRevision.map { "\(expectedRepoId)@\($0)" } ?? expectedRepoId
        return installed != expectedRepoId && installed != expected
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
            if modelID == .zetaNano,
               hasStaleManagedSource(
                in: rootURL,
                expectedRepoId: ZImageTurboRepository.nanoRepoId,
                expectedRevision: ZImageTurboRepository.revision,
                fileManager: fileManager
               ) {
                return [rootURL.appendingPathComponent("\(MereRunModelManifest.filename).upstream-mismatch")]
            }
            return ZImageTurboResources(rootURL: rootURL).validate(fileManager: fileManager)
        default:
            return ZImageTurboResources(rootURL: rootURL).validate(fileManager: fileManager)
        }
    }
}
