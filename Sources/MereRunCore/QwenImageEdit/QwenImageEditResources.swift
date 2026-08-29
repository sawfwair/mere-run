import Foundation

public enum QwenImageEditRepository {
    public enum RepositoryError: LocalizedError {
        case lightningManagedInstallRequired

        public var errorDescription: String? {
            switch self {
            case .lightningManagedInstallRequired:
                return "Qwen Image Edit 2511 Lightning requires its pinned base and adapter; "
                    + "install it with `mere.run model pull \(lightning2511Id)`."
            }
        }
    }

    /// Legacy Qwen Image Edit identity. Keep this source mutable only for compatibility with
    /// existing installs; 2511 uses separate immutable managed identities below.
    public static let modelId = "qwen-image-edit"
    public static let id = "Qwen/Qwen-Image-Edit"
    public static let revision = "main"
    public static let model2511Id = "image-qwen-edit-2511"
    public static let lightning2511Id = "image-qwen-edit-2511-lightning"
    public static let id2511 = "Qwen/Qwen-Image-Edit-2511"
    public static let revision2511 = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
    public static let lightningRepoId = "lightx2v/Qwen-Image-Edit-2511-Lightning"
    public static let lightningRevision = "d74eba145674fd7e31b949324e148e21e7118abd"
    public static let lightningFilename =
        "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    public static let lightningRelativePath = "lightning/\(lightningFilename)"
    public static let lightningSHA256 =
        "22226e8d05d354bb356627d428809f5afd7819399b077238a2b70a82883a904f"
    public static let lightningByteCount: Int64 = 849_608_296
    public static let lightningPin = ModelArtifactPin(
        filename: lightningRelativePath,
        byteCount: lightningByteCount,
        sha256: lightningSHA256
    )

    public static let hubFallbackConfig = HubFallbackConfig(
        repoId: id,
        revision: revision,
        patterns: [
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "transformer/*",
            "vae/*",
        ]
    )
    public static let hubFallback2511Config = HubFallbackConfig(
        repoId: id2511,
        revision: revision2511,
        patterns: hubFallbackConfig.patterns
    )
    public static let lightningHubFallbackConfig = HubFallbackConfig(
        repoId: lightningRepoId,
        revision: lightningRevision,
        patterns: [lightningFilename]
    )

    public static func canonicalModelId(for modelSpec: String) -> String? {
        let raw = modelSpec
            .split(separator: ":", maxSplits: 1)
            .first
            .map(String.init)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? ""
        guard !raw.isEmpty else {
            return nil
        }

        switch raw {
        case modelId, id.lowercased():
            return modelId
        case model2511Id, id2511.lowercased():
            return model2511Id
        case lightning2511Id:
            return lightning2511Id
        default:
            return nil
        }
    }

    public static func resolveModelRoot(
        modelSpec: String,
        fileManager: FileManager = .default,
        progress: (@Sendable (PretrainedModelLoader.ProgressEvent) -> Void)? = nil
    ) async throws -> URL {
        if let local = PretrainedModelLoader.resolveProvidedOrLocalRoot(
            modelPath: nil,
            modelId: modelSpec,
            fileManager: fileManager,
            normalize: { base, manager in
                resolveNestedIfNeeded(base: base, fileManager: manager)
            }
        ) {
            return local
        }

        guard let canonicalId = canonicalModelId(for: modelSpec) else {
            throw PretrainedModelLoader.LoadError.unsupportedModelId(modelSpec)
        }
        if canonicalId == lightning2511Id {
            throw RepositoryError.lightningManagedInstallRequired
        }

        let source = source(for: canonicalId)
        return try await PretrainedModelLoader.fromPretrainedSnapshot(
            modelPath: nil,
            modelId: canonicalId,
            defaultModelIds: [canonicalId],
            storageId: canonicalId,
            hubFallback: source,
            fileManager: fileManager,
            normalize: { base, manager in
                resolveNestedIfNeeded(base: base, fileManager: manager)
            },
            validate: { root, manager in
                QwenImageEditResources(rootURL: root).validate(fileManager: manager)
            },
            progress: progress
        )
    }

    public static func resolveInstalledModelRoot(
        modelSpec: String = modelId,
        fileManager: FileManager = .default
    ) -> URL? {
        guard let canonicalId = canonicalModelId(for: modelSpec) else {
            return nil
        }
        let managed = PretrainedModelLoader.resolveManagedRoot(
            storageId: canonicalId,
            fileManager: fileManager,
            normalize: { base, manager in
                resolveNestedIfNeeded(base: base, fileManager: manager)
            }
        ) { root, manager in
            QwenImageEditResources(rootURL: root).validate(fileManager: manager).isEmpty
        }

        guard managed.isComplete else {
            return nil
        }
        return managed.resolvedRoot
    }

    public static func isLightningModel(_ modelSpec: String) -> Bool {
        canonicalModelId(for: modelSpec) == lightning2511Id
    }

    private static func source(for canonicalId: String) -> HubFallbackConfig {
        switch canonicalId {
        case model2511Id, lightning2511Id:
            return hubFallback2511Config
        default:
            return hubFallbackConfig
        }
    }

    private static func resolveNestedIfNeeded(
        base: URL,
        fileManager: FileManager
    ) -> URL {
        if QwenImageEditResources(rootURL: base).validate(fileManager: fileManager).isEmpty {
            return base
        }

        guard let children = try? fileManager.contentsOfDirectoryResolvingSymlinks(
            at: base,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return base
        }

        let childDirs = children.filter { child in
            var isDirectory: ObjCBool = false
            return fileManager.fileExists(atPath: child.path, isDirectory: &isDirectory)
                && isDirectory.boolValue
        }

        guard childDirs.count == 1 else {
            return base
        }

        let nested = childDirs[0]
        if QwenImageEditResources(rootURL: nested).validate(fileManager: fileManager).isEmpty {
            return nested
        }
        return base
    }
}

public struct QwenImageEditResources: Sendable, Hashable {
    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    // MARK: - Model Index

    public var modelIndexURL: URL {
        rootURL.appending(path: "model_index.json")
    }

    // MARK: - Scheduler

    public var schedulerConfigURL: URL {
        rootURL.appending(path: "scheduler/scheduler_config.json")
    }

    // MARK: - Transformer (MMDiT)

    public var transformerConfigURL: URL {
        rootURL.appending(path: "transformer/config.json")
    }

    public var transformerWeightsIndexURL: URL {
        rootURL.appending(path: "transformer/diffusion_pytorch_model.safetensors.index.json")
    }

    public var transformerWeightsURL: URL {
        rootURL.appending(path: "transformer/diffusion_pytorch_model.safetensors")
    }

    public var lightningWeightsURL: URL {
        rootURL.appending(path: QwenImageEditRepository.lightningRelativePath)
    }

    // MARK: - Text Encoder (Qwen2.5-VL)

    public var textEncoderConfigURL: URL {
        rootURL.appending(path: "text_encoder/config.json")
    }

    public var textEncoderWeightsIndexURL: URL {
        rootURL.appending(path: "text_encoder/model.safetensors.index.json")
    }

    public var textEncoderWeightsURL: URL {
        rootURL.appending(path: "text_encoder/model.safetensors")
    }

    // MARK: - VAE

    public var vaeConfigURL: URL {
        rootURL.appending(path: "vae/config.json")
    }

    public var vaeWeightsURL: URL {
        rootURL.appending(path: "vae/diffusion_pytorch_model.safetensors")
    }

    // MARK: - Tokenizer

    public var tokenizerJSONURL: URL {
        rootURL.appending(path: "tokenizer/tokenizer.json")
    }

    public var tokenizerConfigURL: URL {
        rootURL.appending(path: "tokenizer/tokenizer_config.json")
    }

    public var tokenizerMergesURL: URL {
        rootURL.appending(path: "tokenizer/merges.txt")
    }

    public var tokenizerVocabURL: URL {
        rootURL.appending(path: "tokenizer/vocab.json")
    }

    // MARK: - Processor (for vision-language)

    public var processorConfigURL: URL {
        rootURL.appending(path: "processor/preprocessor_config.json")
    }

    // MARK: - Validation

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []

        // Required config files
        let requiredConfigs = [
            modelIndexURL,
            schedulerConfigURL,
            transformerConfigURL,
            textEncoderConfigURL,
            vaeConfigURL,
            tokenizerConfigURL,
        ]
        for url in requiredConfigs {
            if !fileManager.fileExists(atPath: url.path) {
                missing.append(url)
            }
        }

        // Tokenizer: either tokenizer.json OR (vocab.json + merges.txt)
        let hasTokenizerJSON = fileManager.fileExists(atPath: tokenizerJSONURL.path)
        let hasBPEFiles = fileManager.fileExists(atPath: tokenizerVocabURL.path)
            && fileManager.fileExists(atPath: tokenizerMergesURL.path)
        if !hasTokenizerJSON && !hasBPEFiles {
            missing.append(tokenizerJSONURL)  // Report tokenizer.json as missing
        }

        // Transformer weights: index.json OR single file OR quantized model.safetensors
        let quantizedTransformerURL = rootURL.appending(path: "transformer/model.safetensors")
        let transformerWeightsOK =
            fileManager.fileExists(atPath: transformerWeightsIndexURL.path)
            || fileManager.fileExists(atPath: transformerWeightsURL.path)
            || fileManager.fileExists(atPath: quantizedTransformerURL.path)
        if !transformerWeightsOK {
            missing.append(transformerWeightsIndexURL)
        }

        // Text encoder weights: index.json OR single file OR quantized model.safetensors
        let quantizedEncoderURL = rootURL.appending(path: "text_encoder/model.safetensors")
        let textEncoderWeightsOK =
            fileManager.fileExists(atPath: textEncoderWeightsIndexURL.path)
            || fileManager.fileExists(atPath: textEncoderWeightsURL.path)
            || fileManager.fileExists(atPath: quantizedEncoderURL.path)
        if !textEncoderWeightsOK {
            missing.append(textEncoderWeightsIndexURL)
        }

        // VAE weights
        if !fileManager.fileExists(atPath: vaeWeightsURL.path) {
            missing.append(vaeWeightsURL)
        }

        return missing
    }
}
