import Foundation

public enum QwenImageEditRepository {
    public static let modelId = "qwen-image-edit"
    public static let id = "Qwen/Qwen-Image-Edit"
    public static let revision = "main"
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

        return try await PretrainedModelLoader.fromPretrainedSnapshot(
            modelPath: nil,
            modelId: canonicalId,
            defaultModelIds: [modelId],
            storageId: modelId,
            hubFallback: hubFallbackConfig,
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
        fileManager: FileManager = .default
    ) -> URL? {
        let managed = PretrainedModelLoader.resolveManagedRoot(
            storageId: modelId,
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
