import Foundation
import MLX
@preconcurrency import Hub

public struct Gemma4LoadedTextModel: Sendable {
    public let model: Gemma4TextCausalLM
    public let tokenizerAndTemplate: Gemma4TokenizerAndTemplate
    public let config: Gemma4Config
    public let rootURL: URL

    public init(
        model: Gemma4TextCausalLM,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        config: Gemma4Config,
        rootURL: URL
    ) {
        self.model = model
        self.tokenizerAndTemplate = tokenizerAndTemplate
        self.config = config
        self.rootURL = rootURL
    }
}

public enum Gemma4TextModelLoader {
    public static func load(
        modelId: String,
        modelPath: String? = nil,
        maxContextLength: Int? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> Gemma4LoadedTextModel {
        guard !Gemma4Resources.supportsVision(modelSpec: modelId) else {
            throw Gemma4Error.unsupportedConfiguration("Native text LoRA training supports Gemma4 text models, not vision/unified Gemma4 models.")
        }

        let rootURL = try await resolveModelRoot(
            modelId: modelId,
            modelPath: modelPath,
            progressHandler: progressHandler
        )
        let normalizedRoot = Gemma4Resources.normalizedRootURL(rootURL)
        let resources = Gemma4Resources(rootURL: normalizedRoot)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw Gemma4Error.missingFiles(missing.map(\.lastPathComponent))
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Gemma4 config"))
        let configData = try Data(contentsOf: resources.configURL)
        let config = try JSONDecoder().decode(Gemma4Config.self, from: configData)

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Gemma4 tokenizer"))
        let tokenizer = try await Gemma4TokenizerAndTemplate.load(
            from: normalizedRoot,
            maxLengthOverride: min(
                maxContextLength ?? Gemma4Resources.defaultContextLength,
                config.textConfig.maxPositionEmbeddings
            )
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Gemma4 text weights"))
        let model = Gemma4TextCausalLM(config: config.textConfig)
        try loadWeights(into: model, from: resources, config: config)

        return Gemma4LoadedTextModel(
            model: model,
            tokenizerAndTemplate: tokenizer,
            config: config,
            rootURL: normalizedRoot
        )
    }

    public static func resolveModelRoot(
        modelId: String,
        modelPath: String? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> URL {
        if let explicit = modelPath?.trimmingCharacters(in: .whitespacesAndNewlines), !explicit.isEmpty {
            return try await resolveModelLocation(explicit, progressHandler: progressHandler)
        }

        let trimmedModelId = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        let requestedModelId = trimmedModelId.isEmpty ? Gemma4Resources.defaultModelId : trimmedModelId

        if let modelID = ModelResolver.ModelID(rawValue: requestedModelId),
           let resolved = ModelResolver().resolveIfPresent(modelID) {
            return resolved.rootURL
        }

        if let fallback = resolveInstalledGemmaRoot(for: requestedModelId) {
            return fallback
        }

        if requestedModelId != Gemma4Resources.defaultModelId {
            return try await resolveModelLocation(requestedModelId, progressHandler: progressHandler)
        }

        return try await resolveHubSnapshot(
            repoId: Gemma4Resources.defaultUpstreamModelId,
            progressHandler: progressHandler
        )
    }

    public static func loadWeights(
        into model: Gemma4TextCausalLM,
        from resources: Gemma4Resources,
        config: Gemma4Config
    ) throws {
        let include: (String) -> Bool = { key in
            key.hasPrefix("model.language_model.") || key.hasPrefix("language_model.")
        }
        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            if key.hasPrefix("model.language_model.") {
                return [(String(key.dropFirst("model.".count)), value)]
            }
            if key.hasPrefix("language_model.model.") {
                return [("language_model.\(key.dropFirst("language_model.model.".count))", value)]
            }
            if key.hasPrefix("language_model.") {
                return [(key, value)]
            }
            return []
        }
        let keyMapper: (String) -> String = { key in
            if key.hasPrefix("model.language_model.") {
                return String(key.dropFirst("model.".count))
            }
            if key.hasPrefix("language_model.model.") {
                return "language_model.\(key.dropFirst("language_model.model.".count))"
            }
            if key.hasPrefix("language_model.") {
                return key
            }
            return "__unused__.\(key)"
        }
        let quantizedMapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            key.hasPrefix("__unused__.") ? [] : [(key, value)]
        }
        let quantizedModuleResolver: HFSafetensorsWeightsLoader.QuantizedModuleResolver = { _, _, _, _, biases, fallbackGroupSize, fallbackBits in
            if biases != nil || fallbackBits > 4 {
                return (groupSize: fallbackGroupSize, bits: fallbackBits, mode: QuantizationMode.affine)
            }
            return (groupSize: fallbackGroupSize, bits: fallbackBits, mode: QuantizationMode.nvfp4)
        }

        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            if try indexContainsQuantizedWeights(resources.modelIndexURL) {
                try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                    indexURL: resources.modelIndexURL,
                    to: model,
                    groupSize: config.textConfig.enableMoEBlock ? 16 : 64,
                    bits: 4,
                    quantizedModuleResolver: quantizedModuleResolver,
                    keyMapper: keyMapper,
                    mapper: quantizedMapper
                )
            } else {
                try HFSafetensorsWeightsLoader.applyShardedWeights(
                    indexURL: resources.modelIndexURL,
                    to: model,
                    dtype: .bfloat16,
                    verify: .none,
                    mapper: mapper
                )
            }
        } else if FileManager.default.fileExists(atPath: resources.modelWeightsURL.path) {
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: resources.modelWeightsURL,
                to: model,
                dtype: .bfloat16,
                verify: .none,
                include: include,
                mapper: mapper,
                batchSize: 24
            )
        } else {
            throw Gemma4Error.missingFiles([resources.modelIndexURL.lastPathComponent, resources.modelWeightsURL.lastPathComponent])
        }
    }

    private static func resolveInstalledGemmaRoot(for requestedModelId: String) -> URL? {
        func existingModelDir(_ id: String) -> URL? {
            let url = MereRunModelPaths.modelDir(id)
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
                return nil
            }
            guard (try? FileManager.default.contentsOfDirectory(atPath: url.path))?.isEmpty == false else {
                return nil
            }
            return url.standardizedFileURL
        }

        if requestedModelId == Gemma4Resources.defaultModelId {
            return existingModelDir(Gemma4Resources.defaultModelId)
                ?? existingModelDir(Gemma4Resources.maxModelId)
                ?? existingModelDir(Gemma4Resources.nanoModelId)
        }

        return existingModelDir(requestedModelId)
    }

    private static func resolveModelLocation(
        _ location: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        if let modelID = ModelResolver.ModelID(rawValue: location),
           Gemma4Resources.handles(modelSpec: modelID.rawValue) {
            if let resolved = ModelResolver().resolveIfPresent(modelID) {
                return resolved.rootURL
            }
            do {
                let resolution = try await ManagedModelResolver.resolveForRuntime(
                    requestedModel: modelID.rawValue,
                    defaultModelID: modelID.rawValue,
                    progress: { event in
                        switch event {
                        case .downloading(let percent):
                            progressHandler?(ChatProgress(stage: .loadingModel, message: "Downloading Gemma4... \(percent)%"))
                        case .extracting:
                            progressHandler?(ChatProgress(stage: .loadingModel, message: "Extracting Gemma4..."))
                        }
                    }
                )
                return Gemma4Resources.normalizedRootURL(resolution.url)
            } catch {
                throw Gemma4Error.downloadFailed(error.localizedDescription)
            }
        }

        let fileURL = URL(fileURLWithPath: location).standardizedFileURL
        if FileManager.default.fileExists(atPath: fileURL.path) {
            return Gemma4Resources.normalizedRootURL(fileURL)
        }
        if Gemma4Resources.isLikelyHubRepoID(location) {
            return try await resolveHubSnapshot(repoId: location, progressHandler: progressHandler)
        }
        throw Gemma4Error.unsupportedModelLocation(location)
    }

    private static func resolveHubSnapshot(
        repoId: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        let snapshot = try HubSnapshot(options: HubSnapshotOptions(
            repoId: repoId,
            patterns: Gemma4Resources.snapshotPatterns
        ))
        do {
            return try await snapshot.prepare { progress in
                let percent = Int((progress.fractionCompleted * 100).rounded())
                progressHandler?(ChatProgress(stage: .loadingModel, message: "Downloading Gemma4... \(percent)%"))
            }
        } catch {
            throw Gemma4Error.downloadFailed(error.localizedDescription)
        }
    }

    private static func indexContainsQuantizedWeights(_ indexURL: URL) throws -> Bool {
        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
        return index.weightMap.keys.contains { $0.hasSuffix(".scales") }
    }
}
