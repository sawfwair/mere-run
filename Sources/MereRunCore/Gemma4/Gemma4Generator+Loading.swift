import Foundation
import MLX
import MLXNN

extension Gemma4Generator {
    func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let normalizedRoot = Gemma4Resources.normalizedRootURL(rootURL)
        if loadedModelPath == normalizedRoot.path, model != nil, tokenizerAndTemplate != nil {
            return
        }
        resetPrefixKVCache()

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
            maxLengthOverride: min(Gemma4Resources.defaultContextLength, config.textConfig.maxPositionEmbeddings)
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Gemma4 weights"))
        if Gemma4Resources.supportsVision(modelSpec: modelId) {
            let unifiedModel = try Gemma4UnifiedCausalLM(config: config)
            try Gemma4UnifiedModelLoader.loadWeights(into: unifiedModel, from: resources)
            model = unifiedModel
        } else {
            let textModel = Gemma4TextCausalLM(config: config.textConfig)
            try loadWeights(into: textModel, from: resources, config: config)
            model = textModel
        }
        let loadedMTP = try loadMTPAssistantIfAvailable(
            baseModelRoot: normalizedRoot,
            config: config,
            progressHandler: progressHandler
        )
        mtpModel = loadedMTP.model
        loadedMTPModelPath = loadedMTP.path
        tokenizerAndTemplate = tokenizer
        loadedConfig = config
        loadedModelPath = normalizedRoot.path
    }

    func applyTextLoRAIfNeeded(
        _ lora: LoRA?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        guard let lora else {
            loadedTextLoRASignature = nil
            return
        }
        let signature = Self.loraSignature(lora)
        guard loadedTextLoRASignature != signature else { return }
        guard let model else {
            throw Gemma4Error.modelNotLoaded
        }
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Gemma4 text LoRA"))
        _ = try await Gemma4TextLoRAAdapter.apply(lora, to: model)
        loadedTextLoRASignature = signature
        resetPrefixKVCache()
    }

    static func loraSignature(_ lora: LoRA?) -> String? {
        guard let lora else { return nil }
        switch lora {
        case .local(let path, let scale):
            return "local:\(URL(fileURLWithPath: path).standardizedFileURL.path):\(scale)"
        case .remote(let reference, let scale):
            return "remote:\(reference):\(scale)"
        }
    }

    func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
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

    func resolveInstalledGemmaRoot(for requestedModelId: String) -> URL? {
        func existingModelDir(_ id: String) -> URL? {
            let url = MereRunModelPaths.modelDir(id)
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
                return nil
            }
            guard (try? FileManager.default.contentsOfDirectoryResolvingSymlinks(at: url))?.isEmpty == false else {
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

    func resolveModelLocation(
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

    func resolveHubSnapshot(
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

    func loadMTPAssistantIfAvailable(
        baseModelRoot: URL,
        config: Gemma4Config,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws -> (model: Gemma4AssistantDraftModel?, path: String?) {
        guard Gemma4MTPPolicy.enabled() else {
            return (nil, nil)
        }
        guard supportsManagedMTPAssistant(baseModelRoot: baseModelRoot) else {
            return (nil, nil)
        }
        guard let assistantRoot = ManagedModelResolver.resolveInstalledModel(id: Gemma4MTPResources.modelId) else {
            return (nil, nil)
        }

        let resources = Gemma4MTPResources(rootURL: assistantRoot)
        let missing = resources.validate()
        guard missing.isEmpty else {
            return (nil, nil)
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Gemma4 MTP assistant"))
        let configData = try Data(contentsOf: resources.configURL)
        let assistantConfig = try JSONDecoder().decode(Gemma4AssistantConfig.self, from: configData)
        guard assistantConfig.backboneHiddenSize == config.textConfig.hiddenSize else {
            throw Gemma4Error.unsupportedConfiguration(
                "Gemma4 MTP assistant hidden size \(assistantConfig.backboneHiddenSize) does not match target hidden size \(config.textConfig.hiddenSize)."
            )
        }
        let assistant = try Gemma4AssistantDraftModel(config: assistantConfig)
        try assistant.loadWeights(from: resources)
        return (assistant, assistantRoot.path)
    }

    func supportsManagedMTPAssistant(baseModelRoot: URL) -> Bool {
        let normalizedModelId = modelId.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalizedModelId == Gemma4Resources.twelveBModelId
            || normalizedModelId == Gemma4Resources.twelveB4BitModelId
            || normalizedModelId == Gemma4Resources.visionTwelveBModelId {
            return true
        }
        let path = baseModelRoot.standardizedFileURL.path
        let textManagedRoot = MereRunModelPaths.modelDir(Gemma4Resources.twelveBModelId).standardizedFileURL.path
        let text4BitManagedRoot = MereRunModelPaths.modelDir(Gemma4Resources.twelveB4BitModelId).standardizedFileURL.path
        let visionManagedRoot = MereRunModelPaths.modelDir(Gemma4Resources.visionTwelveBModelId).standardizedFileURL.path
        return path == textManagedRoot || path == text4BitManagedRoot || path == visionManagedRoot
    }

    func loadWeights(
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
            if try Self.indexContainsQuantizedWeights(resources.modelIndexURL) {
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

    static func indexContainsQuantizedWeights(_ indexURL: URL) throws -> Bool {
        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
        return index.weightMap.keys.contains { $0.hasSuffix(".scales") }
    }
}
