import Foundation
import MLX

public actor Gemma4Generator: ChatGenerator {
    private var model: Gemma4TextCausalLM?
    private var tokenizerAndTemplate: Gemma4TokenizerAndTemplate?
    private var loadedModelPath: String?
    private var loadedConfig: Gemma4Config?

    private let modelId: String
    private let kvCacheQuantization: Gemma4KVCacheQuantization

    public init(
        modelId: String = Gemma4Resources.defaultModelId,
        kvCacheQuantization: Gemma4KVCacheQuantization = Gemma4KVCacheQuantization()
    ) {
        self.modelId = modelId
        self.kvCacheQuantization = kvCacheQuantization
    }

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        try await chat(request, modelPath: nil, progressHandler: progressHandler)
    }

    public func chat(
        _ request: ChatRequest,
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
        let loadStart = Date()
        try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
        let loadSeconds = Date().timeIntervalSince(loadStart)

        var response = try await generate(
            request,
            progressHandler: progressHandler,
            maxContextLength: Gemma4Resources.defaultContextLength
        )
        if var timing = response.timing {
            timing.loadSeconds = loadSeconds
            response.timing = timing
        } else {
            response.timing = ChatTiming(loadSeconds: loadSeconds)
        }
        return response
    }

    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws {
        let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
        try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
    }

    public func unload() {
        model = nil
        tokenizerAndTemplate = nil
        loadedModelPath = nil
        loadedConfig = nil
        Memory.clearCache()
    }

    private func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let normalizedRoot = Gemma4Resources.normalizedRootURL(rootURL)
        if loadedModelPath == normalizedRoot.path, model != nil, tokenizerAndTemplate != nil {
            return
        }

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
        let textModel = Gemma4TextCausalLM(config: config.textConfig)
        try loadWeights(into: textModel, from: resources, config: config)

        model = textModel
        tokenizerAndTemplate = tokenizer
        loadedConfig = config
        loadedModelPath = normalizedRoot.path
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        maxContextLength: Int
    ) async throws -> ChatResponse {
        guard let model, let tokenizerAndTemplate, let loadedConfig else {
            throw Gemma4Error.modelNotLoaded
        }
        let kvCacheQuantization = try self.kvCacheQuantization.validated()

        let effectiveContext = min(maxContextLength, loadedConfig.textConfig.maxPositionEmbeddings)
        let prefillStart = Date()
        var promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: request.messages,
            tools: request.tools,
            addGenerationPrompt: true,
            includeThinking: request.showThinking,
            maxLength: effectiveContext
        )
        if promptTokens.count > effectiveContext {
            promptTokens = Array(promptTokens.suffix(effectiveContext))
        }

        let hasTools = request.tools?.isEmpty == false
        let eosSet = Set(loadedConfig.eosTokenIds + tokenizerAndTemplate.stopTokenIds(withTools: hasTools))
        let generationConfig = GenerationConfig(
            maxTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topP: Float(request.topP),
            repetitionPenalty: 1.05,
            repetitionContextSize: 64
        )

        let layerCaches = model.makeCache(quantization: kvCacheQuantization)
        let promptInput = MLXArray(promptTokens.map(Int32.init)).reshaped(1, promptTokens.count)
        var logits = model(promptInput, cache: layerCaches)
        MLX.eval(logits)

        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        let tokenBudget = max(0, min(request.maxTokens, effectiveContext - promptTokens.count))

        progressHandler?(ChatProgress(stage: .generating, message: ""))

        var generated: [Int] = []
        generated.reserveCapacity(tokenBudget)
        var repetitionHistory = promptTokens
        let decodeStart = Date()

        for _ in 0..<tokenBudget {
            let next = sampleToken(
                logits: logits[0, -1, 0...],
                config: generationConfig,
                previousTokens: repetitionHistory
            )

            if eosSet.contains(next) {
                break
            }

            generated.append(next)
            repetitionHistory.append(next)
            let piece = tokenizerAndTemplate.decode(token: next)
            if !piece.isEmpty {
                progressHandler?(ChatProgress(stage: .generating, message: piece))
            }

            let nextInput = MLXArray([Int32(next)]).reshaped(1, 1)
            logits = model(nextInput, cache: layerCaches)
            MLX.eval(logits)
        }

        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        let decoded = tokenizerAndTemplate.decode(tokens: generated)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let toolCalls: [ToolCall]? = hasTools ? {
            let parsed = Gemma4ToolParser.parseToolCalls(decoded)
            return parsed.isEmpty ? nil : parsed
        }() : nil

        return ChatResponse(
            response: decoded,
            tokensGenerated: generated.count,
            timing: ChatTiming(
                loadSeconds: 0,
                prefillSeconds: prefillSeconds,
                decodeSeconds: decodeSeconds
            ),
            toolCalls: toolCalls
        )
    }

    private func resolveModelRoot(
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

    private func resolveInstalledGemmaRoot(for requestedModelId: String) -> URL? {
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

    private func resolveModelLocation(
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

    private func resolveHubSnapshot(
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

    private func loadWeights(
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
            if config.textConfig.enableMoEBlock {
                try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                    indexURL: resources.modelIndexURL,
                    to: model,
                    groupSize: 16,
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
}
