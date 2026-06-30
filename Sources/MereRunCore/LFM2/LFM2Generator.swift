import Foundation
import MLX

private struct LFM2PrefillOutput {
    let logits: MLXArray
    let hidden: MLXArray
}

private struct LFM2DecodeResult {
    let generatedTokens: [Int]
    let decodeSeconds: Double
}

public actor LFM2Generator: ChatGenerator {
    private static let prefillChunkSize = 512

    private var model: LFM2Model?
    private var tokenizerAndTemplate: LFM2TokenizerAndTemplate?
    private var loadedModelPath: String?
    private var loadedConfig: LFM2Config?

    private let modelId: String

    public init(modelId: String = LFM2Resources.defaultModelId) {
        self.modelId = modelId
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

        var response = try await generate(request, progressHandler: progressHandler)
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
        let normalizedRoot = LFM2Resources.normalizedRootURL(rootURL)
        if loadedModelPath == normalizedRoot.path, model != nil, tokenizerAndTemplate != nil {
            return
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading LFM2 config"))
        let configData = try Data(contentsOf: normalizedRoot.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(LFM2Config.self, from: configData)

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading LFM2 tokenizer"))
        let tokenizer = try await LFM2TokenizerAndTemplate.load(
            from: normalizedRoot,
            maxLengthOverride: min(LFM2Resources.defaultContextLength, config.maxPositionEmbeddings)
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading LFM2 weights"))
        let lfm2Model = LFM2Model(config: config)
        let resources = LFM2Resources(rootURL: normalizedRoot)
        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 8

        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                indexURL: resources.modelIndexURL,
                to: lfm2Model,
                groupSize: groupSize,
                bits: bits,
                mapper: LFM2Resources.mapWeight(key:value:),
                progressHandler: { progress in
                    progressHandler?(ChatProgress(
                        stage: .loadingModel,
                        message: "Loading LFM2 shard \(progress.shardIndex + 1)/\(progress.shardCount)"
                    ))
                }
            )
        } else {
            let arrays = try MLX.loadArrays(url: resources.modelWeightsURL)
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                arrays,
                to: lfm2Model,
                groupSize: groupSize,
                bits: bits,
                mapper: LFM2Resources.mapWeight(key:value:)
            )
        }

        self.model = lfm2Model
        self.tokenizerAndTemplate = tokenizer
        self.loadedConfig = config
        self.loadedModelPath = normalizedRoot.path
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        guard let model,
              let tokenizerAndTemplate,
              let loadedConfig else {
            throw LFM2Error.modelNotLoaded
        }

        let requestedContextLength = request.maxContextTokens ?? LFM2Resources.defaultContextLength
        guard requestedContextLength > 0 else {
            throw LFM2Error.generationFailed("maxContextTokens must be greater than zero.")
        }
        let effectiveContext = min(
            LFM2Resources.defaultContextLength,
            requestedContextLength,
            loadedConfig.maxPositionEmbeddings
        )

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

        let eosSet = Set(
            loadedConfig.eosTokenIds
                + tokenizerAndTemplate.stopTokenIds(withTools: request.tools?.isEmpty == false)
        )
        let generationConfig = GenerationConfig(
            maxTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topP: Float(request.topP),
            repetitionPenalty: 1.05,
            repetitionContextSize: 64
        )

        let layerCaches = makeLayerCaches(config: loadedConfig)
        let prefillOutput = try await chunkedPrefill(
            model: model,
            promptTokens: promptTokens,
            cache: layerCaches,
            progressHandler: progressHandler
        )
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        let tokenBudget = max(0, min(request.maxTokens, effectiveContext - promptTokens.count))

        progressHandler?(ChatProgress(stage: .generating, message: ""))
        let decodeResult = try await decodeTokensSerially(
            model: model,
            tokenizerAndTemplate: tokenizerAndTemplate,
            initialLogits: prefillOutput.logits,
            layerCaches: layerCaches,
            eosSet: eosSet,
            generationConfig: generationConfig,
            tokenBudget: tokenBudget,
            promptTokens: promptTokens,
            progressHandler: progressHandler
        )

        let decoded = tokenizerAndTemplate.decode(tokens: decodeResult.generatedTokens)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let toolCalls: [ToolCall]? = request.tools?.isEmpty == false ? {
            let parsed = Gemma4ToolParser.parseToolCalls(decoded)
            return parsed.isEmpty ? nil : parsed
        }() : nil

        return ChatResponse(
            generatedText: decoded,
            tokensGenerated: decodeResult.generatedTokens.count,
            showThinking: request.showThinking,
            timing: ChatTiming(
                loadSeconds: 0,
                prefillSeconds: prefillSeconds,
                decodeSeconds: decodeResult.decodeSeconds
            ),
            toolCalls: toolCalls,
            promptTokens: promptTokens.count
        )
    }

    private func decodeTokensSerially(
        model: LFM2Model,
        tokenizerAndTemplate: LFM2TokenizerAndTemplate,
        initialLogits: MLXArray,
        layerCaches: [LFM2LayerCache?],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int],
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> LFM2DecodeResult {
        guard tokenBudget > 0 else {
            return LFM2DecodeResult(generatedTokens: [], decodeSeconds: 0)
        }

        var logits = initialLogits
        var generated: [Int] = []
        generated.reserveCapacity(tokenBudget)
        var repetitionHistory = promptTokens
        var pendingProgressWhitespace = ""
        let decodeStart = Date()

        func emit(_ token: Int) {
            generated.append(token)
            repetitionHistory.append(token)
            let piece = tokenizerAndTemplate.decode(token: token)
            guard !piece.isEmpty else { return }
            if piece.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                pendingProgressWhitespace += piece
                return
            }
            let visiblePiece = pendingProgressWhitespace.isEmpty
                ? piece
                : pendingProgressWhitespace + piece
            pendingProgressWhitespace = ""
            progressHandler?(ChatProgress(stage: .generating, message: visiblePiece))
        }

        while generated.count < tokenBudget {
            try Task.checkCancellation()
            let next = sampleToken(
                logits: logits[0, -1, 0...],
                config: generationConfig,
                previousTokens: repetitionHistory
            )
            if eosSet.contains(next) {
                break
            }
            emit(next)

            guard generated.count < tokenBudget else {
                break
            }
            let nextInput = MLXArray([Int32(next)]).reshaped(1, 1)
            logits = model(nextInput, cache: layerCaches)
            MLX.eval(logits)
        }

        return LFM2DecodeResult(
            generatedTokens: generated,
            decodeSeconds: Date().timeIntervalSince(decodeStart)
        )
    }

    private func chunkedPrefill(
        model: LFM2Model,
        promptTokens: [Int],
        cache: [LFM2LayerCache?],
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> LFM2PrefillOutput {
        guard !promptTokens.isEmpty else {
            throw LFM2Error.generationFailed("Prompt tokenization produced no tokens.")
        }

        var offset = 0
        var lastOutput: LFM2ForwardOutput?
        while offset < promptTokens.count {
            try Task.checkCancellation()
            let end = min(promptTokens.count, offset + Self.prefillChunkSize)
            let chunk = Array(promptTokens[offset..<end])
            let input = MLXArray(chunk.map(Int32.init)).reshaped(1, chunk.count)
            let output = model.forward(input, cache: cache)
            MLX.eval(output.logits)
            MLX.eval(output.hidden)
            lastOutput = output
            offset = end
            progressHandler?(ChatProgress(
                stage: .encoding,
                message: "Prefilled \(offset)/\(promptTokens.count) tokens"
            ))
            await Task.yield()
        }

        guard let lastOutput else {
            throw LFM2Error.generationFailed("LFM2 prefill produced no logits.")
        }
        return LFM2PrefillOutput(logits: lastOutput.logits, hidden: lastOutput.hidden)
    }

    private func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        let requestedModel = modelPath ?? modelId
        guard requestedModel == LFM2Resources.defaultModelId
            || requestedModel.caseInsensitiveCompare(LFM2Resources.upstreamRepoId) == .orderedSame
            || requestedModel.hasPrefix("/")
            || requestedModel.hasPrefix("~")
            || requestedModel.hasPrefix(".") else {
            throw LFM2Error.unsupportedModelId(requestedModel)
        }

        do {
            let root = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: requestedModel,
                defaultModelID: LFM2Resources.defaultModelId,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(ChatProgress(stage: .loadingModel, message: "Downloading model... \(percent)%"))
                    case .extracting:
                        progressHandler?(ChatProgress(stage: .loadingModel, message: "Extracting model..."))
                    }
                }
            )
            return LFM2Resources.normalizedRootURL(root.url)
        } catch let error as ManagedModelResolver.ResolverError {
            throw LFM2Error.downloadFailed(error.localizedDescription)
        }
    }

    private func makeLayerCaches(config: LFM2Config) -> [LFM2LayerCache?] {
        let attentionLayers = config.fullAttentionLayerIndexes
        return (0..<config.numHiddenLayers).map { layerIndex in
            if attentionLayers.contains(layerIndex) {
                return .attention(KVCacheSimple(step: 256))
            }
            return .conv(LFM2ConvCache())
        }
    }
}
