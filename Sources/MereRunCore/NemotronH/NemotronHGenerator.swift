import Foundation
import MLX

private struct NemotronHDecodeResult {
    let tokens: [Int]
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
    let logprobs: ChatLogprobDiagnostics?
    let acceleration: ChatAccelerationDiagnostics
}

public actor NemotronHGenerator: ChatGenerator {
    private static let prefillChunkSize = 128

    private var model: NemotronHCausalLM?
    private var tokenizer: Q35TokenizerAndTemplate?
    private var config: NemotronHConfig?
    private var dspark: NemotronHDSparkModel?
    private var loadedPath: String?
    private var latestDSparkStats = NemotronHDSparkStats()

    public init() {}

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
        try await Stream.withNewDefaultStream {
            guard request.lora == nil else {
                throw NemotronHError.generationFailed("LoRA is not supported by this runtime yet")
            }
            guard !request.requiresJSON else {
                throw NemotronHError.generationFailed("constrained JSON is not supported yet")
            }
            let root = try await resolveRoot(
                modelPath: modelPath,
                progressHandler: progressHandler
            )
            let loadStart = Date()
            try await ensureLoaded(rootURL: root, progressHandler: progressHandler)
            let loadSeconds = Date().timeIntervalSince(loadStart)
            var response = try await generate(request, progressHandler: progressHandler)
            response.timing?.loadSeconds = loadSeconds
            return response
        }
    }

    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws {
        try await Stream.withNewDefaultStream {
            let root = try await resolveRoot(
                modelPath: modelPath,
                progressHandler: progressHandler
            )
            try await ensureLoaded(rootURL: root, progressHandler: progressHandler)
        }
    }

    public func unload() {
        model = nil
        tokenizer = nil
        config = nil
        dspark = nil
        loadedPath = nil
        Memory.clearCache()
    }

    public func dsparkStats() -> NemotronHDSparkStats {
        latestDSparkStats
    }

    private func resolveRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        if let path = modelPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !path.isEmpty {
            let root = URL(fileURLWithPath: path).standardizedFileURL
            guard FileManager.default.fileExists(atPath: root.path) else {
                throw NemotronHError.modelPathRequired
            }
            return root
        }
        if let installed = ManagedModelResolver.resolveInstalledModel(
            id: NemotronHResources.modelID
        ) {
            return installed.standardizedFileURL
        }
        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: NemotronHResources.modelID,
            defaultModelID: NemotronHResources.modelID,
            progress: { event in
                if case .downloading(let percent) = event {
                    progressHandler?(ChatProgress(
                        stage: .loadingModel,
                        message: "Downloading Nemotron... \(percent)%"
                    ))
                }
            }
        )
        return resolution.url.standardizedFileURL
    }

    private func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        guard loadedPath != rootURL.path || model == nil else { return }
        let loaded = try await NemotronHModelLoader.load(
            rootURL: rootURL,
            dsparkPath: NemotronHResources.installedDSparkPath(),
            maxContextLength: NemotronHResources.maximumContextLength,
            progressHandler: progressHandler
        )
        model = loaded.model
        tokenizer = loaded.tokenizer
        config = loaded.config
        dspark = loaded.dspark
        loadedPath = loaded.rootURL.path
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        guard let model, let tokenizer, let config else {
            throw NemotronHError.generationFailed("model is not loaded")
        }
        let contextLength = min(
            request.maxContextTokens ?? NemotronHResources.defaultContextLength,
            NemotronHResources.maximumContextLength,
            config.maxPositionEmbeddings
        )
        var promptTokens = try tokenizer.encodeForGeneration(
            messages: request.messages,
            tools: request.tools,
            addGenerationPrompt: true,
            includeThinking: request.showThinking,
            maxLength: contextLength
        )
        if promptTokens.count > contextLength {
            promptTokens = Array(promptTokens.suffix(contextLength))
        }
        guard !promptTokens.isEmpty else {
            throw NemotronHError.generationFailed("prompt tokenization produced no tokens")
        }
        let targetCache = model.makeCache()
        let captureIndices = request.logprobCapture.isEnabled
            ? []
            : dspark.map { Set($0.config.speculation.targetLayerIDs) } ?? []
        let draftCache = request.logprobCapture.isEnabled ? nil : dspark?.makeCache()
        let prefillStart = Date()
        var offset = 0
        var initialLogits: MLXArray?
        while offset < promptTokens.count {
            try Task.checkCancellation()
            let end = min(promptTokens.count, offset + Self.prefillChunkSize)
            let chunk = Array(promptTokens[offset..<end])
            let output = model.prefill(
                MLXArray(chunk.map(Int32.init)).reshaped(1, chunk.count),
                cache: targetCache,
                captureLayerIndices: captureIndices
            )
            if let dspark, let draftCache {
                dspark.appendTargetContext(
                    dspark.combineTargetHiddenStates(output.capturedHiddenStates),
                    cache: draftCache
                )
            }
            MLX.eval(output.logits)
            initialLogits = output.logits
            offset = end
            progressHandler?(ChatProgress(
                stage: .encoding,
                message: "Prefilled \(offset)/\(promptTokens.count) tokens"
            ))
            await Task.yield()
        }
        guard let initialLogits else {
            throw NemotronHError.generationFailed("prefill produced no logits")
        }
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        let tokenBudget = max(0, min(request.maxTokens, contextLength - promptTokens.count))
        let generationConfig = GenerationConfig(
            maxTokens: tokenBudget,
            temperature: Float(request.temperature),
            topK: request.topK ?? 0,
            topP: Float(request.topP),
            minP: Float(request.minP),
            repetitionPenalty: 1.05,
            repetitionContextSize: 64
        )
        let eosTokens = Set(config.eosTokenIDs + [2, 11] + [tokenizer.eosTokenId].compactMap { $0 })
        progressHandler?(ChatProgress(stage: .generating, message: ""))

        let decode: NemotronHDecodeResult
        if let dspark,
           let draftCache,
           !request.logprobCapture.isEnabled,
           NemotronHDSparkPolicy.enabled(),
           tokenBudget >= NemotronHDSparkPolicy.minimumOutputTokens() {
            let result = try NemotronHDSparkDecoder.decode(
                initialLogits: initialLogits,
                target: model,
                targetCache: targetCache,
                dspark: dspark,
                draftCache: draftCache,
                generationConfig: generationConfig,
                eosTokens: eosTokens,
                tokenBudget: tokenBudget,
                historySeedTokens: promptTokens,
                speculativeTokens: NemotronHResources.defaultSpeculativeTokens,
                decodeToken: { tokenizer.decode(token: $0) },
                emitPiece: { _, piece in
                    progressHandler?(ChatProgress(stage: .generating, message: piece))
                },
                checkCancellation: { try Task.checkCancellation() }
            )
            latestDSparkStats = result.stats
            decode = NemotronHDecodeResult(
                tokens: result.generatedTokens,
                decodeSeconds: result.decodeSeconds,
                firstTokenSeconds: result.firstTokenSeconds,
                logprobs: nil,
                acceleration: ChatAccelerationDiagnostics(
                    route: "dspark-speculative",
                    draftModel: NemotronHResources.dsparkModelID,
                    rounds: result.stats.rounds,
                    draftedTokens: result.stats.draftedTokens,
                    acceptedDraftTokens: result.stats.acceptedDraftTokens
                )
            )
        } else {
            latestDSparkStats = NemotronHDSparkStats(
                enabled: dspark != nil,
                active: false,
                speculativeTokens: NemotronHResources.defaultSpeculativeTokens,
                reason: request.logprobCapture.isEnabled
                    ? "logprob capture requires final-target decode"
                    : dspark == nil ? "companion model is not installed" : nil
            )
            let result = try AutoregressiveDecodeEngine.decode(
                AutoregressiveDecodeRequest(
                    initialLogits: initialLogits,
                    generationConfig: generationConfig,
                    eosTokens: eosTokens,
                    tokenBudget: tokenBudget,
                    historySeedTokens: promptTokens,
                    logprobCapture: request.logprobCapture,
                    logprobRegion: request.logprobRegionHint ?? .visible
                ),
                stepForward: { token in model.lastPositionLogits(token, cache: targetCache) },
                decodeToken: { tokenizer.decode(token: $0) },
                emitPiece: { _, piece in
                    progressHandler?(ChatProgress(stage: .generating, message: piece))
                },
                checkCancellation: { try Task.checkCancellation() }
            )
            decode = NemotronHDecodeResult(
                tokens: result.generatedTokens,
                decodeSeconds: result.decodeSeconds,
                firstTokenSeconds: result.firstTokenSeconds,
                logprobs: result.logprobs,
                acceleration: ChatAccelerationDiagnostics(route: "final-target-pipelined")
            )
        }
        let raw = tokenizer.decode(tokens: decode.tokens)
        let trimmed = TextGenerationStopSequences.trimming(
            raw,
            sequences: TextGenerationStopSequences.merged(request.stopSequences)
        )
        let toolCalls: [ToolCall]? = request.tools?.isEmpty == false ? {
            let parsed = Gemma4ToolParser.parseToolCalls(trimmed.text)
            return parsed.isEmpty ? nil : parsed
        }() : nil
        return ChatResponse(
            generatedText: trimmed.text,
            tokensGenerated: decode.tokens.count,
            showThinking: request.showThinking,
            timing: ChatTiming(
                prefillSeconds: prefillSeconds,
                decodeSeconds: decode.decodeSeconds,
                firstTokenSeconds: decode.firstTokenSeconds,
                kvCacheMode: .default,
                prefillKVCache: "hybrid-fp32-ssm-bf16-kv",
                decodeKVCache: "hybrid-fp32-ssm-bf16-kv"
            ),
            toolCalls: toolCalls,
            promptTokens: promptTokens.count,
            finishReason: decode.tokens.count >= tokenBudget ? .length : .stop,
            logprobs: decode.logprobs,
            acceleration: decode.acceleration
        )
    }
}
