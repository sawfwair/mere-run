import Foundation
import MLX
import MereRunNemotronBoltModel

public actor NemotronBoltGenerator: ChatGenerator {
    private static let prefillChunkSize = 128

    private var model: NemotronBoltModel?
    private var tokenizer: Q35TokenizerAndTemplate?
    private var config: NemotronBoltConfig?
    private var loadedPath: String?

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
                throw NemotronBoltError.invalidCheckpoint("LoRA is not supported by this runtime")
            }
            guard !request.requiresJSON else {
                throw NemotronBoltError.invalidCheckpoint("constrained JSON is not supported")
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
        loadedPath = nil
        Memory.clearCache()
    }

    private func resolveRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        if let path = modelPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !path.isEmpty {
            let root = URL(fileURLWithPath: path).standardizedFileURL
            guard FileManager.default.fileExists(atPath: root.path) else {
                throw NemotronBoltError.invalidCheckpoint("Install vision-chat-nemotron-35-bolt-24gb before inference")
            }
            return root
        }
        if let installed = ManagedModelResolver.resolveInstalledModel(
            id: NemotronBoltResources.modelID
        ) {
            return installed.standardizedFileURL
        }
        throw NemotronBoltError.invalidCheckpoint("Install vision-chat-nemotron-35-bolt-24gb before inference")
    }

    private func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        guard loadedPath != rootURL.path || model == nil else { return }
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading native Nemotron 3.5 Bolt"))
        let loaded = try NemotronBoltModel(root: rootURL) { completed, total in
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Bolt shard \(completed)/\(total)"))
        }
        model = loaded
        config = loaded.config
        tokenizer = try Q35TokenizerAndTemplate.load(from: rootURL, maxLengthOverride: NemotronBoltResources.maximumContextLength)
        loadedPath = rootURL.path
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        guard let model, let tokenizer, let config, let loadedPath else {
            throw NemotronBoltError.invalidCheckpoint("model is not loaded")
        }
        let contextLength = min(
            request.maxContextTokens ?? NemotronBoltResources.defaultContextLength,
            NemotronBoltResources.maximumContextLength,
            config.language.maxPositionEmbeddings
        )
        guard request.messages.allSatisfy({ $0.audioUrl == nil && $0.videoUrl == nil }) else {
            throw NemotronBoltError.invalidCheckpoint("This Bolt runtime accepts text and images; audio and video inputs are unsupported")
        }
        let processor = try JSONDecoder().decode(NemotronOmniPreprocessorConfig.self,
            from: Data(contentsOf: URL(fileURLWithPath: loadedPath).appendingPathComponent("preprocessor_config.json")))
        var messages = request.messages
        var images: [NemotronBoltPreparedImage] = []
        for index in messages.indices {
            if let reference = messages[index].imageUrl, !reference.isEmpty {
                images.append(try NemotronBoltImageProcessor.prepare(reference: reference, config: processor, contextLength: contextLength))
                messages[index].content = "<image>\n" + messages[index].content
                messages[index].imageUrl = nil
            }
        }
        // Render at the model limit first. Truncating before expanding image
        // placeholders can drop a media span while retaining its encoder output.
        var promptTokens = try tokenizer.encodeForGeneration(
            messages: messages, tools: request.tools, addGenerationPrompt: true,
            includeThinking: request.showThinking, reasoningEffort: request.reasoningEffort.map { String($0) },
            maxLength: NemotronBoltResources.maximumContextLength
        )
        promptTokens = try NemotronOmniGenerator.expandImagePlaceholders(
            promptTokens, tokenCounts: images.map(\.languageTokenCount), imageTokenID: config.imageContextTokenID)
        if promptTokens.count > contextLength {
            guard images.isEmpty else { throw NemotronBoltError.invalidCheckpoint("Image prompt exceeds the requested context") }
            promptTokens = Array(promptTokens.suffix(contextLength))
        }
        guard !promptTokens.isEmpty else { throw NemotronBoltError.invalidCheckpoint("Empty prompt") }
        let targetCache = model.makeCache()
        var promptEmbeddings = model.inputEmbeddings(MLXArray(promptTokens.map(Int32.init), [1, promptTokens.count]))
        var replacements: [MLXArray] = []
        for image in images {
            let encoded = model.visionTower.encodeImage(image.pixelValues)
            MLX.eval(encoded)
            replacements.append(encoded)
        }
        promptEmbeddings = try NemotronOmniGenerator.replaceMediaEmbeddings(promptEmbeddings,
            promptTokens: promptTokens, placeholderTokenID: config.imageContextTokenID, replacements: replacements)
        let prefillStart = Date()
        var offset = 0
        var initialLogits: MLXArray?
        while offset < promptTokens.count {
            try Task.checkCancellation()
            let end = min(promptTokens.count, offset + Self.prefillChunkSize)
            let logits = model.prefill(embeddings: promptEmbeddings[0..., offset..<end, 0...], cache: targetCache)
            MLX.eval(logits)
            initialLogits = logits
            offset = end
            progressHandler?(ChatProgress(
                stage: .encoding,
                message: "Prefilled \(offset)/\(promptTokens.count) tokens"
            ))
            await Task.yield()
        }
        guard let initialLogits else {
            throw NemotronBoltError.invalidCheckpoint("prefill produced no logits")
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
        let eosTokens = Set([config.eosTokenID] + [tokenizer.eosTokenId].compactMap { $0 })
        progressHandler?(ChatProgress(stage: .generating, message: ""))
        let filtersToolProtocol = request.tools?.isEmpty == false
        let stopAtCompletedToolCall = filtersToolProtocol && !request.parallelToolCalls
        var toolCallCompletionDetector = Q35ToolParser.StreamingCompletionDetector()
        var visibleTextFilter = Q35ToolParser.StreamingVisibleTextFilter()

        func emitGeneratedPiece(_: Int, _ piece: String) {
            let visiblePiece = filtersToolProtocol ? visibleTextFilter.feed(piece) : piece
            if !visiblePiece.isEmpty {
                progressHandler?(ChatProgress(stage: .generating, message: visiblePiece))
            }
        }

        func shouldContinueAfterPiece(_: Int, _ piece: String) -> Bool {
            guard stopAtCompletedToolCall else { return true }
            return !toolCallCompletionDetector.feed(piece)
        }

        let decode = try AutoregressiveDecodeEngine.decode(
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
                emitPiece: emitGeneratedPiece,
                shouldContinue: shouldContinueAfterPiece,
                checkCancellation: { try Task.checkCancellation() }
            )
        if filtersToolProtocol {
            let remainingVisibleText = visibleTextFilter.finish()
            if !remainingVisibleText.isEmpty {
                progressHandler?(ChatProgress(stage: .generating, message: remainingVisibleText))
            }
        }
        let raw = tokenizer.decode(tokens: decode.generatedTokens)
        let trimmed = TextGenerationStopSequences.trimming(
            raw,
            sequences: TextGenerationStopSequences.merged(request.stopSequences)
        )
        let parsedToolCalls = request.tools?.isEmpty == false
            ? Q35ToolParser.parseToolCalls(trimmed.text)
            : []
        let toolCalls: [ToolCall]? = request.tools.flatMap { tools -> [ToolCall]? in
            let validated = ToolCallPolicy.validatedCalls(
                parsedToolCalls,
                tools: tools,
                parallelToolCalls: request.parallelToolCalls
            )
            return validated.isEmpty ? nil : validated
        }
        let visibleText = parsedToolCalls.isEmpty
            ? trimmed.text
            : Q35ToolParser.visibleText(trimmed.text)
        return ChatResponse(
            generatedText: visibleText,
            tokensGenerated: decode.generatedTokens.count,
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
            finishReason: decode.generatedTokens.count >= tokenBudget ? .length : .stop,
            logprobs: decode.logprobs,
            acceleration: ChatAccelerationDiagnostics(route: "nemotron-bolt-packed-native")
        )
    }
}
