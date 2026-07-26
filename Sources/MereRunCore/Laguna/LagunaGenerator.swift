import Foundation
import MLX

public enum LagunaError: LocalizedError {
    case modelPathRequired
    case missingFiles([String])
    case modelNotLoaded
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelPathRequired:
            return "Laguna evaluation requires an explicit local MLX checkpoint path."
        case .missingFiles(let files):
            return "Laguna checkpoint is missing required files: \(files.joined(separator: ", "))."
        case .modelNotLoaded:
            return "Laguna model is not loaded."
        case .generationFailed(let message):
            return "Laguna generation failed: \(message)"
        }
    }
}

public enum LagunaResources {
    public static let modelID = "poolside/Laguna-S-2.1-NVFP4-mlx"
    public static let defaultContextLength = 32_768

    static let requiredFiles = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    ]

    static func validate(rootURL: URL) -> [String] {
        requiredFiles.filter {
            !FileManager.default.fileExists(atPath: rootURL.appending(path: $0).path)
        }
    }
}

public actor LagunaGenerator: ChatGenerator {
    private var model: LagunaCausalLM?
    private var tokenizerAndTemplate: LagunaTokenizerAndTemplate?
    private var config: LagunaConfig?
    private var loadedModelPath: String?

    public init() {}

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        throw LagunaError.modelPathRequired
    }

    public func chat(
        _ request: ChatRequest,
        modelPath: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> ChatResponse {
        try await Stream.withNewDefaultStream {
            let rootURL = URL(fileURLWithPath: modelPath).standardizedFileURL
            let loadStart = Date()
            try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
            let loadSeconds = Date().timeIntervalSince(loadStart)

            var response = try generate(request, progressHandler: progressHandler)
            if var timing = response.timing {
                timing.loadSeconds = loadSeconds
                response.timing = timing
            } else {
                response.timing = ChatTiming(loadSeconds: loadSeconds)
            }
            return response
        }
    }

    public func prepare(
        modelPath: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws {
        try await Stream.withNewDefaultStream {
            try await ensureLoaded(
                rootURL: URL(fileURLWithPath: modelPath).standardizedFileURL,
                progressHandler: progressHandler
            )
        }
    }

    public func unload() {
        model = nil
        tokenizerAndTemplate = nil
        config = nil
        loadedModelPath = nil
        Memory.clearCache()
    }

    private func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        if loadedModelPath == rootURL.path, model != nil, tokenizerAndTemplate != nil {
            return
        }

        let missingFiles = LagunaResources.validate(rootURL: rootURL)
        guard missingFiles.isEmpty else {
            throw LagunaError.missingFiles(missingFiles)
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Laguna config"))
        let configData = try Data(contentsOf: rootURL.appending(path: "config.json"))
        let config = try JSONDecoder().decode(LagunaConfig.self, from: configData)

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Laguna tokenizer"))
        let tokenizer = try await LagunaTokenizerAndTemplate.load(
            from: rootURL,
            maxLength: config.maxPositionEmbeddings
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Laguna weights"))
        let model = LagunaCausalLM(config: config)
        try HFSafetensorsWeightsLoader.applyShardedWeights(
            indexURL: rootURL.appending(path: "model.safetensors.index.json"),
            to: model,
            dtype: nil,
            verify: .shapeMismatch,
            progressHandler: { progress in
                progressHandler?(ChatProgress(
                    stage: .loadingModel,
                    message: "Loading Laguna shard \(progress.shardIndex + 1)/\(progress.shardCount)"
                ))
            }
        )

        self.model = model
        self.tokenizerAndTemplate = tokenizer
        self.config = config
        self.loadedModelPath = rootURL.path
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws -> ChatResponse {
        guard let model, let tokenizerAndTemplate, let config else {
            throw LagunaError.modelNotLoaded
        }

        let requestedContext = request.maxContextTokens ?? LagunaResources.defaultContextLength
        guard requestedContext > 0 else {
            throw LagunaError.generationFailed("maxContextTokens must be greater than zero.")
        }
        let effectiveContext = min(requestedContext, config.maxPositionEmbeddings)

        progressHandler?(ChatProgress(stage: .encoding, message: "Encoding Laguna prompt"))
        let prefillStart = Date()
        let promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: request.messages,
            tools: request.tools,
            includeThinking: request.showThinking,
            maxLength: effectiveContext
        )
        guard !promptTokens.isEmpty else {
            throw LagunaError.generationFailed("The rendered prompt contained no tokens.")
        }

        let cache = model.makeCache()
        let prompt = MLXArray(promptTokens.map(Int32.init)).reshaped(1, promptTokens.count)
        let logits = model.lastPositionLogits(prompt, cache: cache)
        MLX.eval(logits)
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        let tokenBudget = max(0, min(request.maxTokens, effectiveContext - promptTokens.count))

        progressHandler?(ChatProgress(stage: .generating, message: ""))
        let decode = try AutoregressiveDecodeEngine.decode(
            AutoregressiveDecodeRequest(
                initialLogits: logits,
                generationConfig: GenerationConfig(
                    maxTokens: tokenBudget,
                    temperature: Float(request.temperature),
                    topK: request.topK ?? 0,
                    topP: Float(request.topP),
                    repetitionPenalty: nil,
                    repetitionContextSize: 64
                ),
                eosTokens: Set(config.eosTokenIDs + tokenizerAndTemplate.stopTokenIDs),
                tokenBudget: tokenBudget,
                historySeedTokens: promptTokens
            ),
            stepForward: { token in
                model.lastPositionLogits(token, cache: cache)
            },
            decodeToken: { tokenizerAndTemplate.decode(token: $0) },
            emitPiece: { _, piece in
                progressHandler?(ChatProgress(stage: .generating, message: piece))
            },
            checkCancellation: { try Task.checkCancellation() }
        )

        let decoded = tokenizerAndTemplate.decode(tokens: decode.generatedTokens)
        let trimmed = TextGenerationStopSequences.trimming(
            decoded,
            sequences: request.stopSequences
        )
        let toolCalls: [ToolCall]? = request.tools?.isEmpty == false ? {
            let parsed = LagunaToolParser.parseToolCalls(trimmed.text)
            return parsed.isEmpty ? nil : parsed
        }() : nil
        let finishReason: ChatFinishReason
        if trimmed.matchedSequence != nil {
            finishReason = .stopSequence
        } else if decode.generatedTokens.count >= tokenBudget, tokenBudget > 0 {
            finishReason = .length
        } else {
            finishReason = .stop
        }

        return ChatResponse(
            generatedText: trimmed.text,
            tokensGenerated: decode.generatedTokens.count,
            showThinking: request.showThinking,
            timing: ChatTiming(
                loadSeconds: 0,
                prefillSeconds: prefillSeconds,
                decodeSeconds: decode.decodeSeconds,
                firstTokenSeconds: decode.firstTokenSeconds,
                kvCacheMode: .default,
                prefillKVCache: "bf16",
                decodeKVCache: "bf16"
            ),
            toolCalls: toolCalls,
            promptTokens: promptTokens.count,
            finishReason: finishReason
        )
    }
}
