import Foundation
import MLX
import MLXRandom
import MereRunKolibriModel
@preconcurrency import Tokenizers

public actor KolibriGenerator: ChatGenerator {
    var model: KolibriCausalLM?
    var tokenizer: (any Tokenizer)?
    var loadedRoot: String?

    public init() {}

    public func prepare(modelPath: String?, progressHandler: (@Sendable (ChatProgress) -> Void)? = nil) async throws {
        try await MLXRequestStreams.withStream(isolation: self) {
            try await ensureLoaded(modelPath: modelPath, progressHandler: progressHandler)
        }
    }

    func ensureLoaded(modelPath: String?, progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws {
        guard let modelPath else { throw ChatRequestIssue("model-root", "Kolibri requires a converted native checkpoint directory.") }
        let root = URL(fileURLWithPath: modelPath).standardizedFileURL
        if loadedRoot == root.path { return }
        try Task.checkCancellation()
        model = nil
        tokenizer = nil
        loadedRoot = nil
        Memory.clearCache()
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading native Kolibri weights"))
        let loaded = try KolibriLoader.load(root: root)
        let loadedTokenizer = try await AutoTokenizer.from(modelFolder: root)
        try Task.checkCancellation()
        model = loaded
        tokenizer = loadedTokenizer
        loadedRoot = root.path
    }

    public func unload() {
        model = nil
        tokenizer = nil
        loadedRoot = nil
        Memory.clearCache()
    }

    public func chat(_ request: ChatRequest, progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws -> ChatResponse {
        try await chat(request, modelPath: loadedRoot, progressHandler: progressHandler)
    }

    public func chat(_ request: ChatRequest, modelPath: String?, progressHandler: (@Sendable (ChatProgress) -> Void)? = nil) async throws -> ChatResponse {
        try await MLXRequestStreams.withStream(isolation: self) {
            guard request.lora == nil, !request.requiresJSON,
                  request.noRepeatNgramSize == nil,
                  request.messages.allSatisfy({ $0.imageUrl == nil && $0.audioUrl == nil && $0.videoUrl == nil }) else {
                throw ChatRequestIssue("request", "Kolibri currently supports text and tool messages without LoRA or constrained JSON.")
            }
            let loadStart = Date()
            try await ensureLoaded(modelPath: modelPath, progressHandler: progressHandler)
            let loadSeconds = Date().timeIntervalSince(loadStart)
            guard let model, let tokenizer else { throw ChatRequestIssue("model", "Kolibri is not loaded.") }
            if let seed = request.seed { MLXRandom.seed(seed) }
            let context = min(request.maxContextTokens ?? KolibriResources.defaultContextLength, model.config.maxPositionEmbeddings)
            let prefillStart = Date()
            let prompt = try tokenizer.applyChatTemplate(
                messages: LagunaTokenizerAndTemplate.renderMessages(request.messages),
                tools: request.tools?.map { $0.toToolSpec() },
                additionalContext: ["enable_thinking": request.showThinking, "preserve_thinking": true]
            )
            guard !prompt.isEmpty, prompt.count < context else {
                throw ChatRequestIssue("max-context-tokens", "Kolibri prompt must fit with space for output; prompt truncation is not implicit.")
            }
            let cache = model.makeCache()
            var lastLogits: MLXArray?
            for start in stride(from: 0, to: prompt.count, by: 32) {
                try Task.checkCancellation()
                let end = min(start + 32, prompt.count)
                let tokens = MLXArray(prompt[start..<end].map(Int32.init)).reshaped(1, end - start)
                let logits = model(tokens, cache: cache, lastPositionOnly: true)
                eval(logits)
                lastLogits = logits
            }
            guard let logits = lastLogits else { throw ChatRequestIssue("prompt", "Kolibri prompt is empty.") }
            let prefillSeconds = Date().timeIntervalSince(prefillStart)
            let budget = min(request.maxTokens, context - prompt.count)
            let sampling = GenerationConfig(
                maxTokens: budget, temperature: Float(request.temperature), topK: request.topK ?? 0,
                topP: Float(request.topP), minP: Float(request.minP),
                repetitionPenalty: Float(request.repetitionPenalty), repetitionContextSize: Int.max,
                presencePenalty: Float(request.presencePenalty), frequencyPenalty: Float(request.frequencyPenalty),
                penaltyPromptTokenCount: prompt.count
            )
            let eos = Set([model.config.eosTokenID, tokenizer.eosTokenId].compactMap { $0 })
            var output = ""
            let result = try AutoregressiveDecodeEngine.decode(
                .init(initialLogits: logits, generationConfig: sampling,
                      eosTokens: request.stopOnEOS ? eos : [], tokenBudget: budget,
                      historySeedTokens: prompt, logprobCapture: request.logprobCapture,
                      logprobRegion: request.logprobRegionHint ?? .visible),
                stepForward: { model($0, cache: cache, lastPositionOnly: true) },
                decodeToken: { tokenizer.decode(tokens: [$0]) },
                decodeTokens: { tokenizer.decode(tokens: $0) },
                emitPiece: { _, piece in progressHandler?(ChatProgress(stage: .generating, message: piece)) },
                shouldContinue: { _, piece in
                    output += piece
                    return TextGenerationStopSequences.trimming(output, sequences: request.stopSequences).matchedSequence == nil
                },
                checkCancellation: { try Task.checkCancellation() }
            )
            let trimmed = TextGenerationStopSequences.trimming(tokenizer.decode(tokens: result.generatedTokens), sequences: request.stopSequences)
            let calls = request.tools?.isEmpty == false ? Q35ToolParser.parseToolCalls(trimmed.text) : []
            return ChatResponse(
                generatedText: trimmed.text, tokensGenerated: result.generatedTokens.count, showThinking: request.showThinking,
                timing: ChatTiming(loadSeconds: loadSeconds, prefillSeconds: prefillSeconds, decodeSeconds: result.decodeSeconds,
                                   firstTokenSeconds: result.firstTokenSeconds, kvCacheMode: .default,
                                   prefillKVCache: "bf16", decodeKVCache: "bf16"),
                toolCalls: calls.isEmpty ? nil : calls, promptTokens: prompt.count,
                finishReason: trimmed.matchedSequence != nil ? .stopSequence : (result.generatedTokens.count == budget ? .length : .stop),
                logprobs: result.logprobs
            )
        }
    }
}
