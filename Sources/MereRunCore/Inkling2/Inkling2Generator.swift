import Foundation
import MLX
import MLXRandom
import MereRunInkling2Model
@preconcurrency import Tokenizers

public actor Inkling2Generator: ChatGenerator {
    var model: Inkling2Model?
    var tokenizer: (any Tokenizer)?
    var template: Inkling2ChatTemplate?
    var loadedRoot: String?

    public init() {}

    public func prepare(modelPath: String?, progressHandler: (@Sendable (ChatProgress) -> Void)? = nil) async throws {
        try await MLXRequestStreams.withStream(isolation: self) { try await ensureLoaded(modelPath, progress: progressHandler) }
    }

    func ensureLoaded(_ modelPath: String?, progress: (@Sendable (ChatProgress) -> Void)?) async throws {
        guard let modelPath else { throw ChatRequestIssue("model", "install Inkling 2 or pass its original checkpoint directory") }
        let root = URL(fileURLWithPath: modelPath).standardizedFileURL
        if loadedRoot == root.path { return }
        unload()
        progress?(ChatProgress(stage: .loadingModel, message: "Opening native Inkling 2 checkpoint"))
        let config = try Inkling2Configuration.decode(Data(contentsOf: root.appendingPathComponent("config.json")))
        let weights = try Inkling2Weights(root: root)
        let loaded = try Inkling2Model(config: config, weights: weights)
        let loadedTokenizer = try await AutoTokenizer.from(modelFolder: root)
        let loadedTemplate = try Inkling2ChatTemplate(source: String(contentsOf: root.appendingPathComponent("chat_template.jinja"), encoding: .utf8))
        try Task.checkCancellation()
        model = loaded
        tokenizer = loadedTokenizer
        template = loadedTemplate
        loadedRoot = root.path
    }

    public func unload() {
        model = nil; tokenizer = nil; template = nil; loadedRoot = nil
        Memory.clearCache()
    }

    public func chat(_ request: ChatRequest, progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws -> ChatResponse {
        try await chat(request, modelPath: loadedRoot, progressHandler: progressHandler)
    }

    public func chat(_ request: ChatRequest, modelPath: String?, progressHandler: (@Sendable (ChatProgress) -> Void)? = nil) async throws -> ChatResponse {
        try await MLXRequestStreams.withStream(isolation: self) {
            try ChatRequestResolver.validate(request)
            guard request.lora == nil, !request.requiresJSON, request.noRepeatNgramSize == nil,
                  request.kvCacheMode == .default || request.kvCacheMode == .auto,
                  request.messages.allSatisfy({ $0.audioUrl == nil && $0.videoUrl == nil }) else {
                throw ChatRequestIssue("request", "Inkling 2 supports text, images and tools with native BF16 caches; LoRA, constrained JSON and audio/video are unavailable")
            }
            let loadStart = Date()
            try await ensureLoaded(modelPath, progress: progressHandler)
            let loadSeconds = Date().timeIntervalSince(loadStart)
            guard let model, let tokenizer, let template else { throw ChatRequestIssue("model", "Inkling 2 is not loaded") }
            if let seed = request.seed { MLXRandom.seed(seed) }
            let context = min(request.maxContextTokens ?? Inkling2Resources.defaultContextLength, model.config.textConfig.modelMaxLength)
            let prefillStart = Date()
            let prompt = try Inkling2Prompt.prepare(request, tokenizer: tokenizer, template: template, model: model)
            guard !prompt.tokens.isEmpty, prompt.tokens.count < context else {
                throw ChatRequestIssue("max-context-tokens", "the complete Inkling 2 prompt must fit with space for output")
            }
            let budget = min(request.maxTokens, context - prompt.tokens.count)
            let cache = Inkling2Cache(layerCount: model.config.textConfig.numHiddenLayers)
            var initial: MLXArray?
            for start in stride(from: 0, to: prompt.tokens.count, by: 8) {
                try Task.checkCancellation()
                let end = min(start + 8, prompt.tokens.count)
                let ids = MLXArray(prompt.tokens[start..<end].map(Int32.init)).reshaped(1, end - start)
                let embedding = try model.embeddings(ids)
                for image in prompt.images {
                    let lo = max(start, image.offset), hi = min(end, image.offset + image.count)
                    if lo < hi {
                        embedding[0, (lo - start)..<(hi - start), 0...] = image.embeddings[(lo - image.offset)..<(hi - image.offset), 0...].asType(embedding.dtype)
                    }
                }
                initial = try model.forward(embeddings: embedding, cache: cache)
                if let initial { eval(initial) }
                progressHandler?(ChatProgress(stage: .encoding, message: "Prefilled \(end)/\(prompt.tokens.count) tokens"))
                await Task.yield()
            }
            guard let initial else { throw ChatRequestIssue("prompt", "empty prompt") }
            let prefillSeconds = Date().timeIntervalSince(prefillStart)
            let sampling = GenerationConfig(maxTokens: budget, temperature: Float(request.temperature), topK: request.topK ?? 0,
                topP: Float(request.topP), minP: Float(request.minP), repetitionPenalty: Float(request.repetitionPenalty),
                repetitionContextSize: Int.max, presencePenalty: Float(request.presencePenalty), frequencyPenalty: Float(request.frequencyPenalty),
                penaltyPromptTokenCount: prompt.tokens.count)
            var raw = ""
            let eos = Set([model.config.eosTokenId, tokenizer.convertTokenToId("<|content_model_end_sampling|>")].compactMap { $0 })
            let decoded = try AutoregressiveDecodeEngine.decode(
                .init(initialLogits: initial, generationConfig: sampling, eosTokens: request.stopOnEOS ? eos : [], tokenBudget: budget,
                      historySeedTokens: prompt.tokens, logprobCapture: request.logprobCapture, logprobRegion: request.logprobRegionHint ?? .visible),
                stepForward: { try model($0, cache: cache) }, decodeToken: { tokenizer.decode(tokens: [$0]) },
                decodeTokens: { tokenizer.decode(tokens: $0) },
                emitPiece: { _, piece in raw += piece },
                shouldContinue: { _, _ in
                    let parsed = Inkling2OutputParser.parse(raw)
                    return TextGenerationStopSequences.trimming(parsed.visible, sequences: request.stopSequences).matchedSequence == nil
                }, checkCancellation: { try Task.checkCancellation() })
            let parsed = Inkling2OutputParser.parse(tokenizer.decode(tokens: decoded.generatedTokens), tools: request.tools ?? [], parallel: request.parallelToolCalls)
            let trimmed = TextGenerationStopSequences.trimming(parsed.visible, sequences: request.stopSequences)
            if request.showThinking, let reasoning = parsed.reasoning {
                progressHandler?(ChatProgress(stage: .generating, event: .reasoning(reasoning)))
            }
            progressHandler?(ChatProgress(stage: .generating, event: .text(trimmed.text)))
            let visible = request.showThinking && parsed.reasoning != nil
                ? "<think>\n\(parsed.reasoning ?? "")\n</think>\n\(trimmed.text)" : trimmed.text
            var response = ChatResponse(response: visible, tokensGenerated: decoded.generatedTokens.count,
                timing: ChatTiming(loadSeconds: loadSeconds, prefillSeconds: prefillSeconds, decodeSeconds: decoded.decodeSeconds,
                    firstTokenSeconds: decoded.firstTokenSeconds, kvCacheMode: .default, prefillKVCache: "bf16", decodeKVCache: "bf16"),
                toolCalls: parsed.toolCalls.isEmpty ? nil : parsed.toolCalls, promptTokens: prompt.tokens.count,
                finishReason: trimmed.matchedSequence != nil ? .stopSequence : (decoded.generatedTokens.count == budget ? .length : .stop),
                reasoningContent: parsed.reasoning, reasoningBlockCount: parsed.reasoning == nil ? 0 : 1, logprobs: decoded.logprobs)
            response.matchedStopSequence = trimmed.matchedSequence
            return response
        }
    }
}
