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
    private var dspark: Inkling2DSpark?
    private var lastDSparkStats: Inkling2DSparkStats?
    let prefixKVCacheEnabled: Bool
    var prefixKVCache: [Inkling2PrefixKVCacheKey: Inkling2PrefixKVCacheEntry] = [:]
    var prefixKVCacheHits = 0
    var prefixKVCacheMisses = 0
    var prefixKVCacheStores = 0
    var prefixKVCacheReusedTokens = 0
    /// Vision features of recent images by pixel marker, so an agent's screenshots are encoded once.
    var imageFeatureCache: [Int: MLXArray] = [:]
    var imageFeatureOrder: [Int] = []

    public init(prefixKVCacheEnabled: Bool = ProcessInfo.processInfo.environment["MERERUN_INKLING2_PREFIX_KV_CACHE"] == "1") {
        self.prefixKVCacheEnabled = prefixKVCacheEnabled
    }

    public func dsparkStats() -> Inkling2DSparkStats? { lastDSparkStats }

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
        let loopCDKey = "MERERUN_INKLING2_LOOPCD_STRENGTH"
        guard let loopCDStrength = Float(ProcessInfo.processInfo.environment[loopCDKey] ?? "0") else {
            throw Inkling2Error.invalid("invalid \(loopCDKey)")
        }
        let weights = try Inkling2Weights(root: root)
        let loaded = try Inkling2Model(config: config, weights: weights, loopCDStrength: loopCDStrength)
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
        dspark = nil; lastDSparkStats = nil
        resetPrefixKVCache()
        imageFeatureCache.removeAll(); imageFeatureOrder.removeAll()
        Memory.clearCache()
    }

    public func chat(_ request: ChatRequest, progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws -> ChatResponse {
        try await chat(request, modelPath: loadedRoot, progressHandler: progressHandler)
    }

    public func chat(_ request: ChatRequest, modelPath: String?, progressHandler: (@Sendable (ChatProgress) -> Void)? = nil) async throws -> ChatResponse {
        try await MLXRequestStreams.withStream(isolation: self) {
            try ChatRequestResolver.validate(request)
            let cacheMode = request.kvCacheMode ?? .default
            guard !request.requiresJSON, request.noRepeatNgramSize == nil,
                  cacheMode == .default || cacheMode == .auto,
                  request.messages.allSatisfy({ $0.audioUrl == nil && $0.videoUrl == nil }) else {
                throw ChatRequestIssue("request", "Inkling 2 supports text, images and tools with native BF16 caches; constrained JSON and audio/video are unavailable")
            }
            let loadStart = Date()
            let adapterFile: (url: URL, scale: Double)?
            if let lora = request.lora {
                let url = try await LoRAWeightLoader.resolveURL(for: lora)
                switch lora {
                case .local(_, let scale), .remote(_, let scale): adapterFile = (url, scale)
                }
            } else {
                adapterFile = nil
            }
            try await ensureLoaded(modelPath, progress: progressHandler)
            guard let model, let tokenizer, let template else { throw ChatRequestIssue("model", "Inkling 2 is not loaded") }
            let adapter = try adapterFile.map { try Inkling2MiniLoRALayer.load(from: $0.url, for: model.config, scale: $0.scale) }
            let draft = try draftForRequest(request, model: model)
            let loadSeconds = Date().timeIntervalSince(loadStart)
            if let seed = request.seed { MLXRandom.seed(seed) }
            let context = min(request.maxContextTokens ?? Inkling2Resources.defaultContextLength, model.config.textConfig.modelMaxLength)
            let prefillStart = Date()
            let prompt = try Inkling2Prompt.prepare(request, tokenizer: tokenizer, template: template) { image, marker in
                try cachedImageFeatures(image, marker: marker, model: model)
            }
            guard !prompt.tokens.isEmpty, prompt.tokens.count < context else {
                throw ChatRequestIssue("max-context-tokens", "the complete Inkling 2 prompt must fit with space for output")
            }
            let budget = min(request.maxTokens, context - prompt.tokens.count)
            // Cache keys mark each image's positions with its pixels, so a stored prefix can extend
            // past images and matches only a prompt with the same images.
            let keyTokens = prompt.keyTokens
            let modelRoot = loadedRoot ?? ""
            let seed = prefixKVCacheSeed(modelPath: modelRoot, speculative: draft != nil,
                                         promptTokens: keyTokens, reusableTokens: keyTokens.count)
            let checkpoints = semanticPrefixCheckpoints(request, template: template, promptTokens: keyTokens,
                                                        imageMarkers: prompt.imageMarkers, imageCounts: prompt.images.map(\.count))
            let cache = seed?.cache ?? model.makeCache()
            let speculation = draft.map { Inkling2DSparkSession(draft: $0, cache: cache, prefix: seed?.draft) }
            var initial: MLXArray?
            let chunkSize = model.prefillChunkSize(for: prompt.tokens.count)
            var start = seed?.tokenCount ?? 0
            if start > 0 {
                progressHandler?(ChatProgress(stage: .encoding, message: "Reusing \(start) prompt KV tokens"))
            }
            while start < prompt.tokens.count {
                try Task.checkCancellation()
                let end = Self.nextPrefillEnd(processed: start, total: prompt.tokens.count, chunkSize: chunkSize, checkpoints: checkpoints)
                let ids = MLXArray(prompt.tokens[start..<end].map(Int32.init)).reshaped(1, end - start)
                let embedding = try model.embeddings(ids)
                for image in prompt.images {
                    let lo = max(start, image.offset), hi = min(end, image.offset + image.count)
                    if lo < hi {
                        embedding[0, (lo - start)..<(hi - start), 0...] = image.embeddings[(lo - image.offset)..<(hi - image.offset), 0...].asType(embedding.dtype)
                    }
                }
                initial = try speculation?.prefill(embedding) ?? model.forward(embeddings: embedding, cache: cache, outputDelta: adapter?.callAsFunction)
                if let initial { eval(initial) }
                // Keep the conversation before the final message and the exact prompt, which the next
                // agent turn extends because the template adds no generation suffix. Both rank equally,
                // so pruning drops the least recently used and keeps the latest turn's exact prompt.
                if checkpoints.contains(end) || end == prompt.tokens.count {
                    storePrefixKVCache(modelPath: modelRoot, speculative: speculation != nil, promptTokens: keyTokens,
                                       tokenCount: end, cache: cache, draft: speculation?.prefix, priority: .semantic)
                }
                progressHandler?(ChatProgress(stage: .encoding, message: "Prefilled \(end)/\(prompt.tokens.count) tokens"))
                start = end
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
            let decoded: AutoregressiveDecodeResult
            if let draft, let speculation {
                let configured = ProcessInfo.processInfo.environment["MERERUN_INKLING2_DSPARK_TOKENS"].flatMap(Int.init)
                let fixedDepth = configured.map { min(max($0, 1), draft.maxProposalCount) }
                    ?? (model.usesSpeedPack && model.usePackedExperts ? min(3, draft.maxProposalCount) : nil)
                let distribution: ((MLXArray) -> MLXArray)? = sampling.temperature == 0 ? nil : { logits in
                    if logits.ndim == 2, sampling.topK > 0 || sampling.topP < 1 || sampling.minP > 0 {
                        return stacked((0..<logits.dim(0)).map {
                            samplingDistribution(logits: logits[$0], config: sampling, previousTokenIndices: nil, banMask: nil)
                        }, axis: 0)
                    }
                    return samplingDistribution(logits: logits, config: sampling, previousTokenIndices: nil, banMask: nil)
                }
                let result = try speculation.decode(initial: initial, budget: budget, eos: request.stopOnEOS ? eos : [],
                                                    fixedDepth: fixedDepth, distribution: distribution)
                lastDSparkStats = Inkling2DSparkStats(rounds: result.rounds, draftedTokens: result.drafted,
                                                    acceptedTokens: result.accepted, depthCounts: result.depthCounts)
                decoded = .init(generatedTokens: result.tokens, decodeSeconds: result.seconds, firstTokenSeconds: result.firstTokenSeconds)
            } else {
                decoded = try AutoregressiveDecodeEngine.decode(
                    .init(initialLogits: initial, generationConfig: sampling, eosTokens: request.stopOnEOS ? eos : [], tokenBudget: budget,
                          historySeedTokens: prompt.tokens, logprobCapture: request.logprobCapture, logprobRegion: request.logprobRegionHint ?? .visible),
                    stepForward: { try model.forward(embeddings: model.embeddings($0), cache: cache, outputDelta: adapter?.callAsFunction) }, decodeToken: { tokenizer.decode(tokens: [$0]) },
                    decodeTokens: { tokenizer.decode(tokens: $0) },
                    emitPiece: { _, piece in raw += piece },
                    shouldContinue: { _, _ in
                        let parsed = Inkling2OutputParser.parse(raw)
                        return TextGenerationStopSequences.trimming(parsed.visible, sequences: request.stopSequences).matchedSequence == nil
                    }, checkCancellation: { try Task.checkCancellation() })
            }
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

    static func usesDSpark(_ request: ChatRequest, modelType: String, packedExperts: Bool, preference: String?) -> Bool {
        guard modelType == "inkling2_loop", request.lora == nil, request.maxTokens > 1,
              request.repetitionPenalty == 1, request.presencePenalty == 0, request.frequencyPenalty == 0,
              !request.logprobCapture.isEnabled, request.stopSequences.isEmpty else { return false }
        if let preference { return preference == "1" }
        return packedExperts && request.temperature > 0 && request.topP == 1
            && (request.topK ?? 0) == 0 && request.minP == 0
    }

    private func draftForRequest(_ request: ChatRequest, model: Inkling2Model) throws -> Inkling2DSpark? {
        lastDSparkStats = nil
        let preference = ProcessInfo.processInfo.environment["MERERUN_INKLING2_DSPARK"]
        guard Self.usesDSpark(request, modelType: model.config.modelType, packedExperts: model.usePackedExperts,
                              preference: preference),
              let loadedRoot else { return nil }
        if let dspark { return dspark }
        let weightsURL = URL(fileURLWithPath: loadedRoot).appendingPathComponent("mtp.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            if preference == "1" {
                throw ChatRequestIssue("model", "DSpark requires mtp.safetensors in the Inkling 2 Mini checkpoint directory")
            }
            return nil
        }
        let draft = try Inkling2DSpark(target: model, weightsURL: weightsURL)
        dspark = draft
        return draft
    }
}

public struct Inkling2DSparkStats: Sendable {
    public let rounds: Int
    public let draftedTokens: Int
    public let acceptedTokens: Int
    public let depthCounts: [Int: Int]
}
