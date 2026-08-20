import Foundation
import MLX

public actor NemotronOmniGenerator: ChatGenerator {
    private static let prefillChunkSize = 128

    private var model: NemotronOmniCausalLM?
    private var visionTower: NemotronOmniVisionTower?
    private var soundTower: NemotronOmniSoundTower?
    private var tokenizer: Q35TokenizerAndTemplate?
    private var config: NemotronOmniConfig?
    private var preprocessorConfig: NemotronOmniPreprocessorConfig?
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
                throw NemotronOmniError.generationFailed("LoRA is not supported by this runtime")
            }
            guard !request.requiresJSON else {
                throw NemotronOmniError.generationFailed(
                    "constrained JSON is not supported by this runtime"
                )
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
        visionTower = nil
        soundTower = nil
        tokenizer = nil
        config = nil
        preprocessorConfig = nil
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
                throw NemotronOmniError.modelPathRequired
            }
            return root
        }
        if let installed = ManagedModelResolver.resolveInstalledModel(
            id: NemotronOmniResources.modelID
        ) {
            return installed.standardizedFileURL
        }
        progressHandler?(ChatProgress(
            stage: .loadingModel,
            message: "Nemotron Omni must be installed explicitly before inference"
        ))
        throw NemotronOmniError.modelPathRequired
    }

    private func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        guard loadedPath != rootURL.path || model == nil else { return }
        let loaded = try await NemotronOmniModelLoader.load(
            rootURL: rootURL,
            maxContextLength: NemotronOmniResources.maximumContextLength,
            progressHandler: progressHandler
        )
        model = loaded.model
        visionTower = loaded.visionTower
        soundTower = loaded.soundTower
        tokenizer = loaded.tokenizer
        config = loaded.config
        preprocessorConfig = loaded.preprocessorConfig
        loadedPath = loaded.rootURL.path
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        guard let model,
              let visionTower,
              let soundTower,
              let tokenizer,
              let config,
              let preprocessorConfig else {
            throw NemotronOmniError.generationFailed("model is not loaded")
        }
        let contextLength = min(
            request.maxContextTokens ?? NemotronOmniResources.defaultContextLength,
            NemotronOmniResources.maximumContextLength,
            config.language.maxPositionEmbeddings
        )
        var preparedImages: [Int: NemotronOmniPreparedImage] = [:]
        var preparedVideos: [Int: NemotronOmniPreparedVideo] = [:]
        var preparedAudio: [Int: NemotronOmniPreparedAudio] = [:]
        for index in request.messages.indices {
            let message = request.messages[index]
            if let reference = message.imageUrl?.trimmingCharacters(in: .whitespacesAndNewlines),
               !reference.isEmpty {
                preparedImages[index] = try NemotronOmniImageProcessor.prepare(
                    reference: reference,
                    config: preprocessorConfig,
                    contextLength: contextLength
                )
            }
            if let reference = message.videoUrl?.trimmingCharacters(in: .whitespacesAndNewlines),
               !reference.isEmpty {
                preparedVideos[index] = try NemotronOmniVideoProcessor.prepare(
                    reference: reference,
                    config: preprocessorConfig,
                    contextLength: contextLength
                )
            }
            if let reference = message.audioUrl?.trimmingCharacters(in: .whitespacesAndNewlines),
               !reference.isEmpty {
                preparedAudio[index] = try NemotronOmniAudioProcessor.prepare(reference: reference)
            }
        }
        var promptMessages = request.messages
        if !preparedImages.isEmpty || !preparedVideos.isEmpty || !preparedAudio.isEmpty {
            for index in promptMessages.indices {
                var mediaPrefix = ""
                if preparedImages[index] != nil {
                    mediaPrefix += "<image>\n"
                }
                if let video = preparedVideos[index] {
                    mediaPrefix += video.promptPrefix
                }
                if preparedAudio[index] != nil {
                    mediaPrefix += "<so_embedding>\n"
                }
                promptMessages[index].content = mediaPrefix + promptMessages[index].content
                promptMessages[index].imageUrl = nil
                promptMessages[index].videoUrl = nil
                promptMessages[index].audioUrl = nil
            }
        }
        var promptTokens = try tokenizer.encodeForGeneration(
            messages: promptMessages,
            tools: request.tools,
            addGenerationPrompt: true,
            includeThinking: request.showThinking,
            reasoningEffort: request.reasoningEffort.map { String($0) },
            maxLength: contextLength
        )
        var visionTokenCounts: [Int] = []
        for index in request.messages.indices {
            if let image = preparedImages[index] {
                visionTokenCounts.append(image.languageTokenCount)
            }
            if let video = preparedVideos[index] {
                visionTokenCounts.append(contentsOf: video.languageTokenCounts)
            }
        }
        if !visionTokenCounts.isEmpty {
            promptTokens = try Self.expandImagePlaceholders(
                promptTokens,
                tokenCounts: visionTokenCounts,
                imageTokenID: config.imageContextTokenID
            )
        }
        if !preparedAudio.isEmpty {
            let audioTokenCounts = request.messages.indices.compactMap {
                preparedAudio[$0]?.languageTokenCount
            }
            promptTokens = try Self.expandAudioPlaceholders(
                promptTokens,
                tokenCounts: audioTokenCounts,
                soundTokenID: config.soundContextTokenID
            )
        }
        if promptTokens.count > contextLength {
            guard preparedImages.isEmpty,
                  preparedVideos.isEmpty,
                  preparedAudio.isEmpty else {
                throw NemotronOmniError.generationFailed(
                    "multimodal prompt exceeds the \(contextLength)-token context"
                )
            }
            promptTokens = Array(promptTokens.suffix(contextLength))
        }
        guard !promptTokens.isEmpty else {
            throw NemotronOmniError.generationFailed("prompt tokenization produced no tokens")
        }

        let cache = model.makeCache()
        var promptEmbeddings = model.inputEmbeddings(
            MLXArray(promptTokens.map(Int32.init)).reshaped(1, promptTokens.count)
        )
        if !preparedImages.isEmpty || !preparedVideos.isEmpty {
            progressHandler?(ChatProgress(
                stage: .encoding,
                message: "Encoding image/video input with C-RADIO v4-H"
            ))
            var replacements: [MLXArray] = []
            for index in request.messages.indices {
                if let image = preparedImages[index] {
                    let embeddings = visionTower.encodeImage(image.pixelValues)
                    MLX.eval(embeddings)
                    replacements.append(embeddings)
                }
                if let video = preparedVideos[index] {
                    let embeddings = visionTower.encodeVideo(video.pixelValues)
                    MLX.eval(embeddings)
                    for group in 0..<video.tubeletCount {
                        replacements.append(embeddings[group..<(group + 1), 0..., 0...])
                    }
                }
            }
            promptEmbeddings = try Self.replaceMediaEmbeddings(
                promptEmbeddings,
                promptTokens: promptTokens,
                placeholderTokenID: config.imageContextTokenID,
                replacements: replacements
            )
        }
        if !preparedAudio.isEmpty {
            progressHandler?(ChatProgress(
                stage: .encoding,
                message: "Encoding \(preparedAudio.count) audio clip(s) with Parakeet Fast-Conformer"
            ))
            let replacements = request.messages.indices.compactMap { index -> MLXArray? in
                guard let prepared = preparedAudio[index] else { return nil }
                let embeddings = soundTower(
                    prepared.melFeatures,
                    validFrameCount: prepared.validMelFrameCount,
                    layerProgress: { completed, total in
                        progressHandler?(ChatProgress(
                            stage: .encoding,
                            message: "Audio encoder block \(completed)/\(total)"
                        ))
                    }
                )
                MLX.eval(embeddings)
                return embeddings
            }
            promptEmbeddings = try Self.replaceMediaEmbeddings(
                promptEmbeddings,
                promptTokens: promptTokens,
                placeholderTokenID: config.soundContextTokenID,
                replacements: replacements
            )
        }
        let prefillStart = Date()
        var offset = 0
        var initialLogits: MLXArray?
        while offset < promptTokens.count {
            try Task.checkCancellation()
            let end = min(promptTokens.count, offset + Self.prefillChunkSize)
            let logits = model.prefill(
                embeddings: promptEmbeddings[0..., offset..<end, 0...],
                cache: cache
            )
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
            throw NemotronOmniError.generationFailed("prefill produced no logits")
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
        let eosTokens = Set(
            [config.language.eosTokenID, 11]
                + [tokenizer.eosTokenId].compactMap { $0 }
        )
        progressHandler?(ChatProgress(stage: .generating, message: ""))
        let decoded = try AutoregressiveDecodeEngine.decode(
            AutoregressiveDecodeRequest(
                initialLogits: initialLogits,
                generationConfig: generationConfig,
                eosTokens: eosTokens,
                tokenBudget: tokenBudget,
                historySeedTokens: promptTokens,
                logprobCapture: request.logprobCapture,
                logprobRegion: request.logprobRegionHint ?? .visible
            ),
            stepForward: { token in model.lastPositionLogits(token, cache: cache) },
            decodeToken: { tokenizer.decode(token: $0) },
            emitPiece: { _, piece in
                progressHandler?(ChatProgress(stage: .generating, message: piece))
            },
            checkCancellation: { try Task.checkCancellation() }
        )
        let raw = tokenizer.decode(tokens: decoded.generatedTokens)
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
            tokensGenerated: decoded.generatedTokens.count,
            showThinking: request.showThinking,
            timing: ChatTiming(
                prefillSeconds: prefillSeconds,
                decodeSeconds: decoded.decodeSeconds,
                firstTokenSeconds: decoded.firstTokenSeconds,
                kvCacheMode: .default,
                prefillKVCache: "hybrid-fp32-ssm-bf16-kv",
                decodeKVCache: "hybrid-fp32-ssm-bf16-kv"
            ),
            toolCalls: toolCalls,
            promptTokens: promptTokens.count,
            finishReason: decoded.generatedTokens.count >= tokenBudget ? .length : .stop,
            logprobs: decoded.logprobs,
            acceleration: ChatAccelerationDiagnostics(route: "nemotron-omni-bf16-native")
        )
    }

    static func expandImagePlaceholders(
        _ tokens: [Int],
        tokenCounts: [Int],
        imageTokenID: Int
    ) throws -> [Int] {
        guard !tokenCounts.isEmpty else { return tokens }
        let occurrenceCount = tokens.filter { $0 == imageTokenID }.count
        guard occurrenceCount == tokenCounts.count else {
            throw NemotronOmniError.generationFailed(
                "prompt has \(occurrenceCount) image placeholders for \(tokenCounts.count) image(s)"
            )
        }
        var imageIndex = 0
        var result: [Int] = []
        result.reserveCapacity(tokens.count + tokenCounts.reduce(0, +))
        for token in tokens {
            guard token == imageTokenID else {
                result.append(token)
                continue
            }
            result.append(19)
            result.append(contentsOf: repeatElement(imageTokenID, count: tokenCounts[imageIndex]))
            result.append(20)
            imageIndex += 1
        }
        return result
    }

    static func expandAudioPlaceholders(
        _ tokens: [Int],
        tokenCounts: [Int],
        soundTokenID: Int
    ) throws -> [Int] {
        guard !tokenCounts.isEmpty else { return tokens }
        let occurrenceCount = tokens.filter { $0 == soundTokenID }.count
        guard occurrenceCount == tokenCounts.count else {
            throw NemotronOmniError.generationFailed(
                "prompt has \(occurrenceCount) audio placeholders for \(tokenCounts.count) clip(s)"
            )
        }
        var audioIndex = 0
        var result: [Int] = []
        result.reserveCapacity(tokens.count + tokenCounts.reduce(0, +))
        for token in tokens {
            guard token == soundTokenID else {
                result.append(token)
                continue
            }
            result.append(28)
            result.append(contentsOf: repeatElement(soundTokenID, count: tokenCounts[audioIndex]))
            result.append(29)
            audioIndex += 1
        }
        return result
    }

    static func replaceMediaEmbeddings(
        _ embeddings: MLXArray,
        promptTokens: [Int],
        placeholderTokenID: Int,
        replacements: [MLXArray]
    ) throws -> MLXArray {
        guard !replacements.isEmpty else { return embeddings }
        var runs: [Range<Int>] = []
        var cursor = 0
        while cursor < promptTokens.count {
            guard promptTokens[cursor] == placeholderTokenID else {
                cursor += 1
                continue
            }
            let start = cursor
            while cursor < promptTokens.count, promptTokens[cursor] == placeholderTokenID {
                cursor += 1
            }
            runs.append(start..<cursor)
        }
        guard runs.count == replacements.count else {
            throw NemotronOmniError.generationFailed(
                "prompt contains \(runs.count) media spans for \(replacements.count) encoder outputs"
            )
        }

        var parts: [MLXArray] = []
        var inputCursor = 0
        for (run, replacement) in zip(runs, replacements) {
            guard replacement.dim(0) == 1, replacement.dim(1) == run.count else {
                throw NemotronOmniError.generationFailed(
                    "media span has \(run.count) placeholders but encoder produced \(replacement.dim(1)) embeddings"
                )
            }
            if run.lowerBound > inputCursor {
                parts.append(embeddings[0..., inputCursor..<run.lowerBound, 0...])
            }
            parts.append(replacement.asType(embeddings.dtype))
            inputCursor = run.upperBound
        }
        if inputCursor < promptTokens.count {
            parts.append(embeddings[0..., inputCursor..., 0...])
        }
        return parts.count == 1 ? parts[0] : MLX.concatenated(parts, axis: 1)
    }
}
