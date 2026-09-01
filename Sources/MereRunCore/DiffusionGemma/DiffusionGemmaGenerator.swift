import Foundation
import MLX
import MLXNN
import MLXRandom
@preconcurrency import Hub

private struct DiffusionGemmaRandomKeyStream {
    private var key: MLXArray

    init(seed: UInt64) {
        key = MLXRandom.key(seed)
    }

    mutating func next() -> MLXArray {
        let (nextKey, sampleKey) = MLXRandom.split(key: key)
        key = nextKey
        return sampleKey
    }
}

public actor DiffusionGemmaGenerator: ChatGenerator {
    private static let minimumCanvasLength = 64
    private static let transferThreshold: Float = 0.9
    private static let maximumUnmaskingCharacters = 160

    private let modelID: String
    private var model: DiffusionGemmaLanguageModel?
    private var tokenizerAndTemplate: Gemma4TokenizerAndTemplate?
    private var generationConfig: DiffusionGemmaGenerationConfig?
    private var loadedModelPath: String?

    public init(modelID: String = DiffusionGemmaResources.modelID) {
        self.modelID = modelID
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
        try await Stream.withNewDefaultStream {
            try validate(request)
            let loadStart = Date()
            let rootURL = try await resolveModelRoot(
                modelPath: modelPath,
                progressHandler: progressHandler
            )
            try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
            let loadSeconds = Date().timeIntervalSince(loadStart)
            var response = try generate(request, progressHandler: progressHandler)
            if var timing = response.timing {
                timing.loadSeconds = loadSeconds
                response.timing = timing
            }
            return response
        }
    }

    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws {
        try await Stream.withNewDefaultStream {
            let rootURL = try await resolveModelRoot(
                modelPath: modelPath,
                progressHandler: progressHandler
            )
            try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
        }
    }

    public func unload() {
        model = nil
        tokenizerAndTemplate = nil
        generationConfig = nil
        loadedModelPath = nil
        Memory.clearCache()
    }

    private func validate(_ request: ChatRequest) throws {
        if request.messages.contains(where: {
            $0.imageUrl != nil || $0.audioUrl != nil || $0.videoUrl != nil
        }) {
            throw DiffusionGemmaError.unsupportedConfiguration(
                "DiffusionGemma image, audio, and video inputs are not qualified in this runtime."
            )
        }
        if request.lora != nil {
            throw DiffusionGemmaError.unsupportedConfiguration(
                "DiffusionGemma does not support LoRA adapters."
            )
        }
        if request.requiresJSON {
            throw DiffusionGemmaError.unsupportedConfiguration(
                "DiffusionGemma does not support constrained JSON decoding."
            )
        }
        if request.logprobCapture != .none {
            throw DiffusionGemmaError.unsupportedConfiguration(
                "DiffusionGemma does not expose autoregressive token logprobs."
            )
        }
        if request.maxTokens < 0 {
            throw DiffusionGemmaError.unsupportedConfiguration("maxTokens must not be negative.")
        }
    }

    private func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let normalizedRoot = rootURL.standardizedFileURL
        if loadedModelPath == normalizedRoot.path,
           model != nil,
           tokenizerAndTemplate != nil,
           generationConfig != nil {
            return
        }

        let resources = DiffusionGemmaResources(rootURL: normalizedRoot)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw DiffusionGemmaError.missingFiles(missing.map(\.lastPathComponent))
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading DiffusionGemma config"))
        let config = try JSONDecoder().decode(
            DiffusionGemmaConfig.self,
            from: Data(contentsOf: resources.configURL)
        )
        guard config.modelType == "diffusion_gemma",
              config.architectures.contains("DiffusionGemmaForBlockDiffusion") else {
            throw DiffusionGemmaError.unsupportedConfiguration(
                "The selected checkpoint is not a DiffusionGemma block-diffusion model."
            )
        }
        guard config.canvasLength > 0,
              config.canvasLength <= DiffusionGemmaResources.maximumCanvasLength else {
            throw DiffusionGemmaError.unsupportedConfiguration(
                "Unsupported DiffusionGemma canvas length \(config.canvasLength)."
            )
        }

        let decodedGenerationConfig = try JSONDecoder().decode(
            DiffusionGemmaGenerationConfig.self,
            from: Data(contentsOf: resources.generationConfigURL)
        )
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading DiffusionGemma tokenizer"))
        let tokenizer = try await Gemma4TokenizerAndTemplate.load(
            from: normalizedRoot,
            maxLengthOverride: min(
                DiffusionGemmaResources.defaultContextLength,
                config.textConfig.maxPositionEmbeddings
            )
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading DiffusionGemma OptiQ weights"))
        let loadedModel = DiffusionGemmaLanguageModel(config: config)
        try HFSafetensorsWeightsLoader.applyQuantizedWeights(
            indexURL: resources.modelIndexURL,
            to: loadedModel,
            groupSize: config.quantization.groupSize,
            bits: config.quantization.bits,
            quantizedModuleResolver: { _, _, _, _, _, groupSize, bits in
                (groupSize: groupSize, bits: bits, mode: .affine)
            }
        )
        Memory.clearCache()

        model = loadedModel
        tokenizerAndTemplate = tokenizer
        generationConfig = decodedGenerationConfig
        loadedModelPath = normalizedRoot.path
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws -> ChatResponse {
        guard let model, let tokenizerAndTemplate, let generationConfig else {
            throw DiffusionGemmaError.unsupportedConfiguration("DiffusionGemma is not loaded.")
        }

        progressHandler?(ChatProgress(stage: .encoding, message: "Encoding DiffusionGemma prompt"))
        let contextLimit = min(
            request.maxContextTokens ?? DiffusionGemmaResources.defaultContextLength,
            model.config.textConfig.maxPositionEmbeddings
        )
        let promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: request.messages,
            tools: request.tools,
            includeThinking: request.showThinking,
            maxLength: contextLimit
        )
        guard !promptTokens.isEmpty else {
            throw DiffusionGemmaError.unsupportedConfiguration("The rendered prompt is empty.")
        }

        let tokenBudget = min(
            request.maxTokens,
            generationConfig.maxNewTokens,
            max(0, contextLimit - promptTokens.count)
        )
        guard tokenBudget > 0 else {
            return ChatResponse(
                response: "",
                tokensGenerated: 0,
                timing: ChatTiming(),
                promptTokens: promptTokens.count,
                finishReason: .length
            )
        }

        let caches = model.makeCaches()
        let promptArray = MLXArray(promptTokens.map(Int32.init)).reshaped(1, promptTokens.count)
        let prefillStart = Date()
        model.updateCache(promptArray, caches: caches)
        evaluateGemma4CacheStorage(caches)
        let prefillSeconds = Date().timeIntervalSince(prefillStart)

        progressHandler?(ChatProgress(stage: .generating, message: "Generating..."))
        let decodeStart = Date()
        let effectiveSeed = request.seed ?? UInt64.random(in: UInt64.min...UInt64.max)
        var randomKeys = DiffusionGemmaRandomKeyStream(seed: effectiveSeed)
        var generated: [Int] = []
        generated.reserveCapacity(tokenBudget)
        let eos = Set(generationConfig.eosTokenIds + model.config.eosTokenIds + tokenizerAndTemplate.stopTokenIds)
        var finishReason: ChatFinishReason = .length
        var canvasCount = 0
        var canvasTokens = 0
        var denoisingSteps = 0
        var workTokens = 0
        var firstDraftSeconds: Double?

        while generated.count < tokenBudget {
            canvasCount += 1
            let remaining = tokenBudget - generated.count
            let canvasLength = min(
                model.config.canvasLength,
                max(remaining, Self.minimumCanvasLength)
            )
            canvasTokens += canvasLength
            var canvas = MLXRandom.randInt(
                Int32(0)..<Int32(model.config.textConfig.vocabSize),
                [1, canvasLength],
                key: randomKeys.next()
            )
            var revealed = MLXArray.zeros([1, canvasLength], dtype: .bool)
            var accepted = canvas
            var selfConditioningLogits: MLXArray?
            var finalCanvas = canvas
            var stabilityHistory: [MLXArray] = []
            var denoisingStepsThisCanvas = 0

            for step in stride(from: generationConfig.maxDenoisingSteps, through: 1, by: -1) {
                denoisingSteps += 1
                denoisingStepsThisCanvas += 1
                workTokens += canvasLength
                var logits = model.canvasLogits(
                    canvas,
                    caches: caches,
                    selfConditioningLogits: selfConditioningLogits
                )
                let fraction = Float(step) / Float(generationConfig.maxDenoisingSteps)
                let scheduledTemperature = generationConfig.minimumTemperature
                    + (generationConfig.maximumTemperature - generationConfig.minimumTemperature) * fraction
                logits = logits / MLXArray(scheduledTemperature).asType(logits.dtype)

                let argmaxCanvas = argMax(logits, axis: -1).asType(.int32)
                finalCanvas = argmaxCanvas
                if step == 1 {
                    break
                }

                let denoised: MLXArray
                if request.temperature <= 0 {
                    denoised = argmaxCanvas
                } else {
                    denoised = MLXRandom.categorical(
                        logits / MLXArray(Float(request.temperature)).asType(logits.dtype),
                        axis: -1,
                        key: randomKeys.next()
                    ).asType(.int32)
                }
                let confidence = Self.tokenProbability(logits: logits, tokenIDs: denoised)
                let unrevealed = revealed .== MLXArray(false)
                var transfer = unrevealed .&& (
                    confidence .>= MLXArray(Self.transferThreshold).asType(confidence.dtype)
                )
                let eligibleConfidence = MLX.where(
                    unrevealed,
                    confidence,
                    MLXArray(-Float.infinity).asType(confidence.dtype)
                )
                let best = argMax(eligibleConfidence, axis: -1)
                let positions = MLXArray(Int32(0)..<Int32(canvasLength)).reshaped(1, canvasLength)
                let forced = positions .== MLX.expandedDimensions(best, axis: -1)
                transfer = MLX.where(MLX.any(transfer), transfer, forced)

                accepted = MLX.where(transfer, denoised, accepted)
                revealed = revealed .|| transfer
                let randomCanvas = MLXRandom.randInt(
                    Int32(0)..<Int32(model.config.textConfig.vocabSize),
                    [1, canvasLength],
                    key: randomKeys.next()
                )
                canvas = MLX.where(revealed, accepted, randomCanvas)
                selfConditioningLogits = logits

                if request.showUnmasking {
                    Self.emitUnmaskingProgress(
                        tokens: accepted,
                        revealed: revealed,
                        tokenizerAndTemplate: tokenizerAndTemplate,
                        step: generationConfig.maxDenoisingSteps - step + 1,
                        totalSteps: generationConfig.maxDenoisingSteps,
                        canvasIndex: canvasCount,
                        blockComplete: false,
                        showThinking: request.showThinking,
                        progressHandler: progressHandler
                    )
                    if firstDraftSeconds == nil {
                        firstDraftSeconds = Date().timeIntervalSince(decodeStart)
                    }
                }

                if MLX.all(revealed).item(Bool.self) {
                    break
                }
                if stableAndConfident(
                    argmaxCanvas,
                    logits: logits,
                    history: &stabilityHistory,
                    generationConfig: generationConfig
                ) {
                    break
                }
            }

            MLX.eval(finalCanvas)
            let block = finalCanvas.asArray(Int32.self).map(Int.init)
            if request.showUnmasking {
                let finalDraft = Self.maskedDraft(
                    tokens: block,
                    revealed: Array(repeating: true, count: block.count),
                    skipTokenIDs: eos,
                    decode: tokenizerAndTemplate.decode(tokens:)
                )
                progressHandler?(ChatProgress(
                    stage: .generating,
                    diffusion: ChatDiffusionProgress(
                        draftText: Self.cleanedResponse(
                            finalDraft,
                            showThinking: request.showThinking
                        ),
                        step: denoisingStepsThisCanvas,
                        totalSteps: generationConfig.maxDenoisingSteps,
                        canvasIndex: canvasCount,
                        blockComplete: true
                    )
                ))
                if firstDraftSeconds == nil {
                    firstDraftSeconds = Date().timeIntervalSince(decodeStart)
                }
            }
            var stopped = false
            for token in block.prefix(remaining) {
                if request.stopOnEOS, eos.contains(token) {
                    finishReason = .stop
                    stopped = true
                    break
                }
                generated.append(token)
            }
            if stopped || generated.count >= tokenBudget {
                break
            }
            model.updateCache(finalCanvas, caches: caches)
            evaluateGemma4CacheStorage(caches)
            Memory.clearCache()
        }

        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        let decodedRaw = tokenizerAndTemplate.decode(tokens: generated)
        let trimmed = TextGenerationStopSequences.trimming(
            decodedRaw,
            sequences: TextGenerationStopSequences.merged(
                request.stopSequences + TextGenerationStopSequences.defaultRenderedChatStops
            )
        )
        if trimmed.matchedSequence != nil {
            finishReason = .stopSequence
        }
        let reasoning = ChatReasoningMarkup.splitThinkBlocks(in: trimmed.text)
        let response = Self.cleanedResponse(
            request.showThinking ? trimmed.text : reasoning.visibleContent,
            showThinking: request.showThinking
        )
        let parsedToolCalls = request.tools?.isEmpty == false
            ? Gemma4ToolParser.parseToolCalls(response)
            : []
        let diffusion = ChatDiffusionDiagnostics(
            seed: effectiveSeed,
            canvasTokens: canvasTokens,
            denoisingSteps: denoisingSteps,
            workTokens: workTokens,
            canvasTokensPerSecond: decodeSeconds > 0
                ? Double(canvasTokens) / decodeSeconds
                : 0,
            workTokensPerSecond: decodeSeconds > 0
                ? Double(workTokens) / decodeSeconds
                : 0,
            firstDraftSeconds: firstDraftSeconds
        )

        return ChatResponse(
            response: response,
            tokensGenerated: generated.count,
            timing: ChatTiming(
                prefillSeconds: prefillSeconds,
                decodeSeconds: decodeSeconds,
                firstTokenSeconds: decodeSeconds,
                prefillKVCache: "native-hybrid",
                decodeKVCache: "native-hybrid",
                prefillTokensPerSecond: prefillSeconds > 0
                    ? Double(promptTokens.count) / prefillSeconds
                    : nil,
                decodeTokensPerSecond: decodeSeconds > 0
                    ? Double(generated.count) / decodeSeconds
                    : nil
            ),
            toolCalls: parsedToolCalls.isEmpty ? nil : parsedToolCalls,
            promptTokens: promptTokens.count,
            finishReason: finishReason,
            reasoningContent: reasoning.reasoningContent,
            hasIncompleteReasoning: reasoning.hasIncompleteReasoning,
            reasoningBlockCount: reasoning.reasoningBlockCount,
            hasReopenedReasoning: reasoning.hasReopenedReasoning,
            diffusion: diffusion
        )
    }

    nonisolated static func tokenProbability(
        logits: MLXArray,
        tokenIDs: MLXArray
    ) -> MLXArray {
        let floatLogits = logits.asType(.float32)
        let selected = takeAlong(
            floatLogits,
            MLX.expandedDimensions(tokenIDs.asType(.int32), axis: -1),
            axis: -1
        ).squeezed(axis: -1)
        return MLX.exp(selected - logSumExp(floatLogits, axis: -1))
    }

    nonisolated static func maskedDraft(
        tokens: [Int],
        revealed: [Bool],
        skipTokenIDs: Set<Int> = [],
        maximumCharacters: Int = maximumUnmaskingCharacters,
        decode: ([Int]) -> String
    ) -> String {
        precondition(tokens.count == revealed.count)
        var pieces: [String] = []
        var pending: [Int] = []

        func flushPending() {
            guard !pending.isEmpty else { return }
            pieces.append(decode(pending))
            pending.removeAll(keepingCapacity: true)
        }

        for (token, isRevealed) in zip(tokens, revealed) {
            if isRevealed {
                if !skipTokenIDs.contains(token) {
                    pending.append(token)
                }
            } else {
                flushPending()
                pieces.append("[Mask]")
            }
        }
        flushPending()

        let text = pieces
            .filter { !$0.isEmpty }
            .joined(separator: " ")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\n", with: "\\n")
        return String(text.prefix(max(0, maximumCharacters)))
    }

    private nonisolated static func emitUnmaskingProgress(
        tokens: MLXArray,
        revealed: MLXArray,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        step: Int,
        totalSteps: Int,
        canvasIndex: Int,
        blockComplete: Bool,
        showThinking: Bool,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) {
        MLX.eval(tokens, revealed)
        let tokenValues = tokens.asArray(Int32.self).map(Int.init)
        let revealValues = revealed.asArray(Bool.self)
        let draft = maskedDraft(
            tokens: tokenValues,
            revealed: revealValues,
            decode: tokenizerAndTemplate.decode(tokens:)
        )
        progressHandler?(ChatProgress(
            stage: .generating,
            diffusion: ChatDiffusionProgress(
                draftText: cleanedResponse(draft, showThinking: showThinking),
                step: step,
                totalSteps: totalSteps,
                canvasIndex: canvasIndex,
                blockComplete: blockComplete
            )
        ))
    }

    nonisolated static func cleanedResponse(_ response: String, showThinking: Bool) -> String {
        Gemma4Generator.cleanedResponse(response, showThinking: showThinking)
    }

    private func stableAndConfident(
        _ canvas: MLXArray,
        logits: MLXArray,
        history: inout [MLXArray],
        generationConfig: DiffusionGemmaGenerationConfig
    ) -> Bool {
        let historyLength = max(1, generationConfig.stabilityThreshold)
        let stable = history.count == historyLength && history.allSatisfy {
            MLX.all(canvas .== $0).item(Bool.self)
        }
        history.append(canvas)
        if history.count > historyLength {
            history.removeFirst(history.count - historyLength)
        }
        guard stable else { return false }

        let logProbabilities = logSoftmax(logits.asType(.float32), axis: -1)
        let entropy = (MLXArray(0) - MLX.exp(logProbabilities) * logProbabilities)
            .sum(axis: -1)
            .mean()
        return entropy.item(Float.self) < generationConfig.confidenceThreshold
    }

    private func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        if let modelPath = modelPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !modelPath.isEmpty {
            if let canonicalID = ModelResolver.ModelID(rawValue: modelPath),
               canonicalID.rawValue == DiffusionGemmaResources.modelID {
                if let resolved = ModelResolver().resolveIfPresent(canonicalID) {
                    return resolved.rootURL
                }
                if let installed = ManagedModelResolver.resolveInstalledModel(id: modelPath) {
                    return installed
                }
            } else {
                let url = URL(fileURLWithPath: modelPath).standardizedFileURL
                guard FileManager.default.fileExists(atPath: url.path) else {
                    throw DiffusionGemmaError.unsupportedModelLocation(modelPath)
                }
                return url
            }
        }

        if let modelID = ModelResolver.ModelID(rawValue: modelID),
           let resolved = ModelResolver().resolveIfPresent(modelID) {
            return resolved.rootURL
        }
        if let installed = ManagedModelResolver.resolveInstalledModel(id: modelID) {
            return installed
        }

        do {
            let resolution = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: modelID,
                defaultModelID: DiffusionGemmaResources.modelID,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(ChatProgress(
                            stage: .loadingModel,
                            message: "Downloading DiffusionGemma... \(percent)%"
                        ))
                    case .extracting:
                        progressHandler?(ChatProgress(
                            stage: .loadingModel,
                            message: "Finalizing DiffusionGemma..."
                        ))
                    }
                }
            )
            return resolution.url.standardizedFileURL
        } catch {
            throw DiffusionGemmaError.downloadFailed(error.localizedDescription)
        }
    }
}
