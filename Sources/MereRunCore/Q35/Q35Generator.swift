import Foundation
import MLX
#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

public actor Q35Generator: ChatGenerator {
    private var model: Q35Model?
    private var tokenizerAndTemplate: Q35TokenizerAndTemplate?
    private var visionTower: Q35VisionTower?
    private var loadedModelPath: String?
    private var loadedConfig: Q35Config?
    private var loadedResources: Q35Resources?

    private let modelId: String

    public init(modelId: String = Q35Resources.defaultModelId) {
        self.modelId = modelId
    }

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        let rootURL = try await resolveModelRoot(modelPath: nil, progressHandler: progressHandler)
        let loadStart = Date()
        try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
        let loadSeconds = Date().timeIntervalSince(loadStart)

        var response = try await generate(
            request,
            progressHandler: progressHandler,
            maxContextLength: Q35Resources.defaultContextLength
        )
        if var timing = response.timing {
            timing.loadSeconds = loadSeconds
            response.timing = timing
        } else {
            response.timing = ChatTiming(loadSeconds: loadSeconds)
        }
        return response
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
            maxContextLength: Q35Resources.defaultContextLength
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
        visionTower = nil
        loadedModelPath = nil
        loadedConfig = nil
        loadedResources = nil
        Memory.clearCache()
    }

    private func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let normalizedRoot = Q35Resources.normalizedRootURL(rootURL)
        if loadedModelPath == normalizedRoot.path, model != nil, tokenizerAndTemplate != nil {
            return
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Q35 config"))
        let configData = try Data(contentsOf: normalizedRoot.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Q35Config.self, from: configData)

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Q35 tokenizer"))
        let tokenizer = try Q35TokenizerAndTemplate.load(
            from: normalizedRoot,
            maxLengthOverride: min(Q35Resources.defaultContextLength, config.textConfig.maxPositionEmbeddings)
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Q35 weights"))
        let q35Model = Q35Model(config: config)
        let resources = Q35Resources(rootURL: normalizedRoot)

        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 4

        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                indexURL: resources.modelIndexURL,
                to: q35Model,
                groupSize: groupSize,
                bits: bits,
                keyMapper: { key in
                    if key.hasPrefix("language_model.") {
                        return String(key.dropFirst("language_model.".count))
                    }
                    return "__unused__.\(key)"
                }
            )
        } else {
            let arrays = try MLX.loadArrays(url: resources.modelWeightsURL)
            let filtered = arrays.filter { $0.key.hasPrefix("language_model.") }
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                filtered,
                to: q35Model,
                groupSize: groupSize,
                bits: bits,
                keyMapper: { key in
                    String(key.dropFirst("language_model.".count))
                }
            )
        }

        let tower = Q35VisionTower(config: config)

        model = q35Model
        tokenizerAndTemplate = tokenizer
        visionTower = tower
        loadedConfig = config
        loadedResources = resources
        loadedModelPath = normalizedRoot.path
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        maxContextLength: Int
    ) async throws -> ChatResponse {
        guard let model,
              let tokenizerAndTemplate,
              let loadedConfig else {
            throw Q35Error.modelNotLoaded
        }

        let messages = request.messages
        let effectiveContext = min(maxContextLength, loadedConfig.textConfig.maxPositionEmbeddings)
        let prefillStart = Date()

        var promptTokens = tokenizerAndTemplate.encodeForGeneration(
            messages: messages,
            tools: nil,
            addGenerationPrompt: true,
            includeThinking: request.showThinking,
            maxLength: effectiveContext
        )
        if promptTokens.count > effectiveContext {
            promptTokens = Array(promptTokens.suffix(effectiveContext))
        }

        let eosSet = Set(loadedConfig.eosTokenIds + [tokenizerAndTemplate.eosTokenId].compactMap { $0 })
        let generationConfig = GenerationConfig(
            maxTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topP: Float(request.topP),
            repetitionPenalty: 1.05,
            repetitionContextSize: 64
        )

        let layerCaches = makeLayerCaches(config: loadedConfig)
        let promptInput = MLXArray(promptTokens.map { Int32($0) }).reshaped(1, promptTokens.count)

        let imageURLs = collectImageURLs(from: messages)
        var logits: MLXArray
        var prefillLength = promptTokens.count

        if imageURLs.isEmpty {
            logits = model(promptInput, cache: layerCaches)
        } else {
            progressHandler?(ChatProgress(stage: .encoding, message: "Encoding images"))
            try ensureVisionWeightsLoaded(progressHandler: progressHandler)

            if let visionTower,
               let imageTokenId = loadedConfig.imageTokenId ?? tokenizerAndTemplate.tokenizer.imageTokenId {
                let replacements = try buildVisionReplacements(
                    imageURLs: imageURLs,
                    visionTower: visionTower
                )

                if replacements.isEmpty {
                    logits = model(promptInput, cache: layerCaches)
                } else {
                    var promptEmbeddings = model.embeddings(for: promptInput)
                    promptEmbeddings = insertVisionEmbeddings(
                        hiddenStates: promptEmbeddings,
                        inputIds: promptInput,
                        imageTokenId: imageTokenId,
                        replacements: replacements
                    )

                    if promptEmbeddings.dim(1) > effectiveContext {
                        promptEmbeddings = promptEmbeddings[0..., (promptEmbeddings.dim(1) - effectiveContext)..., 0...]
                    }
                    prefillLength = promptEmbeddings.dim(1)
                    logits = model(promptInput, cache: layerCaches, inputEmbeddings: promptEmbeddings)
                }
            } else {
                logits = model(promptInput, cache: layerCaches)
            }
        }
        MLX.eval(logits)
        let prefillSeconds = Date().timeIntervalSince(prefillStart)

        let tokenBudget = max(0, min(request.maxTokens, effectiveContext - prefillLength))

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

        return ChatResponse(
            response: decoded,
            tokensGenerated: generated.count,
            timing: ChatTiming(
                loadSeconds: 0,
                prefillSeconds: prefillSeconds,
                decodeSeconds: decodeSeconds
            )
        )
    }

    private func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        guard let profile = Q35Resources.profile(for: modelId) else {
            throw Q35Error.unsupportedModelId(modelId)
        }

        do {
            let root = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: modelPath ?? modelId,
                defaultModelID: profile.modelId,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(ChatProgress(stage: .loadingModel, message: "Downloading model... \(percent)%"))
                    case .extracting:
                        progressHandler?(ChatProgress(stage: .loadingModel, message: "Extracting model..."))
                    }
                }
            )
            return Q35Resources.normalizedRootURL(root.url)
        } catch let error as ManagedModelResolver.ResolverError {
            throw Q35Error.downloadFailed(error.localizedDescription)
        }
    }

    private func mapLoaderError(_ error: PretrainedModelLoader.LoadError) -> Q35Error {
        switch error {
        case .unsupportedModelId(let modelId):
            return .unsupportedModelId(modelId)
        case .missingFiles(let files):
            return .missingFiles(files)
        case .downloadFailed(let message):
            return .downloadFailed(message)
        }
    }

    private func makeLayerCaches(config: Q35Config) -> [Q35LayerCache?] {
        let text = config.textConfig
        let mlpOnly = Set(text.mlpOnlyLayers)
        return (0..<text.numHiddenLayers).map { layerIndex in
            if mlpOnly.contains(layerIndex) {
                return nil
            }
            let layerType = layerIndex < text.layerTypes.count ? text.layerTypes[layerIndex] : "linear_attention"
            if layerType == "full_attention" {
                return .full(KVCacheSimple(step: 256))
            }
            return .linear(Q35LinearCache())
        }
    }

    private func collectImageURLs(from messages: [ChatMessage]) -> [String] {
        messages.compactMap { message in
            guard message.role != .system else { return nil }
            guard let url = message.imageUrl?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !url.isEmpty else { return nil }
            return url
        }
    }

    private func ensureVisionWeightsLoaded(
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws {
        guard let visionTower else { return }
        guard !visionTower.isLoaded else { return }
        guard let loadedResources else { throw Q35Error.modelNotLoaded }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Q35 vision tower"))
        try visionTower.loadWeights(from: loadedResources)
    }

    private func buildVisionReplacements(
        imageURLs: [String],
        visionTower: Q35VisionTower
    ) throws -> [MLXArray] {
        var replacements: [MLXArray] = []
        replacements.reserveCapacity(imageURLs.count)

        for imageURL in imageURLs {
            let prepared = try loadImageTensor(
                from: imageURL,
                patchSize: visionTower.patchSize,
                spatialMergeSize: visionTower.spatialMergeSize
            )
            let embeds = try visionTower.encodeImage(
                pixelValues: prepared.tensor,
                gridTHW: prepared.gridTHW
            )
            replacements.append(embeds)
        }

        return replacements
    }

    private func insertVisionEmbeddings(
        hiddenStates: MLXArray,
        inputIds: MLXArray,
        imageTokenId: Int,
        replacements: [MLXArray]
    ) -> MLXArray {
        guard !replacements.isEmpty else { return hiddenStates }

        let seqLen = hiddenStates.dim(1)
        let tokenArray = inputIds.asType(.int32)
        MLX.eval(tokenArray)
        let tokenValues = tokenArray.asArray(Int32.self)

        var positions: [Int] = []
        positions.reserveCapacity(replacements.count)
        for index in 0..<seqLen where tokenValues[index] == Int32(imageTokenId) {
            positions.append(index)
        }

        guard !positions.isEmpty else { return hiddenStates }
        let pairCount = min(positions.count, replacements.count)

        var parts: [MLXArray] = []
        parts.reserveCapacity(pairCount * 2 + 1)

        var cursor = 0
        for pairIndex in 0..<pairCount {
            let position = positions[pairIndex]
            if position > cursor {
                parts.append(hiddenStates[0..., cursor..<position, 0...])
            }

            var replacement = replacements[pairIndex]
            if replacement.dtype != hiddenStates.dtype {
                replacement = replacement.asType(hiddenStates.dtype)
            }
            parts.append(replacement.expandedDimensions(axis: 0))

            cursor = position + 1
        }

        if cursor < seqLen {
            parts.append(hiddenStates[0..., cursor..., 0...])
        }

        if parts.isEmpty {
            return hiddenStates
        }
        if parts.count == 1 {
            return parts[0]
        }
        return MLX.concatenated(parts, axis: 1)
    }

    #if canImport(CoreGraphics)
    private func loadImageTensor(
        from imageRef: String,
        patchSize: Int,
        spatialMergeSize: Int
    ) throws -> (tensor: MLXArray, gridTHW: (Int, Int, Int)) {
        let image = try loadCGImage(from: imageRef)
        let divisor = max(1, patchSize * max(1, spatialMergeSize))

        let targetWidth = max(divisor, (image.width / divisor) * divisor)
        let targetHeight = max(divisor, (image.height / divisor) * divisor)

        let pixels = try QwenImageIO.resizedPixelArray(
            from: image,
            width: targetWidth,
            height: targetHeight,
            addBatchDimension: true,
            dtype: .float16
        )
        let normalized = (pixels - 0.5) / 0.5
        let gridTHW = (1, targetHeight / patchSize, targetWidth / patchSize)
        return (normalized, gridTHW)
    }

    private func loadCGImage(from imageRef: String) throws -> CGImage {
        if let remoteURL = URL(string: imageRef),
           let scheme = remoteURL.scheme?.lowercased(),
           scheme == "http" || scheme == "https" {
            let data = try Data(contentsOf: remoteURL)
            guard let source = CGImageSourceCreateWithData(data as CFData, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw NSError(
                    domain: "Q35Generator",
                    code: 1002,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to decode image URL: \(imageRef)"]
                )
            }
            return image
        }

        let localURL: URL
        if imageRef.hasPrefix("file://"), let parsed = URL(string: imageRef) {
            localURL = parsed
        } else {
            localURL = URL(fileURLWithPath: imageRef)
        }
        guard FileManager.default.fileExists(atPath: localURL.path) else {
            throw NSError(
                domain: "Q35Generator",
                code: 1003,
                userInfo: [NSLocalizedDescriptionKey: "Image file not found: \(imageRef)"]
            )
        }
        guard let source = CGImageSourceCreateWithURL(localURL as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw NSError(
                domain: "Q35Generator",
                code: 1004,
                userInfo: [NSLocalizedDescriptionKey: "Failed to decode image file: \(imageRef)"]
            )
        }
        return image
    }
    #else
    private func loadImageTensor(
        from _: String,
        patchSize _: Int,
        spatialMergeSize _: Int
    ) throws -> (tensor: MLXArray, gridTHW: (Int, Int, Int)) {
        throw NSError(
            domain: "Q35Generator",
            code: 1001,
            userInfo: [NSLocalizedDescriptionKey: "Image input requires CoreGraphics support."]
        )
    }
    #endif
}
