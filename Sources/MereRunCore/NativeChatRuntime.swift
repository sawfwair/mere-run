import Foundation

/// Native runtime invocation shared by command sessions and resident API models.
public enum NativeChatRuntime: Sendable {
    case textChatPsi(Psi3ChatGenerator, modelPath: String?)
    case textChatInkling(InklingGenerator, modelPath: String?)
    case textCode(CodeGenGenerator, modelPath: String?)
    case textChatKlein(Flux2KleinGenerator, modelPath: String?, useStandalone: Bool)
    case textChatGemma4(Gemma4Generator, modelPath: String?)
    case textChatDiffusionGemma(DiffusionGemmaGenerator, modelPath: String?)
    case textChatLaguna(LagunaGenerator, modelPath: String?)
    case textChatQ35(Q35Generator, modelPath: String?)
    case textChatLFM2(LFM2Generator, modelPath: String?)
    case textChatDeepseekV4Flash(DeepseekV4FlashGenerator, modelPath: String?)
    case textChatMuseGlimmer(MuseGlimmerGenerator, modelPath: String?)
    case textChatNemotronH(NemotronHGenerator, modelPath: String?)
    case textChatNemotronOmni(NemotronOmniGenerator, modelPath: String?)

    public func prepare(progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws {
        switch self {
        case .textChatPsi(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatInkling(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textCode(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatKlein(let generator, let modelPath, let useStandalone):
            guard let modelPath else {
                throw Flux2Error.modelNotFound(ModelResolver.ModelID.mebot.rawValue)
            }
            try await generator.prepareChat(
                modelPath: modelPath,
                standalone: useStandalone,
                progressHandler: progressHandler
            )
        case .textChatGemma4(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatDiffusionGemma(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatLaguna(let generator, let modelPath):
            guard let modelPath else {
                throw LagunaError.modelPathRequired
            }
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatQ35(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatLFM2(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatDeepseekV4Flash(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatMuseGlimmer(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatNemotronH(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        case .textChatNemotronOmni(let generator, let modelPath):
            try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
        }
    }

    public func unload() async {
        switch self {
        case .textChatPsi(let generator, _):
            await generator.unload()
        case .textChatInkling(let generator, _):
            await generator.unload()
        case .textCode(let generator, _):
            await generator.unload()
        case .textChatKlein(let generator, _, _):
            await generator.unload()
        case .textChatGemma4(let generator, _):
            await generator.unload()
        case .textChatDiffusionGemma(let generator, _):
            await generator.unload()
        case .textChatLaguna(let generator, _):
            await generator.unload()
        case .textChatQ35(let generator, _):
            await generator.unload()
        case .textChatLFM2(let generator, _):
            await generator.unload()
        case .textChatDeepseekV4Flash(let generator, _):
            await generator.shutdown()
        case .textChatMuseGlimmer(let generator, _):
            await generator.unload()
        case .textChatNemotronH(let generator, _):
            await generator.unload()
        case .textChatNemotronOmni(let generator, _):
            await generator.unload()
        }
    }

    public func prefixKVCacheStats() async -> PrefixKVCacheStats? {
        switch self {
        case .textChatGemma4(let generator, _):
            return await generator.prefixKVCacheStats()
        case .textChatQ35(let generator, _):
            return await generator.prefixKVCacheStats()
        case .textChatLFM2(let generator, _):
            return await generator.prefixKVCacheStats()
        case .textChatPsi, .textChatInkling, .textCode, .textChatKlein, .textChatDiffusionGemma, .textChatLaguna, .textChatDeepseekV4Flash,
             .textChatMuseGlimmer, .textChatNemotronH, .textChatNemotronOmni:
            return nil
        }
    }

    public func continuousBatchingStats() async -> RuntimeDecodeBatchingStats? {
        switch self {
        case .textChatGemma4(let generator, _):
            return await generator.continuousBatchingStats()
        case .textChatLaguna(let generator, _):
            return await generator.continuousBatchingStats()
        case .textChatQ35(let generator, _):
            return await generator.continuousBatchingStats()
        case .textChatLFM2(let generator, _):
            return await generator.continuousBatchingStats()
        case .textChatPsi, .textChatInkling, .textCode, .textChatKlein, .textChatDiffusionGemma, .textChatDeepseekV4Flash, .textChatMuseGlimmer,
             .textChatNemotronH, .textChatNemotronOmni:
            return nil
        }
    }

    public func mtpStats() async -> Gemma4MTPStats? {
        switch self {
        case .textChatGemma4(let generator, _):
            return await generator.mtpStats()
        case .textChatPsi, .textChatInkling, .textCode, .textChatKlein, .textChatDiffusionGemma, .textChatLaguna, .textChatQ35, .textChatLFM2,
             .textChatDeepseekV4Flash, .textChatMuseGlimmer, .textChatNemotronH,
             .textChatNemotronOmni:
            return nil
        }
    }

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        try await ChatGenerationOperation.run(request) {
            try await generate(request, progressHandler: progressHandler)
        }
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        switch self {
        case .textChatPsi(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatInkling(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textCode(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatKlein(let generator, let modelPath, let useStandalone):
            guard let modelPath else {
                throw Flux2Error.modelNotFound(ModelResolver.ModelID.mebot.rawValue)
            }
            if useStandalone {
                return try await generator.chatStandalone(
                    request,
                    modelPath: modelPath,
                    progressHandler: progressHandler
                )
            }
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatGemma4(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatDiffusionGemma(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatLaguna(let generator, let modelPath):
            guard let modelPath else {
                throw LagunaError.modelPathRequired
            }
            return try await generator.chat(
                request,
                modelPath: modelPath,
                progressHandler: progressHandler
            )
        case .textChatQ35(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatLFM2(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatDeepseekV4Flash(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatMuseGlimmer(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatNemotronH(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        case .textChatNemotronOmni(let generator, let modelPath):
            return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
        }
    }

    public func deepseekChatCompletionsURL(
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        switch self {
        case .textChatDeepseekV4Flash(let generator, let modelPath):
            return try await generator.chatCompletionsURL(
                modelPath: modelPath,
                progressHandler: progressHandler
            )
        case .textChatPsi, .textChatInkling, .textCode, .textChatKlein, .textChatGemma4, .textChatDiffusionGemma, .textChatLaguna, .textChatQ35,
             .textChatLFM2, .textChatMuseGlimmer, .textChatNemotronH,
             .textChatNemotronOmni:
            throw ChatRequestIssue("engine", "does not expose a raw chat proxy")
        }
    }
}
