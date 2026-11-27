#if os(macOS) || os(Linux)
import Foundation

/// Native runtime invocation shared by command sessions and resident API models.
public enum NativeChatRuntime: Sendable {
    case textChatPsi(Psi3ChatGenerator, modelPath: String?)
    case textChatInkling(InklingGenerator, modelPath: String?)
    case textCode(CodeGenGenerator, modelPath: String?)
    case textChatKlein(Flux2KleinGenerator, modelPath: String?, useStandalone: Bool)
    case textChatGemma4(Gemma4Generator, modelPath: String?)
    case textChatDiffusionGemma(DiffusionGemmaGenerator, modelPath: String?)
    case textChatInkling2(Inkling2Generator, modelPath: String?)
    case textChatFlint(FlintGenerator, modelPath: String?)
    case textChatKolibri(KolibriGenerator, modelPath: String?)
    case textChatLaguna(LagunaGenerator, modelPath: String?)
    case textChatQ35(Q35Generator, modelPath: String?)
    case textChatLFM2(LFM2Generator, modelPath: String?)
    case textChatDeepseekV4Flash(DeepseekV4FlashGenerator, modelPath: String?)
    case textChatMuseGlimmer(MuseGlimmerGenerator, modelPath: String?)
    case textChatNemotronH(NemotronHGenerator, modelPath: String?)
    case textChatNemotronBolt(NemotronBoltGenerator, modelPath: String?)
    case textChatNemotronOmni(NemotronOmniGenerator, modelPath: String?)

    var adapter: any NativeChatRuntimeAdapter {
        switch self {
        case .textChatPsi(let generator, let modelPath):
            BoundChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textChatInkling(let generator, let modelPath):
            BoundChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textCode(let generator, let modelPath):
            BoundChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textChatKlein(let generator, let modelPath, let useStandalone):
            KleinChatRuntimeAdapter(generator: generator, modelPath: modelPath, useStandalone: useStandalone)
        case .textChatGemma4(let generator, let modelPath):
            BoundChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textChatDiffusionGemma(let generator, let modelPath):
            BoundChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textChatInkling2(let generator, let modelPath):
            BoundChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textChatFlint(let generator, let modelPath):
            BoundChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textChatKolibri(let generator, let modelPath):
            BoundChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textChatLaguna(let generator, let modelPath):
            LagunaChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textChatQ35(let generator, let modelPath):
            BoundChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textChatLFM2(let generator, let modelPath):
            BoundChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textChatDeepseekV4Flash(let generator, let modelPath):
            DeepseekChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textChatMuseGlimmer(let generator, let modelPath):
            BoundChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textChatNemotronH(let generator, let modelPath):
            BoundChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textChatNemotronBolt(let generator, let modelPath):
            BoundChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        case .textChatNemotronOmni(let generator, let modelPath):
            BoundChatRuntimeAdapter(generator: generator, modelPath: modelPath)
        }
    }

    public func prepare(progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws {
        try await adapter.prepare(progressHandler: progressHandler)
    }

    public func unload() async {
        await adapter.unload()
    }

    public func prefixKVCacheStats() async -> PrefixKVCacheStats? {
        await adapter.prefixKVCacheStats()
    }

    public func continuousBatchingStats() async -> RuntimeDecodeBatchingStats? {
        await adapter.continuousBatchingStats()
    }

    public func mtpStats() async -> Gemma4MTPStats? {
        await adapter.mtpStats()
    }

    public func preparePrompt(_ request: ChatRequest) async throws -> ChatPreparedPrompt {
        try await adapter.preparePrompt(request)
    }

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        try await ChatGenerationOperation.run(request) {
            try await adapter.generate(request, progressHandler: progressHandler)
        }
    }

    public func deepseekChatCompletionsURL(
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        try await adapter.deepseekChatCompletionsURL(progressHandler: progressHandler)
    }
}
#endif
