#if os(macOS) || os(Linux)
import Foundation

/// Uniform lifecycle and optional capability surface for chat generators behind `NativeChatRuntime`.
///
/// New chat runtimes conform here (directly via `StandardNativeChatGenerator` or via a small
/// adapter when lifecycle differs) rather than repeating switch arms across every method on
/// `NativeChatRuntime`.
protocol NativeChatRuntimeAdapter: Sendable {
    func prepare(progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws
    func unload() async
    func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse
    func prefixKVCacheStats() async -> PrefixKVCacheStats?
    func continuousBatchingStats() async -> RuntimeDecodeBatchingStats?
    func mtpStats() async -> Gemma4MTPStats?
    func preparePrompt(_ request: ChatRequest) async throws -> ChatPreparedPrompt
    func deepseekChatCompletionsURL(
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL
    func diagnostics() async -> NativeChatDiagnostics
}

extension NativeChatRuntimeAdapter {
    func prefixKVCacheStats() async -> PrefixKVCacheStats? { nil }
    func continuousBatchingStats() async -> RuntimeDecodeBatchingStats? { nil }
    func mtpStats() async -> Gemma4MTPStats? { nil }
    func preparePrompt(_ request: ChatRequest) async throws -> ChatPreparedPrompt {
        throw ChatRequestIssue("count_tokens", "this runtime has no exact shared prompt preparation adapter")
    }
    func deepseekChatCompletionsURL(
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        throw ChatRequestIssue("engine", "does not expose a raw chat proxy")
    }
    func diagnostics() async -> NativeChatDiagnostics { NativeChatDiagnostics() }
}

/// Standard `(modelPath: String?)` chat generator interface shared by most native runtimes.
protocol StandardNativeChatGenerator: Sendable {
    func prepare(modelPath: String?, progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws
    func unload() async
    func chat(
        _ request: ChatRequest,
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse

    func runtimePrefixKVCacheStats() async -> PrefixKVCacheStats?
    func runtimeContinuousBatchingStats() async -> RuntimeDecodeBatchingStats?
    func runtimeMTPStats() async -> Gemma4MTPStats?
    func runtimePreparePrompt(_ request: ChatRequest, modelPath: String?) async throws -> ChatPreparedPrompt
    func runtimeDiagnostics() async -> NativeChatDiagnostics
}

extension StandardNativeChatGenerator {
    func runtimePrefixKVCacheStats() async -> PrefixKVCacheStats? { nil }
    func runtimeContinuousBatchingStats() async -> RuntimeDecodeBatchingStats? { nil }
    func runtimeMTPStats() async -> Gemma4MTPStats? { nil }
    func runtimePreparePrompt(_ request: ChatRequest, modelPath: String?) async throws -> ChatPreparedPrompt {
        throw ChatRequestIssue("count_tokens", "this runtime has no exact shared prompt preparation adapter")
    }
    func runtimeDiagnostics() async -> NativeChatDiagnostics { NativeChatDiagnostics() }
}

struct BoundChatRuntimeAdapter<Generator: StandardNativeChatGenerator>: NativeChatRuntimeAdapter {
    let generator: Generator
    let modelPath: String?

    func prepare(progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws {
        try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
    }

    func unload() async {
        await generator.unload()
    }

    func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
    }

    func prefixKVCacheStats() async -> PrefixKVCacheStats? {
        await generator.runtimePrefixKVCacheStats()
    }

    func continuousBatchingStats() async -> RuntimeDecodeBatchingStats? {
        await generator.runtimeContinuousBatchingStats()
    }

    func mtpStats() async -> Gemma4MTPStats? {
        await generator.runtimeMTPStats()
    }

    func preparePrompt(_ request: ChatRequest) async throws -> ChatPreparedPrompt {
        try await generator.runtimePreparePrompt(request, modelPath: modelPath)
    }

    func diagnostics() async -> NativeChatDiagnostics {
        await generator.runtimeDiagnostics()
    }
}

// MARK: - Standard generator conformances

extension Psi3ChatGenerator: StandardNativeChatGenerator {}
extension InklingGenerator: StandardNativeChatGenerator {}
extension CodeGenGenerator: StandardNativeChatGenerator {}
extension DiffusionGemmaGenerator: StandardNativeChatGenerator {}
extension FlintGenerator: StandardNativeChatGenerator {}
extension KolibriGenerator: StandardNativeChatGenerator {}
extension NemotronBoltGenerator: StandardNativeChatGenerator {}
extension NemotronOmniGenerator: StandardNativeChatGenerator {}

extension Gemma4Generator: StandardNativeChatGenerator {
    func runtimePrefixKVCacheStats() async -> PrefixKVCacheStats? { prefixKVCacheStats() }
    func runtimeContinuousBatchingStats() async -> RuntimeDecodeBatchingStats? { continuousBatchingStats() }
    func runtimeMTPStats() async -> Gemma4MTPStats? { mtpStats() }
    func runtimePreparePrompt(_ request: ChatRequest, modelPath: String?) async throws -> ChatPreparedPrompt {
        try await preparePrompt(request, modelPath: modelPath)
    }
    func runtimeDiagnostics() async -> NativeChatDiagnostics {
        NativeChatDiagnostics(gemma4MTP: mtpStats())
    }
}

extension Q35Generator: StandardNativeChatGenerator {
    func runtimePrefixKVCacheStats() async -> PrefixKVCacheStats? { prefixKVCacheStats() }
    func runtimeContinuousBatchingStats() async -> RuntimeDecodeBatchingStats? { continuousBatchingStats() }
    func runtimePreparePrompt(_ request: ChatRequest, modelPath: String?) async throws -> ChatPreparedPrompt {
        try await preparePrompt(request, modelPath: modelPath)
    }
}

extension LFM2Generator: StandardNativeChatGenerator {
    func runtimePrefixKVCacheStats() async -> PrefixKVCacheStats? { prefixKVCacheStats() }
    func runtimeContinuousBatchingStats() async -> RuntimeDecodeBatchingStats? { continuousBatchingStats() }
    func runtimeDiagnostics() async -> NativeChatDiagnostics {
        NativeChatDiagnostics(lfm2DSpark: dsparkStats())
    }
}

extension Inkling2Generator: StandardNativeChatGenerator {
    func runtimePrefixKVCacheStats() async -> PrefixKVCacheStats? { prefixKVCacheStats() }
    func runtimeDiagnostics() async -> NativeChatDiagnostics {
        NativeChatDiagnostics(inkling2DSpark: dsparkStats())
    }
}

extension MuseGlimmerGenerator: StandardNativeChatGenerator {
    func runtimePreparePrompt(_ request: ChatRequest, modelPath: String?) async throws -> ChatPreparedPrompt {
        try await preparePrompt(request, modelPath: modelPath)
    }
    func runtimeDiagnostics() async -> NativeChatDiagnostics {
        NativeChatDiagnostics(museDFlash: dflashStats())
    }
}

extension NemotronHGenerator: StandardNativeChatGenerator {
    func runtimeDiagnostics() async -> NativeChatDiagnostics {
        NativeChatDiagnostics(nemotronDSpark: dsparkStats())
    }
}

// MARK: - Specialized lifecycle adapters

struct KleinChatRuntimeAdapter: NativeChatRuntimeAdapter {
    let generator: Flux2KleinGenerator
    let modelPath: String?
    let useStandalone: Bool

    func prepare(progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws {
        guard let modelPath else {
            throw Flux2Error.modelNotFound(ModelResolver.ModelID.mebot.rawValue)
        }
        try await generator.prepareChat(
            modelPath: modelPath,
            standalone: useStandalone,
            progressHandler: progressHandler
        )
    }

    func unload() async {
        await generator.unload()
    }

    func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
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
    }
}

struct LagunaChatRuntimeAdapter: NativeChatRuntimeAdapter {
    let generator: LagunaGenerator
    let modelPath: String?

    func prepare(progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws {
        guard let modelPath else {
            throw LagunaError.modelPathRequired
        }
        try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
    }

    func unload() async {
        await generator.unload()
    }

    func continuousBatchingStats() async -> RuntimeDecodeBatchingStats? {
        await generator.continuousBatchingStats()
    }

    func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        guard let modelPath else {
            throw LagunaError.modelPathRequired
        }
        return try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
    }

    func diagnostics() async -> NativeChatDiagnostics {
        NativeChatDiagnostics(lagunaDFlash: await generator.dflashStats())
    }
}

struct DeepseekChatRuntimeAdapter: NativeChatRuntimeAdapter {
    let generator: DeepseekV4FlashGenerator
    let modelPath: String?

    func prepare(progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws {
        try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
    }

    func unload() async {
        await generator.shutdown()
    }

    func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
    }

    func deepseekChatCompletionsURL(
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        try await generator.chatCompletionsURL(
            modelPath: modelPath,
            progressHandler: progressHandler
        )
    }
}
#endif
