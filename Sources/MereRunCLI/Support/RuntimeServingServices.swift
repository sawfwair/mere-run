import Foundation
import AudioCore
import AudioSTT
import MereRunCore

/// Composes runtime owners independently of HTTP routing and authentication.
/// The outer server owns its machine reservation; these services own request
/// admission and connect text/media residency to the same pressure policy.
struct RuntimeServingServices: Sendable {
    let models: RuntimeModelPool
    let media: APISidecarModelPool
    let admission: RuntimeRequestAdmission
    private let ensureAvailable: @Sendable () throws -> Void
    private let transcriptionExecutor: any SpeechTranscriptionExecutor

    init(
        defaultModelID: String,
        modelPath: String?,
        engine: RuntimeServingEngine,
        maxActiveRequests: Int,
        settingsURL: URL = RuntimeModelSettingsStore.defaultURL(),
        gemma4KVCacheQuantization: Gemma4KVCacheQuantization = Gemma4KVCacheQuantization(),
        memoryPressurePolicy: RuntimeMemoryPressurePolicy = .default
    ) {
        // Continuous batching follows the request-concurrency setting: any
        // --max-active-requests above 1 enables it for the engines that
        // support it, with the per-engine env switches as explicit overrides.
        let batching = RuntimeContinuousBatchingConfiguration(
            maxActiveRequests: maxActiveRequests
        )
        let runtimePool = RuntimeModelPool(
            defaultModelID: defaultModelID,
            defaultEngine: engine,
            startupModelPath: modelPath,
            settingsStore: RuntimeModelSettingsStore(url: settingsURL),
            gemma4KVCacheQuantization: gemma4KVCacheQuantization,
            gemma4ContinuousBatchingEnabled: batching.gemma4,
            lagunaContinuousBatchingEnabled: batching.laguna,
            q35ContinuousBatchingEnabled: batching.q35,
            lfm2ContinuousBatchingEnabled: batching.lfm2,
            memoryPressurePolicy: memoryPressurePolicy
        )
        self.models = runtimePool
        self.media = APISidecarModelPool(
            settingsURL: settingsURL,
            memoryPressurePolicy: memoryPressurePolicy,
            relieveTextModelPressure: {
                _ = await runtimePool.relieveMemoryPressure(preserveDefault: false)
            },
            releaseOneIdleTextModelForLoad: {
                await runtimePool.releaseOneIdleModelForSidecarLoad() != nil
            }
        )
        self.admission = RuntimeRequestAdmission(
            maxActiveRequests: maxActiveRequests,
            pressureProvider: {
                await runtimePool.currentMemoryPressure()
            }
        )
        self.ensureAvailable = { try MLXBundleSupport.ensureAvailable(quiet: true) }
        self.transcriptionExecutor = media
    }

    init(
        models: RuntimeModelPool,
        media: APISidecarModelPool,
        admission: RuntimeRequestAdmission,
        ensureAvailable: @escaping @Sendable () throws -> Void,
        transcriptionExecutor: any SpeechTranscriptionExecutor
    ) {
        self.models = models
        self.media = media
        self.admission = admission
        self.ensureAvailable = ensureAvailable
        self.transcriptionExecutor = transcriptionExecutor
    }

    func prepare(warmupDefaultModel: Bool) async throws {
        try await models.preloadDefault(warmup: warmupDefaultModel)
    }

    func status() async -> RuntimeModelPoolStatus {
        let requests = await admission.snapshot()
        let residents = await media.status()
        return await models.status(admission: requests, sidecars: residents)
    }

    func loadModel(id: String) async throws -> RuntimeModelPoolEntrySnapshot {
        try await withRuntimeRequestAdmission(using: admission) {
            try await models.loadModel(idOrAlias: id)
        }
    }

    func unloadModel(id: String) async throws -> RuntimeModelPoolEntrySnapshot {
        try await withRuntimeRequestAdmission(using: admission) {
            try await models.unloadModel(idOrAlias: id)
        }
    }

    func startChat(
        _ request: OpenAIChatRequest, fallbackLoraPath: String?, contextSize: Int
    ) async throws -> RuntimeChatSession {
        let admitted = try await admission.acquire()
        var session: RuntimeChatSession?
        do {
            try Task.checkCancellation()
            let plan = try await models.makeChatPlan(
                for: request, fallbackLoraPath: fallbackLoraPath, serverContextSize: contextSize
            )
            let prepared = RuntimeChatSession(plan: plan, admission: admitted)
            session = prepared
            await admitted.configure(
                modelID: plan.modelID, streaming: request.stream == true,
                requestedMaxTokens: plan.request.maxTokens, toolCount: plan.request.tools?.count ?? 0
            )
            try Task.checkCancellation()
            return prepared
        } catch {
            let cancelled = Task.isCancelled || error is CancellationError
            if let session {
                await session.finish(cancelled: cancelled)
            } else {
                await admitted.release(cancelled: cancelled)
            }
            throw error
        }
    }

    func transcribe(
        _ plan: SpeechTranscriptionPlan, recording: SpeechTranscriptionRunSession? = nil
    ) async throws -> SpeechTranscriptionOutcome {
        do {
            return try await withRuntimeRequestAdmission(using: admission) {
                try ensureAvailable()
                return try await SpeechTranscriptionOperation.execute(plan, recording: recording, executor: transcriptionExecutor)
            }
        } catch {
            try recording?.fail(error)
            throw error
        }
    }
}
