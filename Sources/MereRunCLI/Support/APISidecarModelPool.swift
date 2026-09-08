import Foundation
import AudioCore
import AudioSTT
import AudioTTS
import MereRunCore
import MLX

typealias APISidecarImageKind = ImageGenerationBackend

private struct APISidecarImageKey: Hashable, Sendable {
    let kind: APISidecarImageKind
    let modelID: String
    let modelPath: String
}

private struct APISidecarSpeechKey: Hashable, Sendable {
    let modelID: String
    let modelPath: String?
}

private struct APISidecarASRKey: Hashable, Sendable {
    let backend: ASRResolvedBackend
    let modelID: String
    let modelPath: String?
}

private struct APISidecarEmbeddingKey: Hashable, Sendable {
    let modelID: String
    let modelPath: String
}

enum APISidecarLane: Int, CaseIterable, Hashable, Sendable {
    case image
    case speech
    case transcription
    case embedding
}

enum APISidecarModelPoolError: LocalizedError, Equatable {
    case memoryPressure

    var errorDescription: String? {
        switch self {
        case .memoryPressure:
            return "Memory guard blocked a cold media-model load while process pressure remained critical."
        }
    }
}

struct APISidecarEvictionCandidate: Equatable, Sendable {
    let lane: APISidecarLane
    let loaded: Bool
    let lastAccess: Date?
    let activeRequests: Int
    let queuedRequests: Int
    let pinned: Bool
    let ttlSeconds: Int
}

struct APISidecarEvictionDecision: Equatable, Sendable {
    let lane: APISidecarLane
    let reason: RuntimeSidecarEvictionReason
}

enum APISidecarEvictionPlanner {
    static func decisions(
        candidates: [APISidecarEvictionCandidate],
        now: Date,
        pressure: RuntimeMemoryPressureLevel,
        excluding excludedLanes: Set<APISidecarLane> = []
    ) -> [APISidecarEvictionDecision] {
        let values = candidates.map(typedCandidate)
        let expired = RuntimeEvictionPlanner.expired(values, now: now, excluding: excludedLanes)
        let pressure = RuntimeEvictionPlanner.memoryPressure(
            values, pressure: pressure, excluding: excludedLanes.union(expired)
        )
        return expired.map { .init(lane: $0, reason: .ttl) }
            + pressure.map { .init(lane: $0, reason: .memoryPressure) }
    }

    static func nextMemoryPressureDecision(
        candidates: [APISidecarEvictionCandidate],
        excluding excludedLanes: Set<APISidecarLane> = []
    ) -> APISidecarEvictionDecision? {
        RuntimeEvictionPlanner.memoryPressure(
            candidates.map(typedCandidate), pressure: .elevated, excluding: excludedLanes
        ).first.map { .init(lane: $0, reason: .memoryPressure) }
    }

    private static func typedCandidate(_ candidate: APISidecarEvictionCandidate) -> RuntimeEvictionCandidate<APISidecarLane> {
        RuntimeEvictionCandidate(
            key: candidate.lane, sortKey: String(candidate.lane.rawValue), loaded: candidate.loaded,
            lastAccess: candidate.lastAccess, activeRequests: candidate.activeRequests,
            queuedRequests: candidate.queuedRequests, pinned: candidate.pinned, ttlSeconds: candidate.ttlSeconds
        )
    }
}


/// Non-actor image generators are safe here because the resident slot holds an
/// exclusive lease for their entire lifetime of use.
private enum APISidecarImageGenerator: @unchecked Sendable {
    case flux1(Flux1Generator)
    case flux2Klein(Flux2KleinGenerator)
    case zImageTurbo(ZImageTurboGenerator)
    case hiDreamO1(HiDreamO1Generator)
    case senseNovaU15(SenseNovaU15Generator)
    case krea2(Krea2Generator)
    case ideogram4(Ideogram4Generator)
    case qwenImageEdit(QwenImageEditGenerator)
}

private enum APISidecarASRGenerator: Sendable {
    case qwen(Qwen3ASRGenerator)
    case parakeet(ParakeetGenerator)
}

private final class APISidecarEmbeddingRuntime: @unchecked Sendable {
    private(set) var model: Qwen3EmbeddingModel?

    init(model: Qwen3EmbeddingModel) {
        self.model = model
    }

    func unload() {
        model = nil
        Memory.clearCache()
    }
}

struct APISidecarEmbeddingResult: Sendable {
    let embeddings: [[Float]]
    let tokenCounts: [Int]
}

/// Resident runtimes for the non-chat OpenAI-compatible endpoints.
///
/// One embedding runtime, one image runtime, one TTS runtime, and one ASR
/// runtime may be resident at a time. This bounds memory even when callers
/// provide different local model paths, while repeated requests avoid reloads.
struct APISidecarModelPool: Sendable, CLIASRTranscriptionExecutor {
    static let defaultIdleTTLSeconds = 300

    private let imageSlot: APISidecarResidentSlot<APISidecarImageKey, APISidecarImageGenerator>
    private let speechSlot: APISidecarResidentSlot<APISidecarSpeechKey, Qwen3TTSGenerator>
    private let asrSlot: APISidecarResidentSlot<APISidecarASRKey, APISidecarASRGenerator>
    private let embeddingSlot: APISidecarResidentSlot<APISidecarEmbeddingKey, APISidecarEmbeddingRuntime>
    private let operationCoordinator: APISidecarOperationCoordinator
    private let settingsURL: URL
    private let memoryPressurePolicy: RuntimeMemoryPressurePolicy
    private let defaultIdleTTLSeconds: Int
    private let currentDate: @Sendable () -> Date
    private let currentMemorySample: @Sendable () -> RuntimeMemorySample
    private let relieveTextModelPressure: @Sendable () async -> Void
    private let releaseOneIdleTextModelForLoad: @Sendable () async -> Bool

    init(
        settingsURL: URL = RuntimeModelSettingsStore.defaultURL(),
        memoryPressurePolicy: RuntimeMemoryPressurePolicy = .default,
        defaultIdleTTLSeconds: Int = APISidecarModelPool.defaultIdleTTLSeconds,
        currentDate: @escaping @Sendable () -> Date = { Date() },
        currentMemorySample: @escaping @Sendable () -> RuntimeMemorySample = { RuntimeMemorySample.current() },
        relieveTextModelPressure: @escaping @Sendable () async -> Void = {},
        releaseOneIdleTextModelForLoad: @escaping @Sendable () async -> Bool = { false }
    ) {
        precondition(defaultIdleTTLSeconds > 0, "Sidecar idle TTL must be positive")
        self.settingsURL = settingsURL
        self.memoryPressurePolicy = memoryPressurePolicy
        self.defaultIdleTTLSeconds = defaultIdleTTLSeconds
        self.currentDate = currentDate
        self.currentMemorySample = currentMemorySample
        self.relieveTextModelPressure = relieveTextModelPressure
        self.releaseOneIdleTextModelForLoad = releaseOneIdleTextModelForLoad
        self.imageSlot = APISidecarResidentSlot(currentDate: currentDate)
        self.speechSlot = APISidecarResidentSlot(currentDate: currentDate)
        self.asrSlot = APISidecarResidentSlot(currentDate: currentDate)
        self.embeddingSlot = APISidecarResidentSlot(currentDate: currentDate)
        self.operationCoordinator = APISidecarOperationCoordinator()
    }

    func generateImage(
        kind: APISidecarImageKind,
        modelID: String,
        modelPath: String,
        request: GenerationRequest
    ) async throws -> GenerationResult {
        let residentModelID = Self.canonicalImageModelID(kind: kind, modelID: modelID)
        let key = APISidecarImageKey(
            kind: kind,
            modelID: residentModelID,
            modelPath: normalizedPath(modelPath)
        )
        return try await withSidecarPressureCoordination(excluding: [.image]) {
            let lifecycle = lifecycleSettings(modelID: residentModelID, settings: loadedSettings())
            return try await imageSlot.withValue(
                for: key,
                idleTTL: .seconds(lifecycle.ttlSeconds),
                pinned: lifecycle.pinned,
                currentIdlePolicy: { key in
                    let current = lifecycleSettings(
                        modelID: key.modelID,
                        settings: loadedSettings()
                    )
                    return APISidecarResidentIdlePolicy(
                        pinned: current.pinned,
                        ttl: .seconds(current.ttlSeconds)
                    )
                },
                operationCoordinator: operationCoordinator,
                // Image pipelines may intentionally release and reload staged
                // components between requests (for example Qwen Image Edit's
                // text encoder), so slot readiness alone cannot prove a warm
                // operation is allocation-free.
                forceColdOperation: true,
                prepareForColdOperation: { residentNeedsLoad in
                    let estimate = estimatedImageLoadBytes(
                        kind: kind,
                        modelID: residentModelID,
                        modelPath: key.modelPath,
                        width: request.width,
                        height: request.height,
                        residentNeedsLoad: residentNeedsLoad
                    )
                    try await prepareForColdSidecarLoad(
                        excluding: [.image],
                        estimatedLoadBytes: estimate
                    )
                },
                make: {
                    switch kind {
                    case .flux1:
                        return .flux1(Flux1Generator())
                    case .flux2Klein:
                        return .flux2Klein(Flux2KleinGenerator())
                    case .zImageTurbo:
                        return .zImageTurbo(ZImageTurboGenerator())
                    case .hiDreamO1:
                        return .hiDreamO1(HiDreamO1Generator())
                    case .senseNovaU15:
                        return .senseNovaU15(SenseNovaU15Generator())
                    case .krea2:
                        return .krea2(Krea2Generator())
                    case .ideogram4:
                        return .ideogram4(Ideogram4Generator())
                    case .qwenImageEdit:
                        return .qwenImageEdit(QwenImageEditGenerator())
                    }
                },
                unload: { generator in
                    switch generator {
                    case .flux1(let generator):
                        await generator.unload()
                    case .flux2Klein(let generator):
                        await generator.unload()
                    case .zImageTurbo(let generator):
                        await generator.unload()
                    case .hiDreamO1(let generator):
                        generator.unload()
                    case .senseNovaU15(let generator):
                        generator.unload()
                    case .krea2(let generator):
                        generator.unload()
                    case .ideogram4(let generator):
                        generator.unload()
                    case .qwenImageEdit(let generator):
                        await generator.clearCache()
                    }
                },
                operation: { generator in
                    switch generator {
                    case .flux1(let generator):
                        return try await generator.generate(request, progressHandler: nil)
                    case .flux2Klein(let generator):
                        return try await generator.generate(request, progressHandler: nil)
                    case .zImageTurbo(let generator):
                        return try await generator.generate(request, progressHandler: nil)
                    case .hiDreamO1(let generator):
                        return try await generator.generate(request, progressHandler: nil)
                    case .senseNovaU15(let generator):
                        return try await generator.generate(request, progressHandler: nil)
                    case .krea2(let generator):
                        return try await generator.generate(request, progressHandler: nil)
                    case .ideogram4(let generator):
                        return try await generator.generate(request, progressHandler: nil)
                    case .qwenImageEdit(let generator):
                        return try await generator.generate(request, progressHandler: nil)
                    }
                }
            )
        }
    }

    func synthesizeSpeech(
        modelID: String,
        modelPath: String?,
        request: TTSRequest
    ) async throws -> TTSResult {
        let key = APISidecarSpeechKey(
            modelID: modelID,
            modelPath: modelPath.map(normalizedPath)
        )
        return try await withSidecarPressureCoordination(excluding: [.speech]) {
            let lifecycle = lifecycleSettings(modelID: modelID, settings: loadedSettings())
            return try await speechSlot.withValue(
                for: key,
                idleTTL: .seconds(lifecycle.ttlSeconds),
                pinned: lifecycle.pinned,
                currentIdlePolicy: { key in
                    let current = lifecycleSettings(
                        modelID: key.modelID,
                        settings: loadedSettings()
                    )
                    return APISidecarResidentIdlePolicy(
                        pinned: current.pinned,
                        ttl: .seconds(current.ttlSeconds)
                    )
                },
                operationCoordinator: operationCoordinator,
                prepareForColdOperation: { _ in
                    try await prepareForColdSidecarLoad(
                        excluding: [.speech],
                        estimatedLoadBytes: estimatedLoadBytes(
                            modelID: modelID,
                            modelPath: modelPath,
                            minimumBytes: 10 * 1_073_741_824
                        )
                    )
                },
                make: { Qwen3TTSGenerator(modelId: modelID) },
                unload: { generator in await generator.unload() },
                operation: { generator in
                    try await generator.generate(
                        request,
                        modelPath: modelPath,
                        progressHandler: nil
                    )
                }
            )
        }
    }

    func transcribeQwen(
        request: ASRRequest,
        modelID: String,
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        try await transcribe(
            request: request,
            key: APISidecarASRKey(
                backend: .qwen,
                modelID: modelID,
                modelPath: modelPath.map(normalizedPath)
            ),
            progressHandler: progressHandler
        )
    }

    func transcribeParakeet(
        request: ASRRequest,
        modelID: String,
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        try await transcribe(
            request: request,
            key: APISidecarASRKey(
                backend: .parakeet,
                modelID: modelID,
                modelPath: modelPath.map(normalizedPath)
            ),
            progressHandler: progressHandler
        )
    }

    private func transcribe(
        request: ASRRequest,
        key: APISidecarASRKey,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        return try await withSidecarPressureCoordination(excluding: [.transcription]) {
            let lifecycle = lifecycleSettings(modelID: key.modelID, settings: loadedSettings())
            return try await asrSlot.withValue(
                for: key,
                idleTTL: .seconds(lifecycle.ttlSeconds),
                pinned: lifecycle.pinned,
                currentIdlePolicy: { key in
                    let current = lifecycleSettings(
                        modelID: key.modelID,
                        settings: loadedSettings()
                    )
                    return APISidecarResidentIdlePolicy(
                        pinned: current.pinned,
                        ttl: .seconds(current.ttlSeconds)
                    )
                },
                operationCoordinator: operationCoordinator,
                prepareForColdOperation: { _ in
                    try await prepareForColdSidecarLoad(
                        excluding: [.transcription],
                        estimatedLoadBytes: estimatedLoadBytes(
                            modelID: key.modelID,
                            modelPath: key.modelPath,
                            minimumBytes: 5 * 1_073_741_824
                        )
                    )
                },
                make: {
                    switch key.backend {
                    case .qwen:
                        return .qwen(Qwen3ASRGenerator(modelId: key.modelID))
                    case .parakeet:
                        return .parakeet(ParakeetGenerator(modelId: key.modelID))
                    }
                },
                unload: { generator in
                    switch generator {
                    case .qwen(let generator):
                        await generator.unload()
                    case .parakeet(let generator):
                        await generator.unload()
                    }
                },
                operation: { generator in
                    switch generator {
                    case .qwen(let generator):
                        return try await generator.transcribe(
                            request,
                            modelPath: key.modelPath,
                            progressHandler: progressHandler
                        )
                    case .parakeet(let generator):
                        return try await generator.transcribe(
                            request,
                            modelPath: key.modelPath,
                            progressHandler: progressHandler
                        )
                    }
                }
            )
        }
    }

    func embed(
        modelID: String,
        modelPath: String,
        texts: [String]
    ) async throws -> APISidecarEmbeddingResult {
        let key = APISidecarEmbeddingKey(
            modelID: modelID,
            modelPath: normalizedPath(modelPath)
        )
        return try await withSidecarPressureCoordination(excluding: [.embedding]) {
            let lifecycle = lifecycleSettings(modelID: modelID, settings: loadedSettings())
            return try await embeddingSlot.withValue(
                for: key,
                idleTTL: .seconds(lifecycle.ttlSeconds),
                pinned: lifecycle.pinned,
                currentIdlePolicy: { key in
                    let current = lifecycleSettings(
                        modelID: key.modelID,
                        settings: loadedSettings()
                    )
                    return APISidecarResidentIdlePolicy(
                        pinned: current.pinned,
                        ttl: .seconds(current.ttlSeconds)
                    )
                },
                operationCoordinator: operationCoordinator,
                prepareForColdOperation: { _ in
                    try await prepareForColdSidecarLoad(
                        excluding: [.embedding],
                        estimatedLoadBytes: estimatedLoadBytes(
                            modelID: modelID,
                            modelPath: key.modelPath,
                            minimumBytes: 3 * 1_073_741_824
                        )
                    )
                },
                make: {
                    APISidecarEmbeddingRuntime(
                        model: try Qwen3EmbeddingModel(
                            resources: Qwen3EmbeddingResources(
                                rootURL: URL(fileURLWithPath: key.modelPath, isDirectory: true)
                            )
                        )
                    )
                },
                unload: { runtime in runtime.unload() },
                operation: { runtime in
                    guard let model = runtime.model else {
                        throw CancellationError()
                    }
                    let result = try model.embed(
                        texts: texts,
                        maxTokens: Qwen3EmbeddingModel.defaultMaxPaddedTokensPerBatch,
                        maxPaddedTokensPerBatch: Qwen3EmbeddingModel.defaultMaxPaddedTokensPerBatch
                    )
                    return APISidecarEmbeddingResult(
                        embeddings: result.embeddings,
                        tokenCounts: result.tokenCounts
                    )
                }
            )
        }
    }

    func status(
        now: Date? = nil,
        memorySample: RuntimeMemorySample? = nil
    ) async -> RuntimeSidecarPoolStatus {
        let timestamp = now ?? currentDate()
        let sample = memorySample ?? currentMemorySample()
        await evictIdleResidents(now: timestamp, memorySample: sample)
        let states = await states()
        let settings = loadedSettings()
        let residents = residentSnapshots(states: states, settings: settings)
        return RuntimeSidecarPoolStatus(
            defaultIdleTTLSeconds: defaultIdleTTLSeconds,
            pressure: memoryPressurePolicy.pressure(for: sample).rawValue,
            loadedCount: residents.filter(\.loaded).count,
            activeRequests: residents.reduce(0) { $0 + $1.activeRequests },
            queuedRequests: residents.reduce(0) { $0 + $1.queuedRequests },
            residents: residents
        )
    }

    private struct SlotStates: Sendable {
        let image: APISidecarResidentSlotState<APISidecarImageKey>
        let speech: APISidecarResidentSlotState<APISidecarSpeechKey>
        let transcription: APISidecarResidentSlotState<APISidecarASRKey>
        let embedding: APISidecarResidentSlotState<APISidecarEmbeddingKey>
    }

    private func states() async -> SlotStates {
        async let image = imageSlot.state()
        async let speech = speechSlot.state()
        async let transcription = asrSlot.state()
        async let embedding = embeddingSlot.state()
        return await SlotStates(
            image: image,
            speech: speech,
            transcription: transcription,
            embedding: embedding
        )
    }

    /// Gives already-resident text runtimes the first opportunity to release
    /// memory before a sidecar constructs another large model. Pressure is
    /// sampled again after text eviction; sidecar LRU eviction only runs if
    /// the process remains above the configured guard.
    func prepareForSidecarLoad(
        excluding excludedLanes: Set<APISidecarLane> = []
    ) async {
        await evictIdleResidents(
            pressure: .nominal,
            excluding: excludedLanes
        )
        let initialPressure = memoryPressurePolicy.pressure(for: currentMemorySample())
        switch initialPressure {
        case .elevated, .critical:
            await relieveTextModelPressure()
        case .disabled, .unknown, .nominal:
            return
        }
        await evictIdleResidents(excluding: excludedLanes)
    }

    /// Cold sidecar work is exclusive across media lanes. Re-sample pressure
    /// under that exclusive lease and release idle text/sidecar residents when
    /// the configured guard is already elevated. Nominal high-memory systems
    /// retain warm peers; the post-cold-load pass handles any pressure caused
    /// by the new resident itself.
    private func prepareForColdSidecarLoad(
        excluding excludedLanes: Set<APISidecarLane>,
        estimatedLoadBytes: UInt64?
    ) async throws {
        let initialSample = currentMemorySample()
        let initialPressure = memoryPressurePolicy.pressure(for: initialSample)
        let projectedPressure = estimatedLoadBytes.map {
            memoryPressurePolicy.projectedPressure(for: initialSample, additionalBytes: $0)
        }

        if projectedPressure == .critical, let estimatedLoadBytes {
            while memoryPressurePolicy.projectedPressure(
                for: currentMemorySample(),
                additionalBytes: estimatedLoadBytes
            ) == .critical {
                if await releaseOneIdleTextModelForLoad() {
                    continue
                }
                guard await evictOneIdleResident(excluding: excludedLanes) else {
                    break
                }
                await Task.yield()
            }
        } else if initialPressure == .elevated || initialPressure == .critical {
            await relieveTextModelPressure()
            await evictIdleResidents(excluding: excludedLanes)
        }

        let finalSample = currentMemorySample()
        let finalPressure = memoryPressurePolicy.pressure(for: finalSample)
        let finalProjectedPressure = estimatedLoadBytes.map {
            memoryPressurePolicy.projectedPressure(for: finalSample, additionalBytes: $0)
        }
        if finalPressure == .critical || finalProjectedPressure == .critical {
            throw APISidecarModelPoolError.memoryPressure
        }
    }

    func withSidecarPressureCoordination<Result: Sendable>(
        excluding excludedLanes: Set<APISidecarLane>,
        operation: @Sendable () async throws -> Result
    ) async throws -> Result {
        await prepareForSidecarLoad(excluding: excludedLanes)
        do {
            let result = try await operation()
            await rebalanceAfterSidecarLoad(excluding: [])
            return result
        } catch {
            await rebalanceAfterSidecarLoad(excluding: [])
            throw error
        }
    }

    private func rebalanceAfterSidecarLoad(
        excluding excludedLanes: Set<APISidecarLane>
    ) async {
        let pressure = memoryPressurePolicy.pressure(for: currentMemorySample())
        switch pressure {
        case .elevated, .critical:
            await relieveTextModelPressure()
            await evictIdleResidents(excluding: excludedLanes)
        case .disabled, .unknown, .nominal:
            break
        }
    }

    private func evictIdleResidents(
        now: Date? = nil,
        memorySample: RuntimeMemorySample? = nil,
        pressure: RuntimeMemoryPressureLevel? = nil,
        excluding excludedLanes: Set<APISidecarLane> = []
    ) async {
        let states = await states()
        let settings = loadedSettings()
        let decisions = APISidecarEvictionPlanner.decisions(
            candidates: evictionCandidates(states: states, settings: settings),
            now: now ?? currentDate(),
            pressure: pressure
                ?? memoryPressurePolicy.pressure(for: memorySample ?? currentMemorySample()),
            excluding: excludedLanes
        )

        for decision in decisions {
            switch decision.lane {
            case .image:
                guard let key = states.image.residentKey else { continue }
                _ = await imageSlot.evictIfIdle(
                    expectedKey: key,
                    reason: decision.reason,
                    using: Self.unloadImage
                )
            case .speech:
                guard let key = states.speech.residentKey else { continue }
                _ = await speechSlot.evictIfIdle(
                    expectedKey: key,
                    reason: decision.reason,
                    using: { generator in await generator.unload() }
                )
            case .transcription:
                guard let key = states.transcription.residentKey else { continue }
                _ = await asrSlot.evictIfIdle(
                    expectedKey: key,
                    reason: decision.reason,
                    using: Self.unloadASR
                )
            case .embedding:
                guard let key = states.embedding.residentKey else { continue }
                _ = await embeddingSlot.evictIfIdle(
                    expectedKey: key,
                    reason: decision.reason,
                    using: { runtime in runtime.unload() }
                )
            }
        }
    }

    private func evictOneIdleResident(
        excluding excludedLanes: Set<APISidecarLane>
    ) async -> Bool {
        let states = await states()
        let settings = loadedSettings()
        guard let decision = APISidecarEvictionPlanner.nextMemoryPressureDecision(
            candidates: evictionCandidates(states: states, settings: settings),
            excluding: excludedLanes
        ) else {
            return false
        }

        switch decision.lane {
        case .image:
            guard let key = states.image.residentKey else { return false }
            return await imageSlot.evictIfIdle(
                expectedKey: key,
                reason: decision.reason,
                using: Self.unloadImage
            )
        case .speech:
            guard let key = states.speech.residentKey else { return false }
            return await speechSlot.evictIfIdle(
                expectedKey: key,
                reason: decision.reason,
                using: { generator in await generator.unload() }
            )
        case .transcription:
            guard let key = states.transcription.residentKey else { return false }
            return await asrSlot.evictIfIdle(
                expectedKey: key,
                reason: decision.reason,
                using: Self.unloadASR
            )
        case .embedding:
            guard let key = states.embedding.residentKey else { return false }
            return await embeddingSlot.evictIfIdle(
                expectedKey: key,
                reason: decision.reason,
                using: { runtime in runtime.unload() }
            )
        }
    }

    private func evictionCandidates(
        states: SlotStates,
        settings: [String: RuntimeModelSettings]
    ) -> [APISidecarEvictionCandidate] {
        let imageSettings = lifecycleSettings(
            modelID: states.image.residentKey?.modelID,
            settings: settings
        )
        let speechSettings = lifecycleSettings(
            modelID: states.speech.residentKey?.modelID,
            settings: settings
        )
        let transcriptionSettings = lifecycleSettings(
            modelID: states.transcription.residentKey?.modelID,
            settings: settings
        )
        let embeddingSettings = lifecycleSettings(
            modelID: states.embedding.residentKey?.modelID,
            settings: settings
        )
        return [
            APISidecarEvictionCandidate(
                lane: .image,
                loaded: states.image.residentKey != nil,
                lastAccess: states.image.lastAccess,
                activeRequests: states.image.activeRequests,
                queuedRequests: states.image.queuedRequests,
                pinned: imageSettings.pinned,
                ttlSeconds: imageSettings.ttlSeconds
            ),
            APISidecarEvictionCandidate(
                lane: .speech,
                loaded: states.speech.residentKey != nil,
                lastAccess: states.speech.lastAccess,
                activeRequests: states.speech.activeRequests,
                queuedRequests: states.speech.queuedRequests,
                pinned: speechSettings.pinned,
                ttlSeconds: speechSettings.ttlSeconds
            ),
            APISidecarEvictionCandidate(
                lane: .transcription,
                loaded: states.transcription.residentKey != nil,
                lastAccess: states.transcription.lastAccess,
                activeRequests: states.transcription.activeRequests,
                queuedRequests: states.transcription.queuedRequests,
                pinned: transcriptionSettings.pinned,
                ttlSeconds: transcriptionSettings.ttlSeconds
            ),
            APISidecarEvictionCandidate(
                lane: .embedding,
                loaded: states.embedding.residentKey != nil,
                lastAccess: states.embedding.lastAccess,
                activeRequests: states.embedding.activeRequests,
                queuedRequests: states.embedding.queuedRequests,
                pinned: embeddingSettings.pinned,
                ttlSeconds: embeddingSettings.ttlSeconds
            ),
        ]
    }

    private func residentSnapshots(
        states: SlotStates,
        settings: [String: RuntimeModelSettings]
    ) -> [RuntimeSidecarResidentSnapshot] {
        let imageKey = states.image.residentKey ?? states.image.lastKey
        let speechKey = states.speech.residentKey ?? states.speech.lastKey
        let transcriptionKey = states.transcription.residentKey ?? states.transcription.lastKey
        let embeddingKey = states.embedding.residentKey ?? states.embedding.lastKey
        return [
            residentSnapshot(
                kind: .image,
                modelID: imageKey?.modelID,
                modelPath: imageKey?.modelPath,
                variant: imageKey?.kind.rawValue,
                loaded: states.image.residentKey != nil,
                state: states.image,
                settings: settings
            ),
            residentSnapshot(
                kind: .speech,
                modelID: speechKey?.modelID,
                modelPath: speechKey?.modelPath,
                variant: "qwen3-tts",
                loaded: states.speech.residentKey != nil,
                state: states.speech,
                settings: settings
            ),
            residentSnapshot(
                kind: .transcription,
                modelID: transcriptionKey?.modelID,
                modelPath: transcriptionKey?.modelPath,
                variant: transcriptionKey?.backend.rawValue,
                loaded: states.transcription.residentKey != nil,
                state: states.transcription,
                settings: settings
            ),
            residentSnapshot(
                kind: .embedding,
                modelID: embeddingKey?.modelID,
                modelPath: embeddingKey?.modelPath,
                variant: "qwen3-embedding",
                loaded: states.embedding.residentKey != nil,
                state: states.embedding,
                settings: settings
            ),
        ]
    }

    private func residentSnapshot<Key: Equatable & Sendable>(
        kind: RuntimeSidecarKind,
        modelID: String?,
        modelPath: String?,
        variant: String?,
        loaded: Bool,
        state: APISidecarResidentSlotState<Key>,
        settings: [String: RuntimeModelSettings]
    ) -> RuntimeSidecarResidentSnapshot {
        let lifecycle = lifecycleSettings(modelID: modelID, settings: settings)
        var snapshot = RuntimeSidecarResidentSnapshot(
            kind: kind,
            modelID: modelID,
            modelPath: modelPath,
            variant: variant,
            loaded: loaded,
            activeRequests: state.activeRequests,
            queuedRequests: state.queuedRequests,
            loadedAt: state.loadedAt,
            lastAccess: state.lastAccess,
            lastEvictedAt: state.lastEvictedAt,
            lastEvictionReason: state.lastEvictionReason,
            pinned: lifecycle.pinned,
            ttlSeconds: lifecycle.ttlSeconds,
            loadCount: state.loadCount,
            replacementCount: state.replacementCount,
            evictionCount: state.evictionCount,
            completedRequests: state.completedRequests,
            failedRequests: state.failedRequests
        )
        snapshot.ready = loaded && state.ready
        return snapshot
    }

    private func lifecycleSettings(
        modelID: String?,
        settings: [String: RuntimeModelSettings]
    ) -> (pinned: Bool, ttlSeconds: Int) {
        let configured = modelID.flatMap { settings[$0] }
        return (
            pinned: configured?.pinned ?? false,
            ttlSeconds: configured?.ttlSeconds ?? defaultIdleTTLSeconds
        )
    }

    private func loadedSettings() -> [String: RuntimeModelSettings] {
        (try? RuntimeModelSettingsStore(url: settingsURL).load())?.models ?? [:]
    }

    private static func unloadImage(_ generator: APISidecarImageGenerator) async {
        switch generator {
        case .flux1(let generator):
            await generator.unload()
        case .flux2Klein(let generator):
            await generator.unload()
        case .zImageTurbo(let generator):
            await generator.unload()
        case .hiDreamO1(let generator):
            generator.unload()
        case .senseNovaU15(let generator):
            generator.unload()
        case .krea2(let generator):
            generator.unload()
        case .ideogram4(let generator):
            generator.unload()
        case .qwenImageEdit(let generator):
            await generator.clearCache()
        }
    }

    private static func unloadASR(_ generator: APISidecarASRGenerator) async {
        switch generator {
        case .qwen(let generator):
            await generator.unload()
        case .parakeet(let generator):
            await generator.unload()
        }
    }

    private func estimatedLoadBytes(
        modelID: String,
        modelPath: String?,
        minimumBytes: UInt64
    ) -> UInt64? {
        var estimate = minimumBytes
        if let catalogEstimate = ManagedModelCatalog.spec(for: modelID)?.estimatedDownloadBytes,
           catalogEstimate > 0 {
            updateMaximum(&estimate, candidate: UInt64(catalogEstimate))
        }
        if let modelPath {
            let url = URL(fileURLWithPath: modelPath, isDirectory: true).standardizedFileURL
            if FileManager.default.fileExists(atPath: url.path) {
                let bytes = FileSystemHelper.directorySize(at: url)
                if bytes > 0 {
                    updateMaximum(&estimate, candidate: UInt64(bytes))
                }
            }
        }
        return estimate > 0 ? estimate : nil
    }

    private func estimatedImageLoadBytes(
        kind: APISidecarImageKind,
        modelID: String,
        modelPath: String,
        width: Int,
        height: Int,
        residentNeedsLoad: Bool
    ) -> UInt64? {
        let baseBytes = residentNeedsLoad
            ? Self.imageBaseLoadBytes(for: kind)
            : 8 * Self.gibibyte
        let dimensions = Self.effectiveImageLoadDimensions(
            for: kind,
            width: width,
            height: height
        )
        let minimumBytes = Self.scaledImageLoadBytes(
            baseBytes: baseBytes,
            width: dimensions.width,
            height: dimensions.height
        )
        guard residentNeedsLoad else {
            // Warm staged pipelines can temporarily reload components, but
            // their checkpoint is already resident/accounted for. Avoid an
            // unnecessary directory walk and project only activation/staging
            // headroom for the requested pixel count.
            return minimumBytes
        }
        return estimatedLoadBytes(
            modelID: modelID,
            modelPath: modelPath,
            minimumBytes: minimumBytes
        )
    }

    private static let gibibyte: UInt64 = 1_073_741_824
    private static let referenceImagePixels: UInt64 = 1_024 * 1_024

    static func canonicalImageModelID(
        kind: APISidecarImageKind,
        modelID: String
    ) -> String {
        guard kind == .qwenImageEdit else { return modelID }
        return QwenImageEditRepository.canonicalModelId(for: modelID) ?? modelID
    }

    private static func imageBaseLoadBytes(for kind: APISidecarImageKind) -> UInt64 {
        switch kind {
        case .flux1:
            return 40 * gibibyte
        case .flux2Klein:
            return 10 * gibibyte
        case .zImageTurbo:
            return 12 * gibibyte
        case .hiDreamO1, .krea2, .ideogram4, .qwenImageEdit:
            return 16 * gibibyte
        case .senseNovaU15:
            return 48 * gibibyte
        }
    }

    private static func effectiveImageLoadDimensions(
        for kind: APISidecarImageKind,
        width: Int,
        height: Int
    ) -> (width: Int, height: Int) {
        switch kind {
        case .hiDreamO1:
            let resolution = HiDreamO1SampleBuilder.closestResolution(
                width: width,
                height: height
            )
            return (resolution.width, resolution.height)
        case .flux1, .flux2Klein, .zImageTurbo, .senseNovaU15, .krea2, .ideogram4, .qwenImageEdit:
            return (width, height)
        }
    }

    private static func scaledImageLoadBytes(
        baseBytes: UInt64,
        width: Int,
        height: Int
    ) -> UInt64 {
        guard width > 0, height > 0 else {
            return baseBytes
        }
        let (pixelCount, pixelOverflow) = UInt64(width).multipliedReportingOverflow(by: UInt64(height))
        guard !pixelOverflow else {
            return UInt64.max
        }
        let wholeUnits = pixelCount / referenceImagePixels
        let scale = max(1, wholeUnits + (pixelCount % referenceImagePixels == 0 ? 0 : 1))
        let (scaledBytes, byteOverflow) = baseBytes.multipliedReportingOverflow(by: scale)
        return byteOverflow ? UInt64.max : scaledBytes
    }

    private func updateMaximum(_ value: inout UInt64, candidate: UInt64) {
        value = max(value, candidate)
    }

    #if DEBUG
    func estimatedImageLoadBytesForTesting(
        kind: APISidecarImageKind,
        modelID: String,
        modelPath: String,
        width: Int,
        height: Int,
        residentNeedsLoad: Bool
    ) -> UInt64? {
        estimatedImageLoadBytes(
            kind: kind,
            modelID: modelID,
            modelPath: modelPath,
            width: width,
            height: height,
            residentNeedsLoad: residentNeedsLoad
        )
    }

    func prepareForColdSidecarLoadForTesting(
        estimatedLoadBytes: UInt64?
    ) async throws {
        try await prepareForColdSidecarLoad(
            excluding: [],
            estimatedLoadBytes: estimatedLoadBytes
        )
    }
    #endif

    private func normalizedPath(_ value: String) -> String {
        guard (value as NSString).isAbsolutePath else {
            return value
        }
        return URL(fileURLWithPath: value).standardizedFileURL.path
    }
}
