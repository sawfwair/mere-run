import Foundation
import AudioCore
import AudioSTT
import AudioTTS
import MereRunCore

/// A bounded, exclusive resident slot for mutable inference runtimes.
///
/// The slot keeps exactly one value resident. Requests for the same key reuse
/// it, while a different key unloads the previous value before constructing the
/// replacement. The execution admission remains held for the complete
/// operation because model generators contain mutable caches and are not
/// generally safe to re-enter while an inference is suspended.
struct APISidecarResidentSlotState<Key: Equatable & Sendable>: Sendable {
    let residentKey: Key?
    let ready: Bool
    let lastKey: Key?
    let loadedAt: Date?
    let lastAccess: Date?
    let lastEvictedAt: Date?
    let lastEvictionReason: RuntimeSidecarEvictionReason?
    let activeRequests: Int
    let queuedRequests: Int
    let loadCount: Int
    let replacementCount: Int
    let evictionCount: Int
    let completedRequests: Int
    let failedRequests: Int
}

struct APISidecarResidentIdlePolicy: Sendable {
    let pinned: Bool
    let ttl: Duration
}

actor APISidecarResidentSlot<Key: Equatable & Sendable, Value: Sendable> {
    private struct Resident: Sendable {
        let key: Key
        let value: Value
        let loadedAt: Date
        var lastAccess: Date
        var ready: Bool
    }

    private let execution = RuntimeRequestAdmission(maxActiveRequests: 1)
    private let currentDate: @Sendable () -> Date
    private var resident: Resident?
    private var lastKey: Key?
    private var lastLoadedAt: Date?
    private var lastAccess: Date?
    private var lastEvictedAt: Date?
    private var lastEvictionReason: RuntimeSidecarEvictionReason?
    private var activeRequests = 0
    private var queuedRequests = 0
    private var loadCount = 0
    private var replacementCount = 0
    private var evictionCount = 0
    private var completedRequests = 0
    private var failedRequests = 0
    private var idleEvictionGeneration: UInt64 = 0
    private var idleEvictionTask: Task<Void, Never>?

    init(currentDate: @escaping @Sendable () -> Date = { Date() }) {
        self.currentDate = currentDate
    }

    func withValue<Result: Sendable>(
        for key: Key,
        idleTTL: Duration? = nil,
        pinned: Bool = false,
        currentIdlePolicy: (@Sendable (Key) -> APISidecarResidentIdlePolicy)? = nil,
        idlePolicyPollInterval: Duration = .seconds(1),
        make: @Sendable () async throws -> Value,
        unload: @escaping @Sendable (Value) async -> Void,
        operation: @Sendable (Value) async throws -> Result
    ) async throws -> Result {
        precondition(idlePolicyPollInterval > .zero, "Idle policy poll interval must be positive")
        queuedRequests += 1
        let lease: RuntimeRequestAdmissionLease
        do {
            lease = try await execution.acquire()
        } catch {
            queuedRequests = max(0, queuedRequests - 1)
            throw error
        }
        queuedRequests = max(0, queuedRequests - 1)
        activeRequests += 1
        do {
            let value = try await value(for: key, make: make, unload: unload)
            let result = try await operation(value)
            completedRequests += 1
            markReady(key: key)
            let idleGeneration = touch(key: key)
            activeRequests = max(0, activeRequests - 1)
            await lease.release()
            scheduleIdleEviction(
                expectedKey: key,
                expectedGeneration: idleGeneration,
                idleTTL: idleTTL,
                pinned: pinned,
                currentIdlePolicy: currentIdlePolicy,
                pollInterval: idlePolicyPollInterval,
                unload: unload
            )
            return result
        } catch {
            failedRequests += 1
            let idleGeneration = touch(key: key)
            activeRequests = max(0, activeRequests - 1)
            await lease.release()
            scheduleIdleEviction(
                expectedKey: key,
                expectedGeneration: idleGeneration,
                idleTTL: idleTTL,
                pinned: pinned,
                currentIdlePolicy: currentIdlePolicy,
                pollInterval: idlePolicyPollInterval,
                unload: unload
            )
            throw error
        }
    }

    func evictIfIdle(
        expectedKey: Key,
        expectedGeneration: UInt64? = nil,
        reason: RuntimeSidecarEvictionReason,
        using unload: @Sendable (Value) async -> Void
    ) async -> Bool {
        guard activeRequests == 0,
              queuedRequests == 0,
              expectedGeneration.map({ $0 == idleEvictionGeneration }) ?? true,
              let lease = await execution.tryAcquire() else {
            return false
        }
        guard activeRequests == 0,
              queuedRequests == 0,
              expectedGeneration.map({ $0 == idleEvictionGeneration }) ?? true,
              let resident,
              resident.key == expectedKey else {
            await lease.release()
            return false
        }

        self.resident = nil
        invalidateIdleEviction()
        lastKey = resident.key
        lastLoadedAt = resident.loadedAt
        lastAccess = resident.lastAccess
        lastEvictedAt = currentDate()
        lastEvictionReason = reason
        evictionCount += 1
        await unload(resident.value)
        await lease.release()
        return true
    }

    func residentKey() -> Key? {
        resident?.key
    }

    func state() -> APISidecarResidentSlotState<Key> {
        APISidecarResidentSlotState(
            residentKey: resident?.key,
            ready: resident?.ready ?? false,
            lastKey: lastKey,
            loadedAt: resident?.loadedAt ?? lastLoadedAt,
            lastAccess: resident?.lastAccess ?? lastAccess,
            lastEvictedAt: lastEvictedAt,
            lastEvictionReason: lastEvictionReason,
            activeRequests: activeRequests,
            queuedRequests: queuedRequests,
            loadCount: loadCount,
            replacementCount: replacementCount,
            evictionCount: evictionCount,
            completedRequests: completedRequests,
            failedRequests: failedRequests
        )
    }

    private func value(
        for key: Key,
        make: @Sendable () async throws -> Value,
        unload: @Sendable (Value) async -> Void
    ) async throws -> Value {
        if let resident, resident.key == key {
            return resident.value
        }
        if let resident {
            self.resident = nil
            invalidateIdleEviction()
            lastKey = resident.key
            lastLoadedAt = resident.loadedAt
            lastAccess = resident.lastAccess
            replacementCount += 1
            await unload(resident.value)
        }
        let value = try await make()
        let now = currentDate()
        resident = Resident(
            key: key,
            value: value,
            loadedAt: now,
            lastAccess: now,
            ready: false
        )
        invalidateIdleEviction()
        lastKey = key
        lastLoadedAt = now
        lastAccess = now
        loadCount += 1
        return value
    }

    private func touch(key: Key) -> UInt64? {
        guard var resident, resident.key == key else { return nil }
        let now = currentDate()
        invalidateIdleEviction()
        lastKey = key
        lastAccess = now
        resident.lastAccess = now
        self.resident = resident
        return idleEvictionGeneration
    }

    private func markReady(key: Key) {
        guard var resident, resident.key == key else { return }
        resident.ready = true
        self.resident = resident
    }

    private func scheduleIdleEviction(
        expectedKey: Key,
        expectedGeneration: UInt64?,
        idleTTL: Duration?,
        pinned: Bool,
        currentIdlePolicy: (@Sendable (Key) -> APISidecarResidentIdlePolicy)?,
        pollInterval: Duration,
        unload: @escaping @Sendable (Value) async -> Void
    ) {
        guard let idleTTL,
              (!pinned || currentIdlePolicy != nil),
              let expectedGeneration,
              expectedGeneration == idleEvictionGeneration,
              resident?.key == expectedKey else {
            return
        }
        idleEvictionTask = Task { [weak self] in
            let clock = ContinuousClock()
            let idleStart = clock.now
            var effectiveTTL = idleTTL
            while !Task.isCancelled {
                if let currentIdlePolicy {
                    let policy = currentIdlePolicy(expectedKey)
                    effectiveTTL = policy.ttl
                    if policy.pinned {
                        do {
                            try await Task.sleep(for: pollInterval)
                        } catch {
                            return
                        }
                        continue
                    }
                }

                let elapsed = idleStart.duration(to: clock.now)
                if elapsed >= effectiveTTL {
                    break
                }
                let remaining = effectiveTTL - elapsed
                let delay = remaining < pollInterval ? remaining : pollInterval
                do {
                    try await Task.sleep(for: delay)
                } catch {
                    return
                }
            }
            guard !Task.isCancelled, let self else { return }
            _ = await self.evictIfIdle(
                expectedKey: expectedKey,
                expectedGeneration: expectedGeneration,
                reason: .ttl,
                using: unload
            )
        }
    }

    private func invalidateIdleEviction() {
        idleEvictionGeneration &+= 1
        idleEvictionTask?.cancel()
        idleEvictionTask = nil
    }
}

enum APISidecarImageKind: String, Hashable, Sendable {
    case flux2Klein
    case zImageTurbo
    case hiDreamO1
    case krea2
    case ideogram4
    case qwenImageEdit
}

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

enum APISidecarLane: Int, CaseIterable, Hashable, Sendable {
    case image
    case speech
    case transcription
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
        let eligible = candidates.filter {
            $0.loaded
                && !excludedLanes.contains($0.lane)
                && !$0.pinned
                && $0.activeRequests == 0
                && $0.queuedRequests == 0
        }
        let ttl = eligible.filter { candidate in
            guard let lastAccess = candidate.lastAccess else { return false }
            return now.timeIntervalSince(lastAccess) >= Double(candidate.ttlSeconds)
        }
        .sorted(by: oldestFirst)
        var decisions = ttl.map {
            APISidecarEvictionDecision(lane: $0.lane, reason: .ttl)
        }
        let ttlLanes = Set(ttl.map(\.lane))
        let pressureEligible = eligible.filter { !ttlLanes.contains($0.lane) }.sorted(by: oldestFirst)
        switch pressure {
        case .elevated:
            if let oldest = pressureEligible.first {
                decisions.append(.init(lane: oldest.lane, reason: .memoryPressure))
            }
        case .critical:
            decisions.append(contentsOf: pressureEligible.map {
                .init(lane: $0.lane, reason: .memoryPressure)
            })
        case .disabled, .unknown, .nominal:
            break
        }
        return decisions
    }

    private static func oldestFirst(
        _ lhs: APISidecarEvictionCandidate,
        _ rhs: APISidecarEvictionCandidate
    ) -> Bool {
        let lhsDate = lhs.lastAccess ?? .distantPast
        let rhsDate = rhs.lastAccess ?? .distantPast
        if lhsDate == rhsDate {
            return lhs.lane.rawValue < rhs.lane.rawValue
        }
        return lhsDate < rhsDate
    }
}

/// Non-actor image generators are safe here because the resident slot holds an
/// exclusive lease for their entire lifetime of use.
private enum APISidecarImageGenerator: @unchecked Sendable {
    case flux2Klein(Flux2KleinGenerator)
    case zImageTurbo(ZImageTurboGenerator)
    case hiDreamO1(HiDreamO1Generator)
    case krea2(Krea2Generator)
    case ideogram4(Ideogram4Generator)
    case qwenImageEdit(QwenImageEditGenerator)
}

private enum APISidecarASRGenerator: Sendable {
    case qwen(Qwen3ASRGenerator)
    case parakeet(ParakeetGenerator)
}

/// Resident runtimes for the non-chat OpenAI-compatible endpoints.
///
/// One image runtime, one TTS runtime, and one ASR runtime may be resident at a
/// time. This bounds memory even when callers provide different local model
/// paths, while repeated requests avoid model reloads.
struct APISidecarModelPool: Sendable, CLIASRTranscriptionExecutor {
    static let defaultIdleTTLSeconds = 300

    private let imageSlot: APISidecarResidentSlot<APISidecarImageKey, APISidecarImageGenerator>
    private let speechSlot: APISidecarResidentSlot<APISidecarSpeechKey, Qwen3TTSGenerator>
    private let asrSlot: APISidecarResidentSlot<APISidecarASRKey, APISidecarASRGenerator>
    private let settingsURL: URL
    private let memoryPressurePolicy: RuntimeMemoryPressurePolicy
    private let defaultIdleTTLSeconds: Int
    private let currentDate: @Sendable () -> Date
    private let currentMemorySample: @Sendable () -> RuntimeMemorySample
    private let relieveTextModelPressure: @Sendable () async -> Void

    init(
        settingsURL: URL = RuntimeModelSettingsStore.defaultURL(),
        memoryPressurePolicy: RuntimeMemoryPressurePolicy = .default,
        defaultIdleTTLSeconds: Int = APISidecarModelPool.defaultIdleTTLSeconds,
        currentDate: @escaping @Sendable () -> Date = { Date() },
        currentMemorySample: @escaping @Sendable () -> RuntimeMemorySample = { RuntimeMemorySample.current() },
        relieveTextModelPressure: @escaping @Sendable () async -> Void = {}
    ) {
        precondition(defaultIdleTTLSeconds > 0, "Sidecar idle TTL must be positive")
        self.settingsURL = settingsURL
        self.memoryPressurePolicy = memoryPressurePolicy
        self.defaultIdleTTLSeconds = defaultIdleTTLSeconds
        self.currentDate = currentDate
        self.currentMemorySample = currentMemorySample
        self.relieveTextModelPressure = relieveTextModelPressure
        self.imageSlot = APISidecarResidentSlot(currentDate: currentDate)
        self.speechSlot = APISidecarResidentSlot(currentDate: currentDate)
        self.asrSlot = APISidecarResidentSlot(currentDate: currentDate)
    }

    func generateImage(
        kind: APISidecarImageKind,
        modelID: String,
        modelPath: String,
        request: GenerationRequest
    ) async throws -> GenerationResult {
        let key = APISidecarImageKey(
            kind: kind,
            modelID: modelID,
            modelPath: normalizedPath(modelPath)
        )
        return try await withSidecarPressureCoordination(excluding: [.image]) {
            let lifecycle = lifecycleSettings(modelID: modelID, settings: loadedSettings())
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
                make: {
                    switch kind {
                    case .flux2Klein:
                        return .flux2Klein(Flux2KleinGenerator())
                    case .zImageTurbo:
                        return .zImageTurbo(ZImageTurboGenerator())
                    case .hiDreamO1:
                        return .hiDreamO1(HiDreamO1Generator())
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
                    case .flux2Klein(let generator):
                        await generator.unload()
                    case .zImageTurbo(let generator):
                        await generator.unload()
                    case .hiDreamO1(let generator):
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
                    case .flux2Klein(let generator):
                        return try await generator.generate(request, progressHandler: nil)
                    case .zImageTurbo(let generator):
                        return try await generator.generate(request, progressHandler: nil)
                    case .hiDreamO1(let generator):
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
    }

    private func states() async -> SlotStates {
        async let image = imageSlot.state()
        async let speech = speechSlot.state()
        async let transcription = asrSlot.state()
        return await SlotStates(
            image: image,
            speech: speech,
            transcription: transcription
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

    func withSidecarPressureCoordination<Result: Sendable>(
        excluding excludedLanes: Set<APISidecarLane>,
        operation: @Sendable () async throws -> Result
    ) async throws -> Result {
        await prepareForSidecarLoad(excluding: excludedLanes)
        do {
            let result = try await operation()
            await rebalanceAfterSidecarLoad(excluding: excludedLanes)
            return result
        } catch {
            await rebalanceAfterSidecarLoad(excluding: excludedLanes)
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
            }
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
        ]
    }

    private func residentSnapshots(
        states: SlotStates,
        settings: [String: RuntimeModelSettings]
    ) -> [RuntimeSidecarResidentSnapshot] {
        let imageKey = states.image.residentKey ?? states.image.lastKey
        let speechKey = states.speech.residentKey ?? states.speech.lastKey
        let transcriptionKey = states.transcription.residentKey ?? states.transcription.lastKey
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
        case .flux2Klein(let generator):
            await generator.unload()
        case .zImageTurbo(let generator):
            await generator.unload()
        case .hiDreamO1(let generator):
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

    private func normalizedPath(_ value: String) -> String {
        guard (value as NSString).isAbsolutePath else {
            return value
        }
        return URL(fileURLWithPath: value).standardizedFileURL.path
    }
}
