import XCTest
import AudioCore
@testable import MereRunCLI

final class APISidecarResidentSlotTests: XCTestCase {
    func testResidentSlotReusesMatchingKeyAndUnloadsBeforeReplacement() async throws {
        let probe = APISidecarSlotProbe()
        let slot = APISidecarResidentSlot<String, Int>()

        let first = try await slot.withValue(
            for: "image-a",
            make: { await probe.makeValue(1) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )
        let repeated = try await slot.withValue(
            for: "image-a",
            make: { await probe.makeValue(2) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )
        let replacement = try await slot.withValue(
            for: "image-b",
            make: { await probe.makeValue(3) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )

        XCTAssertEqual(first, 1)
        XCTAssertEqual(repeated, 1)
        XCTAssertEqual(replacement, 3)
        let madeValues = await probe.madeValues()
        let unloadedValues = await probe.unloadedValues()
        let residentKey = await slot.residentKey()
        XCTAssertEqual(madeValues, [1, 3])
        XCTAssertEqual(unloadedValues, [1])
        XCTAssertEqual(residentKey, "image-b")
    }

    func testResidentSlotSerializesConcurrentOperationsAndDeduplicatesLoad() async throws {
        let probe = APISidecarSlotProbe()
        let slot = APISidecarResidentSlot<String, Int>()

        async let first = slot.withValue(
            for: "speech",
            make: { await probe.makeValue(7) },
            unload: { value in await probe.unload(value) },
            operation: { value in try await probe.use(value) }
        )
        async let second = slot.withValue(
            for: "speech",
            make: { await probe.makeValue(7) },
            unload: { value in await probe.unload(value) },
            operation: { value in try await probe.use(value) }
        )

        let values = try await [first, second]
        let madeValues = await probe.madeValues()
        let maximumConcurrentUses = await probe.maximumConcurrentUses()
        XCTAssertEqual(values, [7, 7])
        XCTAssertEqual(madeValues, [7])
        XCTAssertEqual(maximumConcurrentUses, 1)
    }

    func testASRRoutingUsesResidentExecutorWithoutConstructingGenerator() async throws {
        let executor = APISidecarASRExecutorProbe()
        let request = ASRRequest(
            audioURL: URL(fileURLWithPath: "/tmp/resident-asr-probe.wav"),
            language: "en",
            maxTokens: 32
        )

        let execution = try await CLIASRRouting.transcribe(
            request: request,
            preferredBackend: .qwen,
            executor: executor
        )

        XCTAssertEqual(execution.backend, .qwen)
        XCTAssertEqual(execution.result.text, "resident qwen")
        let qwenRequests = await executor.qwenRequests()
        let parakeetRequestCount = await executor.parakeetRequestCount()
        XCTAssertEqual(qwenRequests, [request])
        XCTAssertEqual(parakeetRequestCount, 0)
    }

    func testIdleEvictionRecordsDiagnosticsAndUnloadsResident() async throws {
        let probe = APISidecarSlotProbe()
        let slot = APISidecarResidentSlot<String, Int>()
        _ = try await slot.withValue(
            for: "image",
            make: { await probe.makeValue(9) },
            unload: { value in await probe.unload(value) },
            operation: { $0 }
        )

        let evicted = await slot.evictIfIdle(
            expectedKey: "image",
            reason: .ttl,
            using: { value in await probe.unload(value) }
        )
        let state = await slot.state()
        let unloadedValues = await probe.unloadedValues()

        XCTAssertTrue(evicted)
        XCTAssertNil(state.residentKey)
        XCTAssertEqual(state.lastKey, "image")
        XCTAssertEqual(state.lastEvictionReason, .ttl)
        XCTAssertEqual(state.evictionCount, 1)
        XCTAssertEqual(unloadedValues, [9])
    }

    func testEvictionNeverTakesAnActiveLease() async throws {
        let probe = APISidecarSlotProbe()
        let slot = APISidecarResidentSlot<String, Int>()
        let request = Task {
            try await slot.withValue(
                for: "speech",
                make: { await probe.makeValue(4) },
                unload: { value in await probe.unload(value) },
                operation: { value in try await probe.use(value, nanoseconds: 100_000_000) }
            )
        }
        for _ in 0..<50 {
            if await probe.activeUseCount() == 1 { break }
            try await Task.sleep(nanoseconds: 1_000_000)
        }

        let evicted = await slot.evictIfIdle(
            expectedKey: "speech",
            reason: .memoryPressure,
            using: { value in await probe.unload(value) }
        )
        let result = try await request.value
        let residentKey = await slot.residentKey()
        let unloadedValues = await probe.unloadedValues()

        XCTAssertFalse(evicted)
        XCTAssertEqual(result, 4)
        XCTAssertEqual(residentKey, "speech")
        XCTAssertTrue(unloadedValues.isEmpty)
    }

    func testEvictionPlannerCombinesTTLAndMemoryPressureWithoutTouchingBusyOrPinnedLanes() {
        let now = Date(timeIntervalSince1970: 1_000)
        let candidates = [
            APISidecarEvictionCandidate(
                lane: .image,
                loaded: true,
                lastAccess: Date(timeIntervalSince1970: 600),
                activeRequests: 0,
                queuedRequests: 0,
                pinned: false,
                ttlSeconds: 300
            ),
            APISidecarEvictionCandidate(
                lane: .speech,
                loaded: true,
                lastAccess: Date(timeIntervalSince1970: 900),
                activeRequests: 0,
                queuedRequests: 0,
                pinned: false,
                ttlSeconds: 300
            ),
            APISidecarEvictionCandidate(
                lane: .transcription,
                loaded: true,
                lastAccess: Date(timeIntervalSince1970: 500),
                activeRequests: 0,
                queuedRequests: 0,
                pinned: true,
                ttlSeconds: 300
            ),
        ]

        let decisions = APISidecarEvictionPlanner.decisions(
            candidates: candidates,
            now: now,
            pressure: .elevated
        )

        XCTAssertEqual(decisions, [
            .init(lane: .image, reason: .ttl),
            .init(lane: .speech, reason: .memoryPressure),
        ])
    }

    func testEmptyPoolStatusReportsBoundedDefaultTTL() async {
        let pool = APISidecarModelPool(
            memoryPressurePolicy: RuntimeMemoryPressurePolicy(tier: .off),
            defaultIdleTTLSeconds: 42
        )

        let status = await pool.status()

        XCTAssertEqual(status.defaultIdleTTLSeconds, 42)
        XCTAssertEqual(status.loadedCount, 0)
        XCTAssertEqual(status.activeRequests, 0)
        XCTAssertEqual(status.queuedRequests, 0)
        XCTAssertEqual(status.residents.count, 3)
        XCTAssertTrue(status.residents.allSatisfy { $0.ttlSeconds == 42 && !$0.loaded })
    }
}

private actor APISidecarSlotProbe {
    private var made: [Int] = []
    private var unloaded: [Int] = []
    private var activeUses = 0
    private var maxActiveUses = 0

    func makeValue(_ value: Int) -> Int {
        made.append(value)
        return value
    }

    func unload(_ value: Int) {
        unloaded.append(value)
    }

    func use(_ value: Int, nanoseconds: UInt64 = 20_000_000) async throws -> Int {
        activeUses += 1
        maxActiveUses = max(maxActiveUses, activeUses)
        try await Task.sleep(nanoseconds: nanoseconds)
        activeUses -= 1
        return value
    }

    func madeValues() -> [Int] {
        made
    }

    func unloadedValues() -> [Int] {
        unloaded
    }

    func maximumConcurrentUses() -> Int {
        maxActiveUses
    }

    func activeUseCount() -> Int {
        activeUses
    }
}

private actor APISidecarASRExecutorProbe: CLIASRTranscriptionExecutor {
    private var qwen: [ASRRequest] = []
    private var parakeet: [ASRRequest] = []

    func transcribeQwen(
        request: ASRRequest,
        modelID: String,
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        qwen.append(request)
        return ASRResult(text: "resident qwen", language: request.language)
    }

    func transcribeParakeet(
        request: ASRRequest,
        modelID: String,
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        parakeet.append(request)
        return ASRResult(text: "resident parakeet", language: request.language)
    }

    func qwenRequests() -> [ASRRequest] {
        qwen
    }

    func parakeetRequestCount() -> Int {
        parakeet.count
    }
}
