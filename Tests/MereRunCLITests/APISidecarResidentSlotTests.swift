import Foundation
import XCTest
import AudioCore
import MereRunCore
@testable import MereRunCLI

final class APISidecarResidentSlotTests: XCTestCase {
    func testSinglePostPassRunsAfterResidentSlotReleasesExecution() async throws {
        let gib = UInt64(1_073_741_824)
        let slotProbe = APISidecarSlotProbe()
        let nominal = RuntimeMemorySample(physicalBytes: 100 * gib, residentBytes: 10 * gib)
        let critical = RuntimeMemorySample(physicalBytes: 100 * gib, residentBytes: 95 * gib)
        let pressureProbe = APISidecarPressureReliefProbe(sample: nominal, relievedSample: nominal)
        let slot = APISidecarResidentSlot<String, Int>()
        let pool = APISidecarModelPool(
            memoryPressurePolicy: RuntimeMemoryPressurePolicy(tier: .balanced),
            currentMemorySample: { pressureProbe.currentSample() },
            relieveTextModelPressure: {
                pressureProbe.relieve()
                _ = await slot.evictIfIdle(
                    expectedKey: "image",
                    reason: .memoryPressure,
                    using: { value in await slotProbe.unload(value) }
                )
            }
        )

        let result = try await pool.withSidecarPressureCoordination(excluding: []) {
            try await slot.withValue(
                for: "image",
                make: { await slotProbe.makeValue(9) },
                unload: { value in await slotProbe.unload(value) },
                operation: { value in
                    pressureProbe.setSample(critical)
                    return value
                }
            )
        }

        let state = await slot.state()
        let unloadedValues = await slotProbe.unloadedValues()
        XCTAssertEqual(result, 9)
        XCTAssertNil(state.residentKey)
        XCTAssertEqual(state.lastEvictionReason, .memoryPressure)
        XCTAssertEqual(unloadedValues, [9])
        XCTAssertEqual(pressureProbe.reliefCount(), 1)
    }

    func testASRRoutingUsesResidentExecutorWithoutConstructingGenerator() async throws {
        let executor = APISidecarASRExecutorProbe()
        let audioURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        try Data().write(to: audioURL)
        defer { try? FileManager.default.removeItem(at: audioURL) }
        let request = ASRRequest(
            audioURL: audioURL,
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
        XCTAssertEqual(status.residents.count, 4)
        XCTAssertTrue(status.residents.contains { $0.kind == .embedding })
        XCTAssertTrue(status.residents.allSatisfy { $0.ttlSeconds == 42 && !$0.loaded })
    }

    func testSidecarLoadRelievesTextPressureBeforeRecheckingSidecars() async {
        let gib = UInt64(1024 * 1024 * 1024)
        let probe = APISidecarPressureReliefProbe(
            sample: RuntimeMemorySample(physicalBytes: 100 * gib, residentBytes: 95 * gib),
            relievedSample: RuntimeMemorySample(physicalBytes: 100 * gib, residentBytes: 10 * gib)
        )
        let pool = APISidecarModelPool(
            memoryPressurePolicy: RuntimeMemoryPressurePolicy(tier: .balanced),
            currentMemorySample: { probe.currentSample() },
            relieveTextModelPressure: { probe.relieve() }
        )

        await pool.prepareForSidecarLoad()

        XCTAssertEqual(probe.reliefCount(), 1)
        XCTAssertGreaterThanOrEqual(probe.sampleCount(), 2)
    }

    func testSidecarLoadRechecksPressureAfterOperationBecomesResident() async throws {
        let gib = UInt64(1024 * 1024 * 1024)
        let nominal = RuntimeMemorySample(physicalBytes: 100 * gib, residentBytes: 10 * gib)
        let critical = RuntimeMemorySample(physicalBytes: 100 * gib, residentBytes: 95 * gib)
        let probe = APISidecarPressureReliefProbe(sample: nominal, relievedSample: nominal)
        let pool = APISidecarModelPool(
            memoryPressurePolicy: RuntimeMemoryPressurePolicy(tier: .balanced),
            currentMemorySample: { probe.currentSample() },
            relieveTextModelPressure: { probe.relieve() }
        )

        let value = try await pool.withSidecarPressureCoordination(excluding: [.image]) {
            probe.setSample(critical)
            return 17
        }

        XCTAssertEqual(value, 17)
        XCTAssertEqual(probe.reliefCount(), 1)
        XCTAssertGreaterThanOrEqual(probe.sampleCount(), 3)
    }

    func testProjectedCriticalColdLoadProactivelyReleasesIdleTextModels() async throws {
        let gib = UInt64(1024 * 1024 * 1024)
        let initial = RuntimeMemorySample(
            physicalBytes: 100 * gib,
            residentBytes: 80 * gib,
            availableBytes: 20 * gib
        )
        let relieved = RuntimeMemorySample(
            physicalBytes: 100 * gib,
            residentBytes: 10 * gib,
            availableBytes: 90 * gib
        )
        let probe = APISidecarPressureReliefProbe(sample: initial, relievedSample: relieved)
        let pool = APISidecarModelPool(
            memoryPressurePolicy: RuntimeMemoryPressurePolicy(tier: .balanced),
            currentMemorySample: { probe.currentSample() },
            releaseOneIdleTextModelForLoad: {
                probe.relieve()
                return true
            }
        )

        try await pool.prepareForColdSidecarLoadForTesting(
            estimatedLoadBytes: 15 * gib
        )

        XCTAssertEqual(probe.reliefCount(), 1)
        XCTAssertGreaterThanOrEqual(probe.sampleCount(), 2)
    }

    func testProjectedCriticalColdLoadIsRejectedWhenHeadroomCannotBeRecovered() async throws {
        let gib = UInt64(1024 * 1024 * 1024)
        let constrained = RuntimeMemorySample(
            physicalBytes: 100 * gib,
            residentBytes: 80 * gib,
            availableBytes: 20 * gib
        )
        let pool = APISidecarModelPool(
            memoryPressurePolicy: RuntimeMemoryPressurePolicy(tier: .balanced),
            currentMemorySample: { constrained }
        )

        do {
            try await pool.prepareForColdSidecarLoadForTesting(
                estimatedLoadBytes: 15 * gib
            )
            XCTFail("Expected projected critical pressure to reject the cold load")
        } catch let error as APISidecarModelPoolError {
            XCTAssertEqual(error, .memoryPressure)
        }
    }

    func testImageLoadProjectionUsesFamilyAndPixelScaledHeadroom() throws {
        let gib = UInt64(1_073_741_824)
        let pool = APISidecarModelPool()
        let missingPath = "/tmp/mere-run-missing-image-model-\(UUID().uuidString)"

        XCTAssertEqual(
            pool.estimatedImageLoadBytesForTesting(
                kind: .flux2Klein,
                modelID: "image-klein-nano",
                modelPath: missingPath,
                width: 1_024,
                height: 1_024,
                residentNeedsLoad: true
            ),
            10 * gib
        )
        XCTAssertEqual(
            pool.estimatedImageLoadBytesForTesting(
                kind: .zImageTurbo,
                modelID: "image-zimage-nano",
                modelPath: missingPath,
                width: 1_536,
                height: 1_024,
                residentNeedsLoad: true
            ),
            24 * gib
        )
        XCTAssertEqual(
            pool.estimatedImageLoadBytesForTesting(
                kind: .qwenImageEdit,
                modelID: "qwen-image-edit",
                modelPath: missingPath,
                width: 2_048,
                height: 1_024,
                residentNeedsLoad: true
            ),
            32 * gib
        )
        XCTAssertEqual(
            pool.estimatedImageLoadBytesForTesting(
                kind: .qwenImageEdit,
                modelID: "qwen-image-edit",
                modelPath: missingPath,
                width: 2_048,
                height: 1_024,
                residentNeedsLoad: false
            ),
            16 * gib
        )
        XCTAssertEqual(
            pool.estimatedImageLoadBytesForTesting(
                kind: .hiDreamO1,
                modelID: "image-hidream-o1",
                modelPath: missingPath,
                width: 1_024,
                height: 1_024,
                residentNeedsLoad: true
            ),
            64 * gib
        )
        XCTAssertEqual(
            pool.estimatedImageLoadBytesForTesting(
                kind: .hiDreamO1,
                modelID: "image-hidream-o1",
                modelPath: missingPath,
                width: 1,
                height: 1,
                residentNeedsLoad: false
            ),
            32 * gib
        )
    }

    func testQwenImageEditUsesCanonicalResidentIdentity() {
        XCTAssertEqual(
            APISidecarModelPool.canonicalImageModelID(
                kind: .qwenImageEdit,
                modelID: "Qwen/Qwen-Image-Edit"
            ),
            "qwen-image-edit"
        )
        XCTAssertEqual(
            APISidecarModelPool.canonicalImageModelID(
                kind: .qwenImageEdit,
                modelID: "Qwen/Qwen-Image-Edit-2511"
            ),
            QwenImageEditRepository.model2511Id
        )
        XCTAssertEqual(
            APISidecarModelPool.canonicalImageModelID(
                kind: .qwenImageEdit,
                modelID: QwenImageEditRepository.lightning2511Id
            ),
            QwenImageEditRepository.lightning2511Id
        )
        XCTAssertEqual(
            APISidecarModelPool.canonicalImageModelID(
                kind: .zImageTurbo,
                modelID: "image-zimage-nano"
            ),
            "image-zimage-nano"
        )
    }

    func testQwenImageEditLoadProjectionAccountsForResolvedModelRoot() throws {
        let gib = UInt64(1_073_741_824)
        let modelRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-qwen-edit-estimate-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelRoot) }

        let weights = modelRoot.appendingPathComponent("weights.safetensors")
        XCTAssertTrue(FileManager.default.createFile(atPath: weights.path, contents: Data()))
        let handle = try FileHandle(forWritingTo: weights)
        let logicalSize = 17 * gib + 123
        try handle.truncate(atOffset: logicalSize)
        try handle.close()

        let estimate = APISidecarModelPool().estimatedImageLoadBytesForTesting(
            kind: .qwenImageEdit,
            modelID: "qwen-image-edit",
            modelPath: modelRoot.path,
            width: 1_024,
            height: 1_024,
            residentNeedsLoad: true
        )

        XCTAssertEqual(estimate, logicalSize)
    }
}

private enum APISidecarSlotTestError: Error {
    case loadFailed
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

    func resetMaximumConcurrentUses() {
        maxActiveUses = activeUses
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

private actor APISidecarColdPreparationProbe {
    private var values: [Bool] = []

    func record(_ residentNeedsLoad: Bool) {
        values.append(residentNeedsLoad)
    }

    func classifications() -> [Bool] {
        values
    }
}

private actor APISidecarColdPreparationGate {
    private var started = false
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        started = true
        await withCheckedContinuation { continuation in
            self.continuation = continuation
        }
    }

    func hasStarted() -> Bool {
        started
    }

    func release() {
        continuation?.resume()
        continuation = nil
    }
}

private actor APISidecarReplacementPreparationProbe {
    private var values: [[Int]] = []

    func record(unloadedValues: [Int]) {
        values.append(unloadedValues)
    }

    func observations() -> [[Int]] {
        values
    }
}

private final class APISidecarIdlePolicyProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var pinned: Bool
    private var ttl: Duration

    init(pinned: Bool, ttl: Duration) {
        self.pinned = pinned
        self.ttl = ttl
    }

    func setPinned(_ value: Bool) {
        lock.lock()
        defer { lock.unlock() }
        pinned = value
    }

    func setTTL(_ value: Duration) {
        lock.lock()
        defer { lock.unlock() }
        ttl = value
    }

    func current() -> APISidecarResidentIdlePolicy {
        lock.lock()
        defer { lock.unlock() }
        return APISidecarResidentIdlePolicy(pinned: pinned, ttl: ttl)
    }
}

private final class APISidecarPressureReliefProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var sample: RuntimeMemorySample
    private let relievedSample: RuntimeMemorySample
    private var samples = 0
    private var reliefs = 0

    init(sample: RuntimeMemorySample, relievedSample: RuntimeMemorySample) {
        self.sample = sample
        self.relievedSample = relievedSample
    }

    func currentSample() -> RuntimeMemorySample {
        lock.lock()
        defer { lock.unlock() }
        samples += 1
        return sample
    }

    func relieve() {
        lock.lock()
        defer { lock.unlock() }
        reliefs += 1
        sample = relievedSample
    }

    func setSample(_ value: RuntimeMemorySample) {
        lock.lock()
        defer { lock.unlock() }
        sample = value
    }

    func sampleCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return samples
    }

    func reliefCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return reliefs
    }
}
