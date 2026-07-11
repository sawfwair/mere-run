import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class RuntimeModelPoolTests: XCTestCase {
    private let gib = UInt64(1024 * 1024 * 1024)

    func testStatusIncludesLegacyStartupDefaultWithoutCatalogScan() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root)
        )

        let status = await pool.status()

        XCTAssertEqual(status.defaultModel, "custom.gguf")
        XCTAssertEqual(status.settingsPath, root.appendingPathComponent(".mere-run/runtime-model-settings.json").path)
        XCTAssertEqual(status.models.first?.id, "custom.gguf")
        XCTAssertEqual(status.models.first?.engine, .textCode)
        XCTAssertFalse(status.models.first?.loaded ?? true)
        XCTAssertTrue(status.capabilities.chunkedPrefill.enabled)
        XCTAssertTrue(status.capabilities.continuousBatching.available)
        XCTAssertFalse(status.capabilities.continuousBatching.enabled)
        XCTAssertTrue(status.capabilities.prefixKVReuse.enabled)
    }

    func testRuntimeStatusDecodesOlderPayloadWithoutSidecarsField() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root)
        )
        let status = await pool.status()
        let encoded = try JSONEncoder().encode(status)
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        )
        object.removeValue(forKey: "sidecars")

        let legacyPayload = try JSONSerialization.data(withJSONObject: object)
        let decoded = try JSONDecoder().decode(RuntimeModelPoolStatus.self, from: legacyPayload)

        XCTAssertNil(decoded.sidecars)
        XCTAssertEqual(decoded.defaultModel, status.defaultModel)
    }

    func testRuntimeModelEntryDecodesOlderPayloadWithoutReadyField() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root)
        )
        let status = await pool.status()
        var entry = try XCTUnwrap(status.models.first)
        let encoded = try JSONEncoder().encode(entry)
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object.removeValue(forKey: "ready")

        entry = try JSONDecoder().decode(
            RuntimeModelPoolEntrySnapshot.self,
            from: JSONSerialization.data(withJSONObject: object)
        )

        XCTAssertNil(entry.ready)
        XCTAssertFalse(entry.loaded)
    }

    func testSidecarEntryDecodesOlderPayloadWithoutReadyField() throws {
        var resident = RuntimeSidecarResidentSnapshot(
            kind: .image,
            modelID: "image-zimage-nano",
            modelPath: "/tmp/image-zimage-nano",
            variant: "zImageTurbo",
            loaded: true,
            activeRequests: 0,
            queuedRequests: 0,
            loadedAt: nil,
            lastAccess: nil,
            lastEvictedAt: nil,
            lastEvictionReason: nil,
            pinned: false,
            ttlSeconds: 300,
            loadCount: 1,
            replacementCount: 0,
            evictionCount: 0,
            completedRequests: 1,
            failedRequests: 0
        )
        resident.ready = true
        let encoded = try JSONEncoder().encode(resident)
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object.removeValue(forKey: "ready")

        resident = try JSONDecoder().decode(
            RuntimeSidecarResidentSnapshot.self,
            from: JSONSerialization.data(withJSONObject: object)
        )

        XCTAssertNil(resident.ready)
        XCTAssertTrue(resident.loaded)
    }

    func testRuntimeStatusIncludesSidecarActivityInTopLevelMemorySummary() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root)
        )
        let resident = RuntimeSidecarResidentSnapshot(
            kind: .speech,
            modelID: "speech-tts-qwen3-nano",
            modelPath: nil,
            variant: "qwen3-tts",
            loaded: true,
            activeRequests: 1,
            queuedRequests: 2,
            loadedAt: Date(timeIntervalSince1970: 10),
            lastAccess: Date(timeIntervalSince1970: 20),
            lastEvictedAt: nil,
            lastEvictionReason: nil,
            pinned: false,
            ttlSeconds: 300,
            loadCount: 1,
            replacementCount: 0,
            evictionCount: 0,
            completedRequests: 2,
            failedRequests: 0
        )
        let sidecars = RuntimeSidecarPoolStatus(
            defaultIdleTTLSeconds: 300,
            pressure: "nominal",
            loadedCount: 1,
            activeRequests: 1,
            queuedRequests: 2,
            residents: [resident]
        )

        let status = await pool.status(admission: nil, sidecars: sidecars)

        XCTAssertEqual(status.activeRequests, 1)
        XCTAssertEqual(status.memory.activeRequests, 1)
        XCTAssertEqual(status.memory.activeModelCount, 1)
        XCTAssertEqual(status.sidecars, sidecars)
    }

    func testModelsResponseIncludesStartupDefault() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root)
        )

        let response = try await pool.modelsResponse(createdAt: Date(timeIntervalSince1970: 10))

        XCTAssertTrue(response.data.contains { $0.id == "custom.gguf" })
        XCTAssertEqual(response.data.first { $0.id == "custom.gguf" }?.created, 10)
    }

    func testStatusReportsGemma4PrefixKVCacheCapabilityWhenEnabled() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            gemma4PrefixKVCacheEnabled: true
        )

        let status = await pool.status()

        XCTAssertTrue(status.capabilities.prefixKVReuse.available)
        XCTAssertTrue(status.capabilities.prefixKVReuse.enabled)
    }

    func testStatusReportsPrefixKVCacheCapabilityWhenExplicitlyDisabled() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            gemma4PrefixKVCacheEnabled: false,
            q35PrefixKVCacheEnabled: false,
            lfm2PrefixKVCacheEnabled: false
        )

        let status = await pool.status()

        XCTAssertTrue(status.capabilities.prefixKVReuse.available)
        XCTAssertFalse(status.capabilities.prefixKVReuse.enabled)
        XCTAssertTrue(status.capabilities.prefixKVReuse.detail.contains("disabled"))
    }

    func testStatusReportsQ35PrefixKVCacheCapabilityWhenEnabled() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            q35PrefixKVCacheEnabled: true
        )

        let status = await pool.status()

        XCTAssertTrue(status.capabilities.prefixKVReuse.available)
        XCTAssertTrue(status.capabilities.prefixKVReuse.enabled)
    }

    func testStatusReportsQ35ContinuousBatchingCapabilityWhenEnabled() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            q35ContinuousBatchingEnabled: true
        )

        let status = await pool.status()

        XCTAssertTrue(status.capabilities.continuousBatching.available)
        XCTAssertTrue(status.capabilities.continuousBatching.enabled)
    }

    func testServeConcurrencyEnablesEveryTypedDecodeBatcher() {
        let serialized = RuntimeContinuousBatchingConfiguration(
            maxActiveRequests: 1,
            environment: [:]
        )
        let concurrent = RuntimeContinuousBatchingConfiguration(
            maxActiveRequests: 2,
            environment: [:]
        )

        XCTAssertFalse(serialized.gemma4)
        XCTAssertFalse(serialized.q35)
        XCTAssertFalse(serialized.lfm2)
        XCTAssertTrue(concurrent.gemma4)
        XCTAssertTrue(concurrent.q35)
        XCTAssertTrue(concurrent.lfm2)
    }

    func testLFM2BatchingEnvironmentOverrideWinsOverServeConcurrency() {
        let forcedOff = RuntimeContinuousBatchingConfiguration(
            maxActiveRequests: 4,
            environment: ["MERERUN_LFM2_CONTINUOUS_BATCHING": "off"]
        )
        let forcedOn = RuntimeContinuousBatchingConfiguration(
            maxActiveRequests: 1,
            environment: ["MERERUN_LFM2_CONTINUOUS_BATCHING": "1"]
        )

        XCTAssertFalse(forcedOff.lfm2)
        XCTAssertTrue(forcedOff.gemma4)
        XCTAssertTrue(forcedOff.q35)
        XCTAssertTrue(forcedOn.lfm2)
    }

    func testConcurrentColdLoadsSharePreparationAndWaitUntilReady() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let preparation = RuntimeModelPreparationProbe()
        let completions = RuntimeCompletionProbe()
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            ensureMLXAvailable: {},
            prepareLoadedModel: { _ in
                await preparation.prepare()
            }
        )

        let first = Task {
            let snapshot = try await pool.loadModel(idOrAlias: "custom.gguf")
            await completions.mark()
            return snapshot
        }
        for _ in 0..<50 {
            if await preparation.callCount() == 1 { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        let firstPreparationCount = await preparation.callCount()
        XCTAssertEqual(firstPreparationCount, 1)

        let second = Task {
            let snapshot = try await pool.loadModel(idOrAlias: "custom.gguf")
            await completions.mark()
            return snapshot
        }
        try await Task.sleep(nanoseconds: 50_000_000)

        let concurrentPreparationCount = await preparation.callCount()
        let completionsBeforeRelease = await completions.count()
        XCTAssertEqual(concurrentPreparationCount, 1)
        XCTAssertEqual(completionsBeforeRelease, 0)

        await preparation.releaseAll()
        let firstSnapshot = try await first.value
        let secondSnapshot = try await second.value

        XCTAssertEqual(firstSnapshot.ready, true)
        XCTAssertEqual(secondSnapshot.ready, true)
        let completionsAfterRelease = await completions.count()
        XCTAssertEqual(completionsAfterRelease, 2)
    }

    func testUnloadDuringColdLoadInvalidatesStalePreparation() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let preparation = RuntimeModelPreparationProbe()
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            ensureMLXAvailable: {},
            prepareLoadedModel: { _ in
                await preparation.prepare()
            }
        )

        let loading = Task {
            try await pool.loadModel(idOrAlias: "custom.gguf")
        }
        for _ in 0..<50 {
            if await preparation.callCount() == 1 { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }

        let unloading = Task {
            try await pool.unloadModel(idOrAlias: "custom.gguf")
        }
        try await Task.sleep(nanoseconds: 20_000_000)

        await preparation.releaseAll()
        let unloaded = try await unloading.value
        XCTAssertFalse(unloaded.loaded)
        XCTAssertEqual(unloaded.ready, false)
        do {
            _ = try await loading.value
            XCTFail("Expected the invalidated load to throw CancellationError")
        } catch is CancellationError {
            // Expected.
        }

        let status = await pool.status()
        let model = try XCTUnwrap(status.models.first { $0.id == "custom.gguf" })
        XCTAssertFalse(model.loaded)
        XCTAssertEqual(model.ready, false)
    }

    func testUnloadAndReloadNeverReturnsNewGenerationToOldPreparation() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let preparation = RuntimeOrderedModelPreparationProbe()
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            ensureMLXAvailable: {},
            prepareLoadedModel: { _ in
                await preparation.prepare()
            }
        )

        let firstLoad = Task {
            try await pool.loadModel(idOrAlias: "custom.gguf")
        }
        for _ in 0..<50 {
            if await preparation.callCount() == 1 { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        let firstPreparationCount = await preparation.callCount()
        XCTAssertEqual(firstPreparationCount, 1)

        let unloading = Task {
            try await pool.unloadModel(idOrAlias: "custom.gguf")
        }
        for _ in 0..<50 {
            let status = await pool.status()
            if status.models.first(where: { $0.id == "custom.gguf" })?.loaded == false {
                break
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }

        let secondLoad = Task {
            try await pool.loadModel(idOrAlias: "custom.gguf")
        }
        for _ in 0..<50 {
            if await preparation.callCount() == 2 { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        let secondPreparationCount = await preparation.callCount()
        XCTAssertEqual(secondPreparationCount, 2)

        await preparation.release(call: 2)
        let secondSnapshot = try await secondLoad.value
        XCTAssertTrue(secondSnapshot.loaded)
        XCTAssertEqual(secondSnapshot.ready, true)

        await preparation.release(call: 1)
        _ = try await unloading.value
        do {
            _ = try await firstLoad.value
            XCTFail("Expected the stale generation to throw CancellationError")
        } catch is CancellationError {
            // Expected.
        }

        let status = await pool.status()
        let model = try XCTUnwrap(status.models.first { $0.id == "custom.gguf" })
        XCTAssertTrue(model.loaded)
        XCTAssertEqual(model.ready, true)
    }

    func testCancelledSoleColdLoadDoesNotLeaveOrphanResident() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let preparation = RuntimeModelPreparationProbe()
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            ensureMLXAvailable: {},
            prepareLoadedModel: { _ in
                await preparation.prepare()
            }
        )

        let cancelledLoad = Task {
            try await pool.loadModel(idOrAlias: "custom.gguf")
        }
        for _ in 0..<50 {
            if await preparation.callCount() == 1 { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        let cancelledPreparationCount = await preparation.callCount()
        XCTAssertEqual(cancelledPreparationCount, 1)

        cancelledLoad.cancel()
        var cancelledStatus = await pool.status()
        for _ in 0..<50 {
            guard cancelledStatus.models.first(where: { $0.id == "custom.gguf" })?.loaded != false else {
                break
            }
            try await Task.sleep(nanoseconds: 10_000_000)
            cancelledStatus = await pool.status()
        }
        let cancelledModel = try XCTUnwrap(
            cancelledStatus.models.first { $0.id == "custom.gguf" }
        )
        XCTAssertFalse(cancelledModel.loaded)
        XCTAssertEqual(cancelledModel.ready, false)

        let replacementLoad = Task {
            try await pool.loadModel(idOrAlias: "custom.gguf")
        }
        for _ in 0..<50 {
            if await preparation.callCount() == 2 { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        let replacementPreparationCount = await preparation.callCount()
        XCTAssertEqual(replacementPreparationCount, 2)

        await preparation.releaseAll()
        do {
            _ = try await cancelledLoad.value
            XCTFail("Expected the cancelled generation to throw CancellationError")
        } catch is CancellationError {
            // Expected.
        }
        let replacementSnapshot = try await replacementLoad.value
        XCTAssertTrue(replacementSnapshot.loaded)
        XCTAssertEqual(replacementSnapshot.ready, true)
    }

    func testPreparationFailureAwaitsMatchingGenerationUnload() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let unload = RuntimeModelUnloadProbe()
        let completions = RuntimeCompletionProbe()
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            ensureMLXAvailable: {},
            prepareLoadedModel: { _ in
                throw RuntimeModelPreparationTestError.failed
            },
            unloadLoadedModel: { _ in
                await unload.unload()
            }
        )

        let loading = Task {
            let failedAsExpected: Bool
            do {
                _ = try await pool.loadModel(idOrAlias: "custom.gguf")
                failedAsExpected = false
            } catch is RuntimeModelPreparationTestError {
                failedAsExpected = true
            } catch {
                failedAsExpected = false
            }
            await completions.mark()
            return failedAsExpected
        }
        for _ in 0..<50 {
            if await unload.callCount() == 1 { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }

        let unloadCount = await unload.callCount()
        let completionsBeforeUnload = await completions.count()
        XCTAssertEqual(unloadCount, 1)
        XCTAssertEqual(completionsBeforeUnload, 0)
        let cleaningStatus = await pool.status()
        let cleaningModel = try XCTUnwrap(
            cleaningStatus.models.first { $0.id == "custom.gguf" }
        )
        XCTAssertFalse(cleaningModel.loaded)

        await unload.releaseAll()
        let failedAsExpected = await loading.value
        let completionsAfterUnload = await completions.count()
        XCTAssertTrue(failedAsExpected)
        XCTAssertEqual(completionsAfterUnload, 1)
    }

    func testStatusReportsReachableLFM2ContinuousBatchingRuntime() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            lfm2ContinuousBatchingEnabled: true
        )
        await pool.seedLoadedLFM2ForTesting(
            id: LFM2Resources.defaultModelId,
            continuousBatchingEnabled: true
        )

        let status = await pool.status()

        XCTAssertTrue(status.capabilities.continuousBatching.enabled)
        let lfm2 = try XCTUnwrap(status.models.first { $0.id == LFM2Resources.defaultModelId })
        XCTAssertTrue(lfm2.loaded)
        XCTAssertEqual(lfm2.prefixKVCache?.enabled, true)
        XCTAssertEqual(lfm2.continuousBatching?.enabled, true)
        XCTAssertEqual(status.cacheStats.prefixKVReuse.enabledModelCount, 1)
        XCTAssertEqual(status.cacheStats.decodeBatching.enabledModelCount, 1)
    }

    func testStatusIncludesRuntimeKVCacheMode() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RuntimeModelSettingsStore(modelsDir: root)
        try store.writeSettings(
            RuntimeModelSettings(kvCacheMode: .auto),
            for: Gemma4Resources.defaultModelId
        )
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: store
        )

        let status = await pool.status()

        let gemma = try XCTUnwrap(status.models.first { $0.id == Gemma4Resources.defaultModelId })
        XCTAssertEqual(gemma.kvCacheMode, .auto)
    }

    func testTTLEvictsExpiredIdleUnpinnedModel() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RuntimeModelSettingsStore(modelsDir: root)
        try store.writeSettings(
            RuntimeModelSettings(ttlSeconds: 60),
            for: Gemma4Resources.defaultModelId
        )
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: store
        )
        await pool.seedLoadedModelForTesting(
            id: Gemma4Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 0)
        )

        let evicted = await pool.evictExpiredIdleModels(now: Date(timeIntervalSince1970: 61))

        XCTAssertEqual(evicted, [Gemma4Resources.defaultModelId])
        let status = await pool.status()
        let gemma = try XCTUnwrap(status.models.first { $0.id == Gemma4Resources.defaultModelId })
        XCTAssertFalse(gemma.loaded)
        XCTAssertEqual(gemma.ttlSeconds, 60)
    }

    func testTTLEvictionSkipsPinnedModel() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RuntimeModelSettingsStore(modelsDir: root)
        try store.writeSettings(
            RuntimeModelSettings(pinned: true, ttlSeconds: 60),
            for: Gemma4Resources.defaultModelId
        )
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: store
        )
        await pool.seedLoadedModelForTesting(
            id: Gemma4Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 0)
        )

        let evicted = await pool.evictExpiredIdleModels(now: Date(timeIntervalSince1970: 61))

        XCTAssertTrue(evicted.isEmpty)
        let status = await pool.status()
        let gemma = try XCTUnwrap(status.models.first { $0.id == Gemma4Resources.defaultModelId })
        XCTAssertTrue(gemma.loaded)
        XCTAssertTrue(gemma.pinned)
    }

    func testTTLEvictionSkipsActiveModel() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RuntimeModelSettingsStore(modelsDir: root)
        try store.writeSettings(
            RuntimeModelSettings(ttlSeconds: 60),
            for: Gemma4Resources.defaultModelId
        )
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: store
        )
        await pool.seedLoadedModelForTesting(
            id: Gemma4Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 0),
            activeRequests: 1
        )

        let evicted = await pool.evictExpiredIdleModels(now: Date(timeIntervalSince1970: 61))

        XCTAssertTrue(evicted.isEmpty)
        let status = await pool.status()
        let gemma = try XCTUnwrap(status.models.first { $0.id == Gemma4Resources.defaultModelId })
        XCTAssertTrue(gemma.loaded)
        XCTAssertEqual(gemma.activeRequests, 1)
    }

    func testMemoryPressureLRUEvictsOldestIdleUnpinnedModel() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root)
        )
        await pool.seedLoadedModelForTesting(
            id: Gemma4Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 10)
        )
        await pool.seedLoadedModelForTesting(
            id: Q35Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 20)
        )

        let evicted = await pool.evictIdleModelsForMemoryPressure(
            sample: RuntimeMemorySample(physicalBytes: 100 * gib, residentBytes: 85 * gib)
        )

        XCTAssertEqual(evicted, [Gemma4Resources.defaultModelId])
        let status = await pool.status()
        let gemma = try XCTUnwrap(status.models.first { $0.id == Gemma4Resources.defaultModelId })
        let q35 = try XCTUnwrap(status.models.first { $0.id == Q35Resources.defaultModelId })
        XCTAssertFalse(gemma.loaded)
        XCTAssertTrue(q35.loaded)
    }

    func testPressureReliefRechecksAfterEachIdleTextEviction() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let samples = RuntimeMemorySampleSequenceProbe([
            RuntimeMemorySample(physicalBytes: 100 * gib, residentBytes: 95 * gib),
            RuntimeMemorySample(physicalBytes: 100 * gib, residentBytes: 10 * gib),
        ])
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            currentMemorySample: { samples.next() }
        )
        await pool.seedLoadedModelForTesting(
            id: Gemma4Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 10)
        )
        await pool.seedLoadedModelForTesting(
            id: Q35Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 20)
        )

        let evicted = await pool.relieveMemoryPressure()

        XCTAssertEqual(evicted, [Gemma4Resources.defaultModelId])
        XCTAssertEqual(samples.readCount(), 2)
        let status = await pool.status()
        let gemma = try XCTUnwrap(status.models.first { $0.id == Gemma4Resources.defaultModelId })
        let q35 = try XCTUnwrap(status.models.first { $0.id == Q35Resources.defaultModelId })
        XCTAssertFalse(gemma.loaded)
        XCTAssertTrue(q35.loaded)
        XCTAssertEqual(q35.ready, true)
    }

    func testPressureReliefPreservesStartupDefaultByDefault() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let gibibyte = gib
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            currentMemorySample: {
                RuntimeMemorySample(physicalBytes: 100 * gibibyte, residentBytes: 95 * gibibyte)
            }
        )
        await pool.seedLoadedModelForTesting(
            id: "custom.gguf",
            lastAccess: Date(timeIntervalSince1970: 0)
        )
        await pool.seedLoadedModelForTesting(
            id: Q35Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 10)
        )

        let evicted = await pool.relieveMemoryPressure()

        XCTAssertEqual(evicted, [Q35Resources.defaultModelId])
        let status = await pool.status()
        let startup = try XCTUnwrap(status.models.first { $0.id == "custom.gguf" })
        XCTAssertTrue(startup.loaded)
        XCTAssertEqual(startup.ready, true)
    }

    func testSidecarPressureReliefCanEvictIdleStartupDefault() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let samples = RuntimeMemorySampleSequenceProbe([
            RuntimeMemorySample(physicalBytes: 100 * gib, residentBytes: 95 * gib),
            RuntimeMemorySample(physicalBytes: 100 * gib, residentBytes: 10 * gib),
        ])
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            currentMemorySample: { samples.next() }
        )
        await pool.seedLoadedModelForTesting(
            id: "custom.gguf",
            lastAccess: Date(timeIntervalSince1970: 0)
        )
        await pool.seedLoadedModelForTesting(
            id: Q35Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 10)
        )

        let evicted = await pool.relieveMemoryPressure(preserveDefault: false)

        XCTAssertEqual(evicted, ["custom.gguf"])
        XCTAssertEqual(samples.readCount(), 2)
    }

    func testMemoryPressureLRUHonorsExcludedModel() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root)
        )
        await pool.seedLoadedModelForTesting(
            id: Gemma4Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 10)
        )
        await pool.seedLoadedModelForTesting(
            id: Q35Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 20)
        )

        let evicted = await pool.evictIdleModelsForMemoryPressure(
            sample: RuntimeMemorySample(physicalBytes: 100 * gib, residentBytes: 85 * gib),
            excluding: [Gemma4Resources.defaultModelId]
        )

        XCTAssertEqual(evicted, [Q35Resources.defaultModelId])
        let status = await pool.status()
        let gemma = try XCTUnwrap(status.models.first { $0.id == Gemma4Resources.defaultModelId })
        let q35 = try XCTUnwrap(status.models.first { $0.id == Q35Resources.defaultModelId })
        XCTAssertTrue(gemma.loaded)
        XCTAssertFalse(q35.loaded)
    }

    func testCriticalMemoryPressureLRUSkipsPinnedAndActiveModels() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RuntimeModelSettingsStore(modelsDir: root)
        try store.writeSettings(
            RuntimeModelSettings(pinned: true),
            for: Gemma4Resources.defaultModelId
        )
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: store
        )
        await pool.seedLoadedModelForTesting(
            id: Gemma4Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 0)
        )
        await pool.seedLoadedModelForTesting(
            id: Q35Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 1),
            activeRequests: 1
        )
        await pool.seedLoadedModelForTesting(
            id: LFM2Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 2)
        )

        let evicted = await pool.evictIdleModelsForMemoryPressure(
            sample: RuntimeMemorySample(physicalBytes: 100 * gib, residentBytes: 95 * gib)
        )

        XCTAssertEqual(evicted, [LFM2Resources.defaultModelId])
        let status = await pool.status()
        let gemma = try XCTUnwrap(status.models.first { $0.id == Gemma4Resources.defaultModelId })
        let q35 = try XCTUnwrap(status.models.first { $0.id == Q35Resources.defaultModelId })
        let lfm2 = try XCTUnwrap(status.models.first { $0.id == LFM2Resources.defaultModelId })
        XCTAssertTrue(gemma.loaded)
        XCTAssertTrue(gemma.pinned)
        XCTAssertTrue(q35.loaded)
        XCTAssertEqual(q35.activeRequests, 1)
        XCTAssertFalse(lfm2.loaded)
    }

    func testStatusReportsResidentMemoryPressure() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let gibibyte = gib
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            currentMemorySample: {
                RuntimeMemorySample(physicalBytes: 100 * gibibyte, residentBytes: 85 * gibibyte)
            }
        )

        let status = await pool.status()

        XCTAssertEqual(status.memory.physicalBytes, 100 * gib)
        XCTAssertEqual(status.memory.residentBytes, 85 * gib)
        XCTAssertEqual(status.memory.currentBytes, 85 * gib)
        XCTAssertEqual(status.memory.ceilingBytes, 94 * gib)
        XCTAssertEqual(status.memory.pressure, RuntimeMemoryPressureLevel.elevated.rawValue)
        XCTAssertEqual(status.memory.guardTier, .balanced)
    }

    func testMemoryPressurePrefersPhysicalFootprintOverResidentSize() {
        let policy = RuntimeMemoryPressurePolicy(tier: .balanced)
        let sample = RuntimeMemorySample(
            physicalBytes: 100 * gib,
            residentBytes: 10 * gib,
            physicalFootprintBytes: 85 * gib
        )

        XCTAssertEqual(policy.currentBytes(for: sample), 85 * gib)
        XCTAssertEqual(policy.pressure(for: sample), .elevated)
    }

    func testDynamicMemoryCeilingUsesPhysicalFootprint() {
        let policy = RuntimeMemoryPressurePolicy(tier: .balanced)
        let sample = RuntimeMemorySample(
            physicalBytes: 100 * gib,
            residentBytes: 5 * gib,
            physicalFootprintBytes: 30 * gib,
            availableBytes: 10 * gib
        )

        let limits = policy.limits(for: sample)

        XCTAssertEqual(limits?.ceiling, 40 * gib)
        XCTAssertEqual(limits?.soft, 36 * gib)
        XCTAssertEqual(limits?.hard, 38 * gib)
    }

    func testStatusKeepsResidentSizeWhileGuardingOnPhysicalFootprint() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let gibibyte = gib
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            currentMemorySample: {
                RuntimeMemorySample(
                    physicalBytes: 100 * gibibyte,
                    residentBytes: 10 * gibibyte,
                    physicalFootprintBytes: 85 * gibibyte
                )
            }
        )

        let status = await pool.status()

        XCTAssertEqual(status.memory.residentBytes, 10 * gib)
        XCTAssertEqual(status.memory.currentBytes, 85 * gib)
        XCTAssertEqual(status.memory.pressure, RuntimeMemoryPressureLevel.elevated.rawValue)
    }

    func testDarwinMemorySampleIncludesPhysicalFootprint() {
        #if canImport(Darwin)
        let sample = RuntimeMemorySample.current()
        XCTAssertNotNil(sample.physicalFootprintBytes)
        XCTAssertGreaterThan(sample.physicalFootprintBytes ?? 0, 0)
        #endif
    }

    func testMemoryGuardOffDisablesPressureEviction() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = RuntimeModelPool(
            defaultModelID: "custom.gguf",
            defaultEngine: .textCode,
            startupModelPath: root.appendingPathComponent("custom.gguf").path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root),
            memoryPressurePolicy: RuntimeMemoryPressurePolicy(tier: .off)
        )
        await pool.seedLoadedModelForTesting(
            id: Gemma4Resources.defaultModelId,
            lastAccess: Date(timeIntervalSince1970: 10)
        )

        let evicted = await pool.evictIdleModelsForMemoryPressure(
            sample: RuntimeMemorySample(physicalBytes: 100 * gib, residentBytes: 99 * gib)
        )

        XCTAssertTrue(evicted.isEmpty)
        let status = await pool.status()
        XCTAssertEqual(status.memory.pressure, RuntimeMemoryPressureLevel.disabled.rawValue)
    }

    func testCustomMemoryGuardUsesConfiguredCeiling() {
        let policy = RuntimeMemoryPressurePolicy(
            tier: .custom,
            customCeilingBytes: 40 * gib
        )
        let sample = RuntimeMemorySample(
            physicalBytes: 100 * gib,
            residentBytes: 37 * gib
        )

        let limits = policy.limits(for: sample)

        XCTAssertEqual(limits?.ceiling, 40 * gib)
        XCTAssertEqual(limits?.soft, 36 * gib)
        XCTAssertEqual(limits?.hard, 38 * gib)
        XCTAssertEqual(policy.pressure(for: sample), .elevated)
    }

    func testRequestAdmissionSerializesByDefaultAndReportsQueue() async throws {
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let first = try await admission.acquire()
        let secondTask = Task {
            try await admission.acquire()
        }

        var snapshot = await admission.snapshot()
        for _ in 0..<20 {
            guard snapshot.queuedRequests == 0 else { break }
            try await Task.sleep(nanoseconds: 10_000_000)
            snapshot = await admission.snapshot()
        }

        XCTAssertEqual(snapshot.maxActiveRequests, 1)
        XCTAssertEqual(snapshot.activeRequests, 1)
        XCTAssertEqual(snapshot.queuedRequests, 1)
        XCTAssertEqual(snapshot.totalAdmittedRequests, 1)
        XCTAssertEqual(snapshot.totalCompletedRequests, 0)
        XCTAssertEqual(snapshot.totalCancelledRequests, 0)
        XCTAssertEqual(snapshot.admissionPaused, false)
        XCTAssertEqual(snapshot.pressure, RuntimeMemoryPressureLevel.nominal.rawValue)

        await first.release()
        let second = try await secondTask.value
        snapshot = await admission.snapshot()

        XCTAssertEqual(snapshot.activeRequests, 1)
        XCTAssertEqual(snapshot.queuedRequests, 0)
        XCTAssertEqual(snapshot.totalAdmittedRequests, 2)
        XCTAssertEqual(snapshot.totalCompletedRequests, 1)
        XCTAssertEqual(snapshot.totalCancelledRequests, 0)

        await second.release()
        snapshot = await admission.snapshot()

        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.queuedRequests, 0)
        XCTAssertEqual(snapshot.totalCompletedRequests, 2)
        XCTAssertEqual(snapshot.totalCancelledRequests, 0)
    }

    func testRequestAdmissionRechecksCapacityAfterSuspendedPressureSample() async throws {
        let pressure = RuntimeSuspendedMemoryPressureProbe(suspendedCallCount: 2)
        let heldLeases = RuntimeAdmissionHoldProbe()
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1) {
            await pressure.current()
        }

        let firstTask = Task {
            let lease = try await admission.acquire()
            await heldLeases.hold(lease)
        }
        for _ in 0..<50 {
            if await pressure.callCount() == 1 { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }

        let secondTask = Task {
            let lease = try await admission.acquire()
            await heldLeases.hold(lease)
        }
        for _ in 0..<50 {
            if await pressure.callCount() == 2 { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        let suspendedSamples = await pressure.callCount()
        XCTAssertEqual(suspendedSamples, 2)

        await pressure.releaseAll()
        var snapshot = await admission.snapshot()
        for _ in 0..<50 {
            if snapshot.activeRequests == 1,
               snapshot.queuedRequests == 1,
               snapshot.totalAdmittedRequests == 1 {
                break
            }
            try await Task.sleep(nanoseconds: 10_000_000)
            snapshot = await admission.snapshot()
        }

        let initiallyHeld = await heldLeases.count()
        XCTAssertEqual(initiallyHeld, 1)
        XCTAssertEqual(snapshot.activeRequests, 1)
        XCTAssertEqual(snapshot.queuedRequests, 1)
        XCTAssertEqual(snapshot.totalAdmittedRequests, 1)

        await heldLeases.releaseNext()
        for _ in 0..<50 {
            if await heldLeases.count() == 1 { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 1)
        XCTAssertEqual(snapshot.queuedRequests, 0)
        XCTAssertEqual(snapshot.totalAdmittedRequests, 2)

        await heldLeases.releaseNext()
        _ = try await (firstTask.value, secondTask.value)
        snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.totalCompletedRequests, 2)
    }

    func testRequestAdmissionPausesAdditionalWorkUnderPressure() async throws {
        let pressure = RuntimeMemoryPressureProbe(.elevated)
        let admission = RuntimeRequestAdmission(maxActiveRequests: 2) {
            await pressure.current()
        }
        let first = try await admission.acquire()
        let secondTask = Task {
            try await admission.acquire()
        }

        var snapshot = await admission.snapshot()
        for _ in 0..<20 {
            guard snapshot.queuedRequests == 0 else { break }
            try await Task.sleep(nanoseconds: 10_000_000)
            snapshot = await admission.snapshot()
        }

        XCTAssertEqual(snapshot.activeRequests, 1)
        XCTAssertEqual(snapshot.queuedRequests, 1)
        XCTAssertEqual(snapshot.admissionPaused, true)
        XCTAssertEqual(snapshot.pressure, RuntimeMemoryPressureLevel.elevated.rawValue)

        await pressure.set(.nominal)
        let thirdTask = Task {
            try await admission.acquire()
        }
        let second = try await secondTask.value
        snapshot = await admission.snapshot()

        XCTAssertEqual(snapshot.activeRequests, 2)
        XCTAssertEqual(snapshot.queuedRequests, 1)
        XCTAssertEqual(snapshot.admissionPaused, false)

        await first.release()
        let third = try await thirdTask.value
        await second.release()
        await third.release()
    }

    func testRequestAdmissionCancelsQueuedWaiterWithoutLaterAdmission() async throws {
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let first = try await admission.acquire()
        let queued = Task {
            try await admission.acquire()
        }

        var snapshot = await admission.snapshot()
        for _ in 0..<20 {
            guard snapshot.queuedRequests == 0 else { break }
            try await Task.sleep(nanoseconds: 10_000_000)
            snapshot = await admission.snapshot()
        }
        XCTAssertEqual(snapshot.queuedRequests, 1)

        queued.cancel()
        do {
            _ = try await queued.value
            XCTFail("Expected queued admission to throw CancellationError")
        } catch is CancellationError {
            // Expected.
        }

        snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 1)
        XCTAssertEqual(snapshot.queuedRequests, 0)
        XCTAssertEqual(snapshot.totalAdmittedRequests, 1)
        XCTAssertEqual(snapshot.totalCompletedRequests, 0)
        XCTAssertEqual(snapshot.totalCancelledRequests, 1)

        await first.release()
        snapshot = await admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.queuedRequests, 0)
        XCTAssertEqual(snapshot.totalAdmittedRequests, 1)
        XCTAssertEqual(snapshot.totalCompletedRequests, 1)
        XCTAssertEqual(snapshot.totalCancelledRequests, 1)
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-runtime-pool-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

private actor RuntimeMemoryPressureProbe {
    private var level: RuntimeMemoryPressureLevel

    init(_ level: RuntimeMemoryPressureLevel) {
        self.level = level
    }

    func set(_ level: RuntimeMemoryPressureLevel) {
        self.level = level
    }

    func current() -> RuntimeMemoryPressureLevel {
        level
    }
}

private actor RuntimeSuspendedMemoryPressureProbe {
    private let suspendedCallCount: Int
    private var calls = 0
    private var continuations: [CheckedContinuation<Void, Never>] = []

    init(suspendedCallCount: Int) {
        self.suspendedCallCount = suspendedCallCount
    }

    func current() async -> RuntimeMemoryPressureLevel {
        calls += 1
        if calls <= suspendedCallCount {
            await withCheckedContinuation { continuation in
                continuations.append(continuation)
            }
        }
        return .nominal
    }

    func callCount() -> Int {
        calls
    }

    func releaseAll() {
        let pending = continuations
        continuations.removeAll()
        pending.forEach { $0.resume() }
    }
}

private actor RuntimeAdmissionHoldProbe {
    private var leases: [RuntimeRequestAdmissionLease] = []

    func hold(_ lease: RuntimeRequestAdmissionLease) {
        leases.append(lease)
    }

    func count() -> Int {
        leases.count
    }

    func releaseNext() async {
        guard !leases.isEmpty else { return }
        let lease = leases.removeFirst()
        await lease.release()
    }
}

private actor RuntimeModelPreparationProbe {
    private var calls = 0
    private var continuations: [CheckedContinuation<Void, Never>] = []

    func prepare() async {
        calls += 1
        await withCheckedContinuation { continuation in
            continuations.append(continuation)
        }
    }

    func callCount() -> Int {
        calls
    }

    func releaseAll() {
        let pending = continuations
        continuations.removeAll()
        pending.forEach { $0.resume() }
    }
}

private actor RuntimeOrderedModelPreparationProbe {
    private var calls = 0
    private var continuations: [Int: CheckedContinuation<Void, Never>] = [:]

    func prepare() async {
        calls += 1
        let call = calls
        await withCheckedContinuation { continuation in
            continuations[call] = continuation
        }
    }

    func callCount() -> Int {
        calls
    }

    func release(call: Int) {
        continuations.removeValue(forKey: call)?.resume()
    }
}

private enum RuntimeModelPreparationTestError: Error {
    case failed
}

private actor RuntimeModelUnloadProbe {
    private var calls = 0
    private var continuations: [CheckedContinuation<Void, Never>] = []

    func unload() async {
        calls += 1
        await withCheckedContinuation { continuation in
            continuations.append(continuation)
        }
    }

    func callCount() -> Int {
        calls
    }

    func releaseAll() {
        let pending = continuations
        continuations.removeAll()
        pending.forEach { $0.resume() }
    }
}

private actor RuntimeCompletionProbe {
    private var completions = 0

    func mark() {
        completions += 1
    }

    func count() -> Int {
        completions
    }
}

private final class RuntimeMemorySampleSequenceProbe: @unchecked Sendable {
    private let lock = NSLock()
    private let samples: [RuntimeMemorySample]
    private var index = 0

    init(_ samples: [RuntimeMemorySample]) {
        precondition(!samples.isEmpty)
        self.samples = samples
    }

    func next() -> RuntimeMemorySample {
        lock.lock()
        defer { lock.unlock() }
        let sample = samples[min(index, samples.count - 1)]
        index += 1
        return sample
    }

    func readCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return index
    }
}
