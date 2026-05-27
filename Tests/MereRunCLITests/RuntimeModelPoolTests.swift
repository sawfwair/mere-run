import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class RuntimeModelPoolTests: XCTestCase {
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
        XCTAssertFalse(status.capabilities.prefixKVReuse.enabled)
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
