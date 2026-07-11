import Foundation
import XCTest
@testable import MereRunCore

final class RuntimeModelSettingsTests: XCTestCase {
    func testSettingsRoundTripAndAliasResolution() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RuntimeModelSettingsStore(modelsDir: root)
        let settings = RuntimeModelSettings(
            alias: "chat-default",
            pinned: true,
            ttlSeconds: 600,
            maxContextTokens: 8192,
            maxTokens: 512,
            temperature: 0.4,
            topP: 0.8,
            engineOverride: .textChatGemma4,
            kvCacheMode: .auto
        )

        try store.writeSettings(settings, for: Gemma4Resources.defaultModelId)
        let loaded = try store.settings(for: Gemma4Resources.defaultModelId)

        XCTAssertEqual(loaded.alias, "chat-default")
        XCTAssertTrue(loaded.pinned)
        XCTAssertEqual(loaded.ttlSeconds, 600)
        XCTAssertEqual(loaded.maxContextTokens, 8192)
        XCTAssertEqual(loaded.maxTokens, 512)
        XCTAssertEqual(loaded.temperature, 0.4)
        XCTAssertEqual(loaded.topP, 0.8)
        XCTAssertEqual(loaded.engineOverride, .textChatGemma4)
        XCTAssertEqual(loaded.kvCacheMode, .auto)
        XCTAssertEqual(
            try store.resolveModelID(aliasOrID: "chat-default", defaultModelID: "fallback"),
            Gemma4Resources.defaultModelId
        )
    }

    func testSettingsRejectUnsupportedAndIncompatibleEngine() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RuntimeModelSettingsStore(modelsDir: root)

        XCTAssertThrowsError(
            try store.writeSettings(
                RuntimeModelSettings(engineOverride: .textChatQ36),
                for: Gemma4Resources.defaultModelId
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("not compatible"))
        }

        XCTAssertThrowsError(
            try store.writeSettings(RuntimeModelSettings(alias: "image"), for: "image-zimage-nano")
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("sidecar models support only"))
        }
    }

    func testSidecarSettingsAllowOnlyResidencyControls() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RuntimeModelSettingsStore(modelsDir: root)

        try store.writeSettings(
            RuntimeModelSettings(pinned: true, ttlSeconds: 45),
            for: "image-zimage-nano"
        )
        let settings = try store.settings(for: "image-zimage-nano")

        XCTAssertTrue(settings.pinned)
        XCTAssertEqual(settings.ttlSeconds, 45)
        XCTAssertTrue(try XCTUnwrap(ManagedModelCatalog.spec(for: "image-zimage-nano"))
            .supportsRuntimeResidencySettings)
    }

    func testSettingsRejectGemmaKVModeForNonGemmaModel() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RuntimeModelSettingsStore(modelsDir: root)

        XCTAssertThrowsError(
            try store.writeSettings(RuntimeModelSettings(kvCacheMode: .polar2), for: Q35Resources.defaultModelId)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("KV cache mode"))
        }
    }

    func testAffineEightKVModeIsAcceptedByNativeAttentionEngines() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RuntimeModelSettingsStore(modelsDir: root)

        for modelID in [
            Gemma4Resources.defaultModelId,
            Q35Resources.defaultModelId,
            LFM2Resources.defaultModelId,
        ] {
            try store.writeSettings(RuntimeModelSettings(kvCacheMode: .affine8), for: modelID)
            XCTAssertEqual(try store.settings(for: modelID).kvCacheMode, .affine8)
        }
    }

    func testGemmaAffineEightMapsToUniformQuantization() {
        let fallback = Gemma4KVCacheQuantization(bits: nil, scheme: .uniform, groupSize: 64, quantizedStart: 128)
        let resolved = RuntimeKVCacheMode.affine8.gemma4Quantization(
            fallback: fallback,
            promptTokenCount: 10
        )
        XCTAssertEqual(resolved.scheme, .uniform)
        XCTAssertEqual(resolved.bits, 8)
        XCTAssertEqual(resolved.groupSize, 64)
        XCTAssertEqual(resolved.quantizedStart, 0)
    }

    func testQ36DefaultRuntimeEngineAcceptsLegacyQ35Alias() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = RuntimeModelSettingsStore(modelsDir: root)
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.defaultModelId))

        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatQ36)

        try store.writeSettings(
            RuntimeModelSettings(engineOverride: .textChatQ35),
            for: Q35Resources.defaultModelId
        )
        let settings = try store.settings(for: Q35Resources.defaultModelId)

        XCTAssertEqual(settings.engineOverride, .textChatQ35)
        XCTAssertTrue(settings.engineOverride?.isCompatible(with: .textChatQ36) == true)
    }

    func testGemmaAutoKVModeUsesPolarOnlyPastPromptThreshold() {
        let fallback = Gemma4KVCacheQuantization(
            bits: Gemma4Resources.defaultTurboKVBits,
            scheme: Gemma4Resources.defaultTurboKVQuantizationScheme,
            groupSize: Gemma4Resources.defaultKVGroupSize,
            quantizedStart: Gemma4Resources.defaultTurboQuantizedKVStart
        )

        let short = RuntimeKVCacheMode.auto.gemma4Quantization(
            fallback: fallback,
            promptTokenCount: RuntimeKVCacheMode.gemma4AutoPolarPromptTokenThreshold - 1
        )
        XCTAssertEqual(short.scheme, .turboquant)
        XCTAssertEqual(short.bits, Gemma4Resources.defaultTurboKVBits)

        let long = RuntimeKVCacheMode.auto.gemma4Quantization(
            fallback: fallback,
            promptTokenCount: RuntimeKVCacheMode.gemma4AutoPolarPromptTokenThreshold
        )
        XCTAssertEqual(long.scheme, .polar)
        XCTAssertEqual(long.bits, 2)
        XCTAssertEqual(long.quantizedStart, 0)
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-runtime-settings-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
