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
            engineOverride: .textChatGemma4
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
                RuntimeModelSettings(engineOverride: .textChatQ35),
                for: Gemma4Resources.defaultModelId
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("not compatible"))
        }

        XCTAssertThrowsError(
            try store.writeSettings(RuntimeModelSettings(alias: "image"), for: "image-zimage-nano")
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("not supported"))
        }
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-runtime-settings-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
