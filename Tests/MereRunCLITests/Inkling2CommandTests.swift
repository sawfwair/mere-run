import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class Inkling2CommandTests: XCTestCase {
    func testLocalCheckpointPathPreservesCaseAndResolvesNativeFamily() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("Inkling2-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(#"{"model_type":"inkling2"}"#.utf8).write(to: root.appendingPathComponent("config.json"))
        let image = root.appendingPathComponent("image.png")
        try Data().write(to: image)
        let command = try TextChat.parse(["--model", root.path, "--prompt", "hello", "--image", image.path, "--reasoning-effort", "0.2"])
        XCTAssertEqual(command.normalizedModelIdentifier, root.path)
        XCTAssertEqual(NativeChatRuntime.commandFamily(modelID: root.path), .inkling2)
        XCTAssertTrue(command.makePreflightReport(modelID: root.path, installedModelPath: root.path).diagnostics.isEmpty)
        let server = try APIServe.parse(["--engine", "text-chat-inkling2", "--model", root.path])
        XCTAssertEqual(server.engine.runtimeServingEngine, .textChatInkling2)
        XCTAssertEqual(try server.resolveModelPath(), root.path)
    }

    func testServingRequiresInstalledCheckpoint() throws {
        let server = try APIServe.parse(["--engine", "text-chat-inkling2", "--model", "/missing-inkling2-checkpoint"])
        XCTAssertThrowsError(try server.resolveModelPath())
    }

    func testAPICapabilitiesRetainNativeToolHistoryAndSampling() throws {
        let profile = try XCTUnwrap(ManagedModelCatalog.spec(for: Inkling2Resources.modelID)?.apiProfile)
        let capabilities = APIEngineCapabilities.catalog(profile)
        XCTAssertTrue(capabilities.usesNativeToolHistory)
        XCTAssertTrue(capabilities.supportsVisionContentParts)
        XCTAssertTrue(capabilities.supportsTopK)
        XCTAssertTrue(capabilities.supportsRepetitionPenalty)
        XCTAssertTrue(capabilities.supportsReasoningEffort)
        XCTAssertFalse(capabilities.supportsStructuredOutputs)
    }
}
