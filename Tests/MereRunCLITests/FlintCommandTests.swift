import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class FlintCommandTests: XCTestCase {
    func testLocalCheckpointPreservesCaseAndPassesPreflight() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("Flint-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(#"{"model_type":"flint"}"#.utf8).write(to: root.appendingPathComponent("config.json"))
        let command = try TextChat.parse(["--model", root.path, "--prompt", "hello"])
        XCTAssertEqual(command.normalizedModelIdentifier, root.path)
        XCTAssertTrue(command.makePreflightReport(modelID: root.path, installedModelPath: root.path).diagnostics.isEmpty)
        let serve = try APIServe.parse(["--engine", "text-chat-flint", "--model", root.path])
        XCTAssertEqual(serve.engine.runtimeServingEngine, .textChatFlint)
        XCTAssertEqual(try serve.resolveModelPath(), root.path)
    }

    func testManagedModelHasNativeServingProfile() throws {
        let command = try APIServe.parse(["--engine", "text-chat-flint"])
        XCTAssertEqual(command.engine.runtimeServingEngine, .textChatFlint)
        let profile = try XCTUnwrap(ManagedModelCatalog.spec(for: FlintResources.modelID)?.apiProfile)
        let capabilities = APIEngineCapabilities.catalog(profile)
        XCTAssertTrue(capabilities.usesNativeToolHistory)
        XCTAssertTrue(capabilities.supportsTopK)
        XCTAssertFalse(capabilities.supportsStructuredOutputs)
        XCTAssertFalse(capabilities.supportsReasoningEffort)
    }
}
