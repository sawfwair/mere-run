import Foundation
import XCTest
@testable import MereRunCLI
import MereRunCore
import MereRunRelayKit

final class LFM25LocalCheckpointCommandTests: XCTestCase {
    private func fixture(modelType: String = "lfm2_moe") throws -> URL {
        let root = FileManager.default.temporaryDirectory.appending(path: "Native Checkpoint-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appending(path: "MereRunCoreTests/Fixtures/LFM25A2B24B/config.json")
        var text = try String(contentsOf: fixtures, encoding: .utf8)
        text = text.replacingOccurrences(of: "\"lfm2_moe\"", with: "\"\(modelType)\"")
        try Data(text.utf8).write(to: root.appending(path: "config.json"))
        return root
    }

    func testLocalLFMCheckpointPreservesCaseAndPassesPreflightIdentityChecks() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let command = try TextChat.parse(["--model", root.path, "--prompt", "hello", "--context-size", "8192"])
        XCTAssertEqual(command.normalizedModelIdentifier, root.path)
        XCTAssertEqual(NativeChatRuntime.commandFamily(modelID: root.path), .lfm2A2B)
        let report = command.makePreflightReport(modelID: root.path, installedModelPath: root.path)
        XCTAssertTrue(report.diagnostics.isEmpty)
    }

    func testLocalCheckpointDoesNotRemoveResourceBlockers() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let command = try TextChat.parse(["--model", root.path, "--prompt", "hello"])
        let blocker = PreflightDiagnostic(
            id: "machine_insufficient_memory", severity: .blocker,
            title: "Insufficient memory", message: "Real admission must still pass."
        )
        let report = command.makePreflightReport(
            modelID: root.path, installedModelPath: root.path, resourceDiagnostics: [blocker]
        )
        XCTAssertEqual(report.status, .blocked)
        XCTAssertEqual(report.diagnostics, [blocker])
    }

    func testOtherModelTypesDoNotBecomeLFMFromSharedConfigFields() throws {
        let root = try fixture(modelType: "qwen3_moe")
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertNil(ModelFamilyIdentifier.lfm2Family(folder: root.path))
        let command = try TextChat.parse(["--model", root.path, "--prompt", "hello"])
        let report = command.makePreflightReport(modelID: root.path, installedModelPath: root.path)
        XCTAssertTrue(report.diagnostics.contains { $0.id == "text_chat_model_unknown" })
    }
}
