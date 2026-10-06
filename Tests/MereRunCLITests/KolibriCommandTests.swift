import XCTest
import Foundation
@testable import MereRunCLI
@testable import MereRunCore

final class KolibriCommandTests: XCTestCase {
    func testLocalCheckpointIdentifierPreservesPathCase() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("Kolibri-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(#"{"model_type":"kolibri1"}"#.utf8).write(to: root.appendingPathComponent("config.json"))
        let command = try TextChat.parse(["--model", root.path, "--prompt", "hello"])
        XCTAssertEqual(command.normalizedModelIdentifier, root.path)
        let report = command.makePreflightReport(modelID: root.path, installedModelPath: root.path)
        XCTAssertTrue(report.diagnostics.isEmpty)
    }

    func testUnpublishedNativeIdentifierRequiresCheckpointInsteadOfManagedDownload() throws {
        let command = try TextChat.parse(["--model", KolibriResources.mixedModelID, "--prompt", "hello"])
        let report = command.makePreflightReport(modelID: KolibriResources.mixedModelID, installedModelPath: nil)
        XCTAssertEqual(report.diagnostics.map(\.id), ["text_chat_kolibri_root_required"])
    }

    func testKolibriServingRequiresAnExplicitConvertedRoot() throws {
        let missing = try APIServe.parse(["--engine", "text-chat-kolibri"])
        XCTAssertThrowsError(try missing.resolveModelPath())
        let command = try APIServe.parse(["--engine", "text-chat-kolibri", "--model", "/models/kolibri-mixed2"])
        XCTAssertEqual(command.engine.runtimeServingEngine, .textChatKolibri)
        XCTAssertEqual(try command.resolveModelPath(), "/models/kolibri-mixed2")
    }

    func testBenchmarkParsesScoringAndCalibrationPaths() throws {
        let command = try ModelBenchmarkKolibri.parse([
            "--model-root", "/models/kolibri-reference", "--suite", "/fixtures/suite.json",
            "--output", "/results/reference", "--chunk-size", "16",
            "--calibration-output", "/results/moments.safetensors"
        ])
        XCTAssertEqual(command.chunkSize, 16)
        XCTAssertEqual(command.calibrationOutput, "/results/moments.safetensors")
    }
}
