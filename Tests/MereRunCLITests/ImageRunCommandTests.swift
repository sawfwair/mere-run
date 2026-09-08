import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class ImageRunCommandTests: XCTestCase {
    func testFlagsAreOptInAndPreflightDoesNotCreateRecord() throws {
        let root = try temporaryDirectory()
        try MereRunModelManifest(id: "fixture", family: .zimage).write(to: root)
        let directory = root.appendingPathComponent("run")
        let command = try ImageGenerate.parse([
            "--prompt", "A camera", "--model", root.path, "--run-dir", directory.path, "--preflight", "--json"
        ])
        XCTAssertEqual(command.runDirectory, directory.path)
        let envelope = command.makePreflightEnvelope(outputURL: root.appendingPathComponent("output.png"))
        XCTAssertEqual(envelope.status, .ok)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
        let replay = try ImageGenerate.parse(envelope.result.runPlan.arguments.generateArguments())
        XCTAssertEqual(replay.runDirectory, directory.path)
        XCTAssertTrue(envelope.actions.contains { $0.command?.argv.contains("--run-dir") == true })
        XCTAssertNil(try ImageGenerate.parse(["--prompt", "A camera"]).runDirectory)
        XCTAssertNil(try APIServe.parse([]).imageRunRecords)
        XCTAssertEqual(try APIServe.parse(["--image-run-records", root.path]).imageRunRecords, root.path)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        XCTAssertTrue(command.makePreflightEnvelope(outputURL: root.appendingPathComponent("output.png"))
            .diagnostics.contains { $0.id == "image_run_directory_exists" && $0.severity == .blocker })
    }

    func testModelResolutionFailureIsRecordedWithoutLoadingRuntime() async throws {
        let root = try temporaryDirectory()
        let directory = root.appendingPathComponent("run")
        let command = try ImageGenerate.parse([
            "--prompt", "A camera", "--model", root.appendingPathComponent("missing-model").path, "--run-dir", directory.path
        ])
        do {
            try await command.run()
            XCTFail("Expected missing model failure")
        } catch {
            let record = try ImageRunRecord.inspect(at: directory)
            XCTAssertEqual(record.state, .failed)
            XCTAssertNil(record.effective)
            XCTAssertEqual(record.requested.prompt, "A camera")
            XCTAssertTrue(record.artifacts.isEmpty)
        }
    }

    func testListInspectAndRetryActionRecognizeRecordWithoutChangingLegacyEncoding() throws {
        let root = try temporaryDirectory()
        let directory = root.appendingPathComponent("run")
        let options = ImageGenerationOptions(prompt: "A camera", outputURL: root.appendingPathComponent("output.png"))
        let session = try ImageRunSession(directory: directory, requested: options, modelSelector: "fixture")
        _ = try session.prepare(ImageGenerationPlan.resolve(options, modelRoot: root, manifest: .init(id: "fixture", family: .zimage)))
        let active = try RunInspect.parse([directory.path, "--json"]).makeInspectionEnvelope()
        XCTAssertEqual(active.result.kind, "image_run")
        XCTAssertEqual(active.result.imageRun?.state, .running)
        XCTAssertEqual(active.actions.first { $0.id == "retry-image-run" }?.enabled, false)
        try session.fail(ImageGenerationIssue("fixture_failure", "Fixture failed"))
        let inspection = try RunInspect.parse([directory.path, "--json"]).makeInspectionEnvelope(now: { Date(timeIntervalSince1970: 10) })
        XCTAssertEqual(inspection.result.imageRun?.state, .failed)
        XCTAssertEqual(inspection.actions.first { $0.id == "retry-image-run" }?.enabled, true)
        let list = try RunList.parse(["--root", root.path, "--json"]).makeListEnvelope()
        XCTAssertEqual(list.result.entryCount, 1)
        XCTAssertEqual(list.result.entries.first?.kind, "image_run")
        XCTAssertEqual(list.result.entries.first?.state, "failed")
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        XCTAssertEqual(try decoder.decode(RunInspectionEnvelope.self, from: Data(StructuredRunOutput.encode(inspection).utf8)), inspection)
        let legacy = RunInspectionResult(kind: "missing", path: "/missing", runDirectory: nil, report: nil, plan: nil)
        XCTAssertFalse(try StructuredRunOutput.encode(legacy).contains("image_run"))
        let retry = try RunRetry.parse([directory.path, "--json"])
        XCTAssertEqual(retry.reference, directory.path)
        XCTAssertTrue(retry.json)
    }

    func testCorruptRecordBlocksInspectionAndIsListedForReview() throws {
        let root = try temporaryDirectory()
        let directory = root.appendingPathComponent("run")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("broken".utf8).write(to: ImageRunRecord.recordURL(at: directory))
        let inspection = try RunInspect.parse([directory.path]).makeInspectionEnvelope()
        XCTAssertEqual(inspection.status, .blocked)
        XCTAssertEqual(inspection.diagnostics.first?.id, "image_run_unreadable")
        let list = try RunList.parse(["--root", root.path]).makeListEnvelope()
        XCTAssertEqual(list.result.entryCount, 1)
        XCTAssertEqual(list.result.entries.first?.status, .blocked)
    }

    func testRetryUsesImageAdmissionClassForInstalledModelProfiles() {
        for id in ["image-zimage-nano", "image-klein-nano", "image-flux2-dev", "image-krea2"] {
            XCTAssertEqual(CLIInferenceAdmissionClassifier.imageGenerationRequest(modelID: id).resourceClass,
                           CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "image", "generate", "--model", id])?.resourceClass)
        }
        XCTAssertNil(CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "run", "retry", "/tmp/run"]))
        XCTAssertNil(CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "run", "retry", "relay://fleet/job"]))
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
}
