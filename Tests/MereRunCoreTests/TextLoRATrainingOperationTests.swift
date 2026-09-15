import Foundation
import XCTest
@testable import MereRunCore

final class TextLoRATrainingOperationTests: XCTestCase {
    func testDryRunSkipsTrainerAndCreatesOnlyManifest() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let plan = try TextLoRATrainingPlan.resolve(TextLoRATrainingOptions(
            data: fixture.data.path, output: fixture.output.path, dryRun: true
        ))
        let outcome = try await TextLoRATrainingOperation.execute(plan, trainer: { _, _, _ in
            XCTFail("Dry runs must not enter a native trainer")
            throw FixtureError.failed
        })
        XCTAssertNil(outcome.report)
        XCTAssertEqual(outcome.manifest.status, "prepared")
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.output.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: TextLoRATrainingManifest.url(nextTo: fixture.output).path))
    }

    func testTrainingReceivesSnapshotAndPublishesAfterSuccess() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let plan = try TextLoRATrainingPlan.resolve(TextLoRATrainingOptions(
            data: fixture.data.path, output: fixture.output.path, model: InklingResources.modelID,
            modelPath: "/explicit/model", eval: fixture.data.path, trainingSteps: 7, rank: 4, reasoningEffort: 0.2
        ))
        let progressReceived = expectation(description: "Loading progress forwarded")
        let trainingProgressReceived = expectation(description: "Training progress forwarded")
        let outcome = try await TextLoRATrainingOperation.execute(
            plan,
            progressHandler: { _ in progressReceived.fulfill() },
            trainingProgressHandler: { _ in trainingProgressReceived.fulfill() },
            trainer: { received, progress, trainingProgress in
                XCTAssertEqual(received.family, .inkling)
                XCTAssertEqual(received.options.modelPath, "/explicit/model")
                XCTAssertEqual(received.dataset.examples.count, 1)
                XCTAssertEqual(received.evaluationDataset?.summary.fingerprint, received.dataset.summary.fingerprint)
                XCTAssertFalse(FileManager.default.fileExists(atPath: TextLoRATrainingManifest.url(nextTo: received.outputURL).path))
                progress?(ChatProgress(stage: .generating, message: "Fixture"))
                trainingProgress?(TextLoRATrainingProgress(stage: .saving))
                try Data("adapter".utf8).write(to: received.outputURL)
                return Self.report(output: received.outputURL)
        })
        XCTAssertEqual(outcome.manifest.status, "trained")
        XCTAssertEqual(outcome.manifest.training.reasoningEffort, 0.2)
        XCTAssertEqual(outcome.manifest.lora.alpha, 4)
        XCTAssertEqual(outcome.manifest.evalPromptCount, 1)
        XCTAssertEqual(outcome.report?.steps, 7)
        await fulfillment(of: [progressReceived, trainingProgressReceived], timeout: 1)
    }

    func testCancellationBeforeExecutionDoesNotCreateOutputDirectory() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let plan = try TextLoRATrainingPlan.resolve(TextLoRATrainingOptions(
            data: fixture.data.path, output: fixture.output.path, dryRun: true
        ))
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await TextLoRATrainingOperation.execute(plan)
        }
        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.output.deletingLastPathComponent().path))
    }

    func testFailurePreservesPreviousManifest() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let manifestURL = TextLoRATrainingManifest.url(nextTo: fixture.output)
        try FileManager.default.createDirectory(at: manifestURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let previous = Data("previous manifest".utf8)
        try previous.write(to: manifestURL)
        let plan = try TextLoRATrainingPlan.resolve(TextLoRATrainingOptions(data: fixture.data.path, output: fixture.output.path))
        do {
            _ = try await TextLoRATrainingOperation.execute(plan, trainer: { _, _, _ in throw FixtureError.failed })
            XCTFail("Expected trainer failure")
        } catch {
            XCTAssertTrue(error is FixtureError)
        }
        XCTAssertEqual(try Data(contentsOf: manifestURL), previous)
    }

    func testCancelledTrainerCannotPublishTrainedManifest() async throws {
        for throwsError in [false, true] {
            let fixture = try makeFixture()
            defer { try? FileManager.default.removeItem(at: fixture.root) }
            let plan = try TextLoRATrainingPlan.resolve(TextLoRATrainingOptions(data: fixture.data.path, output: fixture.output.path))
            let trainer: TextLoRATrainingOperation.Trainer = { received, _, _ in
                withUnsafeCurrentTask { $0?.cancel() }
                if throwsError { throw FixtureError.failed }
                return Self.report(output: received.outputURL)
            }
            let task = Task { try await TextLoRATrainingOperation.execute(plan, trainer: trainer) }
            do {
                _ = try await task.value
                XCTFail("Expected cancellation")
            } catch {
                XCTAssertTrue(error is CancellationError)
            }
            XCTAssertFalse(FileManager.default.fileExists(atPath: TextLoRATrainingManifest.url(nextTo: fixture.output).path))
        }
    }

    func testResumeValidationProtectsSourceAndPreservesGlobalStep() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let checkpoint = fixture.root.appendingPathComponent("resume.safetensors")
        try Data("optimizer fixture".utf8).write(to: checkpoint)
        let options = TextLoRATrainingOptions(data: fixture.data.path, output: fixture.output.path,
                                            trainingSteps: 10, resumeFrom: checkpoint.path, resumeStep: 4)
        let plan = try TextLoRATrainingPlan.resolve(options)
        XCTAssertEqual(plan.options.resumeStep, 4)
        XCTAssertEqual(plan.options.trainingSteps, 10)
        XCTAssertEqual(plan.options.resumeFrom, checkpoint.path)
        XCTAssertThrowsError(try TextLoRATrainingPlan.resolve(TextLoRATrainingOptions(
            data: fixture.data.path, output: checkpoint.path, resumeFrom: checkpoint.path
        ))) { error in
            XCTAssertEqual(String(describing: error), "--resume-from and --output must be different files")
        }
    }

    func testAllTextFamiliesKeepManifestAndTargetDefaults() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let cases: [(String, TextLoRATrainingFamily, String, String)] = [
            (Gemma4Resources.twelveB4BitModelId, .gemma4, TextLoRATrainingManifest.gemma4Format, "o_proj"),
            ("", .gemma4, TextLoRATrainingManifest.gemma4Format, "o_proj"),
            ("  ", .gemma4, TextLoRATrainingManifest.gemma4Format, "o_proj"),
            (LagunaResources.xsModelID, .lagunaXS, TextLoRATrainingManifest.lagunaFormat, "o_proj"),
            (InklingResources.modelID, .inkling, TextLoRATrainingManifest.inklingFormat, "lm_head"),
            (LFM2Resources.defaultModelId, .lfm2A1B, TextLoRATrainingManifest.lfm2Format, "out_proj"),
        ]
        for (model, family, format, lastTarget) in cases {
            let plan = try TextLoRATrainingPlan.resolve(TextLoRATrainingOptions(
                data: fixture.data.path, output: fixture.output.path, model: model, dryRun: true
            ))
            XCTAssertEqual(plan.family, family)
            XCTAssertEqual(plan.family.manifestFormat, format)
            XCTAssertEqual(plan.options.resolvedTargetModules().last, lastTarget)
        }
    }

    private enum FixtureError: Error { case failed }

    private static func report(output: URL) -> TextLoRATrainingReport {
        TextLoRATrainingReport(steps: 7, initialLoss: 1, finalLoss: 0.5,
                               initialEvaluationLoss: nil, finalEvaluationLoss: nil,
                               evaluationExampleCount: 0, evaluationTargetTokenCount: 0,
                               layerCount: 1, outputPath: output.path)
    }

    private func makeFixture() throws -> (root: URL, data: URL, output: URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let data = root.appendingPathComponent("pairs.jsonl")
        let example = TextSFTExample(id: nil, sources: ["test"], messages: [
            ChatMessage(role: .system, content: "Be helpful"),
            ChatMessage(role: .user, content: "Hello"),
            ChatMessage(role: .assistant, content: "Hello there"),
        ])
        try JSONEncoder().encode(example).write(to: data)
        return (root, data, root.appendingPathComponent("output/adapter.safetensors"))
    }
}
