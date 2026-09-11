import Foundation
import MediaIO
import XCTest
@testable import MereRunCore

final class ACEStepAdapterTrainingOperationTests: XCTestCase {
    func testPlanResolvesPathsWithoutLoadingAudioOrModel() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let plan = fixture.plan
        XCTAssertEqual(plan.audioURL(for: plan.records[0]), fixture.root.appendingPathComponent("one.wav"))
        XCTAssertEqual(plan.audioURL(for: .init(audio: "~/one.wav", caption: "a")),
                       URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("one.wav"))
        XCTAssertEqual(plan.audioURL(for: .init(audio: "/tmp/one.wav", caption: "a")), URL(fileURLWithPath: "/tmp/one.wav"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: plan.outputURL.deletingLastPathComponent().path))
        XCTAssertEqual(plan.options.maximumAudioFrames, 480)
    }

    func testAudioPreparationPreservesStereoAndCrops() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let source = fixture.root.appendingPathComponent("one.wav")
        try MediaAudioIO.writeFloatWAV(
            samples: (0..<960).flatMap { _ in [Float(0.25), Float(-0.5)] },
            sampleRate: 48_000, channels: 2, to: source
        )
        let examples = try ACEStepAdapterTrainingOperation.prepareExamples(fixture.plan)
        XCTAssertEqual(examples.count, 1)
        XCTAssertEqual(examples[0].audio48kHz.shape, [1, 480, 2])
        XCTAssertEqual(Array(examples[0].audio48kHz.asArray(Float.self).prefix(2)), [0.25, -0.5])
        XCTAssertEqual(examples[0].caption, " caption ")
        XCTAssertEqual(examples[0].lyrics, "")
    }

    func testExecutionForwardsPlanAndProgressThenFinishes() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let received = expectation(description: "Progress forwarded")
        let report = try await ACEStepAdapterTrainingOperation.execute(
            fixture.plan, progress: { update in
                XCTAssertEqual(update.step, 3)
                received.fulfill()
            }, trainer: { plan, progress in
                XCTAssertEqual(plan.options.configuration.kind, .lokr)
                XCTAssertEqual(plan.options.configuration.rank, 4)
                XCTAssertEqual(plan.options.configuration.factor, 2)
                XCTAssertEqual(plan.options.model, "/fixture/model")
                XCTAssertEqual(plan.records[0].caption, " caption ")
                progress?(.init(step: 3, totalSteps: 3, loss: 0.5))
                try Data("adapter".utf8).write(to: plan.outputURL)
                return Self.report()
            }
        )
        XCTAssertEqual(report.finalLoss, 0.5)
        let events = try events(fixture.plan)
        XCTAssertEqual(events.map(\.type), ["run_started", "progress", "run_finished"])
        XCTAssertEqual(events.last?.metadata["sha256"], "fixture-digest")
        await fulfillment(of: [received], timeout: 1)
    }

    func testFailureAndCancellationDoNotRecordSuccess() async throws {
        for cancellation in [false, true] {
            let fixture = try makeFixture()
            defer { try? FileManager.default.removeItem(at: fixture.root) }
            let task = Task {
                try await ACEStepAdapterTrainingOperation.execute(fixture.plan, trainer: { _, _ in
                    if cancellation { withUnsafeCurrentTask { $0?.cancel() } }
                    throw FixtureError.failed
                })
            }
            do {
                _ = try await task.value
                XCTFail("Expected failure")
            } catch {
                XCTAssertEqual(error is CancellationError, cancellation)
            }
            XCTAssertEqual(try events(fixture.plan).map(\.type), ["run_started", "run_failed"])
            XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.plan.outputURL.path))
        }
    }

    func testCancelledTaskCannotStartOrPublishSuccess() async throws {
        for cancelBefore in [true, false] {
            let fixture = try makeFixture()
            defer { try? FileManager.default.removeItem(at: fixture.root) }
            let trainer: ACEStepAdapterTrainingOperation.Trainer = { _, _ in
                XCTAssertFalse(cancelBefore)
                withUnsafeCurrentTask { $0?.cancel() }
                return Self.report()
            }
            let plan = fixture.plan
            let task = Task {
                if cancelBefore { withUnsafeCurrentTask { $0?.cancel() } }
                return try await ACEStepAdapterTrainingOperation.execute(plan, trainer: trainer)
            }
            do {
                _ = try await task.value
                XCTFail("Expected cancellation")
            } catch { XCTAssertTrue(error is CancellationError) }
            XCTAssertEqual(try events(fixture.plan).map(\.type), cancelBefore ? [] : ["run_started", "run_failed"])
        }
    }

    private enum FixtureError: Error { case failed }

    private static func report() -> ACEStepAdapterTrainingReport {
        .init(kind: .lokr, layerCount: 2, trainingSteps: 3, initialLoss: 1, finalLoss: 0.5, outputSHA256: "fixture-digest")
    }

    private func events(_ plan: ACEStepAdapterTrainingPlan) throws -> [LoRATrainingRunEvent] {
        try LoRATrainingRunEvent.load(from: LoRATrainingRunEvent.url(nextTo: plan.outputURL))
    }

    private func makeFixture() throws -> (root: URL, plan: ACEStepAdapterTrainingPlan) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let manifest = root.appendingPathComponent("data.json")
        try #"[{"audio":"one.wav","caption":" caption "}]"#.write(to: manifest, atomically: true, encoding: .utf8)
        let plan = try ACEStepAdapterTrainingPlan.resolve(.init(
            dataset: manifest.path, output: root.appendingPathComponent("output/adapter.safetensors").path,
            model: "/fixture/model", configuration: .init(kind: .lokr, rank: 4, factor: 2, trainingSteps: 3),
            maxDurationSeconds: 0.01
        ))
        return (root, plan)
    }
}
