import ArgumentParser
import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class TextTrainingCompatibilityTests: XCTestCase {
    func testDashboardStopWaitsForServerCleanup() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let logger = try LoRATrainingEventLogger(baseOutputURL: root.appendingPathComponent("adapter.safetensors"))
        let cleanup = TrainingViewerCleanup()
        let server = Task {
            do { try await Task.sleep(for: .seconds(30)) } catch { }
            await cleanup.finish()
        }
        let visualization = TextLoRATrainingVisualization(logger: logger, serverTask: server)
        await visualization.stop()
        let finished = await cleanup.finished
        XCTAssertTrue(finished)
    }

    func testTrainingKeepsProcessAdmissionAndDryRunBypass() {
        let arguments = ["mere.run", "text", "train-lora", "--data", "pairs.jsonl", "--output", "adapter.safetensors"]
        XCTAssertEqual(CLIInferenceAdmissionClassifier.request(arguments: arguments)?.resourceClass, .large)
        XCTAssertNil(CLIInferenceAdmissionClassifier.request(arguments: arguments + ["--dry-run"]))
    }

    func testInvalidOptionsFailBeforeReadingData() async throws {
        let cases: [([String], String)] = [
            (["--steps", "0"], "--training-steps must be >= 1"),
            (["--batch-size", "0"], "--batch-size must be >= 1"),
            (["--lr", "0"], "--learning-rate must be > 0"),
            (["--rank", "0"], "--rank must be >= 1"),
            (["--alpha", "0"], "--alpha must be > 0"),
            (["--max-sequence-length", "127"], "--max-sequence-length must be >= 128"),
            (["--reasoning-effort", "1"], "--reasoning-effort must be between 0 and 0.99"),
            (["--target-modules", " , "], "--target-modules must include at least one target suffix"),
            (["--resume-step", "1"], "--resume-step requires --resume-from"),
            (["--resume-from", "/missing.safetensors"], "--resume-from cannot be combined with --dry-run"),
            (["--visualize"], "--visualize cannot be combined with --dry-run"),
        ]
        for (options, message) in cases {
            let command = try TextTrainLoRA.parse([
                "--data", "/missing.jsonl", "--output", "/missing.safetensors", "--dry-run",
            ] + options)
            do {
                try await command.run()
                XCTFail("Expected validation: \(message)")
            } catch {
                XCTAssertTrue(error is ValidationError)
                XCTAssertEqual(String(describing: error), message)
            }
        }
    }

    func testDryRunPreservesAdapterAndWritesExistingManifestContract() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let data = root.appendingPathComponent("pairs.jsonl")
        try Data(#"{"sources":["test"],"messages":[{"role":"system","content":"Be helpful"},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hello there"}]}"#.utf8).write(to: data)
        let output = root.appendingPathComponent("adapter.safetensors")
        let original = Data("existing adapter".utf8)
        try original.write(to: output)
        let command = try TextTrainLoRA.parse([
            "--data", data.path, "--eval", data.path, "--output", output.path,
            "--rank", "8", "--target-modules", " q_proj, ,v_proj,q_proj ", "--dry-run", "--json",
        ])
        try await command.run()
        XCTAssertEqual(try Data(contentsOf: output), original)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let manifest = try decoder.decode(TextLoRATrainingManifest.self, from: Data(contentsOf: TextLoRATrainingManifest.url(nextTo: output)))
        XCTAssertEqual(manifest.status, "prepared")
        XCTAssertEqual(manifest.baseModel, Gemma4Resources.twelveB4BitModelId)
        XCTAssertEqual(manifest.training.seed, 42)
        XCTAssertEqual(manifest.training.learningRate, 0.0001)
        XCTAssertEqual(manifest.lora.alpha, 8)
        XCTAssertEqual(manifest.lora.targetModules, ["q_proj", "v_proj", "q_proj"])
        XCTAssertEqual(manifest.evalPromptCount, 1)
    }

    func testEmptyDatasetFailsWithoutCreatingOutputDirectory() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let data = root.appendingPathComponent("empty.jsonl")
        try Data().write(to: data)
        let outputDirectory = root.appendingPathComponent("output")
        let command = try TextTrainLoRA.parse([
            "--data", data.path, "--output", outputDirectory.appendingPathComponent("adapter.safetensors").path, "--dry-run",
        ])
        do {
            try await command.run()
            XCTFail("Expected an empty dataset error")
        } catch {
            XCTAssertTrue(error is TextSFTDatasetError)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: outputDirectory.path))
    }
}

private actor TrainingViewerCleanup {
    var finished = false
    func finish() { finished = true }
}
