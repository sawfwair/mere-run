import Foundation
import MereRunAdmission
import XCTest
@testable import MereRunCore

/// The upstream process is a benchmark reference, never a product inference path.
final class Bonsai2SmallThroughputTests: MereRunCoreTestCase {
    private struct Input: Decodable {
        let modality: String
        let prompt: String
        let prompt_ids: [Int]
        let image_tokens: Int
    }

    private struct Record: Decodable {
        let generated_tokens: Int
        let decode_tokens_per_second: Double
    }

    private struct Report: Decodable {
        let inputs: [Input]
        let records: [Record]
    }

    func testInstalledUpstreamThroughput() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu",
              let path = env["MERERUN_BONSAI_THROUGHPUT_ROOT"],
              let size = env["MERERUN_BONSAI_THROUGHPUT_SIZE"],
              let python = env["MERERUN_BONSAI_REFERENCE_PYTHON"] else {
            throw XCTSkip("Set the Bonsai throughput root, size, reference Python, and GPU opt-in variables.")
        }
        XCTAssertTrue(["2", "4", "9"].contains(size))
        let root = URL(fileURLWithPath: path)
        let coordinator = MachineInferenceCoordinator(stateDirectory:
            MereRunModelPaths.applicationSupportBase.appendingPathComponent("admission", isDirectory: true))
        let lease = try await coordinator.acquire(.init(
            label: "Bonsai \(size)B upstream throughput", resourceClass: .standard))
        defer { lease.release() }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(coordinator.snapshot()).write(to: root.appending(path: "upstream-\(size)-admission.json"))
        let output = root.appending(path: "upstream-\(size)")
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = ["-B", repo.appending(path: "scripts/reference-parity/benchmark_bonsai2_small.py").path,
                             "--root", root.path, "--size", size, "--output", output.path,
                             "--admission-state", MereRunModelPaths.applicationSupportBase
                                .appendingPathComponent("admission/state.json").path]
        var environment = env
        environment["HF_HUB_OFFLINE"] = "1"
        environment["HF_HOME"] = root.appending(path: "hub-cache").path
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        process.environment = environment
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)
        let report = try JSONDecoder().decode(Report.self, from: Data(contentsOf: output.appending(path: "results.json")))
        XCTAssertEqual(report.records.count, 8)
        for record in report.records {
            XCTAssertGreaterThan(record.generated_tokens, 0)
            XCTAssertTrue(record.decode_tokens_per_second.isFinite)
            XCTAssertGreaterThan(record.decode_tokens_per_second, 0)
        }
        let checkpoint = root.appending(path: "models/text-chat-bonsai-2-\(size)b-2bit")
        let template = try Q35TokenizerAndTemplate.load(from: checkpoint, maxLengthOverride: 8192)
        XCTAssertEqual(report.inputs.count, 2)
        for input in report.inputs {
            let image = input.modality == "vision" ? root.appending(path: "fixtures/shapes.png").path : nil
            let ids = try template.encodeForGeneration(
                messages: [ChatMessage(role: .user, content: input.prompt, imageUrl: image)],
                includeThinking: true, maxLength: 8192,
                imageTokenCounts: image == nil ? [] : [input.image_tokens])
            XCTAssertEqual(ids, input.prompt_ids, "\(size)B \(input.modality) benchmark input tokens")
        }
    }
}
