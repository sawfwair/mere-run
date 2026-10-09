import Foundation
import XCTest
import MLX
import MereRunAdmission
@testable import MereRunCore

/// Eight synthetic checkpoint probes; this is not a general capability benchmark.
final class ClefOmniCheckpointParityTests: MereRunCoreTestCase {
    private struct Probe: Decodable {
        let token_ids: [Int]
        let result: Result
        struct Result: Decodable {
            let answers: [String: Answer]
            struct Answer: Decodable {
                let choice: String?
                let noul: Double?
                let score: Double?
                let probabilities: [String: Double]?
            }
        }
    }
    private struct Receipt: Encodable {
        let physicalMemoryBytes: UInt64
        let peakMLXBytes: Int
        let maximumProbabilityDelta: Double
        let elapsedSeconds: Double
        let cases: [String: ClefDecisionResponse]
    }

    func testRealCheckpointTokenizationWhenProvided() throws {
        let env = ProcessInfo.processInfo.environment
        guard let model = env["MERERUN_TEST_CLEF_OMNI_ROOT"],
              let probes = env["MERERUN_TEST_CLEF_OMNI_PARITY_DIR"] else {
            throw XCTSkip("Set MERERUN_TEST_CLEF_OMNI_ROOT and MERERUN_TEST_CLEF_OMNI_PARITY_DIR.")
        }
        let root = URL(fileURLWithPath: model), directory = URL(fileURLWithPath: probes)
        let reference = try JSONDecoder().decode([String: Probe].self,
            from: Data(contentsOf: directory.appending(path: "reference-bf16.json")))
        let tokenizer = try ClefTokenizer.load(root: root)
        for (name, target) in reference {
            let data = try Data(contentsOf: directory.appending(path: "\(name).request.json"))
            let request = try ClefDecisionRequest.decode(data, omni: true)
            let media = try ClefOmniMedia.prepare(request)
            let sequence = try tokenizer.sequence(request, modelID: "clef-omni-mlx-q4",
                mediaIDs: media.blocks.flatMap { tokenizer.encode($0.text) })
            XCTAssertEqual(sequence.ids, target.token_ids, name)
        }
    }

    func testRealQ4CheckpointOnGPUWhenProvided() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu",
              let model = env["MERERUN_TEST_CLEF_OMNI_ROOT"],
              let probes = env["MERERUN_TEST_CLEF_OMNI_PARITY_DIR"] else {
            throw XCTSkip("Set MERERUN_TEST_MLX_DEVICE=gpu, MERERUN_TEST_CLEF_OMNI_ROOT, and MERERUN_TEST_CLEF_OMNI_PARITY_DIR.")
        }
        let root = URL(fileURLWithPath: model), directory = URL(fileURLWithPath: probes)
        let reference = try JSONDecoder().decode([String: Probe].self,
            from: Data(contentsOf: directory.appending(path: "reference-bf16.json")))
        let coordinator = MachineInferenceCoordinator(stateDirectory:
            MereRunModelPaths.applicationSupportBase.appendingPathComponent("admission", isDirectory: true))
        let lease = try await coordinator.acquire(.init(label: "Clef Omni Q4 qualification", resourceClass: .standard))
        defer { lease.release() }
        XCTAssertTrue(ClefCatalog.validate(root: root).isEmpty)
        let operation = try ClefDecisionOperation(root: root, modelID: "clef-omni-mlx-q4")
        defer { operation.unload() }
        Memory.peakMemory = 0
        let started = Date()
        var outputs: [String: ClefDecisionResponse] = [:]
        var maximumDelta = 0.0
        for name in ["route", "json", "multilingual", "score", "image", "audio", "video", "mixed"] {
            let target = try XCTUnwrap(reference[name])
            let data = try Data(contentsOf: directory.appending(path: "\(name).request.json"))
            let request = try ClefDecisionRequest.decode(data, omni: true)
            let media = try ClefOmniMedia.prepare(request)
            let tokenizer = try ClefTokenizer.load(root: root)
            let sequence = try tokenizer.sequence(request, modelID: "clef-omni-mlx-q4",
                mediaIDs: media.blocks.flatMap { tokenizer.encode($0.text) })
            XCTAssertEqual(sequence.ids, target.token_ids, name)
            let response = try operation.predict(request)
            outputs[name] = response
            for (field, expected) in target.result.answers {
                let actual = try XCTUnwrap(response.answers[field])
                XCTAssertEqual(actual.choice, expected.choice, "\(name)/\(field)")
                if let value = expected.noul {
                    maximumDelta = max(maximumDelta, abs(try XCTUnwrap(actual.noul) - value))
                }
                if let value = expected.score {
                    XCTAssertEqual(try XCTUnwrap(actual.score), value, accuracy: 0.1, "\(name)/\(field)")
                }
                for (option, value) in expected.probabilities ?? [:] {
                    let probability = try XCTUnwrap(actual.probabilities?[option])
                    XCTAssertTrue(probability.isFinite)
                    maximumDelta = max(maximumDelta, abs(probability - value))
                }
            }
            let receipt = Receipt(physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                peakMLXBytes: Memory.peakMemory, maximumProbabilityDelta: maximumDelta,
                elapsedSeconds: Date().timeIntervalSince(started), cases: outputs)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(receipt).write(to: directory.appending(path: "native-q4.json"), options: .atomic)
        }
        XCTAssertLessThanOrEqual(maximumDelta, 0.05, "Bounded diagnostic probability tolerance.")
        XCTAssertLessThan(Memory.peakMemory, 30_000_000_000, "Initial 36 GB qualification memory ceiling.")
    }
}
