import Foundation
import XCTest
@testable import MereRunCore

/// Opt-in checkpoint loading and media orchestration smoke; synthetic weights do not qualify model accuracy.
final class D1CheckpointSmokeTests: MereRunCoreTestCase {
    func testCheckpointTextImageAndAudioPaths() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let causal = environment["MERERUN_TEST_D1_CAUSAL_CHECKPOINT"],
              let omni = environment["MERERUN_TEST_D1_OMNI_CHECKPOINT"],
              let requests = environment["MERERUN_TEST_D1_SMOKE_REQUESTS"] else {
            throw XCTSkip("Set D1 checkpoint and smoke-request directories to exercise complete native loading.")
        }
        for (root, media) in [(causal, "text"), (omni, "text"), (causal, "image"), (omni, "image"), (omni, "audio")] {
            let operation = try D1DecisionOperation(root: URL(fileURLWithPath: root), modelID: "smoke")
            defer { operation.unload() }
            let data = try Data(contentsOf: URL(fileURLWithPath: requests).appending(path: "d1-smoke-\(media).json"))
            let request = try D1DecisionRequest.decode(data)
            let plan = try operation.prepare(request)
            let result = try operation.predict(request)
            XCTAssertEqual(result.usage.input_tokens, plan.inputTokens)
            XCTAssertEqual(result.usage.output_tokens, 0)
            XCTAssertEqual(result.answers.count, request.questions.count)
            XCTAssertNotNil(result.answers["route"]?.choice)
            XCTAssertNotNil(result.answers["arrived"]?.noul)
            XCTAssertNotNil(result.answers["quality"]?.score)
            if media != "text" { XCTAssertGreaterThan(plan.mediaTokens, 0) }
        }
    }
}
