import Foundation
import XCTest
@testable import MereRunCore

private final class LockedGeneratedText: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = ""

    func append(_ text: String) {
        lock.lock()
        storage += text
        lock.unlock()
    }

    var value: String {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

final class GuardedAccelerationCheckpointTests: MereRunCoreTestCase {
    func testInstalledLFM2AffineEightGreedyParity() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_TEST_GUARDED_ACCELERATION_CHECKPOINTS"] == "1" else {
            throw XCTSkip("Set MERERUN_TEST_GUARDED_ACCELERATION_CHECKPOINTS=1 to run real-checkpoint parity.")
        }
        guard let modelRoot = ManagedModelResolver.resolveInstalledModel(id: LFM2Resources.defaultModelId) else {
            throw XCTSkip("Install \(LFM2Resources.defaultModelId) to run affine KV parity.")
        }

        let generator = LFM2Generator(prefixKVCacheEnabled: false)
        let passage = """
        Local inference performance depends on memory bandwidth, cache layout,
        kernel launch count, and exact output verification. A valid optimization
        must preserve the target model's answer while reducing measured cost.
        """
        let prompt = Array(repeating: passage, count: 12).joined(separator: "\n")
            + "\nSummarize the passage in one sentence."
        let messages = [ChatMessage(role: .user, content: prompt)]

        func request(_ mode: RuntimeKVCacheMode) -> ChatRequest {
            ChatRequest(
                messages: messages,
                maxTokens: 24,
                temperature: 0,
                topP: 1,
                showThinking: false,
                kvCacheMode: mode
            )
        }

        let baselineWarmup = try await generator.chat(
            request(.default),
            modelPath: modelRoot.path,
            progressHandler: nil
        )
        let affineWarmup = try await generator.chat(
            request(.affine8),
            modelPath: modelRoot.path,
            progressHandler: nil
        )
        let baseline = try await generator.chat(
            request(.default),
            modelPath: modelRoot.path,
            progressHandler: nil
        )
        let affine = try await generator.chat(
            request(.affine8),
            modelPath: modelRoot.path,
            progressHandler: nil
        )
        await generator.unload()

        XCTAssertEqual(affineWarmup.response, baselineWarmup.response)
        XCTAssertEqual(affine.response, baseline.response)
        XCTAssertEqual(affine.tokensGenerated, baseline.tokensGenerated)
        XCTAssertEqual(affine.timing?.kvCacheMode, .affine8)
        print(
            "[guarded-acceleration] lfm2 prompt_tokens=\(baseline.promptTokens ?? 0) "
                + "tokens=\(baseline.tokensGenerated) "
                + "baseline_prefill_s=\(baseline.timing?.prefillSeconds ?? 0) "
                + "baseline_decode_s=\(baseline.timing?.decodeSeconds ?? 0) "
                + "affine8_prefill_s=\(affine.timing?.prefillSeconds ?? 0) "
                + "affine8_decode_s=\(affine.timing?.decodeSeconds ?? 0)"
        )
    }

    func testInstalledQ36AffineEightGreedyParity() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_TEST_GUARDED_ACCELERATION_CHECKPOINTS"] == "1" else {
            throw XCTSkip("Set MERERUN_TEST_GUARDED_ACCELERATION_CHECKPOINTS=1 to run real-checkpoint parity.")
        }
        guard let modelRoot = ManagedModelResolver.resolveInstalledModel(id: Q35Resources.defaultModelId) else {
            throw XCTSkip("Install \(Q35Resources.defaultModelId) to run affine KV parity.")
        }

        let generator = Q35Generator(
            prefixKVCacheEnabled: false,
            continuousBatchingEnabled: false
        )
        let passage = """
        Local inference performance depends on memory bandwidth, cache layout,
        kernel launch count, and exact output verification. A valid optimization
        must preserve the target model's answer while reducing measured cost.
        """
        let prompt = Array(repeating: passage, count: 12).joined(separator: "\n")
            + "\nSummarize the passage in one sentence."
        let messages = [ChatMessage(role: .user, content: prompt)]

        func request(_ mode: RuntimeKVCacheMode) -> ChatRequest {
            ChatRequest(
                messages: messages,
                maxTokens: 24,
                temperature: 0,
                topP: 1,
                showThinking: false,
                kvCacheMode: mode
            )
        }

        let baselineWarmup = try await generator.chat(
            request(.default),
            modelPath: modelRoot.path,
            progressHandler: nil
        )
        let affineWarmup = try await generator.chat(
            request(.affine8),
            modelPath: modelRoot.path,
            progressHandler: nil
        )
        let baseline = try await generator.chat(
            request(.default),
            modelPath: modelRoot.path,
            progressHandler: nil
        )
        let affine = try await generator.chat(
            request(.affine8),
            modelPath: modelRoot.path,
            progressHandler: nil
        )
        await generator.unload()

        XCTAssertEqual(affineWarmup.response, baselineWarmup.response)
        XCTAssertEqual(affine.response, baseline.response)
        XCTAssertEqual(affine.tokensGenerated, baseline.tokensGenerated)
        XCTAssertEqual(affine.timing?.kvCacheMode, .affine8)
        print(
            "[guarded-acceleration] q36 prompt_tokens=\(baseline.promptTokens ?? 0) "
                + "tokens=\(baseline.tokensGenerated) "
                + "baseline_prefill_s=\(baseline.timing?.prefillSeconds ?? 0) "
                + "baseline_decode_s=\(baseline.timing?.decodeSeconds ?? 0) "
                + "affine8_prefill_s=\(affine.timing?.prefillSeconds ?? 0) "
                + "affine8_decode_s=\(affine.timing?.decodeSeconds ?? 0)"
        )
    }

    func testInstalledQ36JSONModeRepeatedOutputsParse() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_TEST_Q36_JSON_MODE"] == "1" else {
            throw XCTSkip("Set MERERUN_TEST_Q36_JSON_MODE=1 to run real Q36 JSON-mode generation.")
        }
        guard let modelRoot = ManagedModelResolver.resolveInstalledModel(id: Q35Resources.q36NanoModelId) else {
            throw XCTSkip("Install \(Q35Resources.q36NanoModelId) to run real JSON-mode generation.")
        }

        let generator = Q35Generator(
            prefixKVCacheEnabled: false,
            continuousBatchingEnabled: true
        )
        defer {
            Task { await generator.unload() }
        }
        let prompts = [
            "Return only a JSON object with keys name, count, and enabled. Use name café, count 3, enabled true.",
            "Return only a JSON object describing two ocean items as a nested array. Include Unicode text.",
            "Return only a JSON object with an escaped quote in a message string and a nested metadata object.",
        ]

        for (index, prompt) in prompts.enumerated() {
            let streamed = LockedGeneratedText()
            let response = try await generator.chat(
                ChatRequest(
                    messages: [ChatMessage(role: .user, content: prompt)],
                    maxTokens: 192,
                    temperature: 0,
                    topP: 1,
                    showThinking: true,
                    requiresJSON: true
                ),
                modelPath: modelRoot.path,
                progressHandler: { progress in
                    guard progress.stage == .generating,
                          let piece = progress.message,
                          !piece.isEmpty else { return }
                    streamed.append(piece)
                }
            )

            let output = response.response.trimmingCharacters(in: .whitespacesAndNewlines)
            XCTAssertTrue(output.hasPrefix("{"), "run \(index) emitted prose before JSON: \(output)")
            XCTAssertTrue(output.hasSuffix("}"), "run \(index) did not close the root object: \(output)")
            XCTAssertFalse(output.contains("```"), "run \(index) emitted a markdown fence")
            XCTAssertNil(response.reasoningContent, "run \(index) emitted thinking in JSON mode")
            XCTAssertNoThrow(try JSONSerialization.jsonObject(with: Data(output.utf8)))
            XCTAssertEqual(
                streamed.value.trimmingCharacters(in: .whitespacesAndNewlines),
                output,
                "run \(index) streamed different JSON than the non-streaming response"
            )
        }

        let stats = await generator.continuousBatchingStats()
        XCTAssertEqual(stats.batchedDecodeSteps, 0)
        XCTAssertEqual(stats.singleDecodeSteps, 0)
    }
}
