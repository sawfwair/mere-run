import Foundation
import MLX
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
    func testInstalledGemmaTwelveBTerminalPrefillRowAB() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_TEST_GUARDED_ACCELERATION_CHECKPOINTS"] == "1" else {
            throw XCTSkip("Set MERERUN_TEST_GUARDED_ACCELERATION_CHECKPOINTS=1 to run real-checkpoint parity.")
        }
        let modelId = Gemma4Resources.twelveBModelId
        guard let modelRoot = ManagedModelResolver.resolveInstalledModel(id: modelId) else {
            throw XCTSkip("Install \(modelId) to run terminal-prefill parity.")
        }

        let loaded = try await Gemma4TextModelLoader.load(
            modelId: modelId,
            modelPath: modelRoot.path,
            maxContextLength: 512
        )
        let passage = """
        Local inference performance depends on memory bandwidth, cache layout,
        kernel launch count, and exact output verification. A valid optimization
        must preserve the target model's answer while reducing measured cost.
        """
        let prompt = Array(repeating: passage, count: 12).joined(separator: "\n")
            + "\nReturn exactly one word that summarizes the passage."
        let tokens = try loaded.tokenizerAndTemplate.encodeForGeneration(
            messages: [ChatMessage(role: .user, content: prompt)],
            addGenerationPrompt: true,
            includeThinking: false,
            maxLength: 512
        )
        let input = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)

        struct Measurement {
            let seconds: Double
            let token: Int
            let logits: MLXArray
        }
        func measure(_ enabled: Bool) -> Measurement {
            let cache = loaded.model.languageModel.makeCache()
            let started = Date()
            let logits = loaded.model.languageModel.lastPositionLogits(
                input,
                cache: cache,
                terminalPrefillRowEnabled: enabled
            )
            MLX.eval(logits)
            let seconds = Date().timeIntervalSince(started)
            let token = argMax(logits[0, -1, 0...]).item(Int.self)
            return Measurement(seconds: seconds, token: token, logits: logits)
        }

        let measurements = [false, true, false, true, true, false].map(measure)
        let reference = measurements[2]
        for measurement in measurements {
            XCTAssertEqual(measurement.token, reference.token)
        }
        let maximumDifference = measurements.map { measurement in
            MLX.max(
                MLX.abs(
                    measurement.logits.asType(.float32)
                        - reference.logits.asType(.float32)
                )
            ).item(Float.self)
        }.max() ?? 0
        // The single-query SDPA path may choose a different reduction kernel
        // from the full-query graph. Keep the observed numerical class bounded
        // while requiring the consumed greedy token to remain identical.
        XCTAssertLessThanOrEqual(maximumDifference, 0.5)

        let baseline = [measurements[2].seconds, measurements[5].seconds]
        let candidate = [measurements[3].seconds, measurements[4].seconds]
        let baselineMean = baseline.reduce(0, +) / Double(baseline.count)
        let candidateMean = candidate.reduce(0, +) / Double(candidate.count)
        print(
            "[guarded-acceleration] gemma_terminal_prefill model=\(modelId) "
                + "prompt_tokens=\(tokens.count) baseline_s=\(baseline) "
                + "candidate_s=\(candidate) baseline_mean_s=\(baselineMean) "
                + "candidate_mean_s=\(candidateMean) ratio=\(baselineMean / candidateMean) "
                + "max_abs_diff=\(maximumDifference)"
        )
    }

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
