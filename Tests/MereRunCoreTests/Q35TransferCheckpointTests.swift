import Foundation
import MLX
import XCTest
@testable import MereRunCore

/// Explicit installed-checkpoint qualification; the default suite never loads weights.
final class Q35TransferCheckpointTests: MereRunCoreTestCase {
    func testInstalledLongPromptReplayAndFollowup() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MERERUN_TEST_Q35_TRANSFER"] == "1", "Installed Qwen transfer qualification is opt-in.")
        let root = try XCTUnwrap(env["MERERUN_TEST_Q35_TRANSFER_MODEL_ROOT"])
        let modelID = try XCTUnwrap(env["MERERUN_TEST_Q35_TRANSFER_MODEL_ID"])
        try XCTSkipUnless([Q35Resources.q38TwentySevenB4BitModelId,
                           Q35Resources.ornith35BMLX4BitModelId].contains(modelID), "Select a Q4 transfer target.")
        let generator = Q35Generator(modelId: modelID, prefixKVCacheEnabled: true, continuousBatchingEnabled: false)
        do {
            try await qualify(generator, root: root)
            try await qualifyGenerationCases(generator, root: root)
        } catch {
            await generator.unload()
            throw error
        }
        await generator.unload()
    }

    private func qualify(_ generator: Q35Generator, root: String) async throws {
        let marker = "spruce-lantern-7429"
        let filler = (1...250).map { index in
            "Archive \(index): routine weather observations, equipment checks, and inventory counts were recorded."
        }.joined(separator: "\n")
        let messages = [ChatMessage(role: .user, content:
            "REFERENCE: Project cobalt-harbor has passphrase \(marker).\n\(filler)\n"
            + "What is the passphrase for project cobalt-harbor? Return only the exact passphrase.")]
        let cold = try await generator.chat(request(messages), modelPath: root, progressHandler: nil)
        trace("cold", cold)
        let promptTokens = try XCTUnwrap(cold.promptTokens)
        XCTAssertGreaterThan(promptTokens, 4_096)
        XCTAssertEqual(cold.acceleration?.route, "mtp-speculative")
        XCTAssertEqual(cold.acceleration?.draftHistoryTokens, promptTokens - 1)
        XCTAssertTrue(cold.response.contains(marker))
        let beforeReplay = await generator.prefixKVCacheStats()
        let replay = try await generator.chat(request(messages), modelPath: root, progressHandler: nil)
        trace("replay", replay)
        let afterReplay = await generator.prefixKVCacheStats()
        XCTAssertEqual(afterReplay.reusedTokens - beforeReplay.reusedTokens, promptTokens)
        XCTAssertEqual(replay.response, cold.response)
        XCTAssertEqual(replay.acceleration?.draftHistoryTokens, promptTokens - 1)

        var targetRequest = request(messages)
        targetRequest.logprobCapture = .tokens
        let target = try await generator.chat(targetRequest, modelPath: root, progressHandler: nil)
        trace("serial", target)
        XCTAssertEqual(target.acceleration?.route, "final-target-pipelined")
        XCTAssertEqual(cold.response, target.response, "MTP must preserve serial greedy output")
        XCTAssertEqual(cold.tokensGenerated, target.tokensGenerated)

        let followup = messages + [ChatMessage(role: .assistant, content: cold.response),
                                   ChatMessage(role: .user, content: "Repeat the exact passphrase.")]
        let beforeFollowup = await generator.prefixKVCacheStats()
        let continued = try await generator.chat(request(followup), modelPath: root, progressHandler: nil)
        trace("followup", continued)
        let afterFollowup = await generator.prefixKVCacheStats()
        XCTAssertGreaterThan(afterFollowup.hits, beforeFollowup.hits)
        XCTAssertLessThanOrEqual(afterFollowup.entries, afterFollowup.maxEntries)
        XCTAssertTrue(continued.response.contains(marker))
        XCTAssertEqual(continued.acceleration?.draftHistoryTokens, (continued.promptTokens ?? 0) - 1)

        let original = try await generator.chat(request(messages), modelPath: root, progressHandler: nil)
        trace("original-after-followup", original)
        XCTAssertEqual(original.response, cold.response, "Follow-up must not mutate the original checkpoint")
        XCTAssertEqual(original.acceleration?.draftHistoryTokens, promptTokens - 1)
    }

    private func request(_ messages: [ChatMessage]) -> ChatRequest {
        ChatRequest(messages: messages, maxTokens: 64, temperature: 0, topP: 1,
                    showThinking: false, maxContextTokens: 16_384)
    }

    private func qualifyGenerationCases(_ generator: Q35Generator, root: String) async throws {
        let codePrompt = "Write a complete Python implementation of an LRU cache using a dictionary and a doubly linked list, "
            + "with get and put operations, type hints, and a short usage example. Return the code directly without introductory prose."
        let prosePrompt = "Explain how a small community could turn an abandoned railway station into a public library. "
            + "Discuss the building, accessibility, staffing, the collection, and a realistic opening-day scene. "
            + "Write approximately 400 words of clear, connected prose."
        for (name, prompt) in [("code", codePrompt), ("prose", prosePrompt)] {
            try await qualifyGeneration(generator, root: root, name: name, prompt: prompt)
        }
    }

    private func qualifyGeneration(_ generator: Q35Generator, root: String, name: String, prompt: String) async throws {
        var candidateRequest = ChatRequest(
            messages: [ChatMessage(role: .user, content: prompt)], maxTokens: 256,
            temperature: 0, topP: 1, showThinking: false, stopOnEOS: false, maxContextTokens: 8_192
        )
        let candidate = try await generator.chat(candidateRequest, modelPath: root, progressHandler: nil)
        trace("\(name)-mtp", candidate)
        candidateRequest.logprobCapture = .tokens
        let target = try await generator.chat(candidateRequest, modelPath: root, progressHandler: nil)
        trace("\(name)-serial", target)
        XCTAssertEqual(candidate.acceleration?.route, "mtp-speculative")
        XCTAssertEqual(target.acceleration?.route, "final-target-pipelined")
        XCTAssertEqual(candidate.tokensGenerated, 256)
        XCTAssertEqual(candidate.tokensGenerated, target.tokensGenerated)
        XCTAssertEqual(candidate.response, target.response, "Speculation and its fallback must preserve serial output for \(name)")
        XCTAssertEqual(candidate.reasoningContent, target.reasoningContent)
    }

    private func trace(_ label: String, _ response: ChatResponse) {
        struct Record: Encodable {
            let label: String
            let promptTokens: Int?
            let tokens: Int
            let prefillSeconds: Double?
            let decodeSeconds: Double?
            let acceleration: ChatAccelerationDiagnostics?
            let output: String
            let activeMemory: Int
        }
        let record = Record(label: label, promptTokens: response.promptTokens, tokens: response.tokensGenerated,
                            prefillSeconds: response.timing?.prefillSeconds, decodeSeconds: response.timing?.decodeSeconds,
                            acceleration: response.acceleration, output: response.response, activeMemory: Memory.activeMemory)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        if let data = try? encoder.encode(record) {
            FileHandle.standardError.write(Data("[q35-transfer] ".utf8) + data + Data("\n".utf8))
        }
    }
}
