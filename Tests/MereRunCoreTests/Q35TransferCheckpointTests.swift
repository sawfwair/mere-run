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

    func testInstalledExtendedContextReplay() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MERERUN_TEST_Q35_TRANSFER"] == "1", "Installed Qwen transfer qualification is opt-in.")
        let root = try XCTUnwrap(env["MERERUN_TEST_Q35_TRANSFER_MODEL_ROOT"])
        try XCTSkipUnless(env["MERERUN_TEST_Q35_TRANSFER_MODEL_ID"] == Q35Resources.ornith35BMLX4BitModelId,
                          "This qualification selects the Ornith Q4 head.")
        let generator = Q35Generator(modelId: Q35Resources.ornith35BMLX4BitModelId,
                                     prefixKVCacheEnabled: true, continuousBatchingEnabled: false)
        do {
            try await qualify(generator, root: root, archiveRows: 1_700, minimumPromptTokens: 32_768)
        } catch {
            await generator.unload()
            throw error
        }
        await generator.unload()
    }

    func testInstalledToolsSamplingAndThinking() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MERERUN_TEST_Q35_TRANSFER"] == "1", "Installed Qwen transfer qualification is opt-in.")
        let root = try XCTUnwrap(env["MERERUN_TEST_Q35_TRANSFER_MODEL_ROOT"])
        try XCTSkipUnless(env["MERERUN_TEST_Q35_TRANSFER_MODEL_ID"] == Q35Resources.ornith35BMLX4BitModelId,
                          "This qualification selects the Ornith Q4 head.")
        let generator = Q35Generator(modelId: Q35Resources.ornith35BMLX4BitModelId,
                                     prefixKVCacheEnabled: true, continuousBatchingEnabled: false)
        do {
            try await qualifyTools(generator, root: root)
            try await qualifySampling(generator, root: root)
            var thinkingRequest = request([ChatMessage(role: .user, content:
                "What is 17 plus 25? Think briefly, then answer with the number.")])
            thinkingRequest.maxTokens = 256
            thinkingRequest.showThinking = true
            let candidate = try await generator.chat(thinkingRequest, modelPath: root, progressHandler: nil)
            trace("thinking-mtp", candidate)
            thinkingRequest.logprobCapture = .tokens
            let target = try await generator.chat(thinkingRequest, modelPath: root, progressHandler: nil)
            trace("thinking-serial", target)
            XCTAssertEqual(candidate.acceleration?.route, "mtp-speculative")
            XCTAssertEqual(target.acceleration?.route, "final-target-pipelined")
            XCTAssertTrue(candidate.response.contains("42"))
            XCTAssertFalse(candidate.reasoningContent?.isEmpty ?? true)
            XCTAssertEqual(candidate.response, target.response)
            XCTAssertEqual(candidate.reasoningContent, target.reasoningContent)
            XCTAssertEqual(candidate.tokensGenerated, target.tokensGenerated)
        } catch {
            await generator.unload()
            throw error
        }
        await generator.unload()
    }

    private func qualify(
        _ generator: Q35Generator, root: String, archiveRows: Int = 250, minimumPromptTokens: Int = 4_096
    ) async throws {
        let marker = "spruce-lantern-7429"
        let filler = (1...archiveRows).map { index in
            "Archive \(index): routine weather observations, equipment checks, and inventory counts were recorded."
        }.joined(separator: "\n")
        let messages = [ChatMessage(role: .user, content:
            "REFERENCE: Project cobalt-harbor has passphrase \(marker).\n\(filler)\n"
            + "What is the passphrase for project cobalt-harbor? Return only the exact passphrase.")]
        let cold = try await generator.chat(request(messages), modelPath: root, progressHandler: nil)
        trace("cold", cold)
        let promptTokens = try XCTUnwrap(cold.promptTokens)
        XCTAssertGreaterThan(promptTokens, minimumPromptTokens)
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
                    showThinking: false, maxContextTokens: 65_536)
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

    private func qualifyTools(_ generator: Q35Generator, root: String) async throws {
        let tool = ToolDefinition(name: "lookup_project", description: "Look up a project passphrase.",
                                  parameters: ["project": ToolParameterProperty(type: "string", description: "Project name")])
        var toolRequest = request([ChatMessage(role: .user, content:
            "Use lookup_project to get the passphrase for cobalt-harbor. Do not guess the passphrase.")])
        toolRequest.maxTokens = 128
        toolRequest.tools = [tool]
        toolRequest.toolChoice = .function(tool.name)
        let candidate = try await generator.chat(toolRequest, modelPath: root, progressHandler: nil)
        trace("tool-default", candidate)
        let expected = [ToolCall(name: tool.name, arguments: ["project": "cobalt-harbor"])]
        XCTAssertEqual(candidate.toolCalls, expected)
        // Tool requests intentionally bypass MTP; changing the installed head must preserve this route.
        XCTAssertEqual(candidate.acceleration?.route, "final-target-pipelined")
        XCTAssertNil(candidate.acceleration?.acceptedDraftTokens)
        toolRequest.logprobCapture = .tokens
        let target = try await generator.chat(toolRequest, modelPath: root, progressHandler: nil)
        trace("tool-serial", target)
        XCTAssertEqual(target.acceleration?.route, "final-target-pipelined")
        XCTAssertEqual(candidate.toolCalls, target.toolCalls)
        XCTAssertEqual(candidate.response, target.response)
        XCTAssertEqual(candidate.tokensGenerated, target.tokensGenerated)
        toolRequest.messages += [
            ChatMessage(role: .assistant, content: candidate.response, toolCalls: [
                ChatMessageToolCall(id: "call_qualification", name: tool.name,
                                    arguments: ["project": .string("cobalt-harbor")]),
            ]),
            ChatMessage(role: .tool, content: "{\"passphrase\":\"spruce-lantern-7429\"}", name: tool.name,
                        toolCallID: "call_qualification"),
            ChatMessage(role: .user, content: "Return only the exact passphrase from the tool result."),
        ]
        toolRequest.toolChoice = .auto
        toolRequest.logprobCapture = .none
        let before = await generator.prefixKVCacheStats()
        let followup = try await generator.chat(toolRequest, modelPath: root, progressHandler: nil)
        trace("tool-followup-default", followup)
        let after = await generator.prefixKVCacheStats()
        XCTAssertGreaterThan(after.hits, before.hits)
        XCTAssertTrue(followup.response.contains("spruce-lantern-7429"))
        XCTAssertEqual(followup.acceleration?.route, "final-target-pipelined")
        toolRequest.logprobCapture = .tokens
        let serialFollowup = try await generator.chat(toolRequest, modelPath: root, progressHandler: nil)
        trace("tool-followup-serial", serialFollowup)
        XCTAssertEqual(followup.response, serialFollowup.response)
        XCTAssertEqual(followup.toolCalls, serialFollowup.toolCalls)
    }

    private func qualifySampling(_ generator: Q35Generator, root: String) async throws {
        for seed: UInt64 in [7, 42, 2_026] {
            var sampledRequest = ChatRequest(
                messages: [ChatMessage(role: .user, content:
                    "Describe how volunteers could turn an abandoned railway station into a public library.")],
                maxTokens: 96, temperature: 0.7, topP: 0.9, topK: 20, minP: 0.05,
                seed: seed, showThinking: false, stopOnEOS: false, maxContextTokens: 8_192
            )
            let candidate = try await generator.chat(sampledRequest, modelPath: root, progressHandler: nil)
            trace("sample-\(seed)-mtp", candidate)
            XCTAssertEqual(candidate.acceleration?.route, "mtp-speculative")
            XCTAssertGreaterThan(candidate.acceleration?.draftedTokens ?? 0, 0)
            XCTAssertGreaterThanOrEqual(candidate.acceleration?.acceptanceRate ?? -1, 0)
            XCTAssertLessThanOrEqual(candidate.acceleration?.acceptanceRate ?? 2, 1)
            if seed == 7 {
                let replay = try await generator.chat(sampledRequest, modelPath: root, progressHandler: nil)
                trace("sample-\(seed)-mtp-replay", replay)
                XCTAssertEqual(candidate.response, replay.response, "The same request seed must replay after a cache hit")
                XCTAssertEqual(candidate.tokensGenerated, replay.tokensGenerated)
            }
            sampledRequest.logprobCapture = .tokens
            let target = try await generator.chat(sampledRequest, modelPath: root, progressHandler: nil)
            trace("sample-\(seed)-serial", target)
            XCTAssertEqual(target.acceleration?.route, "final-target-pipelined")
            // Rejection sampling consumes different random draws; same-seed text equality is not expected.
            for response in [candidate, target] {
                XCTAssertEqual(response.tokensGenerated, 96)
                XCTAssertFalse(response.response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                XCTAssertFalse(response.response.contains("\u{FFFD}"))
                XCTAssertTrue(response.timing?.decodeSeconds.isFinite == true)
            }
        }
    }

    private func trace(_ label: String, _ response: ChatResponse) {
        struct ToolRecord: Encodable {
            let name: String
            let arguments: [String: String]
        }
        struct Record: Encodable {
            let label: String
            let promptTokens: Int?
            let tokens: Int
            let prefillSeconds: Double?
            let decodeSeconds: Double?
            let acceleration: ChatAccelerationDiagnostics?
            let output: String
            let reasoning: String?
            let toolCalls: [ToolRecord]?
            let activeMemory: Int
        }
        let record = Record(label: label, promptTokens: response.promptTokens, tokens: response.tokensGenerated,
                            prefillSeconds: response.timing?.prefillSeconds, decodeSeconds: response.timing?.decodeSeconds,
                            acceleration: response.acceleration, output: response.response,
                            reasoning: response.reasoningContent,
                            toolCalls: response.toolCalls?.map { ToolRecord(name: $0.name, arguments: $0.arguments) },
                            activeMemory: Memory.activeMemory)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        if let data = try? encoder.encode(record) {
            FileHandle.standardError.write(Data("[q35-transfer] ".utf8) + data + Data("\n".utf8))
        }
    }
}
