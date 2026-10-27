import Foundation
import MLX
import MereRunAdmission
import XCTest
@testable import MereRunCore

/// Opt-in checks exercise the released BF16 checkpoint through the production loader.
final class LFM25A2BInstalledQualificationTests: MereRunCoreTestCase {
    private func checkpointRoot() throws -> String {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_LFM25_A2B_BF16_ROOT"] else {
            throw XCTSkip("Set MERERUN_LFM25_A2B_BF16_ROOT to the original pinned 24B-A2B checkpoint.")
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Set MERERUN_TEST_MLX_DEVICE=gpu for released-checkpoint qualification.")
        }
        let root = URL(fileURLWithPath: path)
        let config = try JSONDecoder().decode(
            LFM2Config.self, from: Data(contentsOf: root.appending(path: "config.json"))
        )
        XCTAssertTrue(config.isA2B24B)
        XCTAssertNil(config.quantization, "Qualification requires the original BF16 weights.")
        return root.path
    }

    private func acquireLease() async throws -> MachineInferenceLease {
        let coordinator = MachineInferenceCoordinator(
            stateDirectory: MereRunModelPaths.applicationSupportBase.appending(path: "admission"),
            hostSnapshot: {
                MachineInferenceCoordinator.currentHostSnapshot(
                    diskURL: MereRunModelPaths.applicationSupportBase.deletingLastPathComponent()
                )
            }
        )
        return try await coordinator.acquire(.init(label: "LFM2.5 24B-A2B BF16 qualification", resourceClass: .standard))
    }

    private func assertCompleted(_ response: ChatResponse, expected: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(response.finishReason, .stop, file: file, line: line)
        XCTAssertFalse(response.hasIncompleteReasoning, file: file, line: line)
        XCTAssertEqual(response.response.trimmingCharacters(in: .whitespacesAndNewlines), expected, file: file, line: line)
        XCTAssertGreaterThan(response.tokensGenerated, 0, file: file, line: line)
    }

    func testInstalledBF16TokenizerMatchesPinnedToolHistory() async throws {
        let root = try checkpointRoot()
        let tokenizer = try await LFM2TokenizerAndTemplate.load(from: URL(fileURLWithPath: root))
        for fixture in try LFM25ToolTemplateFixture.load() {
            let tokens = try tokenizer.encodeForGeneration(
                messages: fixture.messages, tools: LFM25ToolTemplateFixture.tools,
                includeThinking: false, maxLength: 8_192
            )
            XCTAssertEqual(tokens, fixture.tokens, fixture.id.rawValue)
        }
    }

    func testInstalledBF16CompletesAnswersAndToolRoundTrip() async throws {
        let root = try checkpointRoot()
        let lease = try await acquireLease()
        defer { lease.release() }
        let generator = LFM2Generator(modelId: LFM2Resources.a2b24BBF16ModelId)
        do {
            for (prompt, expected) in [
                ("Reply with exactly READY and no other text.", "READY"),
                ("What is 17 + 25? Reply with only the decimal integer answer.", "42"),
            ] {
                let response = try await generator.chat(
                    ChatRequest(
                        messages: [.init(role: .user, content: prompt)],
                        maxTokens: 2_048, temperature: 0, showThinking: false, maxContextTokens: 8_192
                    ),
                    modelPath: root, progressHandler: nil
                )
                assertCompleted(response, expected: expected)
            }
            let messages = [
                ChatMessage(
                    role: .system,
                    content: "Use calculate for arithmetic. After receiving its result, reply with only the decimal integer returned by the tool."
                ),
                ChatMessage(role: .user, content: "Use calculate to evaluate the expression 17 + 25."),
            ]
            let response = try await generator.chat(
                ChatRequest(
                    messages: messages, maxTokens: 2_048, temperature: 0,
                    tools: LFM25ToolTemplateFixture.tools, maxContextTokens: 8_192
                ),
                modelPath: root, progressHandler: nil
            )
            let call = try XCTUnwrap(response.toolCalls?.first)
            XCTAssertEqual(response.toolCalls?.count, 1)
            XCTAssertEqual(call.name, "calculate")
            XCTAssertEqual(call.arguments["expression"]?.replacingOccurrences(of: " ", with: ""), "17+25")
            let history = messages + [
                ChatMessage(
                    role: .assistant, content: "", reasoningContent: response.reasoningContent,
                    toolCalls: [.init(name: call.name, arguments: call.arguments.mapValues(OpenAIJSONValue.string))]
                ),
                ChatMessage(role: .tool, content: "42", name: "calculate", toolCallID: "call_0"),
            ]
            let final = try await generator.chat(
                ChatRequest(
                    messages: history, maxTokens: 2_048, temperature: 0,
                    showThinking: false, tools: LFM25ToolTemplateFixture.tools, maxContextTokens: 8_192
                ),
                modelPath: root, progressHandler: nil
            )
            assertCompleted(final, expected: "42")
            await generator.unload()
        } catch {
            await generator.unload()
            throw error
        }
    }

    func testInstalledBF16RetrievesAcrossChunkedPrefill() async throws {
        let root = try checkpointRoot()
        let lease = try await acquireLease()
        defer { lease.release() }
        let generator = LFM2Generator(modelId: LFM2Resources.a2b24BBF16ModelId)
        let prompt = "The access code is TANGERINE-4827.\n"
            + String(repeating: "This is unrelated background text.\n", count: 1_000)
            + "\nCopy the access code from the first line. Reply with exactly that code and no other text."
        do {
            let response = try await generator.chat(
                ChatRequest(
                    messages: [.init(role: .user, content: prompt)],
                    maxTokens: 2_048, temperature: 0, showThinking: false, maxContextTokens: 32_768
                ),
                modelPath: root, progressHandler: nil
            )
            XCTAssertGreaterThan(try XCTUnwrap(response.promptTokens), 4_096)
            assertCompleted(response, expected: "TANGERINE-4827")
            await generator.unload()
        } catch {
            await generator.unload()
            throw error
        }
    }
}
