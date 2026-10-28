import Foundation
import MLX
import MereRunAdmission
import XCTest
@testable import MereRunCore

/// Opt-in checks use the production loader with either pinned released checkpoint.
final class FlintInstalledQualificationTests: MereRunCoreTestCase {
    private func checkpointRoot() throws -> String {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_FLINT_CHECKPOINT_ROOT"] else {
            throw XCTSkip("Set MERERUN_FLINT_CHECKPOINT_ROOT to a pinned Flint BF16 or INT4 checkpoint.")
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Set MERERUN_TEST_MLX_DEVICE=gpu for released-checkpoint qualification.")
        }
        return path
    }

    private func assertCompleted(_ response: ChatResponse, expected: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(response.finishReason, .stop, file: file, line: line)
        XCTAssertFalse(response.hasIncompleteReasoning, file: file, line: line)
        XCTAssertEqual(response.response.trimmingCharacters(in: .whitespacesAndNewlines), expected, file: file, line: line)
        XCTAssertGreaterThan(response.tokensGenerated, 0, file: file, line: line)
    }

    func testInstalledCheckpointAnswersToolsAndChunkedRetrieval() async throws {
        let root = try checkpointRoot()
        let coordinator = MachineInferenceCoordinator(
            stateDirectory: MereRunModelPaths.applicationSupportBase.appending(path: "admission"),
            hostSnapshot: {
                MachineInferenceCoordinator.currentHostSnapshot(
                    diskURL: MereRunModelPaths.applicationSupportBase.deletingLastPathComponent()
                )
            }
        )
        let lease = try await coordinator.acquire(.init(label: "Flint released-checkpoint qualification", resourceClass: .standard))
        defer { lease.release() }
        let generator = FlintGenerator()
        do {
            let answer = try await generator.chat(
                ChatRequest(messages: [.init(role: .user, content: "What is 17 + 25? Reply with only the decimal integer answer.")],
                            maxTokens: 2_048, temperature: 0, showThinking: false, maxContextTokens: 8_192),
                modelPath: root, progressHandler: nil
            )
            assertCompleted(answer, expected: "42")
            let tools = [ToolDefinition(name: "calculate", description: "Evaluate arithmetic.",
                                        parameters: ["expression": .init(type: "string", description: "Arithmetic expression")])]
            let messages = [
                ChatMessage(role: .system, content: "Use calculate for arithmetic. After receiving its result, reply with only the decimal integer returned by the tool."),
                ChatMessage(role: .user, content: "Use calculate to evaluate the expression 17 + 25.")
            ]
            let response = try await generator.chat(
                ChatRequest(messages: messages, maxTokens: 2_048, temperature: 0, tools: tools, maxContextTokens: 8_192),
                modelPath: root, progressHandler: nil
            )
            let call = try XCTUnwrap(response.toolCalls?.first)
            XCTAssertEqual(response.toolCalls?.count, 1)
            XCTAssertEqual(call.name, "calculate")
            XCTAssertEqual(call.arguments["expression"]?.filter { !$0.isWhitespace }, "17+25")
            let history = messages + [
                ChatMessage(role: .assistant, content: "", reasoningContent: response.reasoningContent,
                            toolCalls: [.init(name: call.name, arguments: call.arguments.mapValues(OpenAIJSONValue.string))]),
                ChatMessage(role: .tool, content: "42", name: "calculate", toolCallID: "call_0")
            ]
            let final = try await generator.chat(
                ChatRequest(messages: history, maxTokens: 2_048, temperature: 0, showThinking: false,
                            tools: tools, maxContextTokens: 8_192), modelPath: root, progressHandler: nil
            )
            assertCompleted(final, expected: "42")
            let prompt = "The access code is TANGERINE-4827.\n"
                + String(repeating: "This background line does not change the access code.\n", count: 160)
                + "\nCopy the access code from the first line. Reply with exactly that code and no other text."
            let retrieval = try await generator.chat(
                ChatRequest(messages: [.init(role: .user, content: prompt)], maxTokens: 2_048, temperature: 0,
                            showThinking: false, maxContextTokens: 8_192), modelPath: root, progressHandler: nil
            )
            XCTAssertGreaterThan(try XCTUnwrap(retrieval.promptTokens), 1_024)
            assertCompleted(retrieval, expected: "TANGERINE-4827")
            await generator.unload()
        } catch {
            await generator.unload()
            throw error
        }
    }
}
