import Foundation
import XCTest
@testable import MereRunCore

/// Opt-in checkpoint regression coverage. The local root prevents implicit downloads.
final class LFM2InstalledGenerationTests: MereRunCoreTestCase {
    private final class ProgressText: @unchecked Sendable {
        private let lock = NSLock()
        private var text = ""

        func append(_ progress: ChatProgress) {
            guard progress.stage == .generating, let message = progress.message else { return }
            lock.withLock { text += message }
        }

        var value: String { lock.withLock { text } }
    }

    private func modelRoot() throws -> String {
        guard let root = ProcessInfo.processInfo.environment["MERERUN_LFM25_GENERATION_TEST_ROOT"] else {
            throw XCTSkip("Set MERERUN_LFM25_GENERATION_TEST_ROOT to the installed text-chat-lfm25-2.6b-4bit root.")
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: root))
        return root
    }

    private func request(maxTokens: Int, showThinking: Bool = false) -> ChatRequest {
        ChatRequest(
            messages: [
                ChatMessage(role: .system, content: "Follow the requested output format."),
                ChatMessage(role: .user, content: "Reply with exactly: READY"),
            ],
            maxTokens: maxTokens, temperature: 0, topP: 1, topK: 0,
            showThinking: showThinking, maxContextTokens: 2048
        )
    }

    func testInstalledCheckpointKeepsTruncatedThinkingOutOfVisibleOutput() async throws {
        let root = try modelRoot()
        let generator = LFM2Generator(modelId: "text-chat-lfm25-2.6b-4bit")
        let stream = ProgressText()
        do {
            let response = try await generator.chat(
                request(maxTokens: 16), modelPath: root, progressHandler: stream.append
            )
            XCTAssertEqual(response.finishReason, .length)
            XCTAssertEqual(response.tokensGenerated, 16)
            XCTAssertEqual(response.response, "")
            XCTAssertTrue(response.hasIncompleteReasoning)
            XCTAssertFalse(try XCTUnwrap(response.reasoningContent).isEmpty)
            XCTAssertEqual(stream.value, "")

            let shown = try await generator.chat(
                request(maxTokens: 16, showThinking: true), modelPath: root, progressHandler: nil
            )
            XCTAssertTrue(shown.response.hasPrefix("<think>"))
            XCTAssertEqual(shown.finishReason, .length)
            await generator.unload()
        } catch {
            await generator.unload()
            throw error
        }
    }

    func testInstalledCheckpointCompletesAfterTokenLimitedRequest() async throws {
        let root = try modelRoot()
        let generator = LFM2Generator(modelId: "text-chat-lfm25-2.6b-4bit")
        do {
            _ = try await generator.chat(request(maxTokens: 16), modelPath: root, progressHandler: nil)
            let stream = ProgressText()
            let response = try await generator.chat(
                request(maxTokens: 128), modelPath: root, progressHandler: stream.append
            )
            XCTAssertEqual(response.finishReason, .stop)
            XCTAssertLessThan(response.tokensGenerated, 128)
            XCTAssertEqual(response.response, "READY")
            XCTAssertFalse(response.hasIncompleteReasoning)
            XCTAssertEqual(ChatReasoningMarkup.splitThinkBlocks(in: stream.value).visibleContent, "READY")
            await generator.unload()
        } catch {
            await generator.unload()
            throw error
        }
    }
}
