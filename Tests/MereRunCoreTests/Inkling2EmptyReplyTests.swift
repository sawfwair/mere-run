import XCTest
@testable import MereRunCore

final class Inkling2EmptyReplyTests: XCTestCase {
    private let tool = ToolDefinition(name: "render_preview", description: "Render a slide",
                                      parameters: ["nodeId": .init(type: "string", description: "Slide")])

    private func attempt(text: String = "", calls: [ToolCall]? = nil, finish: ChatFinishReason = .stop) -> Inkling2Generator.Attempt {
        .init(response: ChatResponse(response: text, tokensGenerated: 12, toolCalls: calls, finishReason: finish),
              reasoning: "Let me check the image dimensions.", text: text)
    }

    func testAReplyWithReasoningOnlyIsResampledWhenToolsAreOffered() {
        let request = ChatRequest(messages: [.init(role: .user, content: "Add the photo")], tools: [tool])
        XCTAssertTrue(Inkling2Generator.isEmptyToolReply(request, attempt()))
        XCTAssertTrue(Inkling2Generator.isEmptyToolReply(request, attempt(text: " \n")))
    }

    func testRepliesThatActOrAnswerOrCannotChangeAreKept() {
        let request = ChatRequest(messages: [.init(role: .user, content: "Add the photo")], tools: [tool])
        XCTAssertFalse(Inkling2Generator.isEmptyToolReply(request, attempt(calls: [ToolCall(name: "render_preview", arguments: [:])])))
        XCTAssertFalse(Inkling2Generator.isEmptyToolReply(request, attempt(text: "Done: the photo is on slide 1.")))
        XCTAssertFalse(Inkling2Generator.isEmptyToolReply(request, attempt(finish: .length)))
        let noTools = ChatRequest(messages: [.init(role: .user, content: "Hi")])
        XCTAssertFalse(Inkling2Generator.isEmptyToolReply(noTools, attempt()))
        let greedy = ChatRequest(messages: [.init(role: .user, content: "Add the photo")], temperature: 0, tools: [tool])
        XCTAssertFalse(Inkling2Generator.isEmptyToolReply(greedy, attempt()))
        let seeded = ChatRequest(messages: [.init(role: .user, content: "Add the photo")], seed: 7, tools: [tool])
        XCTAssertFalse(Inkling2Generator.isEmptyToolReply(seeded, attempt()))
    }

    func testRetryCountDefaultsToTwoAndStaysInRange() {
        XCTAssertEqual(Inkling2Generator.emptyReplyRetries([:]), 2)
        XCTAssertEqual(Inkling2Generator.emptyReplyRetries(["MERERUN_INKLING2_EMPTY_REPLY_RETRIES": "0"]), 0)
        XCTAssertEqual(Inkling2Generator.emptyReplyRetries(["MERERUN_INKLING2_EMPTY_REPLY_RETRIES": "9"]), 5)
        XCTAssertEqual(Inkling2Generator.emptyReplyRetries(["MERERUN_INKLING2_EMPTY_REPLY_RETRIES": "x"]), 2)
    }
}
