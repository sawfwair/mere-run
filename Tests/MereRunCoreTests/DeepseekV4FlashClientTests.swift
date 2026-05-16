import Foundation
import XCTest
@testable import MereRunCore

final class DeepseekV4FlashClientTests: XCTestCase {
    func testBuildsChatCompletionsRequest() throws {
        let url = try XCTUnwrap(URL(string: "http://127.0.0.1:8080/v1/chat/completions"))
        let body = Data("{\"stream\":false}".utf8)

        let request = DeepseekV4FlashClient.makeChatCompletionsRequest(
            url: url,
            requestBody: body,
            contentType: "application/json; charset=utf-8",
            timeoutInterval: 12
        )

        XCTAssertEqual(request.url, url)
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Content-Type"), "application/json; charset=utf-8")
        XCTAssertEqual(request.httpBody, body)
        XCTAssertEqual(request.timeoutInterval, 12)
    }

    func testDetectsStreamingOpenAIRequests() throws {
        let streaming = OpenAIChatRequest(
            model: "text-chat-deepseek-v4-flash",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            stream: true
        )
        let nonStreaming = OpenAIChatRequest(
            model: "text-chat-deepseek-v4-flash",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            stream: false
        )

        let encoder = JSONEncoder()
        XCTAssertTrue(DeepseekV4FlashClient.requestWantsStreamingResponse(try encoder.encode(streaming)))
        XCTAssertFalse(DeepseekV4FlashClient.requestWantsStreamingResponse(try encoder.encode(nonStreaming)))
    }

    func testNormalizesJSONBodiesOnly() {
        let rawJSON = Data("{\"content\":\"hello\nworld\"}".utf8)
        let rawText = Data("hello\nworld".utf8)

        XCTAssertNoThrow(
            try JSONSerialization.jsonObject(
                with: DeepseekV4FlashClient.normalizedChatCompletionBody(
                    rawJSON,
                    contentType: "application/json"
                )
            )
        )
        XCTAssertEqual(
            DeepseekV4FlashClient.normalizedChatCompletionBody(rawText, contentType: "text/plain"),
            rawText
        )
    }
}
