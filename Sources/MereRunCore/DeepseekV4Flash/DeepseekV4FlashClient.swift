import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public struct DeepseekV4FlashClientResponse: Sendable {
    public let statusCode: Int
    public let contentType: String
    public let body: Data

    public init(statusCode: Int, contentType: String, body: Data) {
        self.statusCode = statusCode
        self.contentType = contentType
        self.body = body
    }
}

public enum DeepseekV4FlashClient {
    public static let defaultTimeout: TimeInterval = 600

    public static func makeChatCompletionsRequest(
        url: URL,
        requestBody: Data,
        contentType: String? = nil,
        timeoutInterval: TimeInterval = defaultTimeout
    ) -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(contentType ?? "application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = requestBody
        request.timeoutInterval = timeoutInterval
        return request
    }

    public static func normalizedChatCompletionData(
        url: URL,
        requestBody: Data,
        contentType: String? = nil,
        session: URLSession = .shared
    ) async throws -> DeepseekV4FlashClientResponse {
        let request = makeChatCompletionsRequest(
            url: url,
            requestBody: requestBody,
            contentType: contentType
        )
        let (data, response) = try await session.data(for: request)
        let http = response as? HTTPURLResponse
        let upstreamContentType = http?.value(forHTTPHeaderField: "Content-Type") ?? "application/json"
        return DeepseekV4FlashClientResponse(
            statusCode: http?.statusCode ?? 502,
            contentType: upstreamContentType,
            body: normalizedChatCompletionBody(data, contentType: upstreamContentType)
        )
    }

    public static func normalizedChatCompletionBody(_ data: Data, contentType: String?) -> Data {
        guard contentType?.lowercased().contains("json") != false else {
            return data
        }
        return DeepseekV4FlashJSONRepair.escapingControlCharactersInsideStrings(data)
    }

    public static func requestWantsStreamingResponse(_ requestBody: Data) -> Bool {
        (try? JSONDecoder().decode(OpenAIChatRequest.self, from: requestBody).stream) == true
    }

    public static func isEventStreamContentType(_ contentType: String?) -> Bool {
        contentType?.lowercased().hasPrefix("text/event-stream") == true
    }
}
