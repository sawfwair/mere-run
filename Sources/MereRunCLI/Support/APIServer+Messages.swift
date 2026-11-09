import Foundation
import Hummingbird
import HTTPTypes
import Logging
import NIOCore
import MereRunCore
import MediaIO

extension CodeGenServer {
    func handleMessages(_ request: Request, counting: Bool, channel: any Channel) async -> Response {
        var request = request
        let requestID = "req_" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
        var session: RuntimeChatSession?
        do {
            try APIAuthentication.validate(expected: apiKey, headers: request.headers)
            guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
                throw MessagesFailure.invalid("content-type", "must be application/json")
            }
            guard await requestLimiter.allowRequest() else {
                throw MessagesFailure(status: 429, type: "rate_limit_error", message: "Rate limit exceeded.")
            }
            let version = try MessagesVersion.resolve(request.headers[HTTPField.Name("anthropic-version")!])
            let body: ByteBuffer
            do { body = try await request.collectBody(upTo: 32 * 1024 * 1024) }
            catch { throw MessagesFailure(status: 413, type: "request_too_large", message: "Request body exceeds 32 MiB or could not be read.") }
            let wire: MessagesRequest
            do { wire = try JSONDecoder().decode(MessagesRequest.self, from: Data(body.readableBytesView)) }
            catch let failure as MessagesFailure { throw failure }
            catch { throw MessagesFailure.invalid("body", "malformed Messages JSON; check field types and required fields") }
            let continuity = try messagesContinuityOverride ?? MessagesContinuity.installation()
            let input = try RuntimeChatRequest.messages(wire, version: version, counting: counting, continuity: continuity,
                authenticatedScope: MessagesAdapter.digest(Data((apiKey ?? "loopback").utf8)))
            let started = try await services.startChat(input, fallbackLoraPath: fallbackLoraPath, contextSize: contextSize)
            session = started
            let prepared: ChatPreparedPrompt?
            if [.textChatMuseGlimmer, .textChatQ35, .textChatQ36, .textChatGemma4].contains(started.engine) || counting {
                prepared = try await started.preparePrompt()
            } else { prepared = nil }
            if counting {
                var payload: [String: ChatJSONValue] = ["input_tokens": .number(Double(prepared?.tokens.count ?? 0))]
                if wire.context_management != nil, let original = started.originalInputTokens {
                    payload["context_management"] = .object(["original_input_tokens": .number(Double(original))])
                }
                let response = try Self.messagesJSON(payload, requestID: requestID)
                await started.finish(); return response
            }
            let messageID = "msg_" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
            if wire.stream == true {
                return try await messagesStream(started, id: messageID, requestID: requestID,
                                          inputTokens: prepared?.tokens.count ?? 0, continuity: continuity, channel: channel)
            }
            let result = try await started.chat()
            let response = try MessagesAdapter.response(result, session: started, id: messageID, continuity: continuity,
                                                       inputTokens: prepared?.tokens.count ?? 0)
            let http = try Self.messagesJSON(response, requestID: requestID)
            await started.finish(); return http
        } catch {
            await session?.finish(cancelled: Task.isCancelled || error is CancellationError)
            return Self.messagesError(error, requestID: requestID)
        }
    }

    private func messagesStream(_ session: RuntimeChatSession, id: String, requestID: String,
                                inputTokens: Int, continuity: MessagesContinuity, channel: any Channel) async throws -> Response {
        let (stream, continuation) = AsyncStream<ByteBuffer>.makeStream()
        let writer = try MessagesStreamWriter(id: id, inputTokens: inputTokens, model: session.modelID, requestID: requestID,
            showThinking: session.request.showThinking, continuity: continuity) { buffer in continuation.yield(buffer) }
        let cancellation = APIRequestCancellation()
        let connectionHandler = APIConnectionCancellationHandler(cancellation: cancellation)
        try await channel.pipeline.addHandler(connectionHandler, position: .first).get()
        let heartbeat = Task {
            do {
                while !Task.isCancelled {
                    try await Task.sleep(for: messagesHeartbeatInterval)
                    try Task.checkCancellation(); writer.ping()
                }
            } catch { }
        }
        let generation = Task {
            defer { heartbeat.cancel(); continuation.finish() }
            do {
                try cancellation.checkConnection()
                try Task.checkCancellation()
                let result = try await session.chat { progress in
                    if let event = progress.event { writer.observe(event) }
                }
                try Task.checkCancellation()
                let response = try MessagesAdapter.response(result, session: session, id: id, continuity: continuity, inputTokens: inputTokens)
                try writer.finish(response)
                await session.finish()
            } catch {
                await session.finish(cancelled: Task.isCancelled || error is CancellationError)
                if !Task.isCancelled {
                    let failure = Self.messagesFailure(error)
                    Logger(label: "mere.run.api.messages").error("Messages stream failed", metadata: [
                        "request_id": .string(requestID), "error_type": .string(failure.type), "detail": .string(failure.message)
                    ])
                    writer.fail(failure)
                }
            }
            cancellation.finish()
            try? await channel.pipeline.removeHandler(connectionHandler).get()
        }
        cancellation.install { generation.cancel() }
        continuation.onTermination = { termination in
            if case .cancelled = termination {
                session.observeClientDisconnect(); heartbeat.cancel(); generation.cancel()
            }
        }
        return Response(status: .ok, headers: [.contentType: "text/event-stream", .cacheControl: "no-cache",
                                               HTTPField.Name("request-id")!: requestID], body: .init { output in
            do {
                for await buffer in stream { try await output.write(buffer) }
                try await output.finish(nil)
            } catch {
                session.observeClientDisconnect(); heartbeat.cancel(); generation.cancel()
                await generation.value
                throw error
            }
        })
    }

    nonisolated static func messagesJSON<T: Encodable>(_ value: T, requestID: String, status: HTTPResponse.Status = .ok) throws -> Response {
        Response(status: status, headers: [.contentType: "application/json", HTTPField.Name("request-id")!: requestID],
                 body: .init(byteBuffer: ByteBuffer(bytes: try JSONEncoder().encode(value))))
    }
    nonisolated static func messagesFailure(_ error: Error) -> MessagesFailure {
        if let failure = error as? MessagesFailure { return failure }
        if let issue = error as? ChatRequestIssue, issue.field.hasPrefix("tool_use") {
            return .output(issue.field, issue.message)
        }
        if error is APIRequestValidationError || error is ChatRequestIssue || error is MediaDocumentError || error is MediaIOError {
            return .invalid("request", error.localizedDescription)
        }
        if let error = error as? RuntimeModelPoolError {
            switch error {
            case .unknownModel, .modelNotInstalled: return MessagesFailure(status: 404, type: "not_found_error", message: error.localizedDescription)
            case .unsupportedModel, .incompatibleEngine, .invalidSettings, .rawProxyUnavailable: return .invalid("model", error.localizedDescription)
            case .unloadConflict: return MessagesFailure(status: 409, type: "invalid_request_error", message: error.localizedDescription)
            }
        }
        if error is CancellationError { return MessagesFailure(status: 499, type: "api_error", message: "Request cancelled.") }
        return MessagesFailure(status: 500, type: "api_error", message: "Local inference failed: \(error.localizedDescription)")
    }
    nonisolated static func messagesError(_ error: Error, requestID: String) -> Response {
        let failure = messagesFailure(error)
        let envelope = MessagesErrorEnvelope(error: .init(type: failure.type, message: failure.message), request_id: requestID)
        // Internal envelope consists only of strings and cannot fail JSON encoding.
        return (try? messagesJSON(envelope, requestID: requestID, status: .init(code: failure.status))) ?? Response(status: .internalServerError)
    }
}

enum APIAuthentication {
    static func validate(expected: String?, headers: HTTPFields) throws {
        let keyName = HTTPField.Name("x-api-key")!
        guard headers.filter({ $0.name == .authorization }).count <= 1,
              headers.filter({ $0.name == keyName }).count <= 1 else {
            throw MessagesFailure(status: 401, type: "authentication_error", message: "Duplicate authentication credentials are not allowed.")
        }
        let authorization = headers[.authorization]
        let bearer: String?
        if let authorization {
            guard authorization.lowercased().hasPrefix("bearer "), authorization.count > 7 else {
                throw MessagesFailure(status: 401, type: "authentication_error", message: "Use bearer authentication or x-api-key.")
            }
            bearer = String(authorization.dropFirst(7))
        } else { bearer = nil }
        let apiKey = headers[keyName]
        if let bearer, let apiKey, bearer != apiKey {
            throw MessagesFailure(status: 401, type: "authentication_error", message: "Conflicting bearer and x-api-key credentials.")
        }
        if let expected, (bearer ?? apiKey) != expected {
            throw MessagesFailure(status: 401, type: "authentication_error", message: "Unauthorized; provide the configured API key.")
        }
    }
}
