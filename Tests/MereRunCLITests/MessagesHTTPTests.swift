import Foundation
import XCTest
import Hummingbird
import MereRunCore
@testable import MereRunCLI

private actor MessagesHTTPModel: RuntimeChatModelLease {
    var releases = 0
    var cancelled = 0
    var preparedToolSets: [Set<String>] = []
    func preparePrompt(_ request: ChatRequest) async throws -> ChatPreparedPrompt {
        preparedToolSets.append(Set((request.tools ?? []).map(\.name)))
        if request.messages.last?.content == "full" { return try ChatPreparedPrompt(tokens: Array(repeating: 1, count: 32760), contextLimit: 32768) }
        return try ChatPreparedPrompt(tokens: [1, 2, 3, 4, 5, 6, 7], contextLimit: 32768)
    }
    func chat(_ request: ChatRequest, progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws -> ChatResponse {
        let prompt = request.messages.last?.content ?? ""
        if prompt == "prefill_ping" { try await Task.sleep(for: .milliseconds(200)) }
        if prompt == "error" { throw MessagesFailure.invalid("fixture", "requested test failure") }
        if prompt == "slow" {
            do { try await Task.sleep(for: .seconds(30)) }
            catch { cancelled += 1; throw error }
        }
        if prompt == "mixed" {
            return ChatResponse(response: "", tokensGenerated: 5, toolCalls: [
                ToolCall(name: "catalog_search", jsonArguments: ["pattern": .string("calculate")]),
                ToolCall(name: "calculate", jsonArguments: ["value": .number(42), "nested": .object(["ok": .bool(true)])])], promptTokens: 7)
        }
        if request.messages.last?.role == .tool, request.messages.first(where: { $0.role == .user })?.content == "search_call" {
            return ChatResponse(response: "", tokensGenerated: 3,
                toolCalls: [ToolCall(name: "calculate", jsonArguments: ["value": .number(42), "nested": .object(["ok": .bool(true)])])], promptTokens: 7)
        }
        if prompt == "search" || prompt == "search_call" {
            return ChatResponse(response: "", tokensGenerated: 2,
                toolCalls: [ToolCall(name: "catalog_search", jsonArguments: ["pattern": .string("calculate")])], promptTokens: 7)
        }
        if prompt == "bad_tool" {
            return ChatResponse(response: "", tokensGenerated: 5,
                toolCalls: [ToolCall(name: "not_declared", jsonArguments: [:])], promptTokens: 7)
        }
        if prompt == "tool" {
            return ChatResponse(response: "", tokensGenerated: 5,
                toolCalls: [ToolCall(name: "calculate", jsonArguments: ["value": .number(42), "nested": .object(["ok": .bool(true)])])],
                promptTokens: 7)
        }
        let cacheRead = prompt == "cached" ? 5 : 0
        progressHandler?(.init(stage: .encoding, event: .usage(input: 7, output: 0, cacheRead: cacheRead, cacheWrite: 0)))
        if request.showThinking { progressHandler?(.init(stage: .generating, event: .reasoning("work 🧮"))) }
        progressHandler?(.init(stage: .generating, event: .text("4")))
        progressHandler?(.init(stage: .generating, event: .text("2")))
        var response = ChatResponse(response: "42", tokensGenerated: 2, promptTokens: 7, finishReason: .stop,
                                    reasoningContent: request.showThinking ? "work 🧮" : nil)
        response.cacheReadTokens = cacheRead
        return response
    }
    func deepseekChatCompletionsURL(progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws -> URL {
        throw MessagesFailure.unsupported("proxy")
    }
    func release() { releases += 1 }
}

final class MessagesHTTPTests: XCTestCase {
    private func withServer(_ operation: (URL, RuntimeServingServices, MessagesHTTPModel) async throws -> Void) async throws {
        let model = MessagesHTTPModel()
        let modelID = MuseGlimmerResources.modelId
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: modelID))
        let services = RuntimeServingServices(defaultModelID: modelID, modelPath: nil, engine: .textChatMuseGlimmer,
            maxActiveRequests: 1, chatPlanner: { input, _, context in
                let prepared = try input.prepare(RuntimeChatConfiguration(modelID: modelID, engine: .textChatMuseGlimmer,
                    settings: .init(), profile: profile, contextSize: context, fallbackLoraPath: nil))
                return RuntimeChatPlan(lease: model, request: prepared.request, modelID: modelID,
                                       engine: .textChatMuseGlimmer, includeUsage: true, scope: prepared.scope)
            })
        let server = CodeGenServer(services: services, apiKey: "test-key", continuity: try MessagesContinuity(key: Data(repeating: 1, count: 32)),
                                   messagesHeartbeatInterval: .milliseconds(20))
        let (ports, ready) = AsyncStream<Int>.makeStream()
        let app = Application(router: server.buildRouter(), configuration: .init(address: .hostname("127.0.0.1", port: 0)),
                              onServerRunning: { if let port = $0.localAddress?.port { ready.yield(port) }; ready.finish() })
        let task = Task { try await app.runService(gracefulShutdownSignals: []) }
        do {
            var iterator = ports.makeAsyncIterator()
            let nextPort = await iterator.next()
            let port = try XCTUnwrap(nextPort)
            try await operation(URL(string: "http://127.0.0.1:\(port)")!, services, model)
            task.cancel(); _ = await task.result
        } catch { task.cancel(); _ = await task.result; throw error }
    }
    private func post(_ url: URL, prompt: String = "hello", extra: String = "", credential: String? = "test-key",
                      version: String? = nil, route: String = "v1/messages?beta=true") async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: URL(string: route, relativeTo: url)!)
        request.httpMethod = "POST"; request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.setValue(credential, forHTTPHeaderField: "x-api-key")
        request.setValue(version, forHTTPHeaderField: "anthropic-version")
        request.httpBody = Data((#"{"model":"fixture","max_tokens":16,"messages":[{"role":"user","content":""# + prompt + #""}]"# + extra + "}").utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        return (data, try XCTUnwrap(response as? HTTPURLResponse))
    }
    func testHTTPAuthenticationErrorsCountingStreamingAndRecovery() async throws {
        try await withServer { url, services, model in
            let (denied, response) = try await self.post(url, credential: nil)
            XCTAssertEqual(response.statusCode, 401)
            XCTAssertEqual(try JSONDecoder().decode(ChatJSONValue.self, from: denied).objectValue?["type"], .string("error"))
            let (_, removedSampling) = try await self.post(url, extra: #", "temperature":1"#, version: "2026-10-21")
            XCTAssertEqual(removedSampling.statusCode, 400)
            let (_, unsupported) = try await self.post(url, extra: #", "output_config":{"format":{"type":"json_schema","schema":{}}}"#)
            XCTAssertEqual(unsupported.statusCode, 400)
            let (count, counted) = try await self.post(url, route: "v1/messages/count_tokens")
            XCTAssertEqual(counted.statusCode, 200)
            XCTAssertEqual(try JSONDecoder().decode(ChatJSONValue.self, from: count).objectValue?["input_tokens"], .number(7))
            let (stream, streamed) = try await self.post(url, extra: #", "stream":true,"thinking":{"type":"adaptive"}"#)
            XCTAssertEqual(streamed.statusCode, 200)
            let text = String(decoding: stream, as: UTF8.self)
            XCTAssertTrue(text.contains("event: message_start")); XCTAssertTrue(text.contains("event: message_stop"))
            XCTAssertTrue(text.contains("thinking_delta")); XCTAssertTrue(text.contains("signature_delta")); XCTAssertFalse(text.contains("[DONE]"))
            let (failed, failure) = try await self.post(url, prompt: "error")
            XCTAssertEqual(failure.statusCode, 400); XCTAssertTrue(String(decoding: failed, as: UTF8.self).contains("requested test failure"))
            let (failedStream, failureStream) = try await self.post(url, prompt: "error", extra: #", "stream":true"#)
            XCTAssertEqual(failureStream.statusCode, 200)
            XCTAssertTrue(String(decoding: failedStream, as: UTF8.self).contains("event: error"))
            XCTAssertFalse(String(decoding: failedStream, as: UTF8.self).contains("event: message_stop"))
            let (recovered, recovery) = try await self.post(url)
            XCTAssertEqual(recovery.statusCode, 200); XCTAssertTrue(String(decoding: recovered, as: UTF8.self).contains("42"))
            let snapshot = await services.admission.snapshot(); XCTAssertEqual(snapshot.activeRequests, 0)
            let releases = await model.releases; XCTAssertEqual(releases, 5)
        }
    }
    func testProbeAuthenticationAndHostedServiceRejection() async throws {
        try await withServer { url, _, _ in
            do {
                _ = try await ClaudeCodeConnection.inspect(origin: url, model: "fixture", credential: nil, credentialVariable: "LOCAL_KEY")
                XCTFail("Missing server credential was accepted")
            } catch let failure as MessagesFailure { XCTAssertTrue(failure.message.contains("check LOCAL_KEY")) }
            for credential in [nil, "test-key"] {
                var probe = URLRequest(url: url.appendingPathComponent("api/hello")); probe.httpMethod = "HEAD"
                probe.setValue(credential, forHTTPHeaderField: "x-api-key")
                let (_, response) = try await URLSession.shared.data(for: probe)
                XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, credential == nil ? 401 : 200)
            }
            for route in ["v1/files", "v1/files/file_id/content", "v1/messages/batches", "v1/messages/batches/batch_id/results", "v1/mcp_servers"] {
                let (data, response) = try await self.post(url, route: route)
                XCTAssertEqual(response.statusCode, 400)
                XCTAssertTrue(String(decoding: data, as: UTF8.self).contains("does not support"))
                XCTAssertNotNil(response.value(forHTTPHeaderField: "request-id"))
            }
        }
    }
    func testHTTPMalformedBodiesLimitsAndUnsupportedCapabilitiesRecover() async throws {
        try await withServer { url, services, _ in
            for (body, expected) in [(Data("{".utf8), 400), (Data(repeating: 32, count: 32 * 1024 * 1024 + 1), 413)] {
                var request = URLRequest(url: url.appendingPathComponent("v1/messages"))
                request.httpMethod = "POST"; request.httpBody = body
                request.setValue("test-key", forHTTPHeaderField: "x-api-key")
                request.setValue("application/json", forHTTPHeaderField: "content-type")
                let (data, response) = try await URLSession.shared.data(for: request)
                XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, expected)
                XCTAssertEqual(try JSONDecoder().decode(ChatJSONValue.self, from: data).objectValue?["type"], .string("error"))
            }
            let (_, unknownVersion) = try await self.post(url, version: "2099-01-01")
            XCTAssertEqual(unknownVersion.statusCode, 400)
            let (_, strict) = try await self.post(url, extra: #", "tools":[{"name":"read","strict":true,"input_schema":{"type":"object","properties":{}}}]"#)
            XCTAssertEqual(strict.statusCode, 400)
            let (_, schema) = try await self.post(url, extra: #", "tools":[{"name":"read","input_schema":{"type":"object","unsupportedKeyword":true}}]"#)
            XCTAssertEqual(schema.statusCode, 400)
            let (_, recovery) = try await self.post(url)
            XCTAssertEqual(recovery.statusCode, 200)
            let snapshot = await services.admission.snapshot()
            XCTAssertEqual(snapshot.activeRequests, 0); XCTAssertEqual(snapshot.queuedRequests, 0)
        }
    }
    func testMessageStartContainsActualInputAndCacheUsage() async throws {
        try await withServer { url, _, _ in
            let (data, response) = try await self.post(url, prompt: "cached", extra: #", "stream":true"#)
            XCTAssertEqual(response.statusCode, 200)
            let records = try String(decoding: data, as: UTF8.self).split(separator: "\n").filter { $0.hasPrefix("data: ") }.map {
                try JSONDecoder().decode(ChatJSONValue.self, from: Data($0.dropFirst(6).utf8))
            }
            let start = try XCTUnwrap(records.first(where: { $0.objectValue?["type"] == .string("message_start") })?.objectValue?["message"]?.objectValue)
            XCTAssertEqual(start["stop_reason"], .null); XCTAssertEqual(start["stop_sequence"], .null)
            XCTAssertEqual(start["usage"]?.objectValue?["input_tokens"], .number(2))
            XCTAssertEqual(start["usage"]?.objectValue?["cache_read_input_tokens"], .number(5))
            XCTAssertEqual(start["usage"]?.objectValue?["output_tokens"], .number(0))
            let final = try XCTUnwrap(records.first(where: { $0.objectValue?["type"] == .string("message_delta") }))
            XCTAssertEqual(final.objectValue?["usage"]?.objectValue?["input_tokens"], .number(2))
            XCTAssertEqual(final.objectValue?["usage"]?.objectValue?["output_tokens"], .number(2))
        }
    }
    func testPrefillPingsPrecedeAccurateMessageStart() async throws {
        try await withServer { url, _, _ in
            let (data, response) = try await self.post(url, prompt: "prefill_ping", extra: #", "stream":true"#)
            XCTAssertEqual(response.statusCode, 200)
            let text = String(decoding: data, as: UTF8.self)
            XCTAssertTrue(text.hasPrefix("event: ping\n"))
            XCTAssertEqual(text.components(separatedBy: "event: message_start").count, 2)
            XCTAssertTrue(text.contains("event: message_stop"))
        }
    }
    func testDeferredSearchPreparesExpandedPromptBeforeContinuation() async throws {
        try await withServer { url, _, model in
            let tools = #", "tools":[{"type":"tool_search_tool_regex_20251119","name":"catalog_search"},{"name":"calculate","defer_loading":true,"input_schema":{"type":"object","properties":{"value":{"type":"number"},"nested":{"type":"object"}}}}]"#
            let (_, response) = try await self.post(url, prompt: "search_call", extra: tools)
            XCTAssertEqual(response.statusCode, 200)
            let prepared = await model.preparedToolSets
            XCTAssertEqual(prepared, [Set(["catalog_search"]), Set(["catalog_search", "calculate"])])
        }
    }
    func testGenerationReservesOutputWhileCountingDoesNotGenerate() async throws {
        try await withServer { url, services, _ in
            let (count, counted) = try await self.post(url, prompt: "full", route: "v1/messages/count_tokens")
            XCTAssertEqual(counted.statusCode, 200)
            XCTAssertEqual(try JSONDecoder().decode(ChatJSONValue.self, from: count).objectValue?["input_tokens"], .number(32760))
            let (error, rejected) = try await self.post(url, prompt: "full", extra: #", "stream":true"#)
            XCTAssertEqual(rejected.statusCode, 400)
            XCTAssertTrue(String(decoding: error, as: UTF8.self).contains("reduce max_tokens"))
            let (_, recovered) = try await self.post(url)
            XCTAssertEqual(recovered.statusCode, 200)
            let snapshot = await services.admission.snapshot(); XCTAssertEqual(snapshot.activeRequests, 0)
        }
    }
    func testMalformedModelToolOutputFailsWithoutPublishingExecutableCalls() async throws {
        try await withServer { url, _, _ in
            let tools = #", "tools":[{"name":"calculate","input_schema":{"type":"object","properties":{}}}]"#
            let (_, response) = try await self.post(url, prompt: "bad_tool", extra: tools)
            XCTAssertEqual(response.statusCode, 500)
            let (bytes, stream) = try await self.post(url, prompt: "bad_tool", extra: tools + #", "stream":true"#)
            XCTAssertEqual(stream.statusCode, 200)
            let body = String(decoding: bytes, as: UTF8.self)
            XCTAssertTrue(body.contains("event: error"))
            XCTAssertFalse(body.contains("event: message_stop"))
            XCTAssertFalse(body.contains("input_json_delta"))
        }
    }

    func testQueuedDisconnectAndSubsequentRecovery() async throws {
        for streaming in [false, true] {
        try await withServer { url, services, model in
            var slow = URLRequest(url: url.appendingPathComponent("v1/messages"))
            slow.httpMethod = "POST"; slow.setValue("application/json", forHTTPHeaderField: "content-type")
            slow.setValue("test-key", forHTTPHeaderField: "x-api-key")
            let payload = #"{"model":"fixture","max_tokens":16,"messages":[{"role":"user","content":"slow"}],"stream":"# + String(streaming) + "}"
            slow.httpBody = Data(payload.utf8)
            let firstRequest = slow
            let first = Task { try await URLSession.shared.data(for: firstRequest) }
            for _ in 0..<100 {
                if await services.admission.snapshot().activeRequests == 1 { break }
                try await Task.sleep(for: .milliseconds(10))
            }
            var queued = firstRequest
            queued.httpBody = Data(#"{"model":"fixture","max_tokens":16,"messages":[{"role":"user","content":"hello"}]}"#.utf8)
            let secondRequest = queued
            let second = Task { try await URLSession.shared.data(for: secondRequest) }
            try await Task.sleep(for: .milliseconds(50))
            second.cancel(); _ = await second.result
            first.cancel(); _ = await first.result
            for _ in 0..<100 {
                let snapshot = await services.admission.snapshot()
                if snapshot.activeRequests == 0 { break }
                try await Task.sleep(for: .milliseconds(10))
            }
            let snapshot = await services.admission.snapshot()
            XCTAssertEqual(snapshot.activeRequests, 0)
            let cancellations = await model.cancelled; XCTAssertEqual(cancellations, 1)
            let (_, response) = try await self.post(url)
            XCTAssertEqual(response.statusCode, 200)
        }
        }
    }

    func testOfficialAnthropicSDK() async throws {
        guard let python = ProcessInfo.processInfo.environment["MERERUN_ANTHROPIC_SDK_PYTHON"] else {
            throw XCTSkip("Set MERERUN_ANTHROPIC_SDK_PYTHON to a Python with the official anthropic SDK")
        }
        try await withServer { url, _, _ in
            let process = Process(); process.executableURL = URL(fileURLWithPath: python)
            process.arguments = ["scripts/test-messages-sdk.py", url.absoluteString]
            let result = try await Task.detached { try BoundedProcessRunner.run(process, timeout: 30) }.value
            XCTAssertEqual(result.status, 0, result.stderr.text + result.stdout.text)
            XCTAssertEqual(result.completion, .exited)
        }
    }
}
