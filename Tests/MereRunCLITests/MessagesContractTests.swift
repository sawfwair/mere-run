import Foundation
import XCTest
import HTTPTypes
import MereRunCore
import MediaIO
@testable import MereRunCLI

final class MessagesContractTests: XCTestCase {
    func testConsecutiveToolResultMessagesFormOneUserTurn() throws {
        let source = #"{"model":"local","max_tokens":16,"messages":[{"role":"user","content":"read"},{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"read","input":{}}]},{"role":"assistant","content":[{"type":"tool_use","id":"b","name":"read","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"first"}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"b","content":"second"},{"type":"text","text":"continue"}]}]}"#
        let wire = try JSONDecoder().decode(MessagesRequest.self, from: Data(source.utf8))
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: MuseGlimmerResources.modelId))
        let configuration = RuntimeChatConfiguration(modelID: MuseGlimmerResources.modelId, engine: .textChatMuseGlimmer,
            settings: .init(), profile: profile, contextSize: 32768, fallbackLoraPath: nil)
        let request = try RuntimeChatRequest.messages(wire, version: .original, counting: false, continuity: MessagesContinuity(key: Data(repeating: 1, count: 32)))
        let prepared = try request.prepare(configuration)
        XCTAssertEqual(prepared.request.messages.filter { $0.role == .tool }.map(\.toolCallID), ["a", "b"])
        XCTAssertEqual(prepared.request.messages.last?.content, "continue")
        let invalid = source.replacingOccurrences(of: #"[{"type":"tool_result","tool_use_id":"a","content":"first"}]"#,
            with: #"[{"type":"text","text":"before results"},{"type":"tool_result","tool_use_id":"a","content":"first"}]"#)
        let bad = try JSONDecoder().decode(MessagesRequest.self, from: Data(invalid.utf8))
        let invalidRequest = try RuntimeChatRequest.messages(bad, version: .original, counting: false, continuity: MessagesContinuity(key: Data(repeating: 1, count: 32)))
        XCTAssertThrowsError(try invalidRequest.prepare(configuration))
    }
    func testTypedEventsUpdateAdmissionWithoutCountingRawDecodeTwice() {
        XCTAssertEqual(RuntimeRequestProgress(.init(stage: .generating, event: .text("42"))).message, "42")
        XCTAssertEqual(RuntimeRequestProgress(.init(stage: .generating, event: .reasoning("work"))).message, "work")
        XCTAssertNil(RuntimeRequestProgress(.init(stage: .generating, message: "raw", decodedText: "raw")).message)
    }
    func wire(_ extra: String = "") throws -> MessagesRequest {
        try JSONDecoder().decode(MessagesRequest.self, from: Data((#"{"model":"local","max_tokens":16,"messages":[{"role":"user","content":"hello"}]"# + extra + "}").utf8))
    }
    func testVersionDefaultsAndSamplingRemoval() throws {
        XCTAssertEqual(try MessagesVersion.resolve(nil), .original)
        XCTAssertThrowsError(try MessagesVersion.resolve("2024-01-01"))
        let request = try wire(#", "temperature":1, "top_p":1, "top_k":20"#)
        try request.validate(version: .original, counting: false)
        XCTAssertThrowsError(try request.validate(version: .october2026, counting: false))
        XCTAssertThrowsError(try wire(#", "temperature":null"#).validate(version: .october2026, counting: false))
    }
    func testNewVersionRejectsAssistantDocumentsAndUserCitations() throws {
        for content in [
            #"{"role":"assistant","content":[{"type":"document","source":{"type":"text","data":"page"}}]}"#,
            #"{"role":"user","content":[{"type":"text","text":"x","citations":[{"type":"char_location"}]}]}"#
        ] {
            let request = try JSONDecoder().decode(MessagesRequest.self, from: Data((#"{"model":"local","max_tokens":10,"messages":["# + content + "]}").utf8))
            try request.validate(version: .original, counting: false)
            XCTAssertThrowsError(try request.validate(version: .october2026, counting: false))
        }
    }
    func testCountingDoesNotRequireMaxTokens() throws {
        let request = try JSONDecoder().decode(MessagesRequest.self, from: Data(#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#.utf8))
        try request.validate(version: .original, counting: true)
        XCTAssertThrowsError(try request.validate(version: .original, counting: false))
    }
    func testUnknownFeaturesAndBlockTypesFailExplicitly() throws {
        XCTAssertThrowsError(try wire(#", "mcp_servers":[]"#).validate(version: .original, counting: false))
        XCTAssertThrowsError(try JSONDecoder().decode(MessagesBlock.self, from: Data(#"{"type":"web_search_tool_result"}"#.utf8)))
        XCTAssertThrowsError(try wire(#", "service_tier":"priority"# + "\"").validate(version: .original, counting: false))
    }
    func testContinuityBindsExactTextInstallationAndModel() throws {
        let local = try MessagesContinuity(key: Data(repeating: 1, count: 32))
        let foreign = try MessagesContinuity(key: Data(repeating: 2, count: 32))
        let text = "reason 🧮\n"
        let token = local.sign(thinking: text, model: "local")
        try local.verify(signature: token, thinking: text, model: "local")
        XCTAssertThrowsError(try local.verify(signature: token, thinking: text + " ", model: "local"))
        XCTAssertThrowsError(try local.verify(signature: token, thinking: text, model: "other"))
        XCTAssertThrowsError(try foreign.verify(signature: token, thinking: text, model: "local"))
        XCTAssertThrowsError(try local.verify(signature: "hosted-signature", thinking: text, model: "local"))
    }
    func testInstallationContinuityPersistsWithPrivatePermissions() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let key = directory.appendingPathComponent("reasoning.key")
        let first = try MessagesContinuity.installation(url: key)
        let token = first.sign(thinking: "exact\n", model: "local")
        try MessagesContinuity.installation(url: key).verify(signature: token, thinking: "exact\n", model: "local")
        XCTAssertEqual(try Data(contentsOf: key).count, 32)
        let permissions = try FileManager.default.attributesOfItem(atPath: key.path)[.posixPermissions] as? Int
        XCTAssertEqual(permissions, 0o600)
    }

    func testBothCredentialHeadersAndConflicts() throws {
        let key = HTTPField.Name("x-api-key")!
        try APIAuthentication.validate(expected: "test-key", headers: [key: "test-key"])
        try APIAuthentication.validate(expected: "test-key", headers: [.authorization: "Bearer test-key", key: "test-key"])
        XCTAssertThrowsError(try APIAuthentication.validate(expected: "test-key", headers: [.authorization: "Bearer other", key: "test-key"]))
        XCTAssertThrowsError(try APIAuthentication.validate(expected: "test-key", headers: [:]))
        XCTAssertThrowsError(try APIAuthentication.validate(expected: nil, headers: [.authorization: "Bearer other", key: "test-key"]))
    }
    func testQwenAndGemmaExposeNativeMessagesToolsAndStreaming() throws {
        let fixed = MessagesCapabilities.adapter(.textChatQ36, profile: .q38(contextWindow: 32768))
        XCTAssertFalse(fixed.thinking.contains("disabled"))
        XCTAssertEqual(fixed.effort, ["low", "medium", "high", "xhigh"])
        let singlePreset = MessagesCapabilities.adapter(.textChatQ36, profile: .q36(contextWindow: 32768, fixedReasoning: true))
        XCTAssertEqual(singlePreset.effort, ["high"])
        XCTAssertEqual(singlePreset.thinking_display, ["summarized"])
        for model in ["text-agent-ornith-35b-mlx-4bit", "text-chat-gemma4-turbo"] {
            let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: model))
            let engine = try XCTUnwrap(profile.servingEngine)
            let capabilities = MessagesCapabilities.adapter(engine, profile: profile)
            XCTAssertTrue(capabilities.streaming); XCTAssertTrue(capabilities.tools)
            XCTAssertTrue(capabilities.token_counting); XCTAssertTrue(capabilities.tool_search)
            XCTAssertFalse(capabilities.strict_tools); XCTAssertFalse(capabilities.exact_thinking_budget)
            let json = #"{"model":"local","max_tokens":16,"stream":true,"thinking":{"type":"adaptive"},"tools":[{"name":"read","input_schema":{"type":"object","properties":{}}}],"messages":[{"role":"user","content":"read"}]}"#
            let wire = try JSONDecoder().decode(MessagesRequest.self, from: Data(json.utf8))
            let request = try RuntimeChatRequest.messages(wire, version: .original, counting: false, continuity: MessagesContinuity(key: Data(repeating: 1, count: 32)))
            let prepared = try request.prepare(.init(modelID: model, engine: engine, settings: .init(), profile: profile, contextSize: 32768, fallbackLoraPath: nil))
            XCTAssertEqual(prepared.request.tools?.first?.name, "read")
            XCTAssertTrue(prepared.request.showThinking)
        }
    }

    func testDiscoveryPaginationPreservesOpenAIFields() throws {
        let models = APIServerContract.modelsResponse(modelIds: ["b", "a", "c"]).data
        let page = try MessagesModelPage.page(models, limit: 1, after: "a", before: nil)
        XCTAssertEqual(page.data.map(\.id), ["b"]); XCTAssertTrue(page.has_more)
        let preceding = try MessagesModelPage.page(models, limit: 1, after: nil, before: "c")
        XCTAssertEqual(preceding.data.map(\.id), ["b"]); XCTAssertTrue(preceding.has_more)
        let first = try MessagesModelPage.page(models, limit: 1, after: nil, before: preceding.first_id)
        XCTAssertEqual(first.data.map(\.id), ["a"]); XCTAssertFalse(first.has_more)
        let json = try JSONDecoder().decode(ChatJSONValue.self, from: JSONEncoder().encode(page))
        XCTAssertEqual(json.objectValue?["object"], .string("list"))
        XCTAssertEqual(json.objectValue?["data"]?.arrayValue?.first?.objectValue?["display_name"], .string("b"))
        XCTAssertThrowsError(try MessagesModelPage.page(models, limit: 0, after: nil, before: nil))
        XCTAssertThrowsError(try MessagesModelPage.page(models, limit: nil, after: "missing", before: nil))
    }
    func testReplayedSearchErrorsPreserveNamesCodesAndMessages() throws {
        let json = #"{"model":"fixture","max_tokens":16,"tools":[{"type":"tool_search_tool_regex_20251119","name":"catalog_search"}],"messages":[{"role":"user","content":"search"},{"role":"assistant","content":[{"type":"server_tool_use","id":"s1","name":"catalog_search","input":{"pattern":"("}},{"type":"tool_search_tool_result","tool_use_id":"s1","content":{"type":"tool_search_tool_result_error","error_code":"invalid_tool_input","error_message":"bad regex"}}]},{"role":"user","content":"continue"}]}"#
        let request = try JSONDecoder().decode(MessagesRequest.self, from: Data(json.utf8))
        let input = try RuntimeChatRequest.messages(request, version: .original, counting: false, continuity: MessagesContinuity(key: Data(repeating: 1, count: 32)))
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: MuseGlimmerResources.modelId))
        let prepared = try input.prepare(.init(modelID: MuseGlimmerResources.modelId, engine: .textChatMuseGlimmer, settings: .init(), profile: profile, contextSize: 32768, fallbackLoraPath: nil))
        let result = try XCTUnwrap(prepared.request.messages.first(where: { $0.role == .tool }))
        XCTAssertEqual(result.name, "catalog_search")
        XCTAssertEqual(result.toolCallID, "s1")
        XCTAssertEqual(result.toolResultIsError, true)
        XCTAssertTrue(result.content.contains("invalid_tool_input"))
        XCTAssertTrue(result.content.contains("bad regex"))
    }

    func testCacheTTLOrderingAndBreakpointLimits() throws {
        let ordered = try wire(#", "system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral","ttl":"1h"}}], "cache_control":{"type":"ephemeral","ttl":"5m"}"#)
        try ordered.validate(version: .original, counting: false)
        let reversed = try wire(#", "system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral","ttl":"5m"}}], "cache_control":{"type":"ephemeral","ttl":"1h"}"#)
        XCTAssertThrowsError(try reversed.validate(version: .original, counting: false))
        let block = #"{"type":"text","text":"s","cache_control":{"type":"ephemeral"}}"#
        let excess = try wire(#", "system":["# + Array(repeating: block, count: 5).joined(separator: ",") + "]")
        XCTAssertThrowsError(try excess.validate(version: .original, counting: false))
    }

    func testStoredHistoryWithoutBlocksStillDecodes() throws {
        let old = try JSONDecoder().decode(ChatMessage.self, from: Data(#"{"role":"user","content":"old"}"#.utf8))
        XCTAssertNil(old.contentBlocks)
        let message = ChatMessage(role: .user, blocks: [.text("first"), .image(reference: "a", identity: "a"), .text("last")])
        XCTAssertEqual(try JSONDecoder().decode(ChatMessage.self, from: JSONEncoder().encode(message)), message)
    }
    func testTextDocumentsRejectInvalidUTF8AndOversize() throws {
        XCTAssertEqual(try MediaDocumentIO.text(Data("hello\npage".utf8)).pages, ["hello\npage"])
        XCTAssertThrowsError(try MediaDocumentIO.text(Data([0xff])))
        XCTAssertThrowsError(try MediaDocumentIO.text(Data(repeating: 0, count: MediaDocumentIO.maximumBytes + 1)))
        XCTAssertThrowsError(try MediaDocumentIO.pdf(Data("invalid".utf8)))
    }
}
