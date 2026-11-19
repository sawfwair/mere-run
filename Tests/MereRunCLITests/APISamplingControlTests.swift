import Foundation
import XCTest
import MereRunCore
@testable import MereRunCLI

final class APISamplingControlTests: XCTestCase {
    private let modelID = Q35Resources.ornith35BMLX4BitModelId

    private func resolve(_ request: OpenAIChatRequest) throws -> ChatRequest {
        try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: RuntimeServingEngine.textChatQ36.openAICompatibility,
            servedModelID: modelID
        )
    }

    func testCodingProfileSurvivesWireRoundTripAndReachesRuntime() throws {
        let data = Data("""
        {"model":"text-agent-ornith-35b-mlx-4bit","messages":[{"role":"user","content":"hello"}],
         "temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0,
         "presence_penalty":0,"frequency_penalty":0,"repetition_penalty":1}
        """.utf8)
        let wire = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        XCTAssertTrue(wire.unknownFields.isEmpty)
        let roundTrip = try JSONDecoder().decode(OpenAIChatRequest.self, from: JSONEncoder().encode(wire))
        let request = try resolve(roundTrip)
        XCTAssertEqual(request.temperature, 0.6)
        XCTAssertEqual(request.topP, 0.95)
        XCTAssertEqual(request.topK, 20)
        XCTAssertEqual(request.minP, 0)
        XCTAssertEqual(request.presencePenalty, 0)
        XCTAssertEqual(request.frequencyPenalty, 0)
        XCTAssertEqual(request.repetitionPenalty, 1)
    }

    func testSamplingOverridesAreIndependentAndZeroDisablesTopK() throws {
        var wire = OpenAIChatRequest(model: modelID, messages: [.init(role: "user", content: "hello")])
        wire.temperature = 0.6
        XCTAssertEqual(try resolve(wire).topK, 20)
        wire.top_p = 0.8
        wire.min_p = 0.05
        XCTAssertEqual(try resolve(wire).topK, 20)
        wire.top_k = 0
        XCTAssertEqual(try resolve(wire).topK, 0)
        wire.top_k = 7
        wire.presence_penalty = 1.5
        wire.frequency_penalty = -0.5
        wire.repetition_penalty = 1.1
        let request = try resolve(wire)
        XCTAssertEqual(request.topK, 7)
        XCTAssertEqual(request.topP, 0.8)
        XCTAssertEqual(request.minP, 0.05)
        XCTAssertEqual(request.presencePenalty, 1.5)
        XCTAssertEqual(request.frequencyPenalty, -0.5)
        XCTAssertEqual(request.repetitionPenalty, 1.1)
    }

    func testRejectsInvalidSamplerValuesBeforeGeneration() throws {
        let wire = OpenAIChatRequest(model: modelID, messages: [.init(role: "user", content: "hello")])
        var invalid = wire
        invalid.top_k = -1
        XCTAssertThrowsError(try resolve(invalid))
        for value in [-2.1, 2.1, Double.infinity, Double.nan] {
            invalid = wire
            invalid.presence_penalty = value
            XCTAssertThrowsError(try resolve(invalid))
            invalid = wire
            invalid.frequency_penalty = value
            XCTAssertThrowsError(try resolve(invalid))
        }
        for value in [0, -1, Double.infinity, Double.nan, Double.greatestFiniteMagnitude] {
            invalid = wire
            invalid.repetition_penalty = value
            XCTAssertThrowsError(try resolve(invalid))
        }
        for value in [-2.0, 2.0] {
            invalid = wire
            invalid.presence_penalty = value
            invalid.frequency_penalty = value
            XCTAssertNoThrow(try resolve(invalid))
        }
    }

    func testUnsupportedEnginesRejectActiveControls() throws {
        var request = OpenAIChatRequest(model: "test-model", messages: [.init(role: "user", content: "hello")])
        request.top_k = 20
        XCTAssertThrowsError(try APIServerContract.chatRequest(from: request, fallbackLoraPath: nil, contextSize: 4_096))
        request.top_k = 0
        request.repetition_penalty = 1.1
        XCTAssertThrowsError(try APIServerContract.chatRequest(from: request, fallbackLoraPath: nil, contextSize: 4_096))
        request.repetition_penalty = 1
        request.presence_penalty = 1.5
        XCTAssertThrowsError(try APIServerContract.chatRequest(from: request, fallbackLoraPath: nil, contextSize: 4_096))
        request.presence_penalty = 0
        XCTAssertNoThrow(try APIServerContract.chatRequest(from: request, fallbackLoraPath: nil, contextSize: 4_096))
    }

    func testCatalogQwenSeedsReachPreparedRuntimeRequests() throws {
        for id in [Q35Resources.q36NanoModelId, Q35Resources.q38TwentySevenB4BitModelId] {
            let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: id))
            let wire = OpenAIChatRequest(model: id, messages: [.init(role: "user", content: "hello")], seed: 42)
            let prepared = try RuntimeChatRequest.openAI(wire).prepare(.init(
                modelID: id, engine: .textChatQ36, settings: .init(), profile: profile,
                contextSize: 8192, fallbackLoraPath: nil
            ))
            XCTAssertTrue(profile.supportsSeed)
            XCTAssertEqual(prepared.request.seed, 42)
        }
    }

    func testQ38ThinkingOffAliasesDisableGenerationWithoutSelectingLowEffort() throws {
        let id = Q35Resources.q38TwentySevenB4BitModelId
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: id))
        let configuration = RuntimeChatConfiguration(modelID: id, engine: .textChatQ36,
            settings: .init(), profile: profile, contextSize: 8192, fallbackLoraPath: nil)
        for value in ["none", "off", " NONE "] {
            var wire = OpenAIChatRequest(model: id, messages: [.init(role: "user", content: "hello")], seed: 42,
                                        reasoning_effort: value)
            wire.temperature = 0.7
            wire.top_p = 0.8
            wire.top_k = 20
            let request = try RuntimeChatRequest.openAI(wire).prepare(configuration).request
            XCTAssertFalse(request.showThinking)
            XCTAssertNil(request.reasoningEffort)
            XCTAssertEqual(request.seed, 42)
            XCTAssertEqual(request.temperature, 0.7)
            XCTAssertEqual(request.topP, 0.8)
            XCTAssertEqual(request.topK, 20)
        }
        let omitted = OpenAIChatRequest(model: id, messages: [.init(role: "user", content: "hello")])
        XCTAssertTrue(try RuntimeChatRequest.openAI(omitted).prepare(configuration).request.showThinking)
    }

    func testQ38TemplateControlsResolveOfficialPresetsAndPreserveExplicitValues() throws {
        let id = Q35Resources.q38TwentySevenB4BitModelId
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: id))
        let configuration = RuntimeChatConfiguration(modelID: id, engine: .textChatQ36,
            settings: .init(), profile: profile, contextSize: 8192, fallbackLoraPath: nil)
        let json = #"{"model":"qwen","messages":[{"role":"user","content":"hello"}],"chat_template_kwargs":{"enable_thinking":false}}"#
        var wire = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(json.utf8))
        let encoded = try JSONEncoder().encode(wire)
        XCTAssertEqual(try JSONDecoder().decode(OpenAIChatRequest.self, from: encoded).chat_template_kwargs?.enable_thinking, false)
        let request = try RuntimeChatRequest.openAI(wire).prepare(configuration).request
        XCTAssertFalse(request.showThinking)
        XCTAssertEqual(request.temperature, 0.7)
        XCTAssertEqual(request.topP, 0.8)
        XCTAssertEqual(request.topK, 20)
        XCTAssertEqual(request.presencePenalty, 1.5)
        wire.temperature = 0
        XCTAssertEqual(try RuntimeChatRequest.openAI(wire).prepare(configuration).request.presencePenalty, 0)
        wire.presence_penalty = 0
        wire.reasoning_effort = "xhigh"
        let explicit = try RuntimeChatRequest.openAI(wire).prepare(configuration).request
        XCTAssertFalse(explicit.showThinking)
        XCTAssertNil(explicit.reasoningEffort)
        XCTAssertEqual(explicit.temperature, 0)
        XCTAssertEqual(explicit.presencePenalty, 0)
        wire.reasoning_effort = "none"
        wire.chat_template_kwargs = .init(enable_thinking: true)
        XCTAssertFalse(try RuntimeChatRequest.openAI(wire).prepare(configuration).request.showThinking)
    }

    func testSeedAndReasoningValidationStillRejectInvalidOrUnsupportedControls() throws {
        let id = Q35Resources.q38TwentySevenB4BitModelId
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: id))
        for request in [
            OpenAIChatRequest(model: id, messages: [.init(role: "user", content: "hello")], seed: -1),
            OpenAIChatRequest(model: id, messages: [.init(role: "user", content: "hello")], reasoning_effort: "invalid")
        ] {
            XCTAssertThrowsError(try APIServerContract.chatRequest(from: request, fallbackLoraPath: nil,
                contextSize: 8192, capabilities: .catalog(profile), servedModelID: id, apiProfile: profile))
        }
        let seeded = OpenAIChatRequest(model: "local", messages: [.init(role: "user", content: "hello")], seed: 42)
        XCTAssertThrowsError(try APIServerContract.chatRequest(from: seeded, fallbackLoraPath: nil, contextSize: 8192))
    }
}
