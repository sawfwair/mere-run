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
}
