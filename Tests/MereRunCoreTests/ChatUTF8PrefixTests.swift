import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class ChatUTF8PrefixTests: MereRunCoreTestCase {
    func testValidIncompleteAndMalformedSequencesAreDistinguished() {
        XCTAssertEqual(ChatUTF8Prefix.split([0xf0, 0x9f])?.pending, [0xf0, 0x9f])
        XCTAssertEqual(ChatUTF8Prefix.split([0xf0, 0x9f, 0xa6, 0x9c])?.text, "🦜")
        for bytes: [UInt8] in [[0xff], [0x80], [0xc0, 0x80], [0xed, 0xa0], [0xe0, 0x80], [0xf4, 0x90], [0xf0, 0x9f, 0x20]] {
            XCTAssertNil(ChatUTF8Prefix.split(bytes), "\(bytes)")
        }
        XCTAssertEqual(ChatUTF8Prefix.split(Array("literal �".utf8))?.text, "literal �")
    }
    func testPartialBytesMustMatchAnActuallyReachableScalar() throws {
        var request = ChatRequest(messages: [.init(role: .user, content: "test")])
        request.outputConstraint = try ChatOutputConstraint(schema: ["const": .string("🦜")])
        var state = try MuseGlimmerConstraint(request: request)
        var wrongHeader = state
        XCTAssertFalse(try wrongHeader.acceptBytes([0xf0, 0x9f]))
        XCTAssertTrue(try state.acceptBytes(Array(" to=user<|message|>\"".utf8)))
        var wrongEmoji = state
        XCTAssertFalse(try wrongEmoji.acceptBytes([0xf0, 0x9f, 0xa7]))
        XCTAssertTrue(try state.acceptBytes([0xf0, 0x9f]))
        XCTAssertFalse(state.isComplete)
        XCTAssertTrue(try state.acceptBytes([0xa6, 0x9c]))
        XCTAssertTrue(try state.acceptBytes(Array("\"<|eot|>".utf8)))
        XCTAssertTrue(state.isComplete)
        XCTAssertEqual(state.text, "\"🦜\"")
    }
    func testReleasedByteLevelSpellingAndAddedTokensDecodeExactly() throws {
        let decoder = try XCTUnwrap(MuseGlimmerTokenBytes(data: Data(#"{"decoder":{"type":"ByteLevel"},"added_tokens":[{"id":200000}]}"#.utf8)))
        XCTAssertEqual(decoder.bytes(id: 1, spelling: "ĠhelloĊ"), Array(" hello\n".utf8))
        XCTAssertEqual(decoder.bytes(id: 2, spelling: "ðŁ¦ľ"), Array("🦜".utf8))
        XCTAssertEqual(decoder.bytes(id: 200000, spelling: "<|eot|>"), Array("<|eot|>".utf8))
        XCTAssertNil(try MuseGlimmerTokenBytes(data: Data("{}".utf8)))
    }
    func testRareLiteralFallsBackToExactMaskWithSamplingStillEnabled() throws {
        var request = ChatRequest(messages: [.init(role: .user, content: "test")])
        request.outputConstraint = try ChatOutputConstraint(schema: ["type": .string("boolean")])
        let pieces = Array(repeating: Array("!".utf8), count: 300)
            + [Array(" to=user<|message|>".utf8), Array("true".utf8), Array("<|eot|>".utf8)]
        let logits = MLXArray(Array(repeating: Float(10), count: 300) + [Float(0), 0, 0]).reshaped(1, 1, 303)
        let sampling = GenerationConfig(maxTokens: 8, temperature: 1, topK: 64, topP: 0.95, minP: 0, repetitionPenalty: 1)
        let (result, state) = try museGlimmerConstrainedDecode(request: request, initialLogits: logits,
            sampling: sampling, budget: 8, history: [], decode: { String(decoding: $0.flatMap { pieces[$0] }, as: UTF8.self) },
            tokenBytes: { pieces[$0] }, forward: { _ in logits }, progress: nil)
        XCTAssertEqual(result.tokens, [300, 301, 302])
        XCTAssertEqual(state.text, "true")
        XCTAssertTrue(state.isComplete)
    }
    func testNativeStringCapturePreservesLeadingCombiningScalars() throws {
        let value = "\u{0301} text"
        let tool = ToolDefinition(name: "record", description: "record", parameterSchema: ["type": .string("object"),
            "additionalProperties": .bool(false), "properties": .object(["value": .object(["const": .string(value)])]),
            "required": .array([.string("value")])])
        var request = ChatRequest(messages: [.init(role: .user, content: "test")], tools: [tool], toolChoice: .required)
        request.outputConstraint = try ChatOutputConstraint.tools([tool], choice: .required, parallel: false)
        var state = try MuseGlimmerConstraint(request: request)
        let body = ChatJSONValue.object(["tool_calls": .array([.object([
            "name": .string("record"), "arguments": .object(["value": .string(value)])])])]).jsonString
        let source = " to=user<|message|>" + body + "<|eot|>"
        XCTAssertTrue(try state.acceptBytes(Array(source.utf8)))
        XCTAssertEqual(Array(state.calls.first?.jsonArguments?["value"]?.stringValue?.utf8 ?? "".utf8), Array(value.utf8))
    }

}
