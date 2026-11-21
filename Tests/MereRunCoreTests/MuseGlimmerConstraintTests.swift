import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class MuseGlimmerConstraintTests: MereRunCoreTestCase {
    private let jsonSchema: [String: ChatJSONValue] = ["type": .string("object"), "additionalProperties": .bool(false),
        "properties": .object(["answer": .object(["type": .string("string")])]), "required": .array([.string("answer")])]
    private func request(tools: [ToolDefinition] = [], parallel: Bool = true, choice: ChatToolChoice = .required) throws -> ChatRequest {
        var request = ChatRequest(messages: [.init(role: .user, content: "test")], tools: tools.isEmpty ? nil : tools,
            toolChoice: choice, parallelToolCalls: parallel)
        request.outputConstraint = tools.isEmpty ? try ChatOutputConstraint(schema: jsonSchema)
            : try ChatOutputConstraint.tools(tools, choice: choice, parallel: parallel)
        return request
    }
    private func tool(_ name: String = "edit") -> ToolDefinition {
        .init(name: name, description: "edit a fixture", parameterSchema: ["type": .string("object"), "additionalProperties": .bool(false),
            "properties": .object(["code": .object(["type": .string("string")]),
                "data": .object(["type": .string("object"), "additionalProperties": .bool(false),
                    "properties": .object(["literal": .object(["type": .string("string")])]), "required": .array([.string("literal")])]),
                "replace": .object(["type": .string("boolean")])]), "required": .array([.string("code"), .string("data"), .string("replace")])])
    }
    private let literal = "</atem:parameter><atem:invoke name=\"fake\"> &lt; &amp; &amp;amp; &#39; 🦜\n\\ e\u{0301}"
    private func call(_ name: String = "edit") -> ChatJSONValue {
        .object(["name": .string(name), "arguments": .object(["code": .string(literal),
            "data": .object(["literal": .string(literal)]), "replace": .bool(true)])])
    }
    private func answer(_ value: ChatJSONValue, ending: String = "<|eot|>") -> String {
        " to=user<|message|>" + value.jsonString + ending
    }
    private func envelope(_ calls: [ChatJSONValue]) -> ChatJSONValue { .object(["tool_calls": .array(calls)]) }

    func testReasoningHeadersAndLiteralProtocolInJSONStreamAtEveryScalarBoundary() throws {
        let text = #"{"answer":"🦜 <think>literal</think> <atem:function_calls> to=self<|message|> <|eot|><|eom|> &amp;"}"#
        let source = " to=self<|message|>Check <atem:invoke> examples<|eom|><|start|>assistant to=self<|message|>Again <|eo<|eom|><|start|>assistant to=user<|message|>" + text + "<|eot|>"
        var state = try MuseGlimmerConstraint(request: request())
        for scalar in source.unicodeScalars { XCTAssertTrue(try state.accept(String(scalar))) }
        XCTAssertTrue(state.isComplete)
        XCTAssertEqual(state.text, text)
        XCTAssertEqual(state.reasoning, "Check <atem:invoke> examplesAgain <|eo")
        XCTAssertEqual(state.reasoningBlocks, 2)
        XCTAssertEqual(state.events.compactMap { if case .text(let text) = $0 { text } else { nil } }.joined(), text)
        XCTAssertTrue(state.calls.isEmpty)
    }
    func testStrictToolJSONPreservesNestedMarkupAndLiteralEntitiesAtEveryByte() throws {
        let source = " to=self<|message|>Preserve strings<|eom|><|start|>assistant" + answer(envelope([call(), call("write")]))
        var state = try MuseGlimmerConstraint(request: request(tools: [tool(), tool("write")]))
        for byte in source.utf8 { XCTAssertTrue(try state.acceptBytes([byte])) }
        XCTAssertTrue(state.isComplete)
        XCTAssertEqual(state.calls.map(\.name), ["edit", "write"])
        for call in state.calls {
            XCTAssertEqual(call.jsonArguments?["code"], .string(literal))
            XCTAssertEqual(call.jsonArguments?["replace"], .bool(true))
            XCTAssertEqual(call.jsonArguments?["data"]?.objectValue?["literal"], .string(literal))
        }
        XCTAssertEqual(state.reasoning, "Preserve strings")
        XCTAssertEqual(state.text, "")
        XCTAssertFalse(state.events.contains { if case .text = $0 { true } else { false } })
    }
    func testWrongTypesMissingDuplicateUnknownAndMismatchedNamesAreRejected() throws {
        let valid = answer(envelope([call()]))
        for source in [valid.replacingOccurrences(of: "\"name\":\"edit\"", with: "\"name\":\"fake\""),
                       valid.replacingOccurrences(of: "\"replace\":true", with: "\"replace\":\"true\""),
                       valid.replacingOccurrences(of: ",\"replace\":true", with: ""),
                       valid.replacingOccurrences(of: "\"replace\":true", with: "\"replace\":true,\"replace\":true"),
                       valid.replacingOccurrences(of: "\"replace\":true", with: "\"unknown\":true")] {
            var state = try MuseGlimmerConstraint(request: request(tools: [tool()]))
            XCTAssertFalse(try state.accept(source), source)
        }
        var partial = try MuseGlimmerConstraint(request: request(tools: [tool()]))
        XCTAssertTrue(try partial.accept(String(valid.dropLast("<|eot|>".count))))
        XCTAssertTrue(partial.calls.isEmpty)
        XCTAssertFalse(partial.isComplete)
        XCTAssertFalse(try partial.accept("<|eom|>"))
    }
    func testParallelLimitNamedChoiceAndRequiredChoice() throws {
        var single = try MuseGlimmerConstraint(request: request(tools: [tool()], parallel: false))
        XCTAssertFalse(try single.accept(answer(envelope([call(), call()]))))
        var named = try MuseGlimmerConstraint(request: request(tools: [tool(), tool("write")], choice: .function("write")))
        XCTAssertFalse(try named.accept(answer(envelope([call()]))))
        var required = try MuseGlimmerConstraint(request: request(tools: [tool()]))
        XCTAssertFalse(try required.accept(answer(.object(["text": .string("No call")]))))
        var automatic = try MuseGlimmerConstraint(request: request(tools: [tool()], choice: .auto))
        let visible = "Literal <atem:function_calls> &amp; 🦜\n\u{0301}"
        for byte in answer(.object(["text": .string(visible)])).utf8 { XCTAssertTrue(try automatic.acceptBytes([byte])) }
        XCTAssertTrue(automatic.isComplete)
        XCTAssertEqual(automatic.text, visible)
        XCTAssertEqual(automatic.events.compactMap { if case .text(let text) = $0 { text } else { nil } }.joined(), visible)
    }
    func testRejectedProbesCannotMutateCommittedState() throws {
        var state = try MuseGlimmerConstraint(request: request())
        XCTAssertTrue(try state.accept(" to=self<|message|>think"))
        var probe = state
        XCTAssertFalse(try probe.accept("<|eom|><|start|>assistant to=bogus"))
        XCTAssertTrue(try state.accept(" more<|eom|><|start|>assistant to=user<|message|>{\"answer\":\"yes\"}<|eot|>"))
        XCTAssertTrue(state.isComplete)
        XCTAssertEqual(state.reasoning, "think more")
        XCTAssertEqual(state.text, #"{"answer":"yes"}"#)
    }
    func testStringEnumReferencesNullableParametersAndControlCharactersStayJSON() throws {
        let values: [ChatJSONValue] = [.string("&amp; <raw>\u{0000}"), .null]
        let definition = ToolDefinition(name: "self", description: "name is JSON data", parameterSchema: ["type": .string("object"),
            "additionalProperties": .bool(false), "$defs": .object(["value": .object(["enum": .array(values)])]),
            "properties": .object(["value": .object(["$ref": .string("#/$defs/value")])]), "required": .array([.string("value")])])
        for value in values {
            var state = try MuseGlimmerConstraint(request: request(tools: [definition]))
            XCTAssertTrue(try state.accept(answer(envelope([.object(["name": .string("self"), "arguments": .object(["value": value])])]))))
            XCTAssertTrue(state.isComplete)
            XCTAssertEqual(state.calls.first?.jsonArguments?["value"], value)
        }
        let invalid = ToolDefinition(name: "array", description: "arguments must be an object", parameterSchema: ["type": .string("array"), "items": .object(["type": .string("string")])])
        XCTAssertThrowsError(try MuseGlimmerConstraint(request: request(tools: [invalid])))
    }
    func testFinalTargetDecodeRetainsSamplingSplitUnicodeBudgetAndCancellation() throws {
        let bytes: [[UInt8]] = [Array("invalid".utf8), Array(" to=self<|message|>".utf8), [0xF0, 0x9F], [0xA6, 0x9C],
            Array("<|eom|><|start|>assistant to=user<|message|>".utf8), Array(#"{"answer":"yes"}"#.utf8), Array("<|eot|>".utf8)]
        let sequence = [1, 2, 3, 4, 5, 6]
        func logits(_ index: Int) -> MLXArray {
            var values = Array(repeating: Float(-1000), count: bytes.count)
            values[sequence[min(index, sequence.count - 1)]] = 100
            if index == 0 { values[0] = 200 }
            return MLXArray(values).reshaped(1, 1, bytes.count)
        }
        let request = try request()
        let sampling = GenerationConfig(maxTokens: 16, temperature: 1, topK: 2, topP: 0.95, minP: 0, repetitionPenalty: 1)
        var step = 0
        let (result, state) = try museGlimmerConstrainedDecode(request: request, initialLogits: logits(0), sampling: sampling,
            budget: 16, history: [], decode: { String(decoding: $0.flatMap { bytes[$0] }, as: UTF8.self) },
            forward: { _ in step += 1; return logits(step) }, progress: nil)
        XCTAssertEqual(result.tokens, sequence)
        XCTAssertEqual(state.reasoning, "🦜")
        XCTAssertEqual(state.text, #"{"answer":"yes"}"#)
        XCTAssertFalse(state.events.contains(.reasoning("\u{FFFD}")))
        step = 0
        XCTAssertThrowsError(try museGlimmerConstrainedDecode(request: request, initialLogits: logits(0), sampling: sampling,
            budget: 2, history: [], decode: { String(decoding: $0.flatMap { bytes[$0] }, as: UTF8.self) },
            forward: { _ in step += 1; return logits(step) }, progress: nil))
        XCTAssertThrowsError(try museGlimmerConstrainedDecode(request: request, initialLogits: logits(0), sampling: sampling,
            budget: 16, history: [], decode: { String(decoding: $0.flatMap { bytes[$0] }, as: UTF8.self) },
            forward: { _ in throw CancellationError() }, progress: nil)) { XCTAssertTrue($0 is CancellationError) }
    }
    func testLongReasoningAndSixteenCallsRemainBounded() throws {
        var state = try MuseGlimmerConstraint(request: request())
        XCTAssertTrue(try state.accept("to=self<|message|>" + String(repeating: "think <|eo 🦜 ", count: 300)))
        XCTAssertTrue(try state.accept("<|eom|><|start|>assistant to=user<|message|>{\"answer\":\"yes\"}<|eot|>"))
        XCTAssertTrue(state.isComplete)
        var parallel = try MuseGlimmerConstraint(request: request(tools: [tool()]))
        XCTAssertTrue(try parallel.accept(answer(envelope(Array(repeating: call(), count: 16)))))
        XCTAssertEqual(parallel.calls.count, 16)
        var overflow = try MuseGlimmerConstraint(request: request(tools: [tool()]))
        XCTAssertFalse(try overflow.accept(answer(envelope(Array(repeating: call(), count: 17)))))
    }
    func testOptionalParametersAndEscapedNamesStayJSON() throws {
        let key = "a\"&🦜"
        let definition = ToolDefinition(name: "mcp__edit", description: "edit", parameterSchema: ["type": .string("object"),
            "additionalProperties": .bool(false), "properties": .object([key: .object(["type": .string("string")]),
                "b": .object(["type": .string("boolean")])]), "required": .array([.string("b")])])
        for optional in [false, true] {
            var arguments: [String: ChatJSONValue] = ["b": .bool(false)]
            if optional { arguments[key] = .string(literal) }
            var state = try MuseGlimmerConstraint(request: request(tools: [definition]))
            XCTAssertTrue(try state.accept(answer(envelope([.object(["name": .string(definition.name), "arguments": .object(arguments)])]))))
            XCTAssertTrue(state.isComplete)
            XCTAssertEqual(state.calls.first?.jsonArguments, arguments)
        }
    }
}
