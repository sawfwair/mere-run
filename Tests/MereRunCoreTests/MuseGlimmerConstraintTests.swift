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
    private func packet(_ name: String = "edit") -> String {
        " to=\(name)<|message|><atem:function_calls><atem:invoke name=\"\(name)\">"
            + "<atem:parameter name=\"code\">x &lt; y &amp;amp; 🦜</atem:parameter>"
            + #"<atem:parameter name="data">{"literal":"</atem:parameter><atem:invoke name=\"fake\">"}</atem:parameter>"#
            + "<atem:parameter name=\"replace\">true</atem:parameter></atem:invoke></atem:function_calls>"
    }
    func testReasoningHeadersAndLiteralProtocolInJSONStreamAtEveryScalarBoundary() throws {
        let text = #"{"answer":"🦜 <think>literal</think> <atem:function_calls> to=self<|message|> <|eot|><|eom|> &amp;"}"#
        let source = " to=self<|message|>Check <atem:invoke> examples<|eom|><|start|>assistant to=self<|message|>Again <|eo<|eom|><|start|>assistant to=user<|message|>" + text + "<|eot|>"
        var state = try MuseGlimmerConstraint(request: request())
        for scalar in source.unicodeScalars { XCTAssertTrue(try state.accept(String(scalar))) }
        XCTAssertTrue(state.grammar.isComplete)
        XCTAssertEqual(state.text, text)
        XCTAssertEqual(state.reasoning, "Check <atem:invoke> examplesAgain <|eo")
        XCTAssertEqual(state.reasoningBlocks, 2)
        XCTAssertEqual(state.events.compactMap { if case .text(let text) = $0 { text } else { nil } }.joined(), text)
        XCTAssertEqual(state.events.compactMap { if case .reasoning(let text) = $0 { text } else { nil } }.joined(), state.reasoning)
        XCTAssertTrue(state.calls.isEmpty)
    }
    func testCompleteTypedToolsAndReasoningBetweenParallelCalls() throws {
        let source = packet() + "<|eom|><|start|>assistant to=self<|message|>Check the next edit<|eom|><|start|>assistant" + packet("write") + "<|eot|>"
        var state = try MuseGlimmerConstraint(request: request(tools: [tool(), tool("write")]))
        for scalar in source.unicodeScalars { XCTAssertTrue(try state.accept(String(scalar))) }
        XCTAssertTrue(state.grammar.isComplete)
        XCTAssertEqual(state.calls.map(\.name), ["edit", "write"])
        XCTAssertEqual(state.calls.first?.jsonArguments?["code"], .string("x < y &amp; 🦜"))
        XCTAssertEqual(state.calls.first?.jsonArguments?["replace"], .bool(true))
        XCTAssertEqual(state.calls.first?.jsonArguments?["data"]?.objectValue?["literal"], .string("</atem:parameter><atem:invoke name=\"fake\">"))
        XCTAssertEqual(state.reasoning, "Check the next edit")
        XCTAssertEqual(state.text, "")
    }
    func testWrongTypesMissingDuplicateUnknownAndMismatchedNamesAreRejected() throws {
        let valid = packet() + "<|eot|>"
        for source in [valid.replacingOccurrences(of: "name=\"edit\"", with: "name=\"fake\""),
                       valid.replacingOccurrences(of: "to=edit", with: "to=fake"),
                       valid.replacingOccurrences(of: "name=\"replace\"", with: "name=\"unknown\""),
                       valid.replacingOccurrences(of: ">true</atem:parameter>", with: ">\"true\"</atem:parameter>"),
                       valid.replacingOccurrences(of: "<atem:parameter name=\"replace\">true</atem:parameter>", with: ""),
                       valid.replacingOccurrences(of: "</atem:invoke>", with: "<atem:parameter name=\"replace\">true</atem:parameter></atem:invoke>")] {
            var state = try MuseGlimmerConstraint(request: request(tools: [tool()]))
            XCTAssertFalse(try state.accept(source), source)
        }
        var partial = try MuseGlimmerConstraint(request: request(tools: [tool()]))
        XCTAssertTrue(try partial.accept(String(valid.prefix(150))))
        XCTAssertTrue(partial.calls.isEmpty)
        XCTAssertFalse(partial.grammar.isComplete)
    }
    func testParallelLimitNamedChoiceAndRequiredChoice() throws {
        var single = try MuseGlimmerConstraint(request: request(tools: [tool()], parallel: false))
        XCTAssertFalse(try single.accept(packet() + "<|eom|><|start|>assistant" + packet() + "<|eot|>"))
        var named = try MuseGlimmerConstraint(request: request(tools: [tool(), tool("write")], choice: .function("write")))
        XCTAssertFalse(try named.accept(packet() + "<|eot|>"))
        var required = try MuseGlimmerConstraint(request: request(tools: [tool()]))
        XCTAssertFalse(try required.accept(" to=user<|message|>No call<|eot|>"))
        var automatic = try MuseGlimmerConstraint(request: request(tools: [tool()], choice: .auto))
        XCTAssertTrue(try automatic.accept(" to=user<|message|>Literal <atem:function_calls> 🦜<|eot|>"))
        XCTAssertEqual(automatic.text, "Literal <atem:function_calls> 🦜")
    }
    func testRejectedProbesCannotMutateCommittedState() throws {
        var state = try MuseGlimmerConstraint(request: request())
        XCTAssertTrue(try state.accept(" to=self<|message|>think"))
        var probe = state
        XCTAssertFalse(try probe.accept("<|eom|><|start|>assistant to=bogus"))
        XCTAssertTrue(try state.accept(" more<|eom|><|start|>assistant to=user<|message|>{\"answer\":\"yes\"}<|eot|>"))
        XCTAssertTrue(state.grammar.isComplete)
        XCTAssertEqual(state.reasoning, "think more")
        XCTAssertEqual(state.text, #"{"answer":"yes"}"#)
    }
    func testNativeStringEnumAndReferencesAreEnforced() throws {
        let definition = ToolDefinition(name: "choose", description: "choose", parameterSchema: ["type": .string("object"),
            "additionalProperties": .bool(false), "$defs": .object(["value": .object(["type": .string("string"), "enum": .array([.string("a&b"), .string("🦜")])])]),
            "properties": .object(["value": .object(["$ref": .string("#/$defs/value")])]), "required": .array([.string("value")])])
        let prefix = " to=choose<|message|><atem:function_calls><atem:invoke name=\"choose\"><atem:parameter name=\"value\">"
        var valid = try MuseGlimmerConstraint(request: request(tools: [definition]))
        XCTAssertTrue(try valid.accept(prefix + "a&amp;b</atem:parameter></atem:invoke></atem:function_calls><|eot|>"))
        XCTAssertEqual(valid.calls.first?.jsonArguments?["value"], .string("a&b"))
        var invalid = try MuseGlimmerConstraint(request: request(tools: [definition]))
        XCTAssertFalse(try invalid.accept(prefix + "other"))
    }
    func testAmbiguousNativeParameterEncodingFailsAtPreparation() throws {
        let definition = ToolDefinition(name: "union", description: "union", parameterSchema: ["type": .string("object"),
            "additionalProperties": .bool(false), "properties": .object(["value": .object(["type": .array([.string("string"), .string("null")])])])])
        XCTAssertThrowsError(try MuseGlimmerConstraint(request: request(tools: [definition])))
        for name in ["self", "user", "tool<|message|>", "tool\nname"] {
            let reserved = ToolDefinition(name: name, description: "reserved", parameterSchema: ["type": .string("object"), "additionalProperties": .bool(false)])
            XCTAssertThrowsError(try MuseGlimmerConstraint(request: request(tools: [reserved])))
        }
        let control = ToolDefinition(name: "control", description: "control", parameterSchema: ["type": .string("object"),
            "additionalProperties": .bool(false), "properties": .object(["value": .object(["const": .string("\u{0000}")])])])
        XCTAssertThrowsError(try MuseGlimmerConstraint(request: request(tools: [control])))
    }
    func testFinalTargetDecodeRetainsSamplingSplitUnicodeBudgetAndCancellation() throws {
        let bytes: [[UInt8]] = [Array("invalid".utf8), Array(" to=self<|message|>".utf8), [0xF0, 0x9F], [0xA6, 0x9C],
            Array("<|eom|><|start|>assistant to=user<|message|>".utf8), Array(#"{"answer":"yes"}"#.utf8), Array("<|eot|>".utf8)]
        let sequence = [1, 2, 3, 4, 5, 6]
        func logits(_ index: Int) -> MLXArray {
            var values = Array(repeating: Float(-1000), count: bytes.count)
            values[sequence[min(index, sequence.count - 1)]] = 100
            if index == 0 { values[0] = 200 } // Invalid high-ranked candidate is rejected.
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
        XCTAssertTrue(state.grammar.isComplete)
        var parallel = try MuseGlimmerConstraint(request: request(tools: [tool()]))
        let source = (0..<16).map { _ in packet() }.joined(separator: "<|eom|><|start|>assistant") + "<|eot|>"
        XCTAssertTrue(try parallel.accept(source))
        XCTAssertEqual(parallel.calls.count, 16)
        var overflow = try MuseGlimmerConstraint(request: request(tools: [tool()]))
        XCTAssertFalse(try overflow.accept((0..<17).map { _ in packet() }.joined(separator: "<|eom|><|start|>assistant") + "<|eot|>"))
    }

    func testOptionalParametersEscapedNamesAndStructuralWhitespace() throws {
        let name = "a\"&🦜"
        let definition = ToolDefinition(name: "mcp__edit", description: "edit", parameterSchema: ["type": .string("object"),
            "additionalProperties": .bool(false), "properties": .object([name: .object(["type": .string("string")]),
                "b": .object(["type": .string("boolean")]), "c": .object(["type": .string("string")])]), "required": .array([.string("b")])])
        for optional in ["", "<atem:parameter name=\"a&quot;&amp;🦜\">example</atem:parameter>\n"] {
            var state = try MuseGlimmerConstraint(request: request(tools: [definition]))
            XCTAssertTrue(try state.accept("to=mcp__edit<|message|>\n<atem:function_calls>\n<atem:invoke name=\"mcp__edit\">\n"
                + optional + "<atem:parameter name=\"b\">false</atem:parameter>\n</atem:invoke>\n</atem:function_calls>\n<|eot|>"))
            XCTAssertTrue(state.grammar.isComplete)
            XCTAssertEqual(state.calls.first?.jsonArguments?["b"], .bool(false))
            XCTAssertEqual(state.calls.first?.jsonArguments?[name], optional.isEmpty ? nil : .string("example"))
        }
    }

}
