import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class ChatSchemaGrammarTests: MereRunCoreTestCase {
    private func schema(_ text: String) throws -> [String: ChatJSONValue] {
        try JSONDecoder().decode([String: ChatJSONValue].self, from: Data(text.utf8))
    }

    func testNestedObjectsOptionalPropertiesArraysEnumsAndLocalReferences() throws {
        let schema = try schema(##"{"type":"object","additionalProperties":false,"properties":{"label":{"type":"string"},"rows":{"type":"array","minItems":1,"maxItems":2,"items":{"$ref":"#/$defs/row"}}},"required":["rows"],"$defs":{"row":{"type":"object","additionalProperties":false,"properties":{"flag":{"type":"boolean"},"value":{"enum":["é 🧮",null,42]}},"required":["value"]}}}"##)
        for text in [##"{"rows":[{"value":"é 🧮"}]}"##, ##"{"label":"a\\b","rows":[{"flag":true,"value":null},{"value":42}]}"##] {
            var grammar = try ChatSchemaGrammar(schema: schema)
            for scalar in text.unicodeScalars { XCTAssertTrue(grammar.accept(String(scalar))) }
            XCTAssertTrue(grammar.isComplete)
            try ChatToolValidation.validate(JSONDecoder().decode(ChatJSONValue.self, from: Data(text.utf8)), schema: schema)
        }
        for text in [##"{"rows":[]}"##, ##"{"rows":[{"value":false}]}"##, ##"{"rows":[{}]}"##, ##"{"unknown":1}"##,
                     ##"{"rows":[{"value":42},{"value":42},{"value":42}]}"##] {
            var grammar = try ChatSchemaGrammar(schema: schema)
            XCTAssertFalse(grammar.accept(text), text)
        }
    }

    func testUnsupportedConstraintsFailAtCompilationInsteadOfBeingIgnored() throws {
        for source in [##"{"type":"string","pattern":"a"}"##, ##"{"type":"number","minimum":3}"##,
                       ##"{"type":"object"}"##, ##"{"$ref":"#/$defs/a","type":"string","$defs":{"a":{"type":"string"}}}"##,
                       ##"{"type":"array","items":{"type":"integer"},"maxItems":129}"##] {
            XCTAssertThrowsError(try ChatSchemaGrammar(schema: schema(source)), source)
        }
        XCTAssertThrowsError(try ChatSchemaGrammar(schema: schema(##"{"$defs":{"a":{"$ref":"#/$defs/a"}},"type":"object","additionalProperties":false,"properties":{"a":{"$ref":"#/$defs/a"}}}"##)))
    }

    func testToolEnvelopePreservesNamesTypesAndReferences() throws {
        let parameters = try schema(##"{"type":"object","additionalProperties":false,"$defs":{"entry":{"type":"array","items":{"type":["boolean","null"]}}},"properties":{"payload":{"$ref":"#/$defs/entry"}},"required":["payload"]}"##)
        let tool = ToolDefinition(name: "mcp__edit", description: "edit", parameterSchema: parameters)
        let constraint = try ChatOutputConstraint.tools([tool], choice: .function(tool.name), parallel: false)
        var request = ChatRequest(messages: [], tools: [tool])
        XCTAssertEqual(NativeChatToolPolicy.templateTools(request), [tool])
        request.outputConstraint = constraint
        XCTAssertNil(NativeChatToolPolicy.templateTools(request))
        XCTAssertTrue(NativeChatToolPolicy.requestMessages(request).first?.content.contains("edit") == true)
        let native = MuseGlimmerTokenizerAndTemplate.requestMessages(request)
        XCTAssertTrue(native.first?.content.contains("JSON escaping only") == true)
        XCTAssertTrue(native.first?.content.contains("mcp__edit") == true)
        XCTAssertFalse(native.first?.content.contains("XML-escaped") == true)
        let text = ##"{"tool_calls":[{"arguments":{"payload":[true,null]},"name":"mcp__edit"}]}"##
        var grammar = try ChatSchemaGrammar(schema: constraint.schema)
        XCTAssertTrue(grammar.accept(text)); XCTAssertTrue(grammar.isComplete)
        let response = try constraint.finish(.init(response: "", tokensGenerated: 20), raw: text, tools: [tool])
        XCTAssertEqual(response.toolCalls?.first?.name, tool.name)
        XCTAssertEqual(response.toolCalls?.first?.jsonArguments?["payload"], .array([.bool(true), .null]))
        XCTAssertThrowsError(try constraint.finish(.init(response: "", tokensGenerated: 1), raw: String(text.dropLast()), tools: [tool]))
    }

    func testToolSchemaReferenceRebasingDoesNotModifyEnumData() throws {
        let expected: ChatJSONValue = .object(["$ref": .string("#/literal-data")])
        let tool = ToolDefinition(name: "store", description: "store", parameterSchema: [
            "type": .string("object"), "additionalProperties": .bool(false),
            "properties": .object(["value": .object(["enum": .array([expected])])]), "required": .array([.string("value")])])
        let constraint = try ChatOutputConstraint.tools([tool], choice: .required, parallel: false)
        let text = ##"{"tool_calls":[{"arguments":{"value":{"$ref":"#/literal-data"}},"name":"store"}]}"##
        var grammar = try ChatSchemaGrammar(schema: constraint.schema)
        XCTAssertTrue(grammar.accept(text)); XCTAssertTrue(grammar.isComplete)
        let result = try constraint.finish(.init(response: "", tokensGenerated: 1), raw: text, tools: [tool])
        XCTAssertEqual(result.toolCalls?.first?.jsonArguments?["value"], expected)
    }

    func testEscapesLongStringsAndOldDocumentHistories() throws {
        let source = try schema(##"{"type":"string"}"##)
        for text in [##""\ud83e\uddee""##, ##""\u0041""##, "\"" + String(repeating: "a", count: 4096) + "\""] {
            var grammar = try ChatSchemaGrammar(schema: source)
            XCTAssertTrue(grammar.accept(text)); XCTAssertTrue(grammar.isComplete)
        }
        for text in [##""\ud800""##, ##""\udc00""##, ##""\u000z""##] {
            var grammar = try ChatSchemaGrammar(schema: source)
            XCTAssertFalse(grammar.accept(text))
        }
        let document = try JSONDecoder().decode(ChatDocument.self, from: Data(#"{"identity":"old","pages":["page"]}"#.utf8))
        XCTAssertNil(document.pageImages)
    }

    func testByteSplitUnicodeAndMarkupInsideJSONStreamAsLiteralText() throws {
        let constraint = try ChatOutputConstraint(schema: schema(#"{"type":"object","additionalProperties":false,"properties":{"value":{"type":"string"}},"required":["value"]}"#))
        let expected = #"{"value":"é 🧮 <think>literal</think>"}"#
        let bytes = Array(expected.utf8)
        let logits = MLXArray([Float(0)]).reshaped(1, 1, 1)
        let recorder = Events()
        let result = try schemaConstrainedDecode(constraint: constraint, initialLogits: logits,
            sampling: GenerationConfig(temperature: 0.7), eos: [], budget: bytes.count, history: [],
            decode: { String(decoding: bytes.prefix($0.count), as: UTF8.self) }, forward: { _ in logits },
            progress: { recorder.append($0) })
        XCTAssertEqual(result.tokens.count, bytes.count)
        XCTAssertEqual(recorder.text, expected)
    }

    private final class Events: @unchecked Sendable {
        let lock = NSLock()
        var value = ""
        var text: String { lock.withLock { value } }
        func append(_ progress: ChatProgress) {
            if case .text(let piece) = progress.event { lock.withLock { value += piece } }
        }
    }

    func testStrictAutomaticTextStreamsWithoutExposingToolEnvelopes() throws {
        let tool = ToolDefinition(name: "edit", description: "edit", parameterSchema: ["type": .string("object"), "additionalProperties": .bool(false)])
        let constraint = try ChatOutputConstraint.tools([tool], choice: .auto, parallel: false)
        XCTAssertEqual(constraint.visiblePrefix(#"{"text":"Hello"#), "Hello")
        XCTAssertEqual(constraint.visiblePrefix(#"{"text":"Hello\nworld"#), "Hello\nworld")
        XCTAssertEqual(constraint.visiblePrefix(#"{"text":"Hello \ud83e"#), "Hello ")
        XCTAssertEqual(constraint.visiblePrefix(#"{"text":"Hello \ud83e\uddee"#), "Hello 🧮")
        XCTAssertEqual(constraint.visiblePrefix(#"{"tool_calls":[{"arguments":{"text":"secret"}}]}"#), "")
    }

    func testDecoderMasksInvalidTokensAndFailsOnInsufficientBudget() throws {
        let constraint = try ChatOutputConstraint(schema: schema(##"{"enum":[{"ok":true}]}"##))
        let pieces = ["invalid", "{", "\"ok\"", ":", "true", "}"]
        let expected = [1, 2, 3, 4, 5]
        var step = 0
        func logits() -> MLXArray {
            var values = [Float](repeating: -100, count: pieces.count)
            values[0] = 100
            values[expected[min(step, expected.count - 1)]] = 90
            return MLXArray(values).reshaped(1, 1, pieces.count)
        }
        let result = try schemaConstrainedDecode(
            constraint: constraint, initialLogits: logits(), sampling: GenerationConfig(temperature: 0),
            eos: [], budget: 5, history: [], decode: { $0.map { pieces[$0] }.joined() },
            forward: { _ in step += 1; return logits() }, progress: nil)
        XCTAssertEqual(result.tokens, expected)
        step = 0
        XCTAssertThrowsError(try schemaConstrainedDecode(
            constraint: constraint, initialLogits: logits(), sampling: GenerationConfig(temperature: 0),
            eos: [], budget: 2, history: [], decode: { $0.map { pieces[$0] }.joined() },
            forward: { _ in step += 1; return logits() }, progress: nil))
    }
}
