import Foundation
import XCTest
@testable import MereRunCore

final class MessagesContentTests: XCTestCase {
    func testURIFormatValidatesClientToolURLs() throws {
        let schema: [String: ChatJSONValue] = ["type": .string("string"), "format": .string("uri")]
        try ChatToolValidation.validateSchema(schema)
        for uri in ["https://example.com/a%20b?x=1#page", "urn:example:document"] {
            try ChatToolValidation.validate(.string(uri), schema: schema)
        }
        for uri in ["/relative", "https://example.com/a b", "https://example.com/%zz"] {
            XCTAssertThrowsError(try ChatToolValidation.validate(.string(uri), schema: schema))
        }
        XCTAssertThrowsError(try ChatToolValidation.validateSchema(["format": .string("unknown")]))
    }

    func testQwenStringArgumentsPreserveCodeWhitespace() throws {
        let tool = ToolDefinition(name: "edit", description: "", parameterSchema: ["type": .string("object"),
            "properties": .object(["text": .object(["type": .string("string")])])])
        let raw = "<tool_call><function=edit><parameter=text>\n\n    return 42\n\n</parameter></function></tool_call>"
        let calls = try Q35ToolParser.validatedToolCalls(raw, tools: [tool], parallel: false)
        XCTAssertEqual(calls.first?.jsonArguments?["text"], .string("\n    return 42\n"))
        let inline = "<tool_call><function=edit><parameter=text>  42  </parameter></function></tool_call>"
        XCTAssertEqual(try Q35ToolParser.validatedToolCalls(inline, tools: [tool], parallel: false).first?.jsonArguments?["text"], .string("  42  "))
        let crlf = "<tool_call><function=edit><parameter=text>\r\n    return 42\r\n</parameter></function></tool_call>"
        XCTAssertEqual(try Q35ToolParser.validatedToolCalls(crlf, tools: [tool], parallel: false).first?.jsonArguments?["text"], .string("    return 42"))
    }
    func testGlimmerRenderingPreservesTextAndMultipleImageOrder() throws {
        let message = ChatMessage(role: .user, blocks: [.text("before"), .image(reference: "first", identity: "a"),
            .text("between"), .image(reference: "second", identity: "b"), .text("after")])
        let rendered = MuseGlimmerTokenizerAndTemplate.renderMessages([message])
        let content = try XCTUnwrap(rendered[0]["content"] as? [[String: any Sendable]])
        XCTAssertEqual(content.compactMap { $0["type"] as? String }, ["text", "image", "text", "image", "text"])
        XCTAssertEqual(content.compactMap { $0["text"] as? String }, ["before", "between", "after"])
        XCTAssertEqual(content.compactMap { $0["image_url"] as? String }, ["first", "second"])
    }

    func testDeclaredSchemaSubsetIsValidatedBeforeGeneration() throws {
        XCTAssertThrowsError(try ChatToolValidation.validate(.number(.infinity), schema: ["type": .string("number")]))
        XCTAssertThrowsError(try ChatToolValidation.validate(.string("value"), schema: ["oneOf": .array([
            .object(["$ref": .string("#/missing")]), .object(["type": .string("string")])])]))
        let schema: [String: ChatJSONValue] = ["type": .string("object"), "properties": .object([
            "path": .object(["type": .array([.string("string"), .string("null")]), "minLength": .number(2)]),
            "count": .object(["anyOf": .array([.object(["type": .string("integer"), "minimum": .number(1)]), .object(["const": .null])])])
        ]), "additionalProperties": .bool(false)]
        try ChatToolValidation.validateSchema(schema)
        try ChatToolValidation.validate(.object(["path": .null, "count": .number(2)]), schema: schema)
        XCTAssertThrowsError(try ChatToolValidation.validate(.object(["path": .string("a")]), schema: schema))
        XCTAssertThrowsError(try ChatToolValidation.validate(.object(["count": .number(0)]), schema: schema))
        XCTAssertThrowsError(try ChatToolValidation.validateSchema(["type": .string("object"), "format": .string("unknown")]))
        XCTAssertThrowsError(try ChatToolValidation.validateSchema(["$ref": .string("https://example.com/schema")]))
        XCTAssertThrowsError(try ChatToolValidation.validateSchema(["pattern": .string("(")]))
        try ChatToolValidation.validate(.string("abc"), schema: ["pattern": .string("^[a-z]+$")])
        XCTAssertThrowsError(try ChatToolValidation.validate(.string("42"), schema: ["pattern": .string("^[a-z]+$")]))
    }

    func testNativeQwenAndGemmaChannelsAtEverySplit() {
        for raw in ["<think>work 🧮</think>answer <tool_call>hidden</tool_call>",
                    "<|channel>thought\nwork 🧮<channel|>answer <|tool_call>hidden<tool_call|>"] {
            for split in 0...raw.count {
                let boundary = raw.index(raw.startIndex, offsetBy: split)
                var parser = NativeChatContentStream(initialReasoning: false)
                let events = parser.append(String(raw[..<boundary])) + parser.append(String(raw[boundary...]), final: true)
                XCTAssertEqual(events.compactMap { if case .text(let value) = $0 { return value }; return nil }.joined(), "answer ")
                XCTAssertEqual(events.compactMap { if case .reasoning(let value) = $0 { return value }; return nil }.joined(), "work 🧮")
            }
        }
    }

    func testCumulativeUnicodeDecodeAndInitialReasoning() {
        let observer = NativeChatOutputObserver(initialReasoning: true)
        observer.observe(.init(stage: .generating, decodedText: "work \u{FFFD}"), handler: nil)
        observer.observe(.init(stage: .generating, decodedText: "work 🧮</think>4"), handler: nil)
        let events = observer.finish("work 🧮</think>42", handler: nil)
        XCTAssertEqual(events, [.reasoning("work 🧮"), .text("42")])
        var plain = NativeChatContentStream(initialReasoning: false, parseTools: false)
        XCTAssertEqual(plain.append("Example: <tool_call>literal</tool_call>", final: true), [.text("Example: <tool_call>literal</tool_call>")])
        let unicode = NativeChatOutputObserver(initialReasoning: false)
        unicode.observe(.init(stage: .generating, decodedText: "e"), handler: nil)
        unicode.observe(.init(stage: .generating, decodedText: "e\u{301} 👩"), handler: nil)
        let joined = unicode.finish("e\u{301} 👩‍💻", handler: nil).compactMap { if case .text(let text) = $0 { return text }; return nil }.joined()
        XCTAssertEqual(Array(joined.utf8), Array("e\u{301} 👩‍💻".utf8))
    }

    func testNativeToolParsersPreserveNestedJSONAndRejectTruncation() throws {
        let schema: [String: ChatJSONValue] = ["type": .string("object"), "properties": .object([
            "value": .object(["type": .string("object")]), "path": .object(["type": .string("string")])
        ]), "required": .array([.string("value"), .string("path")])]
        let tool = ToolDefinition(name: "write", description: "", parameterSchema: schema)
        let qwen = #"<tool_call><function=write><parameter=path>42</parameter><parameter=value>{"ok":true,"items":[1,null,"</tool_call>"]}</parameter></function></tool_call>"#
        let gemma = #"<|tool_call>call:write{path:<|"|>42<|"|>,value:{ok:true,items:[1,null,<|"|></tool_call><|"|>]}}<tool_call|>"#
        let qwenCalls = try Q35ToolParser.validatedToolCalls(qwen, tools: [tool], parallel: true)
        let gemmaCalls = try Gemma4ToolParser.validatedToolCalls(gemma, tools: [tool], parallel: true)
        XCTAssertEqual(qwenCalls, gemmaCalls)
        XCTAssertEqual(qwenCalls[0].jsonArguments?["path"], .string("42"))
        for malformed in [String(qwen.dropLast(8)), qwen + "extra", qwen.replacingOccurrences(of: "</function>", with: "<parameter=path>duplicate</parameter></function>")] {
            XCTAssertThrowsError(try Q35ToolParser.validatedToolCalls(malformed, tools: [tool], parallel: true))
        }
        for malformed in [String(gemma.dropLast(20)), gemma + "extra", gemma.replacingOccurrences(of: "value:{", with: "path:1,value:{")] {
            XCTAssertThrowsError(try Gemma4ToolParser.validatedToolCalls(malformed, tools: [tool], parallel: true))
        }
        XCTAssertThrowsError(try Q35ToolParser.validatedToolCalls(qwen + qwen, tools: [tool], parallel: false))
        XCTAssertThrowsError(try Gemma4ToolParser.validatedToolCalls(gemma + gemma, tools: [tool], parallel: false))
        XCTAssertThrowsError(try Q35ToolParser.validatedToolCalls(qwen, tools: [], parallel: true))
    }

    func testGlimmerStreamingWithEverySplitAndTools() {
        let raw = "to=self<|message|>work 🧮<|eom|><|start|>assistant to=user<|message|>42 <atem:function_calls><atem:invoke name=\"calc\"><atem:parameter name=\"x\">42</atem:parameter></atem:invoke></atem:function_calls><|eot|>"
        for split in 0...raw.count {
            let boundary = raw.index(raw.startIndex, offsetBy: split)
            var parser = MuseGlimmerContentStream()
            let events = parser.append(String(raw[..<boundary])) + parser.append(String(raw[boundary...]), final: true)
            let text = events.compactMap { if case .text(let text) = $0 { return text }; return nil }.joined()
            let thinking = events.compactMap { if case .reasoning(let text) = $0 { return text }; return nil }.joined()
            XCTAssertEqual(text, "42 ", "split \(split)")
            XCTAssertEqual(thinking, "work 🧮", "split \(split)")
        }
    }
    func testToolRecipientHeadersStayOutOfReasoningAndVisibleText() {
        let raw = "to=self<|message|>Use the tool.<|eom|><|start|>assistant to=multiply<|message|><atem:function_calls><atem:invoke name=\"multiply\"></atem:invoke></atem:function_calls>"
        let parsed = MuseGlimmerOutputParser.parse(raw)
        XCTAssertEqual(parsed.reasoning, "Use the tool.")
        XCTAssertEqual(parsed.visible, "")
        for split in 0...raw.count {
            var parser = MuseGlimmerContentStream()
            let boundary = raw.index(raw.startIndex, offsetBy: split)
            _ = parser.append(String(raw[..<boundary]))
            _ = parser.append(String(raw[boundary...]), final: true)
            XCTAssertEqual(parser.thinking, "Use the tool.")
            XCTAssertEqual(parser.visible, "")
        }
    }

    func testStopsDoNotLeakAcrossChunks() {
        var parser = MuseGlimmerContentStream(stops: ["STOP"])
        let events = parser.append("hello ST") + parser.append("OP ignored", final: true)
        XCTAssertEqual(events.compactMap { if case .text(let text) = $0 { return text }; return nil }.joined(), "hello ")
    }
    func testToolArgumentTypesAndMalformedCalls() throws {
        let schema: [String: ChatJSONValue] = ["type": .string("object"), "properties": .object([
            "path": .object(["$ref": .string("#/$defs/path")]), "value": .object(["type": .string("object"), "properties": .object(["ok": .object(["type": .string("boolean")])])])
        ]), "required": .array([.string("path"), .string("value")]), "$defs": .object(["path": .object(["type": .string("string")])])]
        let tool = ToolDefinition(name: "write", description: "", parameterSchema: schema)
        let raw = #"<atem:function_calls><atem:invoke name="write"><atem:parameter name="path">42</atem:parameter><atem:parameter name="value">{"ok":true}</atem:parameter></atem:invoke></atem:function_calls>"#
        let calls = try MuseGlimmerToolParser.validatedToolCalls(raw, tools: [tool], parallel: false)
        XCTAssertEqual(calls[0].jsonArguments?["path"], .string("42"))
        XCTAssertEqual(calls[0].jsonArguments?["value"], .object(["ok": .bool(true)]))
        XCTAssertThrowsError(try MuseGlimmerToolParser.validatedToolCalls(String(raw.dropLast(10)), tools: [tool], parallel: false))
        XCTAssertThrowsError(try MuseGlimmerToolParser.validatedToolCalls(raw.replacingOccurrences(of: #"{"ok":true}"#, with: "{"), tools: [tool], parallel: false))
        XCTAssertThrowsError(try MuseGlimmerToolParser.validatedToolCalls(raw + raw, tools: [tool], parallel: false))
        XCTAssertThrowsError(try MuseGlimmerToolParser.validatedToolCalls("<atem:function_calls>broken</atem:function_calls>", tools: [tool], parallel: false))
        XCTAssertThrowsError(try MuseGlimmerToolParser.validatedToolCalls("to=write<|message|>broken", tools: [tool], parallel: false))
        XCTAssertThrowsError(try MuseGlimmerToolParser.validatedToolCalls(raw.replacingOccurrences(of: "</atem:invoke>", with: "garbage</atem:invoke>"), tools: [tool], parallel: false))
    }
    func testDocumentTextProjectionPreservesMetadataAndPages() {
        let document = ChatDocument(identity: "source", title: "Document title", context: "Document context", pages: ["first", "second"])
        let message = ChatMessage(role: .user, blocks: [.text("before"), .document(document), .text("after")])
        XCTAssertTrue(message.content.contains("Document title\nDocument context"))
        XCTAssertTrue(message.content.contains("<page number=\"2\">\nsecond\n</page>"))
        XCTAssertEqual(message.contentBlocks, [.text("before"), .document(document), .text("after")])
    }

    func testNestedSchemaAndLocalReferences() throws {
        let schema: [String: ChatJSONValue] = ["type": .string("array"), "items": .object(["$ref": .string("#/$defs/item")]),
            "$defs": .object(["item": .object(["type": .string("string"), "enum": .array([.string("a"), .string("b")])])])]
        try ChatToolValidation.validate(.array([.string("a")]), schema: schema)
        XCTAssertThrowsError(try ChatToolValidation.validate(.array([.string("c")]), schema: schema))
        XCTAssertThrowsError(try ChatToolValidation.validate(.array([.number(2)]), schema: schema))
    }
}
