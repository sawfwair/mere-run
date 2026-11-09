import XCTest
@testable import MereRunCore

final class MuseGlimmerChannelTests: XCTestCase {
    private let tool = ToolDefinition(name: "record", description: "", parameterSchema: [
        "type": .string("object"), "properties": .object(["value": .object(["type": .string("string")])]),
        "required": .array([.string("value")]), "additionalProperties": .bool(false)
    ])

    private func packet(_ value: String = "雪", name: String = "record") -> String {
        "<atem:function_calls><atem:invoke name=\"\(name)\"><atem:parameter name=\"value\">\(value)</atem:parameter></atem:invoke></atem:function_calls>"
    }

    func testReasoningToolExampleIsPreservedButNeverExecutableAtEverySplit() throws {
        let thought = "Plan 🧮: " + packet() + " Then execute once."
        let raw = "to=self<|message|>\(thought)<|eom|><|start|>assistant to=record<|message|>\(packet())<|eot|>"
        for split in 0...raw.count {
            let boundary = raw.index(raw.startIndex, offsetBy: split)
            var stream = MuseGlimmerContentStream()
            let events = stream.append(String(raw[..<boundary])) + stream.append(String(raw[boundary...]), final: true)
            let reasoning = events.compactMap { if case .reasoning(let value) = $0 { return value }; return nil }.joined()
            XCTAssertEqual(Array(reasoning.utf8), Array(thought.utf8), "split \(split)")
            XCTAssertEqual(stream.visible, "")
            XCTAssertEqual(MuseGlimmerToolParser.parseToolCalls(stream.toolSource).count, 1)
        }
        let calls = try MuseGlimmerToolParser.validatedToolCalls(raw, tools: [tool], parallel: false)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].jsonArguments?["value"], .string("雪"))
    }

    func testReasoningExamplesDoNotRequireDeclaredToolsOrCompleteArguments() throws {
        for thought in [packet("example", name: "not_declared"), "<atem:function_calls><atem:invoke name=\"record\">{"] {
            let raw = "to=self<|message|>\(thought)<|eom|><|start|>assistant to=record<|message|>\(packet())"
            XCTAssertEqual(try MuseGlimmerToolParser.validatedToolCalls(raw, tools: [tool], parallel: false).count, 1)
            XCTAssertEqual(MuseGlimmerOutputParser.parse(raw).reasoning, thought)
        }
        XCTAssertTrue(try MuseGlimmerToolParser.validatedToolCalls(
            "to=self<|message|><atem:function_calls>unfinished", tools: [], parallel: false).isEmpty)
    }

    func testBareRecipientExampleWithinReasoningDoesNotChangeChannels() throws {
        let thought = "The literal to=record<|message|> is an example. " + packet()
        let raw = "to=self<|message|>\(thought)<|eom|><|start|>assistant to=user<|message|>READY"
        let parsed = MuseGlimmerOutputParser.parse(raw)
        XCTAssertEqual(parsed.reasoning, thought)
        XCTAssertEqual(parsed.visible, "READY")
        XCTAssertTrue(try MuseGlimmerToolParser.validatedToolCalls(raw, tools: [tool], parallel: false).isEmpty)
    }

    func testHeaderAndEndingTextWithinToolArgumentsRemainData() throws {
        let value = "to=self<|message|>quoted<|eom|>tail"
        for raw in [packet(value), "to=record<|message|>" + packet(value)] {
            var stream = MuseGlimmerContentStream()
            for character in raw { _ = stream.append(String(character)) }
            _ = stream.append("", final: true)
            XCTAssertEqual(stream.visible, "")
            XCTAssertEqual(stream.thinking, "")
            let calls = try MuseGlimmerToolParser.validatedToolCalls(raw, tools: [tool], parallel: false)
            XCTAssertEqual(calls[0].jsonArguments?["value"], .string(value))
        }
    }

    func testRealDuplicateCallsAreRetainedAndParallelControlStillApplies() throws {
        let call = "<|start|>assistant to=record<|message|>\(packet())<|eom|>"
        XCTAssertEqual(try MuseGlimmerToolParser.validatedToolCalls(call + call, tools: [tool], parallel: true).count, 2)
        XCTAssertThrowsError(try MuseGlimmerToolParser.validatedToolCalls(call + call, tools: [tool], parallel: false))
        XCTAssertThrowsError(try MuseGlimmerToolParser.validatedToolCalls(
            "to=record<|message|>" + String(packet().dropLast(4)), tools: [tool], parallel: false))
        XCTAssertThrowsError(try MuseGlimmerToolParser.validatedToolCalls("to=record<|message|>broken", tools: [tool], parallel: false))
    }
}
