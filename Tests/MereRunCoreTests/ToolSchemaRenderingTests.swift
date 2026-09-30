import Foundation
import Jinja
import XCTest
@testable import MereRunCore

final class ToolSchemaRenderingTests: XCTestCase {
    private func schema() throws -> [String: OpenAIJSONValue] {
        let json = #"""
        {
          "type": "object",
          "additionalProperties": false,
          "$defs": {"label": {"type": "string", "minLength": 1}},
          "properties": {
            "action": {"type": "string", "enum": ["navigate", "screenshot"]},
            "width": {"type": "integer", "minimum": 200, "maximum": 2000},
            "edits": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {"oldText": {"$ref": "#/$defs/label"}, "newText": {"type": "string"}},
                "required": ["oldText", "newText"],
                "additionalProperties": false
              }
            },
            "selection": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}
          },
          "required": ["action"]
        }
        """#
        return try JSONDecoder().decode([String: OpenAIJSONValue].self, from: Data(json.utf8))
    }

    func testCompleteSchemaSurvivesBothPromptEncodersAndCodable() throws {
        let expected = try schema()
        let tool = ToolDefinition(name: "browser", description: "Inspect a page.", parameterSchema: expected)
        let restored = try JSONDecoder().decode(ToolDefinition.self, from: JSONEncoder().encode(tool))
        XCTAssertEqual(restored, tool)
        XCTAssertEqual(restored.required, ["action"])
        XCTAssertEqual(restored.parameters["edits"]?.type, "array")

        let template = try Template("{{ tool | tojson }}")
        let rendered = try template.render(["tool": Value(any: restored.toToolSpec())])
        for json in [rendered, try restored.promptSchemaJSONString()] {
            let declaration = try JSONDecoder().decode(OpenAIJSONValue.self, from: Data(json.utf8))
            XCTAssertEqual(declaration.objectValue?["function"]?.objectValue?["parameters"], .object(expected))
        }
    }

    func testLegacyToolDefinitionStillDecodesAndBuildsItsSchema() throws {
        let json = #"{"name":"read","description":"Read a file.","parameters":{"path":{"type":"string","description":"File path"}},"required":["path"]}"#
        let tool = try JSONDecoder().decode(ToolDefinition.self, from: Data(json.utf8))
        XCTAssertNil(tool.parameterSchema)
        XCTAssertEqual(tool.parametersJSONSchema["required"], .array([.string("path")]))
        XCTAssertEqual(
            tool.parametersJSONSchema["properties"]?.objectValue?["path"],
            .object(["type": .string("string"), "description": .string("File path")])
        )
    }

    func testInklingDeclarationKeepsCompleteSchema() throws {
        let expected = try schema()
        let prompt = try InklingTokenizerAndTemplate.renderPrompt(
            messages: [],
            tools: [ToolDefinition(name: "browser", description: "Inspect a page.", parameterSchema: expected)],
            addGenerationPrompt: false,
            reasoningEffort: 0
        )
        let start = try XCTUnwrap(prompt.range(of: "<|content_xml|>")?.upperBound)
        let end = try XCTUnwrap(prompt.range(of: "<|end_message|>", range: start..<prompt.endIndex)?.lowerBound)
        let declarations = try JSONDecoder().decode([OpenAIJSONValue].self, from: Data(prompt[start..<end].utf8))
        XCTAssertEqual(declarations.first?.objectValue?["parameters"], .object(expected))
    }

    func testDatasetFingerprintIncludesNestedSchemaConstraints() throws {
        let first = try schema()
        var second = first
        second["additionalProperties"] = .bool(true)
        func example(_ schema: [String: OpenAIJSONValue]) -> TextSFTExample {
            TextSFTExample(
                id: "schema-test", sources: ["test"],
                messages: [ChatMessage(role: .user, content: "Inspect the page."),
                           ChatMessage(role: .assistant, content: "Ready.")],
                tools: [ToolDefinition(name: "browser", description: "Inspect a page.", parameterSchema: schema)]
            )
        }
        XCTAssertNotEqual(TextSFTDataset.fingerprint([example(first)]), TextSFTDataset.fingerprint([example(second)]))
    }
}
