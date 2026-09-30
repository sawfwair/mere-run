import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class APIToolSchemaTests: XCTestCase {
    func testAPIRequestPreservesCompleteToolSchemaInNativePrompt() throws {
        let data = Data(#"""
        {
          "model": "mererun-test-model",
          "messages": [{"role": "user", "content": "Edit the page."}],
          "tools": [{
            "type": "function",
            "function": {
              "name": "edit",
              "parameters": {
                "type": "object",
                "properties": {
                  "action": {"type": "string", "enum": ["replace", "append"]},
                  "edits": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}},
                    "required": ["oldText", "newText"],
                    "additionalProperties": false
                  }},
                  "limit": {"type": "integer", "minimum": 1, "maximum": 10}
                },
                "required": ["action", "edits"],
                "additionalProperties": false
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let expected = try XCTUnwrap(request.tools?.first?.function?.parameters)
        let converted = try APIServerContract.chatRequest(
            from: request, fallbackLoraPath: nil, contextSize: 4_096,
            capabilities: .localTextWithTools
        )
        let tool = try XCTUnwrap(converted.tools?.first)
        let prompt = try JSONDecoder().decode(
            OpenAIJSONValue.self, from: Data(tool.promptSchemaJSONString().utf8)
        )
        XCTAssertEqual(prompt.objectValue?["function"]?.objectValue?["parameters"], expected)
        XCTAssertEqual(tool.required, ["action", "edits"])
        XCTAssertEqual(tool.parameters["edits"]?.type, "array")
    }

    func testNonObjectToolSchemaIsRejectedInsteadOfSilentlyReplaced() {
        for parameters in [OpenAIJSONValue.string("invalid"), .array([]), .bool(true)] {
            let request = OpenAIChatRequest(
                model: "mererun-test-model",
                messages: [OpenAIChatMessage(role: "user", content: "Inspect the page.")],
                tools: [OpenAIChatTool(function: OpenAIChatToolFunction(name: "inspect", parameters: parameters))]
            )
            XCTAssertThrowsError(try APIServerContract.chatRequest(
                from: request, fallbackLoraPath: nil, contextSize: 4_096,
                capabilities: .localTextWithTools
            )) { error in
                XCTAssertTrue(error.localizedDescription.contains("function parameters must be a JSON object"))
            }
        }
    }

    func testOmittedParametersStillSupportsAZeroArgumentTool() throws {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "Inspect the page.")],
            tools: [OpenAIChatTool(function: OpenAIChatToolFunction(name: "inspect"))]
        )
        let converted = try APIServerContract.chatRequest(
            from: request, fallbackLoraPath: nil, contextSize: 4_096,
            capabilities: .localTextWithTools
        )
        XCTAssertEqual(converted.tools?.first?.parametersJSONSchema, [
            "type": .string("object"), "properties": .object([:]), "required": .array([]),
        ])
    }

    func testNullParametersKeepsTheOmittedParameterBehavior() throws {
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(#"""
        {"model":"mererun-test-model","messages":[{"role":"user","content":"Inspect."}],
         "tools":[{"type":"function","function":{"name":"inspect","parameters":null}}]}
        """#.utf8))
        let converted = try APIServerContract.chatRequest(
            from: request, fallbackLoraPath: nil, contextSize: 4_096,
            capabilities: .localTextWithTools
        )
        XCTAssertEqual(converted.tools?.first?.required, [])
        XCTAssertEqual(converted.tools?.first?.parametersJSONSchema["type"], .string("object"))
    }
}
