import Foundation
import XCTest
import MereRunCore
@testable import MereRunCLI

final class MessagesConstrainedOutputTests: XCTestCase {
    private func prepare(_ options: String, version: MessagesVersion = .original, glimmer: Bool = false) throws -> ChatRequest {
        let source = #"{"model":"local","max_tokens":128,"messages":[{"role":"user","content":"answer"}]"# + options + "}"
        let wire = try JSONDecoder().decode(MessagesRequest.self, from: Data(source.utf8))
        let model = glimmer ? MuseGlimmerResources.modelId : Gemma4Resources.defaultModelId
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: model))
        let request = try RuntimeChatRequest.messages(wire, version: version, counting: false,
            continuity: MessagesContinuity(key: Data(repeating: 1, count: 32)))
        return try request.prepare(.init(modelID: model, engine: glimmer ? .textChatMuseGlimmer : .textChatGemma4,
            settings: .init(), profile: profile, contextSize: 32768, fallbackLoraPath: nil)).request
    }
    func testSchemaAndStrictToolRequestsCompileForBothVersions() throws {
        for version in MessagesVersion.allCases {
            let request = try prepare(#", "output_config":{"format":{"type":"json_schema","schema":{"type":"object","additionalProperties":false,"properties":{"ok":{"type":"boolean"}},"required":["ok"]}}}"#, version: version)
            XCTAssertEqual(request.outputConstraint?.mode, .json)
            XCTAssertTrue(request.requiresJSON)
            let tools = try prepare(#", "tools":[{"name":"edit","strict":true,"input_schema":{"type":"object","additionalProperties":false,"properties":{"replace":{"type":"boolean"}},"required":["replace"]}}],"tool_choice":{"type":"tool","name":"edit","disable_parallel_tool_use":true}"#, version: version)
            XCTAssertEqual(tools.outputConstraint?.mode, .tools)
        }
    }
    func testUnsupportedCombinationsAndSchemasFailDuringPreparation() throws {
        let format = #", "output_config":{"format":{"type":"json_schema","schema":{"type":"string"}}}"#
        for extra in [#", "thinking":{"type":"adaptive"}"#, #", "stop_sequences":["stop"]"#,
                      #", "tools":[{"name":"read","input_schema":{"type":"object"}}]"#] {
            XCTAssertThrowsError(try prepare(format + extra))
        }
        XCTAssertThrowsError(try prepare(#", "output_config":{"format":{"type":"json_schema","schema":{"type":"number","minimum":1}}}"#))
    }
    func testNativeGlimmerAcceptsAdaptiveConstraintsAndRejectsAmbiguousEncoding() throws {
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: MuseGlimmerResources.modelId))
        let capabilities = MessagesCapabilities.adapter(.textChatMuseGlimmer, profile: profile)
        XCTAssertTrue(capabilities.strict_tools); XCTAssertTrue(capabilities.json_schema)
        XCTAssertEqual(capabilities.thinking, ["adaptive"])
        for version in MessagesVersion.allCases {
            let json = try prepare(#", "thinking":{"type":"adaptive"}, "output_config":{"effort":"high","format":{"type":"json_schema","schema":{"type":"string"}}}"#, version: version, glimmer: true)
            XCTAssertEqual(json.outputConstraint?.mode, .json)
            XCTAssertTrue(json.showThinking)
            let tools = try prepare(#", "thinking":{"type":"adaptive"}, "tools":[{"name":"edit","strict":true,"input_schema":{"type":"object","additionalProperties":false,"properties":{"replace":{"type":"boolean"}},"required":["replace"]}}],"tool_choice":{"type":"any"}"#, version: version, glimmer: true)
            XCTAssertEqual(tools.outputConstraint?.mode, .tools)
            XCTAssertTrue(tools.showThinking)
        }
        XCTAssertThrowsError(try prepare(#", "tools":[{"name":"edit","strict":true,"input_schema":{"type":"object","additionalProperties":false,"properties":{"value":{"type":["string","null"]}}}}]"#, glimmer: true))
        XCTAssertThrowsError(try prepare(#", "thinking":{"type":"disabled"}, "output_config":{"format":{"type":"json_schema","schema":{"type":"string"}}}"#, glimmer: true))
    }

}
