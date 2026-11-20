import Foundation
import XCTest
import MereRunCore
@testable import MereRunCLI

final class MessagesConstrainedOutputTests: XCTestCase {
    private func prepare(_ options: String, version: MessagesVersion = .original) throws -> ChatRequest {
        let source = #"{"model":"local","max_tokens":128,"messages":[{"role":"user","content":"answer"}]"# + options + "}"
        let wire = try JSONDecoder().decode(MessagesRequest.self, from: Data(source.utf8))
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: Gemma4Resources.defaultModelId))
        let request = try RuntimeChatRequest.messages(wire, version: version, counting: false,
            continuity: MessagesContinuity(key: Data(repeating: 1, count: 32)))
        return try request.prepare(.init(modelID: Gemma4Resources.defaultModelId, engine: .textChatGemma4,
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
}
