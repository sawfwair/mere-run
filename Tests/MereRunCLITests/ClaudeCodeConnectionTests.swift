import Foundation
import XCTest
import MereRunCore
@testable import MereRunCLI

final class ClaudeCodeConnectionTests: XCTestCase {
    func testNoEffortAdapterConfiguresClientOmissionWithoutAdvertisingControl() throws {
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: "text-chat-gemma4-turbo"))
        let capabilities = MessagesCapabilities.adapter(.textChatGemma4, profile: profile)
        let model = ClaudeCodeServerModel(id: "local-gemma", display_name: nil, max_input_tokens: 65536,
            max_tokens: 4096, local_capabilities: capabilities)
        let plan = try ClaudeCodeConnection.plan(origin: URL(string: "http://127.0.0.1:8080")!, model: model, version: "2.1.316",
            settings: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString), credentialVariable: "KEY")
        XCTAssertEqual(plan.patch["env"]?.objectValue?["CLAUDE_CODE_EFFORT_LEVEL"], .string("auto"))
        XCTAssertTrue(plan.capabilities.effort.isEmpty)
        XCTAssertTrue(plan.qualification.contains { $0.contains("omits the effort field") && $0.contains("preserving adaptive thinking") })
    }

    func testEffortOmissionPreservesCapsAndRestoresCapableModelSelection() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let settings = directory.appendingPathComponent("settings.json")
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: "text-chat-gemma4-turbo"))
        let gemma = ClaudeCodeServerModel(id: "local-gemma", display_name: nil, max_input_tokens: 65536,
            max_tokens: 4096, local_capabilities: .adapter(.textChatGemma4, profile: profile))
        let origin = URL(string: "http://localhost:8080")!
        try JSONEncoder().encode(["maxEffortLevel": ChatJSONValue.string("low")]).write(to: settings)
        XCTAssertThrowsError(try ClaudeCodeConnection.plan(origin: origin, model: gemma, version: "2.1.316", settings: settings, credentialVariable: "KEY"))
        try FileManager.default.removeItem(at: settings)
        let plan = try ClaudeCodeConnection.plan(origin: origin, model: gemma, version: "2.1.316", settings: settings, credentialVariable: "KEY")
        try ClaudeCodeConnection.apply(plan, settings: settings)
        let first = try Data(contentsOf: settings)
        try ClaudeCodeConnection.apply(plan, settings: settings)
        XCTAssertEqual(try Data(contentsOf: settings), first)
        let capable = try ClaudeCodeConnection.plan(origin: origin, model: model(), version: "2.1.316", settings: settings, credentialVariable: "KEY")
        XCTAssertEqual(capable.patch["env"]?.objectValue?["CLAUDE_CODE_EFFORT_LEVEL"], .null)
        try ClaudeCodeConnection.apply(capable, settings: settings)
        let existing = try JSONDecoder().decode([String: ChatJSONValue].self, from: Data(contentsOf: settings))
        XCTAssertNil(existing["env"]?.objectValue?["CLAUDE_CODE_EFFORT_LEVEL"])
        var preferred = existing
        var env = try XCTUnwrap(preferred["env"]?.objectValue)
        env["CLAUDE_CODE_EFFORT_LEVEL"] = .string("low")
        preferred["env"] = .object(env)
        try JSONEncoder().encode(preferred).write(to: settings)
        let preserve = try ClaudeCodeConnection.plan(origin: origin, model: model(), version: "2.1.316", settings: settings, credentialVariable: "KEY")
        XCTAssertEqual(preserve.patch["env"]?.objectValue?["CLAUDE_CODE_EFFORT_LEVEL"], .string("low"))
    }

    func testAdditiveCapabilitiesAllowInspectingAnAlreadyRunningServer() throws {
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: MuseGlimmerResources.modelId))
        let current = MessagesCapabilities.adapter(.textChatMuseGlimmer, profile: profile)
        var legacy = try JSONDecoder().decode([String: ChatJSONValue].self, from: JSONEncoder().encode(current))
        for key in ["thinking_display", "prompt_cache_max_tokens", "tool_schema_subset"] { legacy.removeValue(forKey: key) }
        let decoded = try JSONDecoder().decode(MessagesCapabilities.self, from: JSONEncoder().encode(legacy))
        XCTAssertTrue(decoded.messages); XCTAssertNil(decoded.thinking_display)
        let model = ClaudeCodeServerModel(id: "local", display_name: nil, max_input_tokens: 32768, max_tokens: 4096, local_capabilities: decoded)
        let settings = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathComponent("settings.json")
        let plan = try ClaudeCodeConnection.plan(origin: URL(string: "http://127.0.0.1:8080")!, model: model, version: "2.1.316",
                                               settings: settings, credentialVariable: "KEY")
        XCTAssertEqual(plan.model, "local")
        XCTAssertTrue(plan.qualification.contains(where: { $0.contains("progress-update") }))
    }
    private func model() throws -> ClaudeCodeServerModel {
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: MuseGlimmerResources.modelId))
        let capability = MessagesCapabilities.adapter(.textChatMuseGlimmer, profile: profile)
        let json: [String: ChatJSONValue] = ["id": .string("local-glimmer"), "max_input_tokens": .number(8192),
            "max_tokens": .number(1024), "local_capabilities": try JSONDecoder().decode(ChatJSONValue.self, from: JSONEncoder().encode(capability))]
        return try JSONDecoder().decode(ClaudeCodeServerModel.self, from: JSONEncoder().encode(json))
    }
    func testCommandDefaultsToPreview() throws {
        let command = try APIConnectClaudeCode.parse(["--base-url", "http://127.0.0.1:8080", "--model", "local"])
        XCTAssertFalse(command.write); XCTAssertEqual(command.scope, "project"); XCTAssertEqual(command.apiKeyEnv, "MERERUN_API_KEY")
        XCTAssertThrowsError(try ClaudeCodeConnection.origin("http://secret@example.test"))
        XCTAssertThrowsError(try ClaudeCodeConnection.origin("http://localhost/v1"))
        XCTAssertThrowsError(try ClaudeCodeConnection.validateEnvironmentName("KEY;env"))
    }
    func testEffectiveLimitsAndLocalRoutesWithoutSecrets() throws {
        let plan = try ClaudeCodeConnection.plan(origin: URL(string: "http://127.0.0.1:8080")!, model: model(), version: "2.1.316",
                                               settings: URL(fileURLWithPath: "/tmp/project/.claude/settings.local.json"), credentialVariable: "PRIVATE_LOCAL_KEY")
        XCTAssertTrue(plan.capabilities.messages)
        XCTAssertTrue(plan.capabilities.streaming)
        XCTAssertFalse(plan.capabilities.strict_tools)
        let env = try XCTUnwrap(plan.patch["env"]?.objectValue)
        XCTAssertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], .string("8192"))
        XCTAssertEqual(env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], .string("1024"))
        XCTAssertEqual(env["ANTHROPIC_DEFAULT_HAIKU_MODEL"], .string("local-glimmer"))
        XCTAssertEqual(plan.patch["apiKeyHelper"], .string("printenv PRIVATE_LOCAL_KEY"))
        XCTAssertFalse(String(decoding: try JSONEncoder().encode(plan), as: UTF8.self).contains("DISABLE_EXPERIMENTAL"))
        let unset = try ClaudeCodeConnection.plan(origin: URL(string: "http://127.0.0.1:8080")!, model: model(), version: "2.1.316",
            settings: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString), credentialVariable: "LOCAL_KEY", credentialPresent: false)
        XCTAssertTrue(unset.qualification.contains(where: { $0.contains("LOCAL_KEY is unset") }))
    }
    func testAtomicMergePreservesMCPPermissionsAndIsIdempotent() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let settings = directory.appendingPathComponent("settings.json")
        let existing: [String: ChatJSONValue] = ["mcpServers": .object(["registered": .object(["command": .string("tool")])]),
            "permissions": .object(["allow": .array([.string("Read")])]), "env": .object(["UNRELATED": .string("preserve")])]
        try JSONEncoder().encode(existing).write(to: settings)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: settings.path)
        let plan = try ClaudeCodeConnection.plan(origin: URL(string: "http://localhost:8080")!, model: model(), version: "2.1.316", settings: settings, credentialVariable: "KEY")
        try ClaudeCodeConnection.apply(plan, settings: settings)
        let first = try Data(contentsOf: settings)
        try ClaudeCodeConnection.apply(plan, settings: settings)
        XCTAssertEqual(try Data(contentsOf: settings), first)
        XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: settings.path)[.posixPermissions] as? Int, 0o600)
        let merged = try JSONDecoder().decode([String: ChatJSONValue].self, from: first)
        XCTAssertEqual(merged["mcpServers"], existing["mcpServers"])
        XCTAssertEqual(merged["permissions"], existing["permissions"])
        XCTAssertEqual(merged["env"]?.objectValue?["UNRELATED"], .string("preserve"))
    }
    func testProviderAndCredentialOverridesFailWithRemediation() throws {
        XCTAssertThrowsError(try ClaudeCodeConnection.validateRouting(environment: ["CLAUDE_CODE_USE_BEDROCK": "1"], credentialVariable: "KEY"))
        XCTAssertThrowsError(try ClaudeCodeConnection.validateRouting(environment: ["ANTHROPIC_AUTH_TOKEN": "old", "KEY": "local"], credentialVariable: "KEY"))
        try ClaudeCodeConnection.validateRouting(environment: ["ANTHROPIC_API_KEY": "local", "ANTHROPIC_AUTH_TOKEN": "local", "KEY": "local"], credentialVariable: "KEY")
        let patch: [String: ChatJSONValue] = ["apiKeyHelper": .string("printenv KEY"), "env": .object([:])]
        XCTAssertThrowsError(try ClaudeCodeConnection.merged(["env": .object(["CLAUDE_CODE_USE_VERTEX": .string("1")])], patch: patch))
    }

    func testExistingCredentialsAndHelpersRequireExplicitResolution() throws {
        let patch: [String: ChatJSONValue] = ["apiKeyHelper": .string("printenv LOCAL_KEY"), "env": .object([:])]
        XCTAssertThrowsError(try ClaudeCodeConnection.merged(["env": .object(["ANTHROPIC_API_KEY": .string("must-never-print")])], patch: patch))
        XCTAssertThrowsError(try ClaudeCodeConnection.merged(["apiKeyHelper": .string("existing-helper")], patch: patch))
    }
}
