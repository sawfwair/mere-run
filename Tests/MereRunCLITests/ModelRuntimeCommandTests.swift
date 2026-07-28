import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class ModelRuntimeCommandTests: XCTestCase {
    func testModelCommandExposesRuntimeSubcommand() {
        let commandNames = Set(Model.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("runtime"))
    }

    func testRuntimeGetParsesJSONFlag() throws {
        let cmd = try ModelRuntimeGet.parse(["text-chat-gemma4", "--json"])

        XCTAssertEqual(cmd.model, "text-chat-gemma4")
        XCTAssertTrue(cmd.json)
    }

    func testRuntimeSetParsesTypedFields() throws {
        let cmd = try ModelRuntimeSet.parse([
            "text-chat-gemma4",
            "--alias", "chat-default",
            "--pinned",
            "--ttl-seconds", "600",
            "--max-context-tokens", "8192",
            "--max-tokens", "512",
            "--temperature", "0.4",
            "--top-p", "0.8",
            "--min-p", "0.05",
            "--engine", "text-chat-gemma4",
            "--kv-cache-mode", "auto",
            "--json",
        ])

        XCTAssertEqual(cmd.model, "text-chat-gemma4")
        XCTAssertEqual(cmd.alias, "chat-default")
        XCTAssertTrue(cmd.pinned)
        XCTAssertEqual(cmd.ttlSeconds, 600)
        XCTAssertEqual(cmd.maxContextTokens, 8192)
        XCTAssertEqual(cmd.maxTokens, 512)
        XCTAssertEqual(cmd.temperature, 0.4)
        XCTAssertEqual(cmd.topP, 0.8)
        XCTAssertEqual(cmd.minP, 0.05)
        XCTAssertEqual(cmd.engine, .textChatGemma4)
        XCTAssertEqual(cmd.kvCacheMode, .auto)
        XCTAssertTrue(cmd.json)
    }

    func testRuntimeSetParsesAffineEightKVMode() throws {
        let cmd = try ModelRuntimeSet.parse([
            "text-chat-q36-nano",
            "--kv-cache-mode", "affine8",
        ])

        XCTAssertEqual(cmd.kvCacheMode, .affine8)
    }
}
