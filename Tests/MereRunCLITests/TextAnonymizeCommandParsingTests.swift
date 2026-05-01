import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class TextAnonymizeCommandParsingTests: XCTestCase {
    func testTextAnonymizeParsesDefaults() throws {
        let cmd = try TextAnonymize.parse([
            "My email is alice@example.com",
        ])

        XCTAssertEqual(cmd.texts, ["My email is alice@example.com"])
        XCTAssertNil(cmd.model)
        XCTAssertNil(cmd.maxTokens)
        XCTAssertEqual(cmd.replacement, "[{label}]")
        XCTAssertFalse(cmd.json)
        XCTAssertFalse(cmd.pretty)
    }

    func testTextAnonymizeParsesOverrides() throws {
        let cmd = try TextAnonymize.parse([
            "Phone: 555-1234",
            "--model", "/tmp/privacy-filter",
            "--max-tokens", "128",
            "--replacement", "<{index}:{label}>",
            "--json",
            "--pretty",
            "--output", "/tmp/out.json",
        ])

        XCTAssertEqual(cmd.model, "/tmp/privacy-filter")
        XCTAssertEqual(cmd.maxTokens, 128)
        XCTAssertEqual(cmd.replacement, "<{index}:{label}>")
        XCTAssertTrue(cmd.json)
        XCTAssertTrue(cmd.pretty)
        XCTAssertEqual(cmd.output, "/tmp/out.json")
    }

    func testTextSubcommandsIncludeAnonymize() {
        let textNames = Set(Text.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(textNames, Set(["chat", "code", "embed", "anonymize"]))
    }

    func testPrivacyFilterManagedModelSpec() {
        let spec = ManagedModelCatalog.spec(for: OpenAIPrivacyFilterCatalog.modelId)
        XCTAssertEqual(spec?.category, .textAnonymize)
        XCTAssertEqual(spec?.upstreamRepoId, "openai/privacy-filter")
        XCTAssertEqual(spec?.defaultCLICommands, ["text anonymize"])
    }
}
