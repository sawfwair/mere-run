import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class ModelBenchmarkCommandTests: XCTestCase {
    func testModelCommandExposesBenchmarkSubcommand() {
        let commandNames = Set(Model.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("benchmark"))
    }

    func testGemma4KVBenchmarkParsesDefaults() throws {
        let cmd = try ModelBenchmarkGemma4KV.parse([])

        XCTAssertEqual(cmd.model, Gemma4Resources.turboModelId)
        XCTAssertEqual(cmd.promptRepeat, 220)
        XCTAssertEqual(cmd.decodeTokens, 48)
        XCTAssertEqual(cmd.temperature, 0)
        XCTAssertEqual(cmd.topP, 1)
        XCTAssertFalse(cmd.json)
    }

    func testGemma4KVBenchmarkParsesOverrides() throws {
        let cmd = try ModelBenchmarkGemma4KV.parse([
            "--model", Gemma4Resources.turboModelId,
            "--model-root", "/tmp/gemma4",
            "--prompt", "Benchmark this",
            "--decode-tokens", "32",
            "--temperature", "0.2",
            "--top-p", "0.7",
            "--json",
        ])

        XCTAssertEqual(cmd.model, Gemma4Resources.turboModelId)
        XCTAssertEqual(cmd.modelRoot, "/tmp/gemma4")
        XCTAssertEqual(cmd.prompt, "Benchmark this")
        XCTAssertEqual(cmd.decodeTokens, 32)
        XCTAssertEqual(cmd.temperature, 0.2)
        XCTAssertEqual(cmd.topP, 0.7)
        XCTAssertTrue(cmd.json)
    }
}
