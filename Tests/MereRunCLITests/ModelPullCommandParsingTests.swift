import XCTest
@testable import MereRunCLI

final class ModelPullCommandParsingTests: XCTestCase {
    func testModelPullParsesHardwareOverride() throws {
        let cmd = try ModelPull.parse([
            "text-code-qwen3",
            "--allow-unsupported",
        ])

        XCTAssertEqual(cmd.target, "text-code-qwen3")
        XCTAssertTrue(cmd.allowUnsupported)
    }

    func testModelCommandExposesCapabilities() {
        let commandNames = Set(Model.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("capabilities"))
    }
}
