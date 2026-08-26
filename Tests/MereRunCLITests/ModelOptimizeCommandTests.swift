import XCTest
@testable import MereRunCLI

final class ModelOptimizeCommandTests: XCTestCase {
    func testModelCommandExposesOptimize() {
        let names = Set(Model.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(names.contains("optimize"))
    }

    func testOptimizeFlagsParse() throws {
        let command = try ModelOptimize.parse([
            "video-minimax-h3-fl2va-mlx",
            "--force",
            "--text-encoder-only",
            "--json",
        ])
        XCTAssertEqual(command.target, "video-minimax-h3-fl2va-mlx")
        XCTAssertTrue(command.force)
        XCTAssertTrue(command.textEncoderOnly)
        XCTAssertTrue(command.json)
    }
}
