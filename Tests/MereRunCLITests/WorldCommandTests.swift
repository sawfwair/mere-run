import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class WorldCommandTests: XCTestCase {
    func testWorldCommandExposesServe() {
        XCTAssertEqual(World.configuration.subcommands.map { $0.configuration.commandName }, ["serve"])
    }

    func testWorldServeParsesPersistentRuntimeOptions() throws {
        let command = try WorldServe.parse([
            "--host", "127.0.0.1",
            "--port", "9911",
            "--base-model", "/tmp/wan",
            "--model", "/tmp/dreamx",
            "--state-directory", "/tmp/world",
            "--prepare",
        ])
        XCTAssertEqual(command.port, 9_911)
        XCTAssertEqual(command.baseModel, "/tmp/wan")
        XCTAssertEqual(command.model, "/tmp/dreamx")
        XCTAssertEqual(command.stateDirectory, "/tmp/world")
        XCTAssertTrue(command.prepare)
    }

    func testWorldServeDefaultsToNativeManagedModels() throws {
        let command = try WorldServe.parse([])
        XCTAssertEqual(command.baseModel, Wan2Resources.modelID)
        XCTAssertEqual(command.model, Wan2DreamXCausalResources.modelID)
        XCTAssertEqual(command.port, 8_791)
    }
}
