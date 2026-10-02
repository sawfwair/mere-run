import ArgumentParser
import XCTest
import MereRunCore
@testable import MereRunCLI

final class ClefCommandTests: XCTestCase {
    func testDecideParsesClefAndPreflightWithExistingOutputOptions() throws {
        let command = try TextDecide.parse(["--model", ClefCatalog.modelID, "--input", "request.json", "--preflight", "--pretty"])
        XCTAssertEqual(command.model, ClefCatalog.modelID)
        XCTAssertTrue(command.preflight)
        XCTAssertTrue(command.pretty)
        XCTAssertEqual(command.input, "request.json")
        let alias = try TextDecide.parse(["-m", ClefCatalog.repository, "-i", "-", "-o", "decisions.json"])
        XCTAssertEqual(alias.model, ClefCatalog.repository)
        XCTAssertEqual(alias.output, "decisions.json")
    }

    func testManagedClefHasExplicitInstalledRuntimeSmoke() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ClefCatalog.modelID))
        XCTAssertNotNil(InstalledModelSmokePlans.plan(for: spec, installedIDs: [ClefCatalog.modelID]))
    }
}
