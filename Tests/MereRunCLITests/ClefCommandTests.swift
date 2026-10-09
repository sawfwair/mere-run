import ArgumentParser
import XCTest
import MereRunCore
@testable import MereRunCLI

final class ClefCommandTests: XCTestCase {
    func testDecideParsesClefAndPreflightWithExistingOutputOptions() throws {
        for (id, repository) in [(ClefCatalog.modelID, ClefCatalog.repository), (ClefCatalog.flashModelID, ClefCatalog.flashRepository), (ClefOmniCatalog.modelID, ClefOmniCatalog.repository)] {
            let command = try TextDecide.parse(["--model", id, "--input", "request.json", "--preflight", "--pretty"])
            XCTAssertEqual(command.model, id)
            XCTAssertTrue(command.preflight)
            XCTAssertTrue(command.pretty)
            XCTAssertEqual(command.input, "request.json")
            let alias = try TextDecide.parse(["-m", repository, "-i", "-", "-o", "decisions.json"])
            XCTAssertEqual(alias.model, repository)
            XCTAssertEqual(alias.output, "decisions.json")
            XCTAssertEqual(try ModelGuideRegistry.guide(for: id).topic, "handbook-clef")
        }
    }

    func testManagedClefHasExplicitInstalledRuntimeSmoke() throws {
        for id in ClefCatalog.modelIDs {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))
            XCTAssertNotNil(InstalledModelSmokePlans.plan(for: spec, installedIDs: [id]))
        }
    }
}
