import XCTest
@testable import MereRunCLI
import MereRunCore

final class D1CommandTests: XCTestCase {
    func testManagedD1ParsingAndAdmission() throws {
        for model in D1Catalog.modelIDs {
            let command = try TextDecide.parse(["--model", model, "--input", "-", "--preflight"])
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: model))
            XCTAssertNotNil(InstalledModelSmokePlans.plan(for: spec, installedIDs: [model]))
            XCTAssertEqual(command.model, model)
            XCTAssertTrue(command.preflight)
            XCTAssertNil(CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "text", "decide", "--model", model, "--preflight"]))
            XCTAssertNotNil(CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "text", "decide", "--model", model]))
        }
    }
}
