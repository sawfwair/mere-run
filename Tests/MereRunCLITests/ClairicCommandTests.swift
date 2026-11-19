import ArgumentParser
import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class ClairicCommandTests: XCTestCase {
    func testExtractionRoutesBothClairicCheckpointsAndKeepsGLiNERDefault() throws {
        XCTAssertFalse(TextExtract.isClairicModel(try TextExtract.parse([]).model))
        for alias in ClairicCatalog.modelIDs + ["Cloudflare/clairic", "Cloudflare/clairic-flash"] {
            let command = try TextExtract.parse(["--model", alias, "--input", "request.json", "--preflight"])
            XCTAssertTrue(TextExtract.isClairicModel(command.model))
            XCTAssertTrue(command.preflight)
            XCTAssertNil(CLIInferenceAdmissionClassifier.request(
                arguments: ["mere.run", "text", "extract", "--model", alias, "--preflight"]))
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: alias))
            XCTAssertEqual(spec.defaultCLICommands, ["text extract"])
            XCTAssertNotNil(InstalledModelSmokePlans.plan(for: spec, installedIDs: [spec.id]))
        }
    }

    func testLocalCheckpointMarkerSelectsClairic() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertFalse(TextExtract.isClairicModel(directory.path))
        try Data("{}".utf8).write(to: directory.appending(path: "clair_head_config.json"))
        XCTAssertTrue(TextExtract.isClairicModel(directory.path))
    }
}
