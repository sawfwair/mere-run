import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class TextDecideCommandTests: XCTestCase {
    private let request = #"{"state":"Delivered today.","questions":[{"id":"arrived","type":"noul","instructions":"Did it arrive?"}]}"#

    func testParsingAndDefaultModel() throws {
        let command = try TextDecide.parse(["--input", "request.json", "--output", "answers.json", "--pretty", "--preflight"])
        XCTAssertEqual(command.model, LayaCatalog.modelID)
        XCTAssertEqual(command.input, "request.json")
        XCTAssertEqual(command.output, "answers.json")
        XCTAssertTrue(command.pretty)
        XCTAssertTrue(command.preflight)
        XCTAssertEqual(try TextDecide.parse(["-m", LayaCatalog.multilingualID, "-i", "-"]).input, "-")
    }

    func testRequestValidationBeforeModelResolution() throws {
        XCTAssertEqual(try TextDecide.decodeRequest(Data(request.utf8)).questions.count, 1)
        for invalid in [
            #"{"state":"s","questions":[]}"#,
            #"{"state":"s","questions":[{"id":"q","type":"chat","instructions":"Choose"}]}"#,
            #"{"state":"s","questions":[{"id":"q","type":"choice","instructions":"Choose","criteria":["a","a"]}]}"#,
            #"{"state":"s","questions":[{"id":"q","type":"score","instructions":"Rate"}]}"#,
            request.replacingOccurrences(of: "\"state\":", with: "\"max_tokens\":0,\"state\":")
        ] {
            XCTAssertThrowsError(try TextDecide.decodeRequest(Data(invalid.utf8)))
        }
        XCTAssertThrowsError(try TextDecide.decodeRequest(Data(repeating: 32, count: 2 * 1_024 * 1_024 + 1)))
    }

    func testAPIRequestAndDiscovery() throws {
        let body = request.replacingOccurrences(of: "\"state\":", with: "\"model\":\"text-decide-laya\",\"state\":")
        let decoded = try JSONDecoder().decode(LayaAPIRequest.self, from: Data(body.utf8))
        try decoded.validate()
        XCTAssertEqual(decoded.model, LayaCatalog.modelID)
        let wrong = body.replacingOccurrences(of: LayaCatalog.modelID, with: "text-chat-other")
        XCTAssertThrowsError(try JSONDecoder().decode(LayaAPIRequest.self, from: Data(wrong.utf8)).validate())
        let installed = Set(LayaCatalog.modelIDs)
        let ids = APIServerContract.companionModelIDs(installedModelIDs: installed)
        XCTAssertTrue(installed.isSubset(of: Set(ids)))
        let absent = APIServerContract.companionModelIDs(installedModelIDs: [])
        XCTAssertTrue(installed.isDisjoint(with: absent))
    }

    func testAdmissionAndInstalledSmokeCoverage() throws {
        XCTAssertNil(CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "text", "decide", "--preflight"]))
        XCTAssertNotNil(CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "text", "decide", "--model", LayaCatalog.modelID]))
        for modelID in LayaCatalog.modelIDs {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: modelID))
            XCTAssertNotNil(InstalledModelSmokePlans.plan(for: spec, installedIDs: [modelID]))
        }
    }
}
