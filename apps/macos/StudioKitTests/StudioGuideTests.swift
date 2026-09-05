import XCTest
@testable import StudioKit

final class StudioGuideTests: XCTestCase {
    func testModelGuidesKeepExactVariantIDsAndSupportSearch() throws {
        let json = """
        [{"topic":"handbook-flux2-klein","title":"FLUX.2 Klein","category":"Image",
          "models":["image-klein-max","image-klein-9b"],"resourceName":"handbook-flux2-klein.md"}]
        """
        let guide = try XCTUnwrap(StudioGuideTopic.parseModels(listJSON: json).first)
        XCTAssertTrue(guide.isModelGuide)
        XCTAssertEqual(guide.models, ["image-klein-max", "image-klein-9b"])
        XCTAssertTrue(guide.matches("KLEIN 9B"))
        XCTAssertTrue(guide.matches("image-klein-9b"))
        XCTAssertFalse(guide.matches("Klein speech"))
        XCTAssertTrue(guide.matches(""))
    }

    func testCommandGuidesRemainCompatible() throws {
        let json = """
        [{"topic":"image-generate","title":"Image Generate","commands":["image generate"]}]
        """
        let guide = try XCTUnwrap(StudioGuideTopic.parse(listJSON: json).first)
        XCTAssertFalse(guide.isModelGuide)
        XCTAssertEqual(guide.commandPath, ["image", "generate"])
        XCTAssertTrue(guide.matches("image generate"))
    }

    func testMalformedOrFailedResponsesDoNotBecomeGuideContent() {
        XCTAssertTrue(StudioGuideTopic.parseModels(listJSON: "unavailable").isEmpty)
        XCTAssertTrue(StudioGuideTopic.parseModels(listJSON: "[{}]").isEmpty)
        XCTAssertNil(StudioGuideTopic.parseContent(payloadJSON: "failed"))
    }
}
