@testable import StudioKit
@testable import StudioUI
import XCTest

final class StudioSpecialistPageTests: XCTestCase {
    /// `sfx clap` prints one JSON object; the score comes from its `score` field, wherever that
    /// sits in the object and whatever the log around it says.
    func testTheCLAPScoreIsReadFromTheJSONByName() {
        XCTAssertEqual(
            StudioCLAPScore.parse("Loading model 2.1 GB\n{\"prompt\":\"door slam\",\"score\":0.42,\"audio\":\"/a.wav\",\"model\":\"m\"}\n"),
            0.42
        )
        XCTAssertNil(StudioCLAPScore.parse("error: model missing after 3 tries"))
    }

    /// The checklists match what each command validates.
    func testEarthChecklistsMatchTheCommands() {
        XCTAssertEqual(StudioGeoTool.flood.tensorRequirement, .init(required: ["S2L2A", "S1RTC", "DEM"]))
        XCTAssertEqual(
            StudioGeoTool.tessera.tensorRequirement,
            .init(required: ["S2", "S2_DOY"], oneOf: ["S1_ASC + S1_ASC_DOY", "S1_DESC + S1_DESC_DOY"])
        )
        XCTAssertEqual(StudioGeoTool.olmoEarth.tensorRequirement, .init(required: ["TIMESTAMPS"], oneOf: ["S2L2A", "S1RTC", "LANDSAT"]))
    }
}
