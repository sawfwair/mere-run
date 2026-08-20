import XCTest
@testable import MereRunApp

final class MereRunUpdaterTests: XCTestCase {
    func testStableUpdateConfigurationIsPinnedToHTTPSAndEd25519() throws {
        let feedURL = MereRunUpdateConfiguration.feedURL

        XCTAssertEqual(feedURL.scheme, "https")
        XCTAssertEqual(feedURL.host, "mere.run")
        XCTAssertEqual(feedURL.path, "/releases/appcast.xml")
        XCTAssertEqual(
            Data(base64Encoded: MereRunUpdateConfiguration.publicEDKey)?.count,
            32
        )
    }
}
