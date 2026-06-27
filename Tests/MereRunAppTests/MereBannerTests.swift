@testable import MereRunApp
import XCTest

final class MereBannerTests: XCTestCase {
    func testSeverityVoiceOverPrefix() {
        XCTAssertEqual(MereBanner.Severity.error.voiceOverPrefix, "Error: ")
        XCTAssertEqual(MereBanner.Severity.warning.voiceOverPrefix, "Warning: ")
        XCTAssertEqual(MereBanner.Severity.info.voiceOverPrefix, "")
    }

    func testSeverityIconsAreDistinct() {
        let icons = Set([
            MereBanner.Severity.info.systemImage,
            MereBanner.Severity.warning.systemImage,
            MereBanner.Severity.error.systemImage,
        ])
        XCTAssertEqual(icons.count, 3)
    }
}
