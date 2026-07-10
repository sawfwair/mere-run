@testable import MereRunApp
import XCTest

final class StudioLayoutTests: XCTestCase {
    func testLayoutBecomesCompactBeforeRegularColumnsWouldCollide() {
        XCTAssertEqual(
            StudioLayoutPolicy.layoutClass(for: StudioLayoutPolicy.compactBreakpoint - 1),
            .compact
        )
        XCTAssertEqual(
            StudioLayoutPolicy.layoutClass(for: StudioLayoutPolicy.compactBreakpoint),
            .regular
        )
    }

    func testCompactPanelRespectsWindowInsetsAndPreferredWidth() {
        XCTAssertEqual(
            StudioLayoutPolicy.compactPanelWidth(availableWidth: 480, preferredWidth: 560),
            456
        )
        XCTAssertEqual(
            StudioLayoutPolicy.compactPanelWidth(availableWidth: 900, preferredWidth: 320),
            320
        )
    }
}
