@testable import MereRunApp
import XCTest

final class StudioLayoutTests: XCTestCase {
    func testDefaultWindowFitsSidebarLibraryAndCanvas() {
        // 300 is the sidebar column's maximum width; the Library column and the prompt canvas
        // must still fit beside it at the default size.
        XCTAssertGreaterThanOrEqual(
            StudioLayoutPolicy.defaultWindowWidth,
            300 + StudioLayoutPolicy.libraryWidth + StudioLayoutPolicy.minimumCanvasWidth
        )
        XCTAssertEqual(StudioLayoutPolicy.defaultWindowWidth, 1_280)
        XCTAssertEqual(StudioLayoutPolicy.defaultWindowHeight, 820)
    }

    func testMinimumWindowKeepsLibraryAndCanvasUsableWithSidebarCollapsed() {
        XCTAssertEqual(
            StudioLayoutPolicy.minimumWindowWidth,
            StudioLayoutPolicy.libraryWidth + StudioLayoutPolicy.minimumCanvasWidth
        )
        XCTAssertLessThan(StudioLayoutPolicy.minimumWindowWidth, StudioLayoutPolicy.defaultWindowWidth)
        XCTAssertLessThan(StudioLayoutPolicy.minimumWindowHeight, StudioLayoutPolicy.defaultWindowHeight)
    }
}
