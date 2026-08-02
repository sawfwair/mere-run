@testable import MereRunApp
import XCTest

@MainActor
final class StudioNavigationCoordinatorTests: XCTestCase {
    func testOpenLibraryItemPublishesFreshNavigationRequest() throws {
        let coordinator = StudioNavigationCoordinator()
        let id = UUID()

        coordinator.openLibraryItem(id: id, mode: .createImage)
        let first = try XCTUnwrap(coordinator.libraryRequest)
        coordinator.openLibraryItem(id: id, mode: .createImage)
        let second = try XCTUnwrap(coordinator.libraryRequest)

        XCTAssertEqual(first.itemID, id)
        XCTAssertEqual(first.mode, .createImage)
        XCTAssertNotEqual(first.token, second.token)
    }
}
