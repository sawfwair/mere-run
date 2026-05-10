import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class ImageGenerateCommandParsingTests: XCTestCase {
    func testDefaultManagedImageModelIsNano() {
        XCTAssertEqual(ImageGenerate.defaultManagedModelID, .zetaNano)
    }
}
