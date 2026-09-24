import Foundation
import MediaIO
import XCTest
@testable import MereRunCore

final class Q35VisionImageReferenceTests: MereRunCoreTestCase {
    func testDataImageReferenceDecodesForAPIRequests() async throws {
        let expected = try MediaImage(width: 2, height: 2, rgba8: [
            255, 0, 0, 255, 0, 255, 0, 255,
            0, 0, 255, 255, 255, 255, 255, 255
        ])
        let png = try MediaImageIO.pngData(from: expected)
        let reference = "data:image/png;base64,\(png.base64EncodedString())"

        let decoded = try await Q35Generator().loadImage(from: reference)

        XCTAssertEqual(decoded, expected)
    }

    func testMalformedDataImageReferenceFailsWithoutTreatingItAsAFilePath() async {
        do {
            _ = try await Q35Generator().loadImage(from: "data:image/png;base64,not-base64!")
            XCTFail("Expected the invalid data URL to be rejected")
        } catch {
            XCTAssertTrue(String(describing: error).contains("Invalid base64 Qwen-family image data URL"))
        }
    }
}
