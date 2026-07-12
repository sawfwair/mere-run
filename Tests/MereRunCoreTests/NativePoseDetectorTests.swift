import Foundation
import MereRunCore
import XCTest

final class NativePoseDetectorTests: XCTestCase {
    func testPosePayloadRoundTrips() throws {
        let result = NativePoseResult(
            inputPath: "/tmp/person.png",
            imageWidth: 1920,
            imageHeight: 1080,
            subjects: [
                NativePoseSubject(
                    kind: .body,
                    index: 0,
                    confidence: 0.9,
                    points: [NativePosePoint(name: "nose", x: 0.5, y: 0.75, confidence: 0.8)]
                ),
            ]
        )
        let decoded = try JSONDecoder().decode(NativePoseResult.self, from: JSONEncoder().encode(result))
        XCTAssertEqual(decoded, result)
        XCTAssertEqual(decoded.coordinateSpace, "normalized-bottom-left")
    }

    func testMissingImageFailsBeforePlatformDispatch() {
        XCTAssertThrowsError(try NativePoseDetector().detect(imageURL: URL(fileURLWithPath: "/tmp/missing-pose.png"))) { error in
            XCTAssertEqual(error as? NativePoseDetectorError, .imageNotFound("/tmp/missing-pose.png"))
        }
    }
}
