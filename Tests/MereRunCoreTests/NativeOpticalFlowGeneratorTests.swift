import Foundation
import MereRunCore
import XCTest

final class NativeOpticalFlowGeneratorTests: XCTestCase {
    func testResultRoundTrips() throws {
        let result = NativeOpticalFlowResult(
            fromPath: "/tmp/a.png",
            toPath: "/tmp/b.png",
            outputPath: "/tmp/a-to-b.flo",
            width: 32,
            height: 24,
            vectorCount: 768,
            accuracy: .high,
            meanMagnitude: 1.5,
            maximumMagnitude: 4.0
        )
        let decoded = try JSONDecoder().decode(NativeOpticalFlowResult.self, from: JSONEncoder().encode(result))
        XCTAssertEqual(decoded, result)
    }

    func testMissingSourceFailsBeforePlatformDispatch() {
        XCTAssertThrowsError(try NativeOpticalFlowGenerator().generate(
            from: URL(fileURLWithPath: "/tmp/missing-flow-a.png"),
            to: URL(fileURLWithPath: "/tmp/missing-flow-b.png"),
            outputURL: URL(fileURLWithPath: "/tmp/missing.flo")
        )) { error in
            XCTAssertEqual(error as? NativeOpticalFlowError, .imageNotFound("/tmp/missing-flow-a.png"))
        }
    }
}
