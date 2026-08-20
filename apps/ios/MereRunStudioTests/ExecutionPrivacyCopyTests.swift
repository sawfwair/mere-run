import XCTest

final class ExecutionPrivacyCopyTests: XCTestCase {
    func testHostedCopyNamesRelayBoundary() {
        XCTAssertTrue(ExecutionPrivacyCopy.chat(for: .hostedRelay).contains("relay"))
        XCTAssertTrue(ExecutionPrivacyCopy.create(for: .hostedRelay).contains("relay"))
    }

    func testOnDeviceCopyNamesSingleDeviceBoundary() {
        XCTAssertTrue(ExecutionPrivacyCopy.chat(for: .onDevice).contains("this device"))
        XCTAssertTrue(ExecutionPrivacyCopy.create(for: .onDevice).contains("this device"))
    }

    func testDirectCopyNamesNetworkBoundaryWithoutHostedRelay() {
        let copy = ExecutionPrivacyCopy.create(for: .directMachine)
        XCTAssertTrue(copy.contains("network or tailnet"))
        XCTAssertFalse(copy.contains("relay"))
    }
}
