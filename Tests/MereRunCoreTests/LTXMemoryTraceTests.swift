import XCTest
@testable import MereRunCore

final class LTXMemoryTraceTests: XCTestCase {
    func testMemoryTraceLineReportsPhaseAndMLXResidency() {
        let line = ltxMemoryTraceLine(
            phase: "text-context-ready",
            activeBytes: 3 * 1_073_741_824,
            cacheBytes: 536_870_912,
            peakBytes: 7 * 1_073_741_824
        )
        XCTAssertEqual(
            line,
            "[ltx-memory] phase=text-context-ready"
                + " active=3.00GiB cache=0.50GiB peak=7.00GiB"
        )
    }
}
