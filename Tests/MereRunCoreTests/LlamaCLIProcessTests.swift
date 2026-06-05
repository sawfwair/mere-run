#if os(Linux)
import XCTest
@testable import MereRunCore

final class LlamaCLIProcessTests: XCTestCase {
    func testExtractPerformanceParsesLlamaTimingLine() {
        let stdout = """
        READY READY

        [ Prompt: 100.4 t/s | Generation: 65.1 t/s ]

        Exiting...
        """

        let performance = LlamaCLIProcess.extractPerformance(from: stdout)

        XCTAssertEqual(performance.promptTokensPerSecond, 100.4)
        XCTAssertEqual(performance.generationTokensPerSecond, 65.1)
    }

    func testExtractPerformanceUsesLastTimingLine() {
        let stdout = """
        [ Prompt: 1.0 t/s | Generation: 2.0 t/s ]
        [ Prompt: 3.5 t/s | Generation: 4.5 t/s ]
        """

        let performance = LlamaCLIProcess.extractPerformance(from: stdout)

        XCTAssertEqual(performance.promptTokensPerSecond, 3.5)
        XCTAssertEqual(performance.generationTokensPerSecond, 4.5)
    }
}
#endif
