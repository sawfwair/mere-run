import XCTest
@testable import MereRunQwenModel

final class Q35TreeThinkingLimitTests: XCTestCase {
    func testQualifiedModeAndExplicitOptOut() {
        XCTAssertEqual(Q35SampledTreeMode.resolve(automatic: true, environment: [:]), .vocabulary)
        XCTAssertNil(Q35SampledTreeMode.resolve(automatic: false, environment: [:]))
        XCTAssertNil(Q35SampledTreeMode.resolve(automatic: true, environment: ["MERERUN_Q38_SAMPLED_TREE": "0"]))
        XCTAssertNil(Q35SampledTreeMode.resolve(automatic: true, environment: ["MERERUN_Q38_SAMPLED_CHAIN": "1"]))
        XCTAssertEqual(Q35SampledTreeMode.resolve(automatic: true, environment: ["MERERUN_Q38_SAMPLED_TREE": "1"]), .position)
    }

    func testAutomaticLimitHonorsRequestedAndAvailableOutputBudgets() throws {
        func resolve(_ requested: Int, _ available: Int, automatic: Bool = true,
                     mode: Q35SampledTreeMode? = .vocabulary, override: String? = nil) -> Q35TreeThinkingLimit? {
            Q35TreeThinkingLimit.configured(tokenBudget: available, requestedTokens: requested, automatic: automatic,
                mode: mode, environment: override.map { ["MERERUN_Q38_THINKING_TOKEN_LIMIT": $0] } ?? [:],
                encode: { $0 == "</think>" ? [99] : [20] })
        }
        XCTAssertEqual(try XCTUnwrap(resolve(16_384, 16_384)).forcedToken(at: 14_336), 99)
        XCTAssertNil(resolve(32_768, 32_768))
        XCTAssertNil(resolve(32_768, 16_384))
        XCTAssertNil(resolve(16_384, 15_000))
        XCTAssertNil(resolve(256, 256))
        XCTAssertNil(resolve(16_384, 16_384, automatic: false))
        XCTAssertNil(resolve(16_384, 16_384, mode: nil))
        XCTAssertNil(resolve(16_384, 16_384, override: "0"))
        XCTAssertEqual(try XCTUnwrap(resolve(32_768, 32_768, override: "20000")).forcedToken(at: 20_000), 99)
    }

    func testTransitionCountsCueWithinLimitAndResumesAfterSeparator() throws {
        let limit = try XCTUnwrap(Q35TreeThinkingLimit(limit: 8, tokenBudget: 16, closingToken: 99,
                                                       cue: [20, 21], separator: [30]))
        XCTAssertEqual(limit.childBudget(remaining: 11, generated: 5), 1)
        XCTAssertNil(limit.forcedToken(at: 5))
        XCTAssertTrue(limit.finished(after: 99, at: 5))
        XCTAssertFalse(limit.finished(after: 20, at: 5))
        XCTAssertEqual((6...9).compactMap(limit.forcedToken), [20, 21, 99, 30])
        for index in 6...9 { XCTAssertEqual(limit.childBudget(remaining: 5, generated: index), 0) }
        XCTAssertFalse(limit.finished(after: 99, at: 8))
        XCTAssertTrue(limit.finished(after: 30, at: 9))
        XCTAssertNil(limit.forcedToken(at: 10))
    }

    func testSmallLimitDropsCueAndReservesAnswerSpace() throws {
        let limit = try XCTUnwrap(Q35TreeThinkingLimit(limit: 1, tokenBudget: 4, closingToken: 99,
                                                       cue: [20, 21], separator: [30]))
        XCTAssertEqual((0...3).compactMap(limit.forcedToken), [99, 30])
        XCTAssertNil(Q35TreeThinkingLimit(limit: 1, tokenBudget: 3, closingToken: 99,
                                          cue: [20, 21], separator: [30]))
    }
}
