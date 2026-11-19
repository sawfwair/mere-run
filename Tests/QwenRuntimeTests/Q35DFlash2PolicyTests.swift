import XCTest
@testable import MereRunQwenModel
import MereRunTensor

final class Q35DFlash2PolicyTests: XCTestCase {
    private let costs = Q35DFlash2Costs(secondsByDepth: [1, 1.2, 1.3, 1.4])

    func testCalibrationRequiresExplicitOptIn() {
        let group = "MERERUN_Q38_M5_GROUP_AFFINE", calibration = "MERERUN_Q38_TREE_CALIBRATED_SCORES"
        let cases: [([String: String], Bool)] = [
            ([:], false), ([group: "all"], false), ([group: "decode"], false), ([group: "1"], false),
            ([group: "all", calibration: "0"], false), ([group: "all", calibration: "1"], true),
            ([calibration: "1"], true), ([calibration: "false"], false)
        ]
        for (environment, expected) in cases {
            XCTAssertEqual(Q35DFlash2TreePolicy.calibratedScoresDefault(environment: environment), expected)
        }
    }

    func testQualifiedDefaultsAreScopedAndReferenceOverridesTakePrecedence() {
        let baseline = Q35DFlash2TreePolicy.enabled
        QwenDFlashDefaults.$active.withValue(true) {
            if ProcessInfo.processInfo.environment["MERERUN_Q38_DFLASH2_TREE"] == nil {
                XCTAssertTrue(Q35DFlash2TreePolicy.enabled)
            }
            Q35DFlash2TreePolicy.$enabledOverride.withValue(false) {
                XCTAssertFalse(Q35DFlash2TreePolicy.enabled)
            }
            Q35PrefillProjectionPolicy.$modeOverride.withValue(.reference) {
                XCTAssertEqual(Q35PrefillProjectionPolicy.mode, .reference)
            }
        }
        XCTAssertEqual(Q35DFlash2TreePolicy.enabled, baseline)
    }

    func testHighAcceptanceKeepsWideVerification() {
        var policy = Q35DFlash2Policy(costs: costs)
        for _ in 0..<12 { policy.record(proposed: 3, accepted: 3) }
        XCTAssertEqual(policy.draftDepth(offeredDepth: 3), 3)
        XCTAssertEqual(policy.draftDepth(offeredDepth: 2), 2)
        XCTAssertEqual(policy.draftDepth(offeredDepth: 0), 0)
    }

    func testOnlyUsefulPrefixRemainsWorthDrafting() {
        var policy = Q35DFlash2Policy(costs: costs)
        for _ in 0..<4 { policy.record(proposed: 3, accepted: 1) }
        XCTAssertEqual(policy.draftDepth(offeredDepth: 3), 1)
        for _ in 0..<20 { policy.record(proposed: 1, accepted: 0) }
        XCTAssertEqual(policy.draftDepth(offeredDepth: 3), 0)
    }

    func testFallbackNeedsEvidenceAndMeasuredCost() {
        var expensive = Q35DFlash2Policy(costs: costs)
        var cheap = Q35DFlash2Policy(costs: Q35DFlash2Costs(secondsByDepth: [1, 0.5, 0.6, 0.7]))
        for _ in 0..<3 {
            expensive.record(proposed: 3, accepted: 0)
            cheap.record(proposed: 3, accepted: 0)
        }
        XCTAssertEqual(expensive.draftDepth(offeredDepth: 3), 3)
        expensive.record(proposed: 3, accepted: 0)
        cheap.record(proposed: 3, accepted: 0)
        XCTAssertEqual(expensive.draftDepth(offeredDepth: 3), 0)
        XCTAssertGreaterThan(cheap.draftDepth(offeredDepth: 3), 0)
    }

    func testFivePercentMarginAndPerRequestHistory() {
        let profile = Q35DFlash2Costs(secondsByDepth: [1, 1.9])
        var policy = Q35DFlash2Policy(costs: profile)
        for _ in 0..<4 { policy.record(proposed: 1, accepted: 1) }
        // Expected 1.96 tokens at cost 1.9 does not clear the 5% margin.
        XCTAssertEqual(policy.draftDepth(offeredDepth: 1), 0)
        XCTAssertEqual(Q35DFlash2Policy(costs: profile).draftDepth(offeredDepth: 1), 1)
    }
}
