import XCTest
@testable import MereRunCore

final class Q35MTPPolicyCostTests: XCTestCase {
    func testHigherDraftCostReducesDepthForSameObservedAcceptance() {
        var cheap = Q35MTPAdaptivePolicy(maxDraftDepth: 7, headStepCostRatio: 0.05)
        var expensive = Q35MTPAdaptivePolicy(maxDraftDepth: 7, headStepCostRatio: 0.8)
        for _ in 0..<8 {
            cheap.record(acceptedDrafts: 1, drafted: 7)
            expensive.record(acceptedDrafts: 1, drafted: 7)
        }
        XCTAssertGreaterThan(cheap.draftDepth(offeredDepth: 7), expensive.draftDepth(offeredDepth: 7))
        XCTAssertEqual(cheap.draftDepth(offeredDepth: 1), 1)
    }

    func testCostOverrideRejectsNonFiniteAndInvalidValues() {
        for value in ["nan", "inf", "0", "-1", "6", "invalid"] {
            XCTAssertEqual(Q35MTPAdaptivePolicy.configuredCostRatio(environment: [
                "MERERUN_Q35_MTP_HEAD_COST_RATIO": value,
            ]), 0.18)
        }
        XCTAssertEqual(Q35MTPAdaptivePolicy.configuredCostRatio(environment: [
            "MERERUN_Q35_MTP_HEAD_COST_RATIO": "0.4",
        ]), 0.4)
    }

    func testProfilingIsExplicitAndOlderDiagnosticsRemainDecodable() throws {
        XCTAssertNil(Q35MTPProfile.make(environment: [:]))
        XCTAssertNotNil(Q35MTPProfile.make(environment: ["MERERUN_Q35_MTP_PROFILE": "1"]))
        let data = Data(#"{"route":"mtp-speculative","rounds":12}"#.utf8)
        let decoded = try JSONDecoder().decode(ChatAccelerationDiagnostics.self, from: data)
        XCTAssertNil(decoded.speculationProfile)
        XCTAssertEqual(decoded.rounds, 12)
    }
}
