import Foundation
import XCTest
@testable import MereRunCore

final class MuseGlimmerDFlashCostTests: XCTestCase {
    func testCalibrationWaitsForSustainedLowAcceptance() {
        var cost = MuseGlimmerDFlashCost()
        for _ in 0..<9 { cost.observe(accepted: 0, seconds: 0.07) }
        XCTAssertFalse(cost.needsCalibration(acceptanceRate: 0.1, minimumAcceptanceRate: 0.4))
        cost.observe(accepted: 0, seconds: 0.07)
        XCTAssertFalse(cost.needsCalibration(acceptanceRate: 0.8, minimumAcceptanceRate: 0.4))
        XCTAssertTrue(cost.needsCalibration(acceptanceRate: 0.1, minimumAcceptanceRate: 0.4))
        XCTAssertFalse(cost.shouldFallback)
        cost.calibrate(serialSeconds: 0.05)
        XCTAssertTrue(cost.shouldFallback)
        XCTAssertFalse(cost.needsCalibration(acceptanceRate: 0.1, minimumAcceptanceRate: 0.4))
    }

    func testCostPolicyKeepsUsefulDraftingAfterPoorOpeningRounds() {
        var cost = MuseGlimmerDFlashCost(serialSeconds: 0.05)
        for _ in 0..<2 { cost.observe(accepted: 0, seconds: 2) }
        XCTAssertFalse(cost.shouldFallback)
        XCTAssertEqual(cost.confirmedTokens, 0)
        for _ in 0..<8 { cost.observe(accepted: 2, seconds: 0.06) }
        XCTAssertFalse(cost.shouldFallback)
        XCTAssertEqual(cost.confirmedTokens, 24)
    }

    func testCostPolicyMeasuresWorkRatherThanAcceptanceAlone() {
        var slow = MuseGlimmerDFlashCost(serialSeconds: 0.05)
        var fast = slow
        for _ in 0..<9 {
            slow.observe(accepted: 0, seconds: 0.07)
            fast.observe(accepted: 0, seconds: 0.025)
        }
        XCTAssertFalse(slow.shouldFallback)
        slow.observe(accepted: 0, seconds: 0.07)
        fast.observe(accepted: 0, seconds: 0.025)
        XCTAssertTrue(slow.shouldFallback)
        XCTAssertFalse(fast.shouldFallback)
    }

    func testOlderDFlashTelemetryDecodesWithoutCostMeasurements() throws {
        let data = Data("""
        {"enabled":true,"active":false,"speculativeTokens":3,"rounds":2,"draftedTokens":6,
         "acceptedTokens":1,"rejectedTokens":2,"fullAcceptanceRounds":0,
         "targetVerificationForwards":2,"targetRecoveryForwards":0,"targetFallbackForwards":252,
         "adaptiveFallbacks":1,"reason":"draft acceptance fell below 40%"}
        """.utf8)
        let stats = try JSONDecoder().decode(MuseGlimmerDFlashStats.self, from: data)
        XCTAssertNil(stats.serialCalibrationSeconds)
        XCTAssertNil(stats.speculativeSeconds)
        XCTAssertEqual(stats.targetFallbackForwards, 252)
    }
}
