import XCTest
@testable import MereRunQwenModel

final class Q35DFlash2WidthPolicyTests: XCTestCase {
    private func profile(repair: Double = 0) -> Q35DFlash2Costs {
        let full = (0...15).map { 1 + Double($0) * 0.02 }
        let rejected = (0...15).map { full[$0] + ($0 > 7 ? repair : 0) }
        return Q35DFlash2Costs(secondsByDepth: full, rejectedSecondsByDepth: rejected)
    }

    func testProbesThenKeepsProfitableWideCode() {
        var policy = Q35DFlash2Policy(costs: profile())
        var first: [Int] = []
        for _ in 0..<4 {
            let depth = policy.draftDepth(offeredDepth: 15)
            first.append(depth)
            policy.record(proposed: depth, accepted: depth)
        }
        XCTAssertEqual(first, [7, 15, 15, 15])
        XCTAssertEqual(policy.draftDepth(offeredDepth: 15), 15)
        XCTAssertLessThanOrEqual(policy.draftDepth(offeredDepth: 5), 5)
        XCTAssertEqual(policy.draftDepth(offeredDepth: 0), 0)
    }

    func testPerfectAcceptanceCeilingsPruneShortCodeProbes() {
        var policy = Q35DFlash2Policy(costs: profile())
        var offered: [Int] = []
        for _ in 0..<48 {
            let depth = policy.draftDepth(offeredDepth: 15)
            offered.append(depth)
            policy.record(proposed: depth, accepted: depth)
        }
        XCTAssertTrue(Set(offered).isSubset(of: [7, 15]))
        XCTAssertTrue(offered.contains(15))
        var targetWins = Q35DFlash2Policy(costs: Q35DFlash2Costs(
            secondsByDepth: [0.001] + Array(repeating: 1, count: 15),
            rejectedSecondsByDepth: [0.001] + Array(repeating: 1, count: 15)
        ))
        targetWins.record(proposed: 7, accepted: 7)
        for _ in 0..<40 {
            XCTAssertEqual(targetWins.draftDepth(offeredDepth: 15), 0)
            targetWins.recordTargetOnly()
        }
    }

    func testProseUsesShortWidthsAndFrozenRepairPricesMatter() {
        var policy = Q35DFlash2Policy(costs: profile(repair: 10))
        for depth in [7, 15, 3, 1] { policy.record(proposed: depth, accepted: min(1, depth)) }
        XCTAssertLessThan(policy.draftDepth(offeredDepth: 15), 7)
        var cheap = Q35DFlash2Policy(costs: profile())
        var expensive = Q35DFlash2Policy(costs: profile(repair: 10))
        for _ in 0..<5 {
            for depth in [7, 15, 3, 1] {
                cheap.record(proposed: depth, accepted: min(12, depth))
                expensive.record(proposed: depth, accepted: min(12, depth))
            }
        }
        XCTAssertEqual(cheap.draftDepth(offeredDepth: 15), 15)
        XCTAssertEqual(expensive.draftDepth(offeredDepth: 15), 7)
    }

    func testWidthHistoriesAreSeparateAndProbesCanRecover() {
        var policy = Q35DFlash2Policy(costs: profile())
        for depth in [7, 15, 3, 1] { policy.record(proposed: depth, accepted: depth == 15 ? 0 : depth) }
        XCTAssertEqual(policy.draftDepth(offeredDepth: 15), 7)
        var sawWideProbe = false
        for _ in 0..<101 {
            let depth = policy.draftDepth(offeredDepth: 15)
            sawWideProbe = sawWideProbe || depth == 15
            policy.record(proposed: depth, accepted: depth)
        }
        XCTAssertTrue(sawWideProbe)
        XCTAssertEqual(policy.draftDepth(offeredDepth: 15), 15)
    }

    func testNarrowingDropsTheUnqualifiedWideStrategy() {
        let narrowed = profile().limited(to: 7)
        XCTAssertEqual(narrowed.secondsByDepth.count, 8)
        XCTAssertNil(narrowed.rejectedSecondsByDepth)
        var a = Q35DFlash2Policy(costs: profile())
        var b = Q35DFlash2Policy(costs: profile())
        for _ in 0..<40 {
            let x = a.draftDepth(offeredDepth: 15), y = b.draftDepth(offeredDepth: 15)
            XCTAssertEqual(x, y)
            if x == 0 { break }
            a.record(proposed: x, accepted: min(x, 2))
            b.record(proposed: y, accepted: min(y, 2))
        }
    }
}
