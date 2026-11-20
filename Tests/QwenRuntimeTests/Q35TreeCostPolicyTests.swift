import XCTest
@testable import MereRunQwenModel

final class Q35TreeCostPolicyTests: XCTestCase {
    func testWiderWorkMustPayForItsMeasuredCost() {
        var profitable = bootstrap(narrowTokens: 6, wideTokens: 10, narrowSeconds: 0.06, wideSeconds: 0.08)
        XCTAssertEqual(profitable.nodeLimit(remaining: 128), 15)
        XCTAssertEqual(profitable.nodeLimit(remaining: 14), 7)
        profitable.record(nodes: 15, confirmedTokens: 1, seconds: 0.12)
        profitable.record(nodes: 15, confirmedTokens: 1, seconds: 0.12)
        XCTAssertEqual(profitable.nodeLimit(remaining: 128), 7)

        let expensive = bootstrap(narrowTokens: 3, wideTokens: 3, narrowSeconds: 0.06, wideSeconds: 0.09)
        XCTAssertEqual(expensive.nodeLimit(remaining: 128), 7)
        let marginal = bootstrap(narrowTokens: 6, wideTokens: 8, narrowSeconds: 0.06, wideSeconds: 0.078)
        XCTAssertEqual(marginal.nodeLimit(remaining: 128), 7)
    }

    func testTransitionWaitsDoNotBecomeWidthPrices() {
        var policy = Q35DFlash2TreeCostPolicy()
        XCTAssertEqual(policy.nodeLimit(remaining: 128), 7)
        policy.record(nodes: 7, confirmedTokens: 6, seconds: 20)
        XCTAssertNil(policy.secondsByNodeCount[7])
        XCTAssertEqual(policy.nodeLimit(remaining: 128), 7)
        policy.record(nodes: 7, confirmedTokens: 6, seconds: 0.06)
        XCTAssertEqual(policy.secondsByNodeCount[7], 0.06)
        XCTAssertEqual(policy.nodeLimit(remaining: 128), 15)
        policy.record(nodes: 15, confirmedTokens: 10, seconds: 30)
        XCTAssertNil(policy.secondsByNodeCount[15])
        XCTAssertEqual(policy.nodeLimit(remaining: 128), 15)
        policy.record(nodes: 15, confirmedTokens: 10, seconds: 0.08)
        XCTAssertEqual(policy.secondsByNodeCount[15], 0.08)
    }

    func testSparseProbesRequireAProfitableCeilingAndHoldTwoRounds() {
        var impossible = bootstrap(narrowTokens: 8, wideTokens: 16, narrowSeconds: 0.06, wideSeconds: 0.04)
        for _ in 4..<64 {
            let nodes = impossible.nodeLimit(remaining: 1024)
            XCTAssertEqual(nodes, 15)
            impossible.record(nodes: nodes, confirmedTokens: 16, seconds: 0.04)
        }
        XCTAssertEqual(impossible.nodeLimit(remaining: 1024), 15)

        var viable = bootstrap(narrowTokens: 3, wideTokens: 3, narrowSeconds: 0.06, wideSeconds: 0.09)
        for _ in 4..<64 {
            let nodes = viable.nodeLimit(remaining: 1024)
            XCTAssertEqual(nodes, 7)
            viable.record(nodes: nodes, confirmedTokens: 3, seconds: 0.06)
        }
        XCTAssertEqual(viable.nodeLimit(remaining: 1024), 15)
        viable.record(nodes: 15, confirmedTokens: 3, seconds: 0.09)
        XCTAssertEqual(viable.nodeLimit(remaining: 1024), 15)
        viable.record(nodes: 15, confirmedTokens: 3, seconds: 0.09)
        XCTAssertEqual(viable.nodeLimit(remaining: 1024), 7)
    }

    private func bootstrap(narrowTokens: Int, wideTokens: Int,
                           narrowSeconds: Double, wideSeconds: Double) -> Q35DFlash2TreeCostPolicy {
        var policy = Q35DFlash2TreeCostPolicy()
        for nodes in [7, 7, 15, 15] {
            XCTAssertEqual(policy.nodeLimit(remaining: 128), nodes)
            policy.record(nodes: nodes, confirmedTokens: nodes == 7 ? narrowTokens : wideTokens,
                          seconds: nodes == 7 ? narrowSeconds : wideSeconds)
        }
        return policy
    }
}
