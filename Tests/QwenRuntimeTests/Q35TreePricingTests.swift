import Foundation
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor

final class Q35TreePricingTests: XCTestCase {
    func testPricesEveryParentClosedPrefixAfterUsefulCalibrationRounds() {
        var logProbability = [0.0]
        for row in 1...15 { logProbability.append(logProbability.last! - (row <= 3 ? 0.01 : 6)) }
        let proposal = DFlash2TreeProposal(tokens: Array(0...15), parents: [-1] + Array(0..<15),
                                          logProbabilities: logProbability)
        var policy = Q35DFlash2TreePricing()
        for nodes in [1, 1, 7, 7, 15, 15] {
            let selected = policy.select(proposal, remaining: 64)
            XCTAssertEqual(selected.tokens.count - 1, nodes)
            policy.observe(selected, acceptedPath: Array(0..<min(nodes + 1, 4)))
            policy.record(nodes: nodes, seconds: 0.01 + Double(nodes - 1) * 0.0001)
        }
        // The selected three-node prefix is neither of the legacy 7/15 caps.
        let selected = policy.select(proposal, remaining: 64)
        XCTAssertEqual(selected.tokens, [0, 1, 2, 3])
        XCTAssertEqual(selected.parents, [-1, 0, 1, 2])
        XCTAssertEqual(policy.select(proposal, remaining: 4).tokens, proposal.tokens)
        XCTAssertNotNil(policy.prices[1]); XCTAssertNotNil(policy.prices[7]); XCTAssertNotNil(policy.prices[15])
        XCTAssertNil(policy.prices[3])
    }

    func testContextCopiesMergeWithoutDuplicatingEdgesOrLosingParents() throws {
        let draft = DFlash2TreeProposal(tokens: [1, 10, 11, 20], parents: [-1, 0, 0, 1],
                                       logProbabilities: [0, -0.2, -1, -0.4])
        let merged = draft.mergingCopy([10, 30, 40], maximumNodes: 4)
        XCTAssertEqual(merged.tokens, [1, 10, 20, 30, 40])
        XCTAssertEqual(merged.parents, [-1, 0, 1, 1, 3])
        XCTAssertEqual(merged.copyRows, [3, 4])
        let tree = try Q35VerificationTree(parents: merged.parents)
        XCTAssertEqual(tree.matchingPath(tokens: merged.tokens, targetPicks: [10, 30, 63, 40, 63],
                                        budget: 16, eos: [63]), [0, 1, 3, 4])
        XCTAssertEqual(merged.prefix(nodes: 2).copyRows, [])
    }

    func testCopyIndexUsesOnlyEarlierConfirmedTokenIDs() {
        var index = Q35ContextCopyIndex(tokens: [1, 2, 3, 4, 9, 8, 1, 2, 3])
        index.append([4])
        XCTAssertEqual(index.continuation(limit: 3), [9, 8, 1])
        index.append([99])
        XCTAssertEqual(index.continuation(limit: 15), [])
    }
}
