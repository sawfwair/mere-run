import XCTest
import MereRunKVCache
import MereRunTensor
@testable import MereRunQwenModel

final class Q35WideVerificationPolicyTests: XCTestCase {
    func testWideDepthRequiresTheQualifiedCacheCommitPath() {
        let ordinary: [Q35LayerCache?] = [.linear(Q35LinearCache()), .full(KVCacheSimple())]
        let affine: [Q35LayerCache?] = [.linear(Q35LinearCache()), .full(AffineQuantizedKVCache(groupSize: 32, bits: 8))]
        XCTAssertEqual(Q35DFlash2VerificationPolicy.limitForCaches(15, trainedBlockSize: 8, caches: ordinary), 15)
        XCTAssertEqual(Q35DFlash2VerificationPolicy.limitForCaches(15, trainedBlockSize: 8, caches: affine), 7)
        XCTAssertEqual(Q35DFlash2VerificationPolicy.limitForCaches(3, trainedBlockSize: 8, caches: affine), 3)
    }

    func testExtendedProposalsRequireBothWideVerificationAndExplicitGreedyOptIn() {
        Q35DFlash2VerificationPolicy.$requestedProposalDepth.withValue(nil) {
            XCTAssertEqual(Q35DFlash2VerificationPolicy.proposalLimit(
                trainedBlockSize: 8, verificationRows: 16, sampled: false), 7)
        }
        Q35DFlash2VerificationPolicy.$requestedProposalDepth.withValue(15) {
            XCTAssertEqual(Q35DFlash2VerificationPolicy.proposalLimit(
                trainedBlockSize: 8, verificationRows: 8, sampled: false), 7)
            XCTAssertEqual(Q35DFlash2VerificationPolicy.proposalLimit(
                trainedBlockSize: 8, verificationRows: 16, sampled: false), 15)
            XCTAssertEqual(Q35DFlash2VerificationPolicy.proposalLimit(
                trainedBlockSize: 8, verificationRows: 16, sampled: true), 7)
        }
    }
}
