import Foundation
import MLX
import XCTest
@testable import MereRunCore

/// `KVCacheSimple.update` writes new tokens with subscript assignment, which
/// rebinds the same `MLXArray` wrapper in place. A fork that shares the
/// wrapper objects therefore sees every later write on the parent — which
/// silently corrupted prefix-KV snapshots stored mid-request (the request's
/// remaining prefill and decode kept writing into the stored copy). These
/// tests pin the isolation contract.
final class KVCacheForkTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        MLXTestSupport.ensureMetalLibraryAvailable()
    }

    private func makeKV(_ value: Float, tokens: Int) -> (MLXArray, MLXArray) {
        (
            MLXArray.full([1, 2, tokens, 4], values: MLXArray(value)),
            MLXArray.full([1, 2, tokens, 4], values: MLXArray(value))
        )
    }

    func testForkIsIsolatedFromLaterParentWrites() throws {
        let parent = KVCacheSimple(step: 4)
        let (k1, v1) = makeKV(1.0, tokens: 3)
        _ = parent.update(keys: k1, values: v1)

        let fork = try XCTUnwrap(parent.fork() as? KVCacheSimple)
        XCTAssertEqual(fork.offset, 3, "fork offset must be frozen at fork time")

        // Parent keeps decoding: writes MORE tokens into its buffers. With a
        // wrapper-sharing fork these writes land in the fork's arrays too.
        let (k2, v2) = makeKV(2.0, tokens: 1)
        _ = parent.update(keys: k2, values: v2)
        let (k3, v3) = makeKV(3.0, tokens: 1)
        _ = parent.update(keys: k3, values: v3)

        // Read the fork's snapshot by appending one marker token: the first
        // three tokens it returns must still be the pre-fork 1.0s, not the
        // parent's later 2.0/3.0 writes.
        let (kM, vM) = makeKV(9.0, tokens: 1)
        let forkView = fork.update(keys: kM, values: vM)
        MLX.eval(forkView.0)
        let snapshot = forkView.0[0..., 0..., 0..<3, 0...].asArray(Float.self)
        XCTAssertEqual(snapshot.count, 1 * 2 * 3 * 4)
        XCTAssertTrue(
            snapshot.allSatisfy { $0 == 1.0 },
            "fork observed the parent's post-fork writes (max=\(snapshot.max() ?? 0)) — snapshot corrupted"
        )
    }

    func testForkedCacheCanDivergeIndependently() throws {
        let parent = KVCacheSimple(step: 4)
        let (k1, v1) = makeKV(1.0, tokens: 2)
        _ = parent.update(keys: k1, values: v1)

        let fork = try XCTUnwrap(parent.fork() as? KVCacheSimple)
        let (kF, vF) = makeKV(5.0, tokens: 1)
        let forkView = fork.update(keys: kF, values: vF)
        let (kP, vP) = makeKV(7.0, tokens: 1)
        let parentView = parent.update(keys: kP, values: vP)

        MLX.eval(forkView.0, parentView.0)
        let forkTail = forkView.0[0..., 0..., 2..<3, 0...].asArray(Float.self)
        let parentTail = parentView.0[0..., 0..., 2..<3, 0...].asArray(Float.self)
        XCTAssertTrue(forkTail.allSatisfy { $0 == 5.0 })
        XCTAssertTrue(parentTail.allSatisfy { $0 == 7.0 })
    }
}
