import Foundation
import MLX
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor

#if os(macOS)
final class Q35TreeMetalTests: MLXTestCase {
    func testParentConvolutionDeltaAndEveryCommitAreSerialExact() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Parent-indexed Metal kernels require GPU tests.")
        MLXRandom.seed(719)
        let tree = try Q35VerificationTree(parents: [-1, 0, 0, 1, 2, 3, 4, 5, 2, 8, 7, 6, 9, 10, 11, 12])
        let hk = 2, hv = 4, dim = 128, channels = (2 * hk + hv) * dim
        let qkv = MLXRandom.normal([1, tree.count, channels]).asType(.bfloat16)
        let conv = MLXRandom.normal([1, 3, channels]).asType(.bfloat16)
        let weight = (MLXRandom.normal([channels, 4, 1]) * 0.1).asType(.bfloat16)
        let state = MLXRandom.normal([1, hv, dim, dim]) * 0.1
        let a = MLXRandom.normal([1, tree.count, hv]).asType(.bfloat16)
        let b = MLXRandom.normal([1, tree.count, hv]).asType(.bfloat16)
        let aLog = MLXRandom.normal([hv]).asType(.bfloat16)
        let bias = MLXRandom.normal([hv]).asType(.bfloat16)
        let work = q35TreePrework(qkv: qkv, convState: conv, convWeight: weight,
            numKeyHeads: hk, numValueHeads: hv, keyHeadDim: dim, valueHeadDim: dim,
            rmsNormWeight: MLXArray.ones([dim], dtype: .bfloat16), tree: tree)
        let result = q35TreeDeltaUpdate(q: work.q, k: work.k, v: work.v, a: a, b: b,
            aLog: aLog, dtBias: bias, state: state, numKeyHeads: hk, numValueHeads: hv,
            valueHeadDim: dim, tree: tree).0
        for row in tree.parents.indices {
            let path = tree.paths[row]
            let selected = Q35VerificationTree.take(qkv, rows: path)
            let reference = try XCTUnwrap(q35GDNPreworkMetal(qkv: selected, convState: conv, convWeight: weight,
                numKeyHeads: hk, numValueHeads: hv, keyHeadDim: dim, valueHeadDim: dim))
            let last = path.count - 1
            for (actual, expected) in [(work.q, reference.q), (work.k, reference.k), (work.v, reference.v)] {
                exact(actual[0..., row..<(row + 1), 0..., 0...], expected[0..., last..., 0..., 0...])
            }
            let expected = q35GatedDeltaUpdate(q: reference.q, k: reference.k, v: reference.v,
                a: Q35VerificationTree.take(a, rows: path), b: Q35VerificationTree.take(b, rows: path),
                aLog: aLog, dtBias: bias, state: state, numKeyHeads: hk, numValueHeads: hv, valueHeadDim: dim)
            exact(result[0..., row..<(row + 1), 0..., 0...], expected.0[0..., last..., 0..., 0...])
            let cache = Q35LinearCache()
            cache.verificationTree = tree
            cache.verificationReplay = Q35LinearVerificationReplay(baseConvState: conv, baseRecurrentState: state,
                qkv: qkv, q: work.q, k: work.k, v: work.v, a: a, b: b, aLog: aLog, dtBias: bias,
                numKeyHeads: hk, numValueHeads: hv, valueHeadDim: dim, convKeep: 3)
            cache.restoreVerificationPath(path)
            exact(try XCTUnwrap(cache.convState), reference.convState)
            exact(try XCTUnwrap(cache.recurrentState), expected.1)
            XCTAssertNil(cache.verificationReplay)
            XCTAssertNil(cache.verificationTree)
        }
    }

    private func exact(_ actual: MLXArray, _ expected: MLXArray) {
        XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0)
    }
}
#endif
