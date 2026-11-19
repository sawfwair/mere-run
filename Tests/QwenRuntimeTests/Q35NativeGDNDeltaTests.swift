import MLX
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor
#if os(macOS)
import Metal

final class Q35NativeGDNDeltaTests: MLXTestCase {
    func testEveryBranchAndSlotTopologyMatchesCurrentDelta() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Requires M5 tensor support.")
        Device.setDefault(device: .gpu)
        MLXRandom.seed(939)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        let program = Q35NativeGDNDelta(device: device)
        let base = MLXRandom.uniform(-0.01..<0.01, [1, 48, 128, 128])
        let state = try M5NativeArrayBuffer(base, device: device)
        for rows in [1, 3, 7, 8, 15, 16] {
            let q = MLXRandom.uniform(-0.1..<0.1, [1, rows, 16, 128]).asType(.bfloat16)
            let k = MLXRandom.uniform(-0.1..<0.1, [1, rows, 16, 128]).asType(.bfloat16)
            let v = MLXRandom.uniform(-1..<1, [1, rows, 48, 128]).asType(.bfloat16)
            let g = MLXRandom.uniform(0.5..<1, [1, rows, 48]), beta = MLXRandom.uniform(0..<1, [1, rows, 48])
            let buffers = try [q, k, v, g, beta].map { try XCTUnwrap($0.asMTLBuffer(device: device)) }
            for balanced in [false, true] {
                let tree = try Q35VerificationTree(parents: [-1] + (1..<rows).map { balanced ? ($0 - 1) / 2 : $0 - 1 })
                let expected = q35TreeCompactDelta(q: q, k: k, v: v, g: g, beta: beta, state: base,
                    numKeyHeads: 16, numValueHeads: 48, valueHeadDim: 128, tree: tree)
                MLX.eval(expected)
                let plan = try Q35NativeTreePlan(tree: tree, device: device), prepared = try program.prepare(plan)
                let output = try XCTUnwrap(device.makeBuffer(length: 16 * 48 * 128 * 2, options: .storageModeShared))
                let command = try XCTUnwrap(queue.makeCommandBuffer()), encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                program.encode(into: encoder, q: buffers[0], k: buffers[1], v: buffers[2], g: buffers[3], beta: buffers[4],
                    state: state, output: output, prepared: prepared)
                encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                XCTAssertNil(command.error)
                let actual = MLXArray(rawPointer: output.contents(), expected.shape, dtype: .bfloat16) { _ = output }
                XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self),
                    0, "rows=\(rows), balanced=\(balanced)")
            }
        }
    }
}
#endif
