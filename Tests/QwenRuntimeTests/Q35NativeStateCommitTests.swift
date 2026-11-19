import Foundation
import MLX
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor
#if os(macOS)
import Metal

final class Q35NativeStateCommitTests: MLXTestCase {
    func testBatchedPathsAndMutableForkExports() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Requires M5 tensor support.")
        Device.setDefault(device: .gpu)
        MLXRandom.seed(944)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        let program = try Q35NativeStateCommit(device: device, queue: queue)
        let arena = try Q35NativeArena(device: device, bytes: 855040 + 256, dtype: .bfloat16)
        let floats = try Q35NativeArena(device: device, bytes: 6144 + 256, dtype: .float32)
        _ = arena.allocate(128); _ = floats.allocate(64)
        let qkvz = arena.allocate(16 * 16384), ba = arena.allocate(16 * 96)
        let q = arena.allocate(16 * 16 * 128), k = arena.allocate(16 * 16 * 128), v = arena.allocate(16 * 48 * 128)
        let g = floats.allocate(16 * 48), beta = floats.allocate(16 * 48)
        let projected = qkvz.buffer.contents().advanced(by: qkvz.offset).bindMemory(to: UInt16.self, capacity: 16 * 16384)
        for row in 0..<16 {
            projected.advanced(by: row * 16384).initialize(repeating: UInt16(Float(row + 1).bitPattern >> 16), count: 16384)
        }
        for slice in [ba, q, k, v] {
            slice.buffer.contents().advanced(by: slice.offset).bindMemory(to: UInt16.self, capacity: slice.byteCount / 2)
                .initialize(repeating: 0, count: slice.byteCount / 2)
        }
        g.buffer.contents().advanced(by: g.offset).bindMemory(to: Float.self, capacity: 16 * 48).initialize(repeating: 0.5, count: 16 * 48)
        beta.buffer.contents().advanced(by: beta.offset).bindMemory(to: Float.self, capacity: 16 * 48).initialize(repeating: 0, count: 16 * 48)
        let conv = broadcast(MLXArray([-3, -2, -1]).asType(.bfloat16).reshaped(1, 3, 1), to: [1, 3, 10240]).contiguous()
        let original = MLXRandom.uniform(0.1..<0.2, [1, 48, 128, 128])
        let initial = try [original, original * 2].map { try Q35NativeStoredLinearState(convolution: conv, recurrent: $0, device: device) }
        let inputs = initial.map { state in
            Q35NativeReplayInput(initial: state, qkvz: qkvz, ba: ba, q: q, k: k, v: v, g: g, beta: beta,
                aLog: MLXArray.zeros([48]), dtBias: MLXArray.zeros([48]))
        }
        for balanced in [false, true] {
            let tree = try Q35VerificationTree(parents: [-1] + (1..<16).map { balanced ? ($0 - 1) / 2 : $0 - 1 })
            let selected = tree.paths[15]
            let metadata = try XCTUnwrap(Q35NativeGreedyTail.makeMetadata(device: device))
            let initialCount = program.batchCount
            let prepared = try program.prepare(inputs: inputs, selection: metadata)
            XCTAssertEqual(program.batchCount, initialCount)
            let values = metadata.contents().assumingMemoryBound(to: Int32.self)
            values[0] = Int32(selected.count)
            values[1] = -1
            for (index, row) in selected.enumerated() { values[index + 2] = Int32(row) }
            let batch = Q35NativeReplayBatch(tree: tree, inputs: inputs, program: program, prepared: (selected, prepared))
            for row in 0..<16 {
                let path = tree.paths[row]
                for index in 0..<2 {
                    let state = try batch.state(path: path, index: index), value = try state.materialized()
                    let expected = original * Float(index + 1) * Float(pow(0.5, Double(path.count)))
                    XCTAssertEqual((value.recurrent - expected).abs().max().item(Float.self), 0)
                    let history = Array(([-3, -2, -1] + path.map { $0 + 1 }).suffix(3)).map(Float.init)
                    let expectedConv = broadcast(MLXArray(history).asType(.bfloat16).reshaped(1, 3, 1), to: [1, 3, 10240])
                    XCTAssertEqual((value.convolution.asType(.float32) - expectedConv.asType(.float32)).abs().max().item(Float.self), 0)
                    if row == 0 && index == 0 {
                        let base = Q35LinearCache(); base.installNativeState(state)
                        let fork = base.fork()
                        XCTAssertNotNil(base.nativeState); XCTAssertNotNil(fork.nativeState)
                        Q35NativeVerificationPolicy.$prepareCommitOverride.withValue(true) {
                            XCTAssertTrue(Q35LayerCache.linear(base).pendingTreeState.isEmpty)
                            XCTAssertNotNil(base.nativeState)
                        }
                        fork.convState = MLXArray.ones([1, 3, 10240], dtype: .bfloat16)
                        XCTAssertNil(fork.nativeState); XCTAssertNotNil(base.nativeState)
                        XCTAssertEqual((fork.recurrentState! - expected).abs().max().item(Float.self), 0)
                        fork.recurrentState?[0, 0, 0, 0] = MLXArray(Float(9))
                        XCTAssertEqual((base.recurrentState! - expected).abs().max().item(Float.self), 0)
                        XCTAssertEqual(base.convState![0, 0, 0].item(Float.self), -2)
                        XCTAssertEqual(fork.recurrentState![0, 0, 0, 0].item(Float.self), 9)
                    }
                }
            }
        }
        XCTAssertEqual(program.batchCount, 32, "Each accepted path submits one batch for both layers.")
    }
}
#endif
