#if os(macOS)
import MereRunMLXTestSupport
import Metal
import MLX
import MLXRandom
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor

final class Q35NativeDraftScorerTests: MLXTestCase {
    func testNativeScorerMatchesLearnedPredecessorLattice() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Requires M5 tensor support.")
        Device.setDefault(device: .gpu)
        MLXRandom.seed(1219)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let width = 5120, rank = 256, vocabulary = 4096, k = 16, anchor = 13
        let projection = PortableQuantizedLinear(weight: MLXRandom.randInt(0..<UInt32.max, [rank, width / 8]),
            bias: nil, scales: MLXRandom.uniform(0.001..<0.01, [rank, width / 64]).asType(.bfloat16),
            biases: MLXRandom.uniform(-0.05..<0.05, [rank, width / 64]).asType(.bfloat16), groupSize: 64, bits: 4, mode: .affine)
        let predecessors = MLXRandom.uniform(-0.05..<0.05, [vocabulary, rank]).asType(.bfloat16)
        let successors = MLXRandom.uniform(-0.05..<0.05, [vocabulary, rank]).asType(.bfloat16)
        let scorer = try XCTUnwrap(Q35NativeDraftScorer(projection: projection, predecessors: predecessors,
                                                       successors: successors, device: device))
        let topK = try M5NativeTopK(device: device)
        for rows in [1, 7, 15] {
            let hidden = MLXRandom.uniform(-1..<1, [1, rows, width]).asType(.bfloat16)
            let logits = MLXRandom.uniform(-0.4..<0.4, [1, rows, vocabulary]).asType(.bfloat16)
            let input = try M5NativeArrayBuffer(hidden, device: device)
            let logitBuffer = try M5NativeArrayBuffer(logits, device: device)
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            XCTAssertTrue(scorer.stageInput(input, on: command, rows: rows))
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder(dispatchType: .concurrent))
            topK.encode(into: encoder, logits: logitBuffer.buffer, rows: rows, vocabulary: vocabulary, offset: logitBuffer.offset)
            scorer.encode(into: encoder, topK: topK, anchor: anchor, rows: rows)
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            XCTAssertNil(command.error)

            let shortlist = try XCTUnwrap(topK.read(rows: rows))
            let ids = MLXArray(shortlist.tokens, [rows, k])
            let root = MLXArray(Array(repeating: Int32(anchor), count: k)).reshaped(1, k)
            let parents = rows == 1 ? root : concatenated([root, ids[0..<(rows - 1), 0...]], axis: 0)
            let projected = QuantizedVerificationContext.$maximumM5Rows.withValue(16) { projection(hidden) }
            let edges = (take(predecessors, parents, axis: 0).expandedDimensions(axis: 2)
                * projected.reshaped(rows, 1, 1, rank)
                * take(successors, ids, axis: 0).expandedDimensions(axis: 1)).sum(axis: -1)
            let unary = MLXArray(shortlist.scores, [rows, 1, k]).asType(.bfloat16)
            let conditional = (edges + unary).asType(.float32)
            let expected = conditional - logSumExp(conditional, axis: -1, keepDims: true)
            let actual = try XCTUnwrap(scorer.read(rows: rows))
            let maximumDifference = (MLXArray(actual, expected.shape) - expected).abs().max().item(Float.self)
            XCTAssertLessThan(maximumDifference, 0.003, "rows=\(rows)")
        }
    }
}
#endif
