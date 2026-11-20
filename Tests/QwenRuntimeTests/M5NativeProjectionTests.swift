import MLX
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunTensor
#if os(macOS)
import Metal

final class M5NativeProjectionTests: MLXTestCase {
    func testNativeProjectionFeedsExactWideTopK() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Requires M5 tensor support.")
        Device.setDefault(device: .gpu)
        MLXRandom.seed(1147)
        try M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(true) {
            try checkWideTopK()
            try checkWideTopK(skipping: 1536..<1696)
        }
    }

    @available(macOS 26.0, *)
    private func checkWideTopK(skipping gap: Range<Int>? = nil) throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        let width = 5120, vocabulary = 4352, rows = 15
        let weight = try XCTUnwrap(M5TiledAffineQ4.Weights(
            weight: MLXRandom.randInt(0..<UInt32.max, [vocabulary, width / 8]),
            scales: MLXRandom.uniform(0.001..<0.01, [vocabulary, width / 64]).asType(.bfloat16),
            biases: MLXRandom.uniform(-0.1..<0.1, [vocabulary, width / 64]).asType(.bfloat16)))
        let prepared = try M5NativeFFN.Projection(weight, device: device)
        let projection = try M5NativeProjection(device: device, inputSize: width, outputSize: vocabulary, skippingOutputRows: gap)
        let sum = try M5NativeGroupSums(device: device, inputSize: width), topK = try M5NativeTopK(device: device)
        let sums = try XCTUnwrap(device.makeBuffer(length: width / 64 * 16 * 4, options: .storageModePrivate))
        let output = try XCTUnwrap(device.makeBuffer(length: rows * vocabulary * 2, options: .storageModeShared))
        let x = MLXRandom.uniform(-1..<1, [1, rows, width]).asType(.bfloat16)
        let expected = try XCTUnwrap(M5TiledAffineQ4.matmul(x, weights: weight)).asType(.float32).asArray(Float.self)
        let input = try M5NativeArrayBuffer(x, device: device)
        let command = try XCTUnwrap(queue.makeCommandBuffer()), encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        sum.encode(into: encoder, input: input.buffer, sums: sums, rows: rows, inputOffset: input.offset)
        projection.encode(into: encoder, input: input.buffer, sums: sums, output: output, weights: prepared,
                          rows: rows, inputOffset: input.offset)
        topK.encode(into: encoder, logits: output, rows: rows, vocabulary: projection.projectedSize, skippingTokenIDs: gap)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertNil(command.error)
        let actual = try XCTUnwrap(topK.read(rows: rows))
        for row in 0..<rows {
            let logits = expected[(row * vocabulary)..<((row + 1) * vocabulary)]
            let retained = (0..<vocabulary).filter { gap?.contains($0) != true }
            let ranking = retained.sorted { left, right in
                let a = logits[logits.startIndex + left], b = logits[logits.startIndex + right]
                return a == b ? left < right : a > b
            }
            let shortlist = ranking.prefix(16)
            let range = (row * 16)..<((row + 1) * 16)
            XCTAssertEqual(Array(actual.tokens[range]), shortlist.map(Int32.init))
            XCTAssertEqual(Array(actual.scores[range]), shortlist.map { logits[logits.startIndex + $0] })
        }
    }

    func testAttentionAndHeadReductionGeometriesMatchMLX() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Requires M5 tensor support.")
        Device.setDefault(device: .gpu)
        MLXRandom.seed(935)
        try M5TiledAffineQ4.$groupAffineEnabledOverride.withValue(true) {
            let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
            // The smaller output exercises the head's unsplit K reduction
            // without allocating a full vocabulary matrix in a unit test.
            let geometries: [(Int, Int, GroupAffineArithmetic)] = [
                (5_120, 16_384, .standard), (5_120, 14_336, .standard), (6_144, 5_120, .standard), (5_120, 128, .standard),
                (5_120, 16_384, .centered), (5_120, 16_384, .compensated)
            ]
            for (k, n, arithmetic) in geometries {
                let weight = try XCTUnwrap(M5TiledAffineQ4.Weights(
                    weight: MLXRandom.randInt(0..<UInt32.max, [n, k / 8]),
                    scales: MLXRandom.uniform(0.001..<0.01, [n, k / 64]).asType(.bfloat16),
                    biases: MLXRandom.uniform(-0.1..<0.1, [n, k / 64]).asType(.bfloat16)))
                let prepared = try M5NativeFFN.Projection(weight, device: device)
                let projection = try M5NativeProjection(device: device, inputSize: k, outputSize: n, arithmetic: arithmetic)
                let sum = try M5NativeGroupSums(device: device, inputSize: k)
                let sums = try XCTUnwrap(device.makeBuffer(length: k / 64 * 16 * 4, options: .storageModePrivate))
                let output = try XCTUnwrap(device.makeBuffer(length: 16 * n * 2, options: .storageModeShared))
                for rows in [1, 3, 7, 8, 15, 16] {
                    let full = MLXRandom.uniform(-1..<1, [1, 18, k]).asType(.bfloat16)
                    let x = full[0..., 1..<(1 + rows), 0...]
                    let expected = try XCTUnwrap(M5TiledAffineQ4.matmul(x, weights: weight, arithmetic: arithmetic))
                    MLX.eval(expected)
                    let view = try M5NativeArrayBuffer(x, device: device)
                    let command = try XCTUnwrap(queue.makeCommandBuffer()), encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                    sum.encode(into: encoder, input: view.buffer, sums: sums, rows: rows, inputOffset: view.offset)
                    projection.encode(into: encoder, input: view.buffer, sums: sums, output: output, weights: prepared,
                        rows: rows, inputOffset: view.offset)
                    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                    XCTAssertNil(command.error)
                    let actual = MLXArray(rawPointer: output.contents(), [1, rows, n], dtype: .bfloat16) { _ = output }
                    XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self),
                        0, "K=\(k), N=\(n), rows=\(rows), arithmetic=\(arithmetic)")
                }
            }
        }
    }
}
#endif
