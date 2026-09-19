import MLX
import MLXNN
import XCTest
@testable import MereRunQwenModel

final class Bonsai2FusionTests: MereRunCoreTestCase {
    func testMetalTransformMatchesReferenceAcrossBlocksAndDirections() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Run with MERERUN_TEST_MLX_DEVICE=gpu for the fused Metal transform")
        }
        for width in [1024, 5120, 17408] {
            let signs = MLXArray((0..<width).map { $0 % 3 == 0 ? Float(-1) : Float(1) })
            for dtype: DType in [.float16, .bfloat16, .float32] {
                for inverse in [false, true] {
                    let input = MLXArray((0..<(width * 3)).map { Float($0 % 127 - 63) / 65 })
                        .reshaped(1, 3, width).asType(dtype)
                    let actual = try XCTUnwrap(Q35PrismHadamard.apply(input, block: 1024, signs: signs, inverse: inverse))
                    let reference = Q35PrismTransform.reference(input, block: 1024, signs: signs, inverse: inverse)
                    XCTAssertEqual(MLX.max(MLX.abs(actual - reference)).item(Float.self), 0)
                }
            }
        }
    }

    private func projection(rows: Int, flipped: Bool = false, block: Int = 512) -> Q35PrismLinear {
        let width = 1024
        let words = (0..<(rows * width / 16)).map { UInt32(truncatingIfNeeded: $0 &* 1_664_525 &+ 1_013_904_223) }
        let signs = MLXArray((0..<width).map { ($0 % 3 == 0) != flipped ? Float(-1) : Float(1) })
        return Q35PrismLinear(
            weight: MLXArray(words, [rows, width / 16]),
            scales: MLXArray.full([rows, width / 128], values: MLXArray(Float(0.125))).asType(.float16),
            biases: MLXArray.full([rows, width / 128], values: MLXArray(Float(-0.25))).asType(.float16),
            signs: signs, block: block
        )
    }

    func testFusionMatchesSeparateDecodeAndPrefillProjections() throws {
        let sources = [projection(rows: 32), projection(rows: 64), projection(rows: 16)]
        for fuseWeights in [false, true] {
            let fusion = Q35PrismFusion(fuseWeights: fuseWeights)
            for length in [1, 7] {
                let input = MLXArray((0..<(length * 1024)).map { Float($0 % 19 - 9) / 16 })
                    .reshaped(1, length, 1024).asType(.float16)
                let actual = try XCTUnwrap(fusion.callSplit(input, projections: sources))
                for (source, result) in zip(sources, actual) {
                    let expected = source(input)
                    XCTAssertEqual(result.shape, expected.shape)
                    XCTAssertEqual(result.dtype, expected.dtype)
                    XCTAssertLessThanOrEqual(MLX.max(MLX.abs(result - expected)).item(Float.self), 0.001)
                }
            }
        }
    }

    func testFusionRejectsDifferentSignsBlocksAndOrdinaryLayers() {
        let input = MLXArray.ones([1, 1, 1024], dtype: .float16)
        let source = projection(rows: 32)
        let fusion = Q35PrismFusion(fuseWeights: true)
        XCTAssertNil(fusion.callSplit(input, projections: [source, projection(rows: 32, flipped: true)]))
        XCTAssertNil(fusion.callSplit(input, projections: [source, projection(rows: 32, block: 1024)]))
        XCTAssertNil(fusion.callSplit(input, projections: [source, Linear(1024, 32)]))
        XCTAssertNil(fusion.callSplit(input, projections: [source, nil]))
    }

    func testFusionInvalidatesAfterModuleReplacement() throws {
        let input = MLXArray.ones([1, 1, 1024], dtype: .float16)
        let first = projection(rows: 32)
        let fusion = Q35PrismFusion(fuseWeights: true)
        XCTAssertNotNil(fusion.callSplit(input, projections: [first, projection(rows: 16)]))
        let replacement = projection(rows: 64)
        let result = try XCTUnwrap(fusion.callSplit(input, projections: [first, replacement]))
        XCTAssertEqual(result[1].shape, [1, 1, 64])
        XCTAssertLessThanOrEqual(MLX.max(MLX.abs(result[1] - replacement(input))).item(Float.self), 0.001)
        XCTAssertNil(fusion.callSplit(input, projections: [first, projection(rows: 64, flipped: true)]))
    }
}
