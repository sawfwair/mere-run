import MLX
import MLXNN
import MLXRandom
import MereRunMLXTestSupport
@testable import MereRunLTXModel
import XCTest

final class LTXComputationBoundaryTests: MLXTestCase {
    func testCachedContextMatchesDirectGatedCrossAttentionWithDifferentQueryLengths() {
        MLXRandom.seed(117)
        let attention = LTXDistilledAttention(
            queryDim: 8, contextDim: 12, heads: 2, headDim: 4,
            normEps: 1e-6, applyGatedAttention: true
        )
        let context = MLXRandom.normal([1, 5, 12])
        let keyAngles = MLXRandom.normal([1, 2, 5, 2])
        let keyRope = (cos: MLX.cos(keyAngles), sin: MLX.sin(keyAngles))
        let projected = attention.projectContext(context, keyRope: keyRope)
        for length in [1, 3] {
            let query = MLXRandom.normal([1, length, 8])
            let angles = MLXRandom.normal([1, 2, length, 2])
            let rope = (cos: MLX.cos(angles), sin: MLX.sin(angles))
            let direct = attention(query, context: context, mask: nil, rope: rope, keyRope: keyRope)
            let cached = attention(query, context: nil, mask: nil, rope: rope, projectedContext: projected)
            assertClose(cached, direct)
        }
    }

    func testMaskedContextTokensCannotAffectAttention() {
        MLXRandom.seed(118)
        let attention = LTXDistilledAttention(
            queryDim: 8, contextDim: 8, heads: 2, headDim: 4, normEps: 1e-6
        )
        let query = MLXRandom.normal([1, 2, 8])
        let prefix = MLXRandom.normal([1, 3, 8])
        let suffix = MLXRandom.normal([1, 2, 8]) * 100
        let context = MLX.concatenated([prefix, suffix], axis: 1)
        let weights = MLXArray([Float(1), 1, 1, 0, 0]).reshaped(1, 1, 5)
        let mask = prepareLTXSelfAttentionMask(weights, dtype: .float32)
        let masked = attention(query, context: context, mask: mask, rope: nil)
        let cropped = attention(query, context: prefix, mask: nil, rope: nil)
        assertClose(masked, cropped)
    }

    func testCausalVideoConvolutionPrefixIsIndependentOfFutureFrames() {
        MLXRandom.seed(119)
        let convolution = LTXCausalConv3d(
            inChannels: 2, outChannels: 3, kernelSize: (3, 3, 3)
        )
        let prefix = MLXRandom.normal([1, 2, 2, 4, 4])
        let future = MLXRandom.normal([1, 2, 3, 4, 4]) * 100
        let full = convolution(MLX.concatenated([prefix, future], axis: 2), causal: true)
        let partial = convolution(prefix, causal: true)
        XCTAssertEqual(full.shape, [1, 3, 5, 4, 4])
        assertClose(full[0..., 0..., 0..<2, 0..., 0...], partial)
    }

    func testNeighborhoodAttentionIsIndependentOfQueryTileBudget() {
        MLXRandom.seed(120)
        let shape = [1, 3, 4, 5, 2, 4]
        let query = MLXRandom.normal(shape) / 2
        let key = MLXRandom.normal(shape)
        let value = MLXRandom.normal(shape)
        let whole = ltxDiffVAENeighborhoodAttention(
            query: query, key: key, value: value, kernel: (3, 3, 3), scoreBudget: 1 << 25
        )
        let tiled = ltxDiffVAENeighborhoodAttention(
            query: query, key: key, value: value, kernel: (3, 3, 3), scoreBudget: 54
        )
        assertClose(tiled, whole)
    }

    func testPatchAndChannelLayoutsPreserveTemporalAndSpatialOrder() {
        let pixels = MLXArray(0..<(2 * 4 * 6 * 8)).asType(.float32).reshaped(1, 2, 4, 6, 8)
        let packed = spaceToDepth3D(pixels, stride: (2, 3, 2))
        XCTAssertEqual(packed.shape, [1, 24, 2, 2, 4])
        assertClose(depthToSpace3D(packed, stride: (2, 3, 2)), pixels, tolerance: 0)
        let patches = patchify3D(pixels, patchSizeHW: 2, patchSizeT: 2)
        XCTAssertEqual(patches[0, 0, 0, 0, 0].item(Float.self), 0)
        XCTAssertEqual(patches[0, 1, 0, 0, 0].item(Float.self), 8)
        assertClose(unpatchify3D(patches, patchSizeHW: 2, patchSizeT: 2), pixels, tolerance: 0)
    }

    func testAudioAndVideoPositionsUseTheSameCausalTimeOrigin() {
        let video = createPositionGrid(
            batchSize: 1, numFrames: 3, height: 1, width: 1,
            temporalScale: 8, spatialScale: 32, fps: 24, causalFix: true
        )
        let audio = createAudioPositionGrid(batchSize: 1, audioFrames: 3)
        XCTAssertEqual(video[0, 0, 0, 0].item(Float.self), 0)
        XCTAssertEqual(video[0, 0, 0, 1].item(Float.self), 1 / Float(24), accuracy: 1e-7)
        XCTAssertEqual(video[0, 0, 1, 0].item(Float.self), 1 / Float(24), accuracy: 1e-7)
        XCTAssertEqual(audio[0, 0, 0, 0].item(Float.self), 0)
        XCTAssertEqual(audio[0, 0, 0, 1].item(Float.self), 0.01, accuracy: 1e-7)
        XCTAssertEqual(audio[0, 0, 1, 0].item(Float.self), 0.01, accuracy: 1e-7)
        XCTAssertEqual(computeAudioLatentFrameCount(videoFrames: 121, fps: 24), 126)
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 1e-5,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        MLX.eval(actual, expected)
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertLessThanOrEqual(
            MLX.max(MLX.abs(actual - expected)).item(Float.self), tolerance, file: file, line: line
        )
    }
}
