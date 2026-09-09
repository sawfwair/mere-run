import MLX
import MLXNN
import MLXRandom
import XCTest
import MereRunAudioModels
import MereRunMLXTestSupport
@testable import MereRunH3Model

final class H3AudioVideoBoundaryTests: MLXTestCase {
    func testCausalVideoConvolutionCannotReadFutureFrames() {
        MLXRandom.seed(91)
        let convolution = MiniMaxH3CausalConv3D(
            inputChannels: 2,
            outputChannels: 2,
            kernel: (3, 1, 1)
        )
        convolution.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.ones([2, 3, 1, 1, 2]),
        ]))
        let prefix = MLXRandom.normal([1, 3, 2, 2, 2])
        let future = MLXArray.ones([1, 2, 2, 2, 2]) * 1_000
        let alone = convolution(prefix)
        let extended = convolution(MLX.concatenated([prefix, future], axis: 1))
        MLX.eval(alone, extended)
        XCTAssertEqual(alone.shape, [1, 3, 2, 2, 2])
        XCTAssertLessThan(
            MLX.abs(alone - extended[0..., 0..<3, 0..., 0..., 0...]).max().item(Float.self),
            1e-6
        )
        let firstFrame = prefix[0..., 0..<1, 0..., 0..., 0...].sum(axis: -1, keepDims: true)
        XCTAssertLessThan(
            MLX.abs(alone[0..., 0..<1, 0..., 0..., 0...] - firstFrame).max().item(Float.self),
            1e-6
        )
    }

    func testFrameGroupNormalizationDoesNotMixTimeOrBatchRows() {
        MLXRandom.seed(92)
        let normalization = MiniMaxH3FrameGroupNorm(groups: 2, channels: 4)
        let input = MLXRandom.normal([2, 3, 2, 2, 4])
        let together = normalization(input)
        let oneFrame = normalization(input[1..<2, 1..<2, 0..., 0..., 0...])
        MLX.eval(together, oneFrame)
        XCTAssertLessThan(
            MLX.abs(together[1..<2, 1..<2, 0..., 0..., 0...] - oneFrame).max().item(Float.self),
            1e-6
        )
    }

    func testSharedVocoderPreservesBatchIsolationAndTemporalExpansion() {
        MLXRandom.seed(93)
        let vocoder = MMAudioBigVGAN(
            inputChannels: 2,
            initialChannels: 8,
            upsampleRates: [2, 2],
            upsampleKernelSizes: [4, 4],
            useFloat32: true
        )
        let input = MLXRandom.normal([2, 12, 2])
        let together = vocoder(input)
        let separate = MLX.concatenated([
            vocoder(input[0..<1, 0..., 0...]),
            vocoder(input[1..<2, 0..., 0...]),
        ], axis: 0)
        MLX.eval(together, separate)
        XCTAssertEqual(together.shape, [2, 48, 1])
        XCTAssertEqual(together.dtype, .float32)
        XCTAssertTrue(MLX.isFinite(together).all().item(Bool.self))
        XCTAssertLessThanOrEqual(MLX.abs(together).max().item(Float.self), 1)
        XCTAssertLessThan(MLX.abs(together - separate).max().item(Float.self), 1e-5)
    }
}
