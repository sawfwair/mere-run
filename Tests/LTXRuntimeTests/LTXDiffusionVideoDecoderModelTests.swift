import Foundation
import MLX
import MLXNN
import MLXRandom
import MereRunMLXTestSupport
@testable import MereRunLTXModel
import XCTest

final class LTXDiffusionVideoDecoderModelTests: MLXTestCase {
    func testNativeParameterLayoutMatchesOfficialDecoderInventory() {
        let parameters = Dictionary(
            uniqueKeysWithValues: LTXDiffusionVideoDecoder().parameters().flattened()
        )
        XCTAssertEqual(parameters.count, 407)
        XCTAssertEqual(parameters["conv_in.weight"]?.shape, [2_048, 128])
        XCTAssertEqual(parameters["det_stages.0.blocks.0.attn.to_q.weight"]?.shape, [2_048, 2_048])
        XCTAssertEqual(parameters["upsamples.3.proj.weight"]?.shape, [2_048, 512])
        XCTAssertEqual(parameters["diff_blocks.7.scale_shift_table"]?.shape, [7, 256])
        XCTAssertEqual(parameters["shared_adaln.proj.weight"]?.shape, [1_792, 384])
        XCTAssertEqual(parameters["conv_out.weight"]?.shape, [48, 256])
    }

    func testNattenShiftedBoundaryWindowsMatchOfficialSemantics() {
        let bounds = ltxDiffVAEWindowBounds(length: 8, kernel: 5)
        XCTAssertEqual(bounds.starts, [0, 0, 0, 1, 2, 3, 3, 3])
        XCTAssertEqual(bounds.ends, [5, 5, 5, 6, 7, 8, 8, 8])
    }

    func testMetalNeighborhoodAttentionMatchesMLXReference() throws {
        MLXRandom.seed(73)
        let shape = [1, 3, 4, 5, 1, 64]
        let sourceQuery = MLXRandom.normal(shape)
        let sourceKey = MLXRandom.normal(shape)
        let sourceValue = MLXRandom.normal(shape)
        for dtype in [DType.float32, .bfloat16, .float16] {
            let query = sourceQuery.asType(dtype)
                * MLXArray(1 / Float(64).squareRoot()).asType(dtype)
            let key = sourceKey.asType(dtype)
            let value = sourceValue.asType(dtype)
            guard let accelerated = LTXDiffVAEMetalNeighborhoodAttention.apply(
                query: query,
                key: key,
                value: value,
                kernel: (3, 3, 3)
            ) else {
                throw XCTSkip("The custom DiffVAE neighborhood-attention kernel requires a Metal GPU.")
            }
            let reference = ltxDiffVAENeighborhoodAttention(
                query: query,
                key: key,
                value: value,
                kernel: (3, 3, 3),
                scoreBudget: 1 << 25
            )
            MLX.eval(accelerated, reference)
            let maximumError = MLX.max(MLX.abs(accelerated - reference)).item(Float.self)
            XCTAssertLessThan(maximumError, dtype == .float32 ? 2e-4 : 2e-2, "\(dtype)")
        }
    }
}
