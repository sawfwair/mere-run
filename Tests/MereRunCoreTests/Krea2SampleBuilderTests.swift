import MLX
import XCTest
@testable import MereRunCore

final class Krea2SampleBuilderTests: XCTestCase {
    func testAlignedResolutionUsesLatentPatchMultiple() {
        XCTAssertEqual(Krea2SampleBuilder.alignedResolution(width: 1024, height: 1024).width, 1024)
        XCTAssertEqual(Krea2SampleBuilder.alignedResolution(width: 1025, height: 1025).width, 1040)
        XCTAssertEqual(Krea2SampleBuilder.alignedResolution(width: 1025, height: 1025).height, 1040)
    }

    func testTurboTimestepsUseShiftedOneToZeroGrid() {
        let timesteps = Krea2SampleBuilder.timesteps(
            imageTokenCount: 64 * 64,
            steps: 8,
            mu: Krea2SampleBuilder.defaultMu
        )

        XCTAssertEqual(timesteps.count, 9)
        XCTAssertEqual(timesteps.first, 1.0)
        XCTAssertEqual(timesteps.last, 0.0)
        for (current, next) in zip(timesteps, timesteps.dropFirst()) {
            XCTAssertGreaterThanOrEqual(current, next)
        }
    }

    func testTextEncoderWeightMapperStripsKreaLanguageModelPrefix() {
        let value = MLXArray([Float(1)], [1])

        let mapped = Krea2ModelLoader.mapTextEncoderWeight(
            key: "language_model.embed_tokens.weight",
            value: value
        )

        XCTAssertEqual(mapped.map(\.0), ["embed_tokens.weight"])
    }

    func testAttentionMaskKeepsPaddedQueryRowsFinite() {
        let validMask = MLXArray([Int32(1), Int32(1), Int32(0)]).reshaped(1, 3)
        let mask = Krea2SampleBuilder.attentionMask(validMask: validMask, dtype: .float32)

        XCTAssertEqual(mask[0, 0, 2, 0].item(Float.self), 0)
        XCTAssertEqual(mask[0, 0, 2, 1].item(Float.self), 0)
        XCTAssertLessThan(mask[0, 0, 2, 2].item(Float.self), -1_000_000)
    }

    func testPaddedTextTokenInputsMatchOfficialConditionerShape() {
        let inputs = Krea2SampleBuilder.paddedTextTokenInputs(
            promptTokenIds: [10, 11, 12, 13, 14],
            suffixTokenIds: [20, 21, 22, 23, 24],
            padTokenId: 0,
            maxLength: 512,
            prefixDropCount: 34
        )

        XCTAssertEqual(inputs.tokenIds.count, 546)
        XCTAssertEqual(inputs.attentionMask.count, 546)
        XCTAssertEqual(Array(inputs.tokenIds[0..<5]), [10, 11, 12, 13, 14])
        XCTAssertEqual(Array(inputs.tokenIds[541..<546]), [20, 21, 22, 23, 24])
        XCTAssertEqual(Array(inputs.attentionMask[5..<541]).reduce(Int32(0), +), 0)
        XCTAssertEqual(inputs.tokenIds.count - 34, 512)
    }

    func testQwenActivationLayerIndicesConvertFromHiddenStateIndices() {
        XCTAssertEqual(
            Krea2SampleBuilder.qwenActivationLayerIndices(from: [2, 5, 8, 35]),
            [1, 4, 7, 34]
        )
    }

    func testVAEWeightMapperPreservesQwenPostQuantConv() {
        let value = MLXArray.zeros([16, 16, 1, 1, 1])

        let mapped = QwenImageEditVAE.weightMapper(key: "post_quant_conv.weight", value: value)

        XCTAssertEqual(mapped.map(\.0), ["postQuantConv.weight"])
    }

    func testVAEWeightMapperTransposesConv2DWeightsToMLXLayout() {
        let value = MLXArray.zeros([2, 3, 5, 7])

        let mapped = QwenImageEditVAE.weightMapper(key: "decoder.mid_block.attentions.0.proj.weight", value: value)

        XCTAssertEqual(mapped.map(\.0), ["decoder.mid_block.attentions.0.proj.weight"])
        XCTAssertEqual(mapped[0].1.shape, [2, 5, 7, 3])
    }
}
