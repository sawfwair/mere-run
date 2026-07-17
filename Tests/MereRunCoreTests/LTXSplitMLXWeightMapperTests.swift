import MLX
@testable import MereRunCore
import XCTest

final class LTXSplitMLXWeightMapperTests: XCTestCase {
    func testVideoVAEDecoderPreservesSplitMLXConvLayout() {
        let mapped = mapLTXDecoderWeight(
            key: "vae_decoder.conv_in.conv.weight",
            value: MLXArray.zeros([1024, 3, 3, 3, 128]),
            dtype: .float32,
            sourceLayout: .mlx
        )

        XCTAssertEqual(mapped.map(\.0), ["conv_in.conv.weight"])
        XCTAssertEqual(mapped[0].1.shape, [1024, 3, 3, 3, 128])
    }

    func testVideoVAEDecoderTransposesLegacyPyTorchConvLayout() {
        let mapped = mapLTXDecoderWeight(
            key: "decoder.conv_in.conv.weight",
            value: MLXArray.zeros([1024, 128, 3, 3, 3]),
            dtype: .float32
        )

        XCTAssertEqual(mapped.map(\.0), ["conv_in.conv.weight"])
        XCTAssertEqual(mapped[0].1.shape, [1024, 3, 3, 3, 128])
    }

    func testAudioVAEDecoderPreservesSplitMLXConvLayout() {
        let mapped = mapAudioVaeDecoderWeight(
            key: "audio_vae.decoder.conv_in.conv.weight",
            value: MLXArray.zeros([512, 3, 3, 8]),
            dtype: .float32,
            sourceLayout: .mlx
        )

        XCTAssertEqual(mapped.map(\.0), ["conv_in.conv.weight"])
        XCTAssertEqual(mapped[0].1.shape, [512, 3, 3, 8])
    }

    func testAudioVAEDecoderMapsSplitMLXUnderscoreStatistics() {
        let mean = mapAudioVaeDecoderWeight(
            key: "audio_vae.per_channel_statistics._mean_of_means",
            value: MLXArray.zeros([128]),
            dtype: .float32,
            sourceLayout: .mlx
        )
        let std = mapAudioVaeDecoderWeight(
            key: "audio_vae.per_channel_statistics._std_of_means",
            value: MLXArray.ones([128]),
            dtype: .float32,
            sourceLayout: .mlx
        )

        XCTAssertEqual(mean.map(\.0), ["per_channel_statistics._mean_of_means"])
        XCTAssertEqual(std.map(\.0), ["per_channel_statistics._std_of_means"])
        XCTAssertEqual(mean[0].1.shape, [128])
        XCTAssertEqual(std[0].1.shape, [128])
    }

    func testAudioVAEEncoderPreservesSplitMLXConvLayout() {
        let mapped = mapAudioVaeEncoderWeight(
            key: "audio_vae.encoder.down.1.block.0.conv1.conv.weight",
            value: MLXArray.zeros([256, 3, 3, 128]),
            dtype: .float32,
            sourceLayout: .mlx
        )

        XCTAssertEqual(mapped.map(\.0), ["down.1.block.0.conv1.conv.weight"])
        XCTAssertEqual(mapped[0].1.shape, [256, 3, 3, 128])
    }

    func testAudioVAEEncoderTransposesPyTorchConvLayout() {
        let mapped = mapAudioVaeEncoderWeight(
            key: "audio_vae.encoder.conv_in.conv.weight",
            value: MLXArray.zeros([128, 2, 3, 3]),
            dtype: .float32
        )

        XCTAssertEqual(mapped.map(\.0), ["conv_in.conv.weight"])
        XCTAssertEqual(mapped[0].1.shape, [128, 3, 3, 2])
    }

    func testUpsamplerStripsSplitPrefixAndPreservesMLXConvLayout() {
        let mapped = mapLTXUpsamplerWeight(
            key: "spatial_upscaler_x2_v1_1.initial_conv.weight",
            value: MLXArray.zeros([1024, 3, 3, 3, 128]),
            dtype: .float32,
            sourceLayout: .mlx
        )

        XCTAssertEqual(mapped.map(\.0), ["initial_conv.weight"])
        XCTAssertEqual(mapped[0].1.shape, [1024, 3, 3, 3, 128])
    }

    func testUpsamplerMapsSequentialConvKeyToSwiftModuleName() {
        let mapped = mapLTXUpsamplerWeight(
            key: "spatial_upscaler_x2_v1_1.upsampler.0.weight",
            value: MLXArray.zeros([4096, 3, 3, 1024]),
            dtype: .float32,
            sourceLayout: .mlx
        )

        XCTAssertEqual(mapped.map(\.0), ["upsampler.conv.weight"])
        XCTAssertEqual(mapped[0].1.shape, [4096, 3, 3, 1024])
    }
}
