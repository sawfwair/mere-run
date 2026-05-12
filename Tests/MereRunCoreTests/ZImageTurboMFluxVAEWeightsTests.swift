import XCTest
@testable import MereRunCore

final class ZImageTurboMFluxVAEWeightsTests: XCTestCase {
    func testMapsMFluxDecoderWrapperKeysToSwiftVAEKeys() {
        XCTAssertEqual(
            ZImageTurboMFluxVAEWeights.mapKey("decoder.conv_in.conv.weight"),
            "decoder.conv_in.weight"
        )
        XCTAssertEqual(
            ZImageTurboMFluxVAEWeights.mapKey("decoder.conv_out.conv.bias"),
            "decoder.conv_out.bias"
        )
        XCTAssertEqual(
            ZImageTurboMFluxVAEWeights.mapKey("decoder.conv_norm_out.norm.weight"),
            "decoder.conv_norm_out.weight"
        )
    }

    func testMapsMFluxEncoderWrapperKeysToSwiftVAEKeys() {
        XCTAssertEqual(
            ZImageTurboMFluxVAEWeights.mapKey("encoder.conv_in.conv2d.weight"),
            "encoder.conv_in.weight"
        )
        XCTAssertEqual(
            ZImageTurboMFluxVAEWeights.mapKey("encoder.conv_out.conv2d.bias"),
            "encoder.conv_out.bias"
        )
        XCTAssertEqual(
            ZImageTurboMFluxVAEWeights.mapKey("encoder.conv_norm_out.norm.bias"),
            "encoder.conv_norm_out.bias"
        )
    }

    func testLeavesNestedVAEBlockKeysUnchanged() {
        let key = "decoder.up_blocks.0.resnets.0.conv1.weight"
        XCTAssertEqual(ZImageTurboMFluxVAEWeights.mapKey(key), key)
    }
}
