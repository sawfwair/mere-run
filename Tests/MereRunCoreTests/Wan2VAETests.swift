import MLX
import XCTest
@testable import MereRunCore

final class Wan2VAETests: MereRunCoreTestCase {
    func testPatchifyRoundTripPreservesPixels() {
        let input = MLXArray((0..<48).map(Float.init), [1, 1, 4, 4, 3])
        let packed = Wan2VAEModel.patchify(input)
        let output = Wan2VAEModel.unpatchify(packed)
        eval(output)
        XCTAssertEqual(packed.shape, [1, 1, 2, 2, 12])
        XCTAssertEqual(output.shape, input.shape)
        XCTAssertEqual(output.asArray(Float.self), input.asArray(Float.self))
    }

    func testLatentNormalizationRoundTrip() {
        let input = MLX.zeros([1, 1, 1, 1, 48])
        let output = Wan2VAEModel.denormalize(Wan2VAEModel.normalize(input))
        eval(output)
        XCTAssertTrue(output.asArray(Float.self).allSatisfy { abs($0) < 1e-6 })
    }

    func testTinyVAEEncoderAndDecoderGeometry() {
        let model = Wan2VAEModel(latentChannels: 48, encoderDimensions: 4, decoderDimensions: 4)
        let image = MLX.zeros([1, 1, 32, 32, 3])
        let latent = model.encodeImage(image)
        let decoded = model.decode(MLX.zeros([1, 2, 2, 2, 48]))
        eval(latent, decoded)
        XCTAssertEqual(latent.shape, [1, 1, 2, 2, 48])
        XCTAssertEqual(decoded.shape, [1, 5, 32, 32, 3])
        XCTAssertTrue(decoded.asArray(Float.self).allSatisfy(\.isFinite))
    }
}
