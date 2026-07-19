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

    func testWan21LatentNormalizationAndPixelPacking() {
        let configuration = Wan2VAEConfiguration.wan21
        let model = Wan2VAEModel(configuration: configuration)
        let latents = MLX.zeros([1, 1, 1, 1, 16])
        let roundTrip = model.denormalizeLatents(model.normalizeLatents(latents))
        let pixels = MLXArray((0..<48).map(Float.init), [1, 1, 4, 4, 3])
        let packed = Wan2VAEModel.patchify(pixels, patchSize: configuration.imagePatchSize)
        let unpacked = Wan2VAEModel.unpatchify(packed, patchSize: configuration.imagePatchSize)
        eval(roundTrip, unpacked)

        XCTAssertEqual(packed.shape, pixels.shape)
        XCTAssertEqual(unpacked.asArray(Float.self), pixels.asArray(Float.self))
        XCTAssertTrue(roundTrip.asArray(Float.self).allSatisfy { abs($0) < 1e-6 })
    }

    func testWan21DecoderMatchesOfficialStageChannelTopology() {
        let decoder = Wan2VAEDecoder3D(
            baseDimensions: 4,
            latentChannels: 2,
            outputChannels: 3,
            useBlockShortcuts: false,
            resampleReducesChannels: true
        )
        let parameters = Dictionary(uniqueKeysWithValues: decoder.parameters().flattened())

        XCTAssertEqual(parameters["upsamples.0.upsamples.3.resample_weight"]?.shape, [8, 3, 3, 16])
        XCTAssertEqual(parameters["upsamples.1.upsamples.0.shortcut.weight"]?.shape, [16, 1, 1, 1, 8])
        XCTAssertEqual(parameters["upsamples.1.upsamples.3.resample_weight"]?.shape, [8, 3, 3, 16])
        XCTAssertNil(parameters["upsamples.2.upsamples.0.shortcut.weight"])
        XCTAssertEqual(parameters["upsamples.2.upsamples.3.resample_weight"]?.shape, [4, 3, 3, 8])
        XCTAssertNil(parameters["upsamples.3.upsamples.0.shortcut.weight"])
    }

    func testTinyWan21VAEGeometry() {
        let configuration = Wan2VAEConfiguration(
            latentChannels: 16,
            encoderDimensions: 4,
            decoderDimensions: 4,
            imagePatchSize: 1,
            blockResampleShortcut: false,
            decoderResampleReducesChannels: true,
            latentMean: Array(repeating: 0, count: 16),
            latentStandardDeviation: Array(repeating: 1, count: 16)
        )
        let model = Wan2VAEModel(configuration: configuration)
        let latent = model.encodeImage(MLX.zeros([1, 1, 32, 32, 3]))
        let decoded = model.decode(MLX.zeros([1, 2, 4, 4, 16]))
        eval(latent, decoded)

        XCTAssertEqual(latent.shape, [1, 1, 4, 4, 16])
        XCTAssertEqual(decoded.shape, [1, 5, 32, 32, 3])
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
