import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunImageModels

final class QwenImage21Tests: MLXTestCase {
    private var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/QwenImage21")
    }

    private func decode<T: Decodable>(_ name: String, as type: T.Type) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: fixtures.appending(path: name)))
    }

    func testTransformerMatchesIndependentDiffusersReferenceAndCachedTrajectory() throws {
        let config = try decode("transformer-config.json", as: QwenImage21TransformerConfig.self)
        let weights = try MLX.loadArrays(url: fixtures.appending(path: "transformer.safetensors"))
        XCTAssertEqual(Set(weights.keys), Set(QwenImage21Transformer.weightShapes(config).keys))
        let model = try QwenImage21Transformer(config: config, arrays: weights)
        let data = try MLX.loadArrays(url: fixtures.appending(path: "transformer-results.safetensors"))
        let layout = try QwenImage21Layout(imageSlots: [false, true, true, false, false, true],
                                          imageShapes: [(2, 2), (2, 2), (2, 2)])
        XCTAssertEqual(layout.segments.map(\.range), [0..<1, 1..<5, 5..<9, 9..<11, 11..<15])
        let cache = QwenImage21PrefixCache(layerCount: config.numLayers)
        let first = try model(latents: data["latents"]!, text: data["text"]!, timestep: 0.75, layout: layout, cache: cache)
        assertClose(first, data["full"]!, tolerance: 2e-5)
        XCTAssertTrue(cache.isPopulated)
        let changed = concatenated([data["latents"]![0..., 0..<8, 0...], data["latents"]![0..., 8..., 0...] + 0.125], axis: 1)
        let second = try model(latents: changed, text: data["text"]!, timestep: 0.4, layout: layout, cache: cache)
        let uncached = try model(latents: changed, text: data["text"]!, timestep: 0.4, layout: layout)
        assertClose(second, data["cached"]!, tolerance: 2e-5)
        assertClose(second, uncached, tolerance: 2e-5)
        XCTAssertThrowsError(try QwenImage21Transformer(config: config, arrays: weights.filter { $0.key != "img_in.weight" }))
    }

    func testBFloat16TransformerMatchesReferenceTimestepRounding() throws {
        let config = try decode("transformer-config.json", as: QwenImage21TransformerConfig.self)
        let weights = try MLX.loadArrays(url: fixtures.appending(path: "transformer.safetensors")).mapValues { $0.asType(.bfloat16) }
        let model = try QwenImage21Transformer(config: config, arrays: weights)
        let data = try MLX.loadArrays(url: fixtures.appending(path: "transformer-results.safetensors"))
        let expected = try MLX.loadArrays(url: fixtures.appending(path: "transformer-bf16-results.safetensors"))["expected"]!
        let layout = try QwenImage21Layout(imageSlots: [false, true, true, false, false, true], imageShapes: [(2, 2), (2, 2), (2, 2)])
        let actual = try model(latents: data["latents"]!.asType(.bfloat16), text: data["text"]!.asType(.bfloat16), timestep: 0.413725, layout: layout)
        XCTAssertEqual(actual.dtype, .bfloat16)
        // BF16 kernels round intermediate products differently across CPU PyTorch and MLX.
        assertClose(actual.asType(.float32), expected.asType(.float32), tolerance: 0.02)
    }

    func testBFloat16QueryKeyNormRoundsBeforeLearnedScale() throws {
        let weights = try QwenImage21Weights(
            ["norm.weight": MLXArray([Float(1.19), 0.53, 1.7, -0.9]).asType(.bfloat16)],
            shapes: ["norm.weight": [4]]
        )
        let input = MLXArray([Float(0.132), 1.42, -2.51, 3.37]).reshaped(1, 4).asType(.bfloat16)
        // Independent PyTorch BF16 RMSNorm result, including the intermediate rounding.
        let expected = MLXArray([Float(0.0703125), 0.33984375, -1.9296875, -1.359375]).reshaped(1, 4)
        assertClose(weights.rms(input, "norm", epsilon: 1e-6).asType(.float32), expected, tolerance: 1e-7)
    }

    func testRGBAEncoderAndDecoderMatchIndependentDiffusersReference() throws {
        let config = try decode("vae-config.json", as: QwenImage21VAEConfig.self)
        let weights = try MLX.loadArrays(url: fixtures.appending(path: "vae.safetensors"))
        XCTAssertEqual(Set(weights.keys), Set(QwenImage21VAE.weightShapes(config).keys))
        let model = try QwenImage21VAE(config: config, arrays: weights)
        let data = try MLX.loadArrays(url: fixtures.appending(path: "vae-results.safetensors"))
        func nhwc(_ key: String) -> MLXArray { data[key]![0..., 0..., 0, 0..., 0...].transposed(0, 2, 3, 1) }
        assertClose(try model.encode(nhwc("pixels")), nhwc("encoded"), tolerance: 2e-4)
        let decoded = try model.decode(nhwc("latents"))
        XCTAssertEqual(decoded.shape, [1, 32, 32, 4])
        assertClose(decoded, nhwc("decoded"), tolerance: 2e-4)
    }

    func testNativeWeightSchemasMatchPinnedOfficialCheckpointHeaders() throws {
        let shapes = try decode("official-weight-shapes.json", as: [String: [String: [Int]]].self)
        let transformer = try decode("official-transformer-config.json", as: QwenImage21TransformerConfig.self)
        let vae = try decode("official-vae-config.json", as: QwenImage21VAEConfig.self)
        XCTAssertEqual(QwenImage21Transformer.weightShapes(transformer), shapes["transformer"])
        XCTAssertEqual(QwenImage21VAE.weightShapes(vae), shapes["vae"])
    }

    func testBFloat16VAEKeepsNormalizedLatentsAndPixelsInBFloat16() throws {
        let config = try decode("vae-config.json", as: QwenImage21VAEConfig.self)
        let weights = try MLX.loadArrays(url: fixtures.appending(path: "vae.safetensors"))
            .mapValues { $0.asType(.bfloat16) }
        let model = try QwenImage21VAE(config: config, arrays: weights)
        let latent = try model.encode(zeros([1, 32, 32, 4], dtype: .bfloat16))
        XCTAssertEqual(latent.dtype, .bfloat16)
        XCTAssertEqual(try model.decode(latent).dtype, .bfloat16)
    }

    func testLayoutRejectsInvalidSlotsAndUsesCenteredSpatialCoordinates() throws {
        XCTAssertThrowsError(try QwenImage21Layout(imageSlots: [false, true], imageShapes: [(4, 4)]))
        let layout = try QwenImage21Layout(imageSlots: [false, false, true], imageShapes: [(2, 2)])
        XCTAssertEqual(layout.positions, [[0, 0, 0], [1, 1, 1], [2, -1, -1], [2, -1, 0], [2, 0, -1], [2, 0, 0]])
        XCTAssertEqual(layout.prefixCount, 2)
    }

    private func assertClose(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertLessThanOrEqual(max(abs(actual - expected)).item(Float.self), tolerance, file: file, line: line)
    }
}
