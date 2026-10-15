import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunCore

final class Flux3VAETests: MLXTestCase {
    private var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/Flux3")
    }

    func testOriginalVAENamesAndShapesMatchNativeModel() throws {
        let source = try JSONDecoder().decode([String: [Int]].self,
            from: Data(contentsOf: fixtures.appending(path: "vae-source-shapes.json")))
        let model = AutoencoderKL(configuration: Flux3VAE.configuration)
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        var mapped: [String: [Int]] = [:]
        for (key, shape) in source where !key.hasPrefix("bn.") {
            let name = Ideogram4VAEWeights.mapKey(key)
            let target = try XCTUnwrap(expected[name], key)
            var converted = shape
            if shape.count == 4 {
                converted = target.count == 2 ? [shape[0], shape[1]] : [shape[0], shape[2], shape[3], shape[1]]
            }
            XCTAssertEqual(converted, target, key)
            XCTAssertNil(mapped[name], key)
            mapped[name] = converted
        }
        XCTAssertEqual(Set(mapped.keys), Set(expected.keys))
    }

    func testVAEEncodeDecodeAndPNGChannelOrderMatchPinnedBFLReference() throws {
        let arrays = try MLX.loadArrays(url: fixtures.appending(path: "vae-weights.safetensors"))
        let expected = try MLX.loadArrays(url: fixtures.appending(path: "vae-results.safetensors"))
        let configuration = VAEConfig(latentChannels: 32, scalingFactor: 1, shiftFactor: 0,
            blockOutChannels: [32, 32, 32, 32], useQuantConv: true, usePostQuantConv: true, attentionNormEps: 1e-6)
        let vae = try Flux3VAE(configuration: configuration, arrays: arrays)
        let encoded = vae.encode(expected["pixels"]!.expandedDimensions(axis: 0))
        XCTAssertEqual(encoded.shape, [4, 128])
        XCTAssertLessThanOrEqual(abs(encoded.asType(.float32) - expected["encoded"]!).max().item(Float.self), 0.004)
        // Decode identical inputs so encoder BF16 rounding doesn't hide decoder errors.
        let decoded = vae.decode(expected["decode_tokens"]!, height: 2, width: 2)
        XCTAssertEqual(decoded.shape, [1, 3, 32, 32])
        let media = try QwenImageIO.mediaImage(from: decoded)
        XCTAssertEqual(media.width, 32)
        XCTAssertEqual(media.height, 32)
        XCTAssertLessThanOrEqual(abs(decoded - expected["decoded"]!).max().item(Float.self), 2e-4)
        XCTAssertThrowsError(try Flux3VAE(configuration: configuration, arrays: arrays.filter { $0.key != "encoder.quant_conv.weight" }))
    }
}
