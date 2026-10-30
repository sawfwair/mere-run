import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunImageModels

final class BiRefNetTests: MLXTestCase {
    private var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/Idol") }
    private func arrays(_ name: String) throws -> [String: MLXArray] { try MLX.loadArrays(url: root.appending(path: name + ".safetensors")) }
    private func close(_ actual: MLXArray, _ expected: MLXArray, _ tolerance: Float,
                       file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        let error = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
        XCTAssertLessThanOrEqual(error, tolerance, file: file, line: line)
    }
    private var config: BiRefNetConfiguration {
        .init(embedDim: 8, depths: [2, 2, 2, 2], heads: [2, 4, 8, 16], windowSize: 4, decoderIntermediate: 8, asppChannels: 4)
    }
    func testDeformableConvolutionMatchesTorchvisionAtBordersAndFractionalOffsets() throws {
        let fixture = try arrays("birefnet-results")
        for (kernel, stride) in [(1, 1), (3, 1), (7, 1), (3, 2)] {
            let prefix = "k\(kernel)s\(stride)"
            for dtype in [DType.float32, .float16] {
                let result = try BiRefNetOps.deformConv(fixture[prefix + "_image"]!.transposed(0, 2, 3, 1).asType(dtype),
                    offsets: fixture[prefix + "_offsets"]!.transposed(0, 2, 3, 1).asType(dtype),
                    mask: fixture[prefix + "_mask"]!.transposed(0, 2, 3, 1).asType(dtype),
                    weight: fixture[prefix + "_weight"]!.transposed(0, 2, 3, 1).asType(dtype),
                    stride: stride, padding: kernel / 2, chunkSize: 5)
                let expected = fixture[prefix + (dtype == .float32 ? "_output" : "_half")]!
                close(result.transposed(0, 3, 1, 2), expected, dtype == .float32 ? 1e-5 : 0.012)
            }
        }
    }
    func testSwinMultiscaleDecoderAndResizeMatchUpstream() throws {
        let fixture = try arrays("birefnet-results"), weights = try arrays("birefnet-tiny")
        let input = fixture["network_image"]!.transposed(0, 2, 3, 1)
        let model = try BiRefNet(weights: weights, configuration: config, dtype: .float32)
        close(try model(input).transposed(0, 3, 1, 2), fixture["network_fp32"]!, 2e-5)
        let half = try BiRefNet(weights: weights, configuration: config, dtype: .float16)
        close(try half(input).transposed(0, 3, 1, 2), fixture["network_fp16"]!, 0.002)
        let swin = BiRefNetSwin(weights: try BiRefNetWeights(weights, configuration: config, dtype: .float32), configuration: config)
        for (index, feature) in try swin(input).enumerated() { close(feature.transposed(0, 3, 1, 2), fixture["swin_\(index)"]!, 2e-5) }
        close(BiRefNetOps.resize(input, height: 39, width: 77).transposed(0, 3, 1, 2), fixture["resized"]!, 2e-5)
        close(BiRefNetOps.imagePatches(input, height: 8, width: 16).transposed(0, 3, 1, 2), fixture["patches"]!, 0)
    }
    func testOriginalWeightInventoryAndMalformedPackages() throws {
        let weights = try arrays("birefnet-tiny")
        XCTAssertEqual(Set(weights.keys), Set(BiRefNetWeights.shapes(config).keys))
        var invalid = weights
        invalid.removeValue(forKey: "bb.patch_embed.proj.weight")
        XCTAssertThrowsError(try BiRefNet(weights: invalid, configuration: config))
        invalid = weights; invalid["extra"] = MLXArray(1)
        XCTAssertThrowsError(try BiRefNet(weights: invalid, configuration: config))
    }
}
