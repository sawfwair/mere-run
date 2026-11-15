import Foundation
import AudioCodecs
import XCTest
import MLX
import MLXNN
import MereRunCore
import MereRunMLXTestSupport
@testable import AudioSTT

final class SOVTranscribeWeightsTests: MLXTestCase {
    func testShortWaveformZeroTailMatchesCompletePaddedTransform() throws {
        let samples = (0..<1050).map { sinf(Float($0)*0.17)*0.2 }
        let padded = samples + Array(repeating: Float(0), count: 480_000-samples.count)
        let reference = MelSpectrogram().extract(from: padded)
        let actual = try SOVTranscribeGenerator.audioFeatures(samples: samples)
        XCTAssertEqual(actual.shape, [1, 128, 3000])
        XCTAssertEqual((actual-reference).abs().max().item(Float.self), 0)
        let silent = try SOVTranscribeGenerator.audioFeatures(samples: [0])
        XCTAssertEqual(silent.shape, [1, 128, 3000])
        XCTAssertEqual(silent.min().item(Float.self), -1.5)
        XCTAssertEqual(silent.max().item(Float.self), -1.5)
    }

    func testWhisperFrontendMatchesIndependentNumpyAcrossWindowBoundary() throws {
        struct Point: Decodable { let window: Int; let channel: Int; let frame: Int; let value: Float }
        let url = try XCTUnwrap(Bundle.module.url(forResource: "frontend", withExtension: "json", subdirectory: "Fixtures/SOV"))
        let expected = try JSONDecoder().decode([Point].self, from: Data(contentsOf: url))
        var samples = [Float](repeating: 0, count: 480_160)
        for index in 0..<1600 {
            samples[index] = Float(0.1 * sin(2 * Double.pi * 440 * Double(index) / 16_000)
                + 0.03 * cos(2 * Double.pi * 1234 * Double(index) / 16_000))
        }
        for index in 0..<160 { samples[480_000 + index] = Float(0.8 * sin(2 * Double.pi * 700 * Double(index) / 16_000)) }
        let features = try SOVTranscribeGenerator.audioFeatures(samples: samples)
        XCTAssertEqual(features.shape, [2, 128, 3000])
        for point in expected {
            XCTAssertEqual(features[point.window, point.channel, point.frame].item(Float.self), point.value, accuracy: 0.0002)
        }
        XCTAssertThrowsError(try SOVTranscribeGenerator.audioFeatures(samples: [.nan]))
    }

    func testFusedExpertMappingKeepsGateUpOrderAndRouterPrecision() {
        let fused = MLXArray((0..<24).map(Float.init), [2, 4, 3]).asType(.bfloat16)
        let parts = SOVTranscribeWeights.mappedTarget("model.language_model.layers.0.experts.gate_up_proj", fused)
        XCTAssertEqual(parts.map(\.0), ["language_model.layers.0.experts.switch_glu.gate_proj.weight",
                                       "language_model.layers.0.experts.switch_glu.up_proj.weight"])
        XCTAssertEqual(parts[0].1.shape, [2, 2, 3])
        XCTAssertEqual(parts[0].1.asType(.float32).asArray(Float.self), [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17])
        XCTAssertEqual(parts[1].1.asType(.float32).asArray(Float.self), [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23])
        let router = SOVTranscribeWeights.mappedTarget("model.language_model.layers.0.router.scale", MLXArray.ones([8]))
        XCTAssertEqual(router.first?.1.dtype, .float32)
    }

    func testLoaderRejectsMissingWeightsAndWrongShapes() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("model.safetensors")
        try MLX.save(arrays: ["other.weight": MLXArray.ones([2, 2])], url: file)
        XCTAssertThrowsError(try SOVTranscribeWeights.load(root: root, into: Linear(2, 2, bias: false), sourcePrefix: "prefix."))
        try MLX.save(arrays: ["prefix.weight": MLXArray.ones([3, 2])], url: file)
        XCTAssertThrowsError(try SOVTranscribeWeights.load(root: root, into: Linear(2, 2, bias: false), sourcePrefix: "prefix."))
        try MLX.save(arrays: ["prefix.weight": MLXArray.ones([2, 2]).asType(.bfloat16)], url: file)
        let linear = Linear(2, 2, bias: false)
        try SOVTranscribeWeights.load(root: root, into: linear, sourcePrefix: "prefix.")
        XCTAssertEqual(linear.weight.dtype, .bfloat16)
        XCTAssertEqual(linear(MLXArray.ones([1, 2]).asType(.bfloat16)).asType(.float32).asArray(Float.self), [2, 2])
    }
}
