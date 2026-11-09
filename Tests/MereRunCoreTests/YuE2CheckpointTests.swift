import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunCore

final class YuE2CheckpointTests: MLXTestCase {
    func testReleasedTransformerAndAcousticWeightsMatchUpstreamFP32() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["MERERUN_YUE2_ROOT"],
              let reference = environment["MERERUN_YUE2_CHECKPOINT_REFERENCE"] else {
            throw XCTSkip("Set MERERUN_YUE2_ROOT and MERERUN_YUE2_CHECKPOINT_REFERENCE for trained-weight parity.")
        }
        let root = URL(fileURLWithPath: path)
        let resources = YuE2Resources(rootURL: root)
        let model = try YuE2Model(configuration: resources.configuration(), arrays: SafetensorsStreamingLoader.loadArrays(
            url: root.appendingPathComponent("model.safetensors"), dtype: .float32))
        let expected = try MLX.loadArrays(url: URL(fileURLWithPath: reference))
        var errors: [String: Float] = [:]
        func compare(_ actual: MLXArray, _ key: String) throws {
            let value = try XCTUnwrap(expected[key])
            XCTAssertEqual(actual.shape, value.shape, key)
            guard actual.shape == value.shape else { return }
            let error = max(abs(actual - value)).item(Float.self)
            errors[key] = error
            XCTAssertTrue(error.isFinite, key)
            XCTAssertLessThanOrEqual(error, 3e-3, key)
        }
        let tokens = try XCTUnwrap(expected["tokens"]).asArray(Int32.self).map(Int.init)
        let cache = model.makeCache()
        try compare(model.logits(tokens, cache: cache), "prefill")
        try compare(model.logits([151856], cache: cache), "cached")
        try compare(model.logits(tokens + [151856], cache: model.makeCache()), "uncached")
        let acoustic = try YuE2Acoustic(model: model, tokens: tokens)
        try compare(acoustic.velocity(XCTUnwrap(expected["noise"]), rawTime: 0.4), "velocity")
        if let output = environment["MERERUN_YUE2_CHECKPOINT_MEASUREMENTS"] {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(errors).write(to: URL(fileURLWithPath: output), options: .atomic)
        }
    }
}
