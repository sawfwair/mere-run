import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunSheetSageModel

final class SheetSage2CheckpointTests: MLXTestCase {
    func testReleasedWeightsMatchBoundedUpstreamCPUReference() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["MERERUN_SHEETSAGE2_ROOT"],
              let referencePath = environment["MERERUN_SHEETSAGE2_REFERENCE"] else {
            throw XCTSkip("Set MERERUN_SHEETSAGE2_ROOT and MERERUN_SHEETSAGE2_REFERENCE for trained-weight parity.")
        }
        let root = URL(fileURLWithPath: modelPath)
        let configuration = try JSONDecoder().decode(SheetSage2Configuration.self,
            from: Data(contentsOf: root.appendingPathComponent("config.json")))
        try configuration.validate()
        let model = try SheetSage2Model(configuration: configuration,
            parent: MLX.loadArrays(url: root.appendingPathComponent("mert2/model.safetensors")),
            adapter: MLX.loadArrays(url: root.appendingPathComponent("model.safetensors")))
        let expected = try MLX.loadArrays(url: URL(fileURLWithPath: referencePath))
        var measurements: [String: Float] = [:]
        func compare(_ actual: MLXArray, _ name: String, tolerance: Float) throws {
            let reference = try XCTUnwrap(expected[name])
            XCTAssertEqual(actual.shape, reference.shape, name)
            guard actual.shape == reference.shape else { return }
            let error = max(abs(actual - reference)).item(Float.self)
            measurements[name] = error
            XCTAssertTrue(error.isFinite, name)
            XCTAssertLessThanOrEqual(error, tolerance, name)
        }
        let audio = try XCTUnwrap(expected["audio"])
        let mel = try model.encoder.mel(audio)
        try compare(mel, "mel", tolerance: 3e-4)
        let subsampled = try model.encoder.subsample(mel)
        try compare(subsampled, "subsampled", tolerance: 1e-3)
        try compare(model.encoder.block(subsampled, index: 0), "block0", tolerance: 1e-3)
        let memory = try model.encode(audio, windowSamples: audio.size)
        try compare(memory, "encoder", tolerance: 3e-3)
        let tokens = try XCTUnwrap(expected["tokens"]).asArray(Int32.self).map(Int.init)
        let cache = SheetSage2DecoderCache()
        try compare(model.logits(tokens, memory: memory, cache: cache), "prefill", tolerance: 3e-3)
        let next = try XCTUnwrap(expected["next_token"]).asArray(Int32.self).map(Int.init)
        try compare(model.logits(next, memory: memory, cache: cache), "cached", tolerance: 3e-3)
        try compare(model.logits(tokens + next, memory: memory, cache: SheetSage2DecoderCache()), "uncached", tolerance: 3e-3)
        if let output = environment["MERERUN_SHEETSAGE2_MEASUREMENTS"] {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(measurements).write(to: URL(fileURLWithPath: output), options: .atomic)
        }
    }
}
