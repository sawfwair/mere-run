import AudioSOVTranscribeModel
import CryptoKit
import Foundation
import MereRunCore
import MereRunMLXTestSupport
import MLX
import MLXNN
import XCTest
@testable import AudioSTT

final class SOVPackedWeightsTests: MLXTestCase {
    private let configData = Data(#"{"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":3,"intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"max_position_embeddings":128,"rms_norm_eps":0.000001,"vocab_size":32,"enable_moe_block":true,"num_experts":4,"top_k_experts":2,"moe_intermediate_size":64}"#.utf8)

    private func model() throws -> Gemma4TextCausalLM {
        Gemma4TextCausalLM(config: try JSONDecoder().decode(Gemma4TextConfig.self, from: configData))
    }

    private func source(at root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try configData.write(to: root.appendingPathComponent("config.json"))
        let full = try model()
        full.prepareForFullPrecisionExpertWeights()
        let parameters = Dictionary(uniqueKeysWithValues: full.parameters().flattened())
        var arrays: [String: MLXArray] = [:]
        for (key, value) in parameters {
            if key.contains(".switch_glu.up_proj.") { continue }
            if key.contains(".switch_glu.gate_proj.weight") {
                let up = try XCTUnwrap(parameters[key.replacingOccurrences(of: "gate_proj", with: "up_proj")])
                let name = "model." + key.replacingOccurrences(of: "switch_glu.gate_proj.weight", with: "gate_up_proj")
                arrays[name] = concatenated([value, up], axis: 1).asType(.bfloat16)
            } else if key.contains(".switch_glu.down_proj.weight") {
                arrays["model." + key.replacingOccurrences(of: "switch_glu.down_proj.weight", with: "down_proj")] = value.asType(.bfloat16)
            } else {
                arrays["model." + key] = key.contains(".router.") ? value : value.asType(.bfloat16)
            }
        }
        try MLX.save(arrays: arrays, url: root.appendingPathComponent("model.safetensors"))
    }

    private func hashes(_ model: Module) -> [String: String] {
        // SOV disables per-layer input embeddings. Generic Gemma retains unused,
        // randomly initialized slots for that feature; they are absent from the
        // checkpoint and deliberately excluded by both loader completeness checks.
        Dictionary(uniqueKeysWithValues: model.parameters().flattened().filter {
            !$0.0.contains("per_layer") && !$0.0.contains(".rope.")
        }.map { name, value in
            (name, SOVPackedWeights.digest(value.asData(access: .noCopyIfContiguous).data))
        })
    }

    func testStreamingAndSavedWeightsMatchLegacyPackingForMixedPrecisions() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try source(at: root)
        let bits = [4, 6, 8]
        let legacy = try model()
        legacy.prepareForFullPrecisionExpertWeights()
        try SOVTranscribeWeights.load(root: root, into: legacy, target: true)
        try legacy.prepareExperimentalExperts(groupSize: 64, layerBits: bits)
        let expected = hashes(legacy)
        let destination = root.appendingPathComponent("packed")
        let streamed = try model()
        try streamed.preparePackedExpertWeights(groupSize: 64, layerBits: bits)
        let converted = try SOVPackedWeights.prepare(root: root, destination: destination, model: streamed, groupSize: 64, layerBits: bits)
        XCTAssertFalse(converted.cacheHit)
        XCTAssertLessThan(converted.packedBytes, converted.sourceBytes)
        XCTAssertEqual(hashes(streamed), expected)
        let restored = try model()
        try restored.preparePackedExpertWeights(groupSize: 64, layerBits: bits)
        let reused = try SOVPackedWeights.prepare(root: root, destination: destination, model: restored, groupSize: 64, layerBits: bits)
        XCTAssertTrue(reused.cacheHit)
        XCTAssertEqual(reused.sourceFingerprint, converted.sourceFingerprint)
        XCTAssertEqual(hashes(restored), expected)
        let x = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        XCTAssertEqual(abs(legacy(x) - restored(x)).max().item(Float.self), 0)
        XCTAssertThrowsError(try SOVPackedWeights.prepare(root: root, destination: destination, model: restored, groupSize: 64, layerBits: [8, 8, 8]))
        try Data("changed".utf8).write(to: root.appendingPathComponent("config.json"))
        XCTAssertThrowsError(try SOVPackedWeights.prepare(root: root, destination: destination, model: restored, groupSize: 64, layerBits: bits))
    }

    func testFailedConversionDoesNotPublishPartialPackAndChangedShardIsRejected() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try source(at: root)
        let destination = root.appendingPathComponent("packed")
        let target = try model()
        try target.preparePackedExpertWeights(groupSize: 64, layerBits: [8, 8, 8])
        let complete = try Data(contentsOf: root.appendingPathComponent("model.safetensors"))
        try MLX.save(arrays: ["model.language_model.embed_tokens.weight": MLXArray.zeros([32, 64])],
                     url: root.appendingPathComponent("model.safetensors"))
        XCTAssertThrowsError(try SOVPackedWeights.prepare(root: root, destination: destination, model: target, groupSize: 64, layerBits: [8, 8, 8]))
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: root.path).contains { $0.hasSuffix(".partial") })
        try complete.write(to: root.appendingPathComponent("model.safetensors"))
        _ = try SOVPackedWeights.prepare(root: root, destination: destination, model: target, groupSize: 64, layerBits: [8, 8, 8])
        try Data().write(to: destination.appendingPathComponent("target-00001.safetensors"))
        XCTAssertThrowsError(try SOVPackedWeights.prepare(root: root, destination: destination, model: target, groupSize: 64, layerBits: [8, 8, 8]))
        XCTAssertTrue(FileManager.default.fileExists(atPath: destination.appendingPathComponent(SOVPackedWeights.manifestName).path))
    }

    func testInstalledPackedQ8MatchesEveryLegacyTargetParameter() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu", let sourcePath = env["MERERUN_TEST_SOV_PATH"],
              let packedPath = env["MERERUN_TEST_SOV_PACKED_PATH"] else {
            throw XCTSkip("Requires explicit original SOV and saved Q8 checkpoint paths on the GPU.")
        }
        let root = URL(fileURLWithPath: sourcePath), destination = URL(fileURLWithPath: packedPath)
        let config = try JSONDecoder().decode(SOVTranscribeConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        XCTAssertEqual(config.textConfig.hiddenSizePerLayerInput, 0)
        let expected: [String: String] = try autoreleasepool {
            let legacy = Gemma4TextCausalLM(config: config.textConfig)
            legacy.prepareForFullPrecisionExpertWeights()
            try SOVTranscribeWeights.load(root: root, into: legacy, target: true)
            try legacy.prepareExperimentalExperts(groupSize: 64, bits: 8, metalVerification: true)
            return hashes(legacy)
        }
        Memory.clearCache()
        let target = Gemma4TextCausalLM(config: config.textConfig)
        let bits = Array(repeating: 8, count: config.textConfig.numHiddenLayers)
        try target.preparePackedExpertWeights(groupSize: 64, layerBits: bits, metalVerification: true)
        let result = try SOVPackedWeights.prepare(root: root, destination: destination, model: target, groupSize: 64, layerBits: bits)
        XCTAssertTrue(result.cacheHit)
        let actual = hashes(target)
        let mismatches = Set(actual.keys).union(expected.keys).filter { actual[$0] != expected[$0] }.sorted()
        XCTAssertEqual(mismatches, [])
        if let output = env["MERERUN_TEST_SOV_PACKED_RECEIPT"] {
            struct Receipt: Encodable {
                let sourceFingerprint: String
                let legacy: [String: String]
                let packed: [String: String]
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
            try encoder.encode(Receipt(sourceFingerprint: result.sourceFingerprint, legacy: expected, packed: actual))
                .write(to: URL(fileURLWithPath: output), options: .atomic)
        }
        if mismatches.isEmpty {
            print("SOV packed Q8 verified \(expected.count) checkpoint parameter byte digests; source \(result.sourceFingerprint)")
        }
    }
}
