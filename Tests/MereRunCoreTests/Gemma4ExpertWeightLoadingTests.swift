import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class Gemma4ExpertWeightLoadingTests: MereRunCoreTestCase {
    func testBF16ExpertsPrefillAndDecodeFromIndexedAndSingleFiles() async throws {
        for indexed in [false, true] {
            for inference in [false, true] {
                let (root, config) = try fixture(indexed: indexed, quantized: false)
                defer { try? FileManager.default.removeItem(at: root) }
                let model = Gemma4TextCausalLM(config: config.textConfig)
                try await load(model, root: root, config: config, inference: inference)
                let scales = model.parameters().flattened().filter { $0.0.hasSuffix(".scales") }
                XCTAssertTrue(scales.isEmpty, "BF16 checkpoint retained quantized expert placeholders")
                guard scales.isEmpty else { continue }
                let matrices = model.parameters().flattened().filter {
                    $0.0.contains(".experts.") && $0.0.hasSuffix(".weight")
                }
                XCTAssertEqual(matrices.count, 6)
                XCTAssertTrue(matrices.allSatisfy { $0.1.dtype == .bfloat16 })
                // MLX's CPU gatherMM supports FP32; Metal exercises the loaded BF16 matrices.
                if Device.defaultDevice().deviceType == .cpu {
                    model.apply { $0.dtype.isFloatingPoint ? $0.asType(.float32) : $0 }
                }
                let tokens = MLXArray((0..<32).map(Int32.init)).reshaped(1, 32)
                let prefill = model(tokens)
                let cache = model.makeCache()
                let serial = MLX.concatenated((0..<32).map { index in
                    model(tokens[0..., index..<(index + 1)], cache: cache)
                }, axis: 1)
                MLX.eval(prefill, serial)
                XCTAssertEqual(prefill.shape, [1, 32, 64])
                XCTAssertTrue(prefill.asArray(Float.self).allSatisfy(\.isFinite))
                let difference = MLX.max(MLX.abs(prefill - serial)).item(Float.self)
                let magnitude = max(1, MLX.max(MLX.abs(prefill)).item(Float.self))
                XCTAssertLessThan(difference / magnitude, 0.01)
            }
        }
    }

    func testNVFP4ExpertsRetainCheckpointScalesAndRunPrefill() async throws {
        for inference in [false, true] {
            let (root, config) = try fixture(indexed: true, quantized: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let model = Gemma4TextCausalLM(config: config.textConfig)
            try await load(model, root: root, config: config, inference: inference)
            let scales = model.parameters().flattened().filter { $0.0.hasSuffix(".scales") }
            XCTAssertEqual(scales.count, 6)
            XCTAssertTrue(scales.allSatisfy { $0.1.dtype == .uint8 })
            let output = model(MLXArray((0..<32).map(Int32.init)).reshaped(1, 32))
            MLX.eval(output)
            XCTAssertEqual(output.shape, [1, 32, 64])
            XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
        }
    }

    private func load(
        _ model: Gemma4TextCausalLM, root: URL, config: Gemma4Config, inference: Bool
    ) async throws {
        let resources = Gemma4Resources(rootURL: root)
        if inference {
            try await Gemma4Generator().loadWeights(into: model, from: resources, config: config)
        } else {
            try Gemma4TextModelLoader.loadWeights(into: model, from: resources, config: config)
        }
    }

    private func fixture(indexed: Bool, quantized: Bool) throws -> (URL, Gemma4Config) {
        let json = #"""
        {"model_type":"gemma4","text_config":{
          "model_type":"gemma4_text","hidden_size":32,"num_hidden_layers":2,
          "intermediate_size":32,"num_attention_heads":4,"num_key_value_heads":2,
          "head_dim":8,"max_position_embeddings":128,"rms_norm_eps":0.000001,
          "vocab_size":64,"vocab_size_per_layer_input":64,"hidden_size_per_layer_input":0,
          "sliding_window":32,"layer_types":["sliding_attention","full_attention"],
          "enable_moe_block":true,"num_experts":4,"top_k_experts":2,
          "moe_intermediate_size":16,"tie_word_embeddings":true}}
        """#
        let config = try JSONDecoder().decode(Gemma4Config.self, from: Data(json.utf8))
        MLXRandom.seed(71)
        let model = Gemma4TextCausalLM(config: config.textConfig)
        var weights: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() where !key.hasSuffix(".scales") {
            let mapped = "language_model.model." + key.dropFirst("language_model.".count)
            if quantized && key.contains(".experts.") && key.hasSuffix(".weight") {
                let (packed, scales, _) = MLX.quantized(value, groupSize: 16, bits: 4, mode: .nvfp4)
                weights[mapped] = packed
                weights[String(mapped.dropLast("weight".count)) + "scales"] = scales
            } else {
                weights[mapped] = value.asType(.bfloat16)
            }
        }
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        let resources = Gemma4Resources(rootURL: root)
        try MLX.save(arrays: weights, url: resources.modelWeightsURL)
        if indexed {
            let index = HFSafetensorsIndex(metadata: nil,
                weightMap: weights.mapValues { _ in resources.modelWeightsURL.lastPathComponent })
            try JSONEncoder().encode(index).write(to: resources.modelIndexURL)
        }
        return (root, config)
    }
}
