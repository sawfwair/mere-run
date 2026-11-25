import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore
@testable import MereRunQwenModel

final class Bonsai25Tests: MereRunCoreTestCase {
    func testManagedPackPinsVisionAndDraftFiles() throws {
        let id = Q35Resources.bonsai25ModelId
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))
        XCTAssertEqual(spec.upstreamRepoId, "prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit")
        XCTAssertEqual(spec.hubFallback?.revision, "aa432fd0e17612f6684e728d4d3c42b6656bcabb")
        let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: id))
        XCTAssertEqual(profile.contextWindow, 262_144)
        XCTAssertEqual(profile.thinkingLevels, [.off, .low, .medium, .xhigh])
        XCTAssertEqual(Q35Resources.nativeReasoningEffort(for: 0.5, modelId: id), "medium")
        XCTAssertNil(Q35Resources.nativeReasoningEffort(for: 0.5, modelId: Q35Resources.bonsai2ModelId))
        XCTAssertEqual(MereRunModelManifest.template(for: .bonsai25).quantization?.groupSize, 128)

        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("mtp"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let required = [
            "config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja",
            "hadamard.json", "generation_config.json", "preprocessor_config.json",
            "mtp/mtp-v2-head.safetensors", "mtp/head-manifest.json",
        ]
        for path in required {
            XCTAssertTrue(spec.hubFallback?.patterns.contains(path) == true, path)
            try Data().write(to: root.appendingPathComponent(path))
        }
        XCTAssertTrue(spec.missingPaths(in: root).isEmpty)
        try FileManager.default.removeItem(at: root.appendingPathComponent("mtp/mtp-v2-head.safetensors"))
        XCTAssertEqual(spec.missingPaths(in: root), [root.appendingPathComponent("mtp/mtp-v2-head.safetensors")])
    }

    func testDraftRemainsOptInWithOneDefaultProposal() {
        let id = Q35Resources.bonsai25ModelId
        XCTAssertEqual(Q35Generator.mtpBlockSize(modelId: id, environment: [:]), 2)
        XCTAssertFalse(Q35Generator.shouldSpeculate(
            modelId: id, usesMoE: false, promptTokenCount: 32, maxContextTokens: 4096, environment: [:]
        ))
        XCTAssertTrue(Q35Generator.shouldSpeculate(
            modelId: id, usesMoE: false, promptTokenCount: 32, maxContextTokens: 4096,
            environment: ["MERERUN_Q35_MTP_SPECULATION": "1"]
        ))
    }

    func testDraftLoadsAffineQ4AndDirectNormScales() throws {
        let config = try JSONDecoder().decode(Q35Config.self, from: Data("""
        {"model_type":"prism_hadamard_qwen35","text_config":{
         "model_type":"qwen3_5_text","hidden_size":64,"num_hidden_layers":1,
         "intermediate_size":128,"layer_types":["full_attention"],"num_attention_heads":2,
         "num_key_value_heads":1,"head_dim":32,"vocab_size":8,"max_position_embeddings":1024,
         "rms_norm_eps":0.000001,"attn_output_gate":true,"attention_bias":false,"attention_dropout":0,
         "linear_num_value_heads":2,"linear_num_key_heads":1,"linear_key_head_dim":32,
         "linear_value_head_dim":32,"linear_conv_kernel_dim":4,
         "rope_parameters":{"rope_theta":10000,"partial_rotary_factor":0.25}}}
        """.utf8))
        let draft = Q35MTPModel(config: config)
        var arrays: [String: MLXArray] = [:]
        for name in ["attn_k_norm", "attn_q_norm", "attn_norm", "post_attention_norm",
                     "nextn.enorm", "nextn.hnorm", "nextn.shared_head_norm"] {
            let width = ["attn_k_norm", "attn_q_norm"].contains(name) ? 32 : 64
            arrays[name + ".weight"] = MLXArray(Array(repeating: Float(0.0001), count: width))
        }
        let projections: [(String, Int, Int)] = [
            ("attn_k", 32, 64), ("attn_v", 32, 64), ("attn_q", 128, 64), ("attn_output", 64, 64),
            ("ffn_down", 64, 128), ("ffn_gate", 128, 64), ("ffn_up", 128, 64), ("nextn.eh_proj", 64, 128),
        ]
        for (name, rows, width) in projections {
            arrays[name + ".weight"] = MLXArray.zeros([rows, width / 8], dtype: .uint32)
            arrays[name + ".biases"] = MLXArray.zeros([rows, width / 32], dtype: .float16)
            arrays[name + ".scales"] = MLXArray.ones([rows, width / 32], dtype: .float16)
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try MLX.save(arrays: arrays, url: url)
        try Q35Generator.loadBonsai25MTPWeights(into: draft, from: url)

        let leaves = Dictionary(uniqueKeysWithValues: draft.leafModules().flattened())
        let fc = try XCTUnwrap(leaves["fc"] as? PortableQuantizedLinear)
        XCTAssertEqual(fc.bits, 4)
        XCTAssertEqual(fc.groupSize, 32)
        let norm = try XCTUnwrap(leaves["pre_fc_norm_embedding"] as? Q35RMSNorm)
        XCTAssertEqual(norm.weight.dtype, .float16)
        XCTAssertEqual(norm.resolvedWeight()[0].item(Float.self), Float(Float16(0.0001)))

        arrays["attn_q.scales"] = MLXArray.ones([128, 1], dtype: .float16)
        XCTAssertThrowsError(try Q35Generator.bonsai25MTPWeights(
            arrays, expectedParameters: Q35MTPModel(config: config).parameters().flattened()
        ))
    }
}
