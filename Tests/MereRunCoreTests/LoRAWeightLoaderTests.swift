import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class LoRAWeightLoaderTests: MereRunCoreTestCase {
    func testLoadInfersPerLayerRanks() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let adapterURL = temp.appendingPathComponent("mixed-rank.safetensors")
        try MLX.save(
            arrays: [
                "transformer_blocks.0.attn.to_q.lora_down.weight": MLXArray.zeros([8, 4], dtype: .float32),
                "transformer_blocks.0.attn.to_q.lora_up.weight": MLXArray.zeros([4, 8], dtype: .float32),
                "transformer_blocks.0.ff.linear_in.lora_down.weight": MLXArray.zeros([16, 4], dtype: .float32),
                "transformer_blocks.0.ff.linear_in.lora_up.weight": MLXArray.zeros([4, 16], dtype: .float32),
            ],
            metadata: ["lora_alpha": "4"],
            url: adapterURL
        )

        let weights = try LoRAWeightLoader.load(from: adapterURL)

        XCTAssertEqual(weights.targetRanks["transformer_blocks.0.attn.to_q"], 8)
        XCTAssertEqual(weights.targetRanks["transformer_blocks.0.ff.linear_in"], 16)
        XCTAssertEqual(weights.alpha, 4)
    }

    func testLoadPreservesKrea2NativeGlobalKeys() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let adapterURL = temp.appendingPathComponent("krea-official-style.safetensors")
        try MLX.save(
            arrays: [
                "transformer.img_in.lora_A.weight": MLXArray.zeros([4, 64], dtype: .float32),
                "transformer.img_in.lora_B.weight": MLXArray.zeros([6144, 4], dtype: .float32),
                "transformer.final_layer.linear.lora_A.weight": MLXArray.zeros([4, 6144], dtype: .float32),
                "transformer.final_layer.linear.lora_B.weight": MLXArray.zeros([64, 4], dtype: .float32),
            ],
            url: adapterURL
        )

        let weights = try LoRAWeightLoader.load(from: adapterURL)

        XCTAssertNotNil(weights.weights["img_in"])
        XCTAssertNotNil(weights.weights["final_layer.linear"])
        XCTAssertNil(weights.weights["x_embedder"])
        XCTAssertNil(weights.weights["proj_out"])
    }

    func testLoadPreservesRepresentativeKrea2PublishedAdapterSurface() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let paths = [
            "img_in",
            "txt_in.linear_1",
            "txt_in.linear_2",
            "text_fusion.projector",
            "text_fusion.layerwise_blocks.0.attn.to_q",
            "text_fusion.layerwise_blocks.0.attn.to_k",
            "text_fusion.layerwise_blocks.0.attn.to_v",
            "text_fusion.layerwise_blocks.0.attn.to_gate",
            "text_fusion.layerwise_blocks.0.attn.to_out.0",
            "text_fusion.layerwise_blocks.0.ff.gate",
            "text_fusion.layerwise_blocks.0.ff.up",
            "text_fusion.layerwise_blocks.0.ff.down",
            "text_fusion.refiner_blocks.0.attn.to_q",
            "text_fusion.refiner_blocks.0.attn.to_k",
            "text_fusion.refiner_blocks.0.attn.to_v",
            "text_fusion.refiner_blocks.0.attn.to_gate",
            "text_fusion.refiner_blocks.0.attn.to_out.0",
            "text_fusion.refiner_blocks.0.ff.gate",
            "text_fusion.refiner_blocks.0.ff.up",
            "text_fusion.refiner_blocks.0.ff.down",
            "time_embed.linear_1",
            "time_embed.linear_2",
            "time_mod_proj",
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.to_k",
            "transformer_blocks.0.attn.to_v",
            "transformer_blocks.0.attn.to_gate",
            "transformer_blocks.0.attn.to_out.0",
            "transformer_blocks.0.ff.gate",
            "transformer_blocks.0.ff.up",
            "transformer_blocks.0.ff.down",
            "final_layer.linear",
        ]
        let adapterURL = temp.appendingPathComponent("krea-published-surface.safetensors")
        var arrays: [String: MLXArray] = [:]
        for path in paths {
            arrays["transformer.\(path).lora_A.weight"] = MLXArray.zeros([4, 8], dtype: .float32)
            arrays["transformer.\(path).lora_B.weight"] = MLXArray.zeros([8, 4], dtype: .float32)
        }
        try MLX.save(arrays: arrays, url: adapterURL)

        let weights = try LoRAWeightLoader.load(from: adapterURL)

        XCTAssertEqual(Set(weights.weights.keys), Set(paths))
        XCTAssertNil(weights.weights["x_embedder"])
        XCTAssertNil(weights.weights["proj_out"])
        XCTAssertEqual(weights.targetRanks["transformer_blocks.0.attn.to_q"], 4)
        XCTAssertEqual(weights.targetRanks["text_fusion.layerwise_blocks.0.ff.down"], 4)
    }

    func testLoadMapsFALKrea2AdapterKeys() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let adapterURL = temp.appendingPathComponent("fal-krea2-lora.safetensors")
        let sourceToExpected = [
            "base_model.model.first": "img_in",
            "base_model.model.txtmlp.1": "txt_in.linear_1",
            "base_model.model.txtmlp.3": "txt_in.linear_2",
            "base_model.model.txtfusion.projector": "text_fusion.projector",
            "base_model.model.tmlp.0": "time_embed.linear_1",
            "base_model.model.tmlp.2": "time_embed.linear_2",
            "base_model.model.tproj.1": "time_mod_proj",
            "base_model.model.last.linear": "final_layer.linear",
            "base_model.model.blocks.0.attn.wq": "transformer_blocks.0.attn.to_q",
            "base_model.model.blocks.0.attn.wk": "transformer_blocks.0.attn.to_k",
            "base_model.model.blocks.0.attn.wv": "transformer_blocks.0.attn.to_v",
            "base_model.model.blocks.0.attn.gate": "transformer_blocks.0.attn.to_gate",
            "base_model.model.blocks.0.attn.wo": "transformer_blocks.0.attn.to_out.0",
            "base_model.model.blocks.0.mlp.gate": "transformer_blocks.0.ff.gate",
            "base_model.model.blocks.0.mlp.up": "transformer_blocks.0.ff.up",
            "base_model.model.blocks.0.mlp.down": "transformer_blocks.0.ff.down",
            "base_model.model.txtfusion.layerwise_blocks.0.attn.wq": "text_fusion.layerwise_blocks.0.attn.to_q",
            "base_model.model.txtfusion.refiner_blocks.1.mlp.down": "text_fusion.refiner_blocks.1.ff.down",
        ]
        var arrays: [String: MLXArray] = [:]
        for source in sourceToExpected.keys {
            arrays["\(source).lora_A.weight"] = MLXArray.zeros([4, 8], dtype: .float32)
            arrays["\(source).lora_B.weight"] = MLXArray.zeros([8, 4], dtype: .float32)
        }
        try MLX.save(arrays: arrays, metadata: ["lora_alpha": "32"], url: adapterURL)

        let weights = try LoRAWeightLoader.load(from: adapterURL)

        XCTAssertEqual(Set(weights.weights.keys), Set(sourceToExpected.values))
        XCTAssertNil(weights.weights["blocks.0.attn.wq"])
        XCTAssertNil(weights.weights["txtfusion.layerwise_blocks.0.attn.wq"])
        XCTAssertEqual(weights.alpha, 32)
        XCTAssertEqual(weights.targetRanks["transformer_blocks.0.attn.to_q"], 4)
        XCTAssertEqual(weights.targetRanks["text_fusion.refiner_blocks.1.ff.down"], 4)
    }

    func testLoadMapsDottedBFLKleinAdapterKeys() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let adapterURL = temp.appendingPathComponent("fal-klein-lora.safetensors")
        try MLX.save(
            arrays: [
                "base_model.model.double_blocks.0.img_attn.qkv.lora_A.weight": MLXArray.zeros([4, 8], dtype: .float32),
                "base_model.model.double_blocks.0.img_attn.qkv.lora_B.weight": MLXArray.zeros([24, 4], dtype: .float32),
                "base_model.model.double_blocks.1.txt_attn.proj.lora_A.weight": MLXArray.zeros([4, 8], dtype: .float32),
                "base_model.model.double_blocks.1.txt_attn.proj.lora_B.weight": MLXArray.zeros([8, 4], dtype: .float32),
                "base_model.model.single_blocks.2.linear1.lora_A.weight": MLXArray.zeros([4, 8], dtype: .float32),
                "base_model.model.single_blocks.2.linear1.lora_B.weight": MLXArray.zeros([32, 4], dtype: .float32),
                "base_model.model.single_blocks.3.linear2.lora_A.weight": MLXArray.zeros([4, 8], dtype: .float32),
                "base_model.model.single_blocks.3.linear2.lora_B.weight": MLXArray.zeros([8, 4], dtype: .float32),
            ],
            url: adapterURL
        )

        let weights = try LoRAWeightLoader.load(from: adapterURL)

        XCTAssertNotNil(weights.weights["transformer_blocks.0.attn.to_q"])
        XCTAssertNotNil(weights.weights["transformer_blocks.0.attn.to_k"])
        XCTAssertNotNil(weights.weights["transformer_blocks.0.attn.to_v"])
        XCTAssertNotNil(weights.weights["transformer_blocks.1.attn.to_add_out"])
        XCTAssertNotNil(weights.weights["single_transformer_blocks.2.attn.to_qkv_mlp_proj"])
        XCTAssertNotNil(weights.weights["single_transformer_blocks.3.attn.to_out"])
        XCTAssertNil(weights.weights["double_blocks.0.img_attn.qkv"])
        XCTAssertEqual(weights.targetRanks["transformer_blocks.0.attn.to_q"], 4)
        XCTAssertEqual(weights.targetRanks["single_transformer_blocks.2.attn.to_qkv_mlp_proj"], 4)
    }
}
