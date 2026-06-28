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
}
