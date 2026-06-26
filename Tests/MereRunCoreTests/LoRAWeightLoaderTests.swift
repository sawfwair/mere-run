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
}
