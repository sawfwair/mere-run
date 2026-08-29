import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class Q35VisionQuantizedLoadingTests: MereRunCoreTestCase {
    func testShardedVisionLoadsQuantizedEmbeddingAndLinearWithoutLanguageShards() throws {
        try checkQuantizedLoading(sharded: true, prefix: "visual.")
    }

    func testSingleFileVisionLoadsQuantizedAliases() throws {
        try checkQuantizedLoading(sharded: false, prefix: "model.vision_tower.")
    }

    func testUnquantizedVisionStillRejectsWrongShape() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        try MLX.save(arrays: ["visual.pos_embed.weight": MLXArray.zeros([16, 8])],
                     url: root.appendingPathComponent("model.safetensors"))
        let tower = Q35VisionTower(config: try config())
        XCTAssertThrowsError(try tower.loadWeights(from: Q35Resources(rootURL: root)))
        XCTAssertFalse(tower.isLoaded)
    }

    private func checkQuantizedLoading(sharded: Bool, prefix: String) throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let tower = Q35VisionTower(config: try config())
        var source: [String: MLXArray] = [:]
        for (key, value) in tower.parameters().flattened() {
            source[prefix + key.dropFirst("visionTower.".count)] = value.asType(.bfloat16)
        }
        let embeddingKey = prefix + "pos_embed.weight"
        let embedding = try XCTUnwrap(source[embeddingKey])
        let packed = quantized(embedding, groupSize: 32, bits: 4)
        source[embeddingKey] = packed.0
        source[prefix + "pos_embed.scales"] = packed.1
        source[prefix + "pos_embed.biases"] = packed.2
        let linearKey = prefix + "blocks.0.mlp.fc1.weight"
        let linearPacked = quantized(try XCTUnwrap(source[linearKey]), groupSize: 64, bits: 8)
        source[linearKey] = linearPacked.0
        source[prefix + "blocks.0.mlp.fc1.scales"] = linearPacked.1
        source[prefix + "blocks.0.mlp.fc1.biases"] = linearPacked.2
        // The loader must ignore language tensors even when they share a shard.
        source["language_model.invalid.weight"] = MLXArray.zeros([1])
        try MLX.save(arrays: source, url: root.appendingPathComponent("model.safetensors"))
        if sharded {
            var weightMap = Dictionary(uniqueKeysWithValues: source.keys.map { ($0, "model.safetensors") })
            weightMap["language_model.unavailable.weight"] = "not-downloaded.safetensors"
            weightMap["visual.unindexed-marker.weight"] = "duplicate.safetensors"
            try MLX.save(arrays: [embeddingKey: MLXArray.zeros([1]),
                                 "visual.unindexed-marker.weight": MLXArray.zeros([1])],
                         url: root.appendingPathComponent("duplicate.safetensors"))
            try JSONEncoder().encode(Index(weightMap: weightMap))
                .write(to: root.appendingPathComponent("model.safetensors.index.json"))
        }

        try tower.loadWeights(from: Q35Resources(rootURL: root))

        XCTAssertTrue(tower.isLoaded)
        let modules = Dictionary(uniqueKeysWithValues: tower.leafModules().flattened())
        let loadedEmbedding = try XCTUnwrap(modules["visionTower.pos_embed"] as? PreQuantizedEmbedding)
        XCTAssertEqual(loadedEmbedding.bits, 4)
        XCTAssertEqual(loadedEmbedding.groupSize, 32)
        let loadedLinear = try XCTUnwrap(modules["visionTower.blocks.0.mlp.fc1"] as? PortableQuantizedLinear)
        XCTAssertEqual(loadedLinear.bits, 8)
        XCTAssertEqual(loadedLinear.groupSize, 64)
        let indices = MLXArray([Int32(0), 7, 15])
        let expected = dequantized(packed.0, scales: packed.1, biases: packed.2, groupSize: 32, bits: 4)
        XCTAssertEqual(loadedEmbedding(indices).asArray(Float.self), expected[indices].asArray(Float.self))
        let output = try tower.encodeImage(pixelValues: MLXArray.ones([1, 3, 8, 8]), gridTHW: (1, 4, 4))
        XCTAssertEqual(output.shape, [4, 64])
        XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
    }

    private struct Index: Encodable {
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    private func config() throws -> Q35Config {
        let json = """
        {
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5", "hidden_size": 64, "num_hidden_layers": 1,
            "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 1,
            "head_dim": 16, "layer_types": ["full_attention"], "linear_num_value_heads": 1,
            "linear_num_key_heads": 1, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 2, "max_position_embeddings": 128, "rms_norm_eps": 0.000001,
            "attention_bias": false, "attention_dropout": 0, "vocab_size": 32,
            "rope_parameters": {"rope_theta": 10000, "partial_rotary_factor": 1}
          },
          "vision_config": {
            "depth": 1, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4,
            "out_hidden_size": 64, "patch_size": 2, "temporal_patch_size": 2,
            "in_channels": 3, "spatial_merge_size": 2, "num_position_embeddings": 16
          }
        }
        """
        return try JSONDecoder().decode(Q35Config.self, from: Data(json.utf8))
    }
}
