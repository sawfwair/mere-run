import MLX
@testable import MereRunCore
import XCTest

final class LTXGemmaTextEncoderV2Tests: XCTestCase {
    func testGemma3PartialTextConfigUsesLTX23Defaults() throws {
        let data = Data("""
        {
          "model_type": "gemma3",
          "text_config": {
            "hidden_size": 3840,
            "intermediate_size": 15360,
            "model_type": "gemma3_text",
            "num_attention_heads": 16,
            "num_hidden_layers": 48,
            "num_key_value_heads": 8,
            "rope_scaling": {
              "factor": 8.0,
              "rope_type": "linear"
            },
            "sliding_window": 1024
          }
        }
        """.utf8)

        let topConfig = try JSONDecoder().decode(LTXGemmaTopConfig.self, from: data)
        let textConfig = try XCTUnwrap(topConfig.textConfig)
        let modelConfig = LTXGemmaModelConfig(textConfig: textConfig)

        XCTAssertEqual(modelConfig.vocabSize, 262_208)
        XCTAssertEqual(modelConfig.headDim, 256)
        XCTAssertEqual(modelConfig.rmsNormEps, 1e-6)
        XCTAssertEqual(modelConfig.ropeTheta, 1_000_000)
        XCTAssertEqual(modelConfig.queryPreAttnScalar, 256)
        XCTAssertEqual(modelConfig.slidingWindowPattern, 6)
    }

    func testLTX23ConnectorMapperStripsConnectorPrefixAndKeepsGateWeights() {
        let gate = MLXArray.zeros([32, 4096])
        let mapped = mapLTX23TextConnectorWeight(
            key: "connector.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.weight",
            value: gate,
            dtype: .float32
        )

        XCTAssertEqual(mapped.count, 1)
        XCTAssertEqual(
            mapped[0].0,
            "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.weight"
        )
        XCTAssertEqual(mapped[0].1.shape, [32, 4096])
    }

    func testGemmaLanguageMapperAcceptsStrippedQuantizedKeys() {
        let mapped = mapGemmaLanguageWeight(
            key: "model.layers.0.self_attn.q_proj.weight",
            value: MLXArray.zeros([4096, 480], dtype: .uint32),
            dtype: .bfloat16
        )

        XCTAssertEqual(mapped.map(\.0), ["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertEqual(mapped[0].1.dtype, .uint32)
    }

    func testGemmaIndexDetectsQuantizedScaleKeys() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-gemma-index-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        let index = """
        {
          "metadata": {"total_size": 1},
          "weight_map": {
            "language_model.model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
            "language_model.model.layers.0.self_attn.q_proj.scales": "model-00001.safetensors"
          }
        }
        """
        try Data(index.utf8).write(to: indexURL)

        XCTAssertTrue(gemmaIndexContainsQuantizedWeights(indexURL: indexURL))
        XCTAssertEqual(
            mapGemmaLanguageWeightKey("language_model.model.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        )
    }

    func testLTX23ConnectorMapperNormalizesListWrappedOutputAndFeedForwardKeys() {
        let out = mapLTX23TextConnectorWeight(
            key: "connector.audio_embeddings_connector.transformer_1d_blocks.7.attn1.to_out.0.bias",
            value: MLXArray.zeros([2048]),
            dtype: .float32
        )
        let ff = mapLTX23TextConnectorWeight(
            key: "connector.audio_embeddings_connector.transformer_1d_blocks.7.ff.net.0.proj.weight",
            value: MLXArray.zeros([8192, 2048]),
            dtype: .float32
        )

        XCTAssertEqual(
            out.map(\.0),
            ["audio_embeddings_connector.transformer_1d_blocks.7.attn1.to_out.bias"]
        )
        XCTAssertEqual(
            ff.map(\.0),
            ["audio_embeddings_connector.transformer_1d_blocks.7.ff.proj_in.weight"]
        )
    }

    func testNormalizeAndConcatHiddenStatesV2UsesPerTokenRMSAndDInterleavedFlattening() {
        let layer0 = MLXArray(
            [
                Float(100), Float(100),
                Float(3), Float(4),
                Float(0), Float(2),
            ],
            [1, 3, 2]
        )
        let layer1 = MLXArray(
            [
                Float(200), Float(200),
                Float(6), Float(8),
                Float(1), Float(0),
            ],
            [1, 3, 2]
        )
        let mask = MLXArray([Int32(0), Int32(1), Int32(1)], [1, 3])

        let normalized = normalizeAndConcatHiddenStatesV2(
            hiddenStates: [layer0, layer1],
            attentionMask: mask,
            dtype: .float32
        )
        let values = normalized.asArray(Float.self)

        XCTAssertEqual(normalized.shape, [1, 3, 4])
        XCTAssertEqual(Array(values[0..<4]), [0, 0, 0, 0])

        XCTAssertEqual(values[4], 0.848528, accuracy: 1e-5)
        XCTAssertEqual(values[5], 0.848528, accuracy: 1e-5)
        XCTAssertEqual(values[6], 1.131370, accuracy: 1e-5)
        XCTAssertEqual(values[7], 1.131370, accuracy: 1e-5)
    }
}
