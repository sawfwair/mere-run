import MLX
import XCTest
@testable import MereRunCore

final class Krea2LoRAInjectorTests: XCTestCase {
    func testDefaultTargetsKreaPublishedAdapterSurface() throws {
        let transformer = Krea2Transformer(configuration: try makeTinyConfig(numLayers: 2))

        let layers = try Krea2LoRAInjector.inject(into: transformer, rank: 4, alpha: 2.0)

        XCTAssertEqual(layers.count, 40)
        XCTAssertNotNil(layers["img_in"])
        XCTAssertNotNil(layers["txt_in.linear_1"])
        XCTAssertNotNil(layers["txt_in.linear_2"])
        XCTAssertNotNil(layers["text_fusion.projector"])
        XCTAssertNotNil(layers["time_embed.linear_1"])
        XCTAssertNotNil(layers["time_embed.linear_2"])
        XCTAssertNotNil(layers["time_mod_proj"])
        XCTAssertNotNil(layers["final_layer.linear"])
        XCTAssertNotNil(layers["transformer_blocks.0.attn.to_q"])
        XCTAssertNotNil(layers["transformer_blocks.0.attn.to_k"])
        XCTAssertNotNil(layers["transformer_blocks.0.attn.to_v"])
        XCTAssertNotNil(layers["transformer_blocks.0.attn.to_gate"])
        XCTAssertNotNil(layers["transformer_blocks.0.attn.to_out.0"])
        XCTAssertNotNil(layers["transformer_blocks.0.ff.gate"])
        XCTAssertNotNil(layers["transformer_blocks.0.ff.up"])
        XCTAssertNotNil(layers["transformer_blocks.0.ff.down"])
        XCTAssertNotNil(layers["text_fusion.layerwise_blocks.0.attn.to_gate"])
        XCTAssertNotNil(layers["text_fusion.refiner_blocks.0.attn.to_q"])
    }

    func testDefaultTargetsMatchOfficialFullModelModuleCount() throws {
        let transformer = Krea2Transformer(configuration: try makeTinyConfig(
            numLayers: 28,
            numLayerwiseTextBlocks: 2,
            numRefinerTextBlocks: 2
        ))

        let layers = try Krea2LoRAInjector.inject(into: transformer, rank: 32, alpha: 32.0)

        XCTAssertEqual(layers.count, 264)
    }

    func testLiteTargetsOnlyAttentionQAndVLayers() throws {
        let transformer = Krea2Transformer(configuration: try makeTinyConfig(numLayers: 3))

        let layers = try Krea2LoRAInjector.inject(
            into: transformer,
            rank: 2,
            targetSuffixes: Krea2LoRAInjector.liteTargetSuffixes
        )

        XCTAssertEqual(layers.count, 10)
        XCTAssertNotNil(layers["transformer_blocks.0.attn.to_q"])
        XCTAssertNotNil(layers["transformer_blocks.0.attn.to_v"])
        XCTAssertNotNil(layers["text_fusion.layerwise_blocks.0.attn.to_q"])
        XCTAssertNotNil(layers["text_fusion.refiner_blocks.0.attn.to_v"])
        XCTAssertNil(layers["transformer_blocks.0.attn.to_k"])
        XCTAssertNil(layers["transformer_blocks.0.ff.down"])
        XCTAssertNil(layers["img_in"])
    }

    func testTextFusionAcceptsTrainerSyntheticHiddenStateLayout() throws {
        let config = try makeTinyConfig(numLayers: 1)
        let textFusion = Krea2TextFusionTransformer(configuration: config)
        let textLength = 5
        let hiddenStates = MLXArray.zeros([
            1,
            textLength,
            config.numTextLayers,
            config.textHiddenDim,
        ])
        let validMask = MLXArray.ones([1, textLength], dtype: .int32)
        let attentionMask = Krea2SampleBuilder.attentionMask(validMask: validMask, dtype: .float32)

        let output = textFusion(hiddenStates, mask: attentionMask)
        MLX.eval(output)

        XCTAssertEqual(output.shape, [1, textLength, config.textHiddenDim])
    }

    private func makeTinyConfig(
        numLayers: Int,
        numLayerwiseTextBlocks: Int = 1,
        numRefinerTextBlocks: Int = 1
    ) throws -> Krea2TransformerConfiguration {
        let json = """
        {
          "attention_head_dim": 8,
          "axes_dims_rope": [4, 4, 8],
          "in_channels": 16,
          "intermediate_size": 32,
          "norm_eps": 0.00001,
          "num_attention_heads": 2,
          "num_key_value_heads": 2,
          "num_layers": \(numLayers),
          "num_layerwise_text_blocks": \(numLayerwiseTextBlocks),
          "num_refiner_text_blocks": \(numRefinerTextBlocks),
          "num_text_layers": 4,
          "rope_theta": 10000,
          "text_hidden_dim": 16,
          "text_intermediate_size": 32,
          "text_num_attention_heads": 2,
          "text_num_key_value_heads": 2,
          "timestep_embed_dim": 16
        }
        """
        return try JSONDecoder().decode(Krea2TransformerConfiguration.self, from: Data(json.utf8))
    }
}
