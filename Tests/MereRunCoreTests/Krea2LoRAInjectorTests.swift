import MLX
import MLXNN
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

    func testQuantizedBaseSupportsLiteLoRABackward() throws {
        let transformer = Krea2Transformer(configuration: try makeTinyConfig(
            numLayers: 1,
            numLayerwiseTextBlocks: 2,
            numRefinerTextBlocks: 2
        ))
        Krea2LoRATrainer.quantizeTransformerBase(transformer, bits: 4)

        XCTAssertTrue(transformer.transformerBlocks[0].attn.toQ is QuantizedLinear)
        XCTAssertTrue(transformer.textFusion.layerwiseBlocks[1].attn.toQ is QuantizedLinear)
        XCTAssertTrue(transformer.textFusion.refinerBlocks[1].ff.down is QuantizedLinear)
        #if os(Linux)
        XCTAssertTrue(
            (transformer.textFusion.refinerBlocks[1].ff.down as? PortableQuantizedLinear)?
                .useUncachedDenseFallback == true
        )
        #endif
        let layers = try Krea2LoRAInjector.inject(
            into: transformer,
            rank: 2,
            targetSuffixes: Krea2LoRAInjector.liteTargetSuffixes,
            zeroInitUp: true
        )
        let query = try XCTUnwrap(layers["transformer_blocks.0.attn.to_q"] as? LoRAQuantizedLinear)
        query.freeze(recursive: true)
        try query.unfreeze(recursive: false, keys: ["loraDown", "loraUp"], strict: true)
        let input = MLXArray.ones([1, 3, transformer.configuration.hiddenSize])
        let lossAndGrad = valueAndGrad(model: query) { model, values in
            [MLX.sum(model(values[0]))]
        }

        let (_, gradients) = lossAndGrad(query, [input])
        let upGradient = try XCTUnwrap(gradients.flattened().first { key, _ in
            key == "loraUp"
        }?.1)
        MLX.eval(upGradient)

        XCTAssertEqual(upGradient.shape, query.loraUp.shape)
    }

    func testPortableQuantizedLoRAUsesDenseBaseFallback() throws {
        let base = PortableQuantizedLinear(
            Linear(32, 32, bias: false),
            groupSize: 32,
            bits: 4
        )
        let layer = LoRAQuantizedLinear(base: base, rank: 2, zeroInitUp: true)
        layer.freeze(recursive: true)
        try layer.unfreeze(recursive: false, keys: ["loraDown", "loraUp"], strict: true)
        let input = MLXArray.ones([1, 3, 32])
        let lossAndGrad = valueAndGrad(model: layer) { model, values in
            [MLX.sum(model(values[0]))]
        }

        let (loss, gradients) = lossAndGrad(layer, [input])
        let upGradient = try XCTUnwrap(gradients.flattened().first { key, _ in
            key == "loraUp"
        }?.1)
        MLX.eval(loss, upGradient)

        XCTAssertEqual(loss[0].shape, [])
        XCTAssertEqual(upGradient.shape, layer.loraUp.shape)
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

    func testCheckpointedBlockBackpropagatesWithoutDifferentiatingStaticConditioning() throws {
        let config = try makeTinyConfig(numLayers: 1)
        let block = Krea2SingleStreamBlock(configuration: config)
        let sequenceLength = 4
        let inputs = MLXArray.ones([1, sequenceLength, config.hiddenSize])
        let modulation = MLXArray.zeros([1, config.hiddenSize * 6])
        let rotaryCos = MLXArray.ones([1, sequenceLength, config.attentionHeadDim / 2])
        let rotarySin = MLXArray.zeros([1, sequenceLength, config.attentionHeadDim / 2])
        let attentionMask = MLXArray.zeros([1, 1, sequenceLength, sequenceLength])
        let checkpointedValueAndGrad = valueAndGrad(model: block) { model, values in
            [MLX.sum(model.checkpointed(
                values[0],
                modulation: values[1],
                rotary: (cos: values[2], sin: values[3]),
                mask: values[4]
            ))]
        }

        let (_, gradients) = checkpointedValueAndGrad(
            block,
            [inputs, modulation, rotaryCos, rotarySin, attentionMask]
        )
        let queryGradient = try XCTUnwrap(gradients.flattened().first { key, _ in
            key == "attn.to_q.weight"
        }?.1)
        MLX.eval(queryGradient)

        XCTAssertEqual(queryGradient.shape, block.attn.toQ.weight.shape)
    }

    func testCheckpointedTextFusionBackpropagatesWithAttentionMask() throws {
        let config = try makeTinyConfig(numLayers: 1)
        let textFusion = Krea2TextFusionTransformer(configuration: config)
        let sequenceLength = 5
        let hiddenStates = MLXArray.ones([
            1,
            sequenceLength,
            config.numTextLayers,
            config.textHiddenDim,
        ])
        let validMask = MLXArray.ones([1, sequenceLength], dtype: .int32)
        let attentionMask = Krea2SampleBuilder.attentionMask(validMask: validMask, dtype: .float32)
        let checkpointedValueAndGrad = valueAndGrad(model: textFusion) { model, values in
            [MLX.sum(model(
                values[0],
                mask: values[1],
                gradientCheckpointing: true
            ))]
        }

        let (_, gradients) = checkpointedValueAndGrad(textFusion, [hiddenStates, attentionMask])
        let queryGradient = try XCTUnwrap(gradients.flattened().first { key, _ in
            key == "refiner_blocks.0.attn.to_q.weight"
        }?.1)
        MLX.eval(queryGradient)

        XCTAssertEqual(queryGradient.shape, textFusion.refinerBlocks[0].attn.toQ.weight.shape)
    }

    private func makeTinyConfig(
        numLayers: Int,
        numLayerwiseTextBlocks: Int = 1,
        numRefinerTextBlocks: Int = 1
    ) throws -> Krea2TransformerConfiguration {
        let json = """
        {
          "attention_head_dim": 16,
          "axes_dims_rope": [4, 4, 8],
          "in_channels": 32,
          "intermediate_size": 64,
          "norm_eps": 0.00001,
          "num_attention_heads": 2,
          "num_key_value_heads": 2,
          "num_layers": \(numLayers),
          "num_layerwise_text_blocks": \(numLayerwiseTextBlocks),
          "num_refiner_text_blocks": \(numRefinerTextBlocks),
          "num_text_layers": 4,
          "rope_theta": 10000,
          "text_hidden_dim": 32,
          "text_intermediate_size": 64,
          "text_num_attention_heads": 2,
          "text_num_key_value_heads": 2,
          "timestep_embed_dim": 32
        }
        """
        return try JSONDecoder().decode(Krea2TransformerConfiguration.self, from: Data(json.utf8))
    }
}
