import MLX
import XCTest
@testable import MereRunCore

final class Ideogram4RuntimeTests: XCTestCase {
    func testQwenEncoderWeightMapperStripsUpstreamModelPrefix() {
        let value = MLXArray([Float(1)], [1])

        let mapped = QwenEncoder.mapHFSafetensorWeight(
            key: "model.embed_tokens.weight",
            value: value
        )

        XCTAssertEqual(mapped.map(\.0), ["embed_tokens.weight"])
    }

    func testTransformerConfigurationDecodesDiffusersConfig() throws {
        let json = """
        {
          "_class_name": "Ideogram4Transformer2DModel",
          "adaln_dim": 512,
          "attention_head_dim": 256,
          "in_channels": 128,
          "intermediate_size": 12288,
          "llm_features_dim": 53248,
          "mrope_section": [24, 20, 20],
          "norm_eps": 0.00001,
          "num_attention_heads": 18,
          "num_layers": 34,
          "rope_theta": 5000000
        }
        """

        let config = try JSONDecoder().decode(Ideogram4TransformerConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.embeddingDim, 4_608)
        XCTAssertEqual(config.llmFeaturesDim, 53_248)
        XCTAssertEqual(config.mropeSection, [24, 20, 20])
    }

    func testTextEncoderConfigurationDecodesRopeParametersConfig() throws {
        let json = """
        {
          "head_dim": 128,
          "hidden_size": 4096,
          "intermediate_size": 12288,
          "max_position_embeddings": 262144,
          "num_attention_heads": 32,
          "num_hidden_layers": 36,
          "num_key_value_heads": 8,
          "rms_norm_eps": 0.000001,
          "vocab_size": 151936,
          "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [24, 20, 20],
            "rope_theta": 5000000
          }
        }
        """

        let config = try JSONDecoder().decode(Ideogram4TextEncoderConfiguration.self, from: Data(json.utf8))
        let qwenConfig = config.qwenConfiguration

        XCTAssertEqual(config.ropeTheta, 5_000_000)
        XCTAssertEqual(qwenConfig.mropeSection, [24, 20, 20])
        XCTAssertTrue(qwenConfig.mropeInterleaved)
    }

    func testVAEWeightMapperConvertsOfficialKeysToAutoencoderKeys() {
        XCTAssertEqual(
            Ideogram4VAEWeights.mapKey("decoder.mid.block_1.nin_shortcut.scale"),
            "decoder.mid_block.resnets.0.conv_shortcut.scale"
        )
        XCTAssertEqual(
            Ideogram4VAEWeights.mapKey("decoder.mid.attn_1.proj_out.weight"),
            "decoder.mid_block.attentions.0.to_out.0.weight"
        )
        XCTAssertEqual(
            Ideogram4VAEWeights.mapKey("decoder.up.0.block.2.conv1.scale"),
            "decoder.up_blocks.3.resnets.2.conv1.scale"
        )
        XCTAssertEqual(
            Ideogram4VAEWeights.mapKey("encoder.down.2.downsample.conv.zero_point"),
            "encoder.down_blocks.2.downsamplers.0.conv.zero_point"
        )
        XCTAssertEqual(
            Ideogram4VAEWeights.mapKey("decoder.post_quant_conv.weight"),
            "post_quant_conv.weight"
        )
    }

    func testSchedulerUsesIdeogram12StepPreset() {
        let scheduler = Ideogram4Scheduler.preset(steps: 12, width: 256, height: 256, guidanceScale: 5.0)

        XCTAssertEqual(scheduler.guidanceSchedule.count, 12)
        XCTAssertEqual(scheduler.guidanceSchedule.first, 3.0)
        XCTAssertEqual(scheduler.guidanceSchedule.last, 7.0)
        XCTAssertLessThan(scheduler.value(at: scheduler.interval(12)), scheduler.value(at: scheduler.interval(0)))
    }

    func testSampleBuilderPacksTextAndImageTokens() throws {
        let features = MLXArray(Array(0..<12).map(Float.init), [1, 2, 6])
        let latents = MLXArray(Array(0..<32).map(Float.init), [1, 4, 8])

        let sample = try Ideogram4SampleBuilder.pack(
            llmFeatures: features,
            imageLatents: latents,
            imageWidth: 32,
            imageHeight: 32,
            inChannels: 8
        )

        XCTAssertEqual(sample.llmFeatures.shape, [1, 6, 6])
        XCTAssertEqual(sample.x.shape, [1, 6, 8])
        XCTAssertEqual(sample.positionIds.shape, [3, 1, 6])
        XCTAssertEqual(sample.segmentIds.asArray(Int32.self), [1, 1, 1, 1, 1, 1])
        XCTAssertEqual(sample.indicator.asArray(Int32.self), [3, 3, 2, 2, 2, 2])
        XCTAssertEqual(sample.positionIds[0, 0, 1].item(Int32.self), 1)
        XCTAssertEqual(sample.positionIds[0, 0, 2].item(Int32.self), 65_536)
        XCTAssertEqual(sample.positionIds[1, 0, 5].item(Int32.self), 65_537)
        XCTAssertEqual(sample.positionIds[2, 0, 5].item(Int32.self), 65_537)

        let output = MLXArray(Array(0..<48).map(Float.init), [1, 6, 8])
        XCTAssertEqual(Ideogram4SampleBuilder.imageTokens(from: output, sample: sample).shape, [1, 4, 8])
    }

    func testTextFeatureExtractorConcatenatesRawActivationLayers() {
        let encoder = QwenEncoder(configuration: QwenTextEncoderConfiguration(
            vocabSize: 32,
            hiddenSize: 6,
            numHiddenLayers: 3,
            numAttentionHeads: 3,
            numKeyValueHeads: 3,
            intermediateSize: 12,
            ropeTheta: 10_000,
            maxPositionEmbeddings: 32,
            rmsNormEps: 1e-6,
            headDim: 2
        ))
        let inputIds = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3)
        let attentionMask = MLXArray([Int32(1), Int32(1), Int32(1)]).reshaped(1, 3)

        let features = Ideogram4TextFeatures.encode(
            inputIds: inputIds,
            attentionMask: attentionMask,
            using: encoder,
            activationLayers: [0, 2]
        )
        eval(features)

        XCTAssertEqual(features.shape, [1, 3, 12])
        XCTAssertTrue(MLX.max(MLX.abs(features.asType(.float32))).item(Float.self).isFinite)
    }

    func testTextFeatureConcatenationInterleavesActivationLayersPerHiddenChannel() {
        let layer0 = MLXArray([1, 2, 3, 4], [1, 2, 2])
        let layer1 = MLXArray([10, 20, 30, 40], [1, 2, 2])
        let layer2 = MLXArray([100, 200, 300, 400], [1, 2, 2])

        let features = Ideogram4TextFeatures.concatenate([layer0, layer1, layer2])
        eval(features)

        XCTAssertEqual(features.shape, [1, 2, 6])
        XCTAssertEqual(
            features.asArray(Int32.self),
            [
                1, 10, 100, 2, 20, 200,
                3, 30, 300, 4, 40, 400,
            ]
        )
    }

    func testTinyTransformerForwardProducesImageVelocityShape() throws {
        let config = Ideogram4TransformerConfiguration(
            adalnDim: 4,
            attentionHeadDim: 6,
            inChannels: 8,
            intermediateSize: 16,
            llmFeaturesDim: 10,
            mropeSection: [1, 1, 1],
            normEps: 1e-5,
            numAttentionHeads: 2,
            numLayers: 1,
            ropeTheta: 10_000
        )
        let transformer = Ideogram4Transformer(configuration: config)
        let features = MLXArray(Array(0..<20).map { Float($0) / 20.0 }, [1, 2, 10])
        let latents = MLXArray(Array(0..<16).map { Float($0) / 16.0 }, [1, 2, 8])
        let sample = try Ideogram4SampleBuilder.pack(
            llmFeatures: features,
            imageLatents: latents,
            imageWidth: 32,
            imageHeight: 16,
            inChannels: 8
        )

        let output = transformer(sample: sample, timestep: MLXArray([Float(0.5)]))
        eval(output)

        XCTAssertEqual(output.shape, [1, 4, 8])
        XCTAssertEqual(Ideogram4SampleBuilder.imageTokens(from: output, sample: sample).shape, [1, 2, 8])
        XCTAssertTrue(MLX.max(MLX.abs(output)).item(Float.self).isFinite)
    }
}
