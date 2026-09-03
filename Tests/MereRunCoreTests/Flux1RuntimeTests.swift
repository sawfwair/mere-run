import MLX
import XCTest
@testable import MereRunCore

final class Flux1RuntimeTests: MereRunCoreTestCase {
    func testOfficialComponentConfigurationsDecodeExpectedArchitecture() throws {
        let transformer = try decode(
            Flux1TransformerConfiguration.self,
            """
            {
              "attention_head_dim": 128,
              "guidance_embeds": true,
              "in_channels": 64,
              "joint_attention_dim": 4096,
              "num_attention_heads": 24,
              "num_layers": 19,
              "num_single_layers": 38,
              "patch_size": 1,
              "pooled_projection_dim": 768
            }
            """
        )
        let clip = try decode(
            Flux1CLIPConfiguration.self,
            """
            {
              "hidden_size": 768,
              "intermediate_size": 3072,
              "layer_norm_eps": 0.00001,
              "max_position_embeddings": 77,
              "num_attention_heads": 12,
              "num_hidden_layers": 12,
              "projection_dim": 768,
              "vocab_size": 49408
            }
            """
        )
        let t5 = try decode(
            Flux1T5Configuration.self,
            """
            {
              "d_ff": 10240,
              "d_kv": 64,
              "d_model": 4096,
              "layer_norm_epsilon": 0.000001,
              "num_heads": 64,
              "num_layers": 24,
              "relative_attention_num_buckets": 32,
              "vocab_size": 32128
            }
            """
        )
        let vae = try decode(
            Flux1VAEConfiguration.self,
            """
            {
              "block_out_channels": [128, 256, 512, 512],
              "in_channels": 3,
              "latent_channels": 16,
              "layers_per_block": 2,
              "norm_num_groups": 32,
              "out_channels": 3,
              "scaling_factor": 0.3611,
              "shift_factor": 0.1159
            }
            """
        )

        XCTAssertEqual(transformer.hiddenSize, 3_072)
        XCTAssertEqual(transformer.outChannels, 64)
        XCTAssertEqual(transformer.axesDimsRope, [16, 56, 56])
        XCTAssertEqual(transformer.axesDimsRope.reduce(0, +), transformer.attentionHeadDim)
        XCTAssertTrue(transformer.guidanceEmbeds)
        XCTAssertEqual(clip.maxPositionEmbeddings, 77)
        XCTAssertEqual(t5.wanConfiguration.attentionSize, 4_096)
        XCTAssertEqual(t5.wanConfiguration.layerCount, 24)
        XCTAssertEqual(vae.coreConfiguration.latentChannels, 16)
        XCTAssertEqual(vae.coreConfiguration.scalingFactor, 0.3611, accuracy: 0.00001)
        XCTAssertEqual(vae.coreConfiguration.shiftFactor, 0.1159, accuracy: 0.00001)
    }

    func testFlowMatchScheduleUsesDynamicImageShiftAndTerminalZero() throws {
        let configuration = try decode(
            Flux1SchedulerConfiguration.self,
            """
            {
              "base_image_seq_len": 256,
              "base_shift": 0.5,
              "max_image_seq_len": 4096,
              "max_shift": 1.15,
              "num_train_timesteps": 1000,
              "shift": 1.0,
              "use_dynamic_shifting": true
            }
            """
        )
        let scheduler = Flux1Scheduler(
            steps: 4,
            imageSequenceLength: 1_024,
            configuration: configuration
        )

        XCTAssertEqual(scheduler.sigmas.count, 5)
        XCTAssertEqual(scheduler.sigmas.first, 1)
        XCTAssertEqual(scheduler.sigmas.last, 0)
        XCTAssertTrue(zip(scheduler.sigmas, scheduler.sigmas.dropFirst()).allSatisfy(>))
    }

    func testLatentPackingRoundTripsSixteenChannelGrid() {
        let latents = MLXArray(0..<256).asType(.float32).reshaped(1, 16, 4, 4)
        let packed = Flux1SampleBuilder.pack(latents)
        let unpacked = Flux1SampleBuilder.unpack(packed, height: 4, width: 4)

        XCTAssertEqual(packed.shape, [1, 4, 64])
        XCTAssertEqual(unpacked.shape, latents.shape)
        XCTAssertEqual(unpacked.asArray(Float.self), latents.asArray(Float.self))
    }

    func testT5WeightMapperReplicatesSharedRelativeBias() {
        let relativeBias = MLXArray.zeros([32, 64], dtype: .float32)
        let mapped = Flux1TextEncoderLoader.mapT5Weight(
            key: "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            value: relativeBias,
            layerCount: 24
        )

        XCTAssertEqual(mapped.count, 24)
        XCTAssertEqual(mapped.first?.0, "blocks.0.pos_embedding.embedding.weight")
        XCTAssertEqual(mapped.last?.0, "blocks.23.pos_embedding.embedding.weight")
    }

    func testTransformerWeightMapperPreservesOfficialProjectionNames() {
        let value = MLXArray.zeros([4, 4], dtype: .float32)
        XCTAssertEqual(
            Flux1TransformerWeightMapper.map(
                key: "single_transformer_blocks.0.proj_out.weight",
                value: value
            ).first?.0,
            "single_transformer_blocks.0.proj_out.weight"
        )
        XCTAssertEqual(
            Flux1TransformerWeightMapper.map(
                key: "transformer_blocks.0.ff.net.0.proj.weight",
                value: value
            ).first?.0,
            "transformer_blocks.0.ff.input.proj.weight"
        )
    }

    func testResourcesRequireOfficialCLIPBPEFilesInsteadOfTokenizerJSON() throws {
        let root = try TestFileSystem.makeTempDir(prefix: "flux1-resources")
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = Flux1Resources(rootURL: root)
        let required = [
            resources.transformerConfigURL,
            resources.transformerWeightsIndexURL,
            resources.clipConfigURL,
            resources.clipWeightsURL,
            resources.clipTokenizerURL.appendingPathComponent("tokenizer_config.json"),
            resources.clipTokenizerVocabURL,
            resources.clipTokenizerMergesURL,
            resources.t5ConfigURL,
            resources.t5WeightsIndexURL,
            resources.t5TokenizerURL.appendingPathComponent("tokenizer.json"),
            resources.vaeConfigURL,
            resources.vaeWeightsURL,
            resources.schedulerConfigURL,
        ]
        for url in required {
            try TestFileSystem.writeFile(url)
        }

        XCTAssertTrue(resources.validate().isEmpty)
        XCTAssertFalse(required.contains(resources.clipTokenizerURL.appendingPathComponent("tokenizer.json")))
    }

    func testCLIPBPEFallbackAddsReferenceSpecialTokens() throws {
        let root = try TestFileSystem.makeTempDir(prefix: "flux1-clip-tokenizer")
        defer { try? FileManager.default.removeItem(at: root) }
        try TestFileSystem.writeFile(
            root.appendingPathComponent("tokenizer_config.json"),
            contents: Data(#"{"tokenizer_class":"CLIPTokenizer"}"#.utf8)
        )
        try TestFileSystem.writeFile(
            root.appendingPathComponent("vocab.json"),
            contents: Data(#"{"<|startoftext|>":49406,"<|endoftext|>":49407}"#.utf8)
        )
        try TestFileSystem.writeFile(
            root.appendingPathComponent("merges.txt"),
            contents: Data("#version: 0.2\n".utf8)
        )

        let tokenizer = try Flux1Tokenizer.load(from: root, maximumLength: 4, fallbackPadTokenID: 49_407)
        let encoded = tokenizer.encode("")

        XCTAssertEqual(encoded.ids, [49_406, 49_407, 49_407, 49_407])
        XCTAssertEqual(encoded.mask, [1, 1, 0, 0])
        XCTAssertEqual(encoded.pooledIndex, 1)
    }

    private func decode<Value: Decodable>(_ type: Value.Type, _ json: String) throws -> Value {
        try JSONDecoder().decode(type, from: Data(json.utf8))
    }
}
