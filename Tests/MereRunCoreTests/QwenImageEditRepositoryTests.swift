import Foundation
import MediaIO
import MLX
import XCTest
@testable import MereRunCore

final class QwenImageEditRepositoryTests: MereRunCoreTestCase {
    func testQwenImageEditDeduplicatesInputAndReferenceURLs() {
        let input = URL(fileURLWithPath: "/tmp/qwen-edit/source.png")
        let second = URL(fileURLWithPath: "/tmp/qwen-edit/second.png")

        let resolved = QwenImageEditGenerator.deduplicatedReferenceURLs(
            inputImage: input,
            referenceImages: [input, second, input]
        )

        XCTAssertEqual(resolved, [input, second])
    }

    func testInstalledQwenImageEditVAERoundTripWhenRequested() throws {
        let env = ProcessInfo.processInfo.environment
        guard let root = env["MERERUN_TEST_QWEN_EDIT_ROOT"], !root.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_QWEN_EDIT_ROOT to test an installed Qwen Image Edit VAE.")
        }
        guard let input = env["MERERUN_TEST_QWEN_EDIT_IMAGE"], !input.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_QWEN_EDIT_IMAGE to test an installed Qwen Image Edit VAE.")
        }

        let output = env["MERERUN_TEST_QWEN_EDIT_VAE_OUTPUT"] ?? "/tmp/qwen-image-edit-vae-round-trip.png"
        let resources = QwenImageEditResources(rootURL: URL(fileURLWithPath: root))
        let configs = try QwenImageEditModelConfigs.load(from: resources)
        let vae = QwenImageEditVAE(config: configs.vae)
        try QwenImageEditGenerator.validateVAECheckpointCoverage(
            rawKeys: Set(try SafetensorsStreamingLoader.metadata(url: resources.vaeWeightsURL).keys),
            model: vae
        )
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.vaeWeightsURL,
            to: vae.underlyingVAE,
            dtype: .bfloat16,
            verify: [.shapeMismatch],
            mapper: QwenImageEditVAE.weightMapper
        )
        MLX.eval(vae)

        let decodedInput = try MediaImageIO.decode(URL(fileURLWithPath: input))
        let source = try QwenImageIO.resizedCenterCropPixelArray(
            from: decodedInput,
            width: 256,
            height: 256,
            dtype: .float32
        )
        let normalized = QwenImageIO.normalizeForEncoder(source)
        let latents = vae.encodeConditioning(normalized)
        let reconstruction = MLX.clip(
            QwenImageIO.denormalizeFromDecoder(vae.decodeGenerated(latents)).asType(.float32),
            min: 0,
            max: 1
        )
        MLX.eval(source, latents, reconstruction)
        try QwenImageIO.saveImage(
            array: reconstruction,
            to: URL(fileURLWithPath: output)
        )

        let meanAbsoluteError = MLX.mean(MLX.abs(reconstruction - source)).item(Float.self)
        FileHandle.standardError.write(
            "qwen_image_edit_vae_round_trip mae=\(meanAbsoluteError) "
                .appending("latent_shape=\(latents.shape) output=\(output)\n")
                .data(using: .utf8)!
        )
        XCTAssertLessThan(meanAbsoluteError, 0.25, "The installed VAE should reconstruct the source image.")
    }

    func testRepositoryIdentitiesKeepLegacyAnd2511Separate() {
        XCTAssertEqual(
            QwenImageEditRepository.canonicalModelId(for: "Qwen/Qwen-Image-Edit"),
            QwenImageEditRepository.modelId
        )
        XCTAssertEqual(
            QwenImageEditRepository.canonicalModelId(for: "Qwen/Qwen-Image-Edit-2511"),
            QwenImageEditRepository.model2511Id
        )
        XCTAssertEqual(
            QwenImageEditRepository.canonicalModelId(for: QwenImageEditRepository.lightning2511Id),
            QwenImageEditRepository.lightning2511Id
        )
        XCTAssertEqual(QwenImageEditRepository.revision, "main")
        XCTAssertEqual(QwenImageEditRepository.revision2511.count, 40)
    }

    func testLightningArtifactAndKeyMappingArePinned() {
        XCTAssertEqual(QwenImageEditRepository.lightningByteCount, 849_608_296)
        XCTAssertEqual(
            QwenImageEditRepository.lightningSHA256,
            "22226e8d05d354bb356627d428809f5afd7819399b077238a2b70a82883a904f"
        )
        XCTAssertEqual(QwenImageEditLightningAdapter.expectedPairCount, 720)
        XCTAssertEqual(QwenImageEditLightningAdapter.rank, 64)
        XCTAssertEqual(QwenImageEditLightningAdapter.alpha, 8)
        XCTAssertEqual(
            QwenImageEditLightningAdapter.mappedTargetPath(
                "transformer_blocks.4.attn.to_out.0"
            ),
            "transformer_blocks.4.attn.to_out"
        )
        XCTAssertEqual(
            QwenImageEditLightningAdapter.mappedTargetPath(
                "transformer_blocks.4.txt_mlp.net.0.proj"
            ),
            "transformer_blocks.4.ff_context.linear1"
        )
    }

    func testCFGExecutionModeParsing() {
        XCTAssertEqual(QwenImageEditCFGExecutionMode.parse(nil), .automatic)
        XCTAssertEqual(QwenImageEditCFGExecutionMode.parse("auto"), .automatic)
        XCTAssertEqual(QwenImageEditCFGExecutionMode.parse("on"), .batched)
        XCTAssertEqual(QwenImageEditCFGExecutionMode.parse("batched"), .batched)
        XCTAssertEqual(QwenImageEditCFGExecutionMode.parse("off"), .serial)
        XCTAssertEqual(QwenImageEditCFGExecutionMode.parse("serial"), .serial)
    }

    func testCFGAutoBatchingRequiresUnifiedMemoryHeadroom() {
        let gibibyte = UInt64(1_073_741_824)
        XCTAssertTrue(QwenImageEditCFGExecution.shouldBatch(
            mode: .automatic,
            width: 1_024,
            height: 1_024,
            physicalMemoryBytes: 64 * gibibyte,
            activeMemoryBytes: 20 * Int(gibibyte),
            cacheMemoryBytes: 2 * Int(gibibyte),
            isUnifiedMemory: true
        ))
        XCTAssertFalse(QwenImageEditCFGExecution.shouldBatch(
            mode: .automatic,
            width: 1_024,
            height: 1_024,
            physicalMemoryBytes: 24 * gibibyte,
            activeMemoryBytes: 20 * Int(gibibyte),
            cacheMemoryBytes: 1 * Int(gibibyte),
            isUnifiedMemory: true
        ))
        XCTAssertFalse(QwenImageEditCFGExecution.shouldBatch(
            mode: .automatic,
            width: 1_024,
            height: 1_024,
            physicalMemoryBytes: 64 * gibibyte,
            activeMemoryBytes: 0,
            cacheMemoryBytes: 0,
            isUnifiedMemory: false
        ))
        XCTAssertTrue(QwenImageEditCFGExecution.shouldBatch(
            mode: .batched,
            width: 4_096,
            height: 4_096,
            physicalMemoryBytes: gibibyte,
            activeMemoryBytes: Int(gibibyte),
            cacheMemoryBytes: 0,
            isUnifiedMemory: false
        ))
    }

    func testBatchedCFGCombinationPreservesSerialFormula() {
        let predictions = MLXArray([Float(1), 2, 4, 8], [2, 1, 1, 2])
        let combined = QwenImageEditCFGExecution.combinePredictions(
            predictions,
            guidanceScale: 3
        )
        MLX.eval(combined)

        XCTAssertEqual(combined.shape, [1, 1, 1, 2])
        XCTAssertEqual(combined.asArray(Float.self), [10, 20])
    }

    func testQwenCFGRescalesToPositivePredictionNorm() {
        let predictions = MLXArray([Float(1), 2, 4, 8], [2, 1, 2])
        let combined = QwenImageEditCFGExecution.combineQwenImagePredictions(
            predictions,
            guidanceScale: 3
        )
        MLX.eval(combined)

        let values = combined.asArray(Float.self)
        let actualNorm = sqrt(values.reduce(Float(0)) { $0 + $1 * $1 })
        let positiveNorm = sqrt(Float(4 * 4 + 8 * 8))
        XCTAssertEqual(actualNorm, positiveNorm, accuracy: 0.000_1)
    }

    func testQwen2511ConfigDecodesZeroConditionTimestep() throws {
        let json = #"""
        {
          "num_attention_heads": 24,
          "attention_head_dim": 128,
          "num_layers": 60,
          "joint_attention_dim": 3584,
          "in_channels": 64,
          "out_channels": 16,
          "patch_size": 2,
          "axes_dims_rope": [16, 56, 56],
          "guidance_embeds": false,
          "zero_cond_t": true
        }
        """#
        let config = try JSONDecoder().decode(QwenImageEditTransformerConfig.self, from: Data(json.utf8))
        XCTAssertTrue(config.zeroCondT)
    }

    func testQwenTimestepEmbeddingUsesOfficialThousandScale() {
        let embedding = TimestepEmbedder.sinusoidalEmbedding(
            MLXArray([Float(0.5)]),
            frequencyDim: 4
        )
        MLX.eval(embedding)

        let values = embedding.asArray(Float.self)
        XCTAssertEqual(values[0], cos(500), accuracy: 0.000_01)
        XCTAssertEqual(values[1], cos(5), accuracy: 0.000_01)
        XCTAssertEqual(values[2], sin(500), accuracy: 0.000_01)
        XCTAssertEqual(values[3], sin(5), accuracy: 0.000_01)
    }

    func testTransformerCheckpointCoverageRejectsMissingAndUnexpectedKeys() throws {
        let json = #"""
        {
          "num_attention_heads": 1,
          "attention_head_dim": 8,
          "num_layers": 1,
          "joint_attention_dim": 16,
          "in_channels": 4,
          "out_channels": 4,
          "patch_size": 2,
          "axes_dims_rope": [2, 2, 4],
          "guidance_embeds": false,
          "zero_cond_t": true
        }
        """#
        let config = try JSONDecoder().decode(QwenImageEditTransformerConfig.self, from: Data(json.utf8))
        let model = MMDiT(config: config)
        var rawKeys = Set(model.parameters().flattened().map { key, _ in
            Self.inverseTransformerCheckpointKey(key)
        })

        XCTAssertNoThrow(try QwenImageEditGenerator.validateTransformerCheckpointCoverage(
            rawKeys: rawKeys,
            model: model
        ))

        rawKeys.remove(try XCTUnwrap(rawKeys.first))
        rawKeys.insert("unexpected.weight")
        XCTAssertThrowsError(try QwenImageEditGenerator.validateTransformerCheckpointCoverage(
            rawKeys: rawKeys,
            model: model
        )) { error in
            guard case QwenImageEditGenerator.GeneratorError.checkpointCoverage(
                _, let missing, let unexpected
            ) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(missing.count, 1)
            XCTAssertEqual(unexpected, ["unexpected.weight"])
        }
    }

    func testQwen25VLEncoderUsesGatedVisionMLPAndExactCheckpointCoverage() throws {
        let json = #"""
        {
          "hidden_size": 16,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "intermediate_size": 32,
          "vocab_size": 128,
          "model_type": "qwen2_5_vl",
          "attention_bias": true,
          "rope_scaling": {"mrope_section": [1, 1, 2]},
          "vision_config": {
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 12,
            "hidden_act": "silu",
            "num_heads": 1,
            "patch_size": 2,
            "spatial_patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 16,
            "window_size": 8,
            "fullatt_block_indexes": [0]
          }
        }
        """#
        let config = try JSONDecoder().decode(QwenImageEditTextEncoderConfig.self, from: Data(json.utf8))
        let encoder = Qwen25VLEncoder.fromConfig(textEncoderConfig: config)
        let modelKeys = Set(encoder.parameters().flattened().map(\.0))

        XCTAssertTrue(modelKeys.contains("visionTower.blocks.0.mlp.gate_proj.weight"))
        XCTAssertTrue(modelKeys.contains("visionTower.blocks.0.mlp.up_proj.weight"))
        XCTAssertTrue(modelKeys.contains("visionTower.blocks.0.mlp.down_proj.weight"))
        XCTAssertFalse(modelKeys.contains("visionTower.blocks.0.mlp.fc1.weight"))
        XCTAssertEqual(encoder.textEncoder.configuration.mropeSection, [1, 1, 2])
        XCTAssertTrue(encoder.textEncoder.configuration.attentionBias)
        XCTAssertFalse(encoder.textEncoder.configuration.useQKNorm)
        XCTAssertTrue(modelKeys.contains("textEncoder.encoder.layers.0.self_attn.q_proj.bias"))
        XCTAssertTrue(modelKeys.contains("textEncoder.encoder.layers.0.self_attn.k_proj.bias"))
        XCTAssertTrue(modelKeys.contains("textEncoder.encoder.layers.0.self_attn.v_proj.bias"))
        XCTAssertFalse(modelKeys.contains("textEncoder.encoder.layers.0.self_attn.q_norm.weight"))
        XCTAssertFalse(modelKeys.contains("textEncoder.encoder.layers.0.self_attn.k_norm.weight"))
        XCTAssertFalse(modelKeys.contains { $0.hasPrefix("textEncoder.visionTower.") })
        XCTAssertFalse(modelKeys.contains("visionTower.blocks.0.norm1.bias"))
        XCTAssertFalse(modelKeys.contains("visionTower.blocks.0.norm2.bias"))
        XCTAssertFalse(modelKeys.contains("visionTower.patch_merger.ln_q.bias"))

        var rawKeys = Set(modelKeys.map(Self.inverseTextEncoderCheckpointKey))
        rawKeys.insert("lm_head.weight")
        XCTAssertNoThrow(try QwenImageEditGenerator.validateEncoderCheckpointCoverage(
            rawKeys: rawKeys,
            model: encoder
        ))

        rawKeys.remove(try XCTUnwrap(rawKeys.first { $0 != "lm_head.weight" }))
        rawKeys.insert("unexpected.weight")
        XCTAssertThrowsError(try QwenImageEditGenerator.validateEncoderCheckpointCoverage(
            rawKeys: rawKeys,
            model: encoder
        ))
    }

    func testQwen25VLEncoderTransposesPyTorchPatchEmbeddingWeightForMLX() throws {
        let json = #"""
        {
          "hidden_size": 16,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "intermediate_size": 32,
          "vocab_size": 128
        }
        """#
        let config = try JSONDecoder().decode(QwenImageEditTextEncoderConfig.self, from: Data(json.utf8))
        let mapper = QwenImageEditGenerator.textEncoderWeightMapper(config: config)
        let weight = MLXArray.zeros([8, 3, 2, 4, 4])
        let mapped = try XCTUnwrap(mapper("visual.patch_embed.proj.weight", weight).first)

        XCTAssertEqual(mapped.0, "visionTower.patch_embed.proj.weight")
        XCTAssertEqual(mapped.1.shape, [8, 2, 4, 4, 3])
    }

    func testQwen25VLMultimodalPositionsTrackEachOrderedPictureGrid() {
        let imageTokenID = 99
        let inputIDs = MLXArray([Int32(10), 99, 99, 99, 99, 11, 12, 99, 13], [1, 9])
        let positions = QwenTextEncoder.multimodalPositionIDs(
            inputIds: inputIDs,
            imageTokenId: imageTokenID,
            placeholderGridTHW: [(1, 4, 4), (1, 2, 2)],
            spatialMergeSize: 2
        )
        MLX.eval(positions)

        XCTAssertEqual(positions.shape, [3, 1, 9])
        XCTAssertEqual(positions[0, 0, 0...].asArray(Int32.self), [0, 1, 1, 1, 1, 3, 4, 5, 6])
        XCTAssertEqual(positions[1, 0, 0...].asArray(Int32.self), [0, 1, 1, 2, 2, 3, 4, 5, 6])
        XCTAssertEqual(positions[2, 0, 0...].asArray(Int32.self), [0, 1, 2, 1, 2, 3, 4, 5, 6])
    }

    func testQwenVAELatentNormalizationRoundTripsPerChannelStatistics() throws {
        let json = #"""
        {
          "base_dim": 96,
          "dim_mult": [1, 2, 4, 4],
          "z_dim": 2,
          "num_res_blocks": 2,
          "temperal_downsample": [false, true, true],
          "latents_mean": [1.0, -2.0],
          "latents_std": [2.0, 4.0]
        }
        """#
        let config = try JSONDecoder().decode(QwenImageEditVAEConfig.self, from: Data(json.utf8))
        let raw = MLXArray([Float(3), 5, 2, 6], [1, 2, 1, 2])
        let normalized = QwenImageEditVAE.normalizeLatents(raw, config: config)
        let restored = QwenImageEditVAE.denormalizeLatents(normalized, config: config)
        MLX.eval(normalized, restored)

        XCTAssertEqual(normalized.asArray(Float.self), [1, 2, 1, 2])
        XCTAssertEqual(restored.asArray(Float.self), raw.asArray(Float.self))
    }

    func testQwenVAEUsesExactCheckpointCoverage() throws {
        let json = #"""
        {
          "block_out_channels": [8, 16, 32, 32],
          "in_channels": 3,
          "out_channels": 3,
          "latent_channels": 2,
          "layers_per_block": 2,
          "norm_num_groups": 8,
          "scaling_factor": 1.0,
          "temporal_compression_ratio": 4,
          "mid_block_add_attention": true
        }
        """#
        let config = try JSONDecoder().decode(QwenImageEditVAEConfig.self, from: Data(json.utf8))
        let vae = QwenImageEditVAE(config: config)
        var rawKeys = Set(vae.underlyingVAE.parameters().flattened().map { key, _ in
            Self.inverseVAECheckpointKey(key)
        })

        XCTAssertNoThrow(try QwenImageEditGenerator.validateVAECheckpointCoverage(
            rawKeys: rawKeys,
            model: vae
        ))

        rawKeys.remove(try XCTUnwrap(rawKeys.first))
        rawKeys.insert("unexpected.weight")
        XCTAssertThrowsError(try QwenImageEditGenerator.validateVAECheckpointCoverage(
            rawKeys: rawKeys,
            model: vae
        )) { error in
            guard case QwenImageEditGenerator.GeneratorError.checkpointCoverage(
                _, let missing, let unexpected
            ) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(missing.count, 1)
            XCTAssertEqual(unexpected, ["unexpected.weight"])
        }
    }

    func testReferencePlanPreservesEachAspectRatioIndependently() {
        let landscape = QwenImageEditReferencePlan(
            source: URL(fileURLWithPath: "/tmp/landscape.png"),
            sourceWidth: 1_600,
            sourceHeight: 900
        )
        let portrait = QwenImageEditReferencePlan(
            source: URL(fileURLWithPath: "/tmp/portrait.png"),
            sourceWidth: 900,
            sourceHeight: 1_600
        )
        let plan = QwenImageEditConditioningPlan(
            outputWidth: 1_024,
            outputHeight: 768,
            references: [landscape, portrait]
        )

        XCTAssertEqual(landscape.semanticInputSize, QwenImageEditSize(width: 512, height: 288))
        XCTAssertEqual(landscape.semanticSize, QwenImageEditSize(width: 504, height: 280))
        XCTAssertEqual(landscape.vaeSize, QwenImageEditSize(width: 1_376, height: 768))
        XCTAssertEqual(portrait.vaeSize, QwenImageEditSize(width: 768, height: 1_376))
        XCTAssertEqual(plan.transformerImageShapes.map { [$0.height, $0.width] }, [
            [48, 64],
            [48, 86],
            [86, 48],
        ])
    }

    func testOfficialEditingPromptLabelsOrderedPictures() {
        let prompt = Qwen25VLTokenizer.editingPrompt(
            prompt: "Make Picture 1 match Picture 2.",
            numImageTokens: [2, 3]
        )

        XCTAssertTrue(prompt.hasPrefix("<|im_start|>system\nDescribe the key features"))
        XCTAssertTrue(prompt.contains(
            "Picture 1: <|vision_start|><|image_pad|><|image_pad|><|vision_end|>"
        ))
        XCTAssertTrue(prompt.contains(
            "Picture 2: <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>"
        ))
        XCTAssertTrue(prompt.hasSuffix("<|im_end|>\n<|im_start|>assistant\n"))
    }

    func testVisionPatchPackingMatchesQwenMergedCellAndTemporalOrder() {
        let pixels = MLXArray((0..<16).map(Float.init), [1, 1, 4, 4])
        let patches = Qwen25VLEncoder.preparePatchInputs(
            pixelValues: pixels,
            patchSize: 1,
            temporalPatchSize: 2,
            mergeSize: 2
        )
        MLX.eval(patches)

        XCTAssertEqual(patches.shape, [1, 16, 2])
        XCTAssertEqual(patches.asArray(Float.self), [
            0, 0, 1, 1, 4, 4, 5, 5,
            2, 2, 3, 3, 6, 6, 7, 7,
            8, 8, 9, 9, 12, 12, 13, 13,
            10, 10, 11, 11, 14, 14, 15, 15,
        ])
    }

    func testPackedLatentsRoundTripWithoutChangingTokenOrder() {
        let raw = MLXArray((0..<64).map(Float.init), [1, 4, 4, 4])
        let packed = QwenImageEditLatentCreator.packLatents(raw)
        let unpacked = QwenImageEditLatentCreator.unpackLatents(
            packed,
            height: 4,
            width: 4,
            channels: 4
        )
        MLX.eval(packed, unpacked)

        XCTAssertEqual(packed.shape, [1, 4, 16])
        XCTAssertEqual(unpacked.asArray(Float.self), raw.asArray(Float.self))
    }

    func testQwenRoPEConcatenatesEveryImageGridBeforeText() {
        let rope = MMDiTRoPE(headDim: 128, axesDims: [16, 56, 56])
        let frequencies = rope.frequencies(
            imageShapes: [
                (temporal: 1, height: 2, width: 2),
                (temporal: 1, height: 1, width: 3),
            ],
            textSequenceLength: 5
        )

        XCTAssertEqual(frequencies.image.shape, [7, 64, 2])
        XCTAssertEqual(frequencies.text.shape, [5, 64, 2])
    }

    func testBatchedCFGCombinationSupportsNonImageRanks() {
        let predictions = MLXArray([Float(1), 2, 4, 8], [2, 2])
        let combined = DiffusionCFGExecution.combinePredictions(
            predictions,
            guidanceScale: 3
        )
        MLX.eval(combined)

        XCTAssertEqual(combined.shape, [1, 2])
        XCTAssertEqual(combined.asArray(Float.self), [10, 20])
    }

    func testPositiveAnchoredCFGCombinationPreservesZImageFormula() {
        let predictions = MLXArray([Float(1), 2, 4, 8], [2, 2])
        let combined = DiffusionCFGExecution.combinePositiveAnchoredPredictions(
            predictions,
            guidanceScale: 3
        )
        MLX.eval(combined)

        XCTAssertEqual(combined.shape, [1, 2])
        XCTAssertEqual(combined.asArray(Float.self), [13, 26])
    }

    func testCFGBatchingPairsOnlyShapeCompatibleRows() {
        let unconditional = MLXArray([Float(1), 2, 3, 4], [1, 2, 2])
        let conditional = MLXArray([Float(5), 6, 7, 8], [1, 2, 2])
        let incompatible = MLXArray([Float(1), 2, 3], [1, 3, 1])

        XCTAssertTrue(DiffusionCFGExecution.canPair(unconditional, conditional))
        XCTAssertFalse(DiffusionCFGExecution.canPair(unconditional, incompatible))

        let paired = DiffusionCFGExecution.paired(unconditional, conditional)
        MLX.eval(paired)
        XCTAssertEqual(paired.shape, [2, 2, 2])
        XCTAssertEqual(paired.asArray(Float.self), [1, 2, 3, 4, 5, 6, 7, 8])
    }

    func testModelSpecificHeadroomReserveCanKeepCFGSerial() {
        let gibibyte = DiffusionCFGExecution.gibibyte
        XCTAssertFalse(DiffusionCFGExecution.shouldBatch(
            mode: .automatic,
            width: 1_024,
            height: 1_024,
            physicalMemoryBytes: 32 * gibibyte,
            activeMemoryBytes: 19 * Int(gibibyte),
            cacheMemoryBytes: 2 * Int(gibibyte),
            isUnifiedMemory: true,
            baseReserveBytes: 8 * gibibyte,
            activationBytesPerPixel: 4_096
        ))
    }

    func testResolveInstalledModelRootFindsDirectRoot() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot.appendingPathComponent(QwenImageEditRepository.modelId, isDirectory: true)
        try writeMinimalQwenImageEditModel(at: modelRoot)

        let resolved = QwenImageEditRepository.resolveInstalledModelRoot()
        XCTAssertEqual(resolved?.standardizedFileURL, modelRoot.standardizedFileURL)
    }

    func testResolveInstalled2511RootDoesNotAliasLegacyRoot() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)
        let legacyRoot = modelsRoot.appendingPathComponent(
            QwenImageEditRepository.modelId,
            isDirectory: true
        )
        try writeMinimalQwenImageEditModel(at: legacyRoot)

        XCTAssertNil(QwenImageEditRepository.resolveInstalledModelRoot(
            modelSpec: QwenImageEditRepository.model2511Id
        ))
    }

    func testResolveInstalledModelRootFindsSingleNestedRoot() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let parentRoot = modelsRoot.appendingPathComponent(QwenImageEditRepository.modelId, isDirectory: true)
        let nestedRoot = parentRoot.appendingPathComponent("Qwen-Image-Edit", isDirectory: true)
        try writeMinimalQwenImageEditModel(at: nestedRoot)

        let resolved = QwenImageEditRepository.resolveInstalledModelRoot()
        XCTAssertEqual(resolved?.standardizedFileURL, nestedRoot.standardizedFileURL)
    }

    func testResolveInstalledModelRootReturnsNilWhenMissing() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        XCTAssertNil(QwenImageEditRepository.resolveInstalledModelRoot())
    }

    private func writeMinimalQwenImageEditModel(at root: URL) throws {
        let files: [String] = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "tokenizer/tokenizer_config.json",
            "tokenizer/tokenizer.json",
        ]

        for relativePath in files {
            let url = root.appendingPathComponent(relativePath, isDirectory: false)
            try TestFileSystem.writeFile(url)
        }
    }

    private static func inverseTransformerCheckpointKey(_ modelKey: String) -> String {
        var key = modelKey
        if key.hasPrefix("x_embedder.") {
            return "img_in." + String(key.dropFirst("x_embedder.".count))
        }
        if key.hasPrefix("context_embedder.") {
            return "txt_in." + String(key.dropFirst("context_embedder.".count))
        }
        if key.hasPrefix("t_embedder.mlp.0.") {
            return "time_text_embed.timestep_embedder.linear_1."
                + String(key.dropFirst("t_embedder.mlp.0.".count))
        }
        if key.hasPrefix("t_embedder.mlp.1.") {
            return "time_text_embed.timestep_embedder.linear_2."
                + String(key.dropFirst("t_embedder.mlp.1.".count))
        }
        key = key.replacingOccurrences(of: ".attn.to_out.", with: ".attn.to_out.0.")
        key = key.replacingOccurrences(of: ".ff.linear1.", with: ".img_mlp.net.0.proj.")
        key = key.replacingOccurrences(of: ".ff.linear2.", with: ".img_mlp.net.2.")
        key = key.replacingOccurrences(of: ".ff_context.linear1.", with: ".txt_mlp.net.0.proj.")
        key = key.replacingOccurrences(of: ".ff_context.linear2.", with: ".txt_mlp.net.2.")
        key = key.replacingOccurrences(of: ".adaLN_modulation.linear.", with: ".img_mod.1.")
        key = key.replacingOccurrences(of: ".adaLN_modulation_context.linear.", with: ".txt_mod.1.")
        return key
    }

    private static func inverseTextEncoderCheckpointKey(_ modelKey: String) -> String {
        if modelKey.hasPrefix("textEncoder.encoder.") {
            return "model." + String(modelKey.dropFirst("textEncoder.encoder.".count))
        }
        if modelKey.hasPrefix("visionTower.") {
            var key = "visual." + String(modelKey.dropFirst("visionTower.".count))
            key = key.replacingOccurrences(of: ".patch_merger.", with: ".merger.")
            key = key.replacingOccurrences(of: ".merger.mlp_0.", with: ".merger.mlp.0.")
            key = key.replacingOccurrences(of: ".merger.mlp_2.", with: ".merger.mlp.2.")
            return key
        }
        return modelKey
    }

    private static func inverseVAECheckpointKey(_ modelKey: String) -> String {
        modelKey
            .replacingOccurrences(of: "postQuantConv", with: "post_quant_conv")
            .replacingOccurrences(of: "quantConv", with: "quant_conv")
            .replacingOccurrences(of: ".resample.0.", with: ".resample.1.")
    }
}
