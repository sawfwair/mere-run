import Foundation
import XCTest
import MLX
import MLXRandom
@testable import MereRunCore

#if canImport(CoreGraphics)
import CoreGraphics
#endif

final class FalconPerceptionProcessorTests: MereRunCoreTestCase {
    func testConfigLoadDecodesSpecialTokenFields() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let configURL = temp.appendingPathComponent("config.json")
        let json = """
        {
          "model_type": "falcon_perception",
          "vocab_size": 32000,
          "text_config": {
            "model_type": "falcon_perception",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "head_dim": 128,
            "num_key_value_heads": 4,
            "vocab_size": 32000,
            "intermediate_size": 4096,
            "rms_norm_eps": 1e-05,
            "max_position_embeddings": 8192,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
          },
          "vision_config": {
            "model_type": "falcon_perception",
            "spatial_patch_size": 16,
            "temporal_patch_size": 1,
            "channel_size": 3
          },
          "img_id": 227,
          "eos_id": 11,
          "image_cls_token_id": 244,
          "image_reg_1_token_id": 245,
          "image_reg_2_token_id": 246,
          "image_reg_3_token_id": 247,
          "image_reg_4_token_id": 248,
          "img_end_id": 230,
          "coord_token_id": 240,
          "size_token_id": 241,
          "seg_token_id": 262,
          "coord_enc_dim": 512,
          "coord_dec_dim": 8192,
          "coord_out_dim": 2048,
          "size_enc_dim": 512,
          "size_dec_dim": 8192,
          "size_out_dim": 2048,
          "do_segmentation": true,
          "segm_out_dim": 256,
          "num_segm_layers": 3
        }
        """
        try TestFileSystem.writeFile(configURL, contents: Data(json.utf8))

        let config = try FalconPerceptionModelConfig.load(from: configURL)
        XCTAssertEqual(config.imgID, 227)
        XCTAssertEqual(config.coordTokenID, 240)
        XCTAssertEqual(config.sizeTokenID, 241)
        XCTAssertEqual(config.segTokenID, 262)
        XCTAssertTrue(config.doSegmentation)
    }

    func testConfigLoadDecodesFlatHuggingFaceFalconLayout() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let configURL = temp.appendingPathComponent("config.json")
        let json = """
        {
          "model_type": "falcon_perception",
          "dim": 1024,
          "n_layers": 28,
          "n_heads": 16,
          "head_dim": 128,
          "n_kv_heads": 8,
          "vocab_size": 65536,
          "ffn_dim": 3072,
          "norm_eps": 1e-05,
          "max_seq_len": 8192,
          "rope_theta": 10000,
          "channel_size": 3,
          "spatial_patch_size": 16,
          "temporal_patch_size": 1,
          "do_segmentation": true,
          "segm_out_dim": 256,
          "num_segm_layers": 3,
          "coord_enc_dim": 512,
          "coord_dec_dim": 8192,
          "coord_out_dim": 2048,
          "coord_token_id": 240,
          "size_enc_dim": 512,
          "size_dec_dim": 8192,
          "size_out_dim": 2048,
          "size_token_id": 241,
          "seg_token_id": 262,
          "eos_id": 11,
          "img_id": 227,
          "image_cls_token_id": 244,
          "image_reg_1_token_id": 245,
          "image_reg_2_token_id": 246,
          "image_reg_3_token_id": 247,
          "image_reg_4_token_id": 248,
          "img_end_id": 230
        }
        """
        try TestFileSystem.writeFile(configURL, contents: Data(json.utf8))

        let config = try FalconPerceptionModelConfig.load(from: configURL)
        XCTAssertEqual(config.textConfig.hiddenSize, 1024)
        XCTAssertEqual(config.textConfig.numHiddenLayers, 28)
        XCTAssertEqual(config.textConfig.numAttentionHeads, 16)
        XCTAssertEqual(config.visionConfig.spatialPatchSize, 16)
        XCTAssertEqual(config.visionConfig.channelSize, 3)
        XCTAssertEqual(config.coordTokenID, 240)
        XCTAssertEqual(config.segTokenID, 262)
    }

    func testComputePositionDataBuildsImageAwareAttentionMask() {
        let config = FalconPerceptionModelConfig()
        let tokens = [
            Int32(1),
            Int32(config.imageCLSTokenID),
            Int32(config.imageReg1TokenID),
            Int32(config.imageReg2TokenID),
            Int32(config.imageReg3TokenID),
            Int32(config.imageReg4TokenID),
            Int32(config.imgID),
            Int32(config.imgID),
            Int32(config.imgEndID),
            Int32(2),
        ]
        let inputIDs = MLXArray(tokens, [1, tokens.count])
        let imageGridHW = MLXArray([Int32(1), Int32(2)], [1, 2])

        let result = FalconPerceptionModel.computePositionData(
            inputIDs: inputIDs,
            config: config,
            imageGridHW: imageGridHW
        )

        XCTAssertEqual(result.positionIDs.shape, [tokens.count])
        XCTAssertEqual(result.posHW.shape, [1, tokens.count, 2])
        XCTAssertEqual(result.attentionMask.shape, [1, 1, tokens.count, tokens.count])
        XCTAssertEqual(result.ropeDelta, -7)

        let mask = result.attentionMask.asArray(Bool.self)
        let side = tokens.count

        func value(_ query: Int, _ key: Int) -> Bool {
            mask[(query * side) + key]
        }

        XCTAssertFalse(
            value(7, 8),
            "Image tokens should not be able to attend forward to <img_end>."
        )
        XCTAssertTrue(
            value(8, 7),
            "The closing token should still attend causally to prior image tokens."
        )
    }

    func testFalconAttentionBiasMaskConvertsBooleanMaskToAdditiveBias() {
        let mask = MLXArray([true, false, true, false], [1, 1, 2, 2])
        let additive = falconPerceptionAttentionBiasMask(mask, dtype: .float32).asArray(Float.self)

        XCTAssertEqual(additive[0], 0, accuracy: 0.0001)
        XCTAssertLessThan(additive[1], -1e8)
        XCTAssertEqual(additive[2], 0, accuracy: 0.0001)
        XCTAssertLessThan(additive[3], -1e8)
    }

    func testLanguageModelDecodePositionsFollowRopeDeltaAfterPrefill() {
        let config = FalconPerceptionModelConfig(
            textConfig: FalconPerceptionTextConfig(
                hiddenSize: 16,
                numHiddenLayers: 1,
                numAttentionHeads: 4,
                headDim: 4,
                numKeyValueHeads: 2,
                vocabSize: 512,
                intermediateSize: 32,
                maxPositionEmbeddings: 128
            ),
            visionConfig: FalconPerceptionVisionConfig(
                spatialPatchSize: 16,
                temporalPatchSize: 1,
                channelSize: 3
            ),
            vocabSize: 512
        )
        let languageModel = FalconPerceptionLanguageModel(config: config)
        let positionIDs = MLXArray([Int32(0), Int32(1), Int32(1), Int32(1), Int32(2)], [5])
        let posHW = MLX.zeros([1, 5, 2], dtype: .float32)
        let attentionMask = MLX.zeros([1, 1, 5, 5], dtype: .float32)

        languageModel.prepareGroundingPrefill(
            positionIDs: positionIDs,
            posHW: posHW,
            ropeDelta: -2,
            attentionMask: attentionMask
        )

        let prefill = languageModel.resolveAttentionInputs(
            sequenceLength: 5,
            cacheOffset: 0,
            mask: nil,
            positionIDs: nil,
            posHW: nil
        )
        XCTAssertEqual(prefill.positionIDs?.asArray(Int32.self), [0, 1, 1, 1, 2])
        XCTAssertEqual(prefill.posHW?.shape, [1, 5, 2])
        XCTAssertEqual(prefill.mask?.shape, [1, 1, 5, 5])

        languageModel.finishGroundingPrefill()

        let decode = languageModel.resolveAttentionInputs(
            sequenceLength: 1,
            cacheOffset: 5,
            mask: nil,
            positionIDs: nil,
            posHW: nil
        )
        XCTAssertEqual(decode.positionIDs?.asArray(Int32.self), [3])
        XCTAssertNil(decode.posHW)
        XCTAssertEqual(decode.mask?.shape, [1, 1, 1, 6])
        XCTAssertEqual(decode.mask?.asArray(Bool.self), [true, true, true, true, true, true])
    }

    func testCachedDecodeMatchesFullForwardForPresenceStep() {
        let config = FalconPerceptionModelConfig(
            textConfig: FalconPerceptionTextConfig(
                hiddenSize: 32,
                numHiddenLayers: 2,
                numAttentionHeads: 4,
                headDim: 8,
                numKeyValueHeads: 2,
                vocabSize: 512,
                intermediateSize: 64,
                maxPositionEmbeddings: 256
            ),
            visionConfig: FalconPerceptionVisionConfig(
                spatialPatchSize: 16,
                temporalPatchSize: 1,
                channelSize: 3
            ),
            vocabSize: 512,
            coordDecDim: 64,
            coordOutDim: 32,
            sizeDecDim: 64,
            sizeOutDim: 32
        )

        let promptTokens: [Int32] = [
            17,
            Int32(config.imageCLSTokenID),
            Int32(config.imageReg1TokenID),
            Int32(config.imageReg2TokenID),
            Int32(config.imageReg3TokenID),
            Int32(config.imageReg4TokenID),
            Int32(config.imgID),
            Int32(config.imgID),
            Int32(config.imgID),
            Int32(config.imgID),
            Int32(config.imgEndID),
            23,
        ]
        let presenceToken = Int32(268)
        let fullTokens = promptTokens + [presenceToken]
        let promptIDs = MLXArray(promptTokens, [1, promptTokens.count])
        let fullIDs = MLXArray(fullTokens, [1, fullTokens.count])
        let imageGridHW = MLXArray([Int32(2), Int32(2)], [1, 2])

        MLXRandom.seed(1234)
        let pixelValues = MLXRandom.normal([1, 32, 32, 3]).asType(.float32)

        MLXRandom.seed(99)
        let cachedModel = FalconPerceptionModel(config: config)
        MLXRandom.seed(99)
        let fullModel = FalconPerceptionModel(config: config)

        let promptPositionData = FalconPerceptionModel.computePositionData(
            inputIDs: promptIDs,
            config: config,
            imageGridHW: imageGridHW
        )
        cachedModel.resetGroundingState()
        cachedModel.prepareGroundingPrefill(positionData: promptPositionData)
        let caches = cachedModel.makeCaches()
        let promptEmbeds = cachedModel.makeInputEmbeddings(
            inputIDs: promptIDs,
            pixelValues: pixelValues,
            imageGridHW: imageGridHW
        )
        var cachedLogits = cachedModel.forward(
            inputIDs: promptIDs,
            inputsEmbeds: promptEmbeds,
            caches: caches
        )
        MLX.eval(cachedLogits)
        cachedModel.finishGroundingPrefill()

        let presenceIDs = MLXArray([presenceToken], [1, 1])
        let presenceEmbeds = cachedModel.embedTokens(presenceIDs)
        cachedLogits = cachedModel.forward(
            inputIDs: presenceIDs,
            inputsEmbeds: presenceEmbeds,
            caches: caches
        )
        MLX.eval(cachedLogits)

        let fullPositionData = FalconPerceptionModel.computePositionData(
            inputIDs: fullIDs,
            config: config,
            imageGridHW: imageGridHW
        )
        let fullEmbeds = fullModel.makeInputEmbeddings(
            inputIDs: fullIDs,
            pixelValues: pixelValues,
            imageGridHW: imageGridHW
        )
        let fullLogits = fullModel.forward(
            inputIDs: fullIDs,
            inputsEmbeds: fullEmbeds,
            mask: fullPositionData.attentionMask,
            positionIDs: fullPositionData.positionIDs,
            posHW: fullPositionData.posHW
        )
        MLX.eval(fullLogits)

        let cachedStep = cachedLogits[0, cachedLogits.dim(1) - 1].asType(.float32).asArray(Float.self)
        let fullStep = fullLogits[0, fullLogits.dim(1) - 1].asType(.float32).asArray(Float.self)

        XCTAssertEqual(cachedStep.count, fullStep.count)
        let maxDiff = zip(cachedStep, fullStep).map { abs($0 - $1) }.max() ?? .infinity
        XCTAssertLessThan(
            maxDiff,
            0.001,
            "Cached decode logits should match full-sequence logits for the same next-token step."
        )
    }

    func testBoundingBoxDerivesClampedCorners() {
        let box = FalconPerceptionGrounder.boundingBox(
            xy: FalconPerceptionCenter(x: 0.4, y: 0.75),
            hw: FalconPerceptionSize(h: 0.4, w: 0.5)
        )

        XCTAssertEqual(box.x1, 0.15, accuracy: 0.0001)
        XCTAssertEqual(box.y1, 0.55, accuracy: 0.0001)
        XCTAssertEqual(box.x2, 0.65, accuracy: 0.0001)
        XCTAssertEqual(box.y2, 0.95, accuracy: 0.0001)
    }

    func testBinaryMaskConvertsToUInt8Pixels() {
        let mask = MLXArray([Int32(0), Int32(1), Int32(1), Int32(0)], [2, 2])
        XCTAssertEqual(FalconPerceptionGrounder.binaryMask(from: mask), [0, 1, 1, 0])
    }

    func testAnyUpAdaptiveAveragePoolProducesRequestedShape() {
        let values = (0..<24).map(Float.init)
        let input = MLXArray(values, [1, 4, 6, 1])

        let pooled = FalconPerceptionAnyUp.adaptiveAveragePool2D(input, outputSize: (2, 3))

        XCTAssertEqual(pooled.shape, [1, 2, 3, 1])
        let flattened = pooled.reshaped(-1).asArray(Float.self)
        let expected: [Float] = [3.5, 5.5, 7.5, 15.5, 17.5, 19.5]
        XCTAssertEqual(flattened.count, expected.count)
        for (actual, wanted) in zip(flattened, expected) {
            XCTAssertEqual(actual, wanted, accuracy: 0.0001)
        }
    }

    func testAnyUpCheckpointMappingSplitsAttentionProjection() {
        let weight = MLXArray((0..<(384 * 128)).map(Float.init), [384, 128])
        let bias = MLXArray((0..<384).map(Float.init), [384])

        let mappedWeights = FalconPerceptionGrounder.mapCheckpointKey(
            key: "itok_upsampler.cross_decode.cross_attn.attention.in_proj_weight",
            value: weight
        )
        let mappedBiases = FalconPerceptionGrounder.mapCheckpointKey(
            key: "itok_upsampler.cross_decode.cross_attn.attention.in_proj_bias",
            value: bias
        )

        XCTAssertEqual(mappedWeights.map(\.0), [
            "itok_upsampler.cross_decode.cross_attn.q_proj.weight",
            "itok_upsampler.cross_decode.cross_attn.k_proj.weight",
        ])
        XCTAssertEqual(mappedWeights.map(\.1.shape), [[128, 128], [128, 128]])
        XCTAssertEqual(mappedBiases.map(\.0), [
            "itok_upsampler.cross_decode.cross_attn.q_proj.bias",
            "itok_upsampler.cross_decode.cross_attn.k_proj.bias",
        ])
        XCTAssertEqual(mappedBiases.map(\.1.shape), [[128], [128]])
    }

    func testAnyUpCheckpointMappingTransposesConvolutionWeights() {
        let encoderWeight = MLXArray((0..<36).map(Float.init), [4, 3, 3, 1])
        let basis = MLXArray((0..<25).map(Float.init), [1, 1, 5, 5])

        let mappedEncoder = FalconPerceptionGrounder.mapCheckpointKey(
            key: "itok_upsampler.image_encoder.0.weight",
            value: encoderWeight
        )
        let mappedBasis = FalconPerceptionGrounder.mapCheckpointKey(
            key: "itok_upsampler.key_features_encoder.0.basis",
            value: basis
        )

        XCTAssertEqual(mappedEncoder.first?.0, "itok_upsampler.image_encoder.conv.weight")
        XCTAssertEqual(mappedEncoder.first?.1.shape, [4, 3, 1, 3])
        XCTAssertEqual(mappedBasis.first?.0, "itok_upsampler.key_features_encoder.lfu.basis")
        XCTAssertEqual(mappedBasis.first?.1.shape, [1, 5, 5, 1])
    }

    func testComputeSegmentationFeaturesUpscalesToImageResolution() {
        let config = FalconPerceptionModelConfig(
            textConfig: FalconPerceptionTextConfig(
                hiddenSize: 16,
                numHiddenLayers: 2,
                numAttentionHeads: 4,
                headDim: 4,
                numKeyValueHeads: 2,
                vocabSize: 512,
                intermediateSize: 32,
                maxPositionEmbeddings: 128
            ),
            visionConfig: FalconPerceptionVisionConfig(
                spatialPatchSize: 16,
                temporalPatchSize: 1,
                channelSize: 3
            ),
            vocabSize: 512,
            coordDecDim: 64,
            coordOutDim: 32,
            sizeDecDim: 64,
            sizeOutDim: 32,
            segmOutDim: 8,
            numSegmLayers: 2
        )
        let model = FalconPerceptionModel(config: config)
        let hiddenState = MLX.zeros([1, 4, config.textConfig.hiddenSize], dtype: .float32)
        let inputIDs = MLXArray([
            Int32(config.imgID),
            Int32(config.imgID),
            Int32(config.imgID),
            Int32(config.imgID),
        ], [1, 4])
        let pixelValues = MLX.zeros([1, 32, 32, 3], dtype: .float32)

        let features = model.computeSegmentationFeatures(
            hiddenState: hiddenState,
            inputIDs: inputIDs,
            pixelValues: pixelValues,
            gridH: 2,
            gridW: 2
        )

        XCTAssertEqual(features?.shape, [1, 32, 32, 8])
    }

    func testFalconSegmentationConvExposesCheckpointBiasParameter() {
        let config = FalconPerceptionModelConfig(
            textConfig: FalconPerceptionTextConfig(
                hiddenSize: 16,
                numHiddenLayers: 1,
                numAttentionHeads: 4,
                headDim: 4,
                numKeyValueHeads: 2,
                vocabSize: 512,
                intermediateSize: 32,
                maxPositionEmbeddings: 128
            ),
            visionConfig: FalconPerceptionVisionConfig(
                spatialPatchSize: 16,
                temporalPatchSize: 1,
                channelSize: 3
            ),
            vocabSize: 512,
            segmOutDim: 8
        )
        let model = FalconPerceptionModel(config: config)
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("conv_segm.bias"))
    }

    func testFalconManualNormWeightAssignmentUpdatesNestedLayerWeights() {
        let config = FalconPerceptionModelConfig(
            textConfig: FalconPerceptionTextConfig(
                hiddenSize: 16,
                numHiddenLayers: 1,
                numAttentionHeads: 4,
                headDim: 4,
                numKeyValueHeads: 2,
                vocabSize: 512,
                intermediateSize: 32,
                maxPositionEmbeddings: 128
            ),
            visionConfig: FalconPerceptionVisionConfig(
                spatialPatchSize: 16,
                temporalPatchSize: 1,
                channelSize: 3
            ),
            vocabSize: 512
        )
        let model = FalconPerceptionModel(config: config)
        let normIn = MLXArray((0..<16).map(Float.init), [16])
        let normQK = MLXArray((0..<4).map(Float.init), [4])
        let normMLP = MLXArray((0..<16).map { Float($0) / 10 }, [16])

        XCTAssertTrue(
            FalconPerceptionGrounder.assignManuallyMappedWeight(
                normIn,
                for: "language_model.model.layers.0.self_attn._norm_w_in",
                into: model
            )
        )
        XCTAssertTrue(
            FalconPerceptionGrounder.assignManuallyMappedWeight(
                normQK,
                for: "language_model.model.layers.0.self_attn._norm_w_qk",
                into: model
            )
        )
        XCTAssertTrue(
            FalconPerceptionGrounder.assignManuallyMappedWeight(
                normMLP,
                for: "language_model.model.layers.0.mlp._norm_w",
                into: model
            )
        )

        XCTAssertEqual(model.languageModel.model.layers[0].selfAttn.normWIn.asArray(Float.self), normIn.asArray(Float.self))
        XCTAssertEqual(model.languageModel.model.layers[0].selfAttn.normWQK.asArray(Float.self), normQK.asArray(Float.self))
        XCTAssertEqual(model.languageModel.model.layers[0].mlp.normW.asArray(Float.self), normMLP.asArray(Float.self))
    }

    func testFalconAttentionExposesSinksForCheckpointLoading() {
        let config = FalconPerceptionModelConfig(
            textConfig: FalconPerceptionTextConfig(
                hiddenSize: 16,
                numHiddenLayers: 1,
                numAttentionHeads: 4,
                headDim: 4,
                numKeyValueHeads: 2,
                vocabSize: 512,
                intermediateSize: 32,
                maxPositionEmbeddings: 128
            ),
            visionConfig: FalconPerceptionVisionConfig(
                spatialPatchSize: 16,
                temporalPatchSize: 1,
                channelSize: 3
            ),
            vocabSize: 512
        )
        let model = FalconPerceptionModel(config: config)
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("language_model.model.layers.0.self_attn.sinks"))
        XCTAssertEqual(model.languageModel.model.layers[0].selfAttn.sinks.shape, [4])
    }

    func testCheckpointMappingKeepsAttentionSinksTensor() {
        let sinks = MLXArray((0..<16).map(Float.init), [16])
        let mapped = FalconPerceptionGrounder.mapCheckpointKey(
            key: "layers.0.attention.sinks",
            value: sinks
        )

        XCTAssertEqual(mapped.count, 1)
        XCTAssertEqual(mapped.first?.0, "language_model.model.layers.0.self_attn.sinks")
        XCTAssertEqual(mapped.first?.1.shape, [16])
    }

    #if canImport(CoreGraphics)
    func testSmartResizeSnapsToPatchMultiple() {
        let image = makeImage(width: 513, height: 299)
        let resized = FalconPerceptionProcessor.smartResize(image, factor: 16)

        XCTAssertEqual(resized.width % 16, 0)
        XCTAssertEqual(resized.height % 16, 0)
    }

    private func makeImage(width: Int, height: Int) -> CGImage {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytes = [UInt8](repeating: 255, count: width * height * bytesPerPixel)
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )!
    }
    #endif
}
