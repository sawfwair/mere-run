import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

final class HiDreamO1ConfigTests: MereRunCoreTestCase {
    func testDecodesUpstreamStyleConfig() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data(Self.configJSON.utf8))
        let config = try HiDreamO1Config.load(from: HiDreamO1Resources(rootURL: root))

        XCTAssertEqual(config.modelType, "qwen3_vl")
        XCTAssertEqual(config.imageTokenId, 151655)
        XCTAssertEqual(config.textConfig.hiddenSize, 4096)
        XCTAssertEqual(config.textConfig.ropeScaling.mropeSection, [24, 20, 20])
        XCTAssertEqual(config.visionConfig.hiddenSize, 1152)
        XCTAssertEqual(config.visionConfig.deepstackVisualIndexes, [8, 16, 24])
    }

    func testDevSchedulerUsesReferenceTimesteps() {
        let scheduler = HiDreamO1Scheduler(steps: 28, variant: .distilled, shift: 1.0)
        let timesteps = scheduler.timesteps.asArray(Float.self)
        let sigmas = scheduler.sigmas.asArray(Float.self)

        XCTAssertEqual(timesteps.prefix(4).map(Int.init), [999, 987, 974, 960])
        XCTAssertEqual(timesteps.last.map(Int.init), 8)
        XCTAssertEqual(sigmas.count, 29)
        XCTAssertEqual(sigmas.last ?? -1, 0, accuracy: 0.0001)
    }

    func testFullSchedulerMatchesUpstreamFlowUniPCSchedule() {
        let scheduler = HiDreamO1Scheduler(steps: 5, variant: .base, shift: 3.0)

        XCTAssertFalse(scheduler.usesFlashStep)
        XCTAssertEqual(scheduler.timestepValues.map(Int.init), [999, 922, 818, 666, 428])
        XCTAssertEqual(scheduler.sigmaValues.count, 6)
        XCTAssertEqual(scheduler.sigmaValues[0], 0.9998888, accuracy: 0.000001)
        XCTAssertEqual(scheduler.sigmaValues[1], 0.9229585, accuracy: 0.000001)
        XCTAssertEqual(scheduler.sigmaValues[4], 0.4284693, accuracy: 0.000001)
        XCTAssertEqual(scheduler.sigmaValues[5], 0, accuracy: 0.000001)
    }

    func testFullUniPCOneStepReturnsDenoisedPrediction() {
        let scheduler = HiDreamO1Scheduler(steps: 1, variant: .base, shift: 3.0)
        var state = HiDreamO1UniPCState(scheduler: scheduler)
        let sample = MLXArray([Float(10)], [1])
        let modelOutput = MLXArray([Float(2)], [1])

        let result = state.step(modelOutput: modelOutput, sample: sample, stepIndex: 0)
        MLX.eval(result)

        XCTAssertEqual(
            result.asArray(Float.self)[0],
            10 - scheduler.sigmaValues[0] * 2,
            accuracy: 0.0005
        )
    }

    func testSampleBuilderConstructsTextAndReferenceMasks() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data(Self.configJSON.utf8))
        let config = try HiDreamO1Config.load(from: HiDreamO1Resources(rootURL: root))

        let textSample = HiDreamO1SampleBuilder.textToImageSample(
            textTokenIds: [1, 2, 3],
            height: 64,
            width: 64,
            config: config
        )
        XCTAssertEqual(textSample.targetImageLength, 4)
        XCTAssertEqual(textSample.inputIds, [1, 2, 3])
        XCTAssertEqual(textSample.paddedInputIds, [1, 2, 3, 151652, 151655, 151655, 151655])
        XCTAssertEqual(textSample.tokenTypes, [0, 0, 1, 1, 1, 1, 1])
        XCTAssertEqual(textSample.vinputMask.filter { $0 }.count, 4)
        XCTAssertEqual(textSample.positionIds[0], [0, 1, 2, 4096, 4096, 4096, 4096])
        XCTAssertEqual(textSample.positionIds[1], [0, 1, 2, 4096, 4096, 4097, 4097])
        XCTAssertEqual(textSample.positionIds[2], [0, 1, 2, 4096, 4097, 4096, 4097])

        let refSample = HiDreamO1SampleBuilder.referenceSample(
            textTokenIds: [1, 2, 3],
            targetHeight: 64,
            targetWidth: 64,
            referenceSizes: [.init(width: 64, height: 64), .init(width: 64, height: 64)],
            config: config
        )
        XCTAssertEqual(refSample.targetImageLength, 4)
        XCTAssertEqual(refSample.referenceImageLengths, [4, 4])
        XCTAssertEqual(refSample.inputIds, [1, 2, 3])
        XCTAssertEqual(refSample.paddedInputIds.count, 15)
        XCTAssertEqual(refSample.tokenTypes, [0, 0] + Array(repeating: 1, count: 13))
        XCTAssertEqual(refSample.vinputMask.filter { $0 }.count, 12)

        let conditionSample = HiDreamO1SampleBuilder.referenceSample(
            textTokenIds: [1, 151652, 151655, 151655, 151655, 151655, 151653, 2, 3],
            targetHeight: 64,
            targetWidth: 64,
            referenceSizes: [.init(width: 64, height: 64)],
            conditionGrids: [.init(width: 2, height: 2)],
            config: config
        )
        XCTAssertEqual(conditionSample.paddedInputIds.count, 17)
        XCTAssertEqual(conditionSample.positionIds[0].count, 17)
        XCTAssertEqual(Array(conditionSample.tokenTypes[0..<8]), Array(repeating: 0, count: 8))
        XCTAssertEqual(conditionSample.vinputMask.filter { $0 }.count, 8)
    }

    func testPatchifyRoundTripShape() {
        let image = MLXArray(Array(repeating: Float(0.5), count: 3 * 64 * 64), [3, 64, 64])
        let patches = HiDreamO1SampleBuilder.patchifyCHW(image)
        let restored = HiDreamO1SampleBuilder.unpatchifyCHW(patches, height: 64, width: 64)

        XCTAssertEqual(patches.shape, [4, 3072])
        XCTAssertEqual(restored.shape, [3, 64, 64])
    }

    #if canImport(CoreGraphics)
    func testImagePreprocessorNormalizesAndPatchifiesRGB() throws {
        let image = try Self.makeSolidImage(width: 32, height: 32, rgba: (255, 0, 0, 255))
        let tensor = try HiDreamO1ImagePreprocessor.patchTensor(
            from: image,
            resolution: .init(width: 32, height: 32)
        )

        XCTAssertEqual(tensor.imageCHW.shape, [3, 32, 32])
        XCTAssertEqual(tensor.patches.shape, [1, 3_072])
        let patch = tensor.patches.asArray(Float.self)
        XCTAssertEqual(patch[0], 1.0, accuracy: 0.0001)
        XCTAssertEqual(patch[1_024], -1.0, accuracy: 0.0001)
        XCTAssertEqual(patch[2_048], -1.0, accuracy: 0.0001)
    }

    func testVisionConditionPreprocessorMatchesQwen3PatchShape() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data(Self.configJSON.utf8))
        let config = try HiDreamO1Config.load(from: HiDreamO1Resources(rootURL: root))
        let image = try Self.makeSolidImage(width: 64, height: 64, rgba: (0, 255, 0, 255))

        let tensor = try HiDreamO1ImagePreprocessor.visionConditionTensor(
            from: image,
            resolution: .init(width: 64, height: 64),
            config: config
        )

        XCTAssertEqual(tensor.grid.temporal, 1)
        XCTAssertEqual(tensor.grid.height, 4)
        XCTAssertEqual(tensor.grid.width, 4)
        XCTAssertEqual(tensor.mergedGrid, .init(width: 2, height: 2))
        XCTAssertEqual(tensor.tokenCount, 4)
        XCTAssertEqual(tensor.pixelValues.shape, [1, 16, 1_536])
    }

    func testPreviewRendererWritesPNG() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("hidream-preview.png")
        let image = HiDreamO1PreviewRenderer.promptPreview(
            prompt: "a small brass camera",
            seed: 42,
            resolution: .init(width: 64, height: 64)
        )

        try HiDreamO1ImagePreprocessor.saveNormalizedCHW(image, to: output)

        XCTAssertTrue(FileManager.default.fileExists(atPath: output.path))
        guard let source = CGImageSourceCreateWithURL(output as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] else {
            XCTFail("Expected a readable PNG.")
            return
        }
        XCTAssertEqual(properties[kCGImagePropertyPixelWidth] as? Int, 64)
        XCTAssertEqual(properties[kCGImagePropertyPixelHeight] as? Int, 64)
    }
    #endif

    func testCoreLayerShapesAndTimestepFrequencies() {
        let embedding = HiDreamO1TimestepEmbedder.timestepEmbedding(MLXArray([Float(0)]), dim: 4)
        XCTAssertEqual(embedding.shape, [1, 4])
        XCTAssertEqual(embedding.asArray(Float.self), [1, 1, 0, 0])

        let timestepEmbedder = HiDreamO1TimestepEmbedder(hiddenSize: 8, frequencyEmbeddingSize: 4)
        let timestepOut = timestepEmbedder(MLXArray([Float(0.5)]))
        XCTAssertEqual(timestepOut.shape, [1, 8])

        let patchEmbed = HiDreamO1BottleneckPatchEmbed(patchSize: 2, inChannels: 3, bottleneckDim: 4, hiddenSize: 8)
        let patches = MLXArray(Array(repeating: Float(0.25), count: 2 * 12), [2, 12])
        XCTAssertEqual(patchEmbed(patches).shape, [2, 8])

        let finalLayer = HiDreamO1FinalLayer(hiddenSize: 8, patchSize: 2, outChannels: 3)
        XCTAssertEqual(finalLayer(MLXArray(Array(repeating: Float(0.5), count: 2 * 8), [2, 8])).shape, [2, 12])
    }

    func testGenerationAttentionMaskKeepsPaddedCFGRowsIndependent() {
        MLXRandom.seed(17)
        let encoder = QwenEncoder(configuration: QwenTextEncoderConfiguration(
            vocabSize: 32,
            hiddenSize: 8,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            intermediateSize: 16,
            ropeTheta: 10_000,
            maxPositionEmbeddings: 32,
            rmsNormEps: 1e-6,
            headDim: 2
        ))
        let source = MLXRandom.normal([1, 4, 8]).asType(.bfloat16)
        let serialTypes = MLXArray([Int32(0), 0, 1, 1], [1, 4])
        let serial = encoder.forward(
            embeddings: source,
            attentionMask: nil,
            tokenTypes: serialTypes
        ).lastHiddenState

        let padding = MLXArray(Array(repeating: Float(1_000), count: 8), [1, 1, 8]).asType(.bfloat16)
        let firstRow = MLX.concatenated([source, padding], axis: 1)
        let secondRow = MLXRandom.normal([1, 5, 8]).asType(.bfloat16)
        let batch = MLX.concatenated([firstRow, secondRow], axis: 0)
        let batchTypes = MLXArray([
            Int32(0), 0, 1, 1, 0,
            0, 0, 1, 1, 1,
        ], [2, 5])
        let batchMask = MLXArray([
            Int32(1), 1, 1, 1, 0,
            1, 1, 1, 1, 1,
        ], [2, 5])
        let batched = encoder.forward(
            embeddings: batch,
            attentionMask: batchMask,
            tokenTypes: batchTypes
        ).lastHiddenState
        MLX.eval(serial, batched)

        let difference = MLX.max(MLX.abs(
            serial[0, 0..<4, 0...].asType(.float32)
                - batched[0, 0..<4, 0...].asType(.float32)
        )).item(Float.self)
        XCTAssertLessThan(difference, 1e-4)
    }

    func testPixelHeadWeightMapperKeepsOnlyHiDreamSpecificKeys() {
        let value = MLXArray([Float(1)])

        XCTAssertEqual(
            HiDreamO1WeightMapper.mapPixelHeadKey("model.t_embedder1.mlp.0.weight", value: value).first?.0,
            "t_embedder1.mlp.0.weight"
        )
        XCTAssertEqual(
            HiDreamO1WeightMapper.mapPixelHeadKey("model.x_embedder.proj1.weight", value: value).first?.0,
            "x_embedder.proj1.weight"
        )
        XCTAssertEqual(
            HiDreamO1WeightMapper.mapPixelHeadKey("final_layer2.linear.bias", value: value).first?.0,
            "final_layer2.linear.bias"
        )
        XCTAssertTrue(HiDreamO1WeightMapper.mapPixelHeadKey("model.language_model.layers.0.self_attn.q_proj.weight", value: value).isEmpty)
    }

    func testResolutionAndReferenceSizingMatchUpstreamRules() {
        XCTAssertEqual(HiDreamO1SampleBuilder.closestResolution(width: 1024, height: 1024), .init(width: 2048, height: 2048))
        XCTAssertEqual(HiDreamO1SampleBuilder.closestResolution(width: 1920, height: 1080), .init(width: 2560, height: 1440))
        XCTAssertEqual(HiDreamO1SampleBuilder.closestResolution(width: 900, height: 1600), .init(width: 1440, height: 2560))

        XCTAssertEqual(
            HiDreamO1SampleBuilder.targetResolution(
                width: 1024,
                height: 1024,
                referenceOriginalSizes: [.init(width: 4000, height: 1000)],
                keepOriginalAspect: true
            ),
            .init(width: 4096, height: 1024)
        )
        XCTAssertEqual(HiDreamO1SampleBuilder.referenceMaxSize(targetWidth: 2048, targetHeight: 1024, referenceCount: 2), 1536)
        XCTAssertEqual(HiDreamO1SampleBuilder.conditionSize(referenceCount: 5), 288)
        XCTAssertEqual(
            HiDreamO1SampleBuilder.vlmConditionSize(originalWidth: 2_000, originalHeight: 1_000, maxSize: 384),
            .init(width: 512, height: 256)
        )
    }

    func testPromptBuilderCreatesReferenceImageMessages() throws {
        let t2i = HiDreamO1TokenizerAndTemplate.messages(prompt: "a brass camera", referenceImageCount: 0)
        XCTAssertEqual(t2i.count, 1)
        XCTAssertEqual(t2i[0]["role"] as? String, "user")
        XCTAssertEqual(t2i[0]["content"] as? String, "a brass camera")

        let refs = HiDreamO1TokenizerAndTemplate.messages(prompt: "move the subject to a studio", referenceImageCount: 2)
        XCTAssertEqual(refs[0]["role"] as? String, "user")
        let content = try XCTUnwrap(refs[0]["content"] as? [[String: String]])
        XCTAssertEqual(content.prefix(2).map { $0["type"] }, ["image", "image"])
        XCTAssertEqual(content.last?["type"], "text")
        XCTAssertEqual(content.last?["text"], "move the subject to a studio")
        XCTAssertEqual(HiDreamO1TokenizerAndTemplate.boiToken, "<|boi_token|>")
        XCTAssertEqual(HiDreamO1TokenizerAndTemplate.tmsToken, "<|tms_token|>")
        XCTAssertEqual(HiDreamO1TokenizerAndTemplate.tmsTokenId, 151_673)
    }

    private static let configJSON = """
    {
      "architectures": ["Qwen3VLForConditionalGeneration"],
      "image_token_id": 151655,
      "model_type": "qwen3_vl",
      "text_config": {
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "max_position_embeddings": 262144,
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "rope_scaling": {
          "mrope_interleaved": true,
          "mrope_section": [24, 20, 20],
          "rope_type": "default"
        },
        "rope_theta": 5000000,
        "vocab_size": 151936
      },
      "video_token_id": 151656,
      "vision_config": {
        "deepstack_visual_indexes": [8, 16, 24],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "intermediate_size": 4304,
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 4096,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
      },
      "vision_end_token_id": 151653,
      "vision_start_token_id": 151652
    }
    """

    #if canImport(CoreGraphics)
    private static func makeSolidImage(
        width: Int,
        height: Int,
        rgba: (UInt8, UInt8, UInt8, UInt8)
    ) throws -> CGImage {
        let pixelCount = width * height
        var bytes: [UInt8] = []
        bytes.reserveCapacity(pixelCount * 4)
        for _ in 0..<pixelCount {
            bytes.append(rgba.0)
            bytes.append(rgba.1)
            bytes.append(rgba.2)
            bytes.append(rgba.3)
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else {
            throw HiDreamO1ImagePreprocessorError.pixelBufferFailed
        }
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            throw HiDreamO1ImagePreprocessorError.pixelBufferFailed
        }
        return image
    }
    #endif
}
