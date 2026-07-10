import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class LingBotVideoTests: XCTestCase {
    func testPromptTemplateMatchesReleasedTextOnlyFormat() {
        let prompt = "a blue robot"
        let expected = """
        <|im_start|>system
        Given a user input that may include a text prompt alone, a text prompt with an image reference, or a text prompt with a video reference or a video reference alone, generate an \"Enhanced prompt\" that provides detailed visual descriptions suitable for video generation. Evaluate the level of detail in the user's input: if it is simple, enrich it by adding specifics about colors, shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal progression, and spatial relationships to create vivid, concrete, and temporally coherent scenes to create vivid and concrete scenes. Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:<|im_end|>
        <|im_start|>user
        a blue robot<|im_end|>
        <|im_start|>assistant

        """

        XCTAssertEqual(
            LingBotVideoPromptEncoder.promptPrefix + prompt + LingBotVideoPromptEncoder.promptSuffix,
            expected
        )
    }

    func testTransformerConfigDecodesDenseCheckpointShape() throws {
        let config = try JSONDecoder().decode(
            LingBotVideoTransformerConfig.self,
            from: Data(Self.transformerConfigJSON.utf8)
        )

        XCTAssertEqual(config.hiddenSize, 2048)
        XCTAssertEqual(config.depth, 24)
        XCTAssertEqual(config.numExperts, 0)
        XCTAssertEqual(config.patchSize, [1, 2, 2])
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.axesDims.reduce(0, +), config.headDim)
    }

    func testDenseTransformerTinyForwardPreservesLatentShape() {
        let config = LingBotVideoTransformerConfig(
            axesDims: [2, 2, 4],
            axesLens: [32, 32, 32],
            depth: 1,
            freqDim: 8,
            hiddenSize: 16,
            inChannels: 4,
            intermediateSize: 24,
            numAttentionHeads: 2,
            outChannels: 4,
            patchSize: [1, 2, 2],
            textDim: 12
        )
        let model = LingBotVideoTransformer(config: config)
        let output = model(
            hiddenStates: MLXRandom.normal([1, 4, 1, 4, 4]).asType(.float32),
            timestep: MLXArray([Float(500)]),
            encoderHiddenStates: MLXRandom.normal([1, 3, 12]).asType(.bfloat16)
        )

        MLX.eval(output)
        XCTAssertEqual(output.shape, [1, 4, 1, 4, 4])
        XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testDenseTransformerReportsCompletedBlocksInOrder() {
        let config = LingBotVideoTransformerConfig(
            axesDims: [2, 2, 4],
            axesLens: [32, 32, 32],
            depth: 3,
            freqDim: 8,
            hiddenSize: 16,
            inChannels: 4,
            intermediateSize: 24,
            numAttentionHeads: 2,
            outChannels: 4,
            patchSize: [1, 2, 2],
            textDim: 12
        )
        let model = LingBotVideoTransformer(config: config)
        var completedBlocks: [Int] = []
        let output = model(
            hiddenStates: MLXRandom.normal([1, 4, 1, 4, 4]).asType(.float32),
            timestep: MLXArray([Float(500)]),
            encoderHiddenStates: MLXRandom.normal([1, 3, 12]).asType(.bfloat16),
            blockProgressHandler: { block, total in
                XCTAssertEqual(total, 3)
                completedBlocks.append(block)
            }
        )

        MLX.eval(output)
        XCTAssertEqual(completedBlocks, [1, 2, 3])
    }

    func testBatchedCFGMatchesIndependentMaskedForwards() {
        let config = LingBotVideoTransformerConfig(
            axesDims: [2, 2, 4],
            axesLens: [32, 32, 32],
            depth: 2,
            freqDim: 8,
            hiddenSize: 16,
            inChannels: 4,
            intermediateSize: 24,
            numAttentionHeads: 2,
            outChannels: 4,
            patchSize: [1, 2, 2],
            textDim: 12
        )
        let model = LingBotVideoTransformer(config: config)
        let latents = MLXRandom.normal([1, 4, 1, 4, 4]).asType(.float32)
        let timestep = MLXArray([Float(500)])
        let positive = MLXRandom.normal([1, 3, 12]).asType(.bfloat16)
        let negative = MLXRandom.normal([1, 2, 12]).asType(.bfloat16)
        let conditional = model(
            hiddenStates: latents,
            timestep: timestep,
            encoderHiddenStates: positive
        )
        let unconditional = model(
            hiddenStates: latents,
            timestep: timestep,
            encoderHiddenStates: negative
        )

        let paddedNegative = MLX.padded(negative, widths: [[0, 0], [0, 1], [0, 0]])
        let batched = model(
            hiddenStates: MLX.concatenated([latents, latents], axis: 0),
            timestep: MLX.concatenated([timestep, timestep], axis: 0),
            encoderHiddenStates: MLX.concatenated([positive, paddedNegative], axis: 0),
            encoderAttentionMask: MLXArray([Int32(1), 1, 1, 1, 1, 0], [2, 3]),
            encoderTextLengths: [3, 2]
        )
        MLX.eval(conditional, unconditional, batched)

        let conditionalError = MLX.max(
            MLX.abs(conditional - batched[0..<1])
        ).item(Float.self)
        let unconditionalError = MLX.max(
            MLX.abs(unconditional - batched[1..<2])
        ).item(Float.self)
        XCTAssertLessThan(conditionalError, 0.01)
        XCTAssertLessThan(unconditionalError, 0.01)
    }

    func testLingBotFusedRMSNormMatchesScalarReference() {
        let norm = LingBotVideoRMSNorm(dimensions: 8, eps: 1e-6)
        let input = MLXRandom.normal([2, 5, 8]).asType(.bfloat16)
        let actual = norm(input).asType(.float32)
        let values = input.asType(.float32)
        let variance = values.square().mean(axis: -1, keepDims: true)
        let expected = values * rsqrt(variance + 1e-6)
        MLX.eval(actual, expected)

        let maximumError = MLX.max(MLX.abs(actual - expected)).item(Float.self)
        XCTAssertLessThan(maximumError, 0.01)
    }

    func testDenseTransformerWeightMapperMatchesModulePaths() {
        let config = LingBotVideoTransformerConfig(
            axesDims: [2, 2, 4],
            axesLens: [32, 32, 32],
            depth: 1,
            freqDim: 8,
            hiddenSize: 16,
            inChannels: 4,
            intermediateSize: 24,
            numAttentionHeads: 2,
            outChannels: 4,
            patchSize: [1, 2, 2],
            textDim: 12
        )
        let model = LingBotVideoTransformer(config: config)
        let parameterKeys = Set(model.parameters().flattened().map(\.0))
        let sourceKeys = [
            "patch_embedder.weight",
            "time_embedder.linear_1.weight",
            "time_modulation.1.weight",
            "text_embedder.norm.weight",
            "blocks.0.scale_shift_table",
            "blocks.0.attn.norm_q.weight",
            "blocks.0.attn.to_q.weight",
            "blocks.0.attn.to_out.bias",
            "blocks.0.norm_post_attn.weight",
            "blocks.0.ffn.gate_proj.weight",
            "blocks.0.norm_post_ffn.weight",
            "norm_out_modulation.1.weight",
            "proj_out.weight",
        ]

        for sourceKey in sourceKeys {
            let sourceValue = sourceKey.hasSuffix("scale_shift_table")
                ? MLX.zeros([1, 96], dtype: .float32)
                : MLX.zeros([1], dtype: .float32)
            let mapped = LingBotVideoTransformer.mapWeight(
                key: sourceKey,
                value: sourceValue,
                config: config
            )
            XCTAssertEqual(mapped.count, 1)
            XCTAssertTrue(parameterKeys.contains(mapped[0].0), "Missing mapped path for \(sourceKey): \(mapped[0].0)")
            if sourceKey.hasSuffix("scale_shift_table") {
                XCTAssertEqual(mapped[0].1.shape, [1, 6, 16])
            }
        }
    }

    func testPromptEncoderMapsOnlyQwenLanguageWeights() {
        XCTAssertEqual(
            LingBotVideoPromptEncoder.mapLanguageWeightKey("model.language_model.layers.0.self_attn.q_proj.weight"),
            "encoder.layers.0.self_attn.q_proj.weight"
        )
        XCTAssertNil(LingBotVideoPromptEncoder.mapLanguageWeightKey("model.visual.blocks.0.attn.qkv.weight"))
    }

    func testReleasedQwenTokenizerFixtureWhenDenseRootProvided() throws {
        guard let root = ProcessInfo.processInfo.environment["MERERUN_LINGBOT_DENSE_ROOT"] else {
            throw XCTSkip("Set MERERUN_LINGBOT_DENSE_ROOT to run the released tokenizer fixture.")
        }
        let tokenizer = try QwenTokenizer.load(
            from: URL(fileURLWithPath: root).appendingPathComponent("processor", isDirectory: true),
            maxLengthOverride: LingBotVideoPromptEncoder.maxTokenLength
        )
        let tokenization = LingBotVideoPromptEncoder.tokenize("a blue robot", using: tokenizer)

        XCTAssertEqual(tokenization.cropStart, 140)
        XCTAssertEqual(tokenization.ids.count, 148)
        XCTAssertEqual(fnv1a64(tokenization.ids), 0x6698_CF57_F45E_7EE4)
    }

    func testFlowUniPCScheduleAndStepsRemainFinite() throws {
        let scheduler = LingBotVideoFlowUniPCScheduler()
        try scheduler.setTimesteps(stepCount: 4, shift: 3)

        XCTAssertEqual(scheduler.timesteps.count, 4)
        XCTAssertEqual(scheduler.sigmas.count, 5)
        XCTAssertEqual(scheduler.sigmas.last, 0)
        XCTAssertTrue(zip(scheduler.sigmas, scheduler.sigmas.dropFirst()).allSatisfy { $0 > $1 })

        var sample = MLX.ones([1, 2, 1, 2, 2], dtype: .float32)
        for _ in scheduler.timesteps {
            sample = try scheduler.step(modelOutput: sample * 0.1, sample: sample)
            MLX.eval(sample)
        }
        XCTAssertEqual(sample.shape, [1, 2, 1, 2, 2])
        XCTAssertTrue(sample.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testFlowUniPCMatchesReleasedScalarReferenceFixture() throws {
        let scheduler = LingBotVideoFlowUniPCScheduler()
        try scheduler.setTimesteps(stepCount: 4, shift: 3)
        let expectedSigmas = [
            0.99966644429619733,
            0.89963978387032217,
            0.74962481240620304,
            0.49966655551850619,
            0,
        ]
        for (actual, expected) in zip(scheduler.sigmas, expectedSigmas) {
            XCTAssertEqual(actual, expected, accuracy: 1e-12)
        }

        let modelOutputs: [Float] = [0.1, 0.2, 0.3, 0.4]
        let expectedSamples: [Float] = [
            0.98999733395741252,
            0.95482874738228163,
            0.86234027421555981,
            0.66247365200815733,
        ]
        var sample = MLXArray(Float(1))
        for (modelOutput, expected) in zip(modelOutputs, expectedSamples) {
            sample = try scheduler.step(modelOutput: MLXArray(modelOutput), sample: sample)
            MLX.eval(sample)
            XCTAssertEqual(sample.item(Float.self), expected, accuracy: 2e-6)
        }
    }

    func testRefinerSigmaScheduleMatchesReleasedLowNoiseTailContract() throws {
        let sigmas = try LingBotVideoFlowUniPCScheduler.refinerSigmas(
            stepCount: 8,
            shift: 3,
            threshold: 0.85,
            tailStepCount: 2
        )

        XCTAssertEqual(sigmas.count, 8)
        XCTAssertEqual(sigmas.first ?? 0, Double(Float(0.85)), accuracy: 1e-8)
        XCTAssertTrue(zip(sigmas, sigmas.dropFirst()).allSatisfy { $0 > $1 })
        let tailStart = sigmas[sigmas.count - 3]
        XCTAssertEqual(sigmas[sigmas.count - 2], tailStart * 2 / 3, accuracy: 1e-8)
        XCTAssertEqual(sigmas[sigmas.count - 1], tailStart / 3, accuracy: 1e-8)

        let scheduler = LingBotVideoFlowUniPCScheduler()
        try scheduler.setTimesteps(sigmas: sigmas)
        XCTAssertEqual(scheduler.timesteps.count, 8)
        XCTAssertEqual(scheduler.sigmas.last, 0)
    }

    func testTemporalMetricsDistinguishStableAndCollapsedFrames() {
        let stable = MLX.full([1, 3, 2, 2, 3], values: MLXArray(Float(128))).asType(.uint8)
        let collapsed = MLX.concatenated([
            MLX.zeros([1, 1, 2, 2, 3], dtype: .uint8),
            MLX.full([1, 1, 2, 2, 3], values: MLXArray(UInt8.max)),
            MLX.zeros([1, 1, 2, 2, 3], dtype: .uint8),
        ], axis: 1)

        let stableMetrics = LingBotVideoTemporalMetrics.analyze(frames: stable)
        let collapsedMetrics = LingBotVideoTemporalMetrics.analyze(frames: collapsed)
        XCTAssertEqual(stableMetrics.meanLumaDelta, 0, accuracy: 1e-6)
        XCTAssertFalse(stableMetrics.isLikelyUnstable)
        XCTAssertFalse(stableMetrics.isInformative)
        XCTAssertEqual(collapsedMetrics.meanLumaDelta, 255, accuracy: 1e-3)
        XCTAssertEqual(collapsedMetrics.peakLumaDelta, 255, accuracy: 1e-3)
        XCTAssertTrue(collapsedMetrics.isLikelyUnstable)
        XCTAssertTrue(collapsedMetrics.isInformative)
    }

    func testPromptSampleMatchesReleasedCaptionAndDurationContract() throws {
        let data = Data(#"""
        {
          "caption": {
            "comprehensive_description": {"scene_content_description": "A robot folds a towel."},
            "prominent_elements": []
          },
          "duration": 5
        }
        """#.utf8)

        let sample = try LingBotVideoPromptSample.decode(data)
        let caption = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(sample.caption.utf8)) as? [String: Any]
        )
        XCTAssertNotNil(caption["comprehensive_description"])
        XCTAssertNotNil(caption["prominent_elements"])
        XCTAssertEqual(
            sample.caption,
            #"{"comprehensive_description":{"scene_content_description":"A robot folds a towel."},"prominent_elements":[]}"#
        )
        XCTAssertEqual(sample.duration, 5)
    }

    func testPromptSampleAcceptsListAndExcludesRuntimeFieldsFromFallbackCaption() throws {
        let data = Data(#"[{"scene":"workbench","duration":"6","fps":24,"width":832}]"#.utf8)

        let sample = try LingBotVideoPromptSample.decode(data)
        let caption = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(sample.caption.utf8)) as? [String: Any]
        )
        XCTAssertEqual(caption["scene"] as? String, "workbench")
        XCTAssertNil(caption["duration"])
        XCTAssertNil(caption["fps"])
        XCTAssertNil(caption["width"])
        XCTAssertEqual(sample.caption, #"{"scene":"workbench"}"#)
        XCTAssertEqual(sample.duration, 6)
    }

    func testCompactJSONDocumentPreservesMemberOrderAndStringWhitespace() throws {
        let compact = try LingBotVideoPromptSample.compactJSONDocument(
            Data("{ \"z\" : [\"a b\", 2], \"a\" : { \"x\" : true } }".utf8)
        )
        XCTAssertEqual(compact, #"{"z":["a b",2],"a":{"x":true}}"#)
    }

    func testReleasedRefinerDimensionsAreDefaultedIndependentlyOfBaseSize() {
        let options = LingBotVideoGenerationOptions(prompt: "a robot folds a towel")
        XCTAssertEqual(options.width, 320)
        XCTAssertEqual(options.height, 192)
        XCTAssertEqual(options.numFrames, 9)
        XCTAssertEqual(options.resolvedRefinerWidth, 1_920)
        XCTAssertEqual(options.resolvedRefinerHeight, 1_088)
    }

    func testStreamingWanDecodeProducesFourNPlusOneFrames() {
        let vae = AutoencoderKL3D(config: VAE3DConfig(
            inChannels: 3,
            outChannels: 3,
            latentChannels: 2,
            blockOutChannels: [4, 8, 16, 16],
            layersPerBlock: 1,
            scalingFactor: 1,
            temporalCompressionRatio: 4,
            midBlockAddAttention: true
        ))
        let output = vae.decodeStreaming(MLXRandom.normal([1, 2, 3, 2, 2]).asType(.float32))

        MLX.eval(output)
        XCTAssertEqual(output.shape, [1, 3, 9, 16, 16])
        XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testStreamingWanDecodeIsCausalAcrossLatentWindows() {
        let vae = AutoencoderKL3D(config: VAE3DConfig(
            inChannels: 3,
            outChannels: 3,
            latentChannels: 2,
            blockOutChannels: [4, 8, 16, 16],
            layersPerBlock: 1,
            scalingFactor: 1,
            temporalCompressionRatio: 4,
            midBlockAddAttention: true
        ))
        let latents = MLXRandom.normal(
            [1, 2, 5, 2, 2],
            key: MLXRandom.key(19)
        ).asType(.float32)
        let full = vae.decodeStreaming(latents)
        let prefix = vae.decodeStreaming(latents[0..., 0..., 0..<4, 0..., 0...])
        MLX.eval(full, prefix)

        let fullPrefix = full[0..., 0..., 0..<prefix.dim(2), 0..., 0...]
        let maxDifference = MLX.max(MLX.abs(fullPrefix - prefix)).item(Float.self)
        XCTAssertLessThan(maxDifference, 1e-4)
    }

    func testWanPosteriorSamplingIsDeterministicForExplicitKey() {
        let vae = AutoencoderKL3D(config: VAE3DConfig(
            inChannels: 3,
            outChannels: 3,
            latentChannels: 2,
            blockOutChannels: [4, 8, 16, 16],
            layersPerBlock: 1,
            scalingFactor: 1,
            temporalCompressionRatio: 4,
            midBlockAddAttention: true
        ))
        let images = MLXRandom.normal(
            [1, 3, 5, 16, 16],
            key: MLXRandom.key(11)
        ).asType(.float32)
        let first = vae.encodeSampled(images, key: MLXRandom.key(22))
        let second = vae.encodeSampled(images, key: MLXRandom.key(22))
        let different = vae.encodeSampled(images, key: MLXRandom.key(23))

        MLX.eval(first, second, different)
        XCTAssertEqual(first.shape, [1, 2, 2, 2, 2])
        XCTAssertEqual(first.asArray(Float.self), second.asArray(Float.self))
        XCTAssertNotEqual(first.asArray(Float.self), different.asArray(Float.self))
        XCTAssertTrue(first.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testWanTemporalPixelShuffleUsesScaleMajorChannelOrder() {
        let input = MLXArray([Float(0), 1, 2, 3]).reshaped(1, 4, 1, 1, 1)
        let shuffled = wanTemporalPixelShuffle(input, scale: 2)

        MLX.eval(shuffled)
        XCTAssertEqual(shuffled.shape, [1, 2, 2, 1, 1])
        XCTAssertEqual(shuffled.asArray(Float.self), [0, 2, 1, 3])
    }

    func testResourcesAcceptDenseLayoutAndRequireQuantizedMoEForInference() throws {
        let root = try makeModelRoot(numExperts: 0)
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = try LingBotVideoResources(rootURL: root)
        XCTAssertEqual(resources.transformerConfig.numExperts, 0)
        XCTAssertEqual(resources.vaeConfig.temporalCompressionRatio, 4)

        let moeRoot = try makeModelRoot(numExperts: 64)
        defer { try? FileManager.default.removeItem(at: moeRoot) }
        let moeResources = try LingBotVideoResources(rootURL: moeRoot)
        XCTAssertThrowsError(try moeResources.validateForInference()) { error in
            guard case LingBotVideoResources.ResourceError.unquantizedMoE(64) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testResourcesRequireCompleteQuantizedMoERefiner() throws {
        let root = try makeModelRoot(numExperts: 64)
        defer { try? FileManager.default.removeItem(at: root) }
        let refiner = root.appendingPathComponent("refiner", isDirectory: true)
        try FileManager.default.createDirectory(at: refiner, withIntermediateDirectories: true)
        let transformerConfig = Self.transformerConfigJSON.replacingOccurrences(
            of: "\"num_experts\": 0",
            with: "\"num_experts\": 64"
        )
        try Data(transformerConfig.utf8).write(to: refiner.appendingPathComponent("config.json"))
        try Data("{\"metadata\":{},\"weight_map\":{}}".utf8).write(
            to: refiner.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
        )
        let quantization = LingBotVideoQuantizationConfig(
            bits: 4,
            groupSize: 64,
            includesRefiner: true,
            sourceModelID: "video-lingbot-moe-30b-a3b"
        )
        try JSONEncoder().encode(quantization).write(
            to: root.appendingPathComponent(LingBotVideoQuantizationConfig.filename)
        )

        let resources = try LingBotVideoResources(rootURL: root)
        XCTAssertNoThrow(try resources.validateForInference())
        XCTAssertNoThrow(try resources.validateForRefiner())
        XCTAssertEqual(resources.refinerConfig?.numExperts, 64)
    }

    func testQuantizedMoETransformerTinyForwardPreservesLatentShape() {
        let config = LingBotVideoTransformerConfig(
            axesDims: [2, 2, 4],
            axesLens: [32, 32, 32],
            depth: 1,
            freqDim: 8,
            hiddenSize: 32,
            inChannels: 4,
            intermediateSize: 64,
            moeIntermediateSize: 32,
            numAttentionHeads: 4,
            numExperts: 4,
            numExpertsPerTok: 2,
            nGroup: 2,
            nSharedExperts: 1,
            outChannels: 4,
            patchSize: [1, 2, 2],
            routedScalingFactor: 2.5,
            textDim: 12,
            topKGroup: 1
        )
        let quantization = LingBotVideoQuantizationConfig(
            bits: 4,
            groupSize: 32,
            includesRefiner: false,
            sourceModelID: "tiny-lingbot-moe"
        )
        let model = LingBotVideoTransformer(config: config, quantization: quantization)
        let output = model(
            hiddenStates: MLXRandom.normal([1, 4, 1, 4, 4]).asType(.float32),
            timestep: MLXArray([Float(500)]),
            encoderHiddenStates: MLXRandom.normal([1, 3, 12]).asType(.bfloat16)
        )

        MLX.eval(output)
        XCTAssertEqual(output.shape, [1, 4, 1, 4, 4])
        XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testMoEWeightMapperNestsRawExpertWeightsAndPreservesQuantizedKeys() {
        let config = LingBotVideoTransformerConfig(
            axesDims: [2, 2, 4],
            axesLens: [32, 32, 32],
            depth: 1,
            freqDim: 8,
            hiddenSize: 32,
            inChannels: 4,
            intermediateSize: 64,
            moeIntermediateSize: 32,
            numAttentionHeads: 4,
            numExperts: 4,
            numExpertsPerTok: 2,
            nGroup: 2,
            nSharedExperts: 1,
            outChannels: 4,
            patchSize: [1, 2, 2],
            textDim: 12,
            topKGroup: 1
        )
        let raw = LingBotVideoTransformer.mapWeight(
            key: "blocks.0.ffn.experts.w1",
            value: MLX.zeros([4, 8, 16]),
            config: config
        )
        let quantized = LingBotVideoTransformer.mapWeight(
            key: "blocks.0.ffn.experts.w1.scales",
            value: MLX.zeros([4, 8, 1]),
            config: config
        )

        XCTAssertEqual(raw.first?.0, "blocks.0.ffn.experts.w1.weight")
        XCTAssertEqual(quantized.first?.0, "blocks.0.ffn.experts.w1.scales")

        let model = LingBotVideoTransformer(
            config: config,
            quantization: LingBotVideoQuantizationConfig(
                bits: 4,
                groupSize: 32,
                includesRefiner: false,
                sourceModelID: "tiny-lingbot-moe"
            )
        )
        let parameterKeys = Set(model.parameters().flattened().map(\.0))
        XCTAssertTrue(parameterKeys.contains("blocks.0.ffn.experts.w1.weight"))
        XCTAssertTrue(parameterKeys.contains("blocks.0.ffn.experts.w1.scales"))
        XCTAssertTrue(parameterKeys.contains("blocks.0.ffn.experts.w1.biases"))
        XCTAssertTrue(parameterKeys.contains("blocks.0.ffn.router.weight"))
        XCTAssertTrue(parameterKeys.contains("blocks.0.ffn.router.e_score_correction_bias"))
    }

    func testMoERouterUsesCorrectionOnlyForSelectionAndNormalizesBaseScores() throws {
        let config = LingBotVideoTransformerConfig(
            hiddenSize: 4,
            intermediateSize: 8,
            moeIntermediateSize: 4,
            numAttentionHeads: 1,
            numExperts: 4,
            numExpertsPerTok: 2,
            nGroup: 2,
            routedScalingFactor: 2.5,
            topKGroup: 1
        )
        let router = LingBotVideoRouter(config: config)
        try router.update(parameters: ModuleParameters.unflattened([
            ("weight", MLX.zeros([4, 4])),
            ("e_score_correction_bias", MLXArray([Float(0), 0, 2, 1])),
        ]), verify: .none)

        let routing = router(MLX.ones([1, 1, 4]))
        MLX.eval(routing.indices, routing.scores)
        XCTAssertEqual(routing.indices.asArray(Int32.self).sorted(), [2, 3])
        XCTAssertEqual(routing.scores.asArray(Float.self), [1.25, 1.25])
    }

    func testMoEQuantizerWritesNativeExpertWeightsAndResumesShard() throws {
        let fileManager = FileManager.default
        let source = fileManager.temporaryDirectory
            .appendingPathComponent("LingBotMoESource.\(UUID().uuidString)", isDirectory: true)
        let output = fileManager.temporaryDirectory
            .appendingPathComponent("LingBotMoEOutput.\(UUID().uuidString)", isDirectory: true)
        let transformer = fileManager.temporaryDirectory
            .appendingPathComponent("LingBotMoETransformer.\(UUID().uuidString)", isDirectory: true)
        defer {
            try? fileManager.removeItem(at: source)
            try? fileManager.removeItem(at: output)
            try? fileManager.removeItem(at: transformer)
        }
        try fileManager.createDirectory(at: source, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: transformer, withIntermediateDirectories: true)
        try fileManager.createSymbolicLink(
            at: source.appendingPathComponent("transformer", isDirectory: true),
            withDestinationURL: transformer
        )
        try Data("{\"num_experts\":2}".utf8)
            .write(to: transformer.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(to: source.appendingPathComponent("model_index.json"))
        try MereRunModelManifest.template(for: .lingBotVideoMoE30BA3B).write(to: source)

        let shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
        let expertKey = "blocks.0.ffn.experts.w1"
        let regularKey = "blocks.0.attn.to_q.weight"
        try MLX.save(arrays: [
            expertKey: MLXArray((0..<256).map { Float($0) / 256 }, [2, 4, 32]),
            regularKey: MLXArray((0..<16).map(Float.init), [4, 4]),
        ], url: transformer.appendingPathComponent(shard))
        let index = """
        {"metadata":{"total_size":320},"weight_map":{
          "\(expertKey)":"\(shard)",
          "\(regularKey)":"\(shard)"
        }}
        """
        try Data(index.utf8).write(
            to: transformer.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
        )

        let first = try LingBotVideoMoEQuantizer.quantize(options: .init(
            sourceRoot: source,
            outputRoot: output,
            groupSize: 32,
            includeRefiner: false
        ))
        XCTAssertEqual(first.quantizedShardCount, 1)
        XCTAssertEqual(first.reusedShardCount, 0)

        let metadata = try SafetensorsStreamingLoader.metadata(
            url: output.appendingPathComponent("transformer/\(shard)")
        )
        XCTAssertNotNil(metadata[expertKey + ".weight"])
        XCTAssertNotNil(metadata[expertKey + ".scales"])
        XCTAssertNotNil(metadata[expertKey + ".biases"])
        XCTAssertNotNil(metadata[regularKey])
        XCTAssertNil(metadata[expertKey])

        let manifest = try MereRunModelManifest.loadRequired(from: output)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.quantization?.scheme, "mlx-affine-routed-experts")
        XCTAssertEqual(manifest.components?.tokenizer, .local(path: "processor"))
        let sidecar = try JSONDecoder().decode(
            LingBotVideoQuantizationConfig.self,
            from: Data(contentsOf: output.appendingPathComponent(LingBotVideoQuantizationConfig.filename))
        )
        XCTAssertFalse(sidecar.includesRefiner)

        let resumed = try LingBotVideoMoEQuantizer.quantize(options: .init(
            sourceRoot: source,
            outputRoot: output,
            groupSize: 32,
            includeRefiner: false
        ))
        XCTAssertEqual(resumed.quantizedShardCount, 0)
        XCTAssertEqual(resumed.reusedShardCount, 1)

        XCTAssertThrowsError(try LingBotVideoMoEQuantizer.quantize(options: .init(
            sourceRoot: source,
            outputRoot: output,
            groupSize: 64,
            includeRefiner: false
        ))) { error in
            guard case LingBotVideoMoEQuantizer.QuantizerError.incompatibleOutput(output) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertFalse(output.path.isEmpty)
        }
    }

    func testMoEQuantizerRejectsNestedOutputBeforeWriting() {
        let source = URL(fileURLWithPath: "/tmp/lingbot-source", isDirectory: true)
        let output = source.appendingPathComponent("converted", isDirectory: true)

        XCTAssertThrowsError(try LingBotVideoMoEQuantizer.quantize(options: .init(
            sourceRoot: source,
            outputRoot: output,
            includeRefiner: false
        ))) { error in
            guard case LingBotVideoMoEQuantizer.QuantizerError.sourceAndOutputOverlap = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testRealMoEExpertQuantizationErrorWhenRootsProvided() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let sourcePath = environment["MERERUN_LINGBOT_MOE_SOURCE_ROOT"],
              let quantizedPath = environment["MERERUN_LINGBOT_MOE_QUANTIZED_ROOT"] else {
            throw XCTSkip("Set LingBot MoE source and quantized roots to run real expert parity.")
        }
        let sourceRoot = URL(fileURLWithPath: sourcePath)
        let quantizedRoot = URL(fileURLWithPath: quantizedPath)
        let component = "transformer"
        let expertKey = "blocks.0.ffn.experts.w1"
        let indexName = "diffusion_pytorch_model.safetensors.index.json"
        let sourceIndex = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: sourceRoot.appendingPathComponent("\(component)/\(indexName)"))
        )
        let quantizedIndex = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: quantizedRoot.appendingPathComponent("\(component)/\(indexName)"))
        )
        let sourceShard = try XCTUnwrap(sourceIndex.weightMap[expertKey])
        let quantizedShard = try XCTUnwrap(quantizedIndex.weightMap[expertKey + ".weight"])
        let dense = try XCTUnwrap(SafetensorsStreamingLoader.loadArrays(
            url: sourceRoot.appendingPathComponent("\(component)/\(sourceShard)"),
            where: { $0 == expertKey }
        )[expertKey])
        let quantized = try SafetensorsStreamingLoader.loadArrays(
            url: quantizedRoot.appendingPathComponent("\(component)/\(quantizedShard)"),
            where: { $0.hasPrefix(expertKey + ".") }
        )
        let weight = try XCTUnwrap(quantized[expertKey + ".weight"])
        let scales = try XCTUnwrap(quantized[expertKey + ".scales"])
        let biases = try XCTUnwrap(quantized[expertKey + ".biases"])
        let restored = MLX.dequantized(
            weight,
            scales: scales,
            biases: biases,
            groupSize: 64,
            bits: 4,
            mode: .affine,
            dtype: .float32
        )
        let reference = dense.asType(.float32)
        let errorRMS = MLX.sqrt((restored - reference).square().mean()).item(Float.self)
        let referenceRMS = MLX.sqrt(reference.square().mean()).item(Float.self)
        let relativeError = errorRMS / max(referenceRMS, 1e-12)

        XCTAssertLessThan(relativeError, 0.15, "relative expert weight error: \(relativeError)")

        let layer = Q35SwitchLinear(
            inputDims: 2048,
            outputDims: 768,
            numExperts: 128,
            groupSize: 64,
            bits: 4,
            quantized: true,
            bias: false
        )
        try layer.update(parameters: ModuleParameters.unflattened([
            ("weight", weight),
            ("scales", scales),
            ("biases", biases),
        ]), verify: .none)
        let input = MLXRandom.normal([1, 4, 2048]).asType(.bfloat16)
        let indices = MLX.zeros([1, 4, 1], dtype: .int32)
        let actual = layer(input, indices: indices).squeezed(axis: -2).asType(.float32)
        let expected = MLX.matmul(input.asType(.float32), reference[0].T)
        let outputError = MLX.sqrt((actual - expected).square().mean()).item(Float.self)
        let expectedRMS = MLX.sqrt(expected.square().mean()).item(Float.self)
        let relativeOutputError = outputError / max(expectedRMS, 1e-12)
        XCTAssertLessThan(
            relativeOutputError,
            0.2,
            "relative GatherQMM expert output error: \(relativeOutputError)"
        )
    }

    private func makeModelRoot(numExperts: Int) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("LingBotVideoTests.\(UUID().uuidString)", isDirectory: true)
        for directory in ["processor", "text_encoder", "transformer", "vae"] {
            try FileManager.default.createDirectory(
                at: root.appendingPathComponent(directory, isDirectory: true),
                withIntermediateDirectories: true
            )
        }
        try Data("{}".utf8).write(to: root.appendingPathComponent("model_index.json"))
        try Data("{}".utf8).write(to: root.appendingPathComponent("processor/tokenizer.json"))
        try Data("{}".utf8).write(to: root.appendingPathComponent("processor/tokenizer_config.json"))
        try Data(Self.textConfigJSON.utf8).write(to: root.appendingPathComponent("text_encoder/config.json"))
        try Data("{\"metadata\":{},\"weight_map\":{}}".utf8)
            .write(to: root.appendingPathComponent("text_encoder/model.safetensors.index.json"))
        let transformerConfig = Self.transformerConfigJSON.replacingOccurrences(
            of: "\"num_experts\": 0",
            with: "\"num_experts\": \(numExperts)"
        )
        try Data(transformerConfig.utf8).write(to: root.appendingPathComponent("transformer/config.json"))
        try Data().write(to: root.appendingPathComponent("transformer/diffusion_pytorch_model.safetensors"))
        try Data(Self.vaeConfigJSON.utf8).write(to: root.appendingPathComponent("vae/config.json"))
        try Data().write(to: root.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"))
        return root
    }

    private static let transformerConfigJSON = """
    {
      "axes_dims": [32, 48, 48],
      "axes_lens": [8192, 1024, 1024],
      "depth": 24,
      "freq_dim": 256,
      "hidden_size": 2048,
      "in_channels": 16,
      "intermediate_size": 6144,
      "norm_eps": 0.000001,
      "num_attention_heads": 16,
      "num_experts": 0,
      "out_bias": true,
      "out_channels": 16,
      "patch_embed_bias": true,
      "patch_size": [1, 2, 2],
      "qkv_bias": false,
      "rope_theta": 256.0,
      "text_dim": 2560,
      "timestep_mlp_bias": true
    }
    """

    private static let textConfigJSON = """
    {
      "text_config": {
        "head_dim": 128,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 0.000001,
        "rope_scaling": {"mrope_interleaved": true, "mrope_section": [24, 20, 20]},
        "rope_theta": 5000000,
        "vocab_size": 151936
      }
    }
    """

    private static let vaeConfigJSON = """
    {
      "base_dim": 96,
      "dim_mult": [1, 2, 4, 4],
      "latents_mean": [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921],
      "latents_std": [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916],
      "num_res_blocks": 2,
      "temperal_downsample": [false, true, true],
      "z_dim": 16
    }
    """

    private func fnv1a64(_ values: [Int]) -> UInt64 {
        var hash: UInt64 = 0xCBF2_9CE4_8422_2325
        for value in values {
            let bits = UInt32(value)
            for shift in stride(from: 0, through: 24, by: 8) {
                hash ^= UInt64((bits >> UInt32(shift)) & 0xFF)
                hash &*= 0x0000_0100_0000_01B3
            }
        }
        return hash
    }
}
