import Foundation
import MediaIO
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class Cosmos3EdgeTests: MereRunCoreTestCase {
    private struct CheckpointInventory: Decodable {
        struct Tensor: Decodable {
            let shape: [Int]
            let dtype: String
        }

        let nvidiaFrameworkRevision: String
        let cosmos3EdgeRevision: String
        let transformer: [String: Tensor]
        let vae: [String: Tensor]

        enum CodingKeys: String, CodingKey {
            case nvidiaFrameworkRevision = "nvidia_framework_revision"
            case cosmos3EdgeRevision = "cosmos3_edge_revision"
            case transformer
            case vae
        }
    }

    func testDecodesPublishedEdgeConfiguration() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let transformer = root.appendingPathComponent("transformer")
        let vae = root.appendingPathComponent("vae")
        try FileManager.default.createDirectory(at: transformer, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: vae, withIntermediateDirectories: true)
        try TestFileSystem.writeFile(
            transformer.appendingPathComponent("config.json"),
            contents: Data(Self.transformerConfigJSON.utf8)
        )
        try TestFileSystem.writeFile(
            vae.appendingPathComponent("config.json"),
            contents: Data(Self.vaeConfigJSON.utf8)
        )

        let resources = Cosmos3Resources(rootURL: root)
        let transformerConfig = try resources.loadTransformerConfiguration()
        let vaeConfig = try resources.loadVAEConfiguration()

        XCTAssertEqual(transformerConfig.hiddenSize, 2_048)
        XCTAssertEqual(transformerConfig.intermediateSize, 9_216)
        XCTAssertEqual(transformerConfig.layerCount, 28)
        XCTAssertEqual(transformerConfig.ropeAxesDimensions, [24, 20, 20])
        XCTAssertEqual(transformerConfig.actionDimension, 64)
        XCTAssertEqual(vaeConfig.latentDimension, 48)
        XCTAssertEqual(vaeConfig.spatialScaleFactor, 16)
        XCTAssertEqual(vaeConfig.temporalScaleFactor, 4)
    }

    func testDecodesPublishedReasonerConfiguration() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        try TestFileSystem.writeFile(
            root.appendingPathComponent("config.json"),
            contents: Data(Self.reasonerConfigJSON.utf8)
        )

        let configuration = try Cosmos3Resources(rootURL: root)
            .loadReasonerConfiguration()

        XCTAssertEqual(configuration.vision.hiddenSize, 1_152)
        XCTAssertEqual(configuration.vision.intermediateSize, 4_304)
        XCTAssertEqual(configuration.vision.layerCount, 27)
        XCTAssertEqual(configuration.vision.patchCount, 256)
        XCTAssertEqual(configuration.projector.intermediateSize, 11_520)
        XCTAssertEqual(configuration.projector.outputHiddenSize, 2_048)
        XCTAssertEqual(configuration.imageTokenID, 19)
        XCTAssertEqual(configuration.videoTokenID, 18)
    }

    func testTextAndVAEPositionIDsMatchUpstreamLayout() {
        let text = Cosmos3SequenceLayout.textPositionIDs(tokenCount: 3)
        XCTAssertEqual(text.temporal, [0, 1, 2])
        XCTAssertEqual(text.height, [0, 1, 2])
        XCTAssertEqual(text.nextTemporalOffset, 3)

        let vision = Cosmos3SequenceLayout.vaePositionIDs(
            frames: 2,
            height: 2,
            width: 2,
            temporalOffset: 15_003,
            fps: 24
        )
        XCTAssertEqual(vision.temporal, [
            15_003, 15_003, 15_003, 15_003,
            15_004, 15_004, 15_004, 15_004,
        ])
        XCTAssertEqual(vision.height, [0, 0, 1, 1, 0, 0, 1, 1])
        XCTAssertEqual(vision.width, [0, 1, 0, 1, 0, 1, 0, 1])
        XCTAssertEqual(vision.nextTemporalOffset, 15_005)
    }

    func testFPSModulationUsesBaseTokenRate() {
        let ids = Cosmos3SequenceLayout.vaePositionIDs(
            frames: 3,
            height: 1,
            width: 1,
            temporalOffset: 10,
            fps: 12,
            baseFPS: 24,
            temporalCompressionFactor: 4
        )
        XCTAssertEqual(ids.temporal[0], 10, accuracy: 1e-6)
        XCTAssertEqual(ids.temporal[1], 12, accuracy: 1e-6)
        XCTAssertEqual(ids.temporal[2], 14, accuracy: 1e-6)
    }

    func testVisionPatchRoundTripIncludesOddSpatialPadding() {
        let values = (0..<(2 * 2 * 3 * 5)).map(Float.init)
        let latent = MLXArray(values).reshaped(2, 2, 3, 5)
        let packed = Cosmos3VisionPatches.pack(latent, patchSize: 2)
        let restored = Cosmos3VisionPatches.unpack(
            packed.tokens,
            layout: packed.layout,
            channels: 2
        )
        eval(restored)

        XCTAssertEqual(packed.tokens.shape, [12, 8])
        XCTAssertEqual(packed.layout.paddedHeight, 4)
        XCTAssertEqual(packed.layout.paddedWidth, 6)
        XCTAssertEqual(restored.shape, latent.shape)
        XCTAssertEqual(restored.asArray(Float.self), values)
    }

    func testActionDomainsAndChunkPaddingMatchPublishedContract() throws {
        XCTAssertEqual(Cosmos3ActionDomain.cameraPose.domainID, 2)
        XCTAssertEqual(Cosmos3ActionDomain.cameraPose.rawActionDimension, 9)
        XCTAssertEqual(Cosmos3ActionDomain.handPose.rawActionDimension, 57)
        let source = URL(fileURLWithPath: "/tmp/frame.png")
        let first = (0..<9).map(Float.init)
        let condition = try Cosmos3ActionCondition(
            mode: .forwardDynamics,
            chunkSize: 3,
            domain: .cameraPose,
            rawActions: [first],
            imageURL: source
        )
        let actions = try XCTUnwrap(condition.modelActions())
        XCTAssertEqual(actions.count, 3)
        XCTAssertEqual(actions[0].count, 64)
        XCTAssertEqual(Array(actions[2].prefix(9)), first)
        XCTAssertTrue(actions[2].dropFirst(9).allSatisfy { $0 == 0 })
    }

    func testDomainAwareLinearUsesPublishedOutputInputWeightLayout() throws {
        let layer = Cosmos3DomainAwareLinear(inputSize: 2, outputSize: 3, domainCount: 2)
        try layer.update(
            parameters: ModuleParameters.unflattened([
                (
                    "fc.weight",
                    MLXArray([
                        1, 2, 3, 4, 5, 6,
                        7, 8, 9, 10, 11, 12,
                    ] as [Float]).reshaped(2, 6)
                ),
                (
                    "bias.weight",
                    MLXArray([0, 0, 0, 1, 2, 3] as [Float]).reshaped(2, 3)
                ),
            ]),
            verify: .all
        )
        let output = layer(
            MLXArray([2, 3, 4, 5] as [Float]).reshaped(2, 2),
            domainIDs: MLXArray([Int32(0), 1])
        )
        eval(output)
        XCTAssertEqual(output.asArray(Float.self), [14, 19, 24, 79, 89, 99])
    }

    func testRotaryEmbeddingAndSmallBackboneAreFinite() {
        let config = Cosmos3TransformerConfiguration(
            actionDimension: 4,
            headDimension: 8,
            hiddenSize: 16,
            intermediateSize: 32,
            latentChannels: 2,
            latentPatchSize: 2,
            attentionHeadCount: 2,
            embodimentDomainCount: 3,
            layerCount: 1,
            keyValueHeadCount: 1,
            patchLatentDimension: 8,
            ropeAxesDimensions: [2, 1, 1],
            vocabularySize: 32
        )
        let model = Cosmos3OmniTransformerModel(configuration: config)
        let text = model.embedText(tokenIDs: MLXArray([Int32(1), 2, 3]))
        let vision = model.projectVision(
            MLX.zeros([2, 8]),
            timesteps: MLXArray([Float(1_000), 1_000])
        )
        let positionIDs = Cosmos3PositionIDs(
            temporal: [0, 1, 2, 15_003, 15_003],
            height: [0, 1, 2, 0, 0],
            width: [0, 1, 2, 0, 1],
            nextTemporalOffset: 15_004
        ).mlxArray
        let output = model(understanding: text, generation: vision, positionIDs: positionIDs)
        eval(output.understanding, output.generation)

        XCTAssertEqual(output.understanding.shape, [3, 16])
        XCTAssertEqual(output.generation.shape, [2, 16])
        XCTAssertTrue(output.understanding.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertTrue(output.generation.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testJointVisionActionPredictionPreservesConditionedPositions() throws {
        let config = Cosmos3TransformerConfiguration(
            actionDimension: 4,
            headDimension: 8,
            hiddenSize: 16,
            intermediateSize: 32,
            latentChannels: 2,
            latentPatchSize: 2,
            attentionHeadCount: 2,
            embodimentDomainCount: 32,
            layerCount: 1,
            keyValueHeadCount: 1,
            patchLatentDimension: 8,
            ropeAxesDimensions: [2, 1, 1],
            vocabularySize: 32
        )
        let model = Cosmos3OmniTransformerModel(configuration: config)
        let prediction = model.predict(try Cosmos3DenoisingInput(
            tokenIDs: MLXArray([Int32(1), 2]),
            visionLatents: MLX.zeros([2, 2, 2, 2]),
            conditionedVisionFrames: [0],
            timestep: 700,
            fps: 24,
            actionLatents: MLX.zeros([3, 4]),
            conditionedActionFrames: [0, 1],
            actionDomain: .cameraPose,
            rawActionDimension: 2
        ))
        let action = try XCTUnwrap(prediction.actionVelocity)
        eval(prediction.visionVelocity, action)

        XCTAssertEqual(prediction.visionVelocity.shape, [2, 2, 2, 2])
        XCTAssertEqual(action.shape, [3, 4])
        XCTAssertTrue(
            prediction.visionVelocity[0..., 0, 0..., 0...]
                .asArray(Float.self)
                .allSatisfy { $0 == 0 }
        )
        XCTAssertTrue(action[0..<2].asArray(Float.self).allSatisfy { $0 == 0 })
        XCTAssertTrue(action[0..., 2...].asArray(Float.self).allSatisfy { $0 == 0 })
        XCTAssertTrue(action.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testImageEditPacksCleanSourceAndNoisyTargetAsSeparateVisionItems() throws {
        let config = Cosmos3TransformerConfiguration(
            actionDimension: 4,
            headDimension: 8,
            hiddenSize: 16,
            intermediateSize: 32,
            latentChannels: 2,
            latentPatchSize: 2,
            attentionHeadCount: 2,
            embodimentDomainCount: 32,
            layerCount: 1,
            keyValueHeadCount: 1,
            patchLatentDimension: 8,
            ropeAxesDimensions: [2, 1, 1],
            vocabularySize: 32
        )
        let model = Cosmos3OmniTransformerModel(configuration: config)
        let source = MLX.ones([2, 1, 2, 2])
        let target = MLX.zeros([2, 1, 2, 2])
        let predictions = model.predictVisionItems(
            tokenIDs: MLXArray([Int32(1), 2]),
            items: [
                try Cosmos3VisionDenoisingItem(
                    latents: source,
                    conditionedFrames: [0]
                ),
                try Cosmos3VisionDenoisingItem(latents: target),
            ],
            timestep: 700
        )
        eval(predictions)

        XCTAssertEqual(predictions.count, 2)
        XCTAssertEqual(predictions[0].shape, source.shape)
        XCTAssertTrue(predictions[0].asArray(Float.self).allSatisfy { $0 == 0 })
        XCTAssertEqual(predictions[1].shape, target.shape)
        XCTAssertTrue(predictions[1].asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testReasonerProcessorSizeAndMRoPEMatchPinnedUpstreamRules() throws {
        XCTAssertEqual(
            Cosmos3ReasonerProcessor.targetSize(
                width: 640,
                height: 480,
                minimumPixels: 65_536,
                maximumPixels: 16_777_216
            ).width,
            640
        )
        XCTAssertEqual(
            Cosmos3ReasonerProcessor.targetSize(
                width: 640,
                height: 480,
                minimumPixels: 65_536,
                maximumPixels: 16_777_216
            ).height,
            480
        )
        let configuration = try JSONDecoder().decode(
            Cosmos3ReasonerConfiguration.self,
            from: Data(Self.reasonerConfigJSON.utf8)
        )
        let ids = Cosmos3ReasonerProcessor.positionIDs(
            tokenIDs: [10, 20, 19, 19, 19, 19, 21],
            imageGrids: [Cosmos3ReasonerVisionGrid(time: 1, height: 4, width: 4)],
            videoGrids: [],
            configuration: configuration
        )
        eval(ids)
        XCTAssertEqual(ids.shape, [3, 7])
        XCTAssertEqual(
            ids.asArray(Float.self),
            [
                0, 1, 2, 2, 2, 2, 4,
                0, 1, 2, 2, 3, 3, 4,
                0, 1, 2, 3, 2, 3, 4,
            ]
        )
    }

    func testNVIDIAAndPublishedUniPCSchedulesMatchPinnedUpstreams() {
        let nvidia = Cosmos3UniPCScheduler(
            steps: 4,
            shift: 3,
            schedule: .nvidiaShiftedFlow
        )
        XCTAssertEqual(nvidia.timesteps, [999, 899, 749, 499])
        XCTAssertEqual(nvidia.sigmas.count, 5)
        for (actual, expected) in zip(
            nvidia.sigmas,
            [0.99966645, 0.8996398, 0.7496248, 0.49966657, 0] as [Float]
        ) {
            XCTAssertEqual(actual, expected, accuracy: 2e-7)
        }

        let published = Cosmos3UniPCScheduler(
            steps: 4,
            schedule: .publishedKarrasFlow
        )
        XCTAssertEqual(published.timesteps, [995, 973, 798, 128])
        for (actual, expected) in zip(
            published.sigmas,
            [0.99502486, 0.9736331, 0.7985969, 0.12816042, 0] as [Float]
        ) {
            XCTAssertEqual(actual, expected, accuracy: 2e-7)
        }
    }

    func testPromptMetadataAndActionJSONMatchPublishedTemplates() throws {
        let text = try Cosmos3Tokenizer.renderPrompts(
            prompt: "A quiet station.",
            negativePrompt: "",
            numFrames: 17,
            height: 256,
            width: 320,
            fps: 24,
            action: nil,
            addResolutionTemplate: true,
            addDurationTemplate: true
        )
        XCTAssertEqual(
            text.conditional,
            "A quiet station. The video is 0.7 seconds long and is of 24 FPS. "
                + "This video is of 256x320 resolution."
        )
        XCTAssertEqual(
            text.unconditional,
            "The video is not 0.7 seconds long and is not of 24 FPS. "
                + "This video is not of 256x320 resolution."
        )

        let action = try Cosmos3ActionCondition(
            mode: .forwardDynamics,
            chunkSize: 16,
            domain: .cameraPose,
            resolutionTier: .compact,
            rawActions: [Array(repeating: 0, count: 9)],
            imageURL: URL(fileURLWithPath: "/tmp/frame.png")
        )
        let actionPrompt = try Cosmos3Tokenizer.renderPrompts(
            prompt: "Move forward",
            negativePrompt: "blur",
            numFrames: 17,
            height: 256,
            width: 320,
            fps: 24,
            action: action,
            addResolutionTemplate: true,
            addDurationTemplate: true
        )
        XCTAssertEqual(
            actionPrompt.conditional,
            "{\"cinematography\": {\"framing\": \"This video is captured from a "
                + "first-person perspective looking at the scene.\"}, \"actions\": "
                + "[{\"time\": \"0:00-0:01\", \"description\": \"Move forward.\"}], "
                + "\"duration\": \"0s\", \"fps\": 24.0, "
                + "\"resolution\": {\"H\": 256, \"W\": 320}, \"aspect_ratio\": \"4,3\"}"
        )
        XCTAssertEqual(actionPrompt.unconditional, "blur")
    }

    func testPublishedTransformerInventoryHasExactTensorCount() {
        let keys = Cosmos3CheckpointInventory.transformerKeys(
            configuration: Cosmos3TransformerConfiguration()
        )
        XCTAssertEqual(keys.count, 549)
        XCTAssertTrue(keys.contains("layers.0.self_attn.k_norm_und_for_gen.weight"))
        XCTAssertTrue(keys.contains("layers.27.mlp_moe_gen.up_proj.weight"))
        XCTAssertFalse(keys.contains("layers.0.mlp.gate_proj.weight"))
        XCTAssertFalse(keys.contains("layers.0.self_attn.norm_q.weight"))
    }

    func testDiffusersWanVAEKeysMapToNativeWanLayout() {
        XCTAssertEqual(
            Cosmos3ModelLoader.mapVAEKey("quant_conv.weight"),
            "conv1.weight"
        )
        XCTAssertEqual(
            Cosmos3ModelLoader.mapVAEKey("encoder.down_blocks.1.resnets.0.conv_shortcut.weight"),
            "encoder.downsamples.1.downsamples.0.shortcut.weight"
        )
        XCTAssertEqual(
            Cosmos3ModelLoader.mapVAEKey("encoder.down_blocks.2.downsampler.time_conv.weight"),
            "encoder.downsamples.2.downsamples.2.time_conv.weight"
        )
        XCTAssertEqual(
            Cosmos3ModelLoader.mapVAEKey("decoder.up_blocks.0.upsampler.resample.1.weight"),
            "decoder.upsamples.0.upsamples.3.resample_weight"
        )
        XCTAssertEqual(
            Cosmos3ModelLoader.mapVAEKey("decoder.mid_block.attentions.0.to_qkv.weight"),
            "decoder.middle.1.to_qkv_weight"
        )
        let mappedGamma = Cosmos3ModelLoader.vaeWeightMapper(
            key: "encoder.down_blocks.0.resnets.0.norm1.gamma",
            value: MLX.zeros([160, 1, 1, 1])
        )
        XCTAssertEqual(mappedGamma.count, 1)
        XCTAssertEqual(
            mappedGamma[0].0,
            "encoder.downsamples.0.downsamples.0.residual.layer_0.gamma"
        )
        XCTAssertEqual(mappedGamma[0].1.shape, [160])
    }

    func testPinnedCheckpointInventoriesMatchNativeContractsExactly() throws {
        let inventoryURL = try XCTUnwrap(
            Bundle.module.url(
                forResource: "checkpoint-inventory",
                withExtension: "json",
                subdirectory: "Fixtures/Cosmos3Edge"
            )
        )
        let inventory = try JSONDecoder().decode(
            CheckpointInventory.self,
            from: Data(contentsOf: inventoryURL)
        )
        XCTAssertEqual(
            inventory.nvidiaFrameworkRevision,
            "ed8287fd7477113f8ac4f6b84290514d55cf0cdc"
        )
        XCTAssertEqual(
            inventory.cosmos3EdgeRevision,
            Cosmos3Resources.officialRevision
        )
        XCTAssertEqual(
            Set(inventory.transformer.keys),
            Cosmos3CheckpointInventory.transformerKeys(
                configuration: Cosmos3TransformerConfiguration()
            )
        )
        XCTAssertTrue(inventory.transformer.values.allSatisfy { $0.dtype == "BF16" })
        XCTAssertEqual(inventory.vae.count, 196)
        XCTAssertTrue(inventory.vae.values.allSatisfy { $0.dtype == "BF16" })

        let mappedVAEKeys = try inventory.vae.keys.map {
            try XCTUnwrap(Cosmos3ModelLoader.mapVAEKey($0), "Unmapped VAE key: \($0)")
        }
        XCTAssertEqual(Set(mappedVAEKeys).count, inventory.vae.count)
        let tinyVAE = Wan2VAEModel(configuration: Wan2VAEConfiguration(
            latentChannels: 2,
            encoderDimensions: 4,
            decoderDimensions: 4,
            imagePatchSize: 2,
            blockResampleShortcut: true,
            decoderResampleReducesChannels: false,
            latentMean: [0, 0],
            latentStandardDeviation: [1, 1]
        ))
        let nativeCheckpointKeys = Set(tinyVAE.parameters().flattened().map(\.0))
            .subtracting(["latentMean", "latentStandardDeviation"])
        XCTAssertEqual(Set(mappedVAEKeys), nativeCheckpointKeys)
    }

    func testTinyMoTLayerMatchesPinnedNVIDIAUpstreamNumerically() throws {
        let fixtureURL = try XCTUnwrap(
            Bundle.module.url(
                forResource: "tiny-mot-layer",
                withExtension: "safetensors",
                subdirectory: "Fixtures/Cosmos3Edge"
            )
        )
        let fixture = try MLX.loadArrays(url: fixtureURL)
        let config = Cosmos3TransformerConfiguration(
            actionDimension: 4,
            headDimension: 8,
            hiddenSize: 16,
            intermediateSize: 32,
            latentChannels: 2,
            latentPatchSize: 2,
            attentionHeadCount: 2,
            embodimentDomainCount: 3,
            layerCount: 1,
            keyValueHeadCount: 1,
            patchLatentDimension: 8,
            ropeAxesDimensions: [2, 1, 1],
            vocabularySize: 32
        )
        let model = Cosmos3OmniTransformerModel(configuration: config)
        let layerWeights = fixture
            .filter { $0.key.hasPrefix("layers.") }
            .map { ($0.key, $0.value) }
        try model.update(
            parameters: ModuleParameters.unflattened(layerWeights),
            verify: .none
        )
        let loadedParameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for (key, expected) in layerWeights {
            let actual = try XCTUnwrap(loadedParameters[key], "Missing native parameter: \(key)")
            XCTAssertLessThan(maxAbsoluteError(actual, expected), 1e-7, key)
        }
        let understanding = try XCTUnwrap(fixture["input.understanding"])
        let generation = try XCTUnwrap(fixture["input.generation"])
        let positionIDs = try XCTUnwrap(fixture["input.position_ids"])
        let nativeLayer = model.layers[0]
        let normalizedUnderstanding = nativeLayer.understandingInputNorm(understanding)
        let normalizedGeneration = nativeLayer.generationInputNorm(generation)
        let rotary = model.rotaryEmbedding(positionIDs: positionIDs)
        let understandingLength = understanding.dim(0)
        let attended = nativeLayer.attention(
            understanding: normalizedUnderstanding,
            generation: normalizedGeneration,
            rotary: (
                understandingCosine: rotary.cosine[0..<understandingLength],
                understandingSine: rotary.sine[0..<understandingLength],
                generationCosine: rotary.cosine[understandingLength...],
                generationSine: rotary.sine[understandingLength...]
            )
        )
        let residualUnderstanding = understanding + attended.understanding
        let residualGeneration = generation + attended.generation
        let postNormUnderstanding = nativeLayer.understandingPostAttentionNorm(
            residualUnderstanding
        )
        let postNormGeneration = nativeLayer.generationPostAttentionNorm(
            residualGeneration
        )
        let mlpUnderstanding = nativeLayer.understandingMLP(postNormUnderstanding)
        let mlpGeneration = nativeLayer.generationMLP(postNormGeneration)
        let output = nativeLayer(
            understanding: understanding,
            generation: generation,
            rotary: (
                understandingCosine: rotary.cosine[0..<understandingLength],
                understandingSine: rotary.sine[0..<understandingLength],
                generationCosine: rotary.cosine[understandingLength...],
                generationSine: rotary.sine[understandingLength...]
            )
        )
        let expectedUnderstanding = try XCTUnwrap(fixture["expected.understanding"])
        let expectedGeneration = try XCTUnwrap(fixture["expected.generation"])
        eval(
            output.understanding,
            output.generation,
            rotary.cosine,
            rotary.sine
        )

        let stages: [(String, MLXArray, String)] = [
            (
                "normalized understanding",
                normalizedUnderstanding,
                "expected.normalized_understanding"
            ),
            ("normalized generation", normalizedGeneration, "expected.normalized_generation"),
            ("attention understanding", attended.understanding, "expected.attention_understanding"),
            ("attention generation", attended.generation, "expected.attention_generation"),
            ("residual understanding", residualUnderstanding, "expected.residual_understanding"),
            ("residual generation", residualGeneration, "expected.residual_generation"),
            ("post-norm understanding", postNormUnderstanding, "expected.post_norm_understanding"),
            ("post-norm generation", postNormGeneration, "expected.post_norm_generation"),
            ("MLP understanding", mlpUnderstanding, "expected.mlp_understanding"),
            ("MLP generation", mlpGeneration, "expected.mlp_generation"),
        ]
        for (label, actual, key) in stages {
            XCTAssertLessThan(
                maxAbsoluteError(actual, try XCTUnwrap(fixture[key])),
                2e-5,
                label
            )
        }
        XCTAssertLessThan(
            maxAbsoluteError(output.understanding, expectedUnderstanding),
            2e-5
        )
        XCTAssertLessThan(
            maxAbsoluteError(output.generation, expectedGeneration),
            2e-5
        )
        XCTAssertLessThan(
            maxAbsoluteError(rotary.cosine, try XCTUnwrap(fixture["rotary.cosine"])),
            2e-6
        )
        XCTAssertLessThan(
            maxAbsoluteError(rotary.sine, try XCTUnwrap(fixture["rotary.sine"])),
            2e-6
        )
    }

    func testTinyReasonerVisionMatchesPinnedNVIDIAUpstreamNumerically() throws {
        let fixtureURL = try XCTUnwrap(
            Bundle.module.url(
                forResource: "tiny-reasoner-vision",
                withExtension: "safetensors",
                subdirectory: "Fixtures/Cosmos3Edge"
            )
        )
        let fixture = try MLX.loadArrays(url: fixtureURL)
        let configuration = try JSONDecoder().decode(
            Cosmos3ReasonerConfiguration.self,
            from: Data(Self.tinyReasonerConfigJSON.utf8)
        )
        let model = Cosmos3ReasonerVisionModel(configuration: configuration)
        let weights = fixture.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("visual.") || key.hasPrefix("projector.") else {
                return nil
            }
            return (key, value)
        }
        try model.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: .all
        )
        let output = model(
            patches: try XCTUnwrap(fixture["input.patches"]),
            grids: [Cosmos3ReasonerVisionGrid(time: 1, height: 4, width: 4)]
        )
        eval(output.embeddings)

        XCTAssertEqual(output.mergedGrids, [
            Cosmos3ReasonerVisionGrid(time: 1, height: 2, width: 2),
        ])
        XCTAssertLessThan(
            maxAbsoluteError(
                output.embeddings,
                try XCTUnwrap(fixture["expected.projected"])
            ),
            3e-5
        )
    }

    func testPublishedModeDefaultsResolvePerOperation() throws {
        let textImage = try Cosmos3GenerationOptions(
            prompt: "image",
            width: 256,
            height: 256,
            numFrames: 1
        )
        XCTAssertEqual(textImage.steps, 50)
        XCTAssertEqual(textImage.guidanceScale, 4)
        XCTAssertEqual(textImage.shift, 3)

        let textVideo = try Cosmos3GenerationOptions(
            prompt: "video",
            width: 320,
            height: 256,
            numFrames: 17
        )
        XCTAssertEqual(textVideo.steps, 35)
        XCTAssertEqual(textVideo.guidanceScale, 6)
        XCTAssertEqual(textVideo.shift, 10)

        let imageEdit = try Cosmos3GenerationOptions(
            prompt: "edit",
            imageURL: URL(fileURLWithPath: "/tmp/source.png"),
            width: 256,
            height: 256,
            numFrames: 1
        )
        XCTAssertEqual(imageEdit.mode, .imageToImage)
        XCTAssertEqual(imageEdit.steps, 35)
        XCTAssertEqual(imageEdit.guidanceScale, 6)
        XCTAssertEqual(imageEdit.shift, 5)

        let action = try Cosmos3ActionCondition(
            mode: .forwardDynamics,
            chunkSize: 1,
            domain: .cameraPose,
            rawActions: [Cosmos3CameraModelSpaceTrajectory.forwardReference[0]],
            imageURL: URL(fileURLWithPath: "/tmp/source.png")
        )
        let forwardDynamics = try Cosmos3GenerationOptions(
            prompt: "move",
            action: action,
            width: 256,
            height: 256,
            numFrames: 5
        )
        XCTAssertEqual(forwardDynamics.steps, 30)
        XCTAssertEqual(forwardDynamics.guidanceScale, 1)
        XCTAssertEqual(forwardDynamics.shift, 3)
        XCTAssertEqual(forwardDynamics.fps, 30)
    }

    func testWorldTransitionDefaultsMatchPublishedForwardDynamicsRecipe() {
        let request = Cosmos3WorldTransitionRequest(
            prompt: "move forward",
            camera: .forward()
        )
        XCTAssertEqual(request.width, 320)
        XCTAssertEqual(request.height, 176)
        XCTAssertEqual(request.numFrames, 17)
        XCTAssertEqual(request.steps, 30)
        XCTAssertEqual(request.guidanceScale, 1)
        XCTAssertEqual(request.shift, 3)
        XCTAssertEqual(request.seed, 0)
        XCTAssertEqual(request.fps, 30)
    }

    func testWorldFrameStitchingPreservesExactPriorStateAndGeneratedTail() throws {
        let source = try MediaImage(
            width: 2,
            height: 1,
            rgba8: [10, 20, 30, 255, 40, 50, 60, 255]
        )
        let generated = MLXArray([
            UInt8(1), 2, 3, 4, 5, 6,
            70, 80, 90, 100, 110, 120,
        ]).reshaped(1, 2, 1, 2, 3)
        let stitched = try Cosmos3WorldFrameStitching.replaceFirstFrame(
            in: generated,
            with: source
        )
        eval(stitched)

        XCTAssertEqual(stitched.shape, generated.shape)
        XCTAssertEqual(stitched[0, 0].asArray(UInt8.self), [10, 20, 30, 40, 50, 60])
        XCTAssertEqual(stitched[0, 1].asArray(UInt8.self), [70, 80, 90, 100, 110, 120])
    }

    func testPublishedCameraTrajectoryIsBundledAtExactShapeAndPinnedValues() {
        let reference = Cosmos3CameraModelSpaceTrajectory.forwardReference
        XCTAssertEqual(reference.count, 60)
        XCTAssertTrue(reference.allSatisfy { $0.count == 9 })
        XCTAssertEqual(
            Cosmos3CameraModelSpaceTrajectory.referenceSHA256,
            "547b977ccd6d435d579ba83e82e58618d0bebf2e8e7d8d577bf2e9c8883c5595"
        )
        XCTAssertEqual(reference[0][0], -1.0418891906738281, accuracy: 1e-7)
        XCTAssertEqual(reference[0][3], 0.9999865293502808, accuracy: 1e-7)
        XCTAssertEqual(reference[59][0], -0.6310582160949707, accuracy: 1e-7)
        XCTAssertEqual(reference[59][8], -0.00047919119242578745, accuracy: 1e-9)
    }

    func testSemanticTranslationControlsUseCameraRelativeDeltaAxes() {
        let forward = Cosmos3CameraActionCompiler.compile(
            control: .forward(meters: 2),
            actionCount: 4
        )
        let backward = Cosmos3CameraActionCompiler.compile(
            control: .backward(meters: 2),
            actionCount: 4
        )
        let left = Cosmos3CameraActionCompiler.compile(
            control: Wan2WorldCameraControl(
                motion: .strafeLeft,
                translationMeters: [-2, 0, 0]
            ),
            actionCount: 4
        )
        let right = Cosmos3CameraActionCompiler.compile(
            control: Wan2WorldCameraControl(
                motion: .strafeRight,
                translationMeters: [2, 0, 0]
            ),
            actionCount: 4
        )

        for index in forward.indices {
            let reference = Cosmos3CameraModelSpaceTrajectory.forwardReference[index]
            let magnitude = sqrt(
                reference[0] * reference[0]
                    + reference[1] * reference[1]
                    + reference[2] * reference[2]
            ) * 2
            XCTAssertEqual(Array(forward[index][0..<3]), [0, 0, magnitude])
            XCTAssertEqual(Array(backward[index][0..<3]), [0, 0, -magnitude])
            XCTAssertEqual(Array(left[index][0..<3]), [-magnitude, 0, 0])
            XCTAssertEqual(Array(right[index][0..<3]), [magnitude, 0, 0])
            XCTAssertEqual(
                Array(forward[index][3..<9]),
                Cosmos3CameraModelSpaceTrajectory.identityRotation6D
            )
        }
    }

    func testExplicitModelSpaceTrajectoryFitsChunkWithoutRenormalizing() {
        let first = Cosmos3CameraModelSpaceTrajectory.forwardReference[0]
        let second = Cosmos3CameraModelSpaceTrajectory.forwardReference[1]
        let fitted = Cosmos3CameraModelSpaceTrajectory.fitted(
            [first, second],
            actionCount: 4
        )
        XCTAssertEqual(fitted, [first, second, second, second])
    }

    func testCustomTranslationPreservesRequestedVectorMagnitude() {
        let actions = Cosmos3CameraActionCompiler.compile(
            control: Wan2WorldCameraControl(
                motion: .custom,
                translationMeters: [3, 0, 4]
            ),
            actionCount: 1
        )
        let reference = Cosmos3CameraModelSpaceTrajectory.forwardReference[0]
        let referenceMagnitude = sqrt(
            reference[0] * reference[0]
                + reference[1] * reference[1]
                + reference[2] * reference[2]
        )
        XCTAssertEqual(actions[0][0], referenceMagnitude * 3, accuracy: 1e-6)
        XCTAssertEqual(actions[0][1], 0, accuracy: 1e-6)
        XCTAssertEqual(actions[0][2], referenceMagnitude * 4, accuracy: 1e-6)
    }

    func testYawCompilerUsesStationaryConstantPerFramePoseDeltas() {
        let actions = Cosmos3CameraActionCompiler.compile(
            control: .yawRight(degrees: 15),
            actionCount: 60
        )
        let first = actions[0]
        let last = actions[59]
        XCTAssertEqual(Array(first.prefix(3)), [0, 0, 0])
        XCTAssertEqual(Array(last.prefix(3)), [0, 0, 0])
        XCTAssertEqual(first, last)
        XCTAssertEqual(last[3], cos(Float.pi / 720), accuracy: 1e-6)
        XCTAssertEqual(last[4], 0, accuracy: 1e-6)
        XCTAssertEqual(last[5], -sin(Float.pi / 720), accuracy: 1e-6)
        XCTAssertEqual(last[6], 0, accuracy: 1e-6)
        XCTAssertEqual(last[7], 1, accuracy: 1e-6)
        XCTAssertEqual(last[8], 0, accuracy: 1e-6)
    }

    func testSemanticContinuationDoesNotAccumulatePriorPoseDelta() {
        let startingAction = Cosmos3CameraModelSpaceTrajectory.forwardReference[59]
        let continued = Cosmos3CameraActionCompiler.compile(
            control: .yawRight(degrees: 15),
            actionCount: 60,
            startingAction: startingAction
        )
        let fresh = Cosmos3CameraActionCompiler.compile(
            control: .yawRight(degrees: 15),
            actionCount: 60
        )
        XCTAssertEqual(continued, fresh)
    }

    func testInteractiveYawUsesPinnedCUDARotationRateAtSixteenActions() {
        let startingAction = Cosmos3CameraModelSpaceTrajectory.forwardReference[59]
        let actions = Cosmos3CameraActionCompiler.compile(
            control: .yawRight(degrees: 15),
            actionCount: 16,
            startingAction: startingAction
        )

        XCTAssertEqual(actions.count, 16)
        XCTAssertEqual(actions[0][3], cos(Float.pi / 192), accuracy: 1e-7)
        XCTAssertEqual(actions[0][5], -sin(Float.pi / 192), accuracy: 1e-7)
        XCTAssertEqual(actions[15], actions[0])
    }

    func testAutoregressiveSeedSequenceMatchesPinnedCUDAChunkPolicy() {
        XCTAssertEqual(
            Cosmos3AutoregressiveSeedSequence.seed(baseSeed: 0, chunkIndex: 0),
            0
        )
        XCTAssertEqual(
            Cosmos3AutoregressiveSeedSequence.seed(baseSeed: 0, chunkIndex: 1),
            1
        )
        XCTAssertEqual(
            Cosmos3AutoregressiveSeedSequence.seed(baseSeed: 42, chunkIndex: 19),
            61
        )
    }

    func testWorldGraphCacheRecognizesExactSemanticInverseControls() {
        XCTAssertTrue(Cosmos3WorldGraphCache.areInverse(
            .forward(meters: 1),
            .backward(meters: 1)
        ))
        XCTAssertTrue(Cosmos3WorldGraphCache.areInverse(
            .yawRight(degrees: 15),
            .yawLeft(degrees: 15)
        ))
        XCTAssertFalse(Cosmos3WorldGraphCache.areInverse(
            .forward(meters: 1),
            .backward(meters: 2)
        ))
        XCTAssertFalse(Cosmos3WorldGraphCache.areInverse(
            .yawRight(degrees: 15),
            .yawRight(degrees: 15)
        ))
    }

    func testWorldGraphCacheReversesCompleteRGBFrames() {
        let frames: [UInt8] = [
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18,
        ]
        XCTAssertEqual(
            Cosmos3WorldGraphCache.reversedFrames(
                frames,
                frameCount: 3,
                width: 2,
                height: 1
            ),
            [
                13, 14, 15, 16, 17, 18,
                7, 8, 9, 10, 11, 12,
                1, 2, 3, 4, 5, 6,
            ]
        )
    }

    func testContinuedForwardAndBackwardTrajectoriesAreTranslationInverses() {
        let startingAction = Cosmos3CameraModelSpaceTrajectory.forwardReference[59]
        let forward = Cosmos3CameraActionCompiler.compile(
            control: .forward(),
            actionCount: 60,
            startingAction: startingAction
        )
        let backward = Cosmos3CameraActionCompiler.compile(
            control: .backward(),
            actionCount: 60,
            startingAction: forward[59]
        )

        for index in forward.indices {
            for axis in 0..<3 {
                XCTAssertEqual(backward[index][axis], -forward[index][axis], accuracy: 1e-6)
            }
            XCTAssertEqual(Array(backward[index][3..<9]), Array(forward[index][3..<9]))
        }
    }

    func testWorldRejectsMalformedExplicitModelSpaceTrajectoryBeforeLoadingModel() async throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let session = Cosmos3WorldSession(
            resources: Cosmos3Resources(rootURL: root),
            stateDirectory: root.appendingPathComponent("state")
        )
        let request = Cosmos3WorldTransitionRequest(
            prompt: "move",
            camera: .forward(),
            sourceImageURL: root.appendingPathComponent("source.png"),
            modelSpaceActions: [[0, 1]]
        )
        do {
            _ = try await session.transition(request)
            XCTFail("Expected malformed model-space trajectory to be rejected")
        } catch let error as Cosmos3WorldSessionError {
            XCTAssertEqual(
                error.localizedDescription,
                "Invalid Cosmos3 normalized model-space camera trajectory: expected 9 values per action; received 2"
            )
        }
    }

    private func maxAbsoluteError(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        MLX.max(MLX.abs(lhs.asType(.float32) - rhs.asType(.float32))).item(Float.self)
    }

    private static let transformerConfigJSON = """
    {
      "action_dim": 64,
      "action_gen": true,
      "attention_bias": false,
      "base_fps": 24,
      "enable_fps_modulation": true,
      "head_dim": 128,
      "hidden_size": 2048,
      "intermediate_size": 9216,
      "latent_channel": 48,
      "latent_patch_size": 2,
      "num_attention_heads": 16,
      "num_embodiment_domains": 32,
      "num_hidden_layers": 28,
      "num_key_value_heads": 8,
      "patch_latent_dim": 192,
      "qk_norm_for_text": false,
      "rms_norm_eps": 0.00001,
      "rope_axes_dim": [24, 20, 20],
      "rope_theta": 100000000,
      "sound_dim": null,
      "sound_gen": false,
      "temporal_compression_factor": 4,
      "timestep_scale": 0.001,
      "use_und_k_norm_for_gen": true,
      "unified_3d_mrope_reset_spatial_ids": true,
      "unified_3d_mrope_temporal_modality_margin": 15000,
      "vocab_size": 131072
    }
    """

    private static let vaeConfigJSON = """
    {
      "base_dim": 160,
      "decoder_base_dim": 256,
      "dim_mult": [1, 2, 4, 4],
      "in_channels": 12,
      "latents_mean": \(Array(repeating: 0, count: 48)),
      "latents_std": \(Array(repeating: 1, count: 48)),
      "out_channels": 12,
      "patch_size": 2,
      "scale_factor_spatial": 16,
      "scale_factor_temporal": 4,
      "z_dim": 48
    }
    """

    private static let reasonerConfigJSON = """
    {
      "image_token_id": 19,
      "video_token_id": 18,
      "vision_start_token_id": 20,
      "vision_end_token_id": 21,
      "vision_config": {
        "attention_dropout": 0,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "layer_norm_eps": 0.000001,
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 27,
        "num_patches": 256,
        "patch_size": 16,
        "spatial_merge_size": 2
      },
      "projector_config": {
        "input_hidden_size": 1152,
        "merger_intermediate_size": 11520,
        "out_hidden_size": 2048,
        "spatial_merge_size": 2,
        "use_postshuffle_norm": false
      }
    }
    """

    private static let tinyReasonerConfigJSON = """
    {
      "image_token_id": 19,
      "video_token_id": 18,
      "vision_start_token_id": 20,
      "vision_end_token_id": 21,
      "vision_config": {
        "attention_dropout": 0,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 8,
        "intermediate_size": 16,
        "layer_norm_eps": 0.000001,
        "num_attention_heads": 2,
        "num_channels": 3,
        "num_hidden_layers": 2,
        "num_patches": 4,
        "patch_size": 2,
        "spatial_merge_size": 2
      },
      "projector_config": {
        "input_hidden_size": 8,
        "merger_intermediate_size": 20,
        "out_hidden_size": 16,
        "spatial_merge_size": 2,
        "use_postshuffle_norm": false
      }
    }
    """
}
