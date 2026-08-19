import MLX
import MLXNN
import MediaIO
@testable import MereRunCore
import XCTest

final class MuseGlimmerTests: MereRunCoreTestCase {
    func testConfigDecodesLayerContractsAndQuantization() throws {
        let config = try Self.config()
        XCTAssertEqual(config.modelType, "muse_glimmer")
        XCTAssertEqual(config.textConfig.layerTypes, [
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        ])
        XCTAssertEqual(config.textConfig.layerRopeTheta, [500_000, 500_000, 500_000, 0])
        XCTAssertEqual(config.visionConfig.patchTemporal, 2)
        XCTAssertEqual(config.quantization?.bits, 4)
        XCTAssertEqual(config.quantization?.groupSize, 64)
    }

    func testConfigRejectsPerLayerArrayMismatch() throws {
        let invalid = Self.configJSON.replacingOccurrences(
            of: #""layer_rope_theta":[500000,500000,500000,0]"#,
            with: #""layer_rope_theta":[500000,0]"#
        )
        XCTAssertThrowsError(
            try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(invalid.utf8))
        )
    }

    func testImageTargetSizeIsMergedPatchAlignedAndBounded() {
        let landscape = MuseGlimmerImageProcessor.targetSize(
            originalWidth: 4_000,
            originalHeight: 1_000,
            patchSize: 14,
            mergeSize: 2,
            maxImageTokens: 4_096
        )
        XCTAssertEqual(landscape.width % 28, 0)
        XCTAssertEqual(landscape.height % 28, 0)
        XCTAssertLessThanOrEqual(landscape.width * landscape.height, 4_096 * 28 * 28)
        XCTAssertGreaterThan(landscape.width, landscape.height)

        let tiny = MuseGlimmerImageProcessor.targetSize(
            originalWidth: 3,
            originalHeight: 7,
            patchSize: 14,
            mergeSize: 2,
            maxImageTokens: 4_096
        )
        XCTAssertGreaterThanOrEqual(tiny.width, 28)
        XCTAssertGreaterThanOrEqual(tiny.height, 28)
    }

    func testImageTargetSizeMatchesUpstreamAspectRatioCandidateSelection() {
        let landscape = MuseGlimmerImageProcessor.targetSize(
            originalWidth: 1_920,
            originalHeight: 1_080,
            patchSize: 14,
            mergeSize: 2,
            maxImageTokens: 4_096
        )

        XCTAssertEqual(landscape.width, 1_932)
        XCTAssertEqual(landscape.height, 1_092)

        let nearSquare = MuseGlimmerImageProcessor.targetSize(
            originalWidth: 101,
            originalHeight: 103,
            patchSize: 14,
            mergeSize: 2,
            maxImageTokens: 4_096
        )
        XCTAssertEqual(nearSquare.width, 112)
        XCTAssertEqual(nearSquare.height, 112)
    }

    func testLanczosResizeMatchesPinnedPillowUint8Fixture() throws {
        var rgba: [UInt8] = []
        for y in 0..<7 {
            for x in 0..<5 {
                rgba.append(UInt8((x * 31 + y * 17) % 256))
                rgba.append(UInt8((x * 7 + y * 43 + 11) % 256))
                rgba.append(UInt8((x * 59 + y * 13 + 23) % 256))
                rgba.append(UInt8((x * 19 + y * 29 + 101) % 256))
            }
        }
        let source = try MediaImage(width: 5, height: 7, rgba8: rgba)
        let resized = try MuseGlimmerImageProcessor.resizedLanczos(source, width: 8, height: 4)

        // Exported from torchvision 0.28's uint8 LANCZOS/antialias path and
        // independently matched by Pillow 12.3.0 after RGB conversion.
        XCTAssertEqual(resized.rgba8, [
            6, 27, 24, 255, 15, 29, 46, 255, 39, 35, 91, 255, 60, 39, 122, 255,
            76, 43, 183, 255, 97, 47, 211, 255, 121, 53, 86, 255, 132, 55, 0, 255,
            33, 96, 46, 255, 44, 98, 68, 255, 68, 104, 113, 255, 89, 108, 143, 255,
            105, 112, 210, 255, 126, 116, 246, 255, 150, 122, 114, 255, 161, 124, 7, 255,
            65, 197, 70, 255, 76, 199, 91, 255, 100, 205, 138, 255, 121, 209, 174, 255,
            137, 213, 215, 255, 158, 217, 214, 255, 182, 223, 113, 255, 193, 225, 37, 255,
            94, 111, 92, 255, 105, 113, 109, 255, 129, 119, 163, 255, 150, 123, 229, 255,
            166, 127, 159, 255, 187, 131, 16, 255, 211, 137, 31, 255, 222, 139, 89, 255,
        ])
    }

    func testVisionBilinearGeometryMatchesPinnedFloat32Fixture() {
        let geometry = MuseGlimmerVisionPatchEmbedder.bilinearPositionGeometry(
            grids: [(temporal: 1, height: 3, width: 5)],
            side: 4
        )
        XCTAssertEqual(geometry.indices, [
            [0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8, 9, 10, 11],
            [0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 11],
            [4, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15],
            [4, 5, 6, 7, 7, 8, 9, 10, 11, 11, 12, 13, 14, 15, 15],
        ])
        XCTAssertEqual(geometry.weights.map { $0.map(\.bitPattern) }, [
            [
                0x00000000, 0x3e7ffffd, 0x3ed55555, 0x3f155556, 0x3f3ffffe,
                0x00000000, 0x3e199998, 0x3e800000, 0x3eb33334, 0x3ee66664,
                0x00000000, 0x3d4cccbe, 0x3daaaaa0, 0x3deeeee1, 0x3e19998e,
            ],
            [
                0x3f3fffff, 0x3f155556, 0x3ed55555, 0x3e7ffffd, 0x00000000,
                0x3ee66666, 0x3eb33334, 0x3e800000, 0x3e199998, 0x00000000,
                0x3e199990, 0x3deeeee1, 0x3daaaaa0, 0x3d4cccbe, 0x00000000,
            ],
            [
                0x00000000, 0x3d4ccccc, 0x3daaaaac, 0x3deeeef2, 0x3e199999,
                0x00000000, 0x3e199998, 0x3e800000, 0x3eb33334, 0x3ee66664,
                0x00000000, 0x3e800000, 0x3ed55558, 0x3f155558, 0x3f400000,
            ],
            [
                0x3e19999b, 0x3deeeef2, 0x3daaaaac, 0x3d4ccccc, 0x00000000,
                0x3ee66666, 0x3eb33334, 0x3e800000, 0x3e199998, 0x00000000,
                0x3f400002, 0x3f155558, 0x3ed55558, 0x3e800000, 0x00000000,
            ],
        ])

        let batched = MuseGlimmerVisionPatchEmbedder.bilinearPositionGeometry(
            grids: [
                (temporal: 2, height: 3, width: 5),
                (temporal: 1, height: 2, width: 4),
            ],
            side: 4
        )
        XCTAssertEqual(batched.indices.map(\.count), [38, 38, 38, 38])
        XCTAssertEqual(batched.weights.map(\.count), [38, 38, 38, 38])
    }

    func testTextArchitectureDetailsAreLoadBearing() throws {
        MLXRandom.seed(7_103)
        let baselineModel = MuseGlimmerModel(config: try Self.config())
        baselineModel.update(parameters: baselineModel.parameters().mapValues {
            $0 + MLXArray(Float(0.013))
        })
        let tokens = MLXArray((0..<16).map { Int32(($0 * 7 + 3) % 31) }).reshaped(1, 16)
        let baseline = baselineModel.forward(tokens, cache: baselineModel.makeCache()).logits
        MLX.eval(baseline)

        func maximumDelta(configJSON: String) throws -> Float {
            let config = try JSONDecoder().decode(
                MuseGlimmerConfig.self,
                from: Data(configJSON.utf8)
            )
            let candidate = MuseGlimmerModel(config: config)
            try candidate.update(parameters: baselineModel.parameters(), verify: .all)
            let output = candidate.forward(tokens, cache: candidate.makeCache()).logits
            let delta = MLX.max(MLX.abs(output.asType(.float32) - baseline.asType(.float32)))
            MLX.eval(delta)
            return delta.item(Float.self)
        }

        let brokenQueryScale = Self.configJSON.replacingOccurrences(
            of: #""qk_scale_factor":3.87"#,
            with: #""qk_scale_factor":1.0"#
        )
        XCTAssertGreaterThan(try maximumDelta(configJSON: brokenQueryScale), 1e-5)

        let brokenNoPE = Self.configJSON.replacingOccurrences(
            of: #""layer_rope_theta":[500000,500000,500000,0]"#,
            with: #""layer_rope_theta":[500000,500000,500000,500000]"#
        )
        XCTAssertGreaterThan(try maximumDelta(configJSON: brokenNoPE), 1e-5)

        let brokenPostEpsilon = Self.configJSON.replacingOccurrences(
            of: #""post_norm_eps":0.00000001"#,
            with: #""post_norm_eps":0.00001"#
        )
        XCTAssertGreaterThan(try maximumDelta(configJSON: brokenPostEpsilon), 0)
    }

    func testPublishedMLXWeightKeysAndImplicitAffineModeAreAccepted() throws {
        let published = Self.configJSON.replacingOccurrences(
            of: #""quantization":{"group_size":64,"bits":4,"mode":"affine"}"#,
            with: #""quantization":{"group_size":64,"bits":4}"#
        )
        XCTAssertEqual(
            try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(published.utf8))
                .quantization?.mode,
            "affine"
        )
        XCTAssertEqual(
            MuseGlimmerWeightKeys.normalized("language_model.model.layers.3.mlp.up_proj.weight"),
            "model.language_model.layers.3.mlp.up_proj.weight"
        )
        XCTAssertEqual(
            MuseGlimmerWeightKeys.normalized("language_model.lm_head.scales"),
            "lm_head.scales"
        )
        XCTAssertEqual(
            MuseGlimmerWeightKeys.normalized("vision_tower.layers.2.attn.q_proj.weight"),
            "model.vision_tower.layers.2.attn.q_proj.weight"
        )
        XCTAssertEqual(
            MuseGlimmerWeightKeys.normalized("model.vision_adapter.fc1.weight"),
            "model.vision_adapter.fc1.weight"
        )
    }

    func testImagePlaceholderExpansionPreservesBoundaries() throws {
        let expanded = try MuseGlimmerImageProcessor.expandedPromptTokens(
            [1, 9, 2, 9, 3],
            tokenCounts: [2, 3],
            imageTokenId: 9,
            imageStartTokenId: 10,
            imageEndTokenId: 11
        )
        XCTAssertEqual(expanded, [1, 10, 9, 9, 11, 2, 10, 9, 9, 9, 11, 3])
    }

    func testATEMToolParserPreservesScalarAndJSONValues() {
        let raw = """
        <atem:function_calls>
        <atem:invoke name="files.write">
        <atem:parameter name="path">notes/out.txt</atem:parameter>
        <atem:parameter name="payload">{"ok":true,"items":[1,2]}</atem:parameter>
        </atem:invoke>
        </atem:function_calls>
        """
        let calls = MuseGlimmerToolParser.parseToolCalls(raw)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "files.write")
        XCTAssertEqual(calls[0].arguments["path"], "notes/out.txt")
        XCTAssertEqual(calls[0].arguments["payload"], #"{"ok":true,"items":[1,2]}"#)
        XCTAssertEqual(MuseGlimmerToolParser.visibleText(raw), "")
    }

    func testTinyTextModelUsesFourCenteredNormsAndSerialCache() throws {
        let model = MuseGlimmerModel(config: try Self.config())
        let names = Set(model.parameters().flattened().map(\.0))
        XCTAssertTrue(names.contains("model.language_model.layers.0.input_layernorm.weight"))
        XCTAssertTrue(names.contains("model.language_model.layers.0.post_attention_layernorm.weight"))
        XCTAssertTrue(names.contains("model.language_model.layers.0.pre_feedforward_layernorm.weight"))
        XCTAssertTrue(names.contains("model.language_model.layers.0.post_feedforward_layernorm.weight"))
        XCTAssertTrue(names.contains("model.language_model.layers.0.self_attn.gate_proj.weight"))
        XCTAssertTrue(names.contains("model.vision_tower.patch_embedder.position_embedding_table.weight"))

        let logits = try model.forwardPrefill(
            inputIds: MLXArray([Int32(1), 2, 3]).reshaped(1, 3),
            imageBatch: nil,
            cache: model.makeCache()
        )
        MLX.eval(logits)
        XCTAssertEqual(logits.shape, [1, 1, 32])
    }

    func testManagedSpecPinsArtifactAndSourceProvenanceAndDisablesImplicitDownload() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: MuseGlimmerResources.modelId))
        XCTAssertEqual(spec.category, .visionChat)
        XCTAssertEqual(spec.validationKind, .museGlimmer)
        XCTAssertEqual(spec.upstreamRepoId, MuseGlimmerResources.artifactRepoId)
        XCTAssertEqual(spec.upstreamRevision, MuseGlimmerResources.artifactRevision)
        XCTAssertEqual(
            spec.usageRestriction?.terms.first?.sourceRepoId,
            MuseGlimmerResources.upstreamRepoId
        )
        XCTAssertEqual(
            spec.usageRestriction?.terms.first?.sourceRevision,
            MuseGlimmerResources.upstreamRevision
        )
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatMuseGlimmer)
        XCTAssertEqual(spec.companionModelIDs, [MuseGlimmerResources.dflash2ModelId])

        let dflash2 = try XCTUnwrap(
            ManagedModelCatalog.spec(for: MuseGlimmerResources.dflash2ModelId)
        )
        XCTAssertEqual(dflash2.upstreamRepoId, MuseGlimmerResources.dflash2UpstreamRepoId)
        XCTAssertEqual(dflash2.upstreamRevision, MuseGlimmerResources.dflash2UpstreamRevision)
        XCTAssertEqual(dflash2.validationKind, .museGlimmerAssistant)
        XCTAssertFalse(dflash2.runtimeAutoDownloadAllowed)

        let manifest = MereRunModelManifest.template(for: .museGlimmer30B)
        XCTAssertEqual(manifest.engine, .museGlimmer)
        XCTAssertEqual(manifest.family, .muse)
        XCTAssertEqual(manifest.supports, [.chat, .codeGeneration, .visionChat])
    }

    func testAssistantConfigAndWeightPathsMatchOfficialContract() throws {
        let config = try Self.assistantConfig()
        XCTAssertEqual(config.modelType, "muse_glimmer_assistant")
        XCTAssertEqual(config.blockSize, 4)
        XCTAssertEqual(config.maskTokenId, 31)
        XCTAssertEqual(config.targetLayerIds, [0, 2])

        let assistant = MuseGlimmerAssistantModel(config: config)
        let names = Set(assistant.parameters().flattened().map(\.0))
        XCTAssertTrue(names.contains("encoder.fc.weight"))
        XCTAssertTrue(names.contains("encoder.output_norm_enc.weight"))
        XCTAssertTrue(names.contains("layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(names.contains("layers.0.self_attn.k_norm.weight"))
        XCTAssertTrue(names.contains("layers.0.mlp.down_proj.weight"))
        XCTAssertTrue(names.contains("norm.weight"))
    }

    func testDFlash2ConfigAndWeightPathsMatchPublishedCheckpoint() throws {
        let config = try Self.dflash2Config()
        XCTAssertTrue(config.isDFlash2)
        XCTAssertEqual(config.modelType, "qwen3")
        XCTAssertEqual(config.blockSize, 4)
        XCTAssertEqual(config.maskTokenId, 31)
        XCTAssertEqual(config.targetLayerIds, [0, 2])
        XCTAssertEqual(config.dflash2?.convKernelSize, 2)
        XCTAssertEqual(config.dflash2?.convGroupSize, 2)
        XCTAssertEqual(config.dflash2?.selectorRank, 1)
        XCTAssertEqual(config.dflash2?.selectorTopK, 2)

        let assistant = MuseGlimmerAssistantModel(config: config)
        let names = Set(assistant.parameters().flattened().map(\.0))
        XCTAssertTrue(names.contains("layers.0.attention_conv.base_kernel"))
        XCTAssertTrue(names.contains("layers.0.attention_conv.kernel_projection.weight"))
        XCTAssertTrue(names.contains("layers.0.mlp_conv.base_kernel"))
        XCTAssertTrue(names.contains("candidate_selector.hidden_projection.weight"))
        XCTAssertTrue(names.contains("candidate_selector.predecessor_codebook.weight"))
        XCTAssertTrue(names.contains("candidate_selector.successor_codebook.weight"))

        XCTAssertEqual(MuseGlimmerDFlash2WeightKeys.normalized("fc.weight"), "encoder.fc.weight")
        XCTAssertEqual(
            MuseGlimmerDFlash2WeightKeys.normalized("hidden_norm.weight"),
            "encoder.output_norm_enc.weight"
        )
        XCTAssertEqual(
            MuseGlimmerDFlash2WeightKeys.normalized("candidate_selector.predecessor_codebook"),
            "candidate_selector.predecessor_codebook.weight"
        )
    }

    func testDFlash2GroupedDynamicConvolutionIsBlockLocalAndCausal() {
        let hidden = MLXArray([Float(1), 2, 3, 4, 5, 6], [1, 3, 2])
        let dynamic = MLXArray.zeros([1, 3, 2, 1])
        let base = MLXArray([Float(1), 1, 10, 10], [2, 2])
        let output = museGlimmerGroupedDynamicConvolution(
            hidden: hidden,
            dynamic: dynamic,
            base: base,
            groupSize: 2
        )
        MLX.eval(output)

        XCTAssertEqual(output.asArray(Float.self), [1, 2, 13, 24, 35, 46])
    }

    func testDFlash2SelectorUsesPredecessorEdgesAcrossPositions() throws {
        let selector = MuseGlimmerDFlash2CandidateSelector(
            vocabularySize: 4,
            hiddenSize: 4,
            rank: 1,
            topK: 2
        )
        try selector.update(
            parameters: ModuleParameters.unflattened([
                ("hidden_projection.weight", MLXArray([Float(1), 0, 0, 0], [1, 4])),
                ("predecessor_codebook.weight", MLXArray([Float(1), 2, 3, 4], [4, 1])),
                ("successor_codebook.weight", MLXArray([Float(-2), -1, 1, 2], [4, 1])),
            ]),
            verify: .all
        )
        let hidden = MLXArray(
            [Float(1), 0, 0, 0, 1, 0, 0, 0],
            [1, 2, 4]
        )
        let logits = MLXArray(
            [Float(0), 0, 5, 4, 5, 4, 0, 0],
            [1, 2, 4]
        )
        let selection = selector.select(
            hidden: hidden,
            logits: logits,
            anchorTokenIds: MLXArray([Int32(1)]),
            temperature: 0
        )
        MLX.eval(selection.tokens)

        XCTAssertEqual(selection.tokens.asArray(Int32.self), [3, 1])
        XCTAssertNil(selection.candidateProbabilities)
    }

    func testDFlash2SparseRejectionDistributionSubtractsOnlyCandidates() {
        let residual = MuseGlimmerDFlashDecoder.sparseRejectionDistribution(
            target: MLXArray([Float(0), 1]),
            candidateTokenIds: MLXArray([Int32(0)]),
            draft: MLXArray([Float(1)])
        )
        MLX.eval(residual)

        XCTAssertEqual(residual.asArray(Float.self), [0, 1])
    }

    func testAssistantDraftBlockUsesTargetContextAndTargetEmbeddings() throws {
        let target = MuseGlimmerModel(config: try Self.config())
        let assistant = MuseGlimmerAssistantModel(config: try Self.assistantConfig())
        let targetCache = target.makeCache()
        let prefill = try target.forwardPrefillDetailed(
            inputIds: MLXArray([Int32(1), 2, 3]).reshaped(1, 3),
            imageBatch: nil,
            cache: targetCache,
            captureLayerIndices: [0, 2]
        )
        let assistantCache = assistant.makeCache()
        assistant.appendTargetContext(prefill.capturedHiddenStates, cache: assistantCache)
        let logits = assistant.draftLogits(
            anchorTokens: MLXArray([Int32(4)]).reshaped(1, 1),
            speculativeTokenCount: 3,
            cache: assistantCache.map { $0.fork() },
            target: target
        )
        MLX.eval(logits)

        XCTAssertEqual(assistantCache.map(\.offset), [3, 3])
        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testGreedyDFlashDecodeMatchesSerialTargetTokens() throws {
        let target = MuseGlimmerModel(config: try Self.config())
        let assistant = MuseGlimmerAssistantModel(config: try Self.assistantConfig())
        let prompt = [1, 4, 7]
        let serialCache = target.makeCache()
        let serialInitial = try target.forwardPrefill(
            inputIds: MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count),
            imageBatch: nil,
            cache: serialCache
        )
        let speculativeCache = target.makeCache()
        let speculativeInitial = try target.forwardPrefillDetailed(
            inputIds: MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count),
            imageBatch: nil,
            cache: speculativeCache,
            captureLayerIndices: [0, 2]
        )
        let assistantCache = assistant.makeCache()
        assistant.appendTargetContext(
            speculativeInitial.capturedHiddenStates,
            cache: assistantCache
        )
        MLX.eval(serialInitial, speculativeInitial.logits)
        let generation = GenerationConfig(
            maxTokens: 8,
            temperature: 0,
            topK: 0,
            topP: 1,
            minP: 0,
            repetitionPenalty: 1,
            repetitionContextSize: 8
        )
        let serial = try AutoregressiveDecodeEngine.decode(
            AutoregressiveDecodeRequest(
                initialLogits: serialInitial,
                generationConfig: generation,
                eosTokens: [],
                tokenBudget: 8,
                historySeedTokens: prompt
            ),
            stepForward: { token in target(token, cache: serialCache) }
        )
        let speculative = try MuseGlimmerDFlashDecoder.decode(
            initialLogits: speculativeInitial.logits,
            target: target,
            targetCache: speculativeCache,
            assistant: assistant,
            assistantCache: assistantCache,
            generationConfig: generation,
            eosTokens: [],
            tokenBudget: 8,
            historySeedTokens: prompt,
            speculativeTokens: 3,
            assistantModelPath: nil
        )

        XCTAssertEqual(speculative.generatedTokens, serial.generatedTokens)
        XCTAssertGreaterThan(speculative.stats.targetVerificationForwards, 0)
    }

    func testGreedyDFlash2DecodeMatchesSerialTargetTokens() throws {
        let target = MuseGlimmerModel(config: try Self.config())
        let assistant = MuseGlimmerAssistantModel(config: try Self.dflash2Config())
        let prompt = [1, 4, 7]
        let serialCache = target.makeCache()
        let serialInitial = try target.forwardPrefill(
            inputIds: MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count),
            imageBatch: nil,
            cache: serialCache
        )
        let speculativeCache = target.makeCache()
        let speculativeInitial = try target.forwardPrefillDetailed(
            inputIds: MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count),
            imageBatch: nil,
            cache: speculativeCache,
            captureLayerIndices: [0, 2]
        )
        let assistantCache = assistant.makeCache()
        assistant.appendTargetContext(
            speculativeInitial.capturedHiddenStates,
            cache: assistantCache
        )
        let generation = GenerationConfig(
            maxTokens: 8,
            temperature: 0,
            topK: 0,
            topP: 1,
            minP: 0,
            repetitionPenalty: 1,
            repetitionContextSize: 8
        )
        let serial = try AutoregressiveDecodeEngine.decode(
            AutoregressiveDecodeRequest(
                initialLogits: serialInitial,
                generationConfig: generation,
                eosTokens: [],
                tokenBudget: 8,
                historySeedTokens: prompt
            ),
            stepForward: { token in target(token, cache: serialCache) }
        )
        let speculative = try MuseGlimmerDFlashDecoder.decode(
            initialLogits: speculativeInitial.logits,
            target: target,
            targetCache: speculativeCache,
            assistant: assistant,
            assistantCache: assistantCache,
            generationConfig: generation,
            eosTokens: [],
            tokenBudget: 8,
            historySeedTokens: prompt,
            speculativeTokens: 3,
            assistantModelPath: nil
        )

        XCTAssertEqual(speculative.generatedTokens, serial.generatedTokens)
        XCTAssertGreaterThan(speculative.stats.targetVerificationForwards, 0)
    }

    func testSampledDFlash2DecodeUsesSparseProposalPath() throws {
        MLXRandom.seed(7_103)
        let target = MuseGlimmerModel(config: try Self.config())
        let assistant = MuseGlimmerAssistantModel(config: try Self.dflash2Config())
        let prompt = [1, 4, 7]
        let targetCache = target.makeCache()
        let initial = try target.forwardPrefillDetailed(
            inputIds: MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count),
            imageBatch: nil,
            cache: targetCache,
            captureLayerIndices: [0, 2]
        )
        let assistantCache = assistant.makeCache()
        assistant.appendTargetContext(initial.capturedHiddenStates, cache: assistantCache)
        let result = try MuseGlimmerDFlashDecoder.decode(
            initialLogits: initial.logits,
            target: target,
            targetCache: targetCache,
            assistant: assistant,
            assistantCache: assistantCache,
            generationConfig: GenerationConfig(
                maxTokens: 8,
                temperature: 1,
                topK: 0,
                topP: 1,
                minP: 0,
                repetitionPenalty: nil,
                repetitionContextSize: 8
            ),
            eosTokens: [],
            tokenBudget: 8,
            historySeedTokens: prompt,
            speculativeTokens: 3,
            assistantModelPath: nil
        )

        XCTAssertEqual(result.generatedTokens.count, 8)
        XCTAssertGreaterThan(result.stats.targetVerificationForwards, 0)
    }

    func testWarmUpMaterializesSerialAndProductionDFlashPaths() throws {
        MLXRandom.seed(7_103)
        let target = MuseGlimmerModel(config: try Self.config())

        XCTAssertNil(try MuseGlimmerGenerator.warmUp(model: target, assistant: nil))

        let assistant = MuseGlimmerAssistantModel(config: try Self.assistantConfig())
        let stats = try XCTUnwrap(MuseGlimmerGenerator.warmUp(
            model: target,
            assistant: assistant,
            speculativeTokens: 3
        ))
        XCTAssertEqual(stats.speculativeTokens, 3)
        XCTAssertGreaterThan(stats.rounds, 0)
        XCTAssertGreaterThan(stats.draftedTokens, 0)
        XCTAssertGreaterThan(stats.targetVerificationForwards, 0)
    }

    func testDFlashPolicyUsesMeasuredMLXProposalWidthAndClampsOverrides() {
        XCTAssertEqual(MuseGlimmerDFlashPolicy.defaultSpeculativeTokens, 3)
        XCTAssertEqual(
            MuseGlimmerDFlashPolicy.speculativeTokens(
                maximum: 15,
                environment: ["MERERUN_MUSE_GLIMMER_DFLASH_TOKENS": "9"]
            ),
            9
        )
        XCTAssertEqual(
            MuseGlimmerDFlashPolicy.speculativeTokens(
                maximum: 15,
                environment: ["MERERUN_MUSE_GLIMMER_DFLASH_TOKENS": "99"]
            ),
            15
        )
        XCTAssertFalse(
            MuseGlimmerDFlashPolicy.enabled(
                environment: ["MERERUN_MUSE_GLIMMER_DFLASH": "off"]
            )
        )
    }

    private static func config() throws -> MuseGlimmerConfig {
        try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(configJSON.utf8))
    }

    private static func assistantConfig() throws -> MuseGlimmerAssistantConfig {
        try JSONDecoder().decode(
            MuseGlimmerAssistantConfig.self,
            from: Data(assistantConfigJSON.utf8)
        )
    }

    private static func dflash2Config() throws -> MuseGlimmerAssistantConfig {
        try JSONDecoder().decode(
            MuseGlimmerAssistantConfig.self,
            from: Data(dflash2ConfigJSON.utf8)
        )
    }

    private static let assistantConfigJSON = #"""
    {
      "model_type":"muse_glimmer_assistant",
      "architectures":["MuseGlimmerAssistantModel"],
      "hidden_size":16,
      "intermediate_size":32,
      "num_hidden_layers":2,
      "num_attention_heads":4,
      "num_key_value_heads":2,
      "head_dim":4,
      "rms_norm_eps":0.00001,
      "rope_parameters":{"rope_theta":500000,"rope_type":"default"},
      "max_position_embeddings":64,
      "sliding_window":8,
      "layer_types":["sliding_attention","sliding_attention"],
      "attention_dropout":0,
      "hidden_act":"silu",
      "bos_token_id":1,
      "eos_token_id":2,
      "pad_token_id":0,
      "block_size":4,
      "mask_token_id":31,
      "target_layer_ids":[0,2]
    }
    """#

    private static let dflash2ConfigJSON = #"""
    {
      "model_type":"qwen3",
      "architectures":["DFlash2DraftModel"],
      "is_causal":false,
      "hidden_size":16,
      "intermediate_size":32,
      "num_hidden_layers":2,
      "num_attention_heads":4,
      "num_key_value_heads":2,
      "head_dim":4,
      "rms_norm_eps":0.00001,
      "rope_parameters":{"rope_theta":500000,"rope_type":"default"},
      "max_position_embeddings":64,
      "sliding_window":8,
      "layer_types":["sliding_attention","sliding_attention"],
      "attention_dropout":0,
      "hidden_act":"silu",
      "vocab_size":32,
      "bos_token_id":1,
      "eos_token_id":2,
      "pad_token_id":0,
      "dflash_config":{
        "block_size":4,
        "conv_group_size":2,
        "conv_kernel_size":2,
        "final_logit_softcapping":20,
        "mask_token_id":31,
        "output_multiplier":0.19611613513818404,
        "selector_rank":1,
        "selector_top_k":2,
        "target_layer_ids":[0,2]
      }
    }
    """#

    private static let configJSON = #"""
    {
      "model_type":"muse_glimmer",
      "architectures":["MuseGlimmerForConditionalGeneration"],
      "image_token_id":9,
      "video_token_id":8,
      "out_hidden_size":32,
      "projector_hidden_size":16,
      "projector_hidden_act":"gelu",
      "quantization":{"group_size":64,"bits":4,"mode":"affine"},
      "text_config":{
        "model_type":"muse_glimmer_text",
        "hidden_size":16,
        "intermediate_size":32,
        "num_hidden_layers":4,
        "num_attention_heads":4,
        "num_key_value_heads":2,
        "head_dim":4,
        "max_position_embeddings":64,
        "sliding_window":8,
        "layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention"],
        "layer_rope_theta":[500000,500000,500000,0],
        "rope_parameters":{"rope_theta":500000,"rope_type":"default"},
        "rms_norm_eps":0.00001,
        "post_norm_eps":0.00000001,
        "qk_scale_factor":3.87,
        "output_multiplier":0.19611613513818404,
        "final_logit_softcapping":20,
        "hidden_activation":"silu",
        "attention_bias":false,
        "attention_dropout":0,
        "vocab_size":32,
        "tie_word_embeddings":false,
        "bos_token_id":1,
        "eos_token_id":2,
        "pad_token_id":0
      },
      "vision_config":{
        "model_type":"muse_glimmer_vision",
        "hidden_size":8,
        "intermediate_size":16,
        "num_hidden_layers":2,
        "num_attention_heads":2,
        "patch_size":2,
        "patch_temporal":2,
        "merge_size":2,
        "pos_emb_height":4,
        "pos_emb_width":4,
        "max_position_embeddings":16,
        "layer_norm_eps":0.00001,
        "hidden_act":"gelu",
        "layer_types":["window_attention","full_attention"],
        "rope_parameters":{"rope_theta":10000,"rope_type":"default"}
      }
    }
    """#
}
