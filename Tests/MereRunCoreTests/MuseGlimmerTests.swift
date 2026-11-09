import MLX
import MLXNN
import MereRunTensor
import MediaIO
@testable import MereRunCore
import XCTest

final class MuseGlimmerTests: MereRunCoreTestCase {
    func testPairedPrefixRestorePreservesSpeculativeOutputAndSnapshotIsolation() async throws {
        try await MuseGlimmerGenerator().checkPairedPrefix(config: Self.config(), assistantConfig: Self.dflash2Config())
    }

    func testCachedImageFeaturesMatchMultipleImagesAndRespectScopeAndIdentity() async throws {
        try await MuseGlimmerGenerator().checkImageFeatures(config: Self.config())
    }
    func testDensePrefillRetainsPackedDecodeAndCheckpointParameters() {
        for bits in [4, 8] {
            let weight = MLX.sin(MLXArray(0..<8192).asType(.float32) * 0.037).reshaped(128, 64).asType(.bfloat16)
            let quantized = QuantizedLinear(weight: weight, bias: nil, groupSize: 64, bits: bits)
            let packed = PortableQuantizedLinear(weight: quantized.weight, bias: nil, scales: quantized.scales, biases: quantized.biases, groupSize: 64, bits: bits)
            let adapted = MuseGlimmerPrefillLinear(packed)
            XCTAssertEqual(Set(adapted.parameters().flattened().map(\.0)), Set(packed.parameters().flattened().map(\.0)))
            XCTAssertEqual(adapted.weight.dtype, .uint32)
            for rows in [1, 4, 16] {
                let input = MLXArray.ones([1, rows, 64], dtype: .bfloat16)
                XCTAssertEqual(MLX.max(MLX.abs(adapted(input) - packed(input))).item(Float.self), 0)
            }
            let input = MLX.cos(MLXArray(0..<8192).asType(.float32) * 0.019).reshaped(1, 128, 64).asType(.bfloat16)
            let actual = adapted(input)
            let expected = packed(input)
            XCTAssertEqual(actual.dtype, .bfloat16)
            let relative = MLX.sqrt(MLX.mean(MLX.square(actual.asType(.float32) - expected.asType(.float32)))
                / MLX.mean(MLX.square(expected.asType(.float32)))).item(Float.self)
            XCTAssertLessThan(relative, 0.01)
        }
    }

    func testQuantizedPatchProjectionUsesScaleDTypeAndFloat32PositionInterpolation() throws {
        let config = try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(
            Self.configJSON.replacingOccurrences(of: #""patch_size":2"#, with: #""patch_size":4"#).utf8
        ))
        let embedder = MuseGlimmerVisionPatchEmbedder(config: config.visionConfig)
        embedder.update(parameters: embedder.parameters().mapValues { $0.asType(.bfloat16) })
        quantize(model: embedder, groupSize: 32, bits: 4, filter: { path, _ in path == "patch_embedding" })
        XCTAssertEqual(embedder.patchEmbedding.weight.dtype, .uint32)
        let patches = MLXArray(Array(0..<576).map { Float($0) / 576 }, [6, 96])
        let geometry = MuseGlimmerVisionPatchEmbedder.bilinearPositionGeometry(grids: [(1, 2, 3)], side: 4)
        var expectedPosition = MLXArray.zeros([6, 8])
        for corner in 0..<4 {
            expectedPosition = expectedPosition
                + embedder.positionEmbeddingTable(MLXArray(geometry.indices[corner])).asType(.float32)
                * MLXArray(geometry.weights[corner], [6, 1])
        }
        let expected = embedder.patchEmbedding(patches.asType(.bfloat16)) + expectedPosition.asType(.bfloat16)
        let result = embedder(patches, grids: [(1, 2, 3)])
        XCTAssertEqual(result.dtype, .bfloat16)
        XCTAssertEqual(MLX.max(MLX.abs(result - expected)).item(Float.self), 0)
    }

    func testVisionRotaryEvaluatesAnglesBeforeBF16Rounding() throws {
        let tower = MuseGlimmerVisionTower(config: try Self.config().visionConfig)
        let rotary = tower.makeRotary(positionPairs: [(257, 511)], dtype: .bfloat16)
        let angles: [Float] = [257, 511, 257, 511]
        let expectedCosine = MLXArray(angles.map { cos($0) }).asType(.bfloat16)
        let expectedSine = MLXArray(angles.map { sin($0) }).asType(.bfloat16)
        XCTAssertEqual(MLX.max(MLX.abs(rotary.cosine.reshaped(4) - expectedCosine)).item(Float.self), 0)
        XCTAssertEqual(MLX.max(MLX.abs(rotary.sine.reshaped(4) - expectedSine)).item(Float.self), 0)
    }

    func testFloat32ImageInputPreservesBF16LanguageEmbeddingsAndCache() throws {
        let model = MuseGlimmerModel(config: try Self.config())
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        let images = MuseGlimmerImageBatch(
            pixelValues: MLXArray(Array(0..<192).map { Float($0) / 192 }, [8, 24]),
            grids: [(1, 2, 2), (1, 2, 2)], tokenCounts: [1, 1]
        )
        let ids = MLXArray([Int32(1), 9, 3, 4, 9, 5]).reshaped(1, 6)
        let text = model.inputEmbeddings(for: ids)
        let fused = try model.inputEmbeddings(for: ids, imageBatch: images)
        XCTAssertEqual(text.dtype, .bfloat16)
        XCTAssertEqual(fused.dtype, text.dtype)
        for index in [0, 2, 3, 5] {
            XCTAssertEqual(MLX.max(MLX.abs(fused[0, index] - text[0, index])).item(Float.self), 0)
        }
        let cache = model.makeCache()
        let output = model.forwardPrefillDetailed(embeddings: fused, cache: cache)
        MLX.eval(output.logits)
        for layer in cache {
            let state = try XCTUnwrap(layer.currentState())
            XCTAssertEqual(state.0.dtype, .bfloat16)
            XCTAssertEqual(state.1.dtype, .bfloat16)
        }
    }

    func testTextPrefixCacheCanBeReusedAcrossFutureImageChanges() throws {
        let model = MuseGlimmerModel(config: try Self.config())
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        let ids = MLXArray([Int32(1), 3, 4, 5, 9, 6]).reshaped(1, 6)
        let text = model.inputEmbeddings(for: ids)
        let prefix = model.makeCache()
        MLX.eval(model.forwardPrefillDetailed(embeddings: text[0..., 0..<4, 0...], cache: prefix).logits)
        for scale: Float in [1, -2] {
            let image = MuseGlimmerImageBatch(
                pixelValues: MLXArray(Array(0..<96).map { scale * Float($0) / 96 }, [4, 24]),
                grids: [(1, 2, 2)], tokenCounts: [1]
            )
            let fused = try model.inputEmbeddings(for: ids, imageBatch: image)
            let fresh = model.makeCache()
            MLX.eval(model.forwardPrefillDetailed(embeddings: fused[0..., 0..<4, 0...], cache: fresh).logits)
            let reused = prefix.map { $0.fork() }
            let expected = model.forwardPrefillDetailed(embeddings: fused[0..., 4..<6, 0...], cache: fresh).logits
            let actual = model.forwardPrefillDetailed(embeddings: fused[0..., 4..<6, 0...], cache: reused).logits
            XCTAssertEqual(MLX.max(MLX.abs(actual - expected)).item(Float.self), 0)
        }
    }

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

    func testChunkedImageEmbeddingsPreserveLogitsAndCapturedLayersAcrossSlidingWindow() throws {
        let model = MuseGlimmerModel(config: try Self.config())
        let ids = MLXArray([Int32(1), 3, 9, 4, 5, 6, 7, 10, 11, 12, 13, 9, 14, 15, 16, 17, 18, 19, 20, 21])
            .reshaped(1, 20)
        let images = MuseGlimmerImageBatch(
            pixelValues: MLXArray(Array(0..<192).map { Float($0) / 192 }, [8, 24]),
            grids: [(1, 2, 2), (1, 2, 2)], tokenCounts: [1, 1]
        )
        let layers: Set<Int> = [0, 2]
        let reference = try model.forwardPrefillDetailed(
            inputIds: ids, imageBatch: images, cache: model.makeCache(), captureLayerIndices: layers
        )
        MLX.eval(reference.logits)
        let embeddings = try model.inputEmbeddings(for: ids, imageBatch: images)
        MLX.eval(embeddings)
        for chunkSize in [1, 4, 7] {
            let cache = model.makeCache()
            var captured: [Int: [MLXArray]] = [:]
            var finalLogits: MLXArray?
            for offset in stride(from: 0, to: 20, by: chunkSize) {
                let end = min(20, offset + chunkSize)
                let output = model.forwardPrefillDetailed(
                    embeddings: embeddings[0..., offset..<end, 0...],
                    cache: cache, captureLayerIndices: layers
                )
                MLX.eval(output.logits)
                finalLogits = output.logits
                for layer in layers {
                    captured[layer, default: []].append(try XCTUnwrap(output.capturedHiddenStates[layer]))
                }
            }
            let difference = MLX.max(MLX.abs(try XCTUnwrap(finalLogits) - reference.logits)).item(Float.self)
            XCTAssertLessThan(difference, 0.0001, "chunk size \(chunkSize)")
            XCTAssertEqual(cache.map(\.offset), [20, 20, 20, 20])
            for layer in layers {
                let states = MLX.concatenated(try XCTUnwrap(captured[layer]), axis: 1)
                let expected = try XCTUnwrap(reference.capturedHiddenStates[layer])
                XCTAssertLessThan(MLX.max(MLX.abs(states - expected)).item(Float.self), 0.0001)
            }
        }
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

    func testFusedNormalizationRetainsCheckpointParametersAndFP32Range() throws {
        for dimension in [128, 4096, 6656] {
            let x = (MLX.sin(MLXArray(0..<(4 * dimension)).asType(.float32) * 0.017) * 200_000)
                .asType(.bfloat16).reshaped(1, 4, dimension)
            let weight = (MLX.cos(MLXArray(0..<dimension).asType(.float32) * 0.013) * 0.25).asType(.bfloat16)
            for centered in [false, true] {
                let fused = FusedRMSNorm(weight: weight, eps: 1e-6, centered: centered)
                let promoted = x.asType(.float32)
                let scale = centered ? 1 + weight.asType(.float32) : weight.asType(.float32)
                let reference = (promoted * MLX.rsqrt((promoted * promoted).mean(axis: -1, keepDims: true) + 1e-6) * scale)
                    .asType(.bfloat16)
                let actual = fused(x)
                MLX.eval(actual, reference)
                XCTAssertEqual(actual.dtype, .bfloat16)
                XCTAssertTrue(MLX.all(MLX.isFinite(actual)).item(Bool.self))
                XCTAssertLessThanOrEqual(MLX.max(MLX.abs(actual.asType(.float32) - reference.asType(.float32))).item(Float.self), 0.015625)
            }
        }
        let target = MuseGlimmerModel(config: try Self.config())
        let before = target.parameters().flattened().map(\.0)
        target.configureFusedNormalization(true)
        XCTAssertEqual(target.parameters().flattened().map(\.0), before)
        target.configureFusedNormalization(false)
        XCTAssertEqual(target.parameters().flattened().map(\.0), before)
    }

    func testCostCalibrationUsesForkedStateAndRespectsOutputBudget() throws {
        MLXRandom.seed(7_103)
        let target = MuseGlimmerModel(config: try Self.config())
        let assistant = MuseGlimmerAssistantModel(config: try Self.assistantConfig())
        let prompt = [1, 4, 7]
        let cache = target.makeCache()
        let initial = try target.forwardPrefillDetailed(
            inputIds: MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count), imageBatch: nil,
            cache: cache, captureLayerIndices: [0, 2]
        )
        let draftCache = assistant.makeCache()
        assistant.appendTargetContext(initial.capturedHiddenStates, cache: draftCache)
        let result = try MuseGlimmerDFlashDecoder.decode(
            initialLogits: initial.logits, target: target, targetCache: cache,
            assistant: assistant, assistantCache: draftCache.map { $0.fork() },
            generationConfig: GenerationConfig(maxTokens: 256, temperature: 0, topK: 0, topP: 1, minP: 0, repetitionPenalty: 1),
            eosTokens: [], tokenBudget: 256, historySeedTokens: prompt, speculativeTokens: 3,
            assistantModelPath: nil, scheduling: .deferred, minimumAcceptanceRate: 1, fallback: .throughput
        )
        XCTAssertEqual(result.generatedTokens.count, 256)
        XCTAssertEqual(result.stats.serialCalibrationForwards, 3)
        XCTAssertGreaterThan(try XCTUnwrap(result.stats.calibrationSeconds), 0)
        XCTAssertGreaterThan(try XCTUnwrap(result.stats.serialCalibrationSeconds), 0)
        XCTAssertGreaterThan(try XCTUnwrap(result.stats.speculativeSeconds), 0)
        XCTAssertEqual(cache.map(\.offset), Array(repeating: prompt.count, count: cache.count))
        XCTAssertEqual(draftCache.map(\.offset), Array(repeating: prompt.count, count: draftCache.count))
    }

    func testDeferredDFlashMatchesEagerAtBudgetBoundariesAndWithProfiling() throws {
        for dflash2 in [false, true] {
            MLXRandom.seed(7_103)
            let target = MuseGlimmerModel(config: try Self.config())
            let assistant = MuseGlimmerAssistantModel(config: try dflash2 ? Self.dflash2Config() : Self.assistantConfig())
            let prompt = [1, 4, 7]
            let cache = target.makeCache()
            let initial = try target.forwardPrefillDetailed(
                inputIds: MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count), imageBatch: nil,
                cache: cache, captureLayerIndices: [0, 2]
            )
            let draftCache = assistant.makeCache()
            assistant.appendTargetContext(initial.capturedHiddenStates, cache: draftCache)
            evaluateGemma4CacheStorage(cache + draftCache)
            let configurations = [1, 2, 7, 32].flatMap { budget in
                [Float(0), Float(0.8)].map { (budget, $0) }
            }
            for (budget, temperature) in configurations {
                let sampling = GenerationConfig(maxTokens: budget, temperature: temperature, topK: 0, topP: 1,
                                                minP: 0, repetitionPenalty: 1)
                func run(_ scheduling: MuseGlimmerDFlashScheduling,
                         observer: ((MuseGlimmerDFlashEvent) -> Void)? = nil) throws -> MuseGlimmerDFlashDecodeResult {
                    MLXRandom.seed(17_103)
                    var drawIndex = 0
                    let draws: [Float] = [0.125, 0.875, 0.375, 0.625]
                    return try MuseGlimmerDFlashDecoder.decode(
                        initialLogits: initial.logits, target: target, targetCache: cache.map { $0.fork() },
                        assistant: assistant, assistantCache: draftCache.map { $0.fork() }, generationConfig: sampling,
                        eosTokens: [], tokenBudget: budget, historySeedTokens: prompt, speculativeTokens: 3,
                        assistantModelPath: nil, scheduling: scheduling, minimumAcceptanceRate: 0, observer: observer,
                        acceptanceDraw: {
                            defer { drawIndex += 1 }
                            return draws[drawIndex % draws.count]
                        }
                    )
                }
                let reference = try run(.eager)
                let deferred = try run(.deferred)
                XCTAssertEqual(deferred.generatedTokens, reference.generatedTokens)
                XCTAssertEqual(deferred.stats.acceptedTokens, reference.stats.acceptedTokens)
                XCTAssertEqual(deferred.stats.targetVerificationForwards, reference.stats.targetVerificationForwards)
                var verified = 0
                let profiled = try run(.deferred) { event in
                    switch event {
                    case .drafted(let arrays), .committed(let arrays), .fallback(let arrays): MLX.eval(arrays)
                    case .verified: verified += 1
                    default: break
                    }
                }
                XCTAssertEqual(profiled.generatedTokens, reference.generatedTokens)
                XCTAssertEqual(verified, profiled.stats.targetVerificationForwards)
                XCTAssertEqual(cache.map(\.offset), Array(repeating: prompt.count, count: cache.count))
                XCTAssertEqual(draftCache.map(\.offset), Array(repeating: prompt.count, count: draftCache.count))
            }
        }
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

private extension MuseGlimmerGenerator {
    func checkPairedPrefix(config: MuseGlimmerConfig, assistantConfig: MuseGlimmerAssistantConfig) throws {
        let target = MuseGlimmerModel(config: config)
        let assistant = MuseGlimmerAssistantModel(config: assistantConfig)
        let tokens = [1, 4, 7, 2, 5, 8, 3]
        let targetCache = target.makeCache()
        let draftCache = assistant.makeCache()
        let fresh = try chunkedPrefill(model: target, tokens: tokens, imageBatch: nil, cache: targetCache,
            assistant: assistant, assistantCache: draftCache, progressHandler: nil, checkpoints: [4: 300], cacheIdentity: "fixture")
        let saved = try XCTUnwrap(prefixCache.lookup(identity: "fixture", tokens: tokens)?.value)
        let savedAssistant = try XCTUnwrap(saved.assistantCache)
        XCTAssertTrue(saved.cache.allSatisfy { $0.offset == 4 })
        XCTAssertTrue(savedAssistant.allSatisfy { $0.offset == 4 })
        let restoredTarget = saved.cache.map { $0.fork() }
        let restoredAssistant = savedAssistant.map { $0.fork() }
        let restored = try chunkedPrefill(model: target, tokens: tokens, imageBatch: nil, cache: restoredTarget,
            assistant: assistant, assistantCache: restoredAssistant, progressHandler: nil, offset: 4,
            initialLogits: saved.logits, cacheIdentity: "fixture")
        XCTAssertEqual(MLX.max(MLX.abs(fresh.logits - restored.logits)).item(Float.self), 0)
        let sampling = GenerationConfig(maxTokens: 12, temperature: 0, topK: 0, topP: 1, minP: 0, repetitionPenalty: 1)
        func decode(_ logits: MLXArray, _ cache: [Gemma4AttentionCache], _ draft: [Gemma4AttentionCache]) throws -> MuseGlimmerDFlashDecodeResult {
            try MuseGlimmerDFlashDecoder.decode(initialLogits: logits, target: target, targetCache: cache,
                assistant: assistant, assistantCache: draft, generationConfig: sampling, eosTokens: [], tokenBudget: 12,
                historySeedTokens: tokens, speculativeTokens: 3, assistantModelPath: nil)
        }
        let expected = try decode(fresh.logits, targetCache, draftCache)
        let actual = try decode(restored.logits, restoredTarget, restoredAssistant)
        XCTAssertEqual(actual.generatedTokens, expected.generatedTokens)
        XCTAssertGreaterThan(actual.stats.rounds, 0)
        XCTAssertTrue(saved.cache.allSatisfy { $0.offset == 4 })
        XCTAssertTrue(savedAssistant.allSatisfy { $0.offset == 4 })
    }

    func checkImageFeatures(config: MuseGlimmerConfig) throws {
        let target = MuseGlimmerModel(config: config)
        target.update(parameters: target.parameters().mapValues { $0.asType(.bfloat16) })
        let pixels = Array(0..<192).map { Float($0) / 192 }
        let batch = MuseGlimmerImageBatch(pixelValues: MLXArray(pixels, [8, 24]),
            grids: [(1, 2, 2), (1, 2, 2)], tokenCounts: [1, 1], identities: ["first", "second"])
        let tokens = [1, 9, 3, 9, 4]
        let expected = try target.inputEmbeddings(for: MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count), imageBatch: batch)
        let actual = try imageInputEmbeddings(model: target, tokens: tokens, batch: batch, identity: "scope-one", retain: true)
        XCTAssertEqual(MLX.max(MLX.abs(actual - expected)).item(Float.self), 0)
        XCTAssertEqual(imageFeatureCache.entries.count, 2)
        let repeated = try imageInputEmbeddings(model: target, tokens: tokens, batch: batch, identity: "scope-one", retain: true)
        XCTAssertEqual(MLX.max(MLX.abs(actual - repeated)).item(Float.self), 0)
        XCTAssertEqual(imageFeatureCache.entries.count, 2)
        _ = try imageInputEmbeddings(model: target, tokens: tokens, batch: batch, identity: "scope-two", retain: true)
        XCTAssertEqual(imageFeatureCache.entries.count, 4)
        var changed = batch
        changed.identities[1] = "changed-second"
        _ = try imageInputEmbeddings(model: target, tokens: tokens, batch: changed, identity: "scope-one", retain: true)
        XCTAssertEqual(imageFeatureCache.entries.count, 5)
        imageFeatureCache.removeAll()
        let suffix = try imageInputEmbeddings(model: target, tokens: tokens, batch: batch, identity: "scope-one", retain: true, offset: 3)
        XCTAssertEqual(MLX.max(MLX.abs(suffix - expected[0..., 3..., 0...])).item(Float.self), 0)
        XCTAssertEqual(imageFeatureCache.entries.count, 1, "Images already covered by KV do not run through the vision tower")
        let larger = MuseGlimmerImageBatch(pixelValues: MLXArray(Array(0..<768).map { Float($0) / 768 }, [32, 24]),
            grids: [(1, 4, 4), (1, 4, 4)], tokenCounts: [4, 4], identities: ["larger-first", "larger-second"])
        let expanded = [1, 9, 9, 9, 9, 3, 9, 9, 9, 9, 4]
        let complete = try target.inputEmbeddings(for: MLXArray(expanded.map(Int32.init)).reshaped(1, expanded.count), imageBatch: larger)
        for offset in [3, 7] {
            imageFeatureCache.removeAll()
            let partial = try imageInputEmbeddings(model: target, tokens: expanded, batch: larger,
                identity: "scope-one", retain: true, offset: offset)
            XCTAssertEqual(MLX.max(MLX.abs(partial - complete[0..., offset..., 0...])).item(Float.self), 0,
                "A checkpoint inside an expanded image must restore the remaining feature rows exactly")
        }
        unload()
        XCTAssertTrue(imageFeatureCache.entries.isEmpty)
        XCTAssertTrue(prefixCache.entries.isEmpty)
    }
}
