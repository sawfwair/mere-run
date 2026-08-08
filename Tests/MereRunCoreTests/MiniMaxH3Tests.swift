import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class MiniMaxH3Tests: MereRunCoreTestCase {
    func testRuntimeLoRAAppliesDeltaInActivationSpace() {
        let base = Linear(
            weight: MLXArray([Float(1), 0, 0, 1]).reshaped(2, 2),
            bias: nil
        )
        let layer = MiniMaxH3RuntimeLoRALinear(
            base: base,
            loraDown: MLXArray([Float(1), 0]).reshaped(1, 2),
            loraUp: MLXArray([Float(2), 3]).reshaped(2, 1),
            strength: 0.5
        )
        let output = layer(MLXArray([Float(4), 5]).reshaped(1, 2))
        MLX.eval(output)
        XCTAssertEqual(output.asArray(Float.self), [8, 11])
    }

    func testTurboAdapterUsesFourDenoiseEvaluationsByDefault() throws {
        let options = try MiniMaxH3GenerationOptions(
            prompt: "a cinematic local video",
            width: 256,
            height: 160,
            numFrames: 22,
            adapterURL: URL(fileURLWithPath: "/tmp/minimax-h3-turbo.safetensors")
        )
        XCTAssertEqual(options.steps, 5)
        XCTAssertEqual(options.adapterStrength, 1)

        XCTAssertThrowsError(try MiniMaxH3GenerationOptions(
            prompt: "invalid compounded acceleration",
            width: 256,
            height: 160,
            numFrames: 22,
            accelerationMode: .maximum,
            adapterURL: URL(fileURLWithPath: "/tmp/minimax-h3-turbo.safetensors")
        ))
    }

    func testTransformerQKVUsesConvertedGlobalSlabs() {
        let projected = MLXArray((0..<12).map(Float.init)).reshaped(1, 1, 12)
        let parts = miniMaxH3SplitProjectedQKV(projected, heads: 2, headDimension: 2)
        MLX.eval(parts)

        XCTAssertEqual(parts[0].asArray(Float.self), [0, 1, 2, 3])
        XCTAssertEqual(parts[1].asArray(Float.self), [4, 5, 6, 7])
        XCTAssertEqual(parts[2].asArray(Float.self), [8, 9, 10, 11])
    }

    func testReleasedBF16TransformerQKVIsDeinterleavedAtLoadTime() {
        let raw = MLXArray(0..<12).reshaped(12, 1)
        let mapped = MiniMaxH3ModelLoader.releasedBF16TransformerWeight(
            key: "blocks.0.attn.qkv_proj.weight",
            value: raw,
            omitCachedAdaLNWeights: true,
            headCount: 2,
            headDimension: 2
        )
        MLX.eval(mapped.map(\.1))

        XCTAssertEqual(mapped.map(\.0), ["blocks.0.attn.qkv_proj.weight"])
        XCTAssertEqual(mapped[0].1.asArray(Int32.self), [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11])
        XCTAssertTrue(
            MiniMaxH3ModelLoader.releasedBF16TransformerWeight(
                key: "blocks.0.adaln_proj.linear.weight",
                value: raw,
                omitCachedAdaLNWeights: true
            ).isEmpty
        )
    }

    func testPinnedMLXArtifactConfigurationDecodes() throws {
        let data = Data(#"""
        {
          "model_type": "minimax_h3",
          "partition": "fl2va",
          "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
          "quantization": {"group_size": 64, "bits": 8, "mode": "affine"},
          "transformer": {
            "hidden_size": 5376,
            "num_layers": 50,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "ffn_hidden_size": 14336,
            "latents_dim": 24,
            "audio_latents_dim": 32,
            "text_dim": 5120,
            "time_embed_dim": 2688,
            "rope_inv_freq_len": 16
          }
        }
        """#.utf8)
        let configuration = try JSONDecoder().decode(MiniMaxH3Configuration.self, from: data)
        XCTAssertEqual(configuration.task, "fl2va")
        XCTAssertEqual(configuration.quantization?.bits, 8)
        XCTAssertEqual(configuration.textEncoderQuantization?.bits, 8)
        XCTAssertEqual(configuration.timeEmbeddingHiddenSize, 5_376)
        XCTAssertEqual(configuration.timeEmbeddingDimension, 2_688)
        XCTAssertTrue(configuration.validationIssues().isEmpty)
    }

    func testMixedTransformerAndTextEncoderQuantizationDecodes() throws {
        let data = Data(#"""
        {
          "model_type": "minimax_h3",
          "partition": "fl2va",
          "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
          "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
          "text_encoder_quantization": {"group_size": 64, "bits": 8, "mode": "affine"},
          "transformer": {
            "hidden_size": 5376,
            "num_layers": 50,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "ffn_hidden_size": 14336,
            "latents_dim": 24,
            "audio_latents_dim": 32,
            "text_dim": 5120,
            "time_embed_dim": 2688,
            "rope_inv_freq_len": 16
          }
        }
        """#.utf8)
        let configuration = try JSONDecoder().decode(MiniMaxH3Configuration.self, from: data)
        XCTAssertEqual(configuration.quantization?.bits, 4)
        XCTAssertEqual(configuration.textEncoderQuantization?.bits, 8)
        XCTAssertTrue(configuration.validationIssues().isEmpty)
    }

    func testManagedFL2VAProfileUsesOfficialSourceCompactQ4Artifact() throws {
        XCTAssertEqual(
            MiniMaxH3Resources.artifactRepository,
            "Sawfwair/MiniMax-H3-FL2VA-MLX-4bit"
        )
        XCTAssertEqual(
            MiniMaxH3Resources.sourceRepository,
            "MiniMaxAI/MiniMax-H3"
        )
        XCTAssertEqual(
            MiniMaxH3Resources.sourceRevision,
            "ec19cc6daf5d8add9417c18e86b6b58cc6c55027"
        )
        XCTAssertTrue(MiniMaxH3Resources.compactArtifactFiles.contains("adaln_cache.safetensors"))
        XCTAssertTrue(MiniMaxH3Resources.compactArtifactFiles.contains("SOURCE_MANIFEST.json"))
        XCTAssertTrue(MiniMaxH3Resources.compactArtifactFiles.contains("transformer.conversion.json"))
        XCTAssertTrue(MiniMaxH3Resources.compactArtifactFiles.contains("SHA256SUMS"))

        let manifest = MereRunModelManifest.template(
            for: .miniMaxH3FL2VAMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.quantization?.bits, 4)
        XCTAssertEqual(manifest.quantization?.groupSize, 64)
        XCTAssertEqual(manifest.quantization?.scheme, "mlx-affine")
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(MiniMaxH3Resources.artifactRepository)@\(MiniMaxH3Resources.artifactRevision)"
        )

        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: MiniMaxH3Resources.fl2vaModelID))
        XCTAssertEqual(spec.hubFallback?.repoId, MiniMaxH3Resources.artifactRepository)
        XCTAssertEqual(spec.hubFallback?.revision, MiniMaxH3Resources.artifactRevision)
        XCTAssertEqual(spec.hubFallback?.patterns, MiniMaxH3Resources.compactArtifactFiles)
    }

    func testManagedBF16ProfileUsesPinnedExistingMLXArtifact() throws {
        XCTAssertEqual(
            MiniMaxH3Resources.bf16ArtifactRepository,
            "pipenetwork/MiniMax-H3-MLX-bf16"
        )
        XCTAssertEqual(
            MiniMaxH3Resources.bf16ArtifactRevision,
            "1486555759eed9e3037edf29f9e055a0713bab2f"
        )
        XCTAssertEqual(MiniMaxH3Resources.bf16ShardFilenames.count, 13)
        XCTAssertFalse(MiniMaxH3Resources.bf16SupportArtifactFiles.contains("transformer.safetensors"))
        XCTAssertFalse(MiniMaxH3Resources.bf16SupportArtifactFiles.contains("adaln_cache.safetensors"))

        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: MiniMaxH3Resources.fl2vaBF16ModelID)
        )
        XCTAssertEqual(spec.hubFallback?.repoId, MiniMaxH3Resources.artifactRepository)
        XCTAssertEqual(spec.hubFallback?.patterns, MiniMaxH3Resources.bf16SupportArtifactFiles)
        let transformer = try XCTUnwrap(spec.mountedHubFallbacks.first)
        XCTAssertEqual(transformer.destinationPath, MiniMaxH3Resources.bf16TransformerDirectory)
        XCTAssertEqual(transformer.hubFallback.repoId, MiniMaxH3Resources.bf16ArtifactRepository)
        XCTAssertEqual(transformer.hubFallback.revision, MiniMaxH3Resources.bf16ArtifactRevision)

        let manifest = MereRunModelManifest.template(
            for: .miniMaxH3FL2VABF16MLX,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertNil(manifest.quantization)
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(MiniMaxH3Resources.bf16ArtifactRepository)@\(MiniMaxH3Resources.bf16ArtifactRevision)"
        )
    }

    func testCompactTransformerPreservesAdaLNCacheSourceIdentity() throws {
        let rootURL = FileManager.default.temporaryDirectory
            .appending(path: "minimax-h3-compact-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: rootURL) }
        let transformerURL = rootURL.appending(path: "transformer.safetensors")
        try MLX.save(
            arrays: ["probe": MLXArray.zeros([1])],
            metadata: [
                "adaln_cache_source_identity": "35248980296:1785757316960297728",
                "cache_covered_weights_omitted": "true",
            ],
            url: transformerURL
        )
        let resources = MiniMaxH3Resources(rootURL: rootURL)
        XCTAssertEqual(
            try resources.adaLNCacheSourceIdentity(),
            "35248980296:1785757316960297728"
        )
        XCTAssertEqual(
            try resources.transformerMetadata()["cache_covered_weights_omitted"],
            "true"
        )
        XCTAssertTrue(try resources.requiresAdaLNCache())
    }

    func testPinnedArtifactWeightMaps() {
        XCTAssertEqual(
            MiniMaxH3ModelLoader.conditionerWeightKey("model.layers.0.self_attn.q_proj.weight"),
            "textEncoder.encoder.layers.0.self_attn.q_proj.weight"
        )
        XCTAssertEqual(
            MiniMaxH3ModelLoader.conditionerWeightKey("visual.merger.linear_fc1.weight"),
            "visionTower.patch_merger.mlp_0.weight"
        )
        let convolution = MLXArray(0..<720).reshaped(2, 3, 4, 5, 6)
        let mappedConvolution = MiniMaxH3VideoVAE.mapCheckpointWeight(
            key: "encoder.conv_in.weight",
            value: convolution
        )
        XCTAssertEqual(mappedConvolution.first?.0, "encoder.conv_in.weight")
        XCTAssertEqual(mappedConvolution.first?.1.shape, [2, 4, 5, 6, 3])

        let fused = MLXArray(0..<6_144)
        let mappedQKV = MiniMaxH3VideoVAE.mapCheckpointWeight(
            key: "decoder.transformer_blocks.0.attn.to_qkv.weight",
            value: fused
        )
        XCTAssertEqual(mappedQKV.map(\.0), [
            "decoder.transformer_blocks.0.attn.to_qkv.weight",
        ])
        XCTAssertEqual(mappedQKV[0].1.shape, [6_144])
        let packedQKV = mappedQKV[0].1.asArray(Int32.self)
        XCTAssertEqual(packedQKV[0], 0)
        XCTAssertEqual(packedQKV[64], 192)
        XCTAssertEqual(packedQKV[2_047], 6_015)
        XCTAssertEqual(packedQKV[2_048], 64)
        XCTAssertEqual(packedQKV[2_112], 256)
        XCTAssertEqual(packedQKV[4_095], 6_079)
        XCTAssertEqual(packedQKV[4_096], 128)
        XCTAssertEqual(packedQKV[4_160], 320)
        XCTAssertEqual(packedQKV[6_143], 6_143)

        let audioEncoder = MiniMaxH3AudioVAE.mapConvertedWeight(
            key: "encoder.block.0.weight",
            value: MLXArray.zeros([64, 1, 7])
        )
        XCTAssertEqual(audioEncoder.first?.1.shape, [64, 7, 1])
        let audioInputBias = MiniMaxH3AudioVAE.mapConvertedWeight(
            key: "dec_in_proj.bias",
            value: MLXArray.zeros([2_048])
        )
        XCTAssertEqual(audioInputBias.map(\.0), ["dec_in_proj.bias"])
        XCTAssertEqual(audioInputBias.first?.1.shape, [2_048])
        let audioResidual = MiniMaxH3AudioVAE.mapConvertedWeight(
            key: "encoder.block.1.block.0.block.1.weight",
            value: MLXArray.zeros([64, 64, 7])
        )
        XCTAssertEqual(audioResidual.first?.0, "encoder.block.1.block.0.block.1.weight")
        XCTAssertEqual(audioResidual.first?.1.shape, [64, 7, 64])
        let audioDecoder = MiniMaxH3AudioVAE.mapConvertedWeight(
            key: "decoder.conv_pre.weight",
            value: MLXArray.zeros([4, 2, 7])
        )
        XCTAssertEqual(audioDecoder.map(\.0), [
            "decoder.conv_pre.parametrizations.weight.original0",
            "decoder.conv_pre.parametrizations.weight.original1",
        ])

        let audioUpsample = MiniMaxH3AudioVAE.mapConvertedWeight(
            key: "decoder.ups.0.0.weight",
            value: MLXArray.zeros([1_024, 512, 9])
        )
        XCTAssertEqual(audioUpsample.map(\.0), [
            "decoder.ups.0.convolution.parametrizations.weight.original0",
            "decoder.ups.0.convolution.parametrizations.weight.original1",
        ])
        XCTAssertEqual(audioUpsample[0].1.shape, [1_024, 1, 1])
        XCTAssertEqual(audioUpsample[1].1.shape, [512, 9, 1_024])
    }

    func testReleasedTemporalGeometry() throws {
        XCTAssertEqual(try MiniMaxH3Geometry.alignFrameCount(120), 124)
        XCTAssertEqual(try MiniMaxH3Geometry.videoLatentFrameCount(for: 124), 37)
        XCTAssertEqual(MiniMaxH3Geometry.audioLatentFrameCount(for: 124), 207)
        XCTAssertThrowsError(try MiniMaxH3Geometry.videoLatentFrameCount(for: 120))
    }

    func testFL2VAPackedLayoutRangesAndTags() throws {
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 0, 1],
            videoLatentFrames: 7,
            latentHeight: 4,
            latentWidth: 6,
            audioLatentFrames: 5,
            keyframeAnchors: [.first, .last]
        )
        XCTAssertEqual(layout.textRows, 0..<3)
        XCTAssertEqual(layout.conditionVideoRowCount, 12)
        XCTAssertEqual(layout.targetAudioRows.count, 10)
        XCTAssertEqual(layout.targetVideoRows.count, 42)
        XCTAssertEqual(layout.sequenceLength, 67)
        XCTAssertEqual(layout.positions.shape, [67, 3])
        XCTAssertEqual(layout.tokenTags[0..<3], [1, 0, 1])
        XCTAssertTrue(layout.tokenTags[3..<15].allSatisfy { $0 == MiniMaxH3Modality.video.rawValue })
    }

    func testVideoAndAudioPackingRoundTrip() {
        let video = MLXArray(0..<384).asType(.float32).reshaped(1, 3, 2, 8, 8)
        let rows = MiniMaxH3Geometry.patchifyVideo(video)
        let roundTrip = MiniMaxH3Geometry.unpatchifyVideo(
            rows,
            frames: 2,
            height: 8,
            width: 8,
            channels: 3
        )
        MLX.eval(roundTrip)
        XCTAssertEqual(roundTrip.shape, video.shape)
        XCTAssertEqual(roundTrip.asArray(Float.self), video.asArray(Float.self))

        let audio = MLXArray(0..<40).asType(.float32).reshaped(1, 4, 2, 5)
        let audioRows = MiniMaxH3Geometry.packAudio(audio)
        let audioRoundTrip = MiniMaxH3Geometry.unpackAudio(audioRows)
        MLX.eval(audioRoundTrip)
        XCTAssertEqual(audioRoundTrip.shape, audio.shape)
        XCTAssertEqual(audioRoundTrip.asArray(Float.self), audio.asArray(Float.self))
    }

    func testInstalledAudioVAEDecodeMatchesReference() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_H3_MODEL_ROOT"], !root.isEmpty else {
            throw XCTSkip(
                "Set MERERUN_H3_MODEL_ROOT to run the checkpoint-backed H3 audio parity fixture."
            )
        }

        let latentFrames = 8
        let latentValues = (0..<(32 * 2 * latentFrames)).map { index in
            Float((index * 37) % 257 - 128) / 64
        }
        let latent = MLXArray(latentValues).reshaped(1, 32, 2, latentFrames)
        let resources = MiniMaxH3Resources(rootURL: URL(fileURLWithPath: root, isDirectory: true))
        let waveform = try MiniMaxH3ModelLoader.loadAudioVAE(resources: resources).decode(latent)
        MLX.eval(waveform)

        XCTAssertEqual(waveform.shape, [1, latentFrames * MiniMaxH3AudioVAE.hopLength, 2])
        let values = waveform.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy(\.isFinite))
        // FP32 samples from ComfyUI's MiniMaxH3AudioVAE at
        // 16e3f3034f2bba1fff6c70cbd759339778555cd6 for the latent fixture above.
        let referenceSamples: [(Int, Float)] = [
            (0, 0.044_520_34), (1, 0.011_212_92), (2, -0.013_318_60),
            (63, -0.142_264_11), (255, 0.202_620_71), (511, -0.673_421_03),
            (1_023, 0.361_299_25), (2_047, -0.206_609_98),
            (4_095, -0.600_924_31), (6_143, 0.713_577_15),
            (8_191, 0.200_860_02), (10_239, 0.757_872_64),
            (12_287, -0.011_021_69), (12_799, 0.009_704_24),
        ]
        for (index, expected) in referenceSamples {
            XCTAssertEqual(values[index], expected, accuracy: 0.000_1, "reference sample \(index)")
        }
        let rootMeanSquare = sqrt(values.reduce(0) { $0 + $1 * $1 } / Float(values.count))
        XCTAssertEqual(rootMeanSquare, 0.556_741_18, accuracy: 0.000_1)
    }

    func testDecodedFramesConvertToMediaPixels() {
        let decoded = MLXArray([Float(0), 0.5, 1]).reshaped(1, 1, 1, 1, 3)
        let pixels = MiniMaxH3Generator.mediaFrames(from: decoded)
        MLX.eval(pixels)
        XCTAssertEqual(pixels.dtype, .uint8)
        XCTAssertEqual(pixels.shape, decoded.shape)
        XCTAssertEqual(pixels.asArray(UInt8.self), [0, 127, 255])
    }

    func testVideoVAETilePlansPreserveCanvasAndMinimumOverlap() {
        XCTAssertEqual(MiniMaxH3VideoVAE.defaultSpatialTileSize, 256)
        for tileSize in [256, 304, 320] {
            for length in [480, 832, 1_344] {
                let plan = MiniMaxH3VideoVAE.tilePlan(length: length, tileSize: tileSize)
                XCTAssertEqual(plan.starts.first, 0)
                XCTAssertEqual(plan.starts.last! + plan.lengths.last!, length)
                XCTAssertTrue(plan.lengths.allSatisfy { $0 == tileSize || $0 == length })
                XCTAssertTrue(plan.overlaps.allSatisfy {
                    $0 >= MiniMaxH3VideoVAE.minimumSpatialTileOverlap
                })
            }
        }
    }

    func testRef2VAPackedLayoutPreservesOrderedReferenceBlocks() throws {
        let layout = try MiniMaxH3Geometry.buildRef2VA(
            textTokenTags: [1, 0],
            references: [
                .init(kind: .image, videoLatentFrames: 1, latentHeight: 4, latentWidth: 4),
                .init(
                    kind: .video,
                    videoLatentFrames: 2,
                    latentHeight: 4,
                    latentWidth: 6,
                    audioLatentFrames: 3
                ),
                .init(kind: .audio, audioLatentFrames: 2),
            ],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 4
        )
        XCTAssertEqual(layout.conditionVideoRowCount, 16)
        XCTAssertEqual(layout.conditionAudioRowCount, 10)
        XCTAssertEqual(layout.conditionRows, 2..<28)
        XCTAssertEqual(layout.targetAudioRows, 28..<36)
        XCTAssertEqual(layout.targetVideoRows, 36..<44)
        XCTAssertEqual(layout.conditionSegments.map(\.modality), [.video, .audio, .video, .audio])
        XCTAssertEqual(layout.conditionSegments.map(\.packedRows), [2..<6, 6..<12, 12..<24, 24..<28])
        XCTAssertEqual(layout.conditionSegments.map(\.sourceRows), [0..<4, 0..<6, 4..<16, 6..<10])
        XCTAssertTrue(layout.tokenTags[6..<12].allSatisfy { $0 == MiniMaxH3Modality.audio.rawValue })
        XCTAssertTrue(layout.tokenTags[12..<24].allSatisfy { $0 == MiniMaxH3Modality.video.rawValue })
    }

    func testShiftedSchedulesTerminateAtCleanEndpoint() throws {
        let video = try MiniMaxH3Schedule(pointCount: 5, shift: 12)
        let audio = try MiniMaxH3Schedule(pointCount: 5, shift: 3)
        XCTAssertEqual(video.sigmas.first, 1)
        XCTAssertEqual(video.sigmas.last, 0)
        XCTAssertEqual(audio.sigmas.first, 1)
        XCTAssertEqual(audio.sigmas.last, 0)
        XCTAssertEqual(video.timesteps.count, 4)
        XCTAssertEqual(video.timesteps.first, 0)

        let sample = MLXArray([1, 2, 3, 4]).reshaped(1, 4)
        let velocity = MLXArray.ones([1, 4])
        let final = video.step(sample: sample, velocity: velocity, index: video.timesteps.count - 1)
        MLX.eval(final)
        XCTAssertTrue(final.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testBlockReusePolicyBoundsCacheStepsForPracticalSchedule() throws {
        XCTAssertNil(MiniMaxH3AccelerationMode.quality.blockReusePolicy)
        XCTAssertEqual(
            MiniMaxH3BlockReusePolicy(cacheDepth: 0.5).warmBlockCount(totalBlockCount: 50),
            25
        )
        let policy = try XCTUnwrap(MiniMaxH3AccelerationMode.maximum.blockReusePolicy)
        XCTAssertEqual(policy.warmBlockCount(totalBlockCount: 50), 9)
        XCTAssertEqual(policy.maximumConsecutiveCachedSteps, 4)

        let video = try MiniMaxH3Schedule(pointCount: 9, shift: 12)
        let audio = try MiniMaxH3Schedule(pointCount: 9, shift: 3)
        var hasResidual = false
        var consecutive = 0
        var decisions: [Bool] = []
        for index in video.timesteps.indices {
            let reuses = policy.shouldReuseTail(
                stepIndex: index,
                stepCount: video.timesteps.count,
                videoSigmas: video.sigmas,
                audioSigmas: audio.sigmas,
                hasCachedResidual: hasResidual,
                consecutiveCachedSteps: consecutive
            )
            decisions.append(reuses)
            if reuses {
                consecutive += 1
            } else {
                hasResidual = true
                consecutive = 0
            }
        }
        XCTAssertEqual(decisions, [false, false, true, true, true, true, false, false])

        let longVideo = try MiniMaxH3Schedule(pointCount: 16, shift: 12)
        let longAudio = try MiniMaxH3Schedule(pointCount: 16, shift: 3)
        hasResidual = false
        consecutive = 0
        var longCachedSteps = 0
        for index in longVideo.timesteps.indices {
            let reuses = policy.shouldReuseTail(
                stepIndex: index,
                stepCount: longVideo.timesteps.count,
                videoSigmas: longVideo.sigmas,
                audioSigmas: longAudio.sigmas,
                hasCachedResidual: hasResidual,
                consecutiveCachedSteps: consecutive
            )
            if reuses {
                longCachedSteps += 1
                consecutive += 1
            } else {
                hasResidual = true
                consecutive = 0
            }
        }
        XCTAssertEqual(longCachedSteps, 10)
    }

    func testResidentBF16PolicyAccountsForGeometryAndMemory() throws {
        let denseBytes = 41 * MiniMaxH3ResidentBF16Policy.gibibyte
        XCTAssertTrue(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .automatic,
            physicalMemoryBytes: 64 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 12_925,
            hasAdaLNCache: true,
            isPortableMac: false
        ))
        XCTAssertFalse(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .automatic,
            physicalMemoryBytes: 64 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 37_966,
            hasAdaLNCache: true,
            isPortableMac: false
        ))
        XCTAssertTrue(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .automatic,
            physicalMemoryBytes: 128 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 37_966,
            hasAdaLNCache: true,
            isPortableMac: false
        ))
        XCTAssertFalse(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .quantized,
            physicalMemoryBytes: 128 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 12_925,
            hasAdaLNCache: true,
            isPortableMac: false
        ))
        XCTAssertFalse(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .automatic,
            physicalMemoryBytes: 128 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 12_925,
            hasAdaLNCache: false,
            isPortableMac: false
        ))
        XCTAssertTrue(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .automatic,
            physicalMemoryBytes: 128 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 12_925,
            hasAdaLNCache: true,
            isPortableMac: true
        ))
        XCTAssertFalse(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .automatic,
            physicalMemoryBytes: 64 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 12_925,
            hasAdaLNCache: true,
            isPortableMac: true
        ))
        XCTAssertThrowsError(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .residentBF16,
            physicalMemoryBytes: 48 * MiniMaxH3ResidentBF16Policy.gibibyte,
            estimatedResidentBytes: denseBytes,
            sequenceLength: 12_925,
            hasAdaLNCache: true,
            isPortableMac: true
        ))
    }

    func testDenoiseExecutionPolicyKeepsProfilingOutsideCompiledTransforms() {
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: false,
                sequenceLength: 11_925,
                usesBlockProfiling: false
            ),
            .compiledStep
        )
        XCTAssertFalse(MiniMaxH3DenoiseExecutionMode.compiledStep.usesLayerwiseEvaluation)
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: false,
                sequenceLength: 11_925,
                usesBlockProfiling: true
            ),
            .eagerStep
        )
        XCTAssertTrue(MiniMaxH3DenoiseExecutionMode.eagerStep.usesLayerwiseEvaluation)
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: true,
                sequenceLength: 11_925,
                usesBlockProfiling: false,
                denoiseStepCount: 1
            ),
            .eagerStep
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: true,
                sequenceLength: 11_925,
                usesBlockProfiling: false,
                denoiseStepCount: 2
            ),
            .compiledStep
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: true,
                sequenceLength: 11_925,
                usesBlockProfiling: false,
                profilingOverride: "compiled"
            ),
            .compiledStep
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: false,
                sequenceLength: 11_925,
                usesBlockProfiling: true,
                profilingOverride: "compiled"
            ),
            .eagerStep
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: true,
                sequenceLength: MiniMaxH3DenoiseExecutionPolicy.blockwiseSequenceThreshold + 1,
                usesBlockProfiling: true
            ),
            .blockwiseCompiled
        )
        XCTAssertFalse(MiniMaxH3DenoiseExecutionMode.blockwiseCompiled.usesLayerwiseEvaluation)
    }

    func testDenoiseExecutionPolicyUsesMeasuredBlockwiseKernelSchedule() {
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 12_930)
                .maximumQueryTokens,
            640
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 12_930)
                .maximumKernelsPerEvaluation,
            1
        )
        XCTAssertNil(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 12_930)
                .maximumHeadsPerKernel
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 14_958)
                .maximumQueryTokens,
            1_024
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 14_958)
                .maximumKernelsPerEvaluation,
            1
        )
        XCTAssertNil(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 14_958)
                .maximumHeadsPerKernel
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 37_966)
                .maximumQueryTokens,
            768
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 37_966)
                .maximumKernelsPerEvaluation,
            1
        )
        XCTAssertNil(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 37_966)
                .maximumHeadsPerKernel
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 73_470)
                .maximumQueryTokens,
            640
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 73_470)
                .maximumHeadsPerKernel,
            8
        )
        XCTAssertEqual(
            MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(sequenceLength: 73_470)
                .maximumKernelsPerEvaluation,
            1
        )
    }

    func testStepPolicyUsesPracticalExtendedAndMaximumEnvelopes() throws {
        XCTAssertEqual(
            try MiniMaxH3StepPolicy.recommendedPointCount(
                width: 512,
                height: 512,
                numFrames: 56
            ),
            9
        )
        XCTAssertEqual(
            try MiniMaxH3StepPolicy.recommendedPointCount(
                width: 768,
                height: 448,
                numFrames: 124
            ),
            9
        )
        XCTAssertEqual(
            try MiniMaxH3StepPolicy.recommendedPointCount(
                width: 768,
                height: 512,
                numFrames: 124
            ),
            16
        )
        XCTAssertEqual(
            try MiniMaxH3StepPolicy.recommendedPointCount(
                width: 1_344,
                height: 768,
                numFrames: 124
            ),
            21
        )
        XCTAssertEqual(
            try MiniMaxH3StepPolicy.recommendedPointCount(
                width: 1_344,
                height: 768,
                numFrames: 124,
                accelerationMode: .maximum
            ),
            12
        )
        XCTAssertEqual(
            try MiniMaxH3StepPolicy.recommendedPointCount(
                width: 768,
                height: 448,
                numFrames: 124,
                referenceKinds: [.video]
            ),
            16
        )

        let automatic = try MiniMaxH3GenerationOptions(
            prompt: "a practical local video",
            width: 768,
            height: 448,
            numFrames: 124
        )
        XCTAssertEqual(automatic.steps, 9)
        let explicit = try MiniMaxH3GenerationOptions(
            prompt: "a maximum quality local video",
            width: 768,
            height: 448,
            numFrames: 124,
            steps: 31
        )
        XCTAssertEqual(explicit.steps, 31)
        let longClip = try MiniMaxH3GenerationOptions(
            prompt: "a ten second local video",
            width: 832,
            height: 480,
            numFrames: 243
        )
        XCTAssertEqual(longClip.steps, 21)
        let acceleratedLongClip = try MiniMaxH3GenerationOptions(
            prompt: "a fast ten second local video",
            width: 832,
            height: 480,
            numFrames: 243,
            accelerationMode: .maximum
        )
        XCTAssertEqual(acceleratedLongClip.steps, 12)
    }

    func testExactScheduleCacheSurvivesDiscardingBF16AdaLNWeights() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 32,
            layerCount: 2,
            refinerLayerCount: 1,
            attentionHeadCount: 4,
            attentionHeadDimension: 8,
            feedForwardSize: 64,
            videoLatentChannels: 8,
            audioLatentChannels: 32,
            patchSize: [1, 2, 2],
            textDimension: 32,
            timeFrequencyDimension: 32,
            timeEmbeddingHiddenSize: 32,
            timeEmbeddingDimension: 32,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        let videoSchedule = try MiniMaxH3Schedule(pointCount: 4, shift: 12)
        let audioSchedule = try MiniMaxH3Schedule(pointCount: 4, shift: 3)
        let cache = model.precomputeAdaLN(
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: "test"
        )
        XCTAssertEqual(cache.stepCount, 3)
        let scheduleBytesBefore = model.parameters().flattened()
            .filter { $0.0.contains("adaln_proj") || $0.0.hasPrefix("time_embedder.") }
            .reduce(0) { $0 + $1.1.nbytes }
        XCTAssertGreaterThan(scheduleBytesBefore, 1_000)

        model.discardAdaLNWeights()

        let scheduleParametersAfter = model.parameters().flattened()
            .filter { $0.0.contains("adaln_proj") || $0.0.hasPrefix("time_embedder.") }
        XCTAssertTrue(scheduleParametersAfter.allSatisfy { $0.1.size == 1 })
        XCTAssertLessThan(scheduleParametersAfter.reduce(0) { $0 + $1.1.nbytes }, scheduleBytesBefore)
        XCTAssertEqual(cache.step(at: 0).blockModulations.count, 2)
    }

    func testTinyTransformerResidentBF16MatchesQuantizedExecution() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 32,
            layerCount: 2,
            refinerLayerCount: 1,
            attentionHeadCount: 4,
            attentionHeadDimension: 8,
            feedForwardSize: 64,
            videoLatentChannels: 8,
            audioLatentChannels: 32,
            patchSize: [1, 2, 2],
            textDimension: 32,
            timeFrequencyDimension: 32,
            timeEmbeddingHiddenSize: 32,
            timeEmbeddingDimension: 32,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        quantize(
            model: model,
            groupSize: 32,
            bits: 4,
            filter: { _, _ in true },
            apply: { module, groupSize, bits, mode in
                guard let quantized = quantizeSingle(
                    layer: module,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                ) as? QuantizedLinear else { return nil }
                return PortableQuantizedLinear(
                    weight: quantized.weight,
                    bias: quantized.bias,
                    scales: quantized.scales,
                    biases: quantized.biases,
                    groupSize: quantized.groupSize,
                    bits: quantized.bits,
                    mode: quantized.mode,
                    globalScale: quantized.globalScale
                )
            }
        )
        let quantizedCount = model.leafModules().flattened()
            .count(where: { $0.1 is QuantizedLinear })
        let estimatedBytes = model.estimatedResidentBF16ByteCount
        XCTAssertGreaterThan(quantizedCount, 0)
        XCTAssertGreaterThan(estimatedBytes, 0)

        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 1],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: [.first]
        )
        let video = MLXArray.zeros([1, 12, 32], dtype: .bfloat16)
        let audio = MLXArray.zeros([1, 6, 32], dtype: .bfloat16)
        let text = MLXArray.zeros([1, 2, 32], dtype: .bfloat16)
        let quantizedOutput = model(
            videoRows: video,
            audioRows: audio,
            textStates: text,
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(quantizedOutput.videoVelocityRows, quantizedOutput.audioVelocityRows)

        let materialized = model.materializeResidentBF16()
        XCTAssertTrue(model.usesResidentBF16)
        XCTAssertEqual(materialized.linearCount, quantizedCount)
        XCTAssertEqual(materialized.byteCount, estimatedBytes)
        XCTAssertFalse(model.leafModules().flattened().contains { $0.1 is QuantizedLinear })
        let denseOutput = model(
            videoRows: video,
            audioRows: audio,
            textStates: text,
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(denseOutput.videoVelocityRows, denseOutput.audioVelocityRows)
        XCTAssertLessThanOrEqual(
            MLX.abs(
                quantizedOutput.videoVelocityRows.asType(.float32)
                    - denseOutput.videoVelocityRows.asType(.float32)
            ).max().item(Float.self),
            0.05
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                quantizedOutput.audioVelocityRows.asType(.float32)
                    - denseOutput.audioVelocityRows.asType(.float32)
            ).max().item(Float.self),
            0.05
        )
    }

    func testTinyDenseBF16TransformerIsActuallyMaterialized() {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 32,
            layerCount: 2,
            refinerLayerCount: 1,
            attentionHeadCount: 4,
            attentionHeadDimension: 8,
            feedForwardSize: 64,
            videoLatentChannels: 8,
            audioLatentChannels: 32,
            patchSize: [1, 2, 2],
            textDimension: 32,
            timeFrequencyDimension: 32,
            timeEmbeddingHiddenSize: 32,
            timeEmbeddingDimension: 32,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        let linearCount = model.leafModules().flattened().count { $0.1 is Linear }
        let estimatedBytes = model.estimatedResidentBF16ByteCount

        XCTAssertGreaterThan(linearCount, 0)
        XCTAssertGreaterThan(estimatedBytes, 0)
        XCTAssertFalse(model.usesResidentBF16)

        let materialized = model.materializeResidentBF16()

        XCTAssertTrue(model.usesResidentBF16)
        XCTAssertEqual(materialized.linearCount, linearCount)
        XCTAssertEqual(materialized.byteCount, estimatedBytes)
        XCTAssertFalse(model.leafModules().flattened().contains { $0.1 is QuantizedLinear })
    }

    func testTinyTransformerPreservesTargetShapes() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 12,
            layerCount: 2,
            refinerLayerCount: 1,
            attentionHeadCount: 2,
            attentionHeadDimension: 6,
            feedForwardSize: 16,
            videoLatentChannels: 3,
            audioLatentChannels: 4,
            patchSize: [1, 2, 2],
            textDimension: 10,
            timeFrequencyDimension: 4,
            timeEmbeddingHiddenSize: 12,
            timeEmbeddingDimension: 8,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 1],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: [.first]
        )
        let result = model(
            videoRows: MLXArray.zeros([1, 12, 12]),
            audioRows: MLXArray.zeros([1, 6, 4]),
            textStates: MLXArray.zeros([1, 2, 10]),
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(result.videoVelocityRows, result.audioVelocityRows)
        XCTAssertEqual(result.videoVelocityRows.shape, [1, 8, 12])
        XCTAssertEqual(result.audioVelocityRows.shape, [1, 6, 4])
    }

    func testTinyTransformerAcceptsRef2VAConditionAudioAndVideo() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 12,
            layerCount: 1,
            refinerLayerCount: 1,
            attentionHeadCount: 2,
            attentionHeadDimension: 6,
            feedForwardSize: 16,
            videoLatentChannels: 3,
            audioLatentChannels: 4,
            patchSize: [1, 2, 2],
            textDimension: 10,
            timeFrequencyDimension: 4,
            timeEmbeddingHiddenSize: 12,
            timeEmbeddingDimension: 8,
            ropeFrequencyCount: 1
        )
        let layout = try MiniMaxH3Geometry.buildRef2VA(
            textTokenTags: [1, 1],
            references: [
                .init(
                    kind: .video,
                    videoLatentFrames: 1,
                    latentHeight: 4,
                    latentWidth: 4,
                    audioLatentFrames: 2
                ),
            ],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3
        )
        let result = MiniMaxH3Transformer(configuration: configuration)(
            videoRows: MLXArray.zeros([1, 12, 12]),
            audioRows: MLXArray.zeros([1, 10, 4]),
            textStates: MLXArray.zeros([1, 2, 10]),
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(result.videoVelocityRows, result.audioVelocityRows)
        XCTAssertEqual(result.videoVelocityRows.shape, [1, 8, 12])
        XCTAssertEqual(result.audioVelocityRows.shape, [1, 6, 4])
    }

    func testPreparedAndCompiledTinyTransformerMatchDirectExecution() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 32,
            layerCount: 2,
            refinerLayerCount: 1,
            attentionHeadCount: 4,
            attentionHeadDimension: 8,
            feedForwardSize: 64,
            videoLatentChannels: 3,
            audioLatentChannels: 4,
            patchSize: [1, 2, 2],
            textDimension: 32,
            timeFrequencyDimension: 8,
            timeEmbeddingHiddenSize: 32,
            timeEmbeddingDimension: 32,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        quantize(
            model: model,
            groupSize: 32,
            bits: 4,
            filter: { path, _ in path.hasPrefix("blocks.") },
            apply: { module, groupSize, bits, mode in
                guard let quantized = quantizeSingle(
                    layer: module,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                ) as? QuantizedLinear else { return nil }
                return PortableQuantizedLinear(
                    weight: quantized.weight,
                    bias: quantized.bias,
                    scales: quantized.scales,
                    biases: quantized.biases,
                    groupSize: quantized.groupSize,
                    bits: quantized.bits,
                    mode: quantized.mode,
                    globalScale: quantized.globalScale
                )
            }
        )
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 1],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: [.first]
        )
        let video = MLXArray.zeros([1, 12, 12])
        let audio = MLXArray.zeros([1, 6, 4])
        let text = MLXArray.zeros([1, 2, 32])
        let timesteps = MLXArray([Float(0.2), 0.4, 0.999])
        let direct = model(
            videoRows: video,
            audioRows: audio,
            textStates: text,
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        let context = model.prepare(textStates: text, layout: layout)
        let prepared = model(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: nil
        )
        let compiled = MLX.compile { (inputs: [MLXArray]) -> [MLXArray] in
            let output = model(
                videoRows: inputs[0],
                audioRows: inputs[1],
                context: context,
                timesteps: inputs[2],
                cachedAdaLN: nil
            )
            return [output.videoVelocityRows, output.audioVelocityRows]
        }
        let compiledOutputs = compiled([video, audio, timesteps])
        MLX.eval(compiledOutputs)
        model.maximumAttentionQueryTokensPerKernel = 2
        model.maximumAttentionHeadsPerKernel = 1
        model.usesBlockwiseCompilation = true
        let blockwiseCompiled = model(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: nil
        )
        MLX.eval(blockwiseCompiled.videoVelocityRows, blockwiseCompiled.audioVelocityRows)
        let refreshedReuse = model.callWithBlockResidualReuse(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: nil,
            warmBlockCount: 1,
            cachedTailResidual: nil
        )
        let reusedTail = model.callWithBlockResidualReuse(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: nil,
            warmBlockCount: 1,
            cachedTailResidual: try XCTUnwrap(refreshedReuse.refreshedTailResidual)
        )
        MLX.eval(
            refreshedReuse.output.videoVelocityRows,
            refreshedReuse.output.audioVelocityRows,
            reusedTail.output.videoVelocityRows,
            reusedTail.output.audioVelocityRows
        )
        model.usesBlockwiseCompilation = false
        model.usesLayerwiseEvaluation = true
        model.clearsCacheAfterLayerwiseEvaluation = false
        let layerwise = model(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: nil
        )
        MLX.eval(layerwise.videoVelocityRows, layerwise.audioVelocityRows)
        model.usesLayerwiseEvaluation = false
        MLX.eval(
            direct.videoVelocityRows,
            direct.audioVelocityRows,
            prepared.videoVelocityRows,
            prepared.audioVelocityRows,
            compiledOutputs[0],
            compiledOutputs[1],
            blockwiseCompiled.videoVelocityRows,
            blockwiseCompiled.audioVelocityRows,
            layerwise.videoVelocityRows,
            layerwise.audioVelocityRows
        )
        XCTAssertTrue(direct.videoVelocityRows.allClose(prepared.videoVelocityRows).item(Bool.self))
        XCTAssertTrue(direct.audioVelocityRows.allClose(prepared.audioVelocityRows).item(Bool.self))
        XCTAssertLessThanOrEqual(
            MLX.abs(prepared.videoVelocityRows - compiledOutputs[0]).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(prepared.audioVelocityRows - compiledOutputs[1]).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(prepared.videoVelocityRows - blockwiseCompiled.videoVelocityRows).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(prepared.audioVelocityRows - blockwiseCompiled.audioVelocityRows)
                .max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                blockwiseCompiled.videoVelocityRows
                    - refreshedReuse.output.videoVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                blockwiseCompiled.audioVelocityRows
                    - refreshedReuse.output.audioVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                refreshedReuse.output.videoVelocityRows
                    - reusedTail.output.videoVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                refreshedReuse.output.audioVelocityRows
                    - reusedTail.output.audioVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(prepared.videoVelocityRows - layerwise.videoVelocityRows)
                .max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(prepared.audioVelocityRows - layerwise.audioVelocityRows)
                .max().item(Float.self),
            1e-5
        )

        let videoSchedule = try MiniMaxH3Schedule(pointCount: 3, shift: 12)
        let audioSchedule = try MiniMaxH3Schedule(pointCount: 3, shift: 3)
        let cache = model.precomputeAdaLN(
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: "test-transformer"
        )
        let scheduleTimesteps = MLXArray([
            videoSchedule.timesteps[1],
            audioSchedule.timesteps[1],
            max(videoSchedule.timesteps[1], 0.999),
        ])
        let eagerScheduleOutput = model(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: scheduleTimesteps,
            cachedAdaLN: nil
        )
        let cachedScheduleOutput = model(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: scheduleTimesteps,
            cachedAdaLN: cache.step(at: 1)
        )
        model.usesBlockwiseCompilation = true
        let blockwiseCachedScheduleOutput = model(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: scheduleTimesteps,
            cachedAdaLN: cache.step(at: 1)
        )
        MLX.eval(
            blockwiseCachedScheduleOutput.videoVelocityRows,
            blockwiseCachedScheduleOutput.audioVelocityRows
        )
        model.usesBlockwiseCompilation = false
        MLX.eval(
            eagerScheduleOutput.videoVelocityRows,
            eagerScheduleOutput.audioVelocityRows,
            cachedScheduleOutput.videoVelocityRows,
            cachedScheduleOutput.audioVelocityRows,
            blockwiseCachedScheduleOutput.videoVelocityRows,
            blockwiseCachedScheduleOutput.audioVelocityRows
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                eagerScheduleOutput.videoVelocityRows
                    - cachedScheduleOutput.videoVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                eagerScheduleOutput.audioVelocityRows
                    - cachedScheduleOutput.audioVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                cachedScheduleOutput.videoVelocityRows
                    - blockwiseCachedScheduleOutput.videoVelocityRows
            ).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                cachedScheduleOutput.audioVelocityRows
                    - blockwiseCachedScheduleOutput.audioVelocityRows
            ).max().item(Float.self),
            1e-5
        )

        let cacheURL = FileManager.default.temporaryDirectory
            .appending(path: "minimax-h3-adaln-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: cacheURL) }
        try cache.save(to: cacheURL, replacing: false)
        let loaded = try MiniMaxH3AdaLNCache.load(
            from: cacheURL,
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: "test-transformer"
        )
        XCTAssertEqual(loaded.stepCount, 2)
        XCTAssertEqual(loaded.blockModulations.count, 2)
        XCTAssertTrue(loaded.isCompatible(
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule
        ))
        XCTAssertThrowsError(try MiniMaxH3AdaLNCache.load(
            from: cacheURL,
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: "stale-transformer"
        ))

        let denseVideoSchedule = try MiniMaxH3Schedule(pointCount: 5, shift: 12)
        let denseAudioSchedule = try MiniMaxH3Schedule(pointCount: 5, shift: 3)
        let denseCache = model.precomputeAdaLN(
            videoSchedule: denseVideoSchedule,
            audioSchedule: denseAudioSchedule,
            sourceIdentity: "test-transformer"
        )
        let resampled = try denseCache.resampled(
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule
        )
        MLX.eval(
            [resampled.timeEmbeddings, resampled.finalModulations]
                + resampled.blockModulations
        )
        XCTAssertTrue(resampled.isCompatible(
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule
        ))
        XCTAssertEqual(resampled.timeEmbeddings.dtype, denseCache.timeEmbeddings.dtype)
        XCTAssertEqual(resampled.finalModulations.dtype, denseCache.finalModulations.dtype)
        XCTAssertEqual(resampled.blockModulations[0].dtype, denseCache.blockModulations[0].dtype)
        XCTAssertLessThanOrEqual(
            MLX.abs(resampled.timeEmbeddings - cache.timeEmbeddings).max().item(Float.self),
            1e-5
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(resampled.finalModulations - cache.finalModulations).max().item(Float.self),
            1e-5
        )
        for index in cache.blockModulations.indices {
            XCTAssertLessThanOrEqual(
                MLX.abs(resampled.blockModulations[index] - cache.blockModulations[index])
                    .max().item(Float.self),
                1e-5
            )
        }

        try denseCache.save(to: cacheURL, replacing: true)
        XCTAssertThrowsError(try MiniMaxH3AdaLNCache.load(
            from: cacheURL,
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: "test-transformer"
        ))
        let loadedResampled = try MiniMaxH3AdaLNCache.load(
            from: cacheURL,
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: "test-transformer",
            allowScheduleResampling: true
        )
        XCTAssertTrue(loadedResampled.isCompatible(
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule
        ))
    }

    func testTinyVideoDecoderShape() {
        let decoder = MiniMaxH3VideoDecoder(configuration: .init(
            latentChannels: 2,
            outputChannels: 3,
            patchSize: 2,
            temporalPatchSize: 2,
            layerCount: 1,
            headCount: 2,
            headDimension: 6,
            registerTokenCount: 1,
            feedForwardMultiplier: 2,
            rotaryDimensionRatio: 1
        ))
        let result = decoder(MLXArray.zeros([1, 2, 2, 2, 2]))
        MLX.eval(result)
        XCTAssertEqual(result.shape, [1, 3, 4, 4, 4])
    }
}
