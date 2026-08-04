import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class MiniMaxH3Tests: MereRunCoreTestCase {
    func testTransformerQKVUsesConvertedGlobalSlabs() {
        let projected = MLXArray((0..<12).map(Float.init)).reshaped(1, 1, 12)
        let parts = miniMaxH3SplitProjectedQKV(projected, heads: 2, headDimension: 2)
        MLX.eval(parts)

        XCTAssertEqual(parts[0].asArray(Float.self), [0, 1, 2, 3])
        XCTAssertEqual(parts[1].asArray(Float.self), [4, 5, 6, 7])
        XCTAssertEqual(parts[2].asArray(Float.self), [8, 9, 10, 11])
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
        let split = MiniMaxH3VideoVAE.mapCheckpointWeight(
            key: "decoder.transformer_blocks.0.attn.to_qkv.weight",
            value: fused
        )
        XCTAssertEqual(split.map(\.0), [
            "decoder.transformer_blocks.0.attn.to_q.weight",
            "decoder.transformer_blocks.0.attn.to_k.weight",
            "decoder.transformer_blocks.0.attn.to_v.weight",
        ])
        XCTAssertTrue(split.allSatisfy { $0.1.shape == [2_048] })
        XCTAssertEqual(split[0].1.asArray(Int32.self)[0], 0)
        XCTAssertEqual(split[0].1.asArray(Int32.self)[64], 192)
        XCTAssertEqual(split[0].1.asArray(Int32.self)[2_047], 6_015)
        XCTAssertEqual(split[1].1.asArray(Int32.self)[0], 64)
        XCTAssertEqual(split[1].1.asArray(Int32.self)[64], 256)
        XCTAssertEqual(split[1].1.asArray(Int32.self)[2_047], 6_079)
        XCTAssertEqual(split[2].1.asArray(Int32.self)[0], 128)
        XCTAssertEqual(split[2].1.asArray(Int32.self)[64], 320)
        XCTAssertEqual(split[2].1.asArray(Int32.self)[2_047], 6_143)

        let audioEncoder = MiniMaxH3AudioVAE.mapConvertedWeight(
            key: "encoder.block.0.weight",
            value: MLXArray.zeros([64, 1, 7])
        )
        XCTAssertEqual(audioEncoder.first?.1.shape, [64, 7, 1])
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

    func testDecodedFramesConvertToMediaPixels() {
        let decoded = MLXArray([Float(0), 0.5, 1]).reshaped(1, 1, 1, 1, 3)
        let pixels = MiniMaxH3Generator.mediaFrames(from: decoded)
        MLX.eval(pixels)
        XCTAssertEqual(pixels.dtype, .uint8)
        XCTAssertEqual(pixels.shape, decoded.shape)
        XCTAssertEqual(pixels.asArray(UInt8.self), [0, 127, 255])
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
        XCTAssertFalse(try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: .automatic,
            physicalMemoryBytes: 128 * MiniMaxH3ResidentBF16Policy.gibibyte,
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
            31
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
        model.usesBlockwiseCompilation = true
        let blockwiseCompiled = model(
            videoRows: video,
            audioRows: audio,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: nil
        )
        MLX.eval(blockwiseCompiled.videoVelocityRows, blockwiseCompiled.audioVelocityRows)
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
