import AudioCodecs
import MediaIO
import MLX
import MLXNN
@testable import MereRunCore
import XCTest

final class LTXAudioToVideoSupportTests: MereRunCoreTestCase {
    func testParityIOReplaysTypedNoiseAndWritesNamedArtifact() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        let noiseURL = temp.appendingPathComponent("stage1.npy")
        let prefixURL = temp.appendingPathComponent("swift-reference")
        try MLX.save(
            array: MLXArray([Float(1), 2, 3, 4]).reshaped(1, 1, 2, 1, 2),
            url: noiseURL
        )
        let parity = LTXAudioToVideoParityIO(environment: [
            LTXAudioToVideoParityIO.outputPrefixEnvironmentKey: prefixURL.path,
            LTXAudioToVideoParityIO.stage1NoiseEnvironmentKey: noiseURL.path,
        ])
        let replayed = try parity.resolveNoise(
            stage: .stage1,
            generated: MLX.zeros([1, 1, 2, 1, 2], dtype: .bfloat16)
        )
        try parity.save(replayed, suffix: "a2vid_stage1_noise")
        let outputURL = temp.appendingPathComponent("swift-reference_a2vid_stage1_noise.npy")
        let saved = try MLX.loadArray(url: outputURL).asType(.float32)

        XCTAssertEqual(replayed.dtype, .bfloat16)
        XCTAssertEqual(saved.asArray(Float.self), [1, 2, 3, 4])
    }

    func testParityIORejectsMismatchedInjectedNoiseShape() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        let noiseURL = temp.appendingPathComponent("stage2.npy")
        try MLX.save(array: MLX.zeros([1, 2], dtype: .float32), url: noiseURL)
        let parity = LTXAudioToVideoParityIO(environment: [
            LTXAudioToVideoParityIO.stage2NoiseEnvironmentKey: noiseURL.path,
        ])

        XCTAssertThrowsError(
            try parity.resolveNoise(
                stage: .stage2,
                generated: MLX.zeros([1, 3], dtype: .bfloat16)
            )
        ) { error in
            guard case LTXAudioToVideoParityError.invalidNoiseShape(
                stage: "stage2",
                expected: [1, 3],
                actual: [1, 2]
            ) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testFrozenAudioDenoisingReturnsSourceLatentsUnchanged() throws {
        let video = MLXArray([Float](repeating: 0.25, count: 128))
            .reshaped(1, 128, 1, 1, 1)
        let audioValues = (0..<(8 * 3 * 16)).map { Float($0) / 100 }
        let audio = MLXArray(audioValues).reshaped(1, 8, 3, 16)
        let rope: LTXRope = (
            cos: MLX.ones([1], dtype: .float32),
            sin: MLX.zeros([1], dtype: .float32)
        )

        let result = try denoiseFrozenAudioVideoLoop(
            videoLatents: video,
            audioLatents: audio,
            videoRope: rope,
            audioRope: rope,
            videoCrossRope: rope,
            audioCrossRope: rope,
            positiveVideoContext: MLX.zeros([1, 1, 1], dtype: .float32),
            negativeVideoContext: nil,
            audioContext: MLX.zeros([1, 1, 1], dtype: .float32),
            transformer: FrozenAudioTestTransformer(),
            sigmas: [1, 0.5, 0],
            videoConditioning: nil,
            guidance: nil
        )
        MLX.eval(result.audio)

        XCTAssertEqual(result.audio.shape, audio.shape)
        XCTAssertEqual(result.audio.asArray(Float.self), audioValues)
    }

    func testFrozenAudioDenoisingPreservesStartAndEndImageAnchors() throws {
        let video = MLXArray([Float](repeating: 0.25, count: 128 * 3))
            .reshaped(1, 128, 3, 1, 1)
        let start = MLXArray([Float](repeating: 0.75, count: 128))
            .reshaped(1, 128, 1, 1, 1)
        let end = MLXArray([Float](repeating: -0.5, count: 128))
            .reshaped(1, 128, 1, 1, 1)
        let conditioning = applyLatentConditioning(
            baseLatent: video,
            conditionedLatent: start,
            frameIndex: 0,
            strength: 1,
            endConditionedLatent: end,
            endFrameIndex: -1,
            endStrength: 1
        )
        let audio = MLX.zeros([1, 8, 3, 16], dtype: .float32)
        let rope: LTXRope = (
            cos: MLX.ones([1], dtype: .float32),
            sin: MLX.zeros([1], dtype: .float32)
        )

        let result = try denoiseFrozenAudioVideoLoop(
            videoLatents: conditioning.latent,
            audioLatents: audio,
            videoRope: rope,
            audioRope: rope,
            videoCrossRope: rope,
            audioCrossRope: rope,
            positiveVideoContext: MLX.zeros([1, 1, 1], dtype: .float32),
            negativeVideoContext: nil,
            audioContext: MLX.zeros([1, 1, 1], dtype: .float32),
            transformer: FrozenAudioTestTransformer(),
            sigmas: [1, 0],
            videoConditioning: conditioning,
            guidance: nil
        )
        MLX.eval(result.video)

        XCTAssertEqual(result.video[0, 0, 0, 0, 0].item(Float.self), 0.75, accuracy: 1e-6)
        XCTAssertEqual(result.video[0, 0, 1, 0, 0].item(Float.self), 0.25, accuracy: 1e-6)
        XCTAssertEqual(result.video[0, 0, 2, 0, 0].item(Float.self), -0.5, accuracy: 1e-6)
    }

    func testAudioSegmentValidationRejectsUnsupportedChannelLayout() {
        let metadata = MediaAudioMetadata(
            sampleRate: 48_000,
            channelCount: 6,
            frameCount: 480_000,
            durationSeconds: 10
        )

        XCTAssertThrowsError(
            try validateLTXAudioSegment(metadata: metadata, startTime: 0, duration: 5)
        ) { error in
            guard case LTXUnifiedAVGeneratorError.unsupportedAudioChannels(6) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testAudioSegmentValidationRejectsShortSelectionWithoutPadding() {
        let metadata = MediaAudioMetadata(
            sampleRate: 48_000,
            channelCount: 2,
            frameCount: 240_000,
            durationSeconds: 5
        )

        XCTAssertThrowsError(
            try validateLTXAudioSegment(metadata: metadata, startTime: 1.25, duration: 4)
        ) { error in
            guard case LTXUnifiedAVGeneratorError.audioSegmentTooShort(let required, let available) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(required, 4, accuracy: 1e-9)
            XCTAssertEqual(available, 3.75, accuracy: 1e-9)
        }
    }

    private final class TinyTransformer: Module {
        @ModuleInfo(key: "patchify_proj") var patchifyProjection: Linear

        override init() {
            self._patchifyProjection.wrappedValue = Linear(2, 2, bias: false)
            super.init()
        }
    }

    func testLTX23StageOneSchedulerMatchesUpstreamFixture() {
        let actual = LTX2DiffusionScheduler.sigmas(steps: 30)
        let expected: [Float] = [
            1.000000000, 0.994957011, 0.989603040, 0.983908406, 0.977839528,
            0.971358259, 0.964421088, 0.956978157, 0.948972066, 0.940336383,
            0.930993805, 0.920853830, 0.909809819, 0.897735232, 0.884478755,
            0.869857931, 0.853650705, 0.835584072, 0.815318574, 0.792426769,
            0.766362747, 0.736418039, 0.701656238, 0.660813314, 0.612140591,
            0.553147857, 0.480163684, 0.387540810, 0.266120375, 0.100000000,
            0.000000000,
        ]

        XCTAssertEqual(actual.count, expected.count)
        for (actualSigma, expectedSigma) in zip(actual, expected) {
            XCTAssertEqual(actualSigma, expectedSigma, accuracy: 1e-6)
        }
    }

    func testSplitRoPEMatchesPinnedUpstreamFloat32Fixture() {
        let positions = MLXArray([
            Float(1.0 / 48.0), Float(1.0 / 48.0), Float(1.0 / 48.0), Float(1.0 / 48.0),
            16, 16, 48, 48,
            16, 48, 16, 48,
        ]).reshaped(1, 3, 4, 1)
        let intervals = MLX.concatenated([positions, positions], axis: 3)
        let rope = precomputeSplitRope(
            positions: intervals,
            dim: 4_096,
            theta: 10_000,
            maxPos: [20, 2_048, 2_048],
            numHeads: 32
        )
        let expectedCos: [Float] = [
            1, 1, 0.0032724838, 0.024541136, 0.024541136, -0.01807095,
            0.0034888473, 0.0034888473, -0.039696317, -0.017851403, -0.017851403,
            -0.06159747, -0.039474286, -0.039474286, -0.08376645, -0.061372474,
        ]
        let expectedSin: [Float] = [
            0, 0, -0.99999464, -0.9996989, -0.9996989, -0.9998367,
            -0.9999939, -0.9999939, -0.99921185, -0.9998407, -0.9998407,
            -0.99810106, -0.9992206, -0.9992206, -0.9964855, -0.99811494,
        ]
        let actualCos = rope.cos[0, 0, 0, 0..<expectedCos.count].asArray(Float.self)
        let actualSin = rope.sin[0, 0, 0, 0..<expectedSin.count].asArray(Float.self)

        XCTAssertEqual(actualCos.count, expectedCos.count)
        for (actual, expected) in zip(actualCos, expectedCos) {
            XCTAssertEqual(actual, expected, accuracy: 1e-6)
        }
        for (actual, expected) in zip(actualSin, expectedSin) {
            XCTAssertEqual(actual, expected, accuracy: 1e-6)
        }
        XCTAssertEqual(rope.cos.shape, [1, 32, 4, 64])
        XCTAssertEqual(rope.sin.shape, [1, 32, 4, 64])
    }

    func testGuidanceCombinesCFGSTGAndAudioIsolation() {
        let guidance = LTXAudioToVideoGuidance(
            classifierFreeScale: 3,
            spatioTemporalScale: 1,
            rescale: 0,
            audioToVideoScale: 3
        )
        let output = guidance.combine(
            conditioned: MLXArray([Float(1), 2]),
            negativeText: MLXArray([Float(0), 1]),
            perturbed: MLXArray([Float(0.5), 1.5]),
            isolatedAudio: MLXArray([Float(-1), 0])
        )
        MLX.eval(output)

        XCTAssertEqual(output.asArray(Float.self), [7.5, 8.5])
    }

    func testMultiModalGuidanceCombinesFullAudioGuidance() {
        let guidance = LTXMultiModalGuidance(
            classifierFreeScale: 7,
            spatioTemporalScale: 1,
            rescale: 0,
            modalityScale: 3
        )
        let output = guidance.combine(
            conditioned: MLXArray([Float(1), 2]),
            negativeText: MLXArray([Float(0), 1]),
            perturbed: MLXArray([Float(0.5), 1.5]),
            isolatedModality: MLXArray([Float(-1), 0])
        )
        MLX.eval(output)

        XCTAssertEqual(output.asArray(Float.self), [11.5, 12.5])
    }

    func testJointSpatioTemporalPerturbationTargetsBothStreams() {
        let perturbation = LTXAudioToVideoPerturbation.spatioTemporal(
            videoBlocks: [28],
            audioBlocks: [12, 28]
        )

        XCTAssertEqual(perturbation.skippedVideoSelfAttentionBlocks, [28])
        XCTAssertEqual(perturbation.skippedAudioSelfAttentionBlocks, [12, 28])
        XCTAssertFalse(perturbation.skipsAudioToVideoCrossAttention)
        XCTAssertFalse(perturbation.skipsVideoToAudioCrossAttention)
    }

    func testLTXAudioMelProcessorMatchesZeroSignalContract() {
        let processor = LTXAudioMelProcessor()
        let output = processor.extract(channels: [
            [Float](repeating: 0, count: 16_000),
            [Float](repeating: 0, count: 16_000),
        ])
        MLX.eval(output)

        XCTAssertEqual(output.shape, [1, 2, 101, 64])
        let expected = log(Float(1e-5))
        XCTAssertTrue(output.asArray(Float.self).allSatisfy { abs($0 - expected) < 1e-6 })
    }

    func testLTXAudioMelProcessorMatchesIndependentSlaneyReferenceFixture() {
        let sampleRate = Float(LTXAudioMelProcessor.sampleRate)
        var samples = [Float](repeating: 0, count: 1_600)
        for index in samples.indices {
            samples[index] = 0.2 * sin(2 * .pi * 440 * Float(index) / sampleRate)
        }
        samples[777] += 0.35

        let output = LTXAudioMelProcessor().extract(channels: [samples])
        MLX.eval(output)
        let values = output.asArray(Float.self)
        let frameStride = LTXAudioMelProcessor.melBinCount
        let fixture: [(frame: Int, mel: Int, expected: Float)] = [
            (0, 0, -1.898389565),
            (0, 10, -0.692807314),
            (1, 3, -1.961315376),
            (3, 7, -4.179414677),
            (4, 8, 0.176228990),
            (5, 9, 0.074844644),
            (7, 20, -5.210179510),
            (10, 63, -6.813462774),
        ]
        for point in fixture {
            XCTAssertEqual(
                values[point.frame * frameStride + point.mel],
                point.expected,
                accuracy: 5e-4,
                "Mismatch at frame \(point.frame), mel \(point.mel)"
            )
        }
    }

    func testLTXAudioMelProcessorProducesFiniteImpulseFeatures() {
        var impulse = [Float](repeating: 0, count: 16_000)
        impulse[8_000] = 1
        let output = LTXAudioMelProcessor().extract(channels: [impulse])
        MLX.eval(output)

        let values = output.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(values.max() ?? -.infinity, log(Float(1e-5)))
    }

    func testInstalledLTX23AudioEncoderCheckpointWhenAvailable() throws {
        let rootURL = ProcessInfo.processInfo.environment["MERERUN_TEST_LTX23_ROOT"]
            .map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? MereRunModelPaths.modelDir(ModelResolver.ModelID.ltxVideo23A2VMLX.rawValue)
        let weightsURL = rootURL
            .appendingPathComponent("audio_vae.safetensors", isDirectory: false)
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw XCTSkip(
                "Install video-ltx23-full-mlx or set MERERUN_TEST_LTX23_ROOT to exercise audio VAE weights."
            )
        }
        let metadata = try SafetensorsStreamingLoader.metadata(url: weightsURL)
        XCTAssertEqual(metadata.keys.filter { $0.hasPrefix("audio_vae.encoder.") }.count, 44)
        XCTAssertNotNil(metadata["audio_vae.per_channel_statistics._mean_of_means"])
        XCTAssertNotNil(metadata["audio_vae.per_channel_statistics._std_of_means"])

        let sampleRate = Float(LTXAudioMelProcessor.sampleRate)
        let waveform = (0..<16_000).map { index in
            Float(0.1) * sin(2 * .pi * 220 * Float(index) / sampleRate)
        }
        let spectrogram = LTXAudioMelProcessor().extract(channels: [waveform, waveform])
        let latent = try encodeLTX23AudioLatents(
            spectrogram: spectrogram,
            requiredFrameCount: 25,
            weightsURL: weightsURL,
            dtype: .bfloat16
        )
        let latent32 = latent.asType(.float32)
        MLX.eval(latent32)

        XCTAssertEqual(latent.shape, [1, 8, 25, 16])
        XCTAssertTrue(latent32.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testDistilledLoRAFusesMixedRankPairWithoutAlphaNormalization() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        let url = temp.appendingPathComponent("distilled.safetensors")
        try writeTinyLTXLoRA(to: url)

        let transformer = TinyTransformer()
        try transformer.update(
            parameters: ModuleParameters.unflattened([
                ("patchify_proj.weight", MLX.zeros([2, 2], dtype: .float32)),
            ]),
            verify: .none
        )
        let count = try LTXStreamingLoRAFuser.fuse(
            url: url,
            into: transformer,
            strength: 0.5,
            expectedPairCount: 1
        )
        let weight = try XCTUnwrap(
            Dictionary(uniqueKeysWithValues: transformer.parameters().flattened())["patchify_proj.weight"]
        )
        MLX.eval(weight)

        XCTAssertEqual(count, 1)
        XCTAssertEqual(weight.asArray(Float.self), [11.5, 17, 15.5, 23])
    }

    func testInstalledDistilledLoRAMatchesPinnedUpstreamFusionFixtures() throws {
        let rootURL = ProcessInfo.processInfo.environment["MERERUN_TEST_LTX23_ROOT"]
            .map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? MereRunModelPaths.modelDir(ModelResolver.ModelID.ltxVideo23A2VMLX.rawValue)
        let loraURL = rootURL.appendingPathComponent(
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            isDirectory: false
        )
        let transformerURL = rootURL.appendingPathComponent(
            "transformer-dev.safetensors",
            isDirectory: false
        )
        guard FileManager.default.fileExists(atPath: loraURL.path),
              FileManager.default.fileExists(atPath: transformerURL.path) else {
            throw XCTSkip(
                "Install video-ltx23-full-mlx or set MERERUN_TEST_LTX23_ROOT to exercise pinned LoRA parity."
            )
        }

        let metadata = try SafetensorsStreamingLoader.metadata(url: loraURL)
        let aMetadata = metadata.filter { $0.key.hasSuffix(".lora_A.weight") }
        let rankCounts = Dictionary(grouping: aMetadata.values, by: { $0.shape[0] })
            .mapValues(\.count)
        XCTAssertEqual(aMetadata.count, 1_660)
        XCTAssertEqual(rankCounts, [32: 288, 128: 4, 256: 8, 384: 1_360])
        for key in aMetadata.keys {
            let bKey = key.replacingOccurrences(of: ".lora_A.weight", with: ".lora_B.weight")
            XCTAssertNotNil(metadata[bKey], "Missing pair for \(key)")
        }

        struct Fixture {
            let loraPrefix: String
            let baseKey: String
            let shape: [Int]
            let firstValues: [Float]
        }
        let fixtures = [
            Fixture(
                loraPrefix: "diffusion_model.transformer_blocks.0.audio_attn1.to_gate_logits",
                baseKey: "transformer.transformer_blocks.0.audio_attn1.to_gate_logits.weight",
                shape: [32, 2_048],
                firstValues: [
                    -0.019287109375, 0.00909423828125, 0.03857421875, -0.005645751953125,
                    0.0024566650390625, -0.0032196044921875, -0.02490234375, 0.00537109375,
                    0.0035247802734375, -0.0034027099609375, 0.0005035400390625,
                    -0.021728515625, 0.003387451171875, 0.0037078857421875,
                    -0.0091552734375, 0.004302978515625,
                ]
            ),
            Fixture(
                loraPrefix: "diffusion_model.audio_patchify_proj",
                baseKey: "transformer.audio_patchify_proj.weight",
                shape: [2_048, 128],
                firstValues: [
                    0.01495361328125, -0.003326416015625, 0.0067138671875, 0.020751953125,
                    -0.00537109375, -0.0196533203125, 0.004730224609375, -0.01123046875,
                    0.00104522705078125, 0.007049560546875, 0.016845703125,
                    0.01019287109375, -0.01068115234375, -0.0247802734375,
                    0.0115966796875, -0.00732421875,
                ]
            ),
        ]
        let loraKeys = Set(fixtures.flatMap { fixture in
            [
                fixture.loraPrefix + ".lora_A.weight",
                fixture.loraPrefix + ".lora_B.weight",
            ]
        })
        let baseKeys = Set(fixtures.map(\.baseKey))
        let loraWeights = try SafetensorsStreamingLoader.loadArrays(
            url: loraURL,
            where: { loraKeys.contains($0) }
        )
        let baseWeights = try SafetensorsStreamingLoader.loadArrays(
            url: transformerURL,
            where: { baseKeys.contains($0) }
        )

        for fixture in fixtures {
            let a = try XCTUnwrap(loraWeights[fixture.loraPrefix + ".lora_A.weight"])
            let b = try XCTUnwrap(loraWeights[fixture.loraPrefix + ".lora_B.weight"])
            let base = try XCTUnwrap(baseWeights[fixture.baseKey])
            let fused = try LTXStreamingLoRAFuser.fusedWeight(
                targetKey: fixture.baseKey,
                currentWeight: base,
                a: a,
                b: b,
                strength: 1
            )
            let actual = fused.flattened()[0..<fixture.firstValues.count]
                .asType(.float32)
                .asArray(Float.self)

            XCTAssertEqual(fused.dtype, .bfloat16)
            XCTAssertEqual(fused.shape, fixture.shape)
            XCTAssertEqual(actual.count, fixture.firstValues.count)
            for (value, expected) in zip(actual, fixture.firstValues) {
                XCTAssertEqual(value, expected, accuracy: 0)
            }
        }
    }

    private func writeTinyLTXLoRA(to url: URL) throws {
        let header = """
        {
          "__metadata__": {"lora_alpha": "384"},
          "diffusion_model.patchify_proj.lora_A.weight": {
            "dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]
          },
          "diffusion_model.patchify_proj.lora_B.weight": {
            "dtype": "F32", "shape": [2, 2], "data_offsets": [16, 32]
          }
        }
        """
        let headerData = Data(header.utf8)
        var headerSize = UInt64(headerData.count).littleEndian
        var data = Data()
        withUnsafeBytes(of: &headerSize) { data.append(contentsOf: $0) }
        data.append(headerData)
        let values: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        values.withUnsafeBytes { data.append(contentsOf: $0) }
        try data.write(to: url)
    }
}

private final class FrozenAudioTestTransformer: Module, LTXUnifiedAVTransformerRuntime {
    func forward(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        timestep _: MLXArray,
        videoTimesteps _: MLXArray?,
        audioTimesteps _: MLXArray?,
        videoContext _: MLXArray,
        audioContext _: MLXArray,
        videoRope _: LTXRope,
        audioRope _: LTXRope,
        videoCrossRope _: LTXRope,
        audioCrossRope _: LTXRope,
        audioSigma _: MLXArray,
        perturbation _: LTXAudioToVideoPerturbation
    ) -> (videoVelocity: MLXArray, audioVelocity: MLXArray) {
        (
            MLX.zeros(videoLatent.shape, dtype: videoLatent.dtype),
            MLX.zeros(audioLatent.shape, dtype: audioLatent.dtype)
        )
    }
}
