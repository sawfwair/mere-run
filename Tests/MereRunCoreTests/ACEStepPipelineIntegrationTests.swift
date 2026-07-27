import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class ACEStepPipelineIntegrationTests: MereRunCoreTestCase {
    func testInstalledVAERoundTripPreservesStructuredAudio() throws {
        let env = ProcessInfo.processInfo.environment
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT to run the real VAE round-trip test.")
        }

        let resources = OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot))
        let config = try OobleckVAECheckpointLoader.loadConfig(resources: resources)
        let hopLength = config.downsamplingRatios.reduce(1, *)
        let sampleCount = config.samplingRate
        XCTAssertEqual(sampleCount % hopLength, 0)

        var sourceSamples: [Float] = []
        sourceSamples.reserveCapacity(sampleCount * config.audioChannels)
        for sampleIndex in 0..<sampleCount {
            let time = Double(sampleIndex) / Double(config.samplingRate)
            let envelope = min(1, time * 20) * min(1, (1 - time) * 20)
            let pulse = (sampleIndex % (config.samplingRate / 8)) < 120 ? 0.12 : 0
            let left = envelope * (
                0.22 * sin(2 * Double.pi * 220 * time)
                    + 0.12 * sin(2 * Double.pi * 440 * time)
                    + pulse
            )
            let right = envelope * (
                0.20 * sin(2 * Double.pi * 330 * time)
                    + 0.10 * sin(2 * Double.pi * 660 * time)
                    + pulse
            )
            sourceSamples.append(Float(left))
            sourceSamples.append(Float(right))
        }

        let source = MLXArray(sourceSamples, [1, sampleCount, config.audioChannels]).asType(.float32)
        let vae = try OobleckVAECheckpointLoader.loadVAE(resources: resources, dtype: .float32)
        let latents = vae.encode(source, sample: false).asType(.float32)
        let reconstructed = vae.decode(latents).asType(.float32)
        MLX.eval(latents, reconstructed)

        XCTAssertEqual(latents.shape, [1, sampleCount / hopLength, config.decoderInputChannels])
        XCTAssertEqual(reconstructed.shape, source.shape)

        let reconstructedSamples = reconstructed.asArray(Float.self)
        let correlation = normalizedCorrelation(sourceSamples, reconstructedSamples)
        let normalizedError = normalizedMeanSquaredError(sourceSamples, reconstructedSamples)

        XCTAssertGreaterThan(
            correlation,
            0.2,
            "VAE round-trip lost source structure: correlation=\(correlation), normalizedMSE=\(normalizedError)"
        )
        XCTAssertLessThan(
            normalizedError,
            4,
            "VAE round-trip behaves like unrelated noise: correlation=\(correlation), normalizedMSE=\(normalizedError)"
        )
    }

    func testNonTurboXLSFTEndToEndWithAPGAndHeun() throws {
        let env = ProcessInfo.processInfo.environment
        guard let decoderRoot = env["MERERUN_TEST_ACESTEP_XL_SFT_ROOT"],
              !decoderRoot.isEmpty
        else {
            throw XCTSkip(
                "Set MERERUN_TEST_ACESTEP_XL_SFT_ROOT to run the non-Turbo XL-SFT test."
            )
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT to run this test.")
        }

        let pipeline = try ACEStepPipeline(
            decoderResources: ACEStepResources(
                rootURL: URL(fileURLWithPath: decoderRoot)
            ),
            vaeResources: OobleckVAEResources(
                rootURL: URL(fileURLWithPath: vaeRoot)
            )
        )

        XCTAssertEqual(pipeline.checkpointVariant, .xlSFT)
        let audio = try pipeline.generatePromptToAudio(
            caption: "tight electronic drum groove, instrumental",
            lyrics: "",
            config: .init(
                durationSeconds: 0.4,
                fixNFE: 2,
                shift: 1,
                samplerMode: .heun,
                guidanceScale: 2,
                guidanceMode: .apg,
                velocityNormThreshold: 2,
                velocityEMAFactor: 0.1,
                useTiledVaeDecode: false,
                seed: 42
            )
        )
        MLX.eval(audio)

        XCTAssertEqual(audio.shape, [1, 19_200, 2])
        let maxAbs = MLX.max(MLX.abs(audio.asType(.float32))).item(Float.self)
        XCTAssertTrue(maxAbs.isFinite)
        XCTAssertGreaterThan(maxAbs, 0)
    }

    func testRepaintTurboPreservesSourceWaveformOutsideRequestedRange() throws {
        let env = ProcessInfo.processInfo.environment
        guard let turboRoot = env["MERERUN_TEST_ACESTEP_TURBO_ROOT"], !turboRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TURBO_ROOT to run this test.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT to run this test.")
        }
        guard let textRoot = env["MERERUN_TEST_ACESTEP_TEXT_ROOT"], !textRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TEXT_ROOT to run this test.")
        }

        let pipeline = try ACEStepPipeline(
            decoderResources: ACEStepResources(rootURL: URL(fileURLWithPath: turboRoot)),
            vaeResources: OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot)),
            textEncoderResources: ACEStep5HzLMResources(rootURL: URL(fileURLWithPath: textRoot))
        )

        let durationSeconds: Float = 0.4
        let sampleCount = Int(durationSeconds * 48_000)
        var samples: [Float] = []
        samples.reserveCapacity(sampleCount * 2)
        for sample in 0..<sampleCount {
            let value = Float(sin(2 * Double.pi * 220 * Double(sample) / 48_000)) * 0.1
            samples.append(value)
            samples.append(value)
        }
        let source = MLXArray(samples, [1, sampleCount, 2])
        let output = try pipeline.generatePromptToAudio(
            caption: "replace the middle with a bright synth fill",
            lyrics: "",
            config: .init(
                durationSeconds: durationSeconds,
                fixNFE: 2,
                useTiledVaeDecode: false,
                seed: 404
            ),
            sourceAudio48kHz: source,
            task: .repaint,
            repaintConfiguration: .init(
                startSeconds: 0.12,
                endSeconds: 0.28,
                mode: .conservative
            )
        )
        MLX.eval(output)

        XCTAssertEqual(output.shape, source.shape)
        let outputValues = output.asArray(Float.self)
        let preservedHeadSamples = Int(0.05 * 48_000) * 2
        for index in 0..<preservedHeadSamples {
            XCTAssertEqual(outputValues[index], samples[index], accuracy: 0.000_001)
        }
        let preservedTailStart = Int(0.35 * 48_000) * 2
        for index in preservedTailStart..<samples.count {
            XCTAssertEqual(outputValues[index], samples[index], accuracy: 0.000_001)
        }
    }

    func testUnconditionalTurboEndToEnd() throws {
        let env = ProcessInfo.processInfo.environment
        guard let turboRoot = env["MERERUN_TEST_ACESTEP_TURBO_ROOT"], !turboRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TURBO_ROOT=/path/to/ACE-Step-1.5/checkpoints/acestep-v15-turbo to run this test.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/ACE-Step-1.5/checkpoints/vae to run this test.")
        }

        let decoderResources = ACEStepResources(rootURL: URL(fileURLWithPath: turboRoot))
        let vaeResources = OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot))

        let vaeConfig = try OobleckVAECheckpointLoader.loadConfig(resources: vaeResources)
        let factor = vaeConfig.downsamplingRatios.reduce(1, *)

        let pipeline = try ACEStepPipeline(decoderResources: decoderResources, vaeResources: vaeResources)

        let durationSeconds: Float = 0.2
        let expectedLatentFrames = Int((Double(durationSeconds) * 25.0).rounded())
        let expectedSamples = expectedLatentFrames * factor

        let audio = pipeline.generateUnconditional(.init(durationSeconds: durationSeconds, seed: 123))

        XCTAssertEqual(audio.dim(0), 1)
        XCTAssertEqual(audio.dim(1), expectedSamples)
        XCTAssertEqual(audio.dim(2), vaeConfig.audioChannels)

        let maxAbs = MLX.max(MLX.abs(audio.asType(.float32))).item(Float.self)
        XCTAssertFalse(maxAbs.isNaN)
        XCTAssertFalse(maxAbs.isInfinite)
        XCTAssertGreaterThan(maxAbs, 0)
    }


    func testPromptTurboEndToEndWithoutLM() throws {
        let env = ProcessInfo.processInfo.environment
        guard let turboRoot = env["MERERUN_TEST_ACESTEP_TURBO_ROOT"], !turboRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TURBO_ROOT=/path/to/ACE-Step-1.5/checkpoints/acestep-v15-turbo to run this test.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/ACE-Step-1.5/checkpoints/vae to run this test.")
        }

        let decoderResources = ACEStepResources(rootURL: URL(fileURLWithPath: turboRoot))
        let vaeResources = OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot))

        let vaeConfig = try OobleckVAECheckpointLoader.loadConfig(resources: vaeResources)
        let factor = vaeConfig.downsamplingRatios.reduce(1, *)

        let pipeline = try ACEStepPipeline(decoderResources: decoderResources, vaeResources: vaeResources)

        let durationSeconds: Float = 0.2
        let expectedLatentFrames = Int((Double(durationSeconds) * 25.0).rounded())
        let expectedSamples = expectedLatentFrames * factor

        let audio = try pipeline.generatePromptToAudio(
            caption: "upbeat electronic groove.",
            lyrics: "[verse]\nla la la",
            config: .init(durationSeconds: durationSeconds, seed: 11),
            lmUserMetadata: .init(bpm: "120", duration: "10", keyscale: "C major", timesignature: "4")
        )

        XCTAssertEqual(audio.dim(0), 1)
        XCTAssertEqual(audio.dim(1), expectedSamples)
        XCTAssertEqual(audio.dim(2), vaeConfig.audioChannels)

        let maxAbs = MLX.max(MLX.abs(audio.asType(.float32))).item(Float.self)
        XCTAssertFalse(maxAbs.isNaN)
        XCTAssertFalse(maxAbs.isInfinite)
        XCTAssertGreaterThan(maxAbs, 0)
    }

    func testConstrainedLMTurboEndToEnd() throws {
        let env = ProcessInfo.processInfo.environment
        guard let turboRoot = env["MERERUN_TEST_ACESTEP_TURBO_ROOT"], !turboRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TURBO_ROOT=/path/to/ACE-Step-1.5/checkpoints/acestep-v15-turbo to run this test.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/ACE-Step-1.5/checkpoints/vae to run this test.")
        }
        guard let lmRoot = env["MERERUN_TEST_ACESTEP_5HZ_ROOT"], !lmRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_5HZ_ROOT=/path/to/ACE-Step-1.5/checkpoints/acestep-5Hz-lm-1.7B to run this test.")
        }

        let decoderResources = ACEStepResources(rootURL: URL(fileURLWithPath: turboRoot))
        let vaeResources = OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot))
        let lmResources = ACEStep5HzLMResources(rootURL: URL(fileURLWithPath: lmRoot))
        let textResources = env["MERERUN_TEST_ACESTEP_TEXT_ROOT"].flatMap {
            $0.isEmpty ? nil : ACEStep5HzLMResources(rootURL: URL(fileURLWithPath: $0))
        }

        let vaeConfig = try OobleckVAECheckpointLoader.loadConfig(resources: vaeResources)
        let factor = vaeConfig.downsamplingRatios.reduce(1, *)

        let pipeline = try ACEStepPipeline(
            decoderResources: decoderResources,
            vaeResources: vaeResources,
            lmResources: lmResources,
            textEncoderResources: textResources
        )

        let durationSeconds: Float = 0.2
        let expectedLatentFrames = Int((Double(durationSeconds) * 25.0).rounded())
        let expectedSamples = expectedLatentFrames * factor

        let lmResult = try pipeline.generatePromptToAudioWithLM(
            caption: "upbeat electronic groove.",
            lyrics: "[verse]\nla la la",
            config: .init(durationSeconds: durationSeconds, seed: 7),
            lmConfig: .init(maxNewTokens: 128, temperature: 0.0, topP: 1.0, repetitionPenalty: nil),
            lmUserMetadata: .init(
                bpm: "120",
                duration: "10",
                keyscale: "C major",
                timesignature: "4"
            )
        )

        XCTAssertFalse(lmResult.lmResult.audioCodeValues.isEmpty)
        XCTAssertEqual(lmResult.audio.dim(0), 1)
        XCTAssertEqual(lmResult.audio.dim(1), expectedSamples)
        XCTAssertEqual(lmResult.audio.dim(2), vaeConfig.audioChannels)

        let maxAbs = MLX.max(MLX.abs(lmResult.audio.asType(.float32))).item(Float.self)
        XCTAssertFalse(maxAbs.isNaN)
        XCTAssertFalse(maxAbs.isInfinite)
        XCTAssertGreaterThan(maxAbs, 0)
    }

    func testPreparePromptConditionInputsUsesConfiguredTextEncoder() throws {
        let env = ProcessInfo.processInfo.environment
        guard let turboRoot = env["MERERUN_TEST_ACESTEP_TURBO_ROOT"], !turboRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TURBO_ROOT=/path/to/ACE-Step-1.5/checkpoints/acestep-v15-turbo to run this test.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/ACE-Step-1.5/checkpoints/vae to run this test.")
        }
        guard let textRoot = env["MERERUN_TEST_ACESTEP_TEXT_ROOT"], !textRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TEXT_ROOT=/path/to/ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B to run this test.")
        }

        let decoderResources = ACEStepResources(rootURL: URL(fileURLWithPath: turboRoot))
        let vaeResources = OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot))
        let textResources = ACEStep5HzLMResources(rootURL: URL(fileURLWithPath: textRoot))
        let config = try ACEStepCheckpointLoader.loadConfig(resources: decoderResources)

        let pipeline = try ACEStepPipeline(
            decoderResources: decoderResources,
            vaeResources: vaeResources,
            textEncoderResources: textResources
        )

        let latentFrames = 12
        let srcLatents = MLXArray.zeros([1, latentFrames, config.audioAcousticHiddenDim], dtype: .bfloat16)
        let chunkChannels = config.inChannels - (2 * config.audioAcousticHiddenDim)
        XCTAssertGreaterThan(chunkChannels, 0)

        let inputs = try pipeline.preparePromptConditionInputs(
            caption: "slow ambient piano intro",
            lyrics: "[verse]\nhello from tests",
            srcLatents: srcLatents,
            chunkChannels: chunkChannels,
            lmUserMetadata: .init(bpm: "80", duration: "12", keyscale: "C minor", language: "en", timesignature: "4"),
            vocalLanguage: "en",
            instruction: "Fill the audio semantic mask based on the given conditions",
            task: .cover
        )

        XCTAssertGreaterThan(inputs.textHiddenStates.dim(1), 1)
        XCTAssertEqual(inputs.textHiddenStates.dim(2), config.textHiddenDim)
        XCTAssertGreaterThan(inputs.lyricHiddenStates.dim(1), 1)
        XCTAssertEqual(inputs.lyricHiddenStates.dim(2), config.textHiddenDim)
        XCTAssertEqual(inputs.referAudioAcousticHiddenStatesPacked.dim(0), 1)
        XCTAssertEqual(inputs.referAudioAcousticHiddenStatesPacked.dim(2), config.audioAcousticHiddenDim)
        XCTAssertEqual(inputs.chunkMasks.dim(0), 1)
        XCTAssertEqual(inputs.chunkMasks.dim(1), latentFrames)
        XCTAssertEqual(inputs.chunkMasks.dim(2), chunkChannels)
    }

    func testPreparePromptConditionInputsBuildsNonCoverTextConditionWhenCoverStrengthBelowOne() throws {
        let env = ProcessInfo.processInfo.environment
        guard let turboRoot = env["MERERUN_TEST_ACESTEP_TURBO_ROOT"], !turboRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TURBO_ROOT=/path/to/ACE-Step-1.5/checkpoints/acestep-v15-turbo to run this test.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/ACE-Step-1.5/checkpoints/vae to run this test.")
        }
        guard let textRoot = env["MERERUN_TEST_ACESTEP_TEXT_ROOT"], !textRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TEXT_ROOT=/path/to/ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B to run this test.")
        }

        let decoderResources = ACEStepResources(rootURL: URL(fileURLWithPath: turboRoot))
        let vaeResources = OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot))
        let textResources = ACEStep5HzLMResources(rootURL: URL(fileURLWithPath: textRoot))
        let config = try ACEStepCheckpointLoader.loadConfig(resources: decoderResources)

        let pipeline = try ACEStepPipeline(
            decoderResources: decoderResources,
            vaeResources: vaeResources,
            textEncoderResources: textResources
        )

        let latentFrames = 12
        let srcLatents = MLXArray.zeros([1, latentFrames, config.audioAcousticHiddenDim], dtype: .bfloat16)
        let chunkChannels = config.inChannels - (2 * config.audioAcousticHiddenDim)
        XCTAssertGreaterThan(chunkChannels, 0)

        let fullCoverInputs = try pipeline.preparePromptConditionInputs(
            caption: "cinematic synth intro",
            lyrics: "[verse]\nhello",
            srcLatents: srcLatents,
            chunkChannels: chunkChannels,
            lmUserMetadata: .init(bpm: "90", duration: "12", keyscale: "A minor", language: "en", timesignature: "4"),
            audioCoverStrength: 1.0,
            vocalLanguage: "en",
            instruction: "Use this custom instruction for cover conditioning",
            task: .cover
        )
        XCTAssertNil(fullCoverInputs.nonCoverTextHiddenStates)
        XCTAssertNil(fullCoverInputs.nonCoverTextAttentionMask)

        let partialCoverInputs = try pipeline.preparePromptConditionInputs(
            caption: "cinematic synth intro",
            lyrics: "[verse]\nhello",
            srcLatents: srcLatents,
            chunkChannels: chunkChannels,
            lmUserMetadata: .init(bpm: "90", duration: "12", keyscale: "A minor", language: "en", timesignature: "4"),
            audioCoverStrength: 0.5,
            vocalLanguage: "en",
            instruction: "Use this custom instruction for cover conditioning",
            task: .cover
        )

        guard
            let nonCoverHiddenStates = partialCoverInputs.nonCoverTextHiddenStates,
            let nonCoverAttentionMask = partialCoverInputs.nonCoverTextAttentionMask
        else {
            XCTFail("Expected non-cover text conditioning tensors when audioCoverStrength < 1.")
            return
        }

        XCTAssertGreaterThan(nonCoverHiddenStates.dim(1), 1)
        XCTAssertEqual(nonCoverHiddenStates.dim(2), config.textHiddenDim)
        XCTAssertEqual(nonCoverAttentionMask.dim(0), 1)
        XCTAssertGreaterThan(nonCoverAttentionMask.dim(1), 1)

        let compareLength = min(partialCoverInputs.textHiddenStates.dim(1), nonCoverHiddenStates.dim(1))
        let coverSlice = partialCoverInputs.textHiddenStates[0..., 0..<compareLength, 0...].asType(.float32)
        let nonCoverSlice = nonCoverHiddenStates[0..., 0..<compareLength, 0...].asType(.float32)
        let maxDiff = MLX.max(MLX.abs(coverSlice - nonCoverSlice)).item(Float.self)
        XCTAssertGreaterThan(maxDiff, 1e-4)

        let nonCoverPrepared = try XCTUnwrap(pipeline.prepareNonCoverConditionIfNeeded(conditionInputs: partialCoverInputs))
        let contextLatents = nonCoverPrepared.contextLatents.asType(.float32)
        XCTAssertEqual(contextLatents.dim(1), latentFrames)
        XCTAssertEqual(contextLatents.dim(2), config.inChannels - config.audioAcousticHiddenDim)

        let expectedSilence = pipeline.defaultSourceLatents(targetFrames: latentFrames)
            .asType(partialCoverInputs.srcLatents.dtype)
            .asType(.float32)
        let nonCoverSource = contextLatents[0..., 0..<latentFrames, 0..<config.audioAcousticHiddenDim]
        let silenceDiff = MLX.max(MLX.abs(nonCoverSource - expectedSilence)).item(Float.self)
        XCTAssertEqual(silenceDiff, 0, accuracy: 1e-4)
        XCTAssertGreaterThan(MLX.sum(MLX.abs(nonCoverSource)).item(Float.self), 0)

        let chunkStart = config.audioAcousticHiddenDim
        let nonCoverChunks = contextLatents[0..., 0..<latentFrames, chunkStart..<contextLatents.dim(2)]
        let chunkDiff = MLX.max(MLX.abs(nonCoverChunks - MLXArray(Float(1.0)))).item(Float.self)
        XCTAssertEqual(chunkDiff, 0, accuracy: 1e-4)
    }

    func testPreparePromptConditionInputsUsesReferenceTimbreLatentsWhenProvided() throws {
        let env = ProcessInfo.processInfo.environment
        guard let turboRoot = env["MERERUN_TEST_ACESTEP_TURBO_ROOT"], !turboRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TURBO_ROOT=/path/to/ACE-Step-1.5/checkpoints/acestep-v15-turbo to run this test.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/ACE-Step-1.5/checkpoints/vae to run this test.")
        }

        let decoderResources = ACEStepResources(rootURL: URL(fileURLWithPath: turboRoot))
        let vaeResources = OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot))
        let config = try ACEStepCheckpointLoader.loadConfig(resources: decoderResources)
        let pipeline = try ACEStepPipeline(decoderResources: decoderResources, vaeResources: vaeResources)

        let latentFrames = 12
        let srcLatents = MLXArray.zeros([1, latentFrames, config.audioAcousticHiddenDim], dtype: .bfloat16)
        let chunkChannels = config.inChannels - (2 * config.audioAcousticHiddenDim)
        XCTAssertGreaterThan(chunkChannels, 0)

        let referenceA = MLXArray.ones([1, 6, config.audioAcousticHiddenDim], dtype: .float32)
        let referenceB = MLXArray.ones([6, config.audioAcousticHiddenDim], dtype: .float32) * MLXArray(2.0)

        let inputs = try pipeline.preparePromptConditionInputs(
            caption: "test caption",
            lyrics: "test lyrics",
            srcLatents: srcLatents,
            chunkChannels: chunkChannels,
            lmUserMetadata: .init(),
            referenceTimbreLatents25Hz: [referenceA, referenceB],
            vocalLanguage: "en",
            instruction: "Fill the audio semantic mask based on the given conditions",
            task: .cover
        )

        XCTAssertEqual(inputs.referAudioAcousticHiddenStatesPacked.dim(0), 2)
        XCTAssertEqual(inputs.referAudioAcousticHiddenStatesPacked.dim(1), latentFrames)
        XCTAssertEqual(inputs.referAudioAcousticHiddenStatesPacked.dim(2), config.audioAcousticHiddenDim)
        XCTAssertEqual(inputs.referAudioOrderMask.dim(0), 2)
        XCTAssertEqual(inputs.referAudioOrderMask[0].item(Int32.self), 0)
        XCTAssertEqual(inputs.referAudioOrderMask[1].item(Int32.self), 0)

        let firstRefValue = inputs.referAudioAcousticHiddenStatesPacked[0, 0, 0].item(Float.self)
        let secondRefValue = inputs.referAudioAcousticHiddenStatesPacked[1, 0, 0].item(Float.self)
        XCTAssertGreaterThan(firstRefValue, 0.9)
        XCTAssertGreaterThan(secondRefValue, 1.9)

        let firstRefPadding = inputs.referAudioAcousticHiddenStatesPacked[0, latentFrames - 1, 0].item(Float.self)
        let secondRefPadding = inputs.referAudioAcousticHiddenStatesPacked[1, latentFrames - 1, 0].item(Float.self)
        XCTAssertEqual(firstRefPadding, 0, accuracy: 1e-4)
        XCTAssertEqual(secondRefPadding, 0, accuracy: 1e-4)
    }

    func testPreparePromptConditionInputsUsesReferenceTimbreAudioWhenProvided() throws {
        let env = ProcessInfo.processInfo.environment
        guard let turboRoot = env["MERERUN_TEST_ACESTEP_TURBO_ROOT"], !turboRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TURBO_ROOT=/path/to/ACE-Step-1.5/checkpoints/acestep-v15-turbo to run this test.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/ACE-Step-1.5/checkpoints/vae to run this test.")
        }

        let decoderResources = ACEStepResources(rootURL: URL(fileURLWithPath: turboRoot))
        let vaeResources = OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot))
        let config = try ACEStepCheckpointLoader.loadConfig(resources: decoderResources)
        let pipeline = try ACEStepPipeline(decoderResources: decoderResources, vaeResources: vaeResources)

        let latentFrames = 12
        let srcLatents = MLXArray.zeros([1, latentFrames, config.audioAcousticHiddenDim], dtype: .bfloat16)
        let chunkChannels = config.inChannels - (2 * config.audioAcousticHiddenDim)
        XCTAssertGreaterThan(chunkChannels, 0)

        let monoWave = MLXArray.zeros([4_800], dtype: .float32)
        let stereoWave = MLXArray.ones([4_800, 2], dtype: .float32) * MLXArray(0.1)
        let channelFirstStereoWave = stereoWave.transposed(1, 0)

        let inputs = try pipeline.preparePromptConditionInputs(
            caption: "test caption",
            lyrics: "test lyrics",
            srcLatents: srcLatents,
            chunkChannels: chunkChannels,
            lmUserMetadata: .init(),
            referenceTimbreAudio48kHz: [monoWave, channelFirstStereoWave],
            vocalLanguage: "en",
            instruction: "Fill the audio semantic mask based on the given conditions",
            task: .cover
        )

        XCTAssertEqual(inputs.referAudioAcousticHiddenStatesPacked.dim(0), 2)
        XCTAssertEqual(inputs.referAudioAcousticHiddenStatesPacked.dim(1), latentFrames)
        XCTAssertEqual(inputs.referAudioAcousticHiddenStatesPacked.dim(2), config.audioAcousticHiddenDim)
        XCTAssertEqual(inputs.referAudioOrderMask.dim(0), 2)
        XCTAssertEqual(inputs.referAudioOrderMask[0].item(Int32.self), 0)
        XCTAssertEqual(inputs.referAudioOrderMask[1].item(Int32.self), 0)

        let maxAbs = MLX.max(MLX.abs(inputs.referAudioAcousticHiddenStatesPacked.asType(.float32))).item(Float.self)
        XCTAssertFalse(maxAbs.isNaN)
        XCTAssertFalse(maxAbs.isInfinite)
        XCTAssertGreaterThan(maxAbs, 0)

        let firstRefPadding = inputs.referAudioAcousticHiddenStatesPacked[0, latentFrames - 1, 0].item(Float.self)
        let secondRefPadding = inputs.referAudioAcousticHiddenStatesPacked[1, latentFrames - 1, 0].item(Float.self)
        XCTAssertEqual(firstRefPadding, 0, accuracy: 1e-4)
        XCTAssertEqual(secondRefPadding, 0, accuracy: 1e-4)
    }

    func testPromptTurboEndToEndWithAudioCoverStrength() throws {
        let env = ProcessInfo.processInfo.environment
        guard let turboRoot = env["MERERUN_TEST_ACESTEP_TURBO_ROOT"], !turboRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TURBO_ROOT=/path/to/ACE-Step-1.5/checkpoints/acestep-v15-turbo to run this test.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/ACE-Step-1.5/checkpoints/vae to run this test.")
        }
        guard let textRoot = env["MERERUN_TEST_ACESTEP_TEXT_ROOT"], !textRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TEXT_ROOT=/path/to/ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B to run this test.")
        }

        let decoderResources = ACEStepResources(rootURL: URL(fileURLWithPath: turboRoot))
        let vaeResources = OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot))
        let textResources = ACEStep5HzLMResources(rootURL: URL(fileURLWithPath: textRoot))

        let vaeConfig = try OobleckVAECheckpointLoader.loadConfig(resources: vaeResources)
        let factor = vaeConfig.downsamplingRatios.reduce(1, *)

        let pipeline = try ACEStepPipeline(
            decoderResources: decoderResources,
            vaeResources: vaeResources,
            textEncoderResources: textResources
        )

        let durationSeconds: Float = 0.2
        let expectedLatentFrames = Int((Double(durationSeconds) * 25.0).rounded())
        let expectedSamples = expectedLatentFrames * factor

        let audio = try pipeline.generatePromptToAudio(
            caption: "bright synth-pop groove with layered arps",
            lyrics: "[verse]\ncity lights and moving trains",
            config: .init(durationSeconds: durationSeconds, seed: 77),
            lmUserMetadata: .init(bpm: "118", duration: "12", keyscale: "D major", language: "en", timesignature: "4"),
            audioCoverStrength: 0.5,
            vocalLanguage: "en",
            instruction: "Strong cover conditioning instruction",
            task: .cover
        )

        XCTAssertEqual(audio.dim(0), 1)
        XCTAssertEqual(audio.dim(1), expectedSamples)
        XCTAssertEqual(audio.dim(2), vaeConfig.audioChannels)

        let maxAbs = MLX.max(MLX.abs(audio.asType(.float32))).item(Float.self)
        XCTAssertFalse(maxAbs.isNaN)
        XCTAssertFalse(maxAbs.isInfinite)
        XCTAssertGreaterThan(maxAbs, 0)
    }

    func testPromptTurboEndToEndWithSourceAudioCover() throws {
        let env = ProcessInfo.processInfo.environment
        guard let turboRoot = env["MERERUN_TEST_ACESTEP_TURBO_ROOT"], !turboRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TURBO_ROOT=/path/to/ACE-Step-1.5/checkpoints/acestep-v15-turbo to run this test.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/ACE-Step-1.5/checkpoints/vae to run this test.")
        }

        let decoderResources = ACEStepResources(rootURL: URL(fileURLWithPath: turboRoot))
        let vaeResources = OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot))
        let vaeConfig = try OobleckVAECheckpointLoader.loadConfig(resources: vaeResources)
        let factor = vaeConfig.downsamplingRatios.reduce(1, *)

        let pipeline = try ACEStepPipeline(decoderResources: decoderResources, vaeResources: vaeResources)

        let durationSeconds: Float = 0.2
        let sourceSamples = max(1, Int(durationSeconds * 48_000))
        let left = (0..<sourceSamples).map { index in
            Float(sin((Double(index) / 48_000.0) * 2.0 * Double.pi * 220.0) * 0.1)
        }
        var interleaved: [Float] = []
        interleaved.reserveCapacity(sourceSamples * 2)
        for sample in left {
            interleaved.append(sample)
            interleaved.append(sample)
        }
        let sourceAudio = MLXArray(interleaved, [1, sourceSamples, 2]).asType(.float32)

        let expectedLatentFrames = Int((Double(durationSeconds) * 25.0).rounded())
        let expectedSamples = expectedLatentFrames * factor

        let audio = try pipeline.generatePromptToAudio(
            caption: "soft synth-pop cover with gentle drums",
            lyrics: "[verse]\ncover smoke test",
            config: .init(durationSeconds: durationSeconds, seed: 91),
            sourceAudio48kHz: sourceAudio,
            audioCoverStrength: 0.8,
            task: .cover
        )

        XCTAssertEqual(audio.dim(0), 1)
        XCTAssertEqual(audio.dim(1), expectedSamples)
        XCTAssertEqual(audio.dim(2), vaeConfig.audioChannels)

        let maxAbs = MLX.max(MLX.abs(audio.asType(.float32))).item(Float.self)
        XCTAssertFalse(maxAbs.isNaN)
        XCTAssertFalse(maxAbs.isInfinite)
        XCTAssertGreaterThan(maxAbs, 0)
    }

    private func normalizedCorrelation(_ lhs: [Float], _ rhs: [Float]) -> Double {
        precondition(lhs.count == rhs.count)
        let count = Double(lhs.count)
        let lhsMean = lhs.reduce(0) { $0 + Double($1) } / count
        let rhsMean = rhs.reduce(0) { $0 + Double($1) } / count

        var covariance = 0.0
        var lhsVariance = 0.0
        var rhsVariance = 0.0
        for index in lhs.indices {
            let lhsCentered = Double(lhs[index]) - lhsMean
            let rhsCentered = Double(rhs[index]) - rhsMean
            covariance += lhsCentered * rhsCentered
            lhsVariance += lhsCentered * lhsCentered
            rhsVariance += rhsCentered * rhsCentered
        }
        return covariance / sqrt(max(lhsVariance * rhsVariance, Double.leastNonzeroMagnitude))
    }

    private func normalizedMeanSquaredError(_ lhs: [Float], _ rhs: [Float]) -> Double {
        precondition(lhs.count == rhs.count)
        var squaredError = 0.0
        var sourceEnergy = 0.0
        for index in lhs.indices {
            let source = Double(lhs[index])
            let difference = source - Double(rhs[index])
            squaredError += difference * difference
            sourceEnergy += source * source
        }
        return squaredError / max(sourceEnergy, Double.leastNonzeroMagnitude)
    }
}
