import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class ACEStepPipelineIntegrationTests: MereRunCoreTestCase {

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
            lmUserMetadata: .init(bpm: "120", duration: "10", keyscale: "C major", timesignature: "4"),
            isCover: false
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
            isCover: true
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
            isCover: true
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
            isCover: true
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
            isCover: true
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
            isCover: true
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
            isCover: true
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
            isCover: true
        )

        XCTAssertEqual(audio.dim(0), 1)
        XCTAssertEqual(audio.dim(1), expectedSamples)
        XCTAssertEqual(audio.dim(2), vaeConfig.audioChannels)

        let maxAbs = MLX.max(MLX.abs(audio.asType(.float32))).item(Float.self)
        XCTAssertFalse(maxAbs.isNaN)
        XCTAssertFalse(maxAbs.isInfinite)
        XCTAssertGreaterThan(maxAbs, 0)
    }
}
