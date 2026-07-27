import MLX
import XCTest
@testable import MereRunCore

final class ACEStepTaskAndRepaintTests: MereRunCoreTestCase {
    func testCandidateScoringRejectsSilenceClippingAndNonFiniteAudio() {
        let healthy = (0..<4_800).map { index in
            Float(sin(Double(index) * 2 * .pi * 440 / 48_000)) * 0.2
        }
        let silent = [Float](repeating: 0, count: 4_800)
        let clipped = [Float](repeating: 1, count: 4_800)
        var nonFinite = healthy
        nonFinite[0] = .nan
        nonFinite[1] = .infinity

        let healthyResult = ACEStepCandidateScorer.evaluate(MLXArray(healthy))
        let silentResult = ACEStepCandidateScorer.evaluate(MLXArray(silent))
        let clippedResult = ACEStepCandidateScorer.evaluate(MLXArray(clipped))
        let nonFiniteResult = ACEStepCandidateScorer.evaluate(MLXArray(nonFinite))

        XCTAssertGreaterThan(healthyResult.score, silentResult.score)
        XCTAssertGreaterThan(healthyResult.score, clippedResult.score)
        XCTAssertLessThan(nonFiniteResult.metrics.finiteRatio, 1)
        XCTAssertTrue(healthyResult.score.isFinite)
    }

    func testCandidateScoringPenalizesStationaryBroadbandNoise() {
        let sampleCount = 48_000
        var state: UInt64 = 0x9e37_79b9_7f4a_7c15
        var noise: [Float] = []
        var structured: [Float] = []
        noise.reserveCapacity(sampleCount)
        structured.reserveCapacity(sampleCount)
        for index in 0..<sampleCount {
            state = state &* 6_364_136_223_846_793_005 &+ 1
            let random = Float((state >> 40) & 0xFF_FFFF)
                / Float(0xFF_FFFF)
            noise.append((random * 2 - 1) * 0.2)

            let time = Double(index) / 48_000
            let noteIndex = Int(time * 8) % 4
            let fundamental = [165.0, 220.0, 277.18, 330.0][noteIndex]
            let beatEnvelope = 0.35 + 0.65
                * max(0, sin(2 * Double.pi * 4 * time))
            let tonal = 0.16 * sin(2 * Double.pi * fundamental * time)
                + 0.08 * sin(2 * Double.pi * fundamental * 2 * time)
            structured.append(Float(beatEnvelope * tonal))
        }

        let noiseResult = ACEStepCandidateScorer.evaluate(MLXArray(noise))
        let structuredResult = ACEStepCandidateScorer.evaluate(
            MLXArray(structured)
        )
        var truncated = structured
        truncated.replaceSubrange(
            (sampleCount * 3 / 4)..<sampleCount,
            with: repeatElement(0, count: sampleCount / 4)
        )
        let truncatedResult = ACEStepCandidateScorer.evaluate(
            MLXArray(truncated)
        )

        XCTAssertGreaterThan(structuredResult.score, noiseResult.score + 10)
        XCTAssertGreaterThan(
            structuredResult.score,
            truncatedResult.score + 3
        )
        XCTAssertLessThan(
            structuredResult.metrics.spectralFlatness ?? 1,
            noiseResult.metrics.spectralFlatness ?? 0
        )
        XCTAssertGreaterThan(
            structuredResult.metrics.periodicity ?? 0,
            noiseResult.metrics.periodicity ?? 1
        )
        XCTAssertGreaterThan(
            structuredResult.metrics.temporalSpectralVariation ?? 0,
            noiseResult.metrics.temporalSpectralVariation ?? 1
        )
        XCTAssertGreaterThan(
            structuredResult.metrics.tailEnergyRatio ?? 0,
            0.5
        )
        XCTAssertLessThan(
            truncatedResult.metrics.tailEnergyRatio ?? 1,
            0.02
        )
    }

    func testCandidateRankingIsScoreDescendingAndStableByIndex() {
        let audio = MLXArray([Float](repeating: 0, count: 2))
        let metrics = ACEStepCandidateScorer.evaluate(audio).metrics
        let ranked = ACEStepCandidateScorer.rank([
            .init(index: 2, seed: 12, audio: audio, score: 50, metrics: metrics),
            .init(index: 1, seed: 11, audio: audio, score: 80, metrics: metrics),
            .init(index: 0, seed: 10, audio: audio, score: 80, metrics: metrics),
        ])

        XCTAssertEqual(ranked.map(\.index), [0, 1, 2])
    }

    func testRetakeNoiseInterpolationMatchesUpstreamEndpointsAndMidpoint() {
        let primary = MLXArray([Float(1), 0, -1])
        let retake = MLXArray([Float(0), 1, 0.5])

        XCTAssertEqual(
            ACEStepPipeline.blendRetakeNoise(
                primary: primary,
                retake: retake,
                variance: 0
            ).asArray(Float.self),
            primary.asArray(Float.self)
        )
        XCTAssertEqual(
            ACEStepPipeline.blendRetakeNoise(
                primary: primary,
                retake: retake,
                variance: 1
            ).asArray(Float.self),
            retake.asArray(Float.self)
        )

        let midpoint = ACEStepPipeline.blendRetakeNoise(
            primary: primary,
            retake: retake,
            variance: 0.5
        ).asArray(Float.self)
        let scale = Float(sqrt(0.5))
        XCTAssertEqual(midpoint[0], scale, accuracy: 0.000_001)
        XCTAssertEqual(midpoint[1], scale, accuracy: 0.000_001)
        XCTAssertEqual(midpoint[2], -0.5 * scale, accuracy: 0.000_001)
    }

    func testFlowEditWindowAndVelocityDeltaMatchUpstream() throws {
        let configuration = ACEStepFlowEditConfiguration(
            sourceCaption: "acoustic folk demo",
            nMin: 0.2,
            nMax: 0.8,
            nAverage: 3,
            retakeSeed: 99
        )
        XCTAssertNoThrow(try configuration.validate())
        let window = ACEStepFlowEdit.windowIndices(
            stepCount: 50,
            nMin: configuration.nMin,
            nMax: configuration.nMax
        )
        XCTAssertEqual(window.minimum, 10)
        XCTAssertEqual(window.maximum, 40)

        let edited = MLXArray([Float(1), 2, 3])
        let sourceVelocity = MLXArray([Float(2), 4, 6])
        let targetVelocity = MLXArray([Float(3), 2, 10])
        let result = ACEStepFlowEdit.integrateDelta(
            editedLatents: edited,
            sourceVelocity: sourceVelocity,
            targetVelocity: targetVelocity,
            currentTimestep: 0.8,
            nextTimestep: 0.6
        ).asArray(Float.self)
        XCTAssertEqual(result[0], 0.8, accuracy: 0.000_001)
        XCTAssertEqual(result[1], 2.4, accuracy: 0.000_001)
        XCTAssertEqual(result[2], 2.2, accuracy: 0.000_001)
    }

    func testFlowEditValidationRejectsInvalidWindowsAndAverages() {
        XCTAssertThrowsError(
            try ACEStepFlowEditConfiguration(
                sourceCaption: "source",
                nMin: 0.8,
                nMax: 0.2
            ).validate()
        )
        XCTAssertThrowsError(
            try ACEStepFlowEditConfiguration(
                sourceCaption: "source",
                nAverage: 0
            ).validate()
        )
    }

    func testLRCParsingRenderingAndApproximateTiming() throws {
        let document = try ACEStepLRCDocument.parse(
            """
            [ar:Test Artist]
            [00:01.25]First line
            [00:03.50][00:05.00]Repeat line
            """
        )
        XCTAssertEqual(document.metadata["ar"], "Test Artist")
        XCTAssertEqual(document.lines.count, 3)
        XCTAssertEqual(document.lines[0].timestampSeconds, 1.25, accuracy: 0.001)
        XCTAssertEqual(document.lyrics, "First line\nRepeat line\nRepeat line")
        XCTAssertTrue(document.rendered().contains("[00:03.50]Repeat line"))

        let approximate = ACEStepLRCDocument.approximate(
            lyrics: "one\ntwo\nthree",
            durationSeconds: 12
        )
        XCTAssertTrue(approximate.timingIsApproximate)
        XCTAssertEqual(
            approximate.lines.map(\.timestampSeconds),
            [0, 4, 8]
        )
        XCTAssertTrue(approximate.rendered().contains("approximate line timing"))
    }

    func testContinuousSchedulerMatchesUpstreamBaseSchedule() {
        XCTAssertEqual(
            ACEStepContinuousScheduler(
                inferenceSteps: 4,
                shift: 1
            ).timesteps,
            [1, 0.75, 0.5, 0.25]
        )

        let shifted = ACEStepContinuousScheduler(
            inferenceSteps: 4,
            shift: 2
        ).timesteps
        XCTAssertEqual(shifted.count, 4)
        XCTAssertEqual(shifted[0], 1, accuracy: 0.000_001)
        XCTAssertEqual(shifted[1], 6.0 / 7.0, accuracy: 0.000_001)
        XCTAssertEqual(shifted[2], 2.0 / 3.0, accuracy: 0.000_001)
        XCTAssertEqual(shifted[3], 0.4, accuracy: 0.000_001)
    }

    func testContinuousSchedulerStripsTerminalZeroFromCustomTimesteps() {
        XCTAssertEqual(
            ACEStepContinuousScheduler(
                inferenceSteps: 50,
                shift: 1,
                timesteps: [1, 0.5, 0]
            ).timesteps,
            [1, 0.5]
        )
    }

    func testCFGMatchesUpstreamFormula() {
        let conditional = MLXArray([Float(1), 3], [1, 1, 2])
        let unconditional = MLXArray([Float(-1), 1], [1, 1, 2])
        let guided = ACEStepGuidance.cfg(
            conditional: conditional,
            unconditional: unconditional,
            scale: 2.5
        )
        MLX.eval(guided)

        XCTAssertEqual(guided.asArray(Float.self), [4, 6])
    }

    func testAPGProjectionAndMomentumMatchUpstream() {
        let conditional = MLXArray(
            [
                Float(1), 0,
                0, 1,
            ],
            [1, 2, 2]
        )
        let unconditional = MLXArray(
            [
                Float(1), -1,
                -1, 1,
            ],
            [1, 2, 2]
        )
        var runningAverage: MLXArray?

        let first = ACEStepGuidance.apg(
            conditional: conditional,
            unconditional: unconditional,
            scale: 2,
            runningAverage: &runningAverage
        )
        let second = ACEStepGuidance.apg(
            conditional: conditional,
            unconditional: unconditional,
            scale: 2,
            runningAverage: &runningAverage
        )
        MLX.eval(first, second)

        XCTAssertEqual(first.asArray(Float.self), [1, 1, 1, 1])
        XCTAssertEqual(second.asArray(Float.self), [1, 0.25, 0.25, 1])
    }

    func testADGProducesFiniteVelocityWithUpstreamShape() {
        let latents = MLXArray(
            [
                0.5, -0.5,
                1.0, 0.25,
            ],
            [1, 2, 2]
        )
        let conditional = MLXArray(
            [
                0.1, 0.2,
                -0.3, 0.4,
            ],
            [1, 2, 2]
        )
        let unconditional = MLXArray(
            [
                -0.2, 0.3,
                0.2, -0.1,
            ],
            [1, 2, 2]
        )
        let guided = ACEStepGuidance.adg(
            latents: latents,
            conditional: conditional,
            unconditional: unconditional,
            sigma: 0.75,
            scale: 7
        )
        MLX.eval(guided)

        XCTAssertEqual(guided.shape, conditional.shape)
        XCTAssertTrue(guided.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testTaskCapabilityMatrixMatchesUpstream() throws {
        for variant in [
            ACEStepCheckpointVariant.turbo,
            .sft,
            .xlTurbo,
            .xlSFT,
        ] {
            XCTAssertTrue(variant.supports(.textToMusic))
            XCTAssertTrue(variant.supports(.repaint))
            XCTAssertTrue(variant.supports(.cover))
            XCTAssertTrue(variant.supports(.coverNoFSQ))
            XCTAssertFalse(variant.supports(.extract))
            XCTAssertFalse(variant.supports(.lego))
            XCTAssertFalse(variant.supports(.complete))
        }

        for variant in [ACEStepCheckpointVariant.base, .xlBase] {
            for task in ACEStepTask.allCases {
                XCTAssertTrue(variant.supports(task), "\(variant) should support \(task)")
            }
        }

        XCTAssertThrowsError(try ACEStepCheckpointVariant.xlTurbo.validate(.extract))
        XCTAssertNoThrow(try ACEStepCheckpointVariant.xlBase.validate(.extract))
    }

    func testCheckpointInferenceDefaultsMatchOfficialModelFamilies() {
        XCTAssertEqual(ACEStepCheckpointVariant.xlTurbo.defaultInferenceSteps, 8)
        XCTAssertEqual(ACEStepCheckpointVariant.xlTurbo.defaultShift, 3)
        XCTAssertEqual(ACEStepCheckpointVariant.xlTurbo.defaultGuidanceScale, 1)

        XCTAssertEqual(ACEStepCheckpointVariant.xlSFT.defaultInferenceSteps, 50)
        XCTAssertEqual(ACEStepCheckpointVariant.xlSFT.defaultShift, 1)
        XCTAssertEqual(ACEStepCheckpointVariant.xlSFT.defaultGuidanceScale, 7)

        XCTAssertEqual(ACEStepCheckpointVariant.xlBase.defaultInferenceSteps, 50)
        XCTAssertEqual(ACEStepCheckpointVariant.xlBase.defaultShift, 1)
        XCTAssertEqual(ACEStepCheckpointVariant.xlBase.defaultGuidanceScale, 7)
    }

    func testAdaptiveQualityPresetsScaleByCheckpointAndWorkflow() {
        let turboDraft = ACEStepQualityPreset.draft.defaults(
            for: .xlTurbo,
            task: .textToMusic
        )
        XCTAssertEqual(turboDraft.inferenceSteps, 4)
        XCTAssertFalse(turboDraft.usesLanguageModel)
        XCTAssertEqual(turboDraft.candidateCount, 1)

        let baseFinal = ACEStepQualityPreset.final.defaults(
            for: .xlBase,
            task: .textToMusic
        )
        XCTAssertEqual(baseFinal.inferenceSteps, 50)
        XCTAssertEqual(baseFinal.guidanceScale, 7)
        XCTAssertEqual(baseFinal.samplerMode, .heun)
        XCTAssertEqual(baseFinal.velocityNormThreshold, 2)
        XCTAssertEqual(baseFinal.velocityEMAFactor, 0.1)
        XCTAssertTrue(baseFinal.automaticDuration)
        XCTAssertEqual(baseFinal.candidateCount, 4)

        let repaint = ACEStepQualityPreset.edit.defaults(
            for: .xlTurbo,
            task: .repaint
        )
        XCTAssertFalse(repaint.usesLanguageModel)
        XCTAssertFalse(repaint.automaticDuration)
    }

    func testTaskInstructionsAndLMRoutingMatchUpstream() {
        XCTAssertEqual(
            ACEStepTask.coverNoFSQ.instruction(),
            "Generate audio semantic tokens based on the given conditions:"
        )
        XCTAssertEqual(
            ACEStepTask.extract.instruction(trackName: "drums"),
            "Extract the DRUMS track from the audio:"
        )
        XCTAssertEqual(
            ACEStepTask.complete.instruction(completeTrackClasses: ["Drums", "Bass"]),
            "Complete the input track with DRUMS | BASS:"
        )
        XCTAssertTrue(ACEStepTask.repaint.skipsLanguageModel)
        XCTAssertTrue(ACEStepTask.cover.usesFSQCoverHints)
        XCTAssertFalse(ACEStepTask.coverNoFSQ.usesFSQCoverHints)
        XCTAssertTrue(ACEStepTask.lego.locksDurationToSource)
        XCTAssertFalse(ACEStepTask.complete.locksDurationToSource)
    }

    func testCheckpointVariantDetectionUsesNameAndConfig() {
        let turbo = ACEStepConfig(isTurbo: true)
        let xlNonTurbo = ACEStepConfig(hiddenSize: 2_560, isTurbo: false)

        XCTAssertEqual(
            ACEStepCheckpointVariant.detect(
                modelRootURL: URL(fileURLWithPath: "/models/acestep-v15-xl-turbo"),
                config: turbo
            ),
            .xlTurbo
        )
        XCTAssertEqual(
            ACEStepCheckpointVariant.detect(
                modelRootURL: URL(fileURLWithPath: "/models/acestep-v15-xl-sft"),
                config: xlNonTurbo
            ),
            .xlSFT
        )
        XCTAssertEqual(
            ACEStepCheckpointVariant.detect(
                modelRootURL: URL(fileURLWithPath: "/models/acestep-v15-base"),
                config: ACEStepConfig(isTurbo: false)
            ),
            .base
        )
    }

    func testRepaintModeMappingMatchesUpstream() {
        let conservative = ACEStepRepaintConfiguration(mode: .conservative)
        XCTAssertEqual(conservative.injectionRatio, 1)
        XCTAssertEqual(conservative.latentCrossfadeFrames, 25)
        XCTAssertEqual(conservative.waveformCrossfadeSeconds, 0.05)

        let balanced = ACEStepRepaintConfiguration(mode: .balanced, strength: 0.5)
        XCTAssertEqual(balanced.injectionRatio, 0.5)
        XCTAssertEqual(balanced.latentCrossfadeFrames, 12)
        XCTAssertEqual(balanced.waveformCrossfadeSeconds, 0.025)

        let aggressive = ACEStepRepaintConfiguration(mode: .aggressive)
        XCTAssertEqual(aggressive.injectionRatio, 0)
        XCTAssertEqual(aggressive.latentCrossfadeFrames, 0)
        XCTAssertEqual(aggressive.waveformCrossfadeSeconds, 0)
    }

    func testRepaintConditioningMasksOnlyRequestedRange() {
        let source = MLXArray(Array(repeating: Float(1), count: 6 * 2), [1, 6, 2])
        let silence = MLXArray.zeros([1, 6, 2], dtype: .float32)
        let configuration = ACEStepRepaintConfiguration(
            startSeconds: 0.08,
            endSeconds: 0.16,
            chunkMaskMode: .explicit
        )

        let result = ACEStepRepaint.prepareConditioning(
            cleanSourceLatents: source,
            silenceLatents: silence,
            chunkChannels: 2,
            configuration: configuration,
            task: .repaint
        )
        MLX.eval(result.sourceLatents, result.chunkMasks, result.repaintMask)

        XCTAssertEqual(result.latentRange, 2..<4)
        XCTAssertEqual(
            result.repaintMask.asArray(Bool.self),
            [false, false, true, true, false, false]
        )
        XCTAssertEqual(
            result.sourceLatents.asArray(Float.self),
            [
                1, 1,
                1, 1,
                0, 0,
                0, 0,
                1, 1,
                1, 1,
            ]
        )
        XCTAssertEqual(
            result.chunkMasks.asArray(Float.self),
            [
                0, 0,
                0, 0,
                1, 1,
                1, 1,
                0, 0,
                0, 0,
            ]
        )
    }

    func testAutomaticChunkMaskMatchesUpstreamBooleanMaskValue() {
        let source = MLXArray.ones([1, 6, 2], dtype: .float32)
        let silence = MLXArray.zeros([1, 6, 2], dtype: .float32)
        let configuration = ACEStepRepaintConfiguration(
            startSeconds: 0.08,
            endSeconds: 0.16,
            chunkMaskMode: .auto
        )

        let result = ACEStepRepaint.prepareConditioning(
            cleanSourceLatents: source,
            silenceLatents: silence,
            chunkChannels: 2,
            configuration: configuration,
            task: .repaint
        )

        XCTAssertEqual(
            result.chunkMasks.asArray(Float.self),
            Array(repeating: 1, count: 12)
        )
    }

    func testRepaintStepInjectionAndBoundaryBlendMatchUpstream() {
        let generated = MLXArray(Array(repeating: Float(10), count: 6), [1, 6, 1])
        let source = MLXArray.zeros([1, 6, 1], dtype: .float32)
        let noise = MLXArray(Array(repeating: Float(4), count: 6), [1, 6, 1])
        let mask = MLXArray([0, 0, 1, 1, 0, 0], [1, 6]).asType(.bool)

        let injected = ACEStepRepaint.injectPreservedSource(
            generatedLatents: generated,
            cleanSourceLatents: source,
            repaintMask: mask,
            nextTimestep: 0.5,
            noise: noise
        )
        let blended = ACEStepRepaint.blendLatentBoundaries(
            generatedLatents: generated,
            cleanSourceLatents: source,
            repaintMask: mask,
            crossfadeFrames: 1
        )
        MLX.eval(injected, blended)

        XCTAssertEqual(injected.asArray(Float.self), [2, 2, 10, 10, 2, 2])
        XCTAssertEqual(blended.asArray(Float.self), [0, 5, 10, 10, 5, 0])
    }

    func testLegoRangePreservesSourceLatentsWhileMaskingGenerationRange() {
        let source = MLXArray(Array(repeating: Float(2.5), count: 6 * 2), [1, 6, 2])
        let silence = MLXArray.zeros([1, 6, 2], dtype: .float32)
        let configuration = ACEStepRepaintConfiguration(
            startSeconds: 0.08,
            endSeconds: 0.16,
            chunkMaskMode: .explicit
        )

        let result = ACEStepRepaint.prepareConditioning(
            cleanSourceLatents: source,
            silenceLatents: silence,
            chunkChannels: 2,
            configuration: configuration,
            task: .lego
        )
        MLX.eval(result.sourceLatents, result.chunkMasks)

        XCTAssertEqual(
            result.sourceLatents.asArray(Float.self),
            Array(repeating: 2.5, count: 12)
        )
        XCTAssertEqual(
            result.chunkMasks.asArray(Float.self),
            [
                0, 0,
                0, 0,
                1, 1,
                1, 1,
                0, 0,
                0, 0,
            ]
        )
    }

    func testWaveformSpliceRestoresOriginalOutsideRepaintRange() {
        let generated = MLXArray(Array(repeating: Float(10), count: 10), [1, 10, 1])
        let source = MLXArray.zeros([1, 10, 1], dtype: .float32)
        let spliced = ACEStepRepaint.spliceWaveform(
            generatedAudio: generated,
            sourceAudio: source,
            startSeconds: 0.2,
            endSeconds: 0.6,
            crossfadeSeconds: 0.1,
            sampleRate: 10
        )
        MLX.eval(spliced)

        XCTAssertEqual(
            spliced.asArray(Float.self),
            [0, 5, 10, 10, 10, 10, 5, 0, 0, 0]
        )
    }
}
