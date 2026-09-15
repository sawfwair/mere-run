import AudioCore
import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class MusicGenerationPlanTests: XCTestCase {
    func testBackendAdaptersPreserveShortStereoFramesWithoutGuessing() throws {
        let ace = try MusicWaveformAdapter.aceStep(MLXArray([Float(1), 10, 2, 20], [1, 2, 2]))
        let mini = try MusicWaveformAdapter.miniMaxMusic3(MLXArray([Float(1), 2, 10, 20], [1, 2, 2]), sampleRate: 44_100)
        XCTAssertEqual(ace.samples, [1, 10, 2, 20])
        XCTAssertEqual(mini.samples, ace.samples)
        XCTAssertEqual(mini.frameCount, 2)
        XCTAssertThrowsError(try MusicWaveformAdapter.aceStep(MLXArray([Float(1), 2])))
        XCTAssertThrowsError(try MusicWaveformAdapter.miniMaxMusic3(MLXArray([Float](repeating: 0, count: 9), [1, 9, 1]), sampleRate: 44_100))
    }

    func testACEExplicitPlanPreservesOverridesAndSkipsPlanner() throws {
        let planner = PlannerProbe()
        var options = ACEStepGenerationOptions(prompt: "  guitar  ")
        options.quality = .draft
        options.durationSeconds = 6
        options.steps = 8
        options.seed = 42
        options.instrumental = true
        let plan = try prepare(options, planner: planner)
        XCTAssertEqual(plan.request.caption, "guitar")
        XCTAssertEqual(plan.request.lyrics, "[Instrumental]")
        XCTAssertEqual(plan.request.config.durationSeconds, 6)
        XCTAssertEqual(plan.request.config.fixNFE, 8)
        XCTAssertEqual(plan.request.config.seed, 42)
        XCTAssertEqual(plan.candidateCount, 1)
        XCTAssertFalse(plan.request.useLanguageModel)
        XCTAssertEqual(planner.calls, 0)
    }

    func testACEAutomaticPlanningMergesMetadataAndHonorsRewritePolicy() throws {
        let planner = PlannerProbe()
        var options = ACEStepGenerationOptions(prompt: "guitar")
        options.bpm = 90
        options.planningSeed = 42
        var plan = try prepare(options, planner: planner)
        XCTAssertEqual(plan.request.caption, "planned guitar")
        XCTAssertEqual(plan.request.config.durationSeconds, 240)
        XCTAssertEqual(plan.conditioningMetadata.bpm, "90")
        XCTAssertEqual(plan.conditioningMetadata.keyscale, "C major")
        XCTAssertEqual(planner.seed, 42)
        XCTAssertNil(planner.duration)
        options.rewriteCaption = false
        options.durationSeconds = 12.6
        options.roundMetadataDuration = false
        plan = try prepare(options, planner: planner)
        XCTAssertEqual(plan.request.caption, "guitar")
        XCTAssertEqual(plan.request.config.durationSeconds, 12.6)
        XCTAssertEqual(planner.duration, "12")
        XCTAssertFalse(planner.rewrite)
    }

    func testACEInvalidSettingsFailBeforePlanning() {
        let planner = PlannerProbe()
        for mutate in [
            { (value: inout ACEStepGenerationOptions) in value.shift = .infinity },
            { $0.guidanceScale = .nan },
            { $0.candidates = 17 },
            { $0.cfgIntervalStart = 0.9; $0.cfgIntervalEnd = 0.1 },
            { $0.instrumental = true; $0.lyrics = "words" },
        ] {
            var options = ACEStepGenerationOptions(prompt: "guitar")
            mutate(&options)
            XCTAssertThrowsError(try prepare(options, planner: planner))
        }
        XCTAssertEqual(planner.calls, 0)
    }

    func testMiniMaxSharedSettingsKeepExportAndSampleRateDefaultsExplicit() throws {
        var settings = MiniMaxMusic3GenerationSettings(caption: " guitar ", lyrics: " [Instrumental] ")
        settings.audioDuration = 4
        settings.seed = 42
        settings.samplingTier = .quality
        let cli = try settings.resolve(exportPlan: AudioExportPlan(options: .music), defaultSampleRate: 44_100)
        let api = try settings.resolve(exportPlan: AudioExportPlan(options: .referencePCM16), defaultSampleRate: 32_000)
        XCTAssertEqual(cli.generation.caption, api.generation.caption)
        XCTAssertEqual(cli.generation.maximumFrames, 100)
        XCTAssertEqual(cli.generation.inferenceSteps, api.generation.inferenceSteps)
        XCTAssertEqual(cli.generation.seed, 42)
        XCTAssertEqual(cli.sampleRate, 44_100)
        XCTAssertEqual(api.sampleRate, 32_000)
        XCTAssertEqual(cli.export.options.format, .pcm24)
        XCTAssertEqual(api.export.options.format, .pcm16)
    }

    func testMiniMaxRejectsInvalidDurationsBeforeFrameConversion() throws {
        let export = try AudioExportPlan(options: .music)
        for value: Float in [.nan, .infinity, -.infinity, -1, 361] {
            var settings = MiniMaxMusic3GenerationSettings(caption: "guitar", lyrics: "[Instrumental]")
            settings.minimumAudioDuration = value
            XCTAssertThrowsError(try settings.resolve(exportPlan: export, defaultSampleRate: 44_100))
            settings.minimumAudioDuration = nil
            settings.audioDuration = value
            XCTAssertThrowsError(try settings.resolve(exportPlan: export, defaultSampleRate: 44_100))
        }
    }

    func testMiniMaxDurationFloorsAndConflictingLimits() throws {
        let export = try AudioExportPlan(options: .music)
        var settings = MiniMaxMusic3GenerationSettings(caption: "guitar", lyrics: "[Instrumental]")
        settings.minNewTokens = 2_000
        let cli = try settings.resolve(exportPlan: export, defaultSampleRate: 44_100, extendImplicitDurationToMinimum: true)
        XCTAssertEqual(cli.generation.maximumFrames, 2_000)
        XCTAssertThrowsError(try settings.resolve(exportPlan: export, defaultSampleRate: 32_000))
        settings.minNewTokens = nil
        settings.audioDuration = 4
        settings.maxNewTokens = 99
        XCTAssertThrowsError(try settings.resolve(exportPlan: export, defaultSampleRate: 44_100))
    }

    private func prepare(_ options: ACEStepGenerationOptions, planner: PlannerProbe) throws -> ACEStepGenerationPlan {
        try ACEStepGenerationPreparation.prepare(options, variant: .turbo, languageModelAvailable: true, planner: planner)
    }

    private final class PlannerProbe: ACEStepGenerationPlanning {
        var calls = 0
        var seed: UInt64?
        var duration: String?
        var rewrite = true

        func planMusic(
            caption: String, lyrics: String, instruction: String,
            userMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata,
            useCotCaption: Bool, lmConfig: ACEStep5HzLMGenerationConfig
        ) throws -> ACEStepMusicPlan {
            calls += 1
            seed = lmConfig.seed
            duration = userMetadata.duration
            rewrite = useCotCaption
            return .init(metadata: .init(caption: "planned guitar", bpm: 120, durationSeconds: 500, keyscale: "C major"),
                         lmResult: .init(generatedText: "", generatedTokens: [], audioCodeValues: []))
        }

        func understandSourceAudio(
            sourceAudio48kHz: MLXArray, durationSeconds: Float, lmConfig: ACEStep5HzLMGenerationConfig
        ) throws -> ACEStepMusicUnderstandingResult {
            throw ACEStepPreparationIssue("Unexpected source analysis")
        }
    }
}
