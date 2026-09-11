import AudioCore
import Foundation

/// Optional model settings resolved once before loading or generating audio.
public struct MiniMaxMusic3GenerationSettings: Sendable {
    public var caption: String
    public var lyrics: String
    public var seed: UInt64? = nil
    public var maxNewTokens: Int? = nil
    public var audioDuration: Float? = nil
    public var minimumAudioDuration: Float? = nil
    public var minNewTokens: Int? = nil
    public var samplingTier: MiniMaxMusic3SamplingTier? = nil
    public var flowStrategy: MiniMaxMusic3FlowStrategy? = nil
    public var flowSolver: MiniMaxMusic3FlowSolver? = nil
    public var autoregressiveGuidanceFrames: Int? = nil
    public var flowGuidanceEnd: Float? = nil
    public var seedStrategy: MiniMaxMusic3SeedStrategy? = nil
    public var lyricPreflight: MiniMaxMusic3LyricPreflightPolicy? = nil
    public var numInferenceSteps: Int? = nil
    public var guidanceScale: Float? = nil
    public var sampleRate: Int? = nil
    public var profilingEnabled = false
    public var blueprint: MiniMaxMusic3SongBlueprint?

    public init(caption: String, lyrics: String) {
        self.caption = caption
        self.lyrics = lyrics
    }
}

public struct MiniMaxMusic3GenerationPlan: Sendable {
    public let generation: MiniMaxMusic3GenerationOptions
    public let sampleRate: Int
    public let export: AudioExportPlan
    public let lyricPreflight: MiniMaxMusic3LyricPreflightReport?
}

extension MiniMaxMusic3GenerationSettings {
    public func resolve(exportPlan: AudioExportPlan, defaultSampleRate: Int, defaultDurationSeconds: Float = 60, extendImplicitDurationToMinimum: Bool = false) throws -> MiniMaxMusic3GenerationPlan {
        try Task.checkCancellation()
        let request = self
        let caption = request.caption.trimmingCharacters(in: .whitespacesAndNewlines)
        let lyrics = request.lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !caption.isEmpty else {
            throw MiniMaxMusic3Error.invalidPrompt("instructions must contain a music description.")
        }
        guard !lyrics.isEmpty else {
            throw MiniMaxMusic3Error.invalidPrompt("input must contain lyrics or [Instrumental].")
        }
        let duration = request.audioDuration ?? defaultDurationSeconds
        guard duration > 0, duration <= 360 else {
            throw MiniMaxMusic3Error.invalidPrompt("audio_duration must be greater than 0 and at most 360 seconds.")
        }
        if let minimumAudioDuration = request.minimumAudioDuration,
           !minimumAudioDuration.isFinite || minimumAudioDuration <= 0 || minimumAudioDuration > 360
        {
            throw MiniMaxMusic3Error.invalidPrompt(
                "minimum_audio_duration must be greater than 0 and at most 360 seconds."
            )
        }
        let durationFrames = Int(duration * Float(MiniMaxMusic3Prompt.frameRate))
        let durationFloorFrames = request.minimumAudioDuration.map {
            MiniMaxMusic3Prompt.minimumFrameCount(forDurationSeconds: $0)
        }
        let minimumFrames = request.minNewTokens ?? durationFloorFrames
        let maximumFrames = request.maxNewTokens
            ?? (request.minimumAudioDuration == nil && !extendImplicitDurationToMinimum
                ? durationFrames
                : max(durationFrames, minimumFrames ?? 0))
        guard (1...MiniMaxMusic3Prompt.maxAudioFrames).contains(maximumFrames) else {
            throw MiniMaxMusic3Error.invalidPrompt("max_new_tokens must be between 1 and 9000.")
        }
        if request.audioDuration != nil,
           request.maxNewTokens != nil,
           durationFrames != maximumFrames
        {
            throw MiniMaxMusic3Error.invalidPrompt(
                "audio_duration and max_new_tokens must describe the same 25 Hz frame limit."
            )
        }
        if request.audioDuration != nil,
           let minimumAudioDuration = request.minimumAudioDuration,
           minimumAudioDuration > duration
        {
            throw MiniMaxMusic3Error.invalidPrompt(
                "minimum_audio_duration cannot exceed audio_duration."
            )
        }
        if let minimumFrames,
           !(1...MiniMaxMusic3Prompt.maxAudioFrames).contains(minimumFrames)
        {
            throw MiniMaxMusic3Error.invalidPrompt("min_new_tokens must be between 1 and 9000.")
        }
        if let durationFloorFrames, let requestedMinimum = request.minNewTokens,
           durationFloorFrames != requestedMinimum
        {
            throw MiniMaxMusic3Error.invalidPrompt(
                "minimum_audio_duration and min_new_tokens must describe the same 25 Hz frame floor."
            )
        }
        if let minimumFrames, minimumFrames > maximumFrames {
            throw MiniMaxMusic3Error.invalidPrompt("The requested duration floor cannot exceed the output upper bound.")
        }
        let steps = request.numInferenceSteps
            ?? request.samplingTier?.inferenceSteps
            ?? MiniMaxMusic3SamplingTier.quality.inferenceSteps
        guard steps > 0 else {
            throw MiniMaxMusic3Error.invalidPrompt("num_inference_steps must be positive.")
        }
        let guidanceScale = request.guidanceScale ?? 1.7
        guard guidanceScale >= 1, guidanceScale.isFinite else {
            throw MiniMaxMusic3Error.invalidPrompt("guidance_scale must be finite and at least 1.")
        }
        let sampleRate = request.sampleRate ?? defaultSampleRate
        guard sampleRate == 32_000 || sampleRate == 44_100 else {
            throw MiniMaxMusic3Error.invalidPrompt("sample_rate must be 32000 or 44100.")
        }
        if let guidanceFrames = request.autoregressiveGuidanceFrames,
           !(0...MiniMaxMusic3Prompt.maxAudioFrames).contains(guidanceFrames)
        {
            throw MiniMaxMusic3Error.invalidPrompt("autoregressive_guidance_frames must be between 0 and 9000.")
        }
        let flowGuidanceEnd = request.flowGuidanceEnd ?? 1
        guard (0...1).contains(flowGuidanceEnd) else {
            throw MiniMaxMusic3Error.invalidPrompt("flow_guidance_end must be between 0 and 1.")
        }
        let instrumental = lyrics.lowercased() == "[instrumental]"
        let lyricPreflight = request.lyricPreflight == .off ? nil : MiniMaxMusic3LyricPreflight.inspect(
            lyrics: lyrics, durationSeconds: duration, instrumental: instrumental, blueprint: request.blueprint
        )
        if request.lyricPreflight == .strict, let issue = lyricPreflight?.issues.first {
            throw MiniMaxMusic3Error.invalidPrompt("lyric preflight failed: \(issue.message)")
        }
        return MiniMaxMusic3GenerationPlan(
            generation: MiniMaxMusic3GenerationOptions(
                caption: caption,
                lyrics: lyrics,
                durationSeconds: duration,
                minimumFrames: minimumFrames,
                maximumFrames: maximumFrames,
                inferenceSteps: steps,
                seed: request.seed ?? 0,
                guidanceScale: guidanceScale,
                profilingEnabled: request.profilingEnabled,
                flowStrategy: request.flowStrategy ?? .sequential,
                flowSolver: request.flowSolver ?? .euler,
                autoregressiveGuidanceFrames: request.autoregressiveGuidanceFrames,
                flowGuidanceEnd: flowGuidanceEnd,
                seedStrategy: request.seedStrategy ?? .legacy
            ),
            sampleRate: sampleRate,
            export: exportPlan,
            lyricPreflight: lyricPreflight
        )
    }
}
