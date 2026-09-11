import Foundation
import MLX

protocol ACEStepGenerationPlanning {
    func planMusic(
        caption: String, lyrics: String, instruction: String,
        userMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata,
        useCotCaption: Bool, lmConfig: ACEStep5HzLMGenerationConfig
    ) throws -> ACEStepMusicPlan

    func understandSourceAudio(
        sourceAudio48kHz: MLXArray, durationSeconds: Float, lmConfig: ACEStep5HzLMGenerationConfig
    ) throws -> ACEStepMusicUnderstandingResult
}

extension ACEStepPipeline: ACEStepGenerationPlanning {}

/// Shared planning policy can be exercised without loading a checkpoint.
enum ACEStepGenerationPreparation {
    static func prepare(
        _ payload: ACEStepGenerationOptions,
        variant: ACEStepCheckpointVariant,
        languageModelAvailable: Bool,
        planner: any ACEStepGenerationPlanning
    ) throws -> ACEStepGenerationPlan {
        try Task.checkCancellation()
        let prompt = payload.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw ACEStepPreparationIssue("prompt must not be empty.")
        }
        let task = payload.task ?? .textToMusic
        try variant.validate(task)
        let quality = payload.quality ?? .song
        let defaults = quality.defaults(for: variant, task: task)
        let candidates = payload.candidates ?? defaults.candidateCount
        guard (1...16).contains(candidates) else {
            throw ACEStepPreparationIssue("candidates must be between 1 and 16.")
        }
        let steps = payload.steps ?? defaults.inferenceSteps
        guard steps >= 1 else {
            throw ACEStepPreparationIssue("steps must be at least 1.")
        }
        let shift = payload.shift ?? defaults.shift
        guard shift.isFinite, shift > 0 else {
            throw ACEStepPreparationIssue("shift must be greater than 0.")
        }
        let guidanceScale = payload.guidanceScale ?? defaults.guidanceScale
        guard guidanceScale.isFinite, guidanceScale >= 1 else {
            throw ACEStepPreparationIssue("guidance_scale must be at least 1.")
        }
        let cfgIntervalStart = payload.cfgIntervalStart ?? 0
        let cfgIntervalEnd = payload.cfgIntervalEnd ?? 1
        guard (0...1).contains(cfgIntervalStart),
              (0...1).contains(cfgIntervalEnd),
              cfgIntervalStart <= cfgIntervalEnd
        else {
            throw ACEStepPreparationIssue(
                "CFG intervals require 0 <= cfg_interval_start "
                    + "<= cfg_interval_end <= 1."
            )
        }
        let velocityNormThreshold = payload.velocityNormThreshold
            ?? defaults.velocityNormThreshold
        guard velocityNormThreshold.isFinite, velocityNormThreshold >= 0 else {
            throw ACEStepPreparationIssue(
                "velocity_norm_threshold must be nonnegative."
            )
        }
        let velocityEMAFactor = payload.velocityEMAFactor
            ?? defaults.velocityEMAFactor
        guard (0..<1).contains(velocityEMAFactor) else {
            throw ACEStepPreparationIssue(
                "velocity_ema_factor must be in [0, 1)."
            )
        }
        let audioCoverStrength = payload.audioCoverStrength ?? 1
        guard (0...1).contains(audioCoverStrength) else {
            throw ACEStepPreparationIssue(
                "audio_cover_strength must be between 0 and 1."
            )
        }
        let coverNoiseStrength = payload.coverNoiseStrength ?? 0
        guard (0...1).contains(coverNoiseStrength) else {
            throw ACEStepPreparationIssue(
                "cover_noise_strength must be between 0 and 1."
            )
        }
        let retakeVariance = payload.retakeVariance ?? 0
        guard (0...1).contains(retakeVariance) else {
            throw ACEStepPreparationIssue(
                "retake_variance must be between 0 and 1."
            )
        }
        let repaintStart = payload.repaintStartSeconds ?? 0
        let repaintEnd = payload.repaintEndSeconds ?? -1
        guard repaintStart >= 0,
              repaintEnd == -1 || repaintEnd > repaintStart
        else {
            throw ACEStepPreparationIssue(
                "repaint range requires a nonnegative start and an end "
                    + "greater than the start, or -1."
            )
        }
        let repaintStrength = payload.repaintStrength ?? 0.5
        guard (0...1).contains(repaintStrength) else {
            throw ACEStepPreparationIssue(
                "repaint_strength must be between 0 and 1."
            )
        }
        let lmTopK = payload.lmTopK ?? 0
        let lmTopP = payload.lmTopP ?? 0.9
        let lmTemperature = payload.lmTemperature ?? 0.85
        let lmRepetitionPenalty = payload.lmRepetitionPenalty ?? 1.0
        let lmCFGScale = payload.lmCFGScale ?? 2.0
        let lmNegativePrompt = payload.lmNegativePrompt ?? "NO USER INPUT"
        guard lmTopK >= 0,
              (0...1).contains(lmTopP),
              (0...2).contains(lmTemperature),
              lmRepetitionPenalty.isFinite, lmRepetitionPenalty > 0,
              lmCFGScale >= 1,
              lmCFGScale.isFinite
        else {
            throw ACEStepPreparationIssue(
                "LM sampling requires lm_top_k >= 0, lm_top_p in [0, 1], "
                    + "lm_temperature in [0, 2], lm_repetition_penalty > 0, "
                    + "and lm_cfg_scale >= 1."
            )
        }
        let effectiveLyrics: String
        if payload.instrumental == true {
            guard Self.nonEmpty(payload.lyrics) == nil else {
                throw ACEStepPreparationIssue(
                    "instrumental cannot be combined with lyrics."
                )
            }
            effectiveLyrics = "[Instrumental]"
        } else {
            effectiveLyrics = payload.lyrics ?? ""
        }
        let effectiveLMRepetitionPenalty = lmRepetitionPenalty == 1
            ? nil
            : lmRepetitionPenalty
        let vaeChunkSize = payload.vaeChunkSize ?? 512
        let vaeOverlap = payload.vaeOverlap ?? 64
        guard vaeChunkSize > 0, vaeOverlap >= 0 else {
            throw ACEStepPreparationIssue(
                "VAE tiling requires vae_chunk_size > 0 and vae_overlap >= 0."
            )
        }
        let sourceAudio = try payload.sourceAudioPath.map {
            try ACEStepRuntimePreparation.loadAudio48kHz($0, label: "Source audio")
        }
        let referenceAudio = try (payload.referenceAudioPaths ?? []).map {
            try ACEStepRuntimePreparation.loadAudio48kHz(
                $0,
                label: "Reference audio"
            )
        }
        let isFlowEdit = payload.sourceCaption != nil
        if task.requiresSourceAudio, sourceAudio == nil {
            throw ACEStepPreparationIssue("source_audio_path is required for \(task.rawValue).")
        }
        if isFlowEdit, sourceAudio == nil {
            throw ACEStepPreparationIssue("source_audio_path is required with source_caption.")
        }
        var duration = task.locksDurationToSource || isFlowEdit
            ? sourceAudio.map {
                ACEStepRuntimePreparation.durationSeconds(
                    of: $0,
                    fallback: defaults.fallbackDurationSeconds
                )
            } ?? defaults.fallbackDurationSeconds
            : payload.durationSeconds ?? defaults.fallbackDurationSeconds
        guard duration.isFinite, duration > 0, duration <= 600 else {
            throw ACEStepPreparationIssue("Duration must be positive and at most 600 seconds.")
        }
        let useLM = !isFlowEdit
            && !task.skipsLanguageModel
            && languageModelAvailable
            && (payload.useLanguageModel ?? defaults.usesLanguageModel)
        let instruction = Self.nonEmpty(payload.instruction)
            ?? task.instruction(
                trackName: payload.trackName,
                completeTrackClasses: payload.completeTrackClasses ?? []
            )
        let shouldPlanDuration = useLM
            && payload.durationSeconds == nil
            && defaults.automaticDuration
            && !task.locksDurationToSource
            && !isFlowEdit
        var effectivePrompt = prompt
        var lmCodeGenerationContext: ACEStepLMCodeGenerationContext?
        let effectiveLanguage = ACEStepPlanningPolicy.effectiveLanguage(
            vocalLanguage: payload.vocalLanguage,
            metadataLanguage: payload.metadataLanguage
        )
        var metadata = ACEStep5HzLMConstrainedSampler.UserMetadata(
            bpm: payload.bpm.map(String.init),
            caption: prompt,
            duration: shouldPlanDuration
                ? nil
                : String(max(1, Int(payload.roundMetadataDuration ? duration.rounded() : duration))),
            keyscale: Self.nonEmpty(payload.keyscale),
            language: effectiveLanguage,
            timesignature: Self.nonEmpty(payload.timeSignature)
        )
        if payload.analyzeSourceAudio {
            guard let sourceAudio else { throw ACEStepPreparationIssue("Source analysis requires source audio.") }
            let analysis = try planner.understandSourceAudio(
                sourceAudio48kHz: sourceAudio, durationSeconds: duration,
                lmConfig: .init(maxNewTokens: 2_048, temperature: 0.3, topK: lmTopK,
                                topP: lmTopP, repetitionPenalty: effectiveLMRepetitionPenalty)
            )
            metadata = ACEStepPlanningPolicy.mergeMissing(userMetadata: metadata, analysis: analysis.metadata).metadata
        }
        if useLM {
            let plan = try planner.planMusic(
                caption: prompt,
                lyrics: effectiveLyrics,
                instruction: instruction,
                userMetadata: .init(
                    bpm: metadata.bpm,
                    duration: shouldPlanDuration
                        ? nil
                        : metadata.duration,
                    keyscale: metadata.keyscale,
                    language: metadata.language,
                    timesignature: metadata.timesignature
                ),
                useCotCaption: payload.rewriteCaption,
                lmConfig: .init(
                    maxNewTokens: 1_024,
                    temperature: lmTemperature,
                    topK: lmTopK,
                    topP: lmTopP,
                    repetitionPenalty: effectiveLMRepetitionPenalty,
                    seed: payload.planningSeed
                )
            )
            lmCodeGenerationContext = plan.codeGenerationContext
            effectivePrompt = payload.rewriteCaption ? Self.nonEmpty(plan.metadata.caption) ?? prompt : prompt
            if shouldPlanDuration,
               let plannedDuration = plan.metadata.durationSeconds
            {
                let upperBound: Float = quality == .song ? 240 : 600
                duration = min(max(plannedDuration, 10), upperBound)
            }
            metadata = ACEStepPlanningPolicy.merge(
                userMetadata: metadata,
                plan: plan.metadata,
                caption: effectivePrompt,
                durationSeconds: duration
            )
            lmCodeGenerationContext = lmCodeGenerationContext?.applying(
                userMetadata: metadata
            )
        }
        let config = ACEStepInferenceConfig(
            durationSeconds: duration,
            fixNFE: steps,
            shift: shift,
            coverNoiseStrength: coverNoiseStrength,
            retakeSeed: payload.retakeSeed,
            retakeVariance: retakeVariance,
            inferMethod: payload.inferMethod ?? .ode,
            samplerMode: payload.sampler ?? defaults.samplerMode,
            guidanceScale: guidanceScale,
            guidanceMode: payload.guidanceMode ?? .apg,
            cfgIntervalStart: cfgIntervalStart,
            cfgIntervalEnd: cfgIntervalEnd,
            velocityNormThreshold: velocityNormThreshold,
            velocityEMAFactor: velocityEMAFactor,
            useTiledVaeDecode: payload.useTiledVAEDecode ?? true,
            vaeChunkSize: vaeChunkSize,
            vaeOverlap: vaeOverlap,
            seed: payload.seed
        )
        let repaint = task == .repaint || task == .lego
            ? ACEStepRepaintConfiguration(
                startSeconds: repaintStart,
                endSeconds: repaintEnd,
                chunkMaskMode: payload.chunkMaskMode ?? .auto,
                mode: payload.repaintMode ?? .balanced,
                strength: repaintStrength
            )
            : nil
        let flowEdit = payload.sourceCaption.map {
            ACEStepFlowEditConfiguration(
                sourceCaption: $0,
                sourceLyrics: payload.sourceLyrics ?? "",
                nMin: payload.flowEditNMin ?? 0,
                nMax: payload.flowEditNMax ?? 1,
                nAverage: payload.flowEditNAverage ?? 1,
                retakeSeed: payload.retakeSeed
            )
        }
        try flowEdit?.validate()
        return ACEStepGenerationPlan(
            request: ACEStepSessionRequest(
                caption: effectivePrompt,
                lyrics: effectiveLyrics,
                config: config,
                lmConfig: .init(
                    maxNewTokens: 4_096,
                    temperature: lmTemperature,
                    topK: lmTopK,
                    topP: lmTopP,
                    repetitionPenalty: effectiveLMRepetitionPenalty,
                    cfgScale: lmCFGScale,
                    negativePrompt: lmNegativePrompt
                ),
                lmUserMetadata: metadata,
                lmCodeGenerationContext: lmCodeGenerationContext,
                sourceAudio48kHz: sourceAudio,
                referenceTimbreAudio48kHz:
                    referenceAudio.isEmpty ? nil : referenceAudio,
                audioCoverStrength: audioCoverStrength,
                vocalLanguage: effectiveLanguage,
                instruction: instruction,
                task: task,
                repaintConfiguration: repaint,
                flowEditConfiguration: flowEdit,
                useLanguageModel: useLM
            ),
            quality: quality,
            task: task,
            candidateCount: candidates,
            conditioningMetadata: metadata
        )
    }

    private static func nonEmpty(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(
            in: .whitespacesAndNewlines
        ) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }

}
