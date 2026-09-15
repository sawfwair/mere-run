import Foundation
import MereRunContract

extension VideoGenerationOptions {
    func additionalValidationIssues(profile: VideoGenerationModelProfile) -> [VideoGenerationIssue] {
        var issues: [VideoGenerationIssue] = []
        func reject(_ invalid: Bool, _ id: String, _ message: String) {
            if invalid {
                issues.append(VideoGenerationIssue(id: id, title: "Video settings are incompatible", message: message))
            }
        }
        reject(steps.map { $0 < 1 } == true, "steps_invalid", "--steps must be >= 1.")
        reject(!guidanceScale.isFinite || guidanceScale < 0, "guidance_scale_invalid", "--guidance-scale must be finite and >= 0.")
        reject(!shift.isFinite || shift <= 0, "shift_invalid", "--shift must be finite and > 0.")
        reject(!(0...1).contains(ltxSamplerEta), "ltx_sampler_eta_invalid", "--ltx-sampler-eta must be in [0, 1].")
        reject([videoSTGScale, audioSTGScale].contains { !$0.isFinite || $0 < 0 },
               "ltx_stg_scale_invalid", "LTX STG scales must be finite and >= 0.")
        reject(!(0...1).contains(videoGuidanceRescale) || !(0...1).contains(audioGuidanceRescale),
               "ltx_guidance_rescale_invalid", "LTX guidance rescale values must be in [0, 1].")
        reject(videoGuidanceSkipStep < 0 || audioGuidanceSkipStep < 0,
               "ltx_guidance_skip_step_invalid", "LTX guidance skip-step values must be >= 0.")
        reject(videoSTGBlocks.contains { $0 < 0 } || audioSTGBlocks.contains { $0 < 0 },
               "ltx_stg_block_invalid", "LTX STG block indices must be >= 0.")
        reject(res2sBongMaxIterations < 1, "res2s_iterations_invalid", "--res2s-bong-max-iterations must be >= 1.")
        reject(!gradientEstimationGamma.isFinite, "gradient_estimation_gamma_invalid", "--gradient-estimation-gamma must be finite.")
        reject(!h3AdapterStrength.isFinite || h3AdapterStrength <= 0,
               "h3_adapter_strength_invalid", "--h3-adapter-strength must be finite and > 0.")
        for schedule in [ltxSigmas, ltxStage2Sigmas] where !schedule.isEmpty {
            do { _ = try validatedLTXSigmaSchedule(schedule) }
            catch { reject(true, "ltx_sigma_schedule_invalid", error.localizedDescription) }
        }
        reject(!(0...2).contains(temporalUpsampleRounds), "temporal_upsample_rounds_invalid", "--temporal-upsample-rounds must be 0, 1, or 2.")
        reject(!dfr && (temporalUpsampleRounds > 0 || !detailingLoRAs.isEmpty || detailingReferenceDownscaleFactor != nil),
               "dfr_controls_without_dfr", "Temporal upsampling and detailing controls require --dfr.")
        reject(detailingReferenceDownscaleFactor.map { $0 <= 0 } == true,
               "detailing_reference_scale_invalid", "--detailing-reference-downscale-factor must be positive.")
        reject(!(0...1).contains(conditioningAttentionStrength),
               "conditioning_attention_strength_invalid", "--conditioning-attention-strength must be in [0, 1].")
        reject(vaeSpatialTileOverlap < 0, "vae_spatial_overlap_invalid", "--spatial-overlap must be nonnegative.")
        reject(vaeSpatialTileSize.map { $0 <= vaeSpatialTileOverlap } == true,
               "vae_spatial_tile_invalid", "--spatial-tile must be larger than --spatial-overlap.")
        reject(referenceDownscaleFactor.map { $0 <= 0 } == true || referenceTemporalScaleFactor.map { $0 <= 0 } == true,
               "reference_scale_invalid", "Reference downscale and temporal scale factors must be positive.")
        reject(!videoConditionings.isEmpty && loras.isEmpty,
               "video_conditioning_requires_lora", "--video-conditioning requires at least one IC-LoRA via --lora.")
        reject(skipStage2 && videoConditionings.isEmpty, "skip_stage2_requires_reference", "--skip-stage-2 requires --video-conditioning.")
        reject(dfr && hasSourceAudio, "dfr_source_audio_unsupported", "--dfr generates synchronized audio and cannot be combined with source --audio.")
        reject(dfr && (numGeneratedKeyframes > 0 || !generatedKeyframeIndices.isEmpty),
               "dfr_generated_keyframes_conflict", "--dfr derives its own generated-keyframe slots.")
        reject(dfr && (!videoConditionings.isEmpty || skipStage2),
               "dfr_reference_video_conflict", "IC-LoRA reference-video controls cannot be combined with --dfr.")
        reject(dfr && (ltxPreset != .standard || ltxPipeline != .twoStage || ltxSampler != nil
                      || distilledLoRAStrengthStage1 != nil || distilledLoRAStrengthStage2 != nil),
               "dfr_recipe_conflict", "--dfr owns its distilled two-stage sampler and LoRA recipe.")
        reject(ltxPipeline == .keyframeInterpolation && (hasSourceAudio || numGeneratedKeyframes > 0
                || !generatedKeyframeIndices.isEmpty || !videoConditionings.isEmpty || skipStage2),
               "keyframe_interpolation_conflict", "Keyframe interpolation accepts timed image guides and generates its own synchronized audio.")
        let recipe = VideoGenerationLTXRecipe(options: self)
        reject(!recipe.distilledLoRAStrengthStage1.isFinite || !recipe.distilledLoRAStrengthStage2.isFinite,
               "distilled_lora_strength_invalid", "Distilled LoRA strengths must be finite.")
        reject(ltxPipeline == .devOneStage && (recipe.distilledLoRAStrengthStage1 != 0 || recipe.distilledLoRAStrengthStage2 != 0),
               "dev_one_stage_lora_conflict", "The dev-one-stage pipeline runs without the distilled LoRA.")
        reject(ltxPreset == .hq && ltxPipeline != .twoStage, "hq_pipeline_conflict", "The hq preset is the official two-stage Res2s pipeline.")
        guard profile != .unknown else { return issues }
        let advancedLTX = ltxPreset == .hq || ltxPipeline != .twoStage || ltxSampler != nil
            || !ltxSigmas.isEmpty || !ltxStage2Sigmas.isEmpty
            || distilledLoRAStrengthStage1 != nil || distilledLoRAStrengthStage2 != nil
        reject(advancedLTX && (profile != .ltx25Full || hasSourceAudio),
               "ltx_pipeline_model_incompatible", "LTX pipeline, sampler, preset, sigma, and distilled-LoRA controls require generated full LTX 2.5 output.")
        reject(videoDecoder != nil && !profile.isLTX25, "video_decoder_requires_ltx25", "--video-decoder is available for official LTX 2.5 model roots.")
        reject(hdrColorSpace != nil && !profile.isLTX25, "hdr_requires_ltx25", "--hdr requires an official LTX 2.5 model root.")
        reject(enhancePrompt && !profile.isLTX25, "prompt_enhancer_requires_ltx25", "--enhance-prompt requires an official LTX 2.5 model root.")
        reject(!autoDuration.isEmpty && !profile.isLTX25, "auto_duration_requires_ltx25", "--auto-duration requires an official LTX 2.5 model root.")
        reject(dfr && profile != .ltx25Full, "dfr_requires_ltx25_full", "--dfr requires the full LTX 2.5 checkpoint.")
        reject((!imageConditionings.isEmpty || numGeneratedKeyframes > 0 || !generatedKeyframeIndices.isEmpty) && !profile.isLTX25,
               "ltx_image_conditioning_requires_ltx25", "LTX image conditioning and generated keyframes require an official LTX 2.5 model root.")
        reject(hasSourceAudio && profile != .ltx23Full && profile != .ltx23AudioToVideo && profile != .ltx25Full,
               "audio_model_incompatible", "--audio requires a full LTX 2.3, legacy A2Vid, or full LTX 2.5 checkpoint.")
        reject(quality != nil && profile.quality != nil && quality != profile.quality,
               "video_quality_model_mismatch", "The selected checkpoint does not match the requested --quality.")
        reject(!profile.isH3 && !references.isEmpty, "references_require_h3", "--reference requires a MiniMax-H3 Ref2VA model root.")
        reject(profile == .wan && image == nil, "wan_image_required", "Wan2.2 TI2V requires --image.")
        reject(profile == .wan && endImage != nil, "wan_end_image_unsupported", "Wan2.2 TI2V does not support --end-image yet.")
        reject(profile == .h3FL2VA && !references.isEmpty, "h3_fl2va_references_unsupported", "--reference requires a MiniMax-H3 Ref2VA model root.")
        reject(profile == .h3Ref2VA && (image != nil || endImage != nil || !h3FrameInputs.isEmpty || references.isEmpty),
               "h3_ref2va_conditioning_invalid", "MiniMax-H3 Ref2VA requires ordered --reference inputs without FL2VA frame conditions.")
        if timings || timingsOutput != nil, !hasSourceAudio {
            reject(profile.isH3 || profile == .wan || profile.ltxRoute(outputMode: effectiveOutputMode)?.supportsPhaseTimings == false,
                   "timings_lane_unsupported", "--timings and --timings-output require a split LTX model, --quality final, --output-mode audio-video, or --audio.")
        }
        return issues
    }
}
