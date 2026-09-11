import Foundation
import MereRunContract

/// Stable issues shared by observational preflight and execution.
public struct VideoGenerationIssue: LocalizedError, Sendable, Equatable {
    public enum Severity: String, Sendable { case blocker, warning, note }
    public let id: String
    public let severity: Severity
    public let title: String
    public let message: String
    public var errorDescription: String? { message }

    public init(id: String, severity: Severity = .blocker, title: String, message: String) {
        self.id = id
        self.severity = severity
        self.title = title
        self.message = message
    }
}

extension VideoGenerationOptions {
    public func validationIssues(profile: VideoGenerationModelProfile) -> [VideoGenerationIssue] {
        var diagnostics = basicValidationIssues(profile: profile)
        do { _ = try VideoGenerationPlan(options: self, profile: profile) }
        catch let issue as VideoGenerationIssue {
            if !diagnostics.contains(where: { $0.id == issue.id }) { diagnostics.append(issue) }
        } catch {
            diagnostics.append(VideoGenerationIssue(id: "video_plan_invalid", title: "Video plan is invalid", message: error.localizedDescription))
        }
        return diagnostics
    }

    func basicValidationIssues(profile: VideoGenerationModelProfile) -> [VideoGenerationIssue] {
        let input = self
        let usesWanGeometry = profile == .wan
        let usesMiniMaxH3Geometry = profile.isH3
        let usesAudioConditioning = hasSourceAudio
        var diagnostics: [VideoGenerationIssue] = []
        if let message = input.productSelectionValidationMessage {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "video_product_selection_conflict",
                    severity: .blocker,
                    title: "Video product selection is ambiguous",
                    message: message
                )
            )
        }
        if input.numGeneratedKeyframes < 0 {
            diagnostics.append(VideoGenerationIssue(
                id: "ltx_generated_keyframe_count_invalid",
                severity: .blocker,
                title: "Generated keyframe count is invalid",
                message: "--num-generated-keyframes must be nonnegative."
            ))
        }
        if input.numGeneratedKeyframes > 0, !input.generatedKeyframeIndices.isEmpty {
            diagnostics.append(VideoGenerationIssue(
                id: "ltx_generated_keyframe_mode_conflict",
                severity: .blocker,
                title: "Generated keyframe request is ambiguous",
                message: "Use --num-generated-keyframes or explicit --generated-keyframe positions, not both."
            ))
        }
        if usesWanGeometry, input.quality != nil || input.outputMode != nil {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "ltx_product_selection_with_wan",
                    severity: .blocker,
                    title: "LTX product options do not apply to Wan",
                    message: "--quality and --output-mode currently select native LTX generation, not Wan2.2 TI2V."
                )
            )
        }
        if usesMiniMaxH3Geometry, input.quality != nil || input.outputMode != nil || input.legacyVariant != nil {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "ltx_product_selection_with_minimax_h3",
                    severity: .blocker,
                    title: "LTX product options do not apply to MiniMax-H3",
                    message: "--quality, --output-mode, and --variant cannot be combined with MiniMax-H3."
                )
            )
        }
        if !usesMiniMaxH3Geometry, input.h3Adapter != nil {
            diagnostics.append(VideoGenerationIssue(
                id: "h3_adapter_with_non_h3_model",
                severity: .blocker,
                title: "MiniMax-H3 adapter requires MiniMax-H3",
                message: "--h3-adapter can only be used with a MiniMax-H3 model."
            ))
        }
        if !usesMiniMaxH3Geometry,
           (!input.h3FrameInputs.isEmpty
            || input.h3WindowFrames != nil
            || input.h3RenderWidth != nil
            || input.h3RenderHeight != nil) {
            diagnostics.append(VideoGenerationIssue(
                id: "h3_window_or_frame_with_non_h3_model",
                severity: .blocker,
                title: "MiniMax-H3 controls require MiniMax-H3",
                message: "H3 frame, window, and internal-render controls require a MiniMax-H3 model."
            ))
        }
        if usesMiniMaxH3Geometry,
           (input.h3RenderWidth == nil) != (input.h3RenderHeight == nil) {
            diagnostics.append(VideoGenerationIssue(
                id: "h3_render_canvas_incomplete",
                severity: .blocker,
                title: "MiniMax-H3 internal render canvas is incomplete",
                message: "--h3-render-width and --h3-render-height must be set together."
            ))
        }
        if !input.references.isEmpty, !input.h3FrameInputs.isEmpty {
            diagnostics.append(VideoGenerationIssue(
                id: "h3_frame_ref2va_unsupported",
                severity: .blocker,
                title: "Timed H3 frames require FL2VA",
                message: "Use --h3-frame with FL2VA; Ref2VA uses ordered --reference inputs."
            ))
        }
        if input.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "prompt_empty",
                    severity: .blocker,
                    title: "Prompt is empty",
                    message: "Provide a non-empty video prompt."
                )
            )
        }
        if !input.fps.isFinite || input.fps < 1 {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "fps_invalid",
                    severity: .blocker,
                    title: "FPS is invalid",
                    message: "--fps must be finite and >= 1."
                )
            )
        }
        if usesWanGeometry, input.fps.isFinite, input.fps.rounded() != input.fps {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "wan_fps_fractional",
                    severity: .blocker,
                    title: "Wan frame rate must be integral",
                    message: "Wan2.2 TI2V requires an integer --fps value."
                )
            )
        }
        if let duration = input.duration, !duration.isFinite || duration <= 0 {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "duration_invalid",
                    severity: .blocker,
                    title: "Duration is invalid",
                    message: "--duration must be finite and > 0."
                )
            )
        }
        if !input.autoDuration.isEmpty {
            let valid = input.autoDuration.count == 2
                && input.autoDuration[0].isFinite
                && input.autoDuration[1].isFinite
                && input.autoDuration[0] > 0
                && input.autoDuration[1] >= input.autoDuration[0]
            if !valid {
                diagnostics.append(
                    VideoGenerationIssue(
                        id: "auto_duration_invalid",
                        severity: .blocker,
                        title: "Automatic duration range is invalid",
                        message: "--auto-duration requires 0 < MIN_SECONDS <= MAX_SECONDS."
                    )
                )
            }
            if input.numFramesSpecified {
                diagnostics.append(
                    VideoGenerationIssue(
                        id: "auto_duration_ignored_by_num_frames",
                        severity: .warning,
                        title: "Explicit frame count wins",
                        message: "--auto-duration is ignored because --num-frames was supplied."
                    )
                )
            }
            if input.duration != nil {
                diagnostics.append(
                    VideoGenerationIssue(
                        id: "auto_duration_conflict",
                        severity: .blocker,
                        title: "Duration selection is ambiguous",
                        message: "Use --duration or --auto-duration, not both."
                    )
                )
            }
            if usesAudioConditioning {
                diagnostics.append(
                    VideoGenerationIssue(
                        id: "auto_duration_a2vid_unsupported",
                        severity: .blocker,
                        title: "Automatic duration is unavailable for A2Vid",
                        message: "Source-audio A2Vid derives duration from the selected audio segment."
                    )
                )
            }
        }
        if !input.audioStartTime.isFinite || input.audioStartTime < 0 {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "audio_start_time_invalid",
                    severity: .blocker,
                    title: "Audio start time is invalid",
                    message: "--audio-start-time must be finite and >= 0."
                )
            )
        }
        if let audioMaxDuration = input.audioMaxDuration,
           !audioMaxDuration.isFinite || audioMaxDuration <= 0 {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "audio_max_duration_invalid",
                    severity: .blocker,
                    title: "Audio maximum duration is invalid",
                    message: "--audio-max-duration must be finite and > 0."
                )
            )
        }
        if [input.a2vGuidanceScale, input.videoCFGGuidanceScale, input.audioCFGGuidanceScale, input.v2aGuidanceScale]
            .contains(where: { !$0.isFinite || $0 < 0 }) {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "ltx_guidance_invalid",
                    severity: .blocker,
                    title: "LTX guidance is invalid",
                    message: "LTX full/A2Vid guidance scales must be >= 0."
                )
            )
        }
        if input.a2vSteps < 1 {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "a2v_steps_invalid",
                    severity: .blocker,
                    title: "A2Vid steps are invalid",
                    message: "--a2v-steps must be >= 1."
                )
            )
        }
        let minimumSpatialDimension = 32
        let minimumFrameCount = 5
        if input.resolvedOutputWidth < minimumSpatialDimension {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "width_too_small",
                    severity: .blocker,
                    title: "Width is too small",
                    message: "--width must be >= \(minimumSpatialDimension)."
                )
            )
        }
        if input.resolvedOutputHeight < minimumSpatialDimension {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "height_too_small",
                    severity: .blocker,
                    title: "Height is too small",
                    message: "--height must be >= \(minimumSpatialDimension)."
                )
            )
        }
        if input.requestedFrameCount < minimumFrameCount {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "num_frames_too_small",
                    severity: .blocker,
                    title: "Frame count is too small",
                    message: "--num-frames must be >= \(minimumFrameCount)."
                )
            )
        }
        if !(0...1).contains(input.imageStrength) {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "image_strength_invalid",
                    severity: .blocker,
                    title: "Image strength is invalid",
                    message: "--image-strength must be between 0 and 1."
                )
            )
        }
        if !(0...1).contains(input.endImageStrength) {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "end_image_strength_invalid",
                    severity: .blocker,
                    title: "End image strength is invalid",
                    message: "--end-image-strength must be between 0 and 1."
                )
            )
        }
        if input.endImage != nil, input.image == nil {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "end_image_requires_source_image",
                    severity: .blocker,
                    title: "End keyframe needs a source image",
                    message: "--end-image requires --image so the start keyframe is anchored."
                )
            )
        }
        if (input.variant == .unifiedAV || usesAudioConditioning), input.fps > 0, input.fps != 24 {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "unified_av_fps_unusual",
                    severity: .warning,
                    title: "LTX audio/video is tuned for 24 fps",
                    message: "LTX audio/video is trained around 24 fps; --fps \(input.fps) can make motion look time-stretched relative to audio."
                )
            )
        }
        if usesMiniMaxH3Geometry,
           input.fps != Double(MiniMaxH3Geometry.framesPerSecond) {
            diagnostics.append(
                VideoGenerationIssue(
                    id: "minimax_h3_fps_fixed",
                    severity: .note,
                    title: "MiniMax-H3 uses fixed 24 fps",
                    message: "MiniMax-H3 output will use 24 fps; --fps \(input.fps) is ignored."
                )
            )
        }
        diagnostics.append(contentsOf: additionalValidationIssues(profile: profile))
        return diagnostics
    }

}
