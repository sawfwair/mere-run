import Foundation
import ArgumentParser
import MereRunContract
import MereRunCore

struct VideoGenerationPreflightInput {
    let prompt: String
    let outputURL: URL
    let model: String
    let variant: LTXVideoVariant
    let quality: LTXVideoQuality?
    let outputMode: LTXVideoOutputMode?
    let legacyVariant: LTXVideoVariant?
    let productSelectionValidationMessage: String?
    let modelRoot: String?
    let width: Int
    let height: Int
    let numFrames: Int
    let steps: Int?
    let h3WeightMode: String
    let h3AccelerationMode: String
    let h3RenderWidth: Int?
    let h3RenderHeight: Int?
    let h3Adapter: String?
    let h3AdapterStrength: Float
    let h3FrameInputs: [String]
    let h3WindowFrames: Int?
    let h3WindowOverlap: Int
    let duration: Double?
    let fps: Int
    let seed: Int?
    let negativePrompt: String?
    let audio: String?
    let audioStartTime: Double
    let a2vGuidanceScale: Float
    let videoCFGGuidanceScale: Float
    let audioCFGGuidanceScale: Float
    let v2aGuidanceScale: Float
    let a2vSteps: Int
    let image: String?
    let imageStrength: Float
    let endImage: String?
    let endImageStrength: Float
    let references: [String]
    let timings: Bool
    let timingsOutput: String?
    let generationArgv: [String]
    let cwd: String
}

struct VideoGenerationPreflightRequest: Codable, Equatable {
    let prompt: String
    let output: String
    let model: String
    let variant: String
    let quality: String?
    let outputMode: String?
    let modelRoot: String?
    let width: Int
    let height: Int
    let numFrames: Int
    let steps: Int?
    let h3WeightMode: String?
    let h3AccelerationMode: String?
    let h3RenderWidth: Int?
    let h3RenderHeight: Int?
    let h3Adapter: String?
    let h3AdapterStrength: Float?
    let h3FrameInputs: [String]?
    let h3WindowFrames: Int?
    let h3WindowOverlap: Int?
    let duration: Double?
    let fps: Int
    let seed: Int?
    let negativePrompt: String?
    let audio: String?
    let audioStartTime: Double
    let a2vGuidanceScale: Float
    let videoCFGGuidanceScale: Float
    let audioCFGGuidanceScale: Float
    let v2aGuidanceScale: Float
    let a2vSteps: Int
    let image: String?
    let imageStrength: Float
    let endImage: String?
    let endImageStrength: Float
    let references: [String]?
    let timings: Bool?
    let timingsOutput: String?

    enum CodingKeys: String, CodingKey {
        case prompt
        case output
        case model
        case variant
        case quality
        case outputMode = "output_mode"
        case modelRoot = "model_root"
        case width
        case height
        case numFrames = "num_frames"
        case steps
        case h3WeightMode = "h3_weight_mode"
        case h3AccelerationMode = "h3_acceleration"
        case h3RenderWidth = "h3_render_width"
        case h3RenderHeight = "h3_render_height"
        case h3Adapter = "h3_adapter"
        case h3AdapterStrength = "h3_adapter_strength"
        case h3FrameInputs = "h3_frames"
        case h3WindowFrames = "h3_window_frames"
        case h3WindowOverlap = "h3_window_overlap"
        case duration
        case fps
        case seed
        case negativePrompt = "negative_prompt"
        case audio
        case audioStartTime = "audio_start_time"
        case a2vGuidanceScale = "a2v_guidance_scale"
        case videoCFGGuidanceScale = "video_cfg_guidance_scale"
        case audioCFGGuidanceScale = "audio_cfg_guidance_scale"
        case v2aGuidanceScale = "v2a_guidance_scale"
        case a2vSteps = "a2v_steps"
        case image
        case imageStrength = "image_strength"
        case endImage = "end_image"
        case endImageStrength = "end_image_strength"
        case references
        case timings
        case timingsOutput = "timings_output"
    }
}

struct VideoGenerationPreflightResult: Codable, Equatable {
    let model: VideoGenerationModelPreflightSummary
    let output: VideoGenerationOutputPreflightSummary
    let inputs: VideoGenerationInputPreflightSummary
    let plan: VideoGenerationPlanPreflightSummary
}

struct VideoGenerationModelPreflightSummary: Codable, Equatable {
    let requested: String
    let kind: String
    let installed: Bool
    let path: String?
    let id: String?
    let layout: String?
    let upstreamRepoID: String?
    let estimatedDownloadBytes: Int64?
    let companionModelIDs: [String]

    enum CodingKeys: String, CodingKey {
        case requested
        case kind
        case installed
        case path
        case id
        case layout
        case upstreamRepoID = "upstream_repo_id"
        case estimatedDownloadBytes = "estimated_download_bytes"
        case companionModelIDs = "companion_model_ids"
    }
}

struct VideoGenerationOutputPreflightSummary: Codable, Equatable {
    let path: String
    let parentDirectory: String
    let parentExists: Bool
    let parentWillBeCreated: Bool
    let exists: Bool
    let expectedExtension: String
    let extensionValid: Bool

    enum CodingKeys: String, CodingKey {
        case path
        case parentDirectory = "parent_directory"
        case parentExists = "parent_exists"
        case parentWillBeCreated = "parent_will_be_created"
        case exists
        case expectedExtension = "expected_extension"
        case extensionValid = "extension_valid"
    }
}

struct VideoGenerationInputPreflightSummary: Codable, Equatable {
    let mode: String
    let sourceAudio: VideoGenerationPathPreflightSummary?
    let sourceImage: VideoGenerationPathPreflightSummary?
    let endImage: VideoGenerationPathPreflightSummary?
    let h3Frames: [VideoGenerationPathPreflightSummary]?
    let references: [VideoGenerationPathPreflightSummary]?
    let adapter: VideoGenerationPathPreflightSummary?
    let missingCount: Int

    enum CodingKeys: String, CodingKey {
        case mode
        case sourceAudio = "source_audio"
        case sourceImage = "source_image"
        case endImage = "end_image"
        case h3Frames = "h3_frames"
        case references
        case adapter
        case missingCount = "missing_count"
    }
}

struct VideoGenerationPathPreflightSummary: Codable, Equatable {
    let requested: String
    let path: String
    let exists: Bool
    let isDirectory: Bool

    enum CodingKeys: String, CodingKey {
        case requested
        case path
        case exists
        case isDirectory = "is_directory"
    }
}

struct VideoGenerationPlanPreflightSummary: Codable, Equatable {
    let variant: String
    let quality: String?
    let outputMode: String?
    let inputMode: String
    let requestedWidth: Int
    let requestedHeight: Int
    let resolvedWidth: Int
    let resolvedHeight: Int
    let requestedNumFrames: Int
    let requestedDurationSeconds: Double?
    let resolvedSteps: Int?
    let h3WeightMode: String?
    let h3AccelerationMode: String?
    let h3RenderWidth: Int?
    let h3RenderHeight: Int?
    let h3Adapter: String?
    let h3AdapterStrength: Float?
    let h3FrameCount: Int?
    let h3WindowFrames: Int?
    let h3WindowOverlap: Int?
    let h3WindowCount: Int?
    let fps: Int
    let resolvedNumFrames: Int
    let resolvedDurationSeconds: Double?
    let seed: Int
    let writesAudio: Bool
    let audioConditioning: Bool
    let preservesSourceAudio: Bool
    let resolvedAudioStartTime: Double?

    enum CodingKeys: String, CodingKey {
        case variant
        case quality
        case outputMode = "output_mode"
        case inputMode = "input_mode"
        case requestedWidth = "requested_width"
        case requestedHeight = "requested_height"
        case resolvedWidth = "resolved_width"
        case resolvedHeight = "resolved_height"
        case requestedNumFrames = "requested_num_frames"
        case requestedDurationSeconds = "requested_duration_seconds"
        case resolvedSteps = "resolved_steps"
        case h3WeightMode = "h3_weight_mode"
        case h3AccelerationMode = "h3_acceleration"
        case h3RenderWidth = "h3_render_width"
        case h3RenderHeight = "h3_render_height"
        case h3Adapter = "h3_adapter"
        case h3AdapterStrength = "h3_adapter_strength"
        case h3FrameCount = "h3_frame_count"
        case h3WindowFrames = "h3_window_frames"
        case h3WindowOverlap = "h3_window_overlap"
        case h3WindowCount = "h3_window_count"
        case fps
        case resolvedNumFrames = "resolved_num_frames"
        case resolvedDurationSeconds = "resolved_duration_seconds"
        case seed
        case writesAudio = "writes_audio"
        case audioConditioning = "audio_conditioning"
        case preservesSourceAudio = "preserves_source_audio"
        case resolvedAudioStartTime = "resolved_audio_start_time"
    }
}

typealias VideoGenerationPreflightEnvelope = StructuredRunEnvelope<
    VideoGenerationPreflightRequest,
    VideoGenerationPreflightResult
>

struct VideoGenerationPreflightAnalyzer {
    let input: VideoGenerationPreflightInput
    let fileManager: FileManager
    let now: () -> Date

    init(
        input: VideoGenerationPreflightInput,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) {
        self.input = input
        self.fileManager = fileManager
        self.now = now
    }

    private var usesWanGeometry: Bool {
        if input.model.trimmingCharacters(in: .whitespacesAndNewlines) == Wan2Resources.modelID {
            return true
        }
        let candidate = input.modelRoot ?? input.model
        let url = URL(fileURLWithPath: candidate).standardizedFileURL
        let resources = Wan2Resources(rootURL: url)
        return resources.validate().isEmpty && (try? resources.loadConfiguration()) != nil
    }

    private var usesMiniMaxH3Geometry: Bool {
        let requested = input.model.trimmingCharacters(in: .whitespacesAndNewlines)
        if requested == ModelResolver.ModelID.miniMaxH3FL2VAMLX.rawValue
            || requested == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue
            || requested == ModelResolver.ModelID.miniMaxH3Ref2VAMLX.rawValue {
            return true
        }
        let candidate = input.modelRoot ?? input.model
        let resources = MiniMaxH3Resources(rootURL: URL(fileURLWithPath: candidate).standardizedFileURL)
        return resources.validate().isEmpty && (try? resources.loadConfiguration()) != nil
    }

    private var usesAudioConditioning: Bool {
        guard let audio = input.audio else { return false }
        return !audio.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func envelope() -> VideoGenerationPreflightEnvelope {
        var diagnostics: [PreflightDiagnostic] = []
        validateStaticOptions(diagnostics: &diagnostics)
        let model = modelSummary(diagnostics: &diagnostics)
        validateProductSelection(model: model, diagnostics: &diagnostics)
        validateTimingOptions(model: model, diagnostics: &diagnostics)
        let output = outputSummary(diagnostics: &diagnostics)
        let inputs = inputSummary(diagnostics: &diagnostics)
        let plan = planSummary(model: model, inputs: inputs, diagnostics: &diagnostics)
        let status = StructuredRunOutput.status(for: diagnostics)

        return VideoGenerationPreflightEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["video", "generate"],
            mode: .preflight,
            status: status,
            createdAt: now(),
            cwd: input.cwd,
            summary: summary(status: status, diagnostics: diagnostics),
            request: request(),
            result: VideoGenerationPreflightResult(
                model: model,
                output: output,
                inputs: inputs,
                plan: plan
            ),
            diagnostics: diagnostics,
            actions: actions(status: status, model: model, output: output, inputs: inputs)
        )
    }

    private func request() -> VideoGenerationPreflightRequest {
        VideoGenerationPreflightRequest(
            prompt: input.prompt,
            output: input.outputURL.path,
            model: input.model,
            variant: input.variant.rawValue,
            quality: input.quality?.rawValue,
            outputMode: input.outputMode?.rawValue,
            modelRoot: input.modelRoot,
            width: input.width,
            height: input.height,
            numFrames: input.numFrames,
            steps: input.steps,
            h3WeightMode: usesMiniMaxH3Geometry ? input.h3WeightMode : nil,
            h3AccelerationMode: usesMiniMaxH3Geometry ? input.h3AccelerationMode : nil,
            h3RenderWidth: usesMiniMaxH3Geometry ? input.h3RenderWidth : nil,
            h3RenderHeight: usesMiniMaxH3Geometry ? input.h3RenderHeight : nil,
            h3Adapter: usesMiniMaxH3Geometry ? input.h3Adapter : nil,
            h3AdapterStrength: usesMiniMaxH3Geometry && input.h3Adapter != nil
                ? input.h3AdapterStrength
                : nil,
            h3FrameInputs: usesMiniMaxH3Geometry && !input.h3FrameInputs.isEmpty
                ? input.h3FrameInputs
                : nil,
            h3WindowFrames: usesMiniMaxH3Geometry ? input.h3WindowFrames : nil,
            h3WindowOverlap: usesMiniMaxH3Geometry && input.h3WindowFrames != nil
                ? input.h3WindowOverlap
                : nil,
            duration: input.duration,
            fps: input.fps,
            seed: input.seed,
            negativePrompt: input.negativePrompt,
            audio: input.audio,
            audioStartTime: input.audioStartTime,
            a2vGuidanceScale: input.a2vGuidanceScale,
            videoCFGGuidanceScale: input.videoCFGGuidanceScale,
            audioCFGGuidanceScale: input.audioCFGGuidanceScale,
            v2aGuidanceScale: input.v2aGuidanceScale,
            a2vSteps: input.a2vSteps,
            image: input.image,
            imageStrength: input.imageStrength,
            endImage: input.endImage,
            endImageStrength: input.endImageStrength,
            references: input.references.isEmpty ? nil : input.references,
            timings: input.timings,
            timingsOutput: input.timingsOutput
        )
    }

    private func validateTimingOptions(
        model: VideoGenerationModelPreflightSummary,
        diagnostics: inout [PreflightDiagnostic]
    ) {
        guard input.timings || input.timingsOutput != nil else { return }
        guard !usesAudioConditioning else { return }
        guard !usesMiniMaxH3Geometry else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "timings_lane_unsupported",
                    severity: .blocker,
                    title: "Phase timings are unavailable for this lane",
                    message: "--timings and --timings-output are not available for MiniMax-H3 yet."
                )
            )
            return
        }
        guard !usesWanGeometry else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "timings_lane_unsupported",
                    severity: .blocker,
                    title: "Phase timings are unavailable for this lane",
                    message: "--timings and --timings-output are available for native LTX generation, not Wan2.2 TI2V."
                )
            )
            return
        }

        let route = resolvedLTXRoute(model: model)
        if route?.supportsPhaseTimings == false {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "timings_lane_unsupported",
                    severity: .blocker,
                    title: "Phase timings are unavailable for this lane",
                    message: "Use an LTX 2.3 split model, --quality final, --output-mode audio-video, or --audio for phase timings."
                )
            )
        }
    }

    private func resolvedLTXRoute(
        model: VideoGenerationModelPreflightSummary
    ) -> LTXVideoGenerationRoute? {
        if let path = model.path {
            return resolveLTXVideoGenerationRoute(
                variant: input.variant,
                modelRoot: URL(fileURLWithPath: path),
                fileManager: fileManager
            )
        }
        if input.variant == .unifiedAV {
            return .unifiedAV
        }
        if input.model == ModelResolver.ModelID.ltxVideo23AVMLX.rawValue {
            return .splitDistilledVideo
        }
        return nil
    }

    private func validateStaticOptions(diagnostics: inout [PreflightDiagnostic]) {
        if let message = input.productSelectionValidationMessage {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "video_product_selection_conflict",
                    severity: .blocker,
                    title: "Video product selection is ambiguous",
                    message: message
                )
            )
        }
        if usesWanGeometry, input.quality != nil || input.outputMode != nil {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "ltx_product_selection_with_wan",
                    severity: .blocker,
                    title: "LTX product options do not apply to Wan",
                    message: "--quality and --output-mode currently select native LTX generation, not Wan2.2 TI2V."
                )
            )
        }
        if usesMiniMaxH3Geometry, input.quality != nil || input.outputMode != nil || input.legacyVariant != nil {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "ltx_product_selection_with_minimax_h3",
                    severity: .blocker,
                    title: "LTX product options do not apply to MiniMax-H3",
                    message: "--quality, --output-mode, and --variant cannot be combined with MiniMax-H3."
                )
            )
        }
        if !usesMiniMaxH3Geometry, input.h3Adapter != nil {
            diagnostics.append(PreflightDiagnostic(
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
            diagnostics.append(PreflightDiagnostic(
                id: "h3_window_or_frame_with_non_h3_model",
                severity: .blocker,
                title: "MiniMax-H3 controls require MiniMax-H3",
                message: "H3 frame, window, and internal-render controls require a MiniMax-H3 model."
            ))
        }
        if usesMiniMaxH3Geometry,
           (input.h3RenderWidth == nil) != (input.h3RenderHeight == nil) {
            diagnostics.append(PreflightDiagnostic(
                id: "h3_render_canvas_incomplete",
                severity: .blocker,
                title: "MiniMax-H3 internal render canvas is incomplete",
                message: "--h3-render-width and --h3-render-height must be set together."
            ))
        }
        if !input.references.isEmpty, !input.h3FrameInputs.isEmpty {
            diagnostics.append(PreflightDiagnostic(
                id: "h3_frame_ref2va_unsupported",
                severity: .blocker,
                title: "Timed H3 frames require FL2VA",
                message: "Use --h3-frame with FL2VA; Ref2VA uses ordered --reference inputs."
            ))
        }
        if input.h3Adapter != nil, input.h3AdapterStrength <= 0 {
            diagnostics.append(PreflightDiagnostic(
                id: "h3_adapter_strength_invalid",
                severity: .blocker,
                title: "MiniMax-H3 adapter strength is invalid",
                message: "--h3-adapter-strength must be > 0."
            ))
        }
        if input.h3Adapter != nil, !input.references.isEmpty {
            diagnostics.append(PreflightDiagnostic(
                id: "h3_adapter_ref2va_unsupported",
                severity: .blocker,
                title: "MiniMax-H3 Turbo does not support Ref2VA",
                message: "Use the Turbo adapter for FL2VA text or keyframe generation without --reference."
            ))
        }
        if input.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "prompt_empty",
                    severity: .blocker,
                    title: "Prompt is empty",
                    message: "Provide a non-empty video prompt."
                )
            )
        }
        if input.fps < 1 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "fps_invalid",
                    severity: .blocker,
                    title: "FPS is invalid",
                    message: "--fps must be >= 1."
                )
            )
        }
        if let duration = input.duration, duration <= 0 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "duration_invalid",
                    severity: .blocker,
                    title: "Duration is invalid",
                    message: "--duration must be > 0."
                )
            )
        }
        if !input.audioStartTime.isFinite || input.audioStartTime < 0 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "audio_start_time_invalid",
                    severity: .blocker,
                    title: "Audio start time is invalid",
                    message: "--audio-start-time must be finite and >= 0."
                )
            )
        }
        if input.a2vGuidanceScale < 0
            || input.videoCFGGuidanceScale < 0
            || input.audioCFGGuidanceScale < 0
            || input.v2aGuidanceScale < 0 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "ltx_guidance_invalid",
                    severity: .blocker,
                    title: "LTX guidance is invalid",
                    message: "LTX full/A2Vid guidance scales must be >= 0."
                )
            )
        }
        if input.a2vSteps < 1 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "a2v_steps_invalid",
                    severity: .blocker,
                    title: "A2Vid steps are invalid",
                    message: "--a2v-steps must be >= 1."
                )
            )
        }
        if usesAudioConditioning,
           input.modelRoot == nil,
           ModelResolver.ModelID(rawValue: input.model) != nil,
           input.model != ModelResolver.ModelID.ltxVideo23FullMLX.rawValue,
           input.model != ModelResolver.ModelID.ltxVideo23A2VMLX.rawValue {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "audio_model_incompatible",
                    severity: .blocker,
                    title: "Model does not support audio conditioning",
                    message: "--audio requires \(ModelResolver.ModelID.ltxVideo23FullMLX.rawValue)."
                )
            )
        }
        let minimumSpatialDimension = usesWanGeometry || usesMiniMaxH3Geometry ? 32 : 64
        let minimumFrameCount = usesMiniMaxH3Geometry ? 22 : (usesWanGeometry ? 5 : 9)
        if input.width < minimumSpatialDimension {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "width_too_small",
                    severity: .blocker,
                    title: "Width is too small",
                    message: "--width must be >= \(minimumSpatialDimension)."
                )
            )
        }
        if input.height < minimumSpatialDimension {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "height_too_small",
                    severity: .blocker,
                    title: "Height is too small",
                    message: "--height must be >= \(minimumSpatialDimension)."
                )
            )
        }
        if input.numFrames < minimumFrameCount {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "num_frames_too_small",
                    severity: .blocker,
                    title: "Frame count is too small",
                    message: "--num-frames must be >= \(minimumFrameCount)."
                )
            )
        }
        if !(0...1).contains(input.imageStrength) {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "image_strength_invalid",
                    severity: .blocker,
                    title: "Image strength is invalid",
                    message: "--image-strength must be between 0 and 1."
                )
            )
        }
        if !(0...1).contains(input.endImageStrength) {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "end_image_strength_invalid",
                    severity: .blocker,
                    title: "End image strength is invalid",
                    message: "--end-image-strength must be between 0 and 1."
                )
            )
        }
        if input.endImage != nil, input.image == nil {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "end_image_requires_source_image",
                    severity: .blocker,
                    title: "End keyframe needs a source image",
                    message: "--end-image requires --image so the start keyframe is anchored."
                )
            )
        }
        if (input.variant == .unifiedAV || usesAudioConditioning), input.fps > 0, input.fps != 24 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "unified_av_fps_unusual",
                    severity: .warning,
                    title: "LTX audio/video is tuned for 24 fps",
                    message: "LTX audio/video is trained around 24 fps; --fps \(input.fps) can make motion look time-stretched relative to audio."
                )
            )
        }
        if usesMiniMaxH3Geometry, input.fps != MiniMaxH3Geometry.framesPerSecond {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "minimax_h3_fps_fixed",
                    severity: .note,
                    title: "MiniMax-H3 uses fixed 24 fps",
                    message: "MiniMax-H3 output will use 24 fps; --fps \(input.fps) is ignored."
                )
            )
        }
    }

    private func validateProductSelection(
        model: VideoGenerationModelPreflightSummary,
        diagnostics: inout [PreflightDiagnostic]
    ) {
        if input.h3Adapter != nil, usesMiniMaxH3Geometry {
            let usesBF16: Bool
            if let path = model.path {
                usesBF16 = MiniMaxH3Resources(
                    rootURL: URL(fileURLWithPath: path).standardizedFileURL
                ).usesShardedBF16Transformer
            } else {
                usesBF16 = model.requested == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue
            }
            if !usesBF16 {
                diagnostics.append(PreflightDiagnostic(
                    id: "h3_adapter_requires_bf16",
                    severity: .blocker,
                    title: "MiniMax-H3 Turbo requires the BF16 base model",
                    message: "Use --model \(ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue)."
                ))
            }
            if let steps = input.steps,
               steps != MiniMaxH3TurboAdapter.recommendedSchedulePointCount {
                diagnostics.append(PreflightDiagnostic(
                    id: "h3_adapter_steps_invalid",
                    severity: .blocker,
                    title: "MiniMax-H3 Turbo uses four denoise evaluations",
                    message: "Omit --steps or set --steps 5 (five schedule points)."
                ))
            }
        }
        guard let requestedQuality = input.quality,
              let actualQuality = resolvedQuality(model: model),
              requestedQuality != actualQuality else {
            return
        }
        let requiredModel = requestedQuality == .final
            ? ModelResolver.ModelID.ltxVideo23FullMLX.rawValue
            : ModelResolver.ModelID.ltxVideo23AVMLX.rawValue
        diagnostics.append(
            PreflightDiagnostic(
                id: "video_quality_model_mismatch",
                severity: .blocker,
                title: "Checkpoint does not match requested quality",
                message: "--quality \(requestedQuality.rawValue) requires \(requiredModel)."
            )
        )
    }

    private func resolvedQuality(
        model: VideoGenerationModelPreflightSummary
    ) -> LTXVideoQuality? {
        switch model.layout {
        case "ltx23_full_split", "ltx23_a2vid_split":
            return .final
        case "ltx23_distilled_split", "ltx_merged":
            return .draft
        default:
            return nil
        }
    }

    private func modelSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> VideoGenerationModelPreflightSummary {
        if let modelRoot = input.modelRoot?.trimmingCharacters(in: .whitespacesAndNewlines), !modelRoot.isEmpty {
            return localModelSummary(
                requested: modelRoot,
                kind: "model_root",
                diagnostics: &diagnostics
            )
        }

        let trimmedModel = input.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let requested = trimmedModel.isEmpty ? ModelResolver.ModelID.ltxVideo23AVMLX.rawValue : trimmedModel
        let localURL = URL(fileURLWithPath: requested).standardizedFileURL
        if fileManager.fileExists(atPath: localURL.path) {
            return localModelSummary(
                requested: requested,
                kind: "local_path",
                diagnostics: &diagnostics
            )
        }

        guard let modelID = ModelResolver.ModelID(rawValue: requested),
              let spec = ManagedModelCatalog.spec(for: requested) else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_unknown",
                    severity: .blocker,
                    title: "Unknown model",
                    message: "Model path not found and not a known model id: \(requested)."
                )
            )
            return modelResult(requested: requested, kind: "unknown", installed: false)
        }

        if let resolution = ModelResolver(fileManager: fileManager).resolveIfPresent(modelID) {
            return installedManagedModelSummary(
                requested: requested,
                spec: spec,
                path: resolution.rootURL,
                diagnostics: &diagnostics
            )
        }

        diagnostics.append(
            PreflightDiagnostic(
                id: "model_missing",
                severity: .blocker,
                title: "Model missing",
                message: "Model \(requested) is not installed. Pull it before video generation.",
                suggestedActionIDs: ["pull-model"]
            )
        )
        return modelResult(
            requested: requested,
            kind: "managed_model",
            installed: false,
            id: requested,
            upstreamRepoID: spec.upstreamRepoId,
            estimatedDownloadBytes: spec.estimatedDownloadBytes,
            companionModelIDs: spec.companionModelIDs
        )
    }

    private func localModelSummary(
        requested: String,
        kind: String,
        diagnostics: inout [PreflightDiagnostic]
    ) -> VideoGenerationModelPreflightSummary {
        let url = URL(fileURLWithPath: requested).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_root_missing",
                    severity: .blocker,
                    title: "Model root missing",
                    message: "Model root not found: \(url.path)",
                    locations: [.init(kind: "directory", path: url.path)]
                )
            )
            return modelResult(requested: requested, kind: kind, installed: false, path: url.path)
        }
        guard isDirectory.boolValue else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_root_not_directory",
                    severity: .blocker,
                    title: "Model root is not a directory",
                    message: "Model root is not a directory: \(url.path)",
                    locations: [.init(kind: "file", path: url.path)]
                )
            )
            return modelResult(requested: requested, kind: kind, installed: false, path: url.path)
        }

        do {
            try validateSelectedModelRoot(url)
            return modelResult(
                requested: requested,
                kind: kind,
                installed: true,
                path: url.path,
                layout: videoLayout(at: url)
            )
        } catch {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_root_invalid",
                    severity: .blocker,
                    title: "Model root is invalid",
                    message: error.localizedDescription,
                    locations: [.init(kind: "directory", path: url.path)]
                )
            )
            return modelResult(
                requested: requested,
                kind: kind,
                installed: false,
                path: url.path,
                layout: videoLayout(at: url)
            )
        }
    }

    private func installedManagedModelSummary(
        requested: String,
        spec: ManagedModelSpec,
        path: URL,
        diagnostics: inout [PreflightDiagnostic]
    ) -> VideoGenerationModelPreflightSummary {
        do {
            try validateSelectedModelRoot(path)
            return modelResult(
                requested: requested,
                kind: "managed_model",
                installed: true,
                path: path.path,
                id: spec.id,
                layout: videoLayout(at: path),
                upstreamRepoID: spec.upstreamRepoId,
                estimatedDownloadBytes: spec.estimatedDownloadBytes,
                companionModelIDs: spec.companionModelIDs
            )
        } catch {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_root_invalid",
                    severity: .blocker,
                    title: "Installed model root is invalid",
                    message: error.localizedDescription,
                    locations: [.init(kind: "directory", path: path.path)]
                )
            )
            return modelResult(
                requested: requested,
                kind: "managed_model",
                installed: false,
                path: path.path,
                id: spec.id,
                layout: videoLayout(at: path),
                upstreamRepoID: spec.upstreamRepoId,
                estimatedDownloadBytes: spec.estimatedDownloadBytes,
                companionModelIDs: spec.companionModelIDs
            )
        }
    }

    private func modelResult(
        requested: String,
        kind: String,
        installed: Bool,
        path: String? = nil,
        id: String? = nil,
        layout: String? = nil,
        upstreamRepoID: String? = nil,
        estimatedDownloadBytes: Int64? = nil,
        companionModelIDs: [String] = []
    ) -> VideoGenerationModelPreflightSummary {
        VideoGenerationModelPreflightSummary(
            requested: requested,
            kind: kind,
            installed: installed,
            path: path,
            id: id,
            layout: layout,
            upstreamRepoID: upstreamRepoID,
            estimatedDownloadBytes: estimatedDownloadBytes,
            companionModelIDs: companionModelIDs
        )
    }

    private func videoLayout(at url: URL) -> String? {
        let h3 = MiniMaxH3Resources(rootURL: url)
        if h3.validate().isEmpty, let configuration = try? h3.loadConfiguration() {
            return "minimax_h3_\(configuration.task)_mlx"
        }
        let resources = Wan2Resources(rootURL: url)
        if resources.validate().isEmpty, (try? resources.loadConfiguration()) != nil {
            return "wan22_ti2v_mlx"
        }
        if isLTX23FullModelRoot(url, fileManager: fileManager) {
            return "ltx23_full_split"
        }
        if isLTX23AudioToVideoModelRoot(url, fileManager: fileManager) {
            return "ltx23_a2vid_split"
        }
        return isLTX23SplitModelRoot(url, fileManager: fileManager)
            ? "ltx23_distilled_split"
            : "ltx_merged"
    }

    private func validateSelectedModelRoot(_ url: URL) throws {
        if usesMiniMaxH3Geometry {
            let resources = MiniMaxH3Resources(rootURL: url)
            let missing = resources.validate(fileManager: fileManager)
            guard missing.isEmpty else {
                throw ValidationError("Missing MiniMax-H3 files: \(missing.map(\.path).joined(separator: ", "))")
            }
            _ = try resources.loadConfiguration()
        } else if usesAudioConditioning {
            try validateNativeAudioToVideoModelRoot(url, fileManager: fileManager)
        } else if input.variant == .unifiedAV,
                  isLTX23AudioToVideoModelRoot(url, fileManager: fileManager),
                  !isLTX23FullModelRoot(url, fileManager: fileManager) {
            throw ValidationError(
                "This legacy A2Vid root has no vocoder for unified AV. Pull \(ModelResolver.ModelID.ltxVideo23FullMLX.rawValue)."
            )
        } else {
            try validateNativeModelRoot(url)
        }
    }

    private func outputSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> VideoGenerationOutputPreflightSummary {
        let parent = input.outputURL.deletingLastPathComponent()
        var parentIsDirectory: ObjCBool = false
        let parentExists = fileManager.fileExists(atPath: parent.path, isDirectory: &parentIsDirectory)
        let outputExists = fileManager.fileExists(atPath: input.outputURL.path)
        let extensionValid = input.outputURL.pathExtension.lowercased() == "mp4"

        if parentExists, !parentIsDirectory.boolValue {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "output_parent_not_directory",
                    severity: .blocker,
                    title: "Output parent is not a directory",
                    message: "Output parent is not a directory: \(parent.path)",
                    locations: [.init(kind: "file", path: parent.path)]
                )
            )
        }
        if outputExists {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "output_exists",
                    severity: .warning,
                    title: "Output exists",
                    message: "Output already exists and may be overwritten: \(input.outputURL.path)",
                    locations: [.init(kind: "file", path: input.outputURL.path)]
                )
            )
        }
        if !extensionValid {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "output_extension_unusual",
                    severity: .warning,
                    title: "Output extension is not MP4",
                    message: "Video generation writes MP4 data; use a .mp4 output path for clarity.",
                    locations: [.init(kind: "file", path: input.outputURL.path)]
                )
            )
        }

        return VideoGenerationOutputPreflightSummary(
            path: input.outputURL.path,
            parentDirectory: parent.path,
            parentExists: parentExists && parentIsDirectory.boolValue,
            parentWillBeCreated: !parentExists,
            exists: outputExists,
            expectedExtension: "mp4",
            extensionValid: extensionValid
        )
    }

    private func inputSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> VideoGenerationInputPreflightSummary {
        let sourceAudio = usesAudioConditioning ? input.audio.map { pathSummary(requested: $0) } : nil
        let sourceImage = input.image.map { pathSummary(requested: $0) }
        let endImage = input.endImage.map { pathSummary(requested: $0) }
        let h3Frames = input.h3FrameInputs.map { raw -> VideoGenerationPathPreflightSummary in
            let path = raw.firstIndex(of: ":").map { String(raw[raw.index(after: $0)...]) } ?? raw
            return pathSummary(requested: path)
        }
        let references = input.references.map { raw -> VideoGenerationPathPreflightSummary in
            let path = raw.firstIndex(of: ":").map { String(raw[raw.index(after: $0)...]) } ?? raw
            return pathSummary(requested: path)
        }
        let adapter = input.h3Adapter.map { reference -> VideoGenerationPathPreflightSummary in
            let path = ManagedAdapterCatalog.spec(for: reference)?.installedFileURL().path ?? reference
            return pathSummary(requested: reference, resolvedPath: path)
        }
        for (summary, prefix) in [
            (sourceAudio, "source_audio"),
            (sourceImage, "source_image"),
            (endImage, "end_image"),
        ] {
            guard let summary else { continue }
            if !summary.exists {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "\(prefix)_missing",
                        severity: .blocker,
                        title: "Input media missing",
                        message: "Input media not found: \(summary.path)",
                        locations: [.init(kind: "file", path: summary.path)]
                    )
                )
            } else if summary.isDirectory {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "\(prefix)_is_directory",
                        severity: .blocker,
                        title: "Input media is a directory",
                        message: "Input media path is a directory: \(summary.path)",
                        locations: [.init(kind: "directory", path: summary.path)]
                    )
                )
            }
        }
        for (index, summary) in references.enumerated() {
            if !summary.exists {
                diagnostics.append(PreflightDiagnostic(
                    id: "reference_\(index)_missing",
                    severity: .blocker,
                    title: "Reference media missing",
                    message: "Reference media not found: \(summary.path)",
                    locations: [.init(kind: "file", path: summary.path)]
                ))
            } else if summary.isDirectory {
                diagnostics.append(PreflightDiagnostic(
                    id: "reference_\(index)_is_directory",
                    severity: .blocker,
                    title: "Reference media is a directory",
                    message: "Reference media path is a directory: \(summary.path)",
                    locations: [.init(kind: "directory", path: summary.path)]
                ))
            }
        }
        for (index, summary) in h3Frames.enumerated() {
            if !summary.exists {
                diagnostics.append(PreflightDiagnostic(
                    id: "h3_frame_\(index)_missing",
                    severity: .blocker,
                    title: "Timed H3 frame missing",
                    message: "Timed frame image not found: \(summary.path)",
                    locations: [.init(kind: "file", path: summary.path)]
                ))
            } else if summary.isDirectory {
                diagnostics.append(PreflightDiagnostic(
                    id: "h3_frame_\(index)_is_directory",
                    severity: .blocker,
                    title: "Timed H3 frame is a directory",
                    message: "Timed frame path is a directory: \(summary.path)",
                    locations: [.init(kind: "directory", path: summary.path)]
                ))
            }
        }
        if let adapter, !adapter.exists {
            let pullHint = ManagedAdapterCatalog.spec(for: adapter.requested).map {
                " Run `mere.run adapter pull \($0.id)`."
            } ?? ""
            diagnostics.append(PreflightDiagnostic(
                id: "h3_adapter_missing",
                severity: .blocker,
                title: "MiniMax-H3 adapter missing",
                message: "Adapter not found: \(adapter.path).\(pullHint)",
                locations: [.init(kind: "file", path: adapter.path)]
            ))
        } else if let adapter, adapter.isDirectory {
            diagnostics.append(PreflightDiagnostic(
                id: "h3_adapter_is_directory",
                severity: .blocker,
                title: "MiniMax-H3 adapter path is a directory",
                message: "Adapter path must be a safetensors file: \(adapter.path)",
                locations: [.init(kind: "directory", path: adapter.path)]
            ))
        }

        let mode: String
        if !references.isEmpty {
            mode = "reference_to_video_audio"
        } else if sourceAudio != nil {
            if sourceImage == nil {
                mode = "audio_to_video"
            } else if endImage == nil {
                mode = "audio_and_image_to_video"
            } else {
                mode = "audio_and_directed_image_to_video"
            }
        } else {
            mode = sourceImage == nil
                ? "text_to_video"
                : (endImage == nil ? "image_to_video" : "directed_image_to_video")
        }
        let allInputs = [sourceAudio, sourceImage, endImage, adapter].compactMap { $0 }
            + h3Frames
            + references
        return VideoGenerationInputPreflightSummary(
            mode: mode,
            sourceAudio: sourceAudio,
            sourceImage: sourceImage,
            endImage: endImage,
            h3Frames: h3Frames.isEmpty ? nil : h3Frames,
            references: references.isEmpty ? nil : references,
            adapter: adapter,
            missingCount: allInputs.filter { !$0.exists }.count
        )
    }

    private func pathSummary(
        requested: String,
        resolvedPath: String? = nil
    ) -> VideoGenerationPathPreflightSummary {
        let url = URL(fileURLWithPath: resolvedPath ?? requested).standardizedFileURL
        var isDirectory: ObjCBool = false
        let exists = fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
        return VideoGenerationPathPreflightSummary(
            requested: requested,
            path: url.path,
            exists: exists,
            isDirectory: exists && isDirectory.boolValue
        )
    }

    private func planSummary(
        model: VideoGenerationModelPreflightSummary,
        inputs: VideoGenerationInputPreflightSummary,
        diagnostics: inout [PreflightDiagnostic]
    ) -> VideoGenerationPlanPreflightSummary {
        let spatialMultiple = usesWanGeometry || usesMiniMaxH3Geometry ? 32 : 64
        let temporalMultiple = usesWanGeometry ? 4 : 8
        let minimumFrames = usesMiniMaxH3Geometry ? 22 : (usesWanGeometry ? 5 : 9)
        let resolvedWidth = max(spatialMultiple, (input.width / spatialMultiple) * spatialMultiple)
        let resolvedHeight = max(spatialMultiple, (input.height / spatialMultiple) * spatialMultiple)
        let requestedFrames = input.duration.map {
            usesMiniMaxH3Geometry
                ? Int(($0 * Double(MiniMaxH3Geometry.framesPerSecond)).rounded())
                : usesWanGeometry
                ? nearestWanFrameCount(duration: $0, fps: input.fps)
                : nearestLTXFrameCount(duration: $0, fps: input.fps)
        } ?? input.numFrames
        let resolvedFrames = usesMiniMaxH3Geometry
            ? (try? MiniMaxH3Geometry.alignFrameCount(max(minimumFrames, requestedFrames))) ?? minimumFrames
            : max(minimumFrames, ((requestedFrames - 1) / temporalMultiple) * temporalMultiple + 1)
        var validH3RenderWidth: Int?
        var validH3RenderHeight: Int?
        if usesMiniMaxH3Geometry,
           let renderWidth = input.h3RenderWidth,
           let renderHeight = input.h3RenderHeight {
            let (leftAspect, leftOverflow) = renderWidth.multipliedReportingOverflow(by: resolvedHeight)
            let (rightAspect, rightOverflow) = renderHeight.multipliedReportingOverflow(by: resolvedWidth)
            if renderWidth < 32
                || renderHeight < 32
                || !renderWidth.isMultiple(of: 32)
                || !renderHeight.isMultiple(of: 32)
                || renderWidth > resolvedWidth
                || renderHeight > resolvedHeight
                || leftOverflow
                || rightOverflow
                || leftAspect != rightAspect {
                diagnostics.append(PreflightDiagnostic(
                    id: "h3_render_canvas_invalid",
                    severity: .blocker,
                    title: "MiniMax-H3 internal render canvas is invalid",
                    message: "Internal render dimensions must preserve output aspect, use 32px multiples, and not exceed the resolved output canvas."
                ))
            } else {
                validH3RenderWidth = renderWidth
                validH3RenderHeight = renderHeight
            }
        }
        if usesMiniMaxH3Geometry,
           input.h3WindowFrames != nil,
           validH3RenderWidth != nil,
           (validH3RenderWidth != resolvedWidth || validH3RenderHeight != resolvedHeight) {
            diagnostics.append(PreflightDiagnostic(
                id: "h3_render_canvas_sliding_window_unsupported",
                severity: .blocker,
                title: "Reduced H3 rendering cannot use sliding windows yet",
                message: "Run a single H3 window or remove --h3-render-width and --h3-render-height."
            ))
        }
        let h3FrameIndices = input.h3FrameInputs.compactMap { value -> Int? in
            guard let separator = value.firstIndex(of: ":") else { return nil }
            return Int(value[..<separator])
        }
        if usesMiniMaxH3Geometry, h3FrameIndices.count != input.h3FrameInputs.count {
            diagnostics.append(PreflightDiagnostic(
                id: "h3_frame_syntax_invalid",
                severity: .blocker,
                title: "Timed H3 frame syntax is invalid",
                message: "Every --h3-frame value must use zero-based FRAME:PATH syntax."
            ))
        }
        if usesMiniMaxH3Geometry,
           (Set(h3FrameIndices).count != h3FrameIndices.count
            || h3FrameIndices.contains(where: { !(0..<resolvedFrames).contains($0) })) {
            diagnostics.append(PreflightDiagnostic(
                id: "h3_frame_index_invalid",
                severity: .blocker,
                title: "Timed H3 frame indices are invalid",
                message: "--h3-frame indices must be unique and inside the resolved output timeline."
            ))
        }
        var slidingWindowPlan: MiniMaxH3SlidingWindowPlan?
        if usesMiniMaxH3Geometry, let windowFrames = input.h3WindowFrames {
            do {
                let options = try MiniMaxH3SlidingWindowOptions(
                    totalFrameCount: resolvedFrames,
                    windowFrameCount: windowFrames,
                    overlapFrameCount: input.h3WindowOverlap
                )
                slidingWindowPlan = MiniMaxH3SlidingWindowPlan(options: options)
            } catch {
                diagnostics.append(PreflightDiagnostic(
                    id: "h3_sliding_window_invalid",
                    severity: .blocker,
                    title: "MiniMax-H3 sliding window is invalid",
                    message: error.localizedDescription
                ))
            }
        }

        if input.width >= spatialMultiple,
           input.height >= spatialMultiple,
           resolvedWidth != input.width || resolvedHeight != input.height {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "dimensions_will_be_adjusted",
                    severity: .note,
                    title: "Dimensions will be adjusted",
                    message: "Video dimensions will be snapped from \(input.width)x\(input.height) to \(resolvedWidth)x\(resolvedHeight)."
                )
            )
        }
        if input.numFrames >= minimumFrames, input.duration == nil, resolvedFrames != input.numFrames {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "num_frames_will_be_adjusted",
                    severity: .note,
                    title: "Frame count will be adjusted",
                    message: usesMiniMaxH3Geometry
                        ? "Frame count will be snapped from \(input.numFrames) to \(resolvedFrames) to satisfy 17*n+5."
                        : "Frame count will be snapped from \(input.numFrames) to \(resolvedFrames) to satisfy \(temporalMultiple)n+1."
                )
            )
        }
        if let duration = input.duration, input.fps > 0, duration > 0 {
            let outputFPS = usesMiniMaxH3Geometry ? MiniMaxH3Geometry.framesPerSecond : input.fps
            let resolvedSeconds = Double(resolvedFrames) / Double(outputFPS)
            diagnostics.append(
                PreflightDiagnostic(
                    id: "duration_resolved_to_frame_count",
                    severity: .note,
                    title: "Duration resolved to frame count",
                    message: String(
                        format: "Duration %.2fs resolves to %d frames at %d fps (~%.2fs).",
                        duration,
                        resolvedFrames,
                        outputFPS,
                        resolvedSeconds
                    )
                )
            )
        }

        let routeWritesAudio = resolvedLTXRoute(model: model)?.writesAudio
            ?? (input.variant == .unifiedAV)
        let resolvedOutputMode: LTXVideoOutputMode? = usesWanGeometry || usesMiniMaxH3Geometry
            ? nil
            : (usesAudioConditioning || input.variant == .unifiedAV ? .audioVideo : .videoOnly)
        let h3AccelerationMode = MiniMaxH3AccelerationMode(rawValue: input.h3AccelerationMode) ?? .quality
        let resolvedH3Steps: Int? = if usesMiniMaxH3Geometry {
            input.steps ?? (input.h3Adapter != nil
                ? MiniMaxH3TurboAdapter.recommendedSchedulePointCount
                : (try? MiniMaxH3StepPolicy.recommendedPointCount(
                width: validH3RenderWidth ?? resolvedWidth,
                height: validH3RenderHeight ?? resolvedHeight,
                numFrames: resolvedFrames,
                keyframeCount: [input.image, input.endImage].compactMap { $0 }.count
                    + input.h3FrameInputs.count,
                referenceKinds: input.references.compactMap { reference in
                    MiniMaxH3ReferenceKind(rawValue: String(reference.prefix { $0 != ":" }))
                },
                accelerationMode: h3AccelerationMode
            )))
        } else {
            nil
        }
        return VideoGenerationPlanPreflightSummary(
            variant: usesMiniMaxH3Geometry
                ? "minimax-h3"
                : usesWanGeometry
                ? "wan22-ti2v"
                : (usesAudioConditioning ? "audio-to-video" : input.variant.rawValue),
            quality: resolvedQuality(model: model)?.rawValue,
            outputMode: resolvedOutputMode?.rawValue,
            inputMode: inputs.mode,
            requestedWidth: input.width,
            requestedHeight: input.height,
            resolvedWidth: resolvedWidth,
            resolvedHeight: resolvedHeight,
            requestedNumFrames: input.numFrames,
            requestedDurationSeconds: input.duration,
            resolvedSteps: resolvedH3Steps,
            h3WeightMode: usesMiniMaxH3Geometry ? input.h3WeightMode : nil,
            h3AccelerationMode: usesMiniMaxH3Geometry ? input.h3AccelerationMode : nil,
            h3RenderWidth: usesMiniMaxH3Geometry ? input.h3RenderWidth : nil,
            h3RenderHeight: usesMiniMaxH3Geometry ? input.h3RenderHeight : nil,
            h3Adapter: usesMiniMaxH3Geometry ? input.h3Adapter : nil,
            h3AdapterStrength: usesMiniMaxH3Geometry && input.h3Adapter != nil
                ? input.h3AdapterStrength
                : nil,
            h3FrameCount: usesMiniMaxH3Geometry ? input.h3FrameInputs.count : nil,
            h3WindowFrames: usesMiniMaxH3Geometry ? input.h3WindowFrames : nil,
            h3WindowOverlap: usesMiniMaxH3Geometry && input.h3WindowFrames != nil
                ? input.h3WindowOverlap
                : nil,
            h3WindowCount: slidingWindowPlan?.windows.count,
            fps: usesMiniMaxH3Geometry ? MiniMaxH3Geometry.framesPerSecond : input.fps,
            resolvedNumFrames: resolvedFrames,
            resolvedDurationSeconds: input.fps > 0
                ? Double(resolvedFrames) / Double(usesMiniMaxH3Geometry ? MiniMaxH3Geometry.framesPerSecond : input.fps)
                : nil,
            seed: input.seed ?? 42,
            writesAudio: usesMiniMaxH3Geometry || usesAudioConditioning || (!usesWanGeometry && routeWritesAudio),
            audioConditioning: usesAudioConditioning,
            preservesSourceAudio: usesAudioConditioning,
            resolvedAudioStartTime: usesAudioConditioning ? input.audioStartTime : nil
        )
    }

    private func actions(
        status: StructuredRunStatus,
        model: VideoGenerationModelPreflightSummary,
        output: VideoGenerationOutputPreflightSummary,
        inputs: VideoGenerationInputPreflightSummary
    ) -> [DeclarativeAction] {
        var actions: [DeclarativeAction] = []
        let blocked = status == .blocked
        actions.append(
            DeclarativeAction(
                id: "start-video-generation",
                label: "Start video generation",
                kind: .command,
                style: .primary,
                enabled: !blocked,
                disabledReason: blocked ? "Resolve hard blockers first." : nil,
                command: DeclarativeCommand(
                    argv: input.generationArgv,
                    cwd: input.cwd,
                    commandPath: ["video", "generate"]
                ),
                requires: ["preflight.passed"]
            )
        )

        if model.kind == "managed_model", !model.installed {
            actions.append(
                DeclarativeAction(
                    id: "pull-model",
                    label: "Pull model",
                    kind: .command,
                    style: .secondary,
                    command: DeclarativeCommand(
                        argv: ["mere.run", "model", "pull", model.requested],
                        cwd: input.cwd,
                        commandPath: ["model", "pull"]
                    )
                )
            )
        }

        actions.append(
            DeclarativeAction(
                id: "open-output-directory",
                label: "Open output directory",
                kind: .openDirectory,
                style: .link,
                enabled: output.parentExists,
                disabledReason: output.parentExists ? nil : "Output directory will be created when generation starts.",
                path: output.parentDirectory
            )
        )

        if let sourceAudio = inputs.sourceAudio {
            actions.append(
                DeclarativeAction(
                    id: "reveal-source-audio",
                    label: "Reveal source audio",
                    kind: .revealFile,
                    style: .link,
                    enabled: sourceAudio.exists && !sourceAudio.isDirectory,
                    path: sourceAudio.path
                )
            )
        }
        if let sourceImage = inputs.sourceImage {
            actions.append(
                DeclarativeAction(
                    id: "reveal-source-image",
                    label: "Reveal source image",
                    kind: .revealFile,
                    style: .link,
                    enabled: sourceImage.exists && !sourceImage.isDirectory,
                    path: sourceImage.path
                )
            )
        }
        if let endImage = inputs.endImage {
            actions.append(
                DeclarativeAction(
                    id: "reveal-end-image",
                    label: "Reveal end image",
                    kind: .revealFile,
                    style: .link,
                    enabled: endImage.exists && !endImage.isDirectory,
                    path: endImage.path
                )
            )
        }

        return actions
    }

    private func summary(
        status: StructuredRunStatus,
        diagnostics: [PreflightDiagnostic]
    ) -> String {
        switch status {
        case .ok:
            return "Video generation preflight passed."
        case .warning:
            return "Video generation preflight found \(diagnostics.count) warning(s) or note(s)."
        case .blocked:
            let blockers = diagnostics.filter { $0.severity == .blocker }.count
            return "Video generation preflight blocked by \(blockers) issue(s)."
        default:
            return "Video generation preflight status: \(status.rawValue)."
        }
    }
}
