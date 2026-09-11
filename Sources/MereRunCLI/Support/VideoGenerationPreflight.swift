import Foundation
import MereRunRelayKit
import ArgumentParser
import MereRunContract
import MereRunCore

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
    let numFrames: Int?
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
    let autoDuration: [Double]?
    let videoDecoder: String?
    let hdrColorSpace: String?
    let hdrTransfer: String?
    let highQualityHDR: Bool?
    let textEmbeddings: String?
    let vaeSpatialTileSize: Int?
    let vaeSpatialTileOverlap: Int?
    let skipHDRMP4: Bool?
    let fps: Double
    let seed: Int?
    let negativePrompt: String?
    let enhancePrompt: Bool?
    let promptEnhancerModel: String?
    let promptEnhancerModelRoot: String?
    let audio: String?
    let audioStartTime: Double
    let audioMaxDuration: Double?
    let a2vGuidanceScale: Float
    let videoCFGGuidanceScale: Float
    let audioCFGGuidanceScale: Float
    let v2aGuidanceScale: Float
    let a2vSteps: Int
    let ltxPreset: String?
    let ltxPipeline: String?
    let ltxSampler: String?
    let ltxSigmas: [Float]?
    let ltxStage2Sigmas: [Float]?
    let distilledLoRAStrengthStage1: Float?
    let distilledLoRAStrengthStage2: Float?
    let ltxSamplerEta: Float?
    let videoSTGScale: Float?
    let videoGuidanceRescale: Float?
    let videoSTGBlocks: [Int]?
    let videoGuidanceSkipStep: Int?
    let audioSTGScale: Float?
    let audioGuidanceRescale: Float?
    let audioSTGBlocks: [Int]?
    let audioGuidanceSkipStep: Int?
    let noRes2sBongMath: Bool?
    let res2sBongMaxIterations: Int?
    let gradientEstimationGamma: Float?
    let image: String?
    let imageStrength: Float
    let endImage: String?
    let endImageStrength: Float
    let imageConditionings: [String]?
    let numGeneratedKeyframes: Int?
    let generatedKeyframeIndices: [Int]?
    let loras: [String]?
    let videoConditionings: [String]?
    let conditioningAttentionStrength: Float?
    let conditioningAttentionMask: String?
    let skipStage2: Bool?
    let referenceDownscaleFactor: Int?
    let referenceTemporalScaleFactor: Int?
    let dfr: Bool?
    let temporalUpsampleRounds: Int?
    let detailingLoRAs: [String]?
    let detailingReferenceDownscaleFactor: Int?
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
        case autoDuration = "auto_duration"
        case videoDecoder = "video_decoder"
        case hdrColorSpace = "hdr"
        case hdrTransfer = "hdr_transfer"
        case highQualityHDR = "high_quality_hdr"
        case textEmbeddings = "text_embeddings"
        case vaeSpatialTileSize = "spatial_tile"
        case vaeSpatialTileOverlap = "spatial_overlap"
        case skipHDRMP4 = "skip_mp4"
        case fps
        case seed
        case negativePrompt = "negative_prompt"
        case enhancePrompt = "enhance_prompt"
        case promptEnhancerModel = "prompt_enhancer_model"
        case promptEnhancerModelRoot = "prompt_enhancer_model_root"
        case audio
        case audioStartTime = "audio_start_time"
        case audioMaxDuration = "audio_max_duration"
        case a2vGuidanceScale = "a2v_guidance_scale"
        case videoCFGGuidanceScale = "video_cfg_guidance_scale"
        case audioCFGGuidanceScale = "audio_cfg_guidance_scale"
        case v2aGuidanceScale = "v2a_guidance_scale"
        case a2vSteps = "a2v_steps"
        case ltxPreset = "ltx_preset"
        case ltxPipeline = "ltx_pipeline"
        case ltxSampler = "ltx_sampler"
        case ltxSigmas = "ltx_sigmas"
        case ltxStage2Sigmas = "ltx_stage_2_sigmas"
        case distilledLoRAStrengthStage1 = "distilled_lora_strength_stage_1"
        case distilledLoRAStrengthStage2 = "distilled_lora_strength_stage_2"
        case ltxSamplerEta = "ltx_sampler_eta"
        case videoSTGScale = "video_stg_scale"
        case videoGuidanceRescale = "video_guidance_rescale"
        case videoSTGBlocks = "video_stg_blocks"
        case videoGuidanceSkipStep = "video_guidance_skip_step"
        case audioSTGScale = "audio_stg_scale"
        case audioGuidanceRescale = "audio_guidance_rescale"
        case audioSTGBlocks = "audio_stg_blocks"
        case audioGuidanceSkipStep = "audio_guidance_skip_step"
        case noRes2sBongMath = "no_res2s_bong_math"
        case res2sBongMaxIterations = "res2s_bong_max_iterations"
        case gradientEstimationGamma = "gradient_estimation_gamma"
        case image
        case imageStrength = "image_strength"
        case endImage = "end_image"
        case endImageStrength = "end_image_strength"
        case imageConditionings = "image_conditionings"
        case numGeneratedKeyframes = "num_generated_keyframes"
        case generatedKeyframeIndices = "generated_keyframe_indices"
        case loras
        case videoConditionings = "video_conditionings"
        case conditioningAttentionStrength = "conditioning_attention_strength"
        case conditioningAttentionMask = "conditioning_attention_mask"
        case skipStage2 = "skip_stage_2"
        case referenceDownscaleFactor = "reference_downscale_factor"
        case referenceTemporalScaleFactor = "reference_temporal_scale_factor"
        case dfr
        case temporalUpsampleRounds = "temporal_upsample_rounds"
        case detailingLoRAs = "detailing_loras"
        case detailingReferenceDownscaleFactor = "detailing_reference_downscale_factor"
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
    let ltxLoRAs: [VideoGenerationPathPreflightSummary]?
    let detailingLoRAs: [VideoGenerationPathPreflightSummary]?
    let imageConditionings: [VideoGenerationPathPreflightSummary]?
    let videoConditionings: [VideoGenerationPathPreflightSummary]?
    let textEmbeddings: VideoGenerationPathPreflightSummary?
    let conditioningAttentionMask: VideoGenerationPathPreflightSummary?
    let missingCount: Int

    enum CodingKeys: String, CodingKey {
        case mode
        case sourceAudio = "source_audio"
        case sourceImage = "source_image"
        case endImage = "end_image"
        case h3Frames = "h3_frames"
        case references
        case adapter
        case ltxLoRAs = "ltx_loras"
        case detailingLoRAs = "detailing_loras"
        case imageConditionings = "image_conditionings"
        case videoConditionings = "video_conditionings"
        case textEmbeddings = "text_embeddings"
        case conditioningAttentionMask = "conditioning_attention_mask"
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
    let requestedNumFrames: Int?
    let requestedDurationSeconds: Double?
    let autoDuration: [Double]?
    let videoDecoder: String?
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
    let fps: Double
    let resolvedNumFrames: Int?
    let resolvedDurationSeconds: Double?
    let seed: Int
    let writesAudio: Bool
    let audioConditioning: Bool
    let preservesSourceAudio: Bool
    let resolvedAudioStartTime: Double?
    let resolvedAudioMaxDuration: Double?

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
        case autoDuration = "auto_duration"
        case videoDecoder = "video_decoder"
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
        case resolvedAudioMaxDuration = "resolved_audio_max_duration"
    }
}

typealias VideoGenerationPreflightEnvelope = StructuredRunEnvelope<
    VideoGenerationPreflightRequest,
    VideoGenerationPreflightResult
>

struct VideoGenerationPreflightAnalyzer {
    let input: VideoGenerationOptions
    let generationArgv: [String]
    let cwd: String
    let fileManager: FileManager
    let adaptersRoot: URL
    let now: () -> Date
    private let modelProfile: VideoGenerationModelProfile

    init(
        input: VideoGenerationOptions,
        generationArgv: [String],
        cwd: String,
        fileManager: FileManager = .default,
        adaptersRoot: URL = MereRunModelPaths.adaptersDir,
        now: @escaping () -> Date = Date.init
    ) {
        self.input = input
        self.generationArgv = generationArgv
        self.cwd = cwd
        self.fileManager = fileManager
        self.adaptersRoot = adaptersRoot
        self.now = now
        self.modelProfile = input.observedProfile(fileManager: fileManager)
    }

    private var usesWanGeometry: Bool { modelProfile == .wan }
    private var usesMiniMaxH3Geometry: Bool { modelProfile.isH3 }
    private var usesMiniMaxH3Ref2VA: Bool { modelProfile == .h3Ref2VA }
    private var usesAudioConditioning: Bool { input.hasSourceAudio }

    private func effectiveAutoDuration(
        model: VideoGenerationModelPreflightSummary
    ) -> [Double]? {
        guard let range = try? VideoGenerationPlan(options: input, profile: modelProfile).autoDuration else { return nil }
        return [range.minimumSeconds, range.maximumSeconds]
    }

    private var h3AdapterInferenceRecipe: MiniMaxH3TurboAdapter.InferenceRecipe? {
        input.h3AdapterInferenceRecipe
    }
    private var usesEmbeddedFastH3Adapter: Bool { input.usesEmbeddedFastH3Adapter }

    func envelope() -> VideoGenerationPreflightEnvelope {
        var diagnostics: [PreflightDiagnostic] = []
        validateStaticOptions(diagnostics: &diagnostics)
        let model = modelSummary(diagnostics: &diagnostics)
        let output = outputSummary(diagnostics: &diagnostics)
        let inputs = inputSummary(model: model, diagnostics: &diagnostics)
        let plan = planSummary(model: model, inputs: inputs, diagnostics: &diagnostics)
        let status = StructuredRunOutput.status(for: diagnostics)

        return VideoGenerationPreflightEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["video", "generate"],
            mode: .preflight,
            status: status,
            createdAt: now(),
            cwd: cwd,
            summary: summary(status: status, diagnostics: diagnostics),
            request: request(model: model),
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

    private func request(model: VideoGenerationModelPreflightSummary) -> VideoGenerationPreflightRequest {
        VideoGenerationPreflightRequest(
            prompt: input.prompt,
            output: input.outputURL.path,
            model: input.resolvedRequestedModel,
            variant: input.variant.rawValue,
            quality: input.quality?.rawValue,
            outputMode: input.outputMode?.rawValue,
            modelRoot: input.modelRoot,
            width: input.resolvedOutputWidth,
            height: input.resolvedOutputHeight,
            numFrames: input.numFramesSpecified ? input.requestedFrameCount : nil,
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
            autoDuration: effectiveAutoDuration(model: model),
            videoDecoder: input.videoDecoder?.rawValue,
            hdrColorSpace: input.hdrColorSpace?.rawValue,
            hdrTransfer: input.hdrTransfer?.rawValue,
            highQualityHDR: input.highQualityHDR ? true : nil,
            textEmbeddings: input.textEmbeddings,
            vaeSpatialTileSize: input.vaeSpatialTileSize,
            vaeSpatialTileOverlap: input.vaeSpatialTileSize == nil
                ? nil
                : input.vaeSpatialTileOverlap,
            skipHDRMP4: input.skipHDRMP4 ? true : nil,
            fps: input.fps,
            seed: input.seed,
            negativePrompt: input.negativePrompt,
            enhancePrompt: input.enhancePrompt ? true : nil,
            promptEnhancerModel: input.promptEnhancerModel,
            promptEnhancerModelRoot: input.promptEnhancerModelRoot,
            audio: input.audio,
            audioStartTime: input.audioStartTime,
            audioMaxDuration: input.audioMaxDuration,
            a2vGuidanceScale: input.a2vGuidanceScale,
            videoCFGGuidanceScale: input.videoCFGGuidanceScale,
            audioCFGGuidanceScale: input.audioCFGGuidanceScale,
            v2aGuidanceScale: input.v2aGuidanceScale,
            a2vSteps: input.a2vSteps,
            ltxPreset: input.ltxPreset == .standard ? nil : input.ltxPreset.rawValue,
            ltxPipeline: input.ltxPipeline == .twoStage ? nil : input.ltxPipeline.rawValue,
            ltxSampler: input.ltxSampler?.rawValue,
            ltxSigmas: input.ltxSigmas.isEmpty ? nil : input.ltxSigmas,
            ltxStage2Sigmas: input.ltxStage2Sigmas.isEmpty ? nil : input.ltxStage2Sigmas,
            distilledLoRAStrengthStage1: input.distilledLoRAStrengthStage1,
            distilledLoRAStrengthStage2: input.distilledLoRAStrengthStage2,
            ltxSamplerEta: input.ltxSamplerEta == 0.5 ? nil : input.ltxSamplerEta,
            videoSTGScale: input.videoSTGScale == 1 ? nil : input.videoSTGScale,
            videoGuidanceRescale: input.videoGuidanceRescale == 0.7 ? nil : input.videoGuidanceRescale,
            videoSTGBlocks: input.videoSTGBlocks.isEmpty ? nil : input.videoSTGBlocks,
            videoGuidanceSkipStep: input.videoGuidanceSkipStep == 0 ? nil : input.videoGuidanceSkipStep,
            audioSTGScale: input.audioSTGScale == 1 ? nil : input.audioSTGScale,
            audioGuidanceRescale: input.audioGuidanceRescale == 0.7 ? nil : input.audioGuidanceRescale,
            audioSTGBlocks: input.audioSTGBlocks.isEmpty ? nil : input.audioSTGBlocks,
            audioGuidanceSkipStep: input.audioGuidanceSkipStep == 0 ? nil : input.audioGuidanceSkipStep,
            noRes2sBongMath: input.noRes2sBongMath ? true : nil,
            res2sBongMaxIterations: input.res2sBongMaxIterations == 100
                ? nil
                : input.res2sBongMaxIterations,
            gradientEstimationGamma: input.gradientEstimationGamma == 2
                ? nil
                : input.gradientEstimationGamma,
            image: input.image,
            imageStrength: input.imageStrength,
            endImage: input.endImage,
            endImageStrength: input.endImageStrength,
            imageConditionings: input.imageConditionings.isEmpty ? nil : input.imageConditionings,
            numGeneratedKeyframes: input.numGeneratedKeyframes == 0
                ? nil
                : input.numGeneratedKeyframes,
            generatedKeyframeIndices: input.generatedKeyframeIndices.isEmpty
                ? nil
                : input.generatedKeyframeIndices,
            loras: input.loras.isEmpty ? nil : input.loras,
            videoConditionings: input.videoConditionings.isEmpty ? nil : input.videoConditionings,
            conditioningAttentionStrength: input.videoConditionings.isEmpty
                ? nil
                : input.conditioningAttentionStrength,
            conditioningAttentionMask: input.conditioningAttentionMask,
            skipStage2: input.skipStage2 ? true : nil,
            referenceDownscaleFactor: input.videoConditionings.isEmpty
                ? nil
                : input.referenceDownscaleFactor,
            referenceTemporalScaleFactor: input.videoConditionings.isEmpty
                ? nil
                : input.referenceTemporalScaleFactor,
            dfr: input.dfr ? true : nil,
            temporalUpsampleRounds: input.dfr ? input.temporalUpsampleRounds : nil,
            detailingLoRAs: input.detailingLoRAs.isEmpty ? nil : input.detailingLoRAs,
            detailingReferenceDownscaleFactor: input.detailingReferenceDownscaleFactor,
            references: input.references.isEmpty ? nil : input.references,
            timings: input.timings,
            timingsOutput: input.timingsOutput
        )
    }

    private func resolvedLTXRoute(model: VideoGenerationModelPreflightSummary) -> LTXVideoGenerationRoute? {
        modelProfile.ltxRoute(outputMode: input.effectiveOutputMode)
    }

    private func validateStaticOptions(diagnostics: inout [PreflightDiagnostic]) {
        diagnostics += input.validationIssues(profile: modelProfile).map { issue in
            PreflightDiagnostic(
                id: issue.id,
                severity: issue.severity == .blocker ? .blocker : (issue.severity == .warning ? .warning : .note),
                title: issue.title,
                message: issue.message
            )
        }
    }

    private func resolvedQuality(model: VideoGenerationModelPreflightSummary) -> LTXVideoQuality? {
        modelProfile.quality
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

        let trimmedModel = input.resolvedRequestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
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
        let profile = VideoGenerationModelProfile.observe(root: url, fileManager: fileManager)
        return profile == .unknown ? nil : profile.rawValue
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
        model: VideoGenerationModelPreflightSummary,
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
            let path = ManagedAdapterCatalog.spec(for: reference)?
                .installedFileURL(adaptersRoot: adaptersRoot).path ?? reference
            return pathSummary(requested: reference, resolvedPath: path)
        }
        func loraReference(_ raw: String) -> String {
            guard let separator = raw.lastIndex(of: "=") else { return raw }
            let suffix = raw[raw.index(after: separator)...]
            return Float(suffix).map { _ in String(raw[..<separator]) } ?? raw
        }
        func adapterPathSummary(_ raw: String) -> VideoGenerationPathPreflightSummary {
            let reference = loraReference(raw)
            let resolved = ManagedAdapterCatalog.spec(for: reference)?
                .installedFileURL(adaptersRoot: adaptersRoot).path
                ?? reference
            return pathSummary(requested: reference, resolvedPath: resolved)
        }
        let ltxLoRAs = input.loras.map(adapterPathSummary)
        let detailingLoRAs = input.detailingLoRAs.map(adapterPathSummary)
        let imageConditionings = input.imageConditionings.map { raw in
            let pieces = raw.split(separator: ":", maxSplits: 3, omittingEmptySubsequences: false)
            let path = pieces.count >= 2 ? String(pieces[1]) : raw
            return pathSummary(requested: path)
        }
        let videoConditionings = input.videoConditionings.map { raw in
            pathSummary(requested: loraReference(raw))
        }
        let textEmbeddings = input.textEmbeddings.map { pathSummary(requested: $0) }
        let conditioningAttentionMask = input.conditioningAttentionMask.map {
            pathSummary(requested: $0)
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
        let expectedLTXBaseModelID: String? = switch model.layout {
        case "ltx25_full": ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
        case "ltx25_distilled": ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue
        default: nil
        }
        for (kind, summaries) in [
            ("ltx_lora", ltxLoRAs),
            ("detailing_lora", detailingLoRAs),
        ] {
            for (index, summary) in summaries.enumerated() {
                if let expectedLTXBaseModelID,
                   let spec = ManagedAdapterCatalog.spec(for: summary.requested),
                   !spec.supports(baseModelID: expectedLTXBaseModelID) {
                    diagnostics.append(PreflightDiagnostic(
                        id: "\(kind)_\(index)_base_model_mismatch",
                        severity: .blocker,
                        title: "LTX adapter base model mismatch",
                        message: "Adapter \(spec.id) requires \(spec.baseModelID), not \(expectedLTXBaseModelID)."
                    ))
                }
                if !summary.exists {
                    let pullHint = ManagedAdapterCatalog.spec(for: summary.requested).map {
                        " Run `mere.run adapter pull \($0.id)`."
                    } ?? ""
                    diagnostics.append(PreflightDiagnostic(
                        id: "\(kind)_\(index)_missing",
                        severity: .blocker,
                        title: "LTX adapter missing",
                        message: "Adapter not found: \(summary.path).\(pullHint)",
                        locations: [.init(kind: "file", path: summary.path)]
                    ))
                } else if summary.isDirectory {
                    diagnostics.append(PreflightDiagnostic(
                        id: "\(kind)_\(index)_is_directory",
                        severity: .blocker,
                        title: "LTX adapter path is a directory",
                        message: "Adapter path must be a safetensors file: \(summary.path)",
                        locations: [.init(kind: "directory", path: summary.path)]
                    ))
                }
            }
        }
        for (kind, summaries) in [
            ("image_conditioning", imageConditionings),
            ("video_conditioning", videoConditionings),
        ] {
            for (index, summary) in summaries.enumerated() where !summary.exists {
                diagnostics.append(PreflightDiagnostic(
                    id: "\(kind)_\(index)_missing",
                    severity: .blocker,
                    title: "LTX conditioning input missing",
                    message: "Conditioning input not found: \(summary.path)",
                    locations: [.init(kind: "file", path: summary.path)]
                ))
            }
        }
        for (kind, summary) in [
            ("text_embeddings", textEmbeddings),
            ("conditioning_attention_mask", conditioningAttentionMask),
        ] {
            guard let summary, !summary.exists else { continue }
            diagnostics.append(PreflightDiagnostic(
                id: "\(kind)_missing",
                severity: .blocker,
                title: "LTX auxiliary input missing",
                message: "Auxiliary input not found: \(summary.path)",
                locations: [.init(kind: "file", path: summary.path)]
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
            + ltxLoRAs
            + detailingLoRAs
            + imageConditionings
            + videoConditionings
            + [textEmbeddings, conditioningAttentionMask].compactMap { $0 }
        return VideoGenerationInputPreflightSummary(
            mode: mode,
            sourceAudio: sourceAudio,
            sourceImage: sourceImage,
            endImage: endImage,
            h3Frames: h3Frames.isEmpty ? nil : h3Frames,
            references: references.isEmpty ? nil : references,
            adapter: adapter,
            ltxLoRAs: ltxLoRAs.isEmpty ? nil : ltxLoRAs,
            detailingLoRAs: detailingLoRAs.isEmpty ? nil : detailingLoRAs,
            imageConditionings: imageConditionings.isEmpty ? nil : imageConditionings,
            videoConditionings: videoConditionings.isEmpty ? nil : videoConditionings,
            textEmbeddings: textEmbeddings,
            conditioningAttentionMask: conditioningAttentionMask,
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
        let plan: VideoGenerationPlan?
        do {
            let arguments = VideoGenerationArgumentParser(
                options: input, fileManager: fileManager, adaptersRoot: adaptersRoot, requireFiles: false
            )
            let baseModelID = modelProfile == .ltx25Full ? ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
                : modelProfile.isLTX25 ? ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue
                : input.resolvedRequestedModel
            let loras = try arguments.parseLTXLoRAConfigurations(input.loras, optionName: "--lora", baseModelID: baseModelID)
            _ = try arguments.parseLTXImageConditionings()
            // Missing adapters already have path diagnostics. Their metadata is unknown.
            let preparation: VideoGenerationLTXPreparation?
            if loras.allSatisfy({ fileManager.fileExists(atPath: $0.url.path) }) {
                preparation = try VideoGenerationLTXPreparation(options: input, profile: modelProfile, loras: loras)
            } else {
                preparation = nil
            }
            _ = try arguments.parseLTXReferenceVideoConditionings(
                downscaleFactor: preparation?.referenceDownscaleFactor ?? 1,
                temporalScaleFactor: preparation?.referenceTemporalScaleFactor ?? 1
            )
            plan = try VideoGenerationPlan(options: input, profile: modelProfile, preparation: preparation)
        } catch {
            plan = nil
            let issue = error as? VideoGenerationIssue
            if !diagnostics.contains(where: { $0.id == issue?.id }) {
                diagnostics.append(PreflightDiagnostic(
                    id: issue?.id ?? "video_plan_invalid", severity: .blocker,
                    title: issue?.title ?? "Video plan is invalid", message: error.localizedDescription
                ))
            }
        }
        let resolvedWidth = plan?.width ?? input.resolvedOutputWidth
        let resolvedHeight = plan?.height ?? input.resolvedOutputHeight
        let resolvedFrames = plan?.numFrames ?? 0
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
        let slidingWindowPlan = plan?.h3SlidingWindowOptions.map(MiniMaxH3SlidingWindowPlan.init(options:))

        if plan != nil, input.resolvedOutputWidth >= spatialMultiple,
           input.resolvedOutputHeight >= spatialMultiple,
           resolvedWidth != input.resolvedOutputWidth || resolvedHeight != input.resolvedOutputHeight {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "dimensions_will_be_adjusted",
                    severity: .note,
                    title: "Dimensions will be adjusted",
                    message: "Video dimensions will be snapped from \(input.resolvedOutputWidth)x\(input.resolvedOutputHeight) to \(resolvedWidth)x\(resolvedHeight)."
                )
            )
        }
        if plan != nil, input.requestedFrameCount >= minimumFrames, input.duration == nil, resolvedFrames != input.requestedFrameCount {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "num_frames_will_be_adjusted",
                    severity: .note,
                    title: "Frame count will be adjusted",
                    message: usesMiniMaxH3Geometry
                        ? "Frame count will be snapped from \(input.requestedFrameCount) to \(resolvedFrames) to satisfy 17*n+5."
                        : "Frame count will be snapped from \(input.requestedFrameCount) to \(resolvedFrames) to satisfy \(temporalMultiple)n+1."
                )
            )
        }
        if plan != nil, let duration = input.duration, input.fps > 0, duration > 0 {
            let outputFPS = usesMiniMaxH3Geometry
                ? Double(MiniMaxH3Geometry.framesPerSecond)
                : input.fps
            let resolvedSeconds = Double(resolvedFrames) / outputFPS
            diagnostics.append(
                PreflightDiagnostic(
                    id: "duration_resolved_to_frame_count",
                    severity: .note,
                    title: "Duration resolved to frame count",
                    message: String(
                        format: "Duration %.2fs resolves to %d frames at %g fps (~%.2fs).",
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
        let resolvedVideoDecoder = modelProfile.isLTX25 ? plan?.videoDecoder : nil
        var resolvedH3Steps: Int?
        if usesMiniMaxH3Geometry, let plan {
            do {
                let frames = try parseMiniMaxH3FrameArguments(input.h3FrameInputs, requireFiles: false)
                let references = try parseMiniMaxH3ReferenceArguments(input.references, requireFiles: false)
                let preparation = try VideoGenerationH3Preparation(
                    options: input, profile: modelProfile,
                    modelRoot: model.path.map { URL(fileURLWithPath: $0) },
                    adaptersRoot: adaptersRoot, fileManager: fileManager, requireInstalled: false
                )
                resolvedH3Steps = try plan.miniMaxH3Options(
                    adapterURL: preparation.adapterURL,
                    firstFrameURL: input.image.map { URL(fileURLWithPath: $0) },
                    lastFrameURL: input.endImage.map { URL(fileURLWithPath: $0) },
                    frames: frames, references: references
                ).steps
            } catch {
                let issue = error as? VideoGenerationIssue
                diagnostics.append(PreflightDiagnostic(
                    id: issue?.id ?? "h3_generation_options_invalid", severity: .blocker,
                    title: issue?.title ?? "MiniMax-H3 request is invalid", message: error.localizedDescription
                ))
            }
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
            requestedWidth: input.resolvedOutputWidth,
            requestedHeight: input.resolvedOutputHeight,
            resolvedWidth: resolvedWidth,
            resolvedHeight: resolvedHeight,
            requestedNumFrames: input.numFramesSpecified ? input.requestedFrameCount : nil,
            requestedDurationSeconds: input.duration,
            autoDuration: effectiveAutoDuration(model: model),
            videoDecoder: resolvedVideoDecoder?.rawValue,
            resolvedSteps: resolvedH3Steps,
            h3WeightMode: usesMiniMaxH3Geometry ? input.h3WeightMode : nil,
            h3AccelerationMode: usesMiniMaxH3Geometry ? input.h3AccelerationMode : nil,
            h3RenderWidth: usesMiniMaxH3Geometry ? input.h3RenderWidth : nil,
            h3RenderHeight: usesMiniMaxH3Geometry ? input.h3RenderHeight : nil,
            h3Adapter: usesMiniMaxH3Geometry
                ? (usesEmbeddedFastH3Adapter
                    ? MiniMaxH3TurboAdapter.fastH3VSADataFreeFilename
                    : input.h3Adapter)
                : nil,
            h3AdapterStrength: usesMiniMaxH3Geometry && h3AdapterInferenceRecipe != nil
                ? input.h3AdapterStrength
                : nil,
            h3FrameCount: usesMiniMaxH3Geometry ? input.h3FrameInputs.count : nil,
            h3WindowFrames: usesMiniMaxH3Geometry ? input.h3WindowFrames : nil,
            h3WindowOverlap: usesMiniMaxH3Geometry && input.h3WindowFrames != nil
                ? input.h3WindowOverlap
                : nil,
            h3WindowCount: slidingWindowPlan?.windows.count,
            fps: usesMiniMaxH3Geometry
                ? Double(MiniMaxH3Geometry.framesPerSecond)
                : input.fps,
            resolvedNumFrames: effectiveAutoDuration(model: model) == nil
                ? plan?.numFrames
                : nil,
            resolvedDurationSeconds: plan != nil && effectiveAutoDuration(model: model) == nil && input.fps > 0
                ? Double(resolvedFrames) / (usesMiniMaxH3Geometry
                    ? Double(MiniMaxH3Geometry.framesPerSecond)
                    : input.fps)
                : nil,
            seed: plan?.seed ?? input.seed ?? (modelProfile.isLTX25 ? 10 : 42),
            writesAudio: usesMiniMaxH3Geometry || usesAudioConditioning || (!usesWanGeometry && routeWritesAudio),
            audioConditioning: usesAudioConditioning,
            preservesSourceAudio: usesAudioConditioning,
            resolvedAudioStartTime: usesAudioConditioning ? input.audioStartTime : nil,
            resolvedAudioMaxDuration: usesAudioConditioning
                ? input.audioMaxDuration
                : nil
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
                    argv: generationArgv,
                    cwd: cwd,
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
                        cwd: cwd,
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
