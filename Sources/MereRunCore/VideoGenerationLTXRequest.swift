import Foundation

/// A prepared LTX request. Tensor loading, admission, progress, and export stay
/// with the caller; every native options object comes from these same settings.
public struct VideoGenerationLTXRequest: Sendable {
    public let settings: VideoGenerationOptions
    public let plan: VideoGenerationPlan
    public let preparation: VideoGenerationLTXPreparation
    public let prompt: String
    public let sourceImageURL: URL?
    public let endImageURL: URL?
    public let imageConditionings: [LTXVideoConditioningInput]
    public let referenceVideos: [LTXReferenceVideoConditioningInput]

    public init(
        plan: VideoGenerationPlan,
        preparation: VideoGenerationLTXPreparation,
        prompt: String,
        sourceImageURL: URL? = nil,
        endImageURL: URL? = nil,
        imageConditionings: [LTXVideoConditioningInput] = [],
        referenceVideos: [LTXReferenceVideoConditioningInput] = []
    ) {
        self.settings = plan.options
        self.plan = plan
        self.preparation = preparation
        self.prompt = prompt
        self.sourceImageURL = sourceImageURL
        self.endImageURL = endImageURL
        self.imageConditionings = imageConditionings
        self.referenceVideos = referenceVideos
    }

    public func unifiedOptions(numFrames: Int? = nil) -> LTXUnifiedAVGenerationOptions {
        LTXUnifiedAVGenerationOptions(
            prompt: prompt,
            negativePrompt: settings.negativePrompt ?? LTXUnifiedAVGenerationOptions.defaultNegativePrompt,
            width: plan.width,
            height: plan.height,
            numFrames: numFrames ?? plan.numFrames,
            fps: plan.fps,
            seed: plan.seed,
            inferenceSteps: plan.ltxRecipe.inferenceSteps,
            videoGuidance: plan.ltxRecipe.videoGuidance,
            audioGuidance: plan.ltxRecipe.audioGuidance,
            sourceImageURL: sourceImageURL,
            imageStrength: settings.imageStrength,
            endImageURL: endImageURL,
            endImageStrength: settings.endImageStrength,
            imageConditionings: imageConditionings,
            generatedKeyframeCount: settings.numGeneratedKeyframes,
            generatedKeyframeIndices: settings.generatedKeyframeIndices,
            referenceVideos: referenceVideos,
            loras: preparation.loras,
            dfr: settings.dfr ? LTX25DFROptions(
                temporalUpsampleRounds: settings.temporalUpsampleRounds,
                detailingLoRAs: preparation.detailingLoRAs,
                detailingReferenceDownscaleFactor: settings.detailingReferenceDownscaleFactor
            ) : nil,
            sigmas: settings.ltxSigmas.isEmpty ? nil : settings.ltxSigmas,
            stage2Sigmas: settings.ltxStage2Sigmas.isEmpty ? nil : settings.ltxStage2Sigmas,
            sampler: plan.ltxRecipe.sampler,
            pipeline: settings.ltxPipeline,
            distilledLoRAStrengthStage1: plan.ltxRecipe.distilledLoRAStrengthStage1,
            distilledLoRAStrengthStage2: plan.ltxRecipe.distilledLoRAStrengthStage2,
            hdrColorSpace: preparation.hdrColorSpace,
            hdrTransfer: preparation.hdrTransfer,
            hdrICLoRA: preparation.hdrICLoRA,
            vaeSpatialTileSize: settings.vaeSpatialTileSize,
            vaeSpatialTileOverlap: settings.vaeSpatialTileOverlap,
            skipStage2: settings.skipStage2,
            precomputedTextEmbeddingsURL: settings.textEmbeddings.map { URL(fileURLWithPath: $0).standardizedFileURL },
            transformerExecution: settings.ltxTransformerExecution,
            guidanceProjectionCache: settings.ltxGuidanceProjectionCache,
            teaCache: settings.ltxTeaCache || settings.ltxTeaCacheCalibrationOutput != nil
                ? LTXTeaCacheConfiguration(
                    threshold: settings.ltxTeaCacheThreshold,
                    calibrationOutputURL: settings.ltxTeaCacheCalibrationOutput.map { URL(fileURLWithPath: $0).standardizedFileURL }
                ) : nil
        )
    }

    public func distilledOptions() -> LTXDistilledLatentGenerationOptions {
        LTXDistilledLatentGenerationOptions(
            prompt: prompt, width: plan.width, height: plan.height,
            numFrames: plan.numFrames, fps: plan.fps, seed: plan.seed,
            sourceImageURL: sourceImageURL, imageStrength: settings.imageStrength,
            endImageURL: endImageURL, endImageStrength: settings.endImageStrength
        )
    }

    public func audioToVideoOptions(audioURL: URL) -> LTXAudioToVideoGenerationOptions {
        LTXAudioToVideoGenerationOptions(
            prompt: prompt,
            negativePrompt: settings.negativePrompt ?? LTXAudioToVideoGenerationOptions.defaultNegativePrompt,
            audioURL: audioURL,
            audioStartTime: settings.audioStartTime,
            audioMaxDuration: settings.audioMaxDuration,
            width: plan.width,
            height: plan.height,
            numFrames: plan.numFrames,
            fps: plan.fps,
            seed: plan.seed,
            inferenceSteps: settings.a2vSteps,
            guidance: LTXAudioToVideoGuidance(
                classifierFreeScale: settings.videoCFGGuidanceScale,
                audioToVideoScale: settings.a2vGuidanceScale
            ),
            sourceImageURL: sourceImageURL,
            imageStrength: settings.imageStrength,
            endImageURL: endImageURL,
            endImageStrength: settings.endImageStrength,
            imageConditionings: imageConditionings,
            generatedKeyframeCount: settings.numGeneratedKeyframes,
            generatedKeyframeIndices: settings.generatedKeyframeIndices,
            hdrColorSpace: preparation.hdrColorSpace,
            hdrTransfer: preparation.hdrTransfer,
            transformerExecution: settings.ltxTransformerExecution
        )
    }
}
