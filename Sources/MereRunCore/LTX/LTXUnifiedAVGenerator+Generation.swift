import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

extension LTXUnifiedAVGenerator {
    public func generate(
        options: LTXUnifiedAVGenerationOptions
    ) async throws -> LTXUnifiedAVGenerationResult {
        guard !loadedForVideoOnlyOutput else {
            throw LTXUnifiedAVGeneratorError.audioDecoderNotLoaded
        }
        let output = try await generate(options: options, decodeAudio: true)
        guard let audioWaveform = output.audioWaveform,
              let audioSampleRate = output.audioSampleRate else {
            throw LTXUnifiedAVGeneratorError.audioDecoderNotLoaded
        }
        return LTXUnifiedAVGenerationResult(
            frames: output.frames,
            videoLatents: output.videoLatents,
            audioLatents: output.audioLatents,
            audioWaveform: audioWaveform,
            audioSampleRate: audioSampleRate,
            hdrOutput: output.hdrOutput,
            generatedKeyframeLatents: output.generatedKeyframeLatents,
            generatedKeyframeIndices: output.generatedKeyframeIndices,
            playbackFPS: output.playbackFPS,
            timings: output.timings
        )
    }

    public func generateVideoOnly(
        options: LTXUnifiedAVGenerationOptions
    ) async throws -> LTXUnifiedVideoGenerationResult {
        let output = try await generate(options: options, decodeAudio: false)
        return LTXUnifiedVideoGenerationResult(
            frames: output.frames,
            hdrOutput: output.hdrOutput,
            videoLatents: output.videoLatents,
            generatedKeyframeLatents: output.generatedKeyframeLatents,
            generatedKeyframeIndices: output.generatedKeyframeIndices,
            playbackFPS: output.playbackFPS,
            timings: output.timings
        )
    }

    func generate(
        options: LTXUnifiedAVGenerationOptions,
        decodeAudio: Bool
    ) async throws -> LTXUnifiedGenerationOutput {
        let totalStart = ltxMonotonicSeconds()
        var textEncodingSeconds = 0.0
        var textEncoderReloadSeconds = 0.0
        var promptCacheHits = 0
        var promptCacheMisses = 0
        let guidanceProjectionCacheMetrics = LTXGuidanceProjectionCacheMetrics()
        let teaCacheController = options.teaCache.map {
            LTXTeaCacheController(configuration: $0, sampler: options.sampler.mode)
        }
        var preparationSeconds = 0.0
        var stage1DenoiseSeconds = 0.0
        var loraFusionSeconds = 0.0
        var upsampleSeconds = 0.0
        var stage2DenoiseSeconds = 0.0
        var videoDecodeSeconds = 0.0
        var audioDecodeSeconds = 0.0
        let prompt = options.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw LTXUnifiedAVGeneratorError.emptyPrompt
        }
        let usesDFR = options.dfr != nil
        let stage1DistilledLoRAStrength = usesDFR && options.distilledLoRAStrengthStage1 == 0
            ? 1
            : options.distilledLoRAStrengthStage1
        let usesHDRICLoRA = options.hdrICLoRA != nil
        let resolutionMultiple = options.retake == nil ? 64 : 32
        let generationWidth = usesHDRICLoRA
            ? max(64, ((options.width + 63) / 64) * 64)
            : options.width
        let generationHeight = usesHDRICLoRA
            ? max(64, ((options.height + 63) / 64) * 64)
            : options.height
        guard options.width > 0,
              options.height > 0,
              usesHDRICLoRA || (
                  options.width.isMultiple(of: resolutionMultiple)
                      && options.height.isMultiple(of: resolutionMultiple)
              ) else {
            throw LTXUnifiedAVGeneratorError.invalidResolution(width: options.width, height: options.height)
        }
        guard options.numFrames >= 9, options.numFrames % 8 == 1 else {
            throw LTXUnifiedAVGeneratorError.invalidFrameCount(options.numFrames)
        }
        guard options.vaeSpatialTileOverlap >= 0,
              options.vaeSpatialTileSize.map({ $0 > options.vaeSpatialTileOverlap }) ?? true else {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "The VAE spatial tile must be positive and larger than its overlap."
            )
        }
        if usesDFR, options.retake != nil {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "Retake and DFR cannot run in the same generation."
            )
        }
        if usesDFR,
           options.generatedKeyframeCount > 0 || !options.generatedKeyframeIndices.isEmpty {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "DFR derives its generated-keyframe slots from the official segment grid."
            )
        }
        if usesDFR, !options.referenceVideos.isEmpty {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "DFR detailing references are supplied through detailing LoRAs, not IC-LoRA reference videos."
            )
        }
        if options.dubIt != nil, options.retake != nil || usesDFR {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "Dub-It cannot be combined with Retake or DFR."
            )
        }
        if usesHDRICLoRA {
            guard options.hdrColorSpace != nil, loadedForLTX25, !decodeAudio else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "HDR IC-LoRA is an LTX-2.5 video-only HDR pipeline."
                )
            }
            guard !usesDFR,
                  options.retake == nil,
                  options.dubIt == nil,
                  options.sourceImageURL == nil,
                  options.endImageURL == nil,
                  options.imageConditionings.isEmpty,
                  options.generatedKeyframeCount == 0,
                  options.generatedKeyframeIndices.isEmpty,
                  !options.referenceVideos.isEmpty,
                  !options.loras.isEmpty else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "HDR IC-LoRA accepts one or more reference videos and HDR LoRA adapters only."
                )
            }
            for phase in options.hdrICLoRA!.stage2Phases {
                _ = try validatedLTXSigmaSchedule(phase.sigmas)
            }
        }
        if let embeddingsURL = options.precomputedTextEmbeddingsURL {
            guard usesHDRICLoRA else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "Precomputed text embeddings are currently accepted by the dedicated HDR IC-LoRA pipeline."
                )
            }
            guard FileManager.default.fileExists(atPath: embeddingsURL.path) else {
                throw LTXUnifiedAVGeneratorError.referenceVideoNotFound(embeddingsURL)
            }
        }
        if options.skipStage2 {
            guard !usesHDRICLoRA,
                  !usesDFR,
                  options.retake == nil,
                  options.dubIt == nil,
                  !options.referenceVideos.isEmpty,
                  !loadedForFullTwoStage else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "Skipping stage two is available for distilled IC-LoRA reference generation."
                )
            }
        }
        if usesDFR, !loadedForDFR {
            if let loadedRoot {
                throw LTXUnifiedAVGeneratorError.dfrRequiresLTX25Full(loadedRoot)
            }
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }
        let dfrCanvas = try options.dfr.map { _ in
            try LTX25DFRLayout.resolveCanvas(frameCount: options.numFrames)
        }
        let generationFrameCount = dfrCanvas?.frameCount
            ?? (options.hdrICLoRA?.highQuality == true ? 2 * options.numFrames - 1 : options.numFrames)
        guard options.fps.isFinite, options.fps > 0 else {
            throw LTXUnifiedAVGeneratorError.invalidFrameRate(options.fps)
        }
        guard options.inferenceSteps > 0 else {
            throw LTXUnifiedAVGeneratorError.invalidInferenceSteps(options.inferenceSteps)
        }
        guard options.imageFrameIndex >= 0 else {
            throw LTXUnifiedAVGeneratorError.invalidImageFrameIndex(options.imageFrameIndex)
        }
        guard stage1DistilledLoRAStrength.isFinite,
              options.distilledLoRAStrengthStage2.isFinite else {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "Distilled LoRA strengths must be finite."
            )
        }
        guard options.imageStrength >= 0, options.imageStrength <= 1 else {
            throw LTXUnifiedAVGeneratorError.invalidImageStrength(options.imageStrength)
        }
        guard options.endImageStrength >= 0, options.endImageStrength <= 1 else {
            throw LTXUnifiedAVGeneratorError.invalidImageStrength(options.endImageStrength)
        }
        let imageConditionings = ltx25ImageConditionings(options: options)
        var referenceVideos = options.referenceVideos
        if !referenceVideos.isEmpty {
            _ = try ltxLoRAReferenceScaleConfiguration(options.loras)
        }
        let hasEXRInput = imageConditionings.contains { MediaHDRImageIO.isEXR($0.imageURL) }
            || referenceVideos.contains { MediaHDRImageIO.isEXRDirectory($0.videoURL) }
            || options.retake.map { MediaHDRImageIO.isEXRDirectory($0.sourceVideoURL) } == true
        if hasEXRInput, options.hdrColorSpace == nil {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "EXR conditioning requires an explicit HDR color space."
            )
        }
        if options.hdrColorSpace != nil, !loadedForLTX25 {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "HDR generation requires an official LTX-2.5 checkpoint."
            )
        }
        if options.hdrColorSpace != nil, options.dubIt != nil {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "Dub-It does not support HDR output."
            )
        }
        if let dubIt = options.dubIt {
            guard loadedForLTX25, !loadedForFullTwoStage else {
                throw LTXUnifiedAVGeneratorError.dubItRequiresLTX25(loadedRoot)
            }
            guard options.referenceVideos.isEmpty else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "Dub-It owns its single reference video; do not also supply referenceVideos."
                )
            }
            guard options.loras.count == 1 else {
                throw LTXUnifiedAVGeneratorError.dubItRequiresOneICLoRA(options.loras.count)
            }
            let icLoRA = options.loras[0]
            guard FileManager.default.fileExists(atPath: icLoRA.url.path) else {
                throw LTXUnifiedAVGeneratorError.loraMissing(icLoRA.url)
            }
            guard FileManager.default.fileExists(atPath: dubIt.referenceVideoURL.path) else {
                throw LTXUnifiedAVGeneratorError.referenceVideoNotFound(dubIt.referenceVideoURL)
            }
            guard MediaVideoIO.hasAudioTrack(dubIt.referenceVideoURL) else {
                throw LTXUnifiedAVGeneratorError.dubItReferenceAudioMissing(dubIt.referenceVideoURL)
            }
            referenceVideos = [
                LTXReferenceVideoConditioningInput(
                    videoURL: dubIt.referenceVideoURL,
                    strength: dubIt.referenceStrength,
                    attentionStrength: 1,
                    downscaleFactor: ltxLoRAReferenceDownscaleFactor(icLoRA),
                    temporalScaleFactor: ltxLoRAReferenceTemporalScaleFactor(icLoRA)
                ),
            ]
        }
        guard options.generatedKeyframeCount >= 0,
              options.generatedKeyframeCount == 0 || options.generatedKeyframeIndices.isEmpty else {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "Use a generated-keyframe count or explicit indices, not both."
            )
        }
        let requestedGeneratedIndices = options.generatedKeyframeCount > 0
            ? try ltxEvenlySpacedGeneratedKeyframePositions(
                count: options.generatedKeyframeCount,
                numFrames: generationFrameCount
            )
            : options.generatedKeyframeIndices
        let requestsLTX25Conditioning = !options.imageConditionings.isEmpty
            || !requestedGeneratedIndices.isEmpty
            || !referenceVideos.isEmpty
            || options.retake != nil
            || options.dubIt != nil
            || usesDFR
        guard loadedForLTX25 || !requestsLTX25Conditioning else {
            throw LTXUnifiedAVGeneratorError.ltx25ConditioningRequiresLTX25
        }
        if loadedForLTX25 {
            for input in imageConditionings {
                guard input.pixelFrameIndex >= 0, input.pixelFrameIndex < generationFrameCount else {
                    throw LTXUnifiedAVGeneratorError.invalidImageFrameIndex(input.pixelFrameIndex)
                }
                guard input.strength >= 0, input.strength <= 1 else {
                    throw LTXUnifiedAVGeneratorError.invalidImageStrength(input.strength)
                }
                guard FileManager.default.fileExists(atPath: input.imageURL.path) else {
                    throw LTXUnifiedAVGeneratorError.imageNotFound(input.imageURL)
                }
            }
            for reference in referenceVideos {
                guard FileManager.default.fileExists(atPath: reference.videoURL.path) else {
                    throw LTXUnifiedAVGeneratorError.referenceVideoNotFound(reference.videoURL)
                }
                if let maskURL = reference.attentionMaskVideoURL,
                   !FileManager.default.fileExists(atPath: maskURL.path) {
                    throw LTXUnifiedAVGeneratorError.referenceVideoNotFound(maskURL)
                }
                guard (generationWidth / 2).isMultiple(of: reference.downscaleFactor),
                      (generationHeight / 2).isMultiple(of: reference.downscaleFactor) else {
                    throw LTXUnifiedAVGeneratorError.invalidResolution(
                        width: options.width,
                        height: options.height
                    )
                }
            }
        }
        if let retake = options.retake {
            guard loadedForLTX25 else {
                throw LTXUnifiedAVGeneratorError.retakeRequiresLTX25(loadedRoot)
            }
            guard FileManager.default.fileExists(atPath: retake.sourceVideoURL.path) else {
                throw LTXUnifiedAVGeneratorError.referenceVideoNotFound(retake.sourceVideoURL)
            }
            let duration = Double(generationFrameCount) / Double(options.fps)
            guard retake.startTime >= 0,
                  retake.startTime < retake.endTime,
                  retake.endTime <= duration else {
                throw LTXUnifiedAVGeneratorError.invalidRetakeRange(
                    start: retake.startTime,
                    end: retake.endTime
                )
            }
        }
        let generatedIndices = Array(
            Set(requestedGeneratedIndices + (dfrCanvas?.keyframePositions ?? []))
        ).sorted()
        let generatedIndicesAreOrdered = zip(
            generatedIndices,
            generatedIndices.dropFirst()
        ).allSatisfy(<)
        guard generatedIndicesAreOrdered,
              generatedIndices.allSatisfy({ $0 >= 0 && $0 < generationFrameCount }) else {
            throw LTXUnifiedAVGeneratorError.invalidGeneratedKeyframes(generatedIndices)
        }
        // LTX-2.5 is natively token-state based even when no conditioning tokens are appended.
        // Keeping the plain latent path for it would bypass generated-slot masks, attention masks,
        // and the upstream sampler/guidance implementation for ordinary text-to-video requests.
        let usesLTX25TokenState = loadedForLTX25

        let usesDevOneStage = options.pipeline == .devOneStage
        let usesKeyframeInterpolation = options.pipeline == .keyframeInterpolation
        let usesRetakeOneStage = options.retake != nil
        if usesDevOneStage {
            guard loadedForLTX25, loadedForFullTwoStage else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "The dev one-stage pipeline requires the full LTX-2.5 checkpoint."
                )
            }
            guard !usesDFR,
                  options.retake == nil,
                  options.dubIt == nil,
                  referenceVideos.isEmpty else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "The dev one-stage pipeline supports text, image, and generated-keyframe conditioning only."
                )
            }
            guard stage1DistilledLoRAStrength == 0,
                  options.distilledLoRAStrengthStage2 == 0 else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "The dev one-stage pipeline runs the full transformer without the distilled LoRA."
                )
            }
        }
        if usesKeyframeInterpolation {
            guard loadedForLTX25, loadedForFullTwoStage else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "The keyframe-interpolation pipeline requires the full LTX-2.5 checkpoint."
                )
            }
            guard !usesDFR,
                  options.retake == nil,
                  options.dubIt == nil,
                  referenceVideos.isEmpty,
                  requestedGeneratedIndices.isEmpty else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "The keyframe-interpolation pipeline accepts timed image guides but not DFR, retake, Dub-It, IC-LoRA references, or generated-keyframe slots."
                )
            }
        }
        let usesFullTwoStage = loadedForFullTwoStage
        let usesReusableFullTwoStage = loadedForReusableFullTwoStage
        let usesGuidedFullTwoStage = usesFullTwoStage && !usesDFR && options.dubIt == nil
        if let teaCache = options.teaCache {
            guard usesGuidedFullTwoStage else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "TeaCache is calibrated only for full LTX-2.5 two-stage generation."
                )
            }
            guard options.transformerExecution == .eager else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "TeaCache requires eager transformer execution so its calibrated gate remains stable."
                )
            }
            guard options.sampler.mode == .euler || options.sampler.mode == .res2s else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "TeaCache supports the calibrated Euler and Res2S full-generation paths."
                )
            }
            guard options.inferenceSteps >= 8 else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "TeaCache requires at least 8 stage-one inference steps."
                )
            }
            if let threshold = teaCache.threshold,
               !threshold.isFinite || threshold <= 0 {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "TeaCache threshold must be finite and positive."
                )
            }
        }
        let usesPlainLTX25DistilledPipeline = ltx25UsesDistilledAncestralStage1(
            isLTX25: loadedForLTX25,
            isFullTwoStage: usesFullTwoStage,
            usesDFR: usesDFR,
            usesHDRICLoRA: usesHDRICLoRA,
            usesRetake: options.retake != nil,
            usesDubIt: options.dubIt != nil,
            hasReferenceVideos: !referenceVideos.isEmpty
        )
        let positivePromptCacheKey = LTXPromptEmbeddingCacheKey(
            prompt: prompt,
            maxLength: options.maxTextLength
        )
        let negativePromptCacheKey = LTXPromptEmbeddingCacheKey(
            prompt: options.negativePrompt,
            maxLength: options.maxTextLength
        )
        let needsPositiveTextEncoder = options.precomputedTextEmbeddingsURL == nil
            && !promptEmbeddingCache.contains(positivePromptCacheKey)
        let needsNegativeTextEncoder = usesGuidedFullTwoStage
            && !promptEmbeddingCache.contains(negativePromptCacheKey)
        let needsTextEncoder = needsPositiveTextEncoder || needsNegativeTextEncoder
        if needsTextEncoder, textEncoder == nil {
            let reloadStart = ltxMonotonicSeconds()
            try await loadTextEncoderIfNeeded()
            textEncoderReloadSeconds = ltxMonotonicSeconds() - reloadStart
        }
        guard let transformer else {
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }
        if let transformerV2 = transformer as? LTXUnifiedAVTransformerV2 {
            transformerV2.execution = options.transformerExecution
        }
        if needsTextEncoder, textEncoder == nil {
            throw LTXUnifiedAVGeneratorError.generatorNotLoaded
        }
        guard let decoder else {
            throw LTXUnifiedAVGeneratorError.decoderNotLoaded
        }
        guard let upsampler else {
            throw LTXUnifiedAVGeneratorError.upsamplerNotLoaded
        }
        try installRuntimeLoRAsIfNeeded(
            user: options.loras,
            detailing: options.dfr?.detailingLoRAs ?? []
        )
        runtimeUserLoRAAdapters.forEach { $0.setActive(true) }
        runtimeDetailingLoRAAdapters.forEach { $0.setActive(false) }
        let fullLoRAURL: URL?
        if usesFullTwoStage, !usesDevOneStage {
            guard !twoStageGenerationConsumed else {
                throw LTXUnifiedAVGeneratorError.fullGenerationRequiresReload
            }
            if usesReusableFullTwoStage {
                guard runtimeLoRAAdapter != nil else {
                    throw LTXUnifiedAVGeneratorError.generatorNotLoaded
                }
                fullLoRAURL = nil
            } else {
                guard let distilledLoRAURL else {
                    throw LTXUnifiedAVGeneratorError.generatorNotLoaded
                }
                fullLoRAURL = distilledLoRAURL
            }
            twoStageGenerationConsumed = true
        } else {
            fullLoRAURL = nil
        }
        if stage1DistilledLoRAStrength != 0 {
            guard usesReusableFullTwoStage, let runtimeLoRAAdapter else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "A non-zero stage-one distilled LoRA strength requires the reusable full-model load path."
                )
            }
            runtimeLoRAAdapter.setStrength(stage1DistilledLoRAStrength)
            runtimeLoRAAdapter.setActive(true)
        }
        defer {
            runtimeUserLoRAAdapters.forEach { $0.setActive(false) }
            runtimeDetailingLoRAAdapters.forEach { $0.setActive(false) }
            if usesReusableFullTwoStage {
                runtimeLoRAAdapter?.setActive(false)
                twoStageGenerationConsumed = false
            }
        }

        let textEncodingStart = ltxMonotonicSeconds()
        let videoContext: MLXArray
        let audioContext: MLXArray
        if let embeddingsURL = options.precomputedTextEmbeddingsURL {
            let embeddings = try MLX.loadArrays(url: embeddingsURL)
            guard let loadedVideoContext = embeddings["video_context"],
                  let loadedAudioContext = embeddings["audio_context"] else {
                throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                    "Precomputed embeddings must contain video_context and audio_context tensors."
                )
            }
            videoContext = loadedVideoContext.asType(loadedDType)
            audioContext = loadedAudioContext.asType(loadedDType)
        } else {
            let result = try await cachedPromptEmbeddings(
                prompt: prompt,
                maxLength: options.maxTextLength
            )
            promptCacheHits += result.cacheHit ? 1 : 0
            promptCacheMisses += result.cacheHit ? 0 : 1
            videoContext = result.embeddings.video
            guard let loadedAudioContext = result.embeddings.audio else {
                throw LTXUnifiedAVGeneratorError.audioEmbeddingsMissing
            }
            audioContext = loadedAudioContext
        }
        var negativeVideoContext: MLXArray?
        var negativeAudioContext: MLXArray?
        if usesGuidedFullTwoStage {
            let result = try await cachedPromptEmbeddings(
                prompt: options.negativePrompt,
                maxLength: options.maxTextLength
            )
            promptCacheHits += result.cacheHit ? 1 : 0
            promptCacheMisses += result.cacheHit ? 0 : 1
            negativeVideoContext = result.embeddings.video
            guard let audioEmbeddings = result.embeddings.audio else {
                throw LTXUnifiedAVGeneratorError.audioEmbeddingsMissing
            }
            negativeAudioContext = audioEmbeddings
        }
        if let negativeVideoContext, let negativeAudioContext {
            MLX.eval(videoContext, audioContext, negativeVideoContext, negativeAudioContext)
        } else {
            MLX.eval(videoContext, audioContext)
        }
        if let textEncoder {
            await textEncoder.unload()
            self.textEncoder = nil
        }
        Memory.clearCache()
        ltxTraceMemory("text-context-ready")
        textEncodingSeconds = textEncoderReloadSeconds + ltxMonotonicSeconds() - textEncodingStart
        let preparationStart = ltxMonotonicSeconds()

        let latentFrames = 1 + ((generationFrameCount - 1) / 8)
        let usesTargetResolutionStageOne = usesDevOneStage || usesRetakeOneStage
        let stage1H = generationHeight / (usesTargetResolutionStageOne ? 32 : 64)
        let stage1W = generationWidth / (usesTargetResolutionStageOne ? 32 : 64)
        let stage2H = generationHeight / 32
        let stage2W = generationWidth / 32
        let audioFrames = computeAudioLatentFrameCount(
            videoFrames: generationFrameCount,
            fps: max(1, options.fps)
        )
        let stage1Sigmas = try validatedLTXSigmaSchedule(
            options.sigmas ?? (usesGuidedFullTwoStage
                ? LTX2DiffusionScheduler.sigmas(
                    steps: options.inferenceSteps,
                    tokenCount: latentFrames * stage1H * stage1W
                )
                : STAGE1Sigmas)
        )
        let stage2Sigmas = try validatedLTXSigmaSchedule(options.stage2Sigmas ?? STAGE2Sigmas)

        MLXRandom.seed(UInt64(bitPattern: Int64(options.seed)))
        let modelDType = videoContext.dtype

        let isImageToVideo = options.sourceImageURL != nil
        var stage1ConditioningState: LTXLatentConditioningState?
        var stage2ConditioningState: LTXLatentConditioningState?
        var stage2ConditioningLatent: MLXArray?
        var stage2EndConditioningLatent: MLXArray?
        var stage1TokenState: LTX25VideoTokenState?
        var stage2TokenState: LTX25VideoTokenState?
        var generatedKeyframeLatents: MLXArray?
        var stage2LTX25ImageLatents: [MLXArray] = []
        var stage2LTX25ReferenceLatents: [MLXArray] = []
        var stage2LTX25ReferenceAttentionWeights: [MLXArray?] = []
        var stage1RetakeLatent: MLXArray?

        var videoLatents: MLXArray
        if usesLTX25TokenState {
            if !imageConditionings.isEmpty || !referenceVideos.isEmpty || options.retake != nil {
                try loadEncoderIfNeeded()
            }
            if let retake = options.retake {
                guard let encoder else {
                    throw LTXUnifiedAVGeneratorError.encoderNotLoaded
                }
                let sourceVideo = try loadVideoForEncoding(
                    url: retake.sourceVideoURL,
                    width: generationWidth,
                    height: generationHeight,
                    frameCap: generationFrameCount,
                    temporalScaleFactor: 1,
                    dtype: modelDType,
                    hdrColorSpace: options.hdrColorSpace
                )
                stage1RetakeLatent = encoder.encode(image: sourceVideo)
                if let stage1RetakeLatent {
                    MLX.eval(stage1RetakeLatent)
                }
            }
            videoLatents = stage1RetakeLatent ?? MLX.zeros(
                [1, 128, latentFrames, stage1H, stage1W],
                dtype: modelDType
            )
        } else if isImageToVideo {
            let sourceImageURL = options.sourceImageURL!
            guard FileManager.default.fileExists(atPath: sourceImageURL.path) else {
                throw LTXUnifiedAVGeneratorError.imageNotFound(sourceImageURL)
            }
            try loadEncoderIfNeeded()
            guard let encoder else {
                throw LTXUnifiedAVGeneratorError.encoderNotLoaded
            }
            if options.imageFrameIndex >= latentFrames {
                throw LTXUnifiedAVGeneratorError.invalidImageFrameIndex(options.imageFrameIndex)
            }

            let stage1Image = try loadImageForEncoding(
                url: sourceImageURL,
                width: generationWidth / 2,
                height: generationHeight / 2,
                dtype: modelDType,
                hdrColorSpace: options.hdrColorSpace
            )
            let stage1ImageLatent = encoder.encode(image: stage1Image)

            let stage2Image = try loadImageForEncoding(
                url: sourceImageURL,
                width: generationWidth,
                height: generationHeight,
                dtype: modelDType,
                hdrColorSpace: options.hdrColorSpace
            )
            let stage2ImageLatent = encoder.encode(image: stage2Image)
            stage2ConditioningLatent = stage2ImageLatent

            // Optional end keyframe -> conditions the tail latent frame so the clip
            // interpolates a directed start->end motion.
            var stage1EndImageLatent: MLXArray?
            if let endImageURL = options.endImageURL {
                let stage1EndImage = try loadImageForEncoding(
                    url: endImageURL,
                    width: generationWidth / 2,
                    height: generationHeight / 2,
                    dtype: modelDType,
                    hdrColorSpace: options.hdrColorSpace
                )
                stage1EndImageLatent = encoder.encode(image: stage1EndImage)
                let stage2EndImage = try loadImageForEncoding(
                    url: endImageURL,
                    width: generationWidth,
                    height: generationHeight,
                    dtype: modelDType,
                    hdrColorSpace: options.hdrColorSpace
                )
                stage2EndConditioningLatent = encoder.encode(image: stage2EndImage)
            }

            var state1 = applyLatentConditioning(
                baseLatent: MLX.zeros([1, 128, latentFrames, stage1H, stage1W], dtype: modelDType),
                conditionedLatent: stage1ImageLatent,
                frameIndex: options.imageFrameIndex,
                strength: options.imageStrength,
                endConditionedLatent: stage1EndImageLatent,
                endFrameIndex: -1,
                endStrength: options.endImageStrength
            )
            let stage1Noise = MLXRandom.normal(state1.latent.shape).asType(modelDType)
            let stage1Sigma = MLXArray(stage1Sigmas[0]).asType(modelDType)
            let one = MLXArray(1.0).asType(modelDType)
            let scaledMask = state1.denoiseMask * stage1Sigma
            state1.latent = stage1Noise * scaledMask + state1.latent * (one - scaledMask)
            videoLatents = state1.latent
            MLX.eval(videoLatents)
            stage1ConditioningState = state1
        } else {
            videoLatents = MLXRandom.normal([1, 128, latentFrames, stage1H, stage1W]).asType(modelDType)
            MLX.eval(videoLatents)
        }
        let baseStage1VideoPositions = createPositionGrid(
            batchSize: 1,
            numFrames: latentFrames,
            height: stage1H,
            width: stage1W,
            temporalScale: 8,
            spatialScale: 32,
            fps: Float(max(1, options.fps)),
            causalFix: true
        )
        if usesLTX25TokenState {
            var state = try makeConditionedLTX25VideoTokenState(
                initialLatent: videoLatents,
                positions: baseStage1VideoPositions,
                imageConditionings: imageConditionings,
                generatedKeyframeIndices: generatedIndices,
                initialGeneratedKeyframes: nil,
                encoder: encoder,
                pixelWidth: generationWidth / (usesTargetResolutionStageOne ? 1 : 2),
                pixelHeight: generationHeight / (usesTargetResolutionStageOne ? 1 : 2),
                fps: options.fps,
                replaceFirstImage: !usesKeyframeInterpolation,
                hdrColorSpace: options.hdrColorSpace
            )
            if let retake = options.retake, let stage1RetakeLatent {
                state.applyTemporalRetake(
                    cleanVideoLatent: stage1RetakeLatent,
                    startTime: retake.startTime,
                    endTime: retake.endTime,
                    fps: options.fps,
                    regenerate: retake.regenerateVideo
                )
            }
            if !referenceVideos.isEmpty {
                guard let encoder else {
                    throw LTXUnifiedAVGeneratorError.encoderNotLoaded
                }
                for reference in referenceVideos {
                    let pixelVideo = try loadVideoForEncoding(
                        url: reference.videoURL,
                        width: (generationWidth / 2) / reference.downscaleFactor,
                        height: (generationHeight / 2) / reference.downscaleFactor,
                        frameCap: generationFrameCount,
                        temporalScaleFactor: reference.temporalScaleFactor,
                        dtype: modelDType,
                        hdrColorSpace: options.hdrColorSpace,
                        duplicateEachFrame: options.hdrICLoRA?.highQuality == true,
                        hdrICLoRAReference: usesHDRICLoRA
                    )
                    let encoded = encoder.encode(image: pixelVideo)
                    MLX.eval(encoded)
                    let attentionWeights = try loadLTXReferenceAttentionWeights(
                        reference: reference,
                        width: (generationWidth / 2) / reference.downscaleFactor,
                        height: (generationHeight / 2) / reference.downscaleFactor,
                        frameCap: generationFrameCount,
                        targetLatent: encoded,
                        dtype: modelDType
                    )
                    state.appendReferenceLatent(
                        encoded,
                        downscaleFactor: reference.downscaleFactor,
                        temporalScaleFactor: reference.temporalScaleFactor,
                        strength: reference.strength,
                        attentionStrength: reference.attentionStrength,
                        attentionWeights: attentionWeights,
                        fps: options.fps
                    )
                    if !options.skipStage2 {
                        let stage2PixelVideo = try loadVideoForEncoding(
                            url: reference.videoURL,
                            width: generationWidth / reference.downscaleFactor,
                            height: generationHeight / reference.downscaleFactor,
                            frameCap: generationFrameCount,
                            temporalScaleFactor: reference.temporalScaleFactor,
                            dtype: modelDType,
                            hdrColorSpace: options.hdrColorSpace,
                            duplicateEachFrame: options.hdrICLoRA?.highQuality == true,
                            hdrICLoRAReference: usesHDRICLoRA
                        )
                        let stage2Encoded = encoder.encode(image: stage2PixelVideo)
                        MLX.eval(stage2Encoded)
                        stage2LTX25ReferenceLatents.append(stage2Encoded)
                        stage2LTX25ReferenceAttentionWeights.append(
                            try loadLTXReferenceAttentionWeights(
                                reference: reference,
                                width: generationWidth / reference.downscaleFactor,
                                height: generationHeight / reference.downscaleFactor,
                                frameCap: generationFrameCount,
                                targetLatent: stage2Encoded,
                                dtype: modelDType
                            )
                        )
                    }
                }
            }
            state.addNoise(scale: stage1Sigmas[0])
            MLX.eval(state.latent)
            stage1TokenState = state
            if !usesDevOneStage, !imageConditionings.isEmpty {
                guard let encoder else {
                    throw LTXUnifiedAVGeneratorError.encoderNotLoaded
                }
                for input in imageConditionings {
                    let image = try loadImageForEncoding(
                        url: input.imageURL,
                        width: generationWidth,
                        height: generationHeight,
                        dtype: modelDType,
                        hdrColorSpace: options.hdrColorSpace,
                        crf: input.crf ?? 18
                    )
                    let encoded = encoder.encode(image: image)
                    MLX.eval(encoded)
                    stage2LTX25ImageLatents.append(encoded)
                }
            }
        }
        encoder = nil
        Memory.clearCache()
        ltxTraceMemory("image-conditioning-ready")

        if usesGuidedFullTwoStage {
            MLXRandom.seed(UInt64(bitPattern: Int64(options.seed &+ 1)))
        }
        var audioLatents = MLXRandom.normal(
            [1, LTXAudioLatentChannels, audioFrames, LTXAudioLatentMelBins]
        ).asType(modelDType)
        var stage1AudioConditioning: LTXLatentConditioningState?
        var stage1DubItAudioReference: LTXAudioReferenceConditioningState?
        var stage1DubItAudioLatent: MLXArray?
        if let retake = options.retake,
           !MediaHDRImageIO.isEXRDirectory(retake.sourceVideoURL),
           MediaVideoIO.hasAudioTrack(retake.sourceVideoURL),
           let audioVAEWeightsURL {
            let duration = Double(generationFrameCount) / Double(options.fps)
            let sourceAudio = try MediaAudioIO.decodeSegment(
                retake.sourceVideoURL,
                startTime: 0,
                duration: duration,
                targetSampleRate: LTXAudioMelProcessor.sampleRate,
                channels: 2
            )
            let spectrogram = LTXAudioMelProcessor().extract(
                channels: planarAudioChannels(sourceAudio)
            )
            let cleanAudioLatent = try encodeLTX23AudioLatents(
                spectrogram: spectrogram,
                requiredFrameCount: audioFrames,
                weightsURL: audioVAEWeightsURL,
                dtype: loadedDType,
                sourceLayout: loadedForLTX25 ? .pytorch : .mlx
            ).asType(modelDType)
            let conditioning = makeLTXAudioTemporalConditioning(
                cleanLatent: cleanAudioLatent,
                startTime: retake.startTime,
                endTime: retake.endTime,
                regenerate: retake.regenerateAudio
            )
            let one = MLXArray(1).asType(modelDType)
            let noiseScale = MLXArray(stage1Sigmas[0]).asType(modelDType)
            let noisedAudio = audioLatents * noiseScale + cleanAudioLatent * (one - noiseScale)
            audioLatents = noisedAudio * conditioning.denoiseMask
                + cleanAudioLatent * (one - conditioning.denoiseMask)
            stage1AudioConditioning = conditioning
        } else if let dubIt = options.dubIt {
            guard let audioVAEWeightsURL else {
                throw LTXUnifiedAVGeneratorError.audioVAEWeightsMissing(
                    loadedRoot ?? dubIt.referenceVideoURL.deletingLastPathComponent()
                )
            }
            let duration = Double(generationFrameCount) / Double(options.fps)
            let referenceAudio = try MediaAudioIO.decodeSegment(
                dubIt.referenceVideoURL,
                startTime: 0,
                duration: duration,
                targetSampleRate: LTXAudioMelProcessor.sampleRate,
                channels: 2
            )
            let referenceSpectrogram = LTXAudioMelProcessor().extract(
                channels: planarAudioChannels(referenceAudio)
            )
            let referenceAudioLatent = try encodeLTX23AudioLatents(
                spectrogram: referenceSpectrogram,
                requiredFrameCount: audioFrames,
                weightsURL: audioVAEWeightsURL,
                dtype: loadedDType,
                sourceLayout: loadedForLTX25 ? .pytorch : .mlx
            ).asType(modelDType)
            let conditioning = makeLTXAudioReferenceConditioning(
                targetLatent: audioLatents,
                referenceLatent: referenceAudioLatent,
                frozenTarget: false
            )
            audioLatents = conditioning.state.latent
            stage1AudioConditioning = conditioning.state
            stage1DubItAudioReference = conditioning
        }
        MLX.eval(audioLatents)

        let stage1VideoPositions = stage1TokenState?.positions ?? baseStage1VideoPositions
        let stage1VideoRope = precomputeSplitRope(
            positions: stage1VideoPositions,
            dim: 4096,
            theta: 10_000.0,
            maxPos: [20, 2048, 2048],
            numHeads: 32
        )
        let stage1VideoCrossPositions = stage1VideoPositions[0..., 0..<1, 0..., 0...]
        let stage1VideoCrossRope = precomputeSplitRope(
            positions: stage1VideoCrossPositions,
            dim: 2048,
            theta: 10_000.0,
            maxPos: [20],
            numHeads: 32
        )
        let stage1AudioPositions = stage1DubItAudioReference?.positions
            ?? createAudioPositionGrid(batchSize: 1, audioFrames: audioFrames)
        let stage1AudioRope = precomputeSplitRope(
            positions: stage1AudioPositions,
            dim: 2048,
            theta: 10_000.0,
            maxPos: [20],
            numHeads: 32
        )

        preparationSeconds = ltxMonotonicSeconds() - preparationStart
        let stage1DenoiseStart = ltxMonotonicSeconds()
        if usesHDRICLoRA, let tokenState = stage1TokenState {
            guard let transformerV2 = transformer as? LTXUnifiedAVTransformerV2 else {
                throw LTXUnifiedAVGeneratorError.generatorNotLoaded
            }
            let result = denoiseLTX25VideoTokenLoop(
                videoState: tokenState,
                videoRope: stage1VideoRope,
                videoContext: videoContext,
                transformer: transformerV2,
                sigmas: stage1Sigmas,
                ancestralNoiseSeed: options.seed,
                ancestralEta: 0
            )
            stage1TokenState = result
            videoLatents = result.mainLatent()
            generatedKeyframeLatents = result.generatedKeyframes()
        } else if let tokenState = stage1TokenState {
            let result: (video: LTX25VideoTokenState, audio: MLXArray)
            if usesGuidedFullTwoStage {
                guard let negativeVideoContext, let negativeAudioContext else {
                    throw LTXUnifiedAVGeneratorError.generatorNotLoaded
                }
                result = denoiseGuidedLTX25AVTokenLoop(
                    videoState: tokenState,
                    audioLatents: audioLatents,
                    videoRope: stage1VideoRope,
                    audioRope: stage1AudioRope,
                    videoCrossRope: stage1VideoCrossRope,
                    audioCrossRope: stage1AudioRope,
                    positiveVideoContext: videoContext,
                    negativeVideoContext: negativeVideoContext,
                    positiveAudioContext: audioContext,
                    negativeAudioContext: negativeAudioContext,
                    transformer: transformer,
                    sigmas: stage1Sigmas,
                    videoGuidance: options.videoGuidance,
                    audioGuidance: options.audioGuidance,
                    sampler: options.sampler,
                    seed: options.seed,
                    guidanceProjectionCache: options.teaCache == nil
                        ? options.guidanceProjectionCache
                        : .disabled,
                    guidanceProjectionCacheMetrics: guidanceProjectionCacheMetrics,
                    teaCacheController: teaCacheController,
                    teaCachePipelineStage: .coarse,
                    audioConditioning: stage1AudioConditioning
                )
            } else {
                result = denoiseLTX25AVTokenLoop(
                    videoState: tokenState,
                    audioLatents: audioLatents,
                    videoRope: stage1VideoRope,
                    audioRope: stage1AudioRope,
                    videoCrossRope: stage1VideoCrossRope,
                    audioCrossRope: stage1AudioRope,
                    videoContext: videoContext,
                    audioContext: audioContext,
                    transformer: transformer,
                    sigmas: stage1Sigmas,
                    ancestralNoiseSeed: usesPlainLTX25DistilledPipeline
                        ? options.seed &+ 10_000
                        : nil,
                    audioConditioning: stage1AudioConditioning
                )
            }
            stage1TokenState = result.video
            videoLatents = result.video.mainLatent()
            generatedKeyframeLatents = result.video.generatedKeyframes()
            audioLatents = result.audio
        } else if usesGuidedFullTwoStage {
            guard let negativeVideoContext, let negativeAudioContext else {
                throw LTXUnifiedAVGeneratorError.generatorNotLoaded
            }
            (videoLatents, audioLatents) = denoiseGuidedAVLoop(
                videoLatents: videoLatents,
                audioLatents: audioLatents,
                videoRope: stage1VideoRope,
                audioRope: stage1AudioRope,
                videoCrossRope: stage1VideoCrossRope,
                audioCrossRope: stage1AudioRope,
                positiveVideoContext: videoContext,
                negativeVideoContext: negativeVideoContext,
                positiveAudioContext: audioContext,
                negativeAudioContext: negativeAudioContext,
                transformer: transformer,
                sigmas: stage1Sigmas,
                videoConditioning: stage1ConditioningState,
                videoGuidance: options.videoGuidance,
                audioGuidance: options.audioGuidance
            )
        } else {
            (videoLatents, audioLatents) = denoiseAVLoop(
                videoLatents: videoLatents,
                audioLatents: audioLatents,
                videoRope: stage1VideoRope,
                audioRope: stage1AudioRope,
                videoCrossRope: stage1VideoCrossRope,
                audioCrossRope: stage1AudioRope,
                videoContext: videoContext,
                audioContext: audioContext,
                transformer: transformer,
                sigmas: stage1Sigmas,
                videoConditioning: stage1ConditioningState,
                ancestralNoiseSeed: usesPlainLTX25DistilledPipeline
                    ? options.seed &+ 10_000
                    : nil
            )
        }
        if let stage1DubItAudioReference {
            let mainAudio = stage1DubItAudioReference.mainLatent(from: audioLatents)
            MLX.eval(mainAudio)
            stage1DubItAudioLatent = mainAudio
            audioLatents = mainAudio
        }
        MLX.eval(videoLatents, audioLatents)
        // The official full/HQ two-stage pipelines refine stage-two video only
        // and decode the full-context stage-one audio. DFR follows the same rule.
        let preservedStage1AudioLatents = (usesDFR || usesFullTwoStage) ? audioLatents : nil
        let dfrDetailingReferenceLatent = options.dfr?.detailingLoRAs.isEmpty == false
            ? videoLatents[0..<1, 0..., 0..., 0..., 0...]
            : nil
        if let dfrDetailingReferenceLatent {
            MLX.eval(dfrDetailingReferenceLatent)
        }
        stage1DenoiseSeconds = ltxMonotonicSeconds() - stage1DenoiseStart
        ltxTraceMemory("stage1-denoise-ready")

        if !usesDevOneStage, !usesRetakeOneStage, !options.skipStage2 {
            let loraFusionStart = ltxMonotonicSeconds()
            if usesReusableFullTwoStage {
                runtimeLoRAAdapter?.setStrength(options.distilledLoRAStrengthStage2)
                runtimeLoRAAdapter?.setActive(options.distilledLoRAStrengthStage2 != 0)
                loraFusionSeconds = ltxMonotonicSeconds() - loraFusionStart
            } else if let fullLoRAURL, options.distilledLoRAStrengthStage2 != 0 {
                try LTXStreamingLoRAFuser.fuse(
                    url: fullLoRAURL,
                    into: transformer,
                    strength: options.distilledLoRAStrengthStage2
                )
                loraFusionSeconds = ltxMonotonicSeconds() - loraFusionStart
            }

        let upsampleStart = ltxMonotonicSeconds()
        videoLatents = upsampleLatents(
            videoLatents,
            upsampler: upsampler,
            latentMean: decoder.latentsMean,
            latentStd: decoder.latentsStd
        )
        MLX.eval(videoLatents)
        let stage2InitialGeneratedKeyframes = generatedKeyframeLatents.map {
            upsampleLatents(
                $0,
                upsampler: upsampler,
                latentMean: decoder.latentsMean,
                latentStd: decoder.latentsStd
            )
        }
        if let stage2InitialGeneratedKeyframes {
            MLX.eval(stage2InitialGeneratedKeyframes)
        }
        upsampleSeconds = ltxMonotonicSeconds() - upsampleStart

        let baseStage2VideoPositions = createPositionGrid(
            batchSize: 1,
            numFrames: latentFrames,
            height: stage2H,
            width: stage2W,
            temporalScale: 8,
            spatialScale: 32,
            fps: Float(max(1, options.fps)),
            causalFix: true
        )
        var stage2AudioConditioning = stage1AudioConditioning
        var stage2DubItAudioReference: LTXAudioReferenceConditioningState?
        if usesHDRICLoRA {
            // HDR IC-LoRA runs video-only stage-two phases below. Each phase
            // applies its own initial noise and optional reference tokens.
        } else if usesLTX25TokenState {
            var state = LTX25VideoTokenState(
                initialLatent: videoLatents,
                positions: baseStage2VideoPositions
            )
            for (input, encoded) in zip(imageConditionings, stage2LTX25ImageLatents) {
                state.applyImageLatent(
                    encoded,
                    pixelFrameIndex: input.pixelFrameIndex,
                    strength: input.strength,
                    fps: options.fps,
                    replaceFirstFrame: !usesKeyframeInterpolation
                )
            }
            for index in referenceVideos.indices {
                let reference = referenceVideos[index]
                let encoded = stage2LTX25ReferenceLatents[index]
                state.appendReferenceLatent(
                    encoded,
                    downscaleFactor: reference.downscaleFactor,
                    temporalScaleFactor: reference.temporalScaleFactor,
                    strength: reference.strength,
                    attentionStrength: reference.attentionStrength,
                    attentionWeights: stage2LTX25ReferenceAttentionWeights[index],
                    fps: options.fps
                )
            }
            if !generatedIndices.isEmpty {
                state.appendGeneratedKeyframeSlots(
                    pixelFrameIndices: generatedIndices,
                    initialKeyframes: stage2InitialGeneratedKeyframes,
                    fps: options.fps
                )
            }
            if let dfrDetailingReferenceLatent, let dfr = options.dfr {
                state.appendReferenceLatent(
                    dfrDetailingReferenceLatent,
                    downscaleFactor: dfr.resolvedDetailingReferenceDownscaleFactor,
                    strength: 1,
                    fps: options.fps
                )
            }
            if usesFullTwoStage {
                MLXRandom.seed(UInt64(bitPattern: Int64(options.seed &+ 2)))
            }
            state.addNoise(scale: stage2Sigmas[0])
            MLX.eval(state.latent)
            stage2TokenState = state

            if stage1DubItAudioLatent == nil {
                if usesFullTwoStage {
                    MLXRandom.seed(UInt64(bitPattern: Int64(options.seed &+ 2)))
                }
                let noiseScale = MLXArray(stage2Sigmas[0]).asType(modelDType)
                let audioNoise = MLXRandom.normal(audioLatents.shape).asType(modelDType)
                audioLatents = audioNoise * noiseScale
                    + audioLatents * (MLXArray(1).asType(modelDType) - noiseScale)
                MLX.eval(audioLatents)
            }
        } else if let stage2State = stage2ConditioningLatent.map({
            applyLatentConditioning(
                baseLatent: videoLatents,
                conditionedLatent: $0,
                frameIndex: options.imageFrameIndex,
                strength: options.imageStrength,
                endConditionedLatent: stage2EndConditioningLatent,
                endFrameIndex: -1,
                endStrength: options.endImageStrength
            )
        }) {
            if usesFullTwoStage {
                MLXRandom.seed(UInt64(bitPattern: Int64(options.seed &+ 2)))
            }
            let noise = MLXRandom.normal(videoLatents.shape).asType(modelDType)
            let noiseScale = MLXArray(stage2Sigmas[0]).asType(modelDType)
            let one = MLXArray(1.0).asType(modelDType)
            let scaledMask = stage2State.denoiseMask * noiseScale
            videoLatents = noise * scaledMask + stage2State.latent * (one - scaledMask)
            MLX.eval(videoLatents)
            stage2ConditioningState = stage2State

            if usesFullTwoStage {
                MLXRandom.seed(UInt64(bitPattern: Int64(options.seed &+ 2)))
            }
            let audioNoise = MLXRandom.normal(audioLatents.shape).asType(modelDType)
            let oneMinusScale = MLXArray(1.0).asType(modelDType) - noiseScale
            audioLatents = audioNoise * noiseScale + audioLatents * oneMinusScale
            MLX.eval(audioLatents)
        } else {
            let noiseScale = MLXArray(stage2Sigmas[0]).asType(modelDType)
            let oneMinusScale = MLXArray(1.0 - stage2Sigmas[0]).asType(modelDType)
            if usesFullTwoStage {
                MLXRandom.seed(UInt64(bitPattern: Int64(options.seed &+ 2)))
            }
            let videoNoise = MLXRandom.normal(videoLatents.shape).asType(modelDType)
            if usesFullTwoStage {
                MLXRandom.seed(UInt64(bitPattern: Int64(options.seed &+ 2)))
            }
            let audioNoise = MLXRandom.normal(audioLatents.shape).asType(modelDType)
            videoLatents = videoNoise * noiseScale + videoLatents * oneMinusScale
            audioLatents = audioNoise * noiseScale + audioLatents * oneMinusScale
            MLX.eval(videoLatents, audioLatents)
        }
        if let stage1DubItAudioLatent {
            let conditioning = makeLTXAudioReferenceConditioning(
                targetLatent: stage1DubItAudioLatent,
                referenceLatent: stage1DubItAudioLatent,
                frozenTarget: true
            )
            audioLatents = conditioning.state.latent
            stage2AudioConditioning = conditioning.state
            stage2DubItAudioReference = conditioning
            MLX.eval(audioLatents)
        } else if let stage2AudioConditioning {
            let one = MLXArray(1).asType(modelDType)
            audioLatents = audioLatents * stage2AudioConditioning.denoiseMask
                + stage2AudioConditioning.cleanLatent
                    * (one - stage2AudioConditioning.denoiseMask)
            MLX.eval(audioLatents)
        }

        let stage2VideoPositions = stage2TokenState?.positions ?? baseStage2VideoPositions
        let stage2VideoRope = precomputeSplitRope(
            positions: stage2VideoPositions,
            dim: 4096,
            theta: 10_000.0,
            maxPos: [20, 2048, 2048],
            numHeads: 32
        )
        let stage2VideoCrossPositions = stage2VideoPositions[0..., 0..<1, 0..., 0...]
        let stage2VideoCrossRope = precomputeSplitRope(
            positions: stage2VideoCrossPositions,
            dim: 2048,
            theta: 10_000.0,
            maxPos: [20],
            numHeads: 32
        )
        let stage2AudioPositions = stage2DubItAudioReference?.positions ?? stage1AudioPositions
        let stage2AudioRope = precomputeSplitRope(
            positions: stage2AudioPositions,
            dim: 2048,
            theta: 10_000.0,
            maxPos: [20],
            numHeads: 32
        )

        let stage2DenoiseStart = ltxMonotonicSeconds()
        runtimeDetailingLoRAAdapters.forEach { $0.setActive(true) }
        if let hdrICLoRA = options.hdrICLoRA {
            guard let transformerV2 = transformer as? LTXUnifiedAVTransformerV2 else {
                throw LTXUnifiedAVGeneratorError.generatorNotLoaded
            }
            videoLatents = try denoiseLTXHDRICLoRAStage2(
                initialLatent: videoLatents,
                phases: hdrICLoRA.stage2Phases,
                referenceVideos: referenceVideos,
                referenceLatents: stage2LTX25ReferenceLatents,
                videoContext: videoContext,
                transformer: transformerV2,
                fps: options.fps,
                seed: options.seed
            )
        } else if let tokenState = stage2TokenState {
            let result: (video: LTX25VideoTokenState, audio: MLXArray)
            if usesGuidedFullTwoStage, options.sampler.mode == .res2s {
                guard let negativeVideoContext, let negativeAudioContext else {
                    throw LTXUnifiedAVGeneratorError.generatorNotLoaded
                }
                let neutralGuidance = LTXMultiModalGuidance(
                    classifierFreeScale: 1,
                    spatioTemporalScale: 0,
                    rescale: 0,
                    modalityScale: 1,
                    spatioTemporalBlocks: []
                )
                result = denoiseGuidedLTX25AVTokenLoop(
                    videoState: tokenState,
                    audioLatents: audioLatents,
                    videoRope: stage2VideoRope,
                    audioRope: stage2AudioRope,
                    videoCrossRope: stage2VideoCrossRope,
                    audioCrossRope: stage2AudioRope,
                    positiveVideoContext: videoContext,
                    negativeVideoContext: negativeVideoContext,
                    positiveAudioContext: audioContext,
                    negativeAudioContext: negativeAudioContext,
                    transformer: transformer,
                    sigmas: stage2Sigmas,
                    videoGuidance: neutralGuidance,
                    audioGuidance: neutralGuidance,
                    sampler: options.sampler,
                    seed: options.seed &+ 2,
                    guidanceProjectionCache: options.teaCache == nil
                        ? options.guidanceProjectionCache
                        : .disabled,
                    guidanceProjectionCacheMetrics: guidanceProjectionCacheMetrics,
                    teaCacheController: nil,
                    teaCachePipelineStage: .detail,
                    audioConditioning: stage2AudioConditioning
                )
            } else {
                result = denoiseLTX25AVTokenLoop(
                    videoState: tokenState,
                    audioLatents: audioLatents,
                    videoRope: stage2VideoRope,
                    audioRope: stage2AudioRope,
                    videoCrossRope: stage2VideoCrossRope,
                    audioCrossRope: stage2AudioRope,
                    videoContext: videoContext,
                    audioContext: audioContext,
                    transformer: transformer,
                    sigmas: stage2Sigmas,
                    audioConditioning: stage2AudioConditioning
                )
            }
            stage2TokenState = result.video
            videoLatents = result.video.mainLatent()
            generatedKeyframeLatents = result.video.generatedKeyframes()
            audioLatents = result.audio
            if let stage2DubItAudioReference {
                audioLatents = stage2DubItAudioReference.mainLatent(from: audioLatents)
            }
        } else {
            (videoLatents, audioLatents) = denoiseAVLoop(
                videoLatents: videoLatents,
                audioLatents: audioLatents,
                videoRope: stage2VideoRope,
                audioRope: stage2AudioRope,
                videoCrossRope: stage2VideoCrossRope,
                audioCrossRope: stage2AudioRope,
                videoContext: videoContext,
                audioContext: audioContext,
                transformer: transformer,
                sigmas: stage2Sigmas,
                videoConditioning: stage2ConditioningState
            )
        }
        MLX.eval(videoLatents, audioLatents)
        stage2DenoiseSeconds = ltxMonotonicSeconds() - stage2DenoiseStart
        ltxTraceMemory("stage2-denoise-ready")
        runtimeDetailingLoRAAdapters.forEach { $0.setActive(false) }
        }

        var finalFrameCount = generationFrameCount
        var finalFPS = options.fps
        var outputGeneratedIndices = generatedIndices
        if let dfr = options.dfr, dfr.temporalUpsampleRounds > 0 {
            guard let temporalUpsampler,
                  let transformerV2 = transformer as? LTXUnifiedAVTransformerV2,
                  var carryKeyframes = generatedKeyframeLatents,
                  !generatedIndices.isEmpty else {
                throw LTXUnifiedAVGeneratorError.dfrRequiresLTX25Full(
                    loadedRoot ?? URL(fileURLWithPath: "", isDirectory: true)
                )
            }
            var carryPositions = generatedIndices
            let temporalSigmas = Array(STAGE1Sigmas.dropFirst(4))

            for roundIndex in 1...dfr.temporalUpsampleRounds {
                videoLatents = upsampleLatentsTemporally(
                    videoLatents,
                    upsampler: temporalUpsampler,
                    latentMean: decoder.latentsMean,
                    latentStd: decoder.latentsStd
                )
                MLX.eval(videoLatents)
                finalFrameCount = 2 * (finalFrameCount - 1) + 1
                finalFPS *= 2
                let seamPositions = carryPositions.map { 2 * $0 }
                let seamIndex = Dictionary(
                    uniqueKeysWithValues: seamPositions.enumerated().map { ($0.element, $0.offset) }
                )
                let ranges = try LTX25DFRLayout.tileRanges(
                    seamPositions: seamPositions,
                    frameCount: finalFrameCount,
                    tileCount: 1 << roundIndex
                )
                var tileLatents: [MLXArray] = []
                var slotPositions: [Int] = []
                var slotLatents: [MLXArray] = []

                for (tileIndex, range) in ranges.enumerated() {
                    let tileVideo = videoLatents[
                        0...,
                        0...,
                        range.latentStart..<range.latentEndExclusive,
                        0...,
                        0...
                    ]
                    let localLatentFrames = range.latentEndExclusive - range.latentStart
                    let positions = createPositionGrid(
                        batchSize: 1,
                        numFrames: localLatentFrames,
                        height: stage2H,
                        width: stage2W,
                        temporalScale: 8,
                        spatialScale: 32,
                        fps: Float(min(finalFPS, 60)),
                        causalFix: true
                    )
                    var state = LTX25VideoTokenState(
                        initialLatent: tileVideo,
                        positions: positions
                    )

                    for (input, encoded) in zip(imageConditionings, stage2LTX25ImageLatents) {
                        if range.pixelStart == 0
                            || (input.pixelFrameIndex >= range.pixelStart
                                && input.pixelFrameIndex <= range.pixelEnd) {
                            state.applyImageLatent(
                                encoded,
                                pixelFrameIndex: input.pixelFrameIndex - range.pixelStart,
                                strength: input.strength,
                                fps: min(finalFPS, 60)
                            )
                        }
                    }

                    for globalPosition in range.anchorKeyframes {
                        guard let index = seamIndex[globalPosition] else {
                            throw LTX25DFRLayoutError.invalidSeams(seamPositions)
                        }
                        state.applyImageLatent(
                            carryKeyframes[0..., 0..., index..<index + 1, 0..., 0...],
                            pixelFrameIndex: globalPosition - range.pixelStart,
                            strength: 0.95,
                            fps: min(finalFPS, 60),
                            replaceFirstFrame: false
                        )
                    }

                    let localSlots = LTX25DFRLayout.remapPositionsToLocal(
                        range.slotKeyframes,
                        pixelStart: range.pixelStart
                    )
                    if !localSlots.isEmpty {
                        let initials = MLX.concatenated(
                            localSlots.map { localPosition in
                                let index = min(
                                    max((localPosition + 4) / 8, 0),
                                    tileVideo.dim(2) - 1
                                )
                                return tileVideo[0..., 0..., index..<index + 1, 0..., 0...]
                            },
                            axis: 2
                        )
                        state.appendGeneratedKeyframeSlots(
                            pixelFrameIndices: localSlots,
                            initialKeyframes: initials,
                            fps: min(finalFPS, 60)
                        )
                    }
                    state.addNoise(scale: temporalSigmas[0])
                    let rope = precomputeSplitRope(
                        positions: state.positions,
                        dim: 4096,
                        theta: 10_000,
                        maxPos: [20, 2048, 2048],
                        numHeads: 32
                    )
                    state = denoiseLTX25VideoTokenLoop(
                        videoState: state,
                        videoRope: rope,
                        videoContext: videoContext,
                        transformer: transformerV2,
                        sigmas: temporalSigmas,
                        ancestralNoiseSeed: options.seed + 1_000 * roundIndex + tileIndex,
                        ancestralEta: 0.5
                    )
                    tileLatents.append(state.mainLatent())
                    if let generated = state.generatedKeyframes() {
                        slotPositions.append(contentsOf: range.slotKeyframes)
                        slotLatents.append(generated)
                    }
                }

                videoLatents = try LTX25DFRLayout.stitchTileLatents(
                    tileLatents,
                    ranges: ranges
                )
                MLX.eval(videoLatents)

                var latentsByPosition: [Int: MLXArray] = [:]
                for (index, position) in seamPositions.enumerated() {
                    latentsByPosition[position] = carryKeyframes[
                        0...,
                        0...,
                        index..<index + 1,
                        0...,
                        0...
                    ]
                }
                var flatSlotIndex = 0
                for group in slotLatents {
                    for index in 0..<group.dim(2) {
                        let position = slotPositions[flatSlotIndex]
                        if latentsByPosition[position] == nil {
                            latentsByPosition[position] = group[
                                0...,
                                0...,
                                index..<index + 1,
                                0...,
                                0...
                            ]
                        }
                        flatSlotIndex += 1
                    }
                }
                carryPositions = latentsByPosition.keys.sorted()
                carryKeyframes = MLX.concatenated(
                    carryPositions.map { latentsByPosition[$0]! },
                    axis: 2
                )
                MLX.eval(carryKeyframes)
            }

            let targetFrameCount = (options.numFrames - 1) * dfr.playbackRateMultiplier + 1
            if targetFrameCount != finalFrameCount {
                let keepLatents = (targetFrameCount - 1) / 8 + 1
                videoLatents = videoLatents[0..., 0..., 0..<keepLatents, 0..., 0...]
                finalFrameCount = targetFrameCount
                MLX.eval(videoLatents)
            }
            let retainedCarry = carryPositions.enumerated().filter { $0.element < targetFrameCount }
            outputGeneratedIndices = retainedCarry.map(\.element)
            if retainedCarry.isEmpty {
                generatedKeyframeLatents = nil
            } else {
                generatedKeyframeLatents = MLX.concatenated(
                    retainedCarry.map { index, _ in
                        carryKeyframes[0..., 0..., index..<index + 1, 0..., 0...]
                    },
                    axis: 2
                )
                MLX.eval(generatedKeyframeLatents!)
            }
        } else if options.dfr != nil {
            let targetFrameCount = options.numFrames
            if targetFrameCount != finalFrameCount {
                let keepLatents = (targetFrameCount - 1) / 8 + 1
                videoLatents = videoLatents[0..., 0..., 0..<keepLatents, 0..., 0...]
                finalFrameCount = targetFrameCount
                MLX.eval(videoLatents)
            }
            let retained = generatedIndices.enumerated().filter { $0.element < targetFrameCount }
            outputGeneratedIndices = retained.map(\.element)
            if let currentGeneratedKeyframes = generatedKeyframeLatents, !retained.isEmpty {
                let filtered = MLX.concatenated(
                    retained.map { index, _ in
                        currentGeneratedKeyframes[0..., 0..., index..<index + 1, 0..., 0...]
                    },
                    axis: 2
                )
                MLX.eval(filtered)
                generatedKeyframeLatents = filtered
            } else if retained.isEmpty {
                generatedKeyframeLatents = nil
            }
        }
        if let preservedStage1AudioLatents {
            audioLatents = preservedStage1AudioLatents
            MLX.eval(audioLatents)
        }
        if usesReusableFullTwoStage {
            runtimeLoRAAdapter?.setActive(false)
        }

        let videoDecodeStart = ltxMonotonicSeconds()
        let decodedVideo: MLXArray?
        var frames: MLXArray
        var hdrOutput: LTXHDROutputFrames?
        if let diffusionDecoder {
            let fullDecoded = try diffusionDecoder.decode(
                sample: videoLatents,
                seed: options.seed
            )
            decodedVideo = fullDecoded
            if let colorSpace = options.hdrColorSpace {
                let output = LTXHDRColorPipeline.decode(
                    fullDecoded,
                    transfer: options.hdrTransfer,
                    exrColorSpace: colorSpace
                )
                hdrOutput = output
                frames = (output.working * MLXArray(Float(255))).asType(.uint8)
            } else {
                hdrOutput = nil
                frames = postprocessDecodedVideo(fullDecoded)
            }
        } else if let tiling = selectDecodeTilingConfig(
            width: generationWidth,
            height: generationHeight,
            numFrames: finalFrameCount,
            fps: finalFPS,
            spatialTileSizeInPixels: options.vaeSpatialTileSize
                ?? (usesHDRICLoRA ? 1_280 : nil),
            spatialTileOverlapInPixels: options.vaeSpatialTileSize != nil || usesHDRICLoRA
                ? options.vaeSpatialTileOverlap
                : 0
        ) {
            if let colorSpace = options.hdrColorSpace {
                let fullDecoded = decodeWithTilingRaw(
                    decoder: decoder,
                    latents: videoLatents,
                    spatialTileSizeInPixels: tiling.spatialTileSizeInPixels,
                    spatialOverlapInPixels: tiling.spatialTileOverlapInPixels,
                    temporalTileSizeInFrames: tiling.temporalTileSizeInFrames,
                    temporalOverlapInFrames: tiling.temporalTileOverlapInFrames,
                    spatialScale: 32,
                    temporalScale: 8
                )
                decodedVideo = fullDecoded
                let output = LTXHDRColorPipeline.decode(
                    fullDecoded,
                    transfer: options.hdrTransfer,
                    exrColorSpace: colorSpace
                )
                hdrOutput = output
                frames = (output.working * MLXArray(Float(255))).asType(.uint8)
            } else {
                decodedVideo = nil
                hdrOutput = nil
                frames = decodeWithTiling(
                    decoder: decoder,
                    latents: videoLatents,
                    spatialTileSizeInPixels: tiling.spatialTileSizeInPixels,
                    spatialOverlapInPixels: tiling.spatialTileOverlapInPixels,
                    temporalTileSizeInFrames: tiling.temporalTileSizeInFrames,
                    temporalOverlapInFrames: tiling.temporalTileOverlapInFrames,
                    spatialScale: 32,
                    temporalScale: 8
                )
            }
        } else {
            let fullDecoded = decoder.decode(sample: videoLatents, timestep: nil)
            decodedVideo = fullDecoded
            if let colorSpace = options.hdrColorSpace {
                let output = LTXHDRColorPipeline.decode(
                    fullDecoded,
                    transfer: options.hdrTransfer,
                    exrColorSpace: colorSpace
                )
                hdrOutput = output
                frames = (output.working * MLXArray(Float(255))).asType(.uint8)
            } else {
                hdrOutput = nil
                frames = postprocessDecodedVideo(fullDecoded)
            }
        }
        if options.hdrICLoRA?.highQuality == true {
            let indices = MLXArray(
                Array(stride(from: 0, to: frames.dim(0), by: 2)).map(Int32.init)
            )
            frames = MLX.take(frames, indices, axis: 0)
            hdrOutput = hdrOutput?.selectingFrames(indices)
            finalFrameCount = options.numFrames
        }
        if usesHDRICLoRA,
           generationWidth != options.width || generationHeight != options.height {
            frames = frames[0..., 0..<options.height, 0..<options.width, 0...]
            hdrOutput = hdrOutput?.cropped(width: options.width, height: options.height)
        }
        MLX.eval(frames)
        ltxTraceMemory("video-decode-ready")
        videoDecodeSeconds = ltxMonotonicSeconds() - videoDecodeStart

        if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
            let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
            let parent = base.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            let stem = base.lastPathComponent
            if let decodedVideo {
                try? MLX.save(array: decodedVideo, url: parent.appendingPathComponent("\(stem)_decoded.npy"))
            }
            try? MLX.save(array: frames, url: parent.appendingPathComponent("\(stem)_frames_postprocess.npy"))
            try? MLX.save(array: videoLatents, url: parent.appendingPathComponent("\(stem)_video_latents.npy"))
            try? MLX.save(array: audioLatents, url: parent.appendingPathComponent("\(stem)_audio_latents.npy"))
        }

        _ = decodedVideo
        var audioWaveform: MLXArray?
        var audioSampleRate: Int?
        if decodeAudio {
            let audioDecodeStart = ltxMonotonicSeconds()
            let activeAudioDecoder: LTXAudioDecoder
            let activeVocoder: LTXAudioVocoderBase
            if usesFullTwoStage {
                guard let loadedRoot else {
                    throw LTXUnifiedAVGeneratorError.generatorNotLoaded
                }
                if !usesReusableFullTwoStage {
                    self.transformer = nil
                    self.upsampler = nil
                    Memory.clearCache()
                }
                let audioWeightsURL = loadedForLTX25
                    ? LTX25Resources(rootURL: loadedRoot).audioVAEURL
                    : loadedRoot.appendingPathComponent("audio_vae.safetensors", isDirectory: false)
                let vocoderWeightsURL = loadedForLTX25
                    ? audioWeightsURL
                    : loadedRoot.appendingPathComponent("vocoder.safetensors", isDirectory: false)
                let sourceLayout: LTXTensorWeightLayout = loadedForLTX25 ? .pytorch : .mlx
                activeAudioDecoder = try loadLTXAudioDecoder(
                    weightsURL: audioWeightsURL,
                    sourceLayout: sourceLayout
                )
                activeVocoder = try loadLTXVocoder(
                    weightsURL: vocoderWeightsURL,
                    sourceLayout: loadedForLTX25 ? .pytorch : .mlx,
                    configurationRoot: loadedRoot,
                    usesPackedConfiguration: loadedForLTX25
                )
                if !usesReusableFullTwoStage {
                    self.audioDecoder = activeAudioDecoder
                    self.vocoder = activeVocoder
                }
            } else {
                guard let audioDecoder else {
                    throw LTXUnifiedAVGeneratorError.audioDecoderNotLoaded
                }
                guard let vocoder else {
                    throw LTXUnifiedAVGeneratorError.vocoderNotLoaded
                }
                activeAudioDecoder = audioDecoder
                activeVocoder = vocoder
            }
            let mel = activeAudioDecoder.decode(latents: audioLatents.asType(.float32))
            saveLTXAVDebugArray(mel, suffix: "audio_mel")
            let vocodedAudio = activeVocoder(mel)
            saveLTXAVDebugAudio(
                vocodedAudio,
                suffix: "audio_vocoded_raw",
                sampleRate: activeVocoder.outputSamplingRate
            )
            let waveform = matchLTXAudioWaveformDuration(
                vocodedAudio,
                videoFrames: finalFrameCount,
                fps: finalFPS,
                sampleRate: activeVocoder.outputSamplingRate
            )
            MLX.eval(waveform)
            audioDecodeSeconds = ltxMonotonicSeconds() - audioDecodeStart
            saveLTXAVDebugAudio(
                waveform,
                suffix: "audio_waveform_matched",
                sampleRate: activeVocoder.outputSamplingRate
            )
            audioWaveform = waveform
            audioSampleRate = activeVocoder.outputSamplingRate
        }

        try teaCacheController?.writeCalibrationReport()

        return LTXUnifiedGenerationOutput(
            frames: frames,
            hdrOutput: hdrOutput,
            videoLatents: videoLatents,
            audioLatents: audioLatents,
            audioWaveform: audioWaveform,
            audioSampleRate: audioSampleRate,
            generatedKeyframeLatents: generatedKeyframeLatents,
            generatedKeyframeIndices: generatedKeyframeLatents == nil ? [] : outputGeneratedIndices,
            playbackFPS: finalFPS,
            timings: LTXGenerationTimings(
                textEncodingSeconds: textEncodingSeconds,
                promptCacheHits: promptCacheHits,
                promptCacheMisses: promptCacheMisses,
                guidanceProjectionCacheBuildSeconds: guidanceProjectionCacheMetrics.buildSeconds,
                guidanceProjectionCacheBuilds: guidanceProjectionCacheMetrics.buildCount,
                guidanceProjectionCacheReuses: guidanceProjectionCacheMetrics.reuseCount,
                guidanceProjectionCacheFallbacks: guidanceProjectionCacheMetrics.fallbackCount,
                teaCacheDecisionSeconds: teaCacheController?.metrics.decisionSeconds ?? 0,
                teaCacheComputedBlockStacks: teaCacheController?.metrics.computedBlockStacks ?? 0,
                teaCacheReusedBlockStacks: teaCacheController?.metrics.reusedBlockStacks ?? 0,
                preparationSeconds: preparationSeconds,
                stage1DenoiseSeconds: stage1DenoiseSeconds,
                loraFusionSeconds: loraFusionSeconds,
                upsampleSeconds: upsampleSeconds,
                stage2DenoiseSeconds: stage2DenoiseSeconds,
                videoDecodeSeconds: videoDecodeSeconds,
                audioDecodeSeconds: audioDecodeSeconds,
                totalSeconds: ltxMonotonicSeconds() - totalStart
            )
        )
    }
}
