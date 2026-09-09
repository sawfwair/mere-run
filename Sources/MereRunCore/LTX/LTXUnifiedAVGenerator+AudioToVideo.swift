import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

extension LTXUnifiedAVGenerator {
    public func generateAudioToVideo(
        options: LTXAudioToVideoGenerationOptions
    ) async throws -> LTXAudioToVideoGenerationResult {
        let totalStart = ltxMonotonicSeconds()
        var preparationSeconds = 0.0
        var textEncodingSeconds = 0.0
        var textEncoderReloadSeconds = 0.0
        var stage1DenoiseSeconds = 0.0
        var loraFusionSeconds = 0.0
        var upsampleSeconds = 0.0
        var stage2DenoiseSeconds = 0.0
        var videoDecodeSeconds = 0.0
        let prompt = options.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let negativePrompt = options.negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !negativePrompt.isEmpty else {
            throw LTXUnifiedAVGeneratorError.emptyPrompt
        }
        guard options.width > 0, options.height > 0,
              options.width % 64 == 0, options.height % 64 == 0 else {
            throw LTXUnifiedAVGeneratorError.invalidResolution(width: options.width, height: options.height)
        }
        guard options.numFrames >= 9, options.numFrames % 8 == 1 else {
            throw LTXUnifiedAVGeneratorError.invalidFrameCount(options.numFrames)
        }
        guard options.fps.isFinite, options.fps > 0 else {
            throw LTXUnifiedAVGeneratorError.invalidFrameRate(options.fps)
        }
        guard options.inferenceSteps > 0 else {
            throw LTXUnifiedAVGeneratorError.invalidInferenceSteps(options.inferenceSteps)
        }
        guard options.audioStartTime.isFinite, options.audioStartTime >= 0 else {
            throw LTXUnifiedAVGeneratorError.invalidAudioStartTime(options.audioStartTime)
        }
        if let audioMaxDuration = options.audioMaxDuration,
           !audioMaxDuration.isFinite || audioMaxDuration <= 0 {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "audioMaxDuration must be finite and positive."
            )
        }
        guard options.imageFrameIndex >= 0 else {
            throw LTXUnifiedAVGeneratorError.invalidImageFrameIndex(options.imageFrameIndex)
        }
        guard options.imageStrength >= 0, options.imageStrength <= 1 else {
            throw LTXUnifiedAVGeneratorError.invalidImageStrength(options.imageStrength)
        }
        guard options.endImageStrength >= 0, options.endImageStrength <= 1 else {
            throw LTXUnifiedAVGeneratorError.invalidImageStrength(options.endImageStrength)
        }
        let imageConditionings = ltx25ImageConditionings(options: options)
        let hasEXRInput = imageConditionings.contains { MediaHDRImageIO.isEXR($0.imageURL) }
        if hasEXRInput, options.hdrColorSpace == nil {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "EXR conditioning requires an explicit HDR color space."
            )
        }
        if options.hdrColorSpace != nil, !loadedForLTX25 {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "HDR audio-to-video requires an official LTX-2.5 checkpoint."
            )
        }
        guard options.generatedKeyframeCount >= 0,
              options.generatedKeyframeCount == 0 || options.generatedKeyframeIndices.isEmpty else {
            throw LTXUnifiedAVGeneratorError.incompatibleLTX25Workflows(
                "Use a generated-keyframe count or explicit indices, not both."
            )
        }
        let generatedIndices = options.generatedKeyframeCount > 0
            ? try ltxEvenlySpacedGeneratedKeyframePositions(
                count: options.generatedKeyframeCount,
                numFrames: options.numFrames
            )
            : options.generatedKeyframeIndices
        let requestsLTX25Conditioning = !options.imageConditionings.isEmpty
            || !generatedIndices.isEmpty
        guard loadedForLTX25 || !requestsLTX25Conditioning else {
            throw LTXUnifiedAVGeneratorError.ltx25ConditioningRequiresLTX25
        }
        if loadedForLTX25 {
            for input in imageConditionings {
                guard input.pixelFrameIndex >= 0, input.pixelFrameIndex < options.numFrames else {
                    throw LTXUnifiedAVGeneratorError.invalidImageFrameIndex(input.pixelFrameIndex)
                }
                guard input.strength >= 0, input.strength <= 1 else {
                    throw LTXUnifiedAVGeneratorError.invalidImageStrength(input.strength)
                }
                guard FileManager.default.fileExists(atPath: input.imageURL.path) else {
                    throw LTXUnifiedAVGeneratorError.imageNotFound(input.imageURL)
                }
            }
        }
        let generatedIndicesAreOrdered = zip(
            generatedIndices,
            generatedIndices.dropFirst()
        ).allSatisfy(<)
        guard generatedIndicesAreOrdered,
              generatedIndices.allSatisfy({ $0 >= 0 && $0 < options.numFrames }) else {
            throw LTXUnifiedAVGeneratorError.invalidGeneratedKeyframes(generatedIndices)
        }
        let usesLTX25TokenState = loadedForLTX25
            && (!imageConditionings.isEmpty || !generatedIndices.isEmpty)
        guard FileManager.default.fileExists(atPath: options.audioURL.path) else {
            throw LTXUnifiedAVGeneratorError.audioSourceNotFound(options.audioURL)
        }
        let usesReusableFullTwoStage = loadedForReusableFullTwoStage
        if usesReusableFullTwoStage {
            let reloadStart = ltxMonotonicSeconds()
            try await loadTextEncoderIfNeeded()
            textEncoderReloadSeconds = ltxMonotonicSeconds() - reloadStart
        }
        guard loadedForAudioToVideo,
              let textEncoder,
              let transformer,
              let decoder,
              let upsampler,
              let audioVAEWeightsURL,
              let distilledLoRAURL else {
            throw LTXUnifiedAVGeneratorError.audioToVideoGeneratorNotLoaded
        }
        if let transformerV2 = transformer as? LTXUnifiedAVTransformerV2 {
            transformerV2.execution = options.transformerExecution
        }
        guard !twoStageGenerationConsumed else {
            throw LTXUnifiedAVGeneratorError.audioToVideoRequiresReload
        }
        if usesReusableFullTwoStage, runtimeLoRAAdapter == nil {
            throw LTXUnifiedAVGeneratorError.audioToVideoGeneratorNotLoaded
        }
        let parityIO = LTXAudioToVideoParityIO()

        let inputPreparationStart = ltxMonotonicSeconds()
        let duration = Double(options.numFrames) / Double(options.fps)
        let audioDecodeDuration = options.audioMaxDuration ?? duration
        let audioMetadata = try MediaAudioIO.probe(options.audioURL)
        try validateLTXAudioSegment(
            metadata: audioMetadata,
            startTime: options.audioStartTime,
            duration: audioDecodeDuration
        )

        let sourceAudio = try decodeExactStereoAudioSegment(
            url: options.audioURL,
            startTime: options.audioStartTime,
            duration: audioDecodeDuration,
            sampleRate: audioMetadata.sampleRate
        )
        let conditioningAudio = try decodeExactStereoAudioSegment(
            url: options.audioURL,
            startTime: options.audioStartTime,
            duration: audioDecodeDuration,
            sampleRate: LTXAudioMelProcessor.sampleRate
        )
        preparationSeconds += ltxMonotonicSeconds() - inputPreparationStart

        twoStageGenerationConsumed = true
        defer {
            if usesReusableFullTwoStage {
                runtimeLoRAAdapter?.setActive(false)
                twoStageGenerationConsumed = false
            }
        }
        let textEncodingStart = ltxMonotonicSeconds()
        let positiveEncoding = try await textEncoder.encode(
            prompt: prompt,
            maxLength: options.maxTextLength
        )
        let negativeEncoding = try await textEncoder.encode(
            prompt: negativePrompt,
            maxLength: options.maxTextLength
        )
        let positiveVideoContext = positiveEncoding.videoEmbeddings
        let negativeVideoContext = negativeEncoding.videoEmbeddings
        guard let positiveAudioContext = positiveEncoding.audioEmbeddings else {
            throw LTXUnifiedAVGeneratorError.audioEmbeddingsMissing
        }
        MLX.eval(positiveVideoContext, negativeVideoContext, positiveAudioContext)
        try parityIO.save(positiveVideoContext, suffix: "a2vid_positive_video_context")
        try parityIO.save(negativeVideoContext, suffix: "a2vid_negative_video_context")
        try parityIO.save(positiveAudioContext, suffix: "a2vid_positive_audio_context")
        await textEncoder.unload()
        self.textEncoder = nil
        Memory.clearCache()
        textEncodingSeconds = textEncoderReloadSeconds + ltxMonotonicSeconds() - textEncodingStart

        let latentPreparationStart = ltxMonotonicSeconds()
        let audioFrameCount = computeAudioLatentFrameCount(
            videoFrames: options.numFrames,
            fps: options.fps
        )
        let spectrogram = LTXAudioMelProcessor().extract(
            channels: planarAudioChannels(conditioningAudio)
        )
        try parityIO.save(spectrogram, suffix: "a2vid_audio_mel")
        var audioLatents = try encodeLTX23AudioLatents(
            spectrogram: spectrogram,
            requiredFrameCount: audioFrameCount,
            weightsURL: audioVAEWeightsURL,
            dtype: loadedDType,
            sourceLayout: loadedForLTX25 ? .pytorch : .mlx
        )
        audioLatents = audioLatents.asType(positiveVideoContext.dtype)
        MLX.eval(audioLatents)
        Memory.clearCache()

        let latentFrames = 1 + ((options.numFrames - 1) / 8)
        let stage1Height = options.height / 2 / 32
        let stage1Width = options.width / 2 / 32
        let stage2Height = options.height / 32
        let stage2Width = options.width / 32
        let modelDType = positiveVideoContext.dtype
        try parityIO.save(audioLatents, suffix: "a2vid_audio_latents")

        MLXRandom.seed(UInt64(bitPattern: Int64(options.seed)))
        var stage1ConditioningState: LTXLatentConditioningState?
        var stage2ConditioningState: LTXLatentConditioningState?
        var stage2ConditioningLatent: MLXArray?
        var stage2EndConditioningLatent: MLXArray?
        var stage1TokenState: LTX25VideoTokenState?
        var stage2TokenState: LTX25VideoTokenState?
        var generatedKeyframeLatents: MLXArray?
        var stage2LTX25ImageLatents: [MLXArray] = []
        var videoLatents: MLXArray

        let baseStage1VideoPositions = createPositionGrid(
            batchSize: 1,
            numFrames: latentFrames,
            height: stage1Height,
            width: stage1Width,
            temporalScale: 8,
            spatialScale: 32,
            fps: Float(options.fps),
            causalFix: true
        )
        if usesLTX25TokenState {
            if !imageConditionings.isEmpty {
                try loadEncoderIfNeeded()
            }
            videoLatents = MLX.zeros(
                [1, 128, latentFrames, stage1Height, stage1Width],
                dtype: modelDType
            )
            var state = try makeConditionedLTX25VideoTokenState(
                initialLatent: videoLatents,
                positions: baseStage1VideoPositions,
                imageConditionings: imageConditionings,
                generatedKeyframeIndices: generatedIndices,
                initialGeneratedKeyframes: nil,
                encoder: encoder,
                pixelWidth: options.width / 2,
                pixelHeight: options.height / 2,
                fps: options.fps,
                hdrColorSpace: options.hdrColorSpace
            )
            let noise = try parityIO.resolveNoise(
                stage: .stage1,
                generated: MLXRandom.normal(state.latent.shape).asType(modelDType)
            )
            state.addNoise(noise, scale: 1)
            MLX.eval(state.latent)
            stage1TokenState = state
            if !imageConditionings.isEmpty {
                guard let imageEncoder = encoder else {
                    throw LTXUnifiedAVGeneratorError.encoderNotLoaded
                }
                for input in imageConditionings {
                    let image = try loadImageForEncoding(
                        url: input.imageURL,
                        width: options.width,
                        height: options.height,
                        dtype: modelDType,
                        hdrColorSpace: options.hdrColorSpace,
                        crf: input.crf ?? 18
                    )
                    let encoded = imageEncoder.encode(image: image)
                    MLX.eval(encoded)
                    stage2LTX25ImageLatents.append(encoded)
                }
            }
            encoder = nil
            Memory.clearCache()
        } else if let sourceImageURL = options.sourceImageURL {
            guard FileManager.default.fileExists(atPath: sourceImageURL.path) else {
                throw LTXUnifiedAVGeneratorError.imageNotFound(sourceImageURL)
            }
            if let endImageURL = options.endImageURL,
               !FileManager.default.fileExists(atPath: endImageURL.path) {
                throw LTXUnifiedAVGeneratorError.imageNotFound(endImageURL)
            }
            guard options.imageFrameIndex < latentFrames else {
                throw LTXUnifiedAVGeneratorError.invalidImageFrameIndex(options.imageFrameIndex)
            }

            try loadEncoderIfNeeded()
            do {
                guard let imageEncoder = encoder else {
                    throw LTXUnifiedAVGeneratorError.encoderNotLoaded
                }
                let stage1Image = try loadImageForEncoding(
                    url: sourceImageURL,
                    width: options.width / 2,
                    height: options.height / 2,
                    dtype: modelDType
                )
                let stage1ImageLatent = imageEncoder.encode(image: stage1Image)
                let stage2Image = try loadImageForEncoding(
                    url: sourceImageURL,
                    width: options.width,
                    height: options.height,
                    dtype: modelDType
                )
                let stage2ImageLatent = imageEncoder.encode(image: stage2Image)
                stage2ConditioningLatent = stage2ImageLatent

                var stage1EndImageLatent: MLXArray?
                if let endImageURL = options.endImageURL {
                    let stage1EndImage = try loadImageForEncoding(
                        url: endImageURL,
                        width: options.width / 2,
                        height: options.height / 2,
                        dtype: modelDType
                    )
                    stage1EndImageLatent = imageEncoder.encode(image: stage1EndImage)
                    let stage2EndImage = try loadImageForEncoding(
                        url: endImageURL,
                        width: options.width,
                        height: options.height,
                        dtype: modelDType
                    )
                    stage2EndConditioningLatent = imageEncoder.encode(image: stage2EndImage)
                }

                var state = applyLatentConditioning(
                    baseLatent: MLX.zeros(
                        [1, 128, latentFrames, stage1Height, stage1Width],
                        dtype: modelDType
                    ),
                    conditionedLatent: stage1ImageLatent,
                    frameIndex: options.imageFrameIndex,
                    strength: options.imageStrength,
                    endConditionedLatent: stage1EndImageLatent,
                    endFrameIndex: -1,
                    endStrength: options.endImageStrength
                )
                let noise = try parityIO.resolveNoise(
                    stage: .stage1,
                    generated: MLXRandom.normal(state.latent.shape).asType(modelDType)
                )
                try parityIO.save(noise, suffix: "a2vid_stage1_noise")
                let sigma = MLXArray(1).asType(modelDType)
                let scaledMask = state.denoiseMask * sigma
                state.latent = noise * scaledMask + state.latent * (MLXArray(1).asType(modelDType) - scaledMask)
                videoLatents = state.latent
                stage1ConditioningState = state
                MLX.eval(videoLatents, stage2ImageLatent)
                if let stage2EndConditioningLatent {
                    MLX.eval(stage2EndConditioningLatent)
                }
            }
            encoder = nil
            Memory.clearCache()
        } else {
            videoLatents = try parityIO.resolveNoise(
                stage: .stage1,
                generated: MLXRandom.normal(
                    [1, 128, latentFrames, stage1Height, stage1Width]
                ).asType(modelDType)
            )
            try parityIO.save(videoLatents, suffix: "a2vid_stage1_noise")
            MLX.eval(videoLatents)
        }

        let audioPositions = createAudioPositionGrid(
            batchSize: 1,
            audioFrames: audioFrameCount
        )
        let audioRope = precomputeSplitRope(
            positions: audioPositions,
            dim: 2_048,
            theta: 10_000,
            maxPos: [20],
            numHeads: 32
        )
        let stage1Ropes = stage1TokenState.map {
            makeLTXAudioToVideoVideoRopes(positions: $0.positions)
        } ?? makeLTXAudioToVideoVideoRopes(
            latentFrames: latentFrames,
            height: stage1Height,
            width: stage1Width,
            fps: options.fps
        )
        let stage1Sigmas = LTX2DiffusionScheduler.sigmas(steps: options.inferenceSteps)
        preparationSeconds += ltxMonotonicSeconds() - latentPreparationStart
        let stage1DenoiseStart = ltxMonotonicSeconds()
        if let tokenState = stage1TokenState {
            let result = denoiseFrozenLTX25AudioVideoTokenLoop(
                videoState: tokenState,
                audioLatents: audioLatents,
                videoRope: stage1Ropes.selfAttention,
                audioRope: audioRope,
                videoCrossRope: stage1Ropes.crossAttention,
                audioCrossRope: audioRope,
                positiveVideoContext: positiveVideoContext,
                negativeVideoContext: negativeVideoContext,
                audioContext: positiveAudioContext,
                transformer: transformer,
                sigmas: stage1Sigmas,
                guidance: options.guidance
            )
            stage1TokenState = result
            videoLatents = result.mainLatent()
            generatedKeyframeLatents = result.generatedKeyframes()
        } else {
            videoLatents = try denoiseFrozenAudioVideoLoop(
                videoLatents: videoLatents,
                audioLatents: audioLatents,
                videoRope: stage1Ropes.selfAttention,
                audioRope: audioRope,
                videoCrossRope: stage1Ropes.crossAttention,
                audioCrossRope: audioRope,
                positiveVideoContext: positiveVideoContext,
                negativeVideoContext: negativeVideoContext,
                audioContext: positiveAudioContext,
                transformer: transformer,
                sigmas: stage1Sigmas,
                videoConditioning: stage1ConditioningState,
                guidance: options.guidance,
                debugLabel: "a2vid_stage1"
            ).video
        }
        MLX.eval(videoLatents)
        stage1DenoiseSeconds = ltxMonotonicSeconds() - stage1DenoiseStart
        try parityIO.save(videoLatents, suffix: "a2vid_stage1_output")

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
        try parityIO.save(videoLatents, suffix: "a2vid_upsampled_latents")

        let stage2Sigma = STAGE2Sigmas[0]
        let baseStage2VideoPositions = createPositionGrid(
            batchSize: 1,
            numFrames: latentFrames,
            height: stage2Height,
            width: stage2Width,
            temporalScale: 8,
            spatialScale: 32,
            fps: Float(options.fps),
            causalFix: true
        )
        if usesLTX25TokenState {
            var state = LTX25VideoTokenState(
                initialLatent: videoLatents,
                positions: baseStage2VideoPositions
            )
            for (input, encoded) in zip(imageConditionings, stage2LTX25ImageLatents) {
                state.applyImageLatent(
                    encoded,
                    pixelFrameIndex: input.pixelFrameIndex,
                    strength: input.strength,
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
            let noise = try parityIO.resolveNoise(
                stage: .stage2,
                generated: MLXRandom.normal(state.latent.shape).asType(modelDType)
            )
            state.addNoise(noise, scale: stage2Sigma)
            MLX.eval(state.latent)
            stage2TokenState = state
            videoLatents = state.mainLatent()
        } else if let conditionedLatent = stage2ConditioningLatent {
            var state = applyLatentConditioning(
                baseLatent: videoLatents,
                conditionedLatent: conditionedLatent,
                frameIndex: options.imageFrameIndex,
                strength: options.imageStrength,
                endConditionedLatent: stage2EndConditioningLatent,
                endFrameIndex: -1,
                endStrength: options.endImageStrength
            )
            let noise = try parityIO.resolveNoise(
                stage: .stage2,
                generated: MLXRandom.normal(videoLatents.shape).asType(modelDType)
            )
            try parityIO.save(noise, suffix: "a2vid_stage2_noise")
            let scaledMask = state.denoiseMask * MLXArray(stage2Sigma).asType(modelDType)
            state.latent = noise * scaledMask + state.latent * (MLXArray(1).asType(modelDType) - scaledMask)
            videoLatents = state.latent
            stage2ConditioningState = state
        } else {
            let noise = try parityIO.resolveNoise(
                stage: .stage2,
                generated: MLXRandom.normal(videoLatents.shape).asType(modelDType)
            )
            try parityIO.save(noise, suffix: "a2vid_stage2_noise")
            videoLatents = MLXArray(stage2Sigma).asType(modelDType) * noise
                + MLXArray(1 - stage2Sigma).asType(modelDType) * videoLatents
        }
        MLX.eval(videoLatents)
        try parityIO.save(videoLatents, suffix: "a2vid_stage2_input")

        let loraFusionStart = ltxMonotonicSeconds()
        if usesReusableFullTwoStage {
            runtimeLoRAAdapter?.setActive(true)
        } else {
            try LTXStreamingLoRAFuser.fuse(
                url: distilledLoRAURL,
                into: transformer,
                debugOutputPrefix: parityIO.outputPrefix
            )
        }
        loraFusionSeconds = ltxMonotonicSeconds() - loraFusionStart

        let stage2Ropes = stage2TokenState.map {
            makeLTXAudioToVideoVideoRopes(positions: $0.positions)
        } ?? makeLTXAudioToVideoVideoRopes(
            latentFrames: latentFrames,
            height: stage2Height,
            width: stage2Width,
            fps: options.fps
        )
        let stage2DenoiseStart = ltxMonotonicSeconds()
        if let tokenState = stage2TokenState {
            let result = denoiseFrozenLTX25AudioVideoTokenLoop(
                videoState: tokenState,
                audioLatents: audioLatents,
                videoRope: stage2Ropes.selfAttention,
                audioRope: audioRope,
                videoCrossRope: stage2Ropes.crossAttention,
                audioCrossRope: audioRope,
                positiveVideoContext: positiveVideoContext,
                negativeVideoContext: nil,
                audioContext: positiveAudioContext,
                transformer: transformer,
                sigmas: STAGE2Sigmas,
                guidance: nil
            )
            stage2TokenState = result
            videoLatents = result.mainLatent()
            generatedKeyframeLatents = result.generatedKeyframes()
        } else {
            videoLatents = try denoiseFrozenAudioVideoLoop(
                videoLatents: videoLatents,
                audioLatents: audioLatents,
                videoRope: stage2Ropes.selfAttention,
                audioRope: audioRope,
                videoCrossRope: stage2Ropes.crossAttention,
                audioCrossRope: audioRope,
                positiveVideoContext: positiveVideoContext,
                negativeVideoContext: nil,
                audioContext: positiveAudioContext,
                transformer: transformer,
                sigmas: STAGE2Sigmas,
                videoConditioning: stage2ConditioningState,
                guidance: nil,
                debugLabel: "a2vid_stage2"
            ).video
        }
        MLX.eval(videoLatents)
        if usesReusableFullTwoStage {
            runtimeLoRAAdapter?.setActive(false)
        }
        stage2DenoiseSeconds = ltxMonotonicSeconds() - stage2DenoiseStart
        try parityIO.save(videoLatents, suffix: "a2vid_stage2_output")

        let videoDecodeStart = ltxMonotonicSeconds()
        let frames: MLXArray
        let hdrOutput: LTXHDROutputFrames?
        if let diffusionDecoder {
            let decoded = try diffusionDecoder.decode(sample: videoLatents, seed: options.seed)
            if let colorSpace = options.hdrColorSpace {
                let output = LTXHDRColorPipeline.decode(
                    decoded,
                    transfer: options.hdrTransfer,
                    exrColorSpace: colorSpace
                )
                hdrOutput = output
                frames = (output.working * MLXArray(Float(255))).asType(.uint8)
            } else {
                hdrOutput = nil
                frames = postprocessDecodedVideo(decoded)
            }
        } else if let tiling = selectDecodeTilingConfig(
            width: options.width,
            height: options.height,
            numFrames: options.numFrames,
            fps: options.fps
        ) {
            if let colorSpace = options.hdrColorSpace {
                let decoded = decodeWithTilingRaw(
                    decoder: decoder,
                    latents: videoLatents,
                    spatialTileSizeInPixels: tiling.spatialTileSizeInPixels,
                    spatialOverlapInPixels: tiling.spatialTileOverlapInPixels,
                    temporalTileSizeInFrames: tiling.temporalTileSizeInFrames,
                    temporalOverlapInFrames: tiling.temporalTileOverlapInFrames,
                    spatialScale: 32,
                    temporalScale: 8
                )
                let output = LTXHDRColorPipeline.decode(
                    decoded,
                    transfer: options.hdrTransfer,
                    exrColorSpace: colorSpace
                )
                hdrOutput = output
                frames = (output.working * MLXArray(Float(255))).asType(.uint8)
            } else {
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
            let decoded = decoder.decode(sample: videoLatents, timestep: nil)
            if let colorSpace = options.hdrColorSpace {
                let output = LTXHDRColorPipeline.decode(
                    decoded,
                    transfer: options.hdrTransfer,
                    exrColorSpace: colorSpace
                )
                hdrOutput = output
                frames = (output.working * MLXArray(Float(255))).asType(.uint8)
            } else {
                hdrOutput = nil
                frames = postprocessDecodedVideo(decoded)
            }
        }
        MLX.eval(frames)
        ltxTraceMemory("video-decode-ready")
        videoDecodeSeconds = ltxMonotonicSeconds() - videoDecodeStart

        return LTXAudioToVideoGenerationResult(
            frames: frames,
            hdrOutput: hdrOutput,
            videoLatents: videoLatents,
            audioLatents: audioLatents,
            sourceAudio: sourceAudio,
            generatedKeyframeLatents: generatedKeyframeLatents,
            generatedKeyframeIndices: generatedKeyframeLatents == nil ? [] : generatedIndices,
            timings: LTXGenerationTimings(
                textEncodingSeconds: textEncodingSeconds,
                preparationSeconds: preparationSeconds,
                stage1DenoiseSeconds: stage1DenoiseSeconds,
                loraFusionSeconds: loraFusionSeconds,
                upsampleSeconds: upsampleSeconds,
                stage2DenoiseSeconds: stage2DenoiseSeconds,
                videoDecodeSeconds: videoDecodeSeconds,
                totalSeconds: ltxMonotonicSeconds() - totalStart
            )
        )
    }
}
