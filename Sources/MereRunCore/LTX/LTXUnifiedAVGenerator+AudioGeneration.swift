import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

extension LTXUnifiedAVGenerator {
    public func generateTextToAudio(
        options: LTXTextToAudioGenerationOptions
    ) async throws -> LTXTextToAudioGenerationResult {
        let totalStart = ltxMonotonicSeconds()
        let prompt = options.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let negativePrompt = options.negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !negativePrompt.isEmpty else {
            throw LTXUnifiedAVGeneratorError.emptyPrompt
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
        guard let textEncoder,
              let audioOnlyTransformer,
              let audioDecoder,
              let vocoder else {
            throw LTXUnifiedAVGeneratorError.textToAudioGeneratorNotLoaded
        }
        let sigmas = try validatedLTXSigmaSchedule(
            options.sigmas ?? LTX2DiffusionScheduler.sigmas(steps: options.inferenceSteps)
        )

        if runtimeAudioLoRAAdapters.isEmpty, !options.loras.isEmpty {
            runtimeAudioLoRAAdapters = try options.loras.map { configuration in
                guard FileManager.default.fileExists(atPath: configuration.url.path) else {
                    throw LTXUnifiedAVGeneratorError.loraMissing(configuration.url)
                }
                return try LTXRuntimeLoRAAdapter.install(
                    url: configuration.url,
                    into: audioOnlyTransformer,
                    strength: configuration.strength,
                    expectedPairCount: nil,
                    ignoreMissingTargets: true
                )
            }
        }
        runtimeAudioLoRAAdapters.forEach { $0.setActive(true) }
        defer { runtimeAudioLoRAAdapters.forEach { $0.setActive(false) } }

        let textStart = ltxMonotonicSeconds()
        let positiveEncoding = try await textEncoder.encode(
            prompt: prompt,
            maxLength: options.maxTextLength
        )
        let negativeEncoding = try await textEncoder.encode(
            prompt: negativePrompt,
            maxLength: options.maxTextLength
        )
        guard let positiveAudioContext = positiveEncoding.audioEmbeddings,
              let negativeAudioContext = negativeEncoding.audioEmbeddings else {
            throw LTXUnifiedAVGeneratorError.audioEmbeddingsMissing
        }
        MLX.eval(positiveAudioContext, negativeAudioContext)
        await textEncoder.unload()
        self.textEncoder = nil
        Memory.clearCache()
        let textSeconds = ltxMonotonicSeconds() - textStart

        let preparationStart = ltxMonotonicSeconds()
        let audioFrameCount = computeAudioLatentFrameCount(
            videoFrames: options.numFrames,
            fps: options.fps
        )
        MLXRandom.seed(UInt64(bitPattern: Int64(options.seed)))
        let initialAudio = MLXRandom.normal(
            [1, LTXAudioLatentChannels, audioFrameCount, LTXAudioLatentMelBins]
        ).asType(positiveAudioContext.dtype)
        let positions = createAudioPositionGrid(batchSize: 1, audioFrames: audioFrameCount)
        let rope = precomputeSplitRope(
            positions: positions,
            dim: 2_048,
            theta: 10_000,
            maxPos: [20],
            numHeads: 32
        )
        MLX.eval(initialAudio, rope.cos, rope.sin)
        let preparationSeconds = ltxMonotonicSeconds() - preparationStart

        let denoiseStart = ltxMonotonicSeconds()
        let audioLatents = denoiseLTX25AudioOnlyLoop(
            audioLatents: initialAudio,
            audioRope: rope,
            positiveContext: positiveAudioContext,
            negativeContext: negativeAudioContext,
            transformer: audioOnlyTransformer,
            sigmas: sigmas,
            guidance: options.guidance
        )
        MLX.eval(audioLatents)
        let denoiseSeconds = ltxMonotonicSeconds() - denoiseStart

        let decodeStart = ltxMonotonicSeconds()
        let mel = audioDecoder.decode(latents: audioLatents.asType(.float32))
        let vocoded = vocoder(mel)
        let waveform = matchLTXAudioWaveformDuration(
            vocoded,
            videoFrames: options.numFrames,
            fps: options.fps,
            sampleRate: vocoder.outputSamplingRate
        )
        MLX.eval(waveform)
        let decodeSeconds = ltxMonotonicSeconds() - decodeStart
        return LTXTextToAudioGenerationResult(
            audioLatents: audioLatents,
            audioWaveform: waveform,
            audioSampleRate: vocoder.outputSamplingRate,
            timings: LTXGenerationTimings(
                textEncodingSeconds: textSeconds,
                preparationSeconds: preparationSeconds,
                stage1DenoiseSeconds: denoiseSeconds,
                audioDecodeSeconds: decodeSeconds,
                totalSeconds: ltxMonotonicSeconds() - totalStart
            )
        )
    }

    public func predictFrameCount(
        prompt: String,
        frameRate: Double,
        range: LTX25AutoDuration = LTX25AutoDuration(),
        conditioning: LTX25DurationConditioning = .audioVideo,
        maxTextLength: Int = 1_024
    ) async throws -> Int {
        guard loadedForLTX25, let loadedRoot else {
            throw LTXUnifiedAVGeneratorError.durationPredictionRequiresLTX25(self.loadedRoot)
        }
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else {
            throw LTXUnifiedAVGeneratorError.emptyPrompt
        }
        let cacheKey = LTXPromptEmbeddingCacheKey(
            prompt: trimmedPrompt,
            maxLength: maxTextLength
        )
        if !promptEmbeddingCache.contains(cacheKey), textEncoder == nil {
            try await loadTextEncoderIfNeeded()
        }
        let encoding = try await cachedPromptEmbeddings(
            prompt: trimmedPrompt,
            maxLength: maxTextLength
        ).embeddings
        let head = try LTX25DurationHead.load(
            weightsURL: LTX25Resources(rootURL: loadedRoot).durationHeadURL,
            dtype: loadedDType
        )
        let videoTokens: MLXArray? = conditioning == .audioVideo
            ? encoding.video
            : nil
        let audioTokens = encoding.audio
        if conditioning == .audioOnly, audioTokens == nil {
            throw LTXUnifiedAVGeneratorError.audioEmbeddingsMissing
        }
        let frameCount = try head.predictFrameCount(
            videoTokens: videoTokens,
            audioTokens: audioTokens,
            frameRate: frameRate,
            range: range
        )
        if let textEncoder {
            await textEncoder.unload()
            self.textEncoder = nil
        }
        Memory.clearCache()
        ltxTraceMemory("duration-context-ready")
        return frameCount
    }
}
