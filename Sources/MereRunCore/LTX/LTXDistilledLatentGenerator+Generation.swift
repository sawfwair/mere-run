import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

extension LTXDistilledLatentGenerator {
    public func generate(
        options: LTXDistilledLatentGenerationOptions
    ) async throws -> LTXDistilledLatentGenerationResult {
        let prompt = options.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw LTXDistilledLatentGeneratorError.emptyPrompt
        }

        guard options.width % 64 == 0, options.height % 64 == 0 else {
            throw LTXDistilledLatentGeneratorError.invalidResolution(width: options.width, height: options.height)
        }
        guard options.numFrames >= 9, options.numFrames % 8 == 1 else {
            throw LTXDistilledLatentGeneratorError.invalidFrameCount(options.numFrames)
        }
        guard options.imageFrameIndex >= 0 else {
            throw LTXDistilledLatentGeneratorError.invalidImageFrameIndex(options.imageFrameIndex)
        }
        guard options.imageStrength >= 0, options.imageStrength <= 1 else {
            throw LTXDistilledLatentGeneratorError.invalidImageStrength(options.imageStrength)
        }

        guard let textEncoder, let transformer else {
            throw LTXDistilledLatentGeneratorError.generatorNotLoaded
        }
        guard let decoder else {
            throw LTXDistilledLatentGeneratorError.decoderNotLoaded
        }
        guard let upsampler else {
            throw LTXDistilledLatentGeneratorError.upsamplerNotLoaded
        }

        let encoding = try await textEncoder.encode(prompt: prompt, maxLength: options.maxTextLength)
        let context = encoding.videoEmbeddings
        let debugDenoise = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_DENOISE"] == "1"
        if debugDenoise {
            print("[LTX] context shape=\(context.shape) \(tensorStatsString(context))")
        }

        let latentFrames = 1 + ((options.numFrames - 1) / 8)
        let stage1H = options.height / 2 / 32
        let stage1W = options.width / 2 / 32
        let stage2H = options.height / 32
        let stage2W = options.width / 32

        MLXRandom.seed(UInt64(bitPattern: Int64(options.seed)))

        let modelDType = context.dtype
        let isImageToVideo = options.sourceImageURL != nil
        var stage1ConditioningState: LTXLatentConditioningState?
        var stage2ConditioningState: LTXLatentConditioningState?
        var stage2ConditioningLatent: MLXArray?
        var stage2EndConditioningLatent: MLXArray?

        var latents: MLXArray
        if isImageToVideo {
            let sourceImageURL = options.sourceImageURL!
            guard FileManager.default.fileExists(atPath: sourceImageURL.path) else {
                throw LTXDistilledLatentGeneratorError.imageNotFound(sourceImageURL)
            }
            try loadEncoderIfNeeded()
            guard let encoder else {
                throw LTXDistilledLatentGeneratorError.encoderNotLoaded
            }
            if options.imageFrameIndex >= latentFrames {
                throw LTXDistilledLatentGeneratorError.invalidImageFrameIndex(options.imageFrameIndex)
            }

            let stage1Image = try loadImageForEncoding(
                url: sourceImageURL,
                width: options.width / 2,
                height: options.height / 2,
                dtype: modelDType
            )
            let stage1ImageLatent = encoder.encode(image: stage1Image)

            let stage2Image = try loadImageForEncoding(
                url: sourceImageURL,
                width: options.width,
                height: options.height,
                dtype: modelDType
            )
            let stage2ImageLatent = encoder.encode(image: stage2Image)
            stage2ConditioningLatent = stage2ImageLatent

            // Optional end keyframe -> conditions the tail latent frame so the clip
            // interpolates a directed start->end motion.
            var stage1EndImageLatent: MLXArray?
            if let endImageURL = options.endImageURL {
                let stage1EndImage = try loadImageForEncoding(
                    url: endImageURL,
                    width: options.width / 2,
                    height: options.height / 2,
                    dtype: modelDType
                )
                stage1EndImageLatent = encoder.encode(image: stage1EndImage)
                let stage2EndImage = try loadImageForEncoding(
                    url: endImageURL,
                    width: options.width,
                    height: options.height,
                    dtype: modelDType
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
            let stage1Sigma = MLXArray(STAGE1Sigmas[0]).asType(modelDType)
            let one = MLXArray(1.0).asType(modelDType)
            let scaledMask = state1.denoiseMask * stage1Sigma
            state1.latent = stage1Noise * scaledMask + state1.latent * (one - scaledMask)
            latents = state1.latent
            MLX.eval(latents)
            stage1ConditioningState = state1

        } else {
            latents = MLXRandom.normal([1, 128, latentFrames, stage1H, stage1W]).asType(modelDType)
            MLX.eval(latents)
        }

        let stage1Positions = createPositionGrid(
            batchSize: 1,
            numFrames: latentFrames,
            height: stage1H,
            width: stage1W,
            temporalScale: 8,
            spatialScale: 32,
            fps: Float(max(1, options.fps)),
            causalFix: true
        )
        let stage1Rope = precomputeSplitRope(
            positions: stage1Positions,
            dim: 4096,
            theta: 10_000.0,
            maxPos: [20, 2048, 2048],
            numHeads: 32
        )

        latents = denoiseLoop(
            latents: latents,
            rope: stage1Rope,
            context: context,
            transformer: transformer,
            label: "stage1",
            sigmas: STAGE1Sigmas,
            conditioning: stage1ConditioningState
        )
        MLX.eval(latents)

        let stage1Latents = latents
        if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
            let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
            let parent = base.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            let stem = base.lastPathComponent
            try? MLX.save(array: stage1Latents, url: parent.appendingPathComponent("\(stem)_stage1_latents.npy"))
        }

        // Python reference initializes and loads stage-2 models after stage 1.
        // Replay that init/load flow to keep stage-2 RNG stream aligned.
        guard let modelWeightsURL else {
            throw LTXDistilledLatentGeneratorError.generatorNotLoaded
        }
        guard let loadedRoot else {
            throw LTXDistilledLatentGeneratorError.generatorNotLoaded
        }
        let upsamplerWeightsURL = loadedRoot.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors", isDirectory: false)
        try advanceRandomStreamForPythonParityAfterStage1(
            modelWeightsURL: modelWeightsURL,
            upsamplerWeightsURL: upsamplerWeightsURL,
            dtype: loadedDType
        )

        latents = upsampleLatents(
            latents,
            upsampler: upsampler,
            latentMean: decoder.latentsMean,
            latentStd: decoder.latentsStd
        )
        MLX.eval(latents)
        if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
            let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
            let parent = base.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            let stem = base.lastPathComponent
            try? MLX.save(array: latents, url: parent.appendingPathComponent("\(stem)_upsampled_latents.npy"))
        }

        if let stage2State = stage2ConditioningLatent.map({
            applyLatentConditioning(
                baseLatent: latents,
                conditionedLatent: $0,
                frameIndex: options.imageFrameIndex,
                strength: options.imageStrength,
                endConditionedLatent: stage2EndConditioningLatent,
                endFrameIndex: -1,
                endStrength: options.endImageStrength
            )
        }) {
            let noise = MLXRandom.normal(latents.shape).asType(modelDType)
            let noiseScale = MLXArray(STAGE2Sigmas[0]).asType(modelDType)
            let one = MLXArray(1.0).asType(modelDType)
            let scaledMask = stage2State.denoiseMask * noiseScale
            latents = noise * scaledMask + stage2State.latent * (one - scaledMask)
            MLX.eval(latents)
            stage2ConditioningState = stage2State
        } else {
            let noiseScale = MLXArray(STAGE2Sigmas[0]).asType(modelDType)
            let oneMinusScale = MLXArray(1.0 - STAGE2Sigmas[0]).asType(modelDType)
            let noise = MLXRandom.normal(latents.shape).asType(modelDType)
            latents = noise * noiseScale + latents * oneMinusScale
            MLX.eval(latents)
        }
        if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
            let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
            let parent = base.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            let stem = base.lastPathComponent
            try? MLX.save(array: latents, url: parent.appendingPathComponent("\(stem)_stage2_init_latents.npy"))
        }

        let stage2Positions = createPositionGrid(
            batchSize: 1,
            numFrames: latentFrames,
            height: stage2H,
            width: stage2W,
            temporalScale: 8,
            spatialScale: 32,
            fps: Float(max(1, options.fps)),
            causalFix: true
        )
        let stage2Rope = precomputeSplitRope(
            positions: stage2Positions,
            dim: 4096,
            theta: 10_000.0,
            maxPos: [20, 2048, 2048],
            numHeads: 32
        )

        latents = denoiseLoop(
            latents: latents,
            rope: stage2Rope,
            context: context,
            transformer: transformer,
            label: "stage2",
            sigmas: STAGE2Sigmas,
            conditioning: stage2ConditioningState
        )
        MLX.eval(latents)

        return LTXDistilledLatentGenerationResult(latents: latents, stage1Latents: stage1Latents)
    }

    public func generateVideo(
        options: LTXDistilledLatentGenerationOptions
    ) async throws -> LTXDistilledVideoGenerationResult {
        guard let decoder else {
            throw LTXDistilledLatentGeneratorError.decoderNotLoaded
        }

        let latentResult = try await generate(options: options)
        let decoded: MLXArray?
        let frames: MLXArray
        if let tiling = selectDecodeTilingConfig(
            width: options.width,
            height: options.height,
            numFrames: options.numFrames,
            fps: options.fps
        ) {
            decoded = nil
            frames = decodeWithTiling(
                decoder: decoder,
                latents: latentResult.latents,
                spatialTileSizeInPixels: tiling.spatialTileSizeInPixels,
                spatialOverlapInPixels: tiling.spatialTileOverlapInPixels,
                temporalTileSizeInFrames: tiling.temporalTileSizeInFrames,
                temporalOverlapInFrames: tiling.temporalTileOverlapInFrames,
                spatialScale: 32,
                temporalScale: 8
            )
        } else {
            let fullDecoded = decoder.decode(sample: latentResult.latents, timestep: nil)
            decoded = fullDecoded
            frames = postprocessDecodedVideo(fullDecoded)
        }
        MLX.eval(frames)
        ltxTraceMemory("video-decode-ready")

        if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
            let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
            let parent = base.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            let stem = base.lastPathComponent
            if let decoded {
                try? MLX.save(array: decoded, url: parent.appendingPathComponent("\(stem)_decoded.npy"))
            }
            try? MLX.save(array: frames, url: parent.appendingPathComponent("\(stem)_frames_postprocess.npy"))
        }
        return LTXDistilledVideoGenerationResult(frames: frames, latents: latentResult.latents)
    }
}
