import Foundation
import MediaIO
import MLX
#if canImport(Darwin)
import Darwin
#endif
#if canImport(IOKit.ps)
import IOKit.ps
#endif

extension MiniMaxH3Generator {
    public func generate(
        options: MiniMaxH3GenerationOptions,
        resources: MiniMaxH3Resources,
        continuation: MiniMaxH3ContinuationInput? = nil,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)? = nil
    ) throws -> MiniMaxH3GenerationResult {
        let phaseProfileLogger = MereRunRuntimeDebug.logger(
            keys: ["MERERUN_H3_PROFILE_PHASES"],
            prefix: "[minimax-h3-phase-profile]"
        )
        let generationStarted = CFAbsoluteTimeGetCurrent()
        let missing = resources.validate()
        guard missing.isEmpty else { throw MiniMaxH3GeneratorError.missingModelFiles(missing) }
        let configuration = try resources.loadConfiguration()
        if let adapterInferenceRecipe = options.adapterInferenceRecipe,
           !adapterInferenceRecipe.supports(task: configuration.task) {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "MiniMax-H3 adapter recipe \(adapterInferenceRecipe.name) requires \(adapterInferenceRecipe.task.rawValue), not \(configuration.task)"
            )
        }
        if configuration.task == "ref2va" {
            return try generateRef2VA(
                options: options,
                resources: resources,
                configuration: configuration,
                continuation: continuation,
                progressHandler: progressHandler,
                phaseProfileLogger: phaseProfileLogger,
                generationStarted: generationStarted
            )
        }
        guard options.references.isEmpty else {
            throw MiniMaxH3GeneratorError.invalidOptions("--reference requires a Ref2VA model root")
        }
        let latentFrames = try MiniMaxH3Geometry.videoLatentFrameCount(for: options.numFrames)
        let latentHeight = options.internalHeight / 16
        let latentWidth = options.internalWidth / 16
        let audioFrames = MiniMaxH3Geometry.audioLatentFrameCount(for: options.numFrames)
        if let continuation {
            if options.adapterInferenceRecipe?.requiresTextOnlyConditioning == true {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "FastH3 Preview v1 supports text-to-audio/video only; continuation is unsupported"
                )
            }
            guard !options.usesReducedRenderCanvas else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "reduced internal rendering does not yet support continuation or sliding windows"
                )
            }
            guard continuation.frames.dim(2) == options.height,
                  continuation.frames.dim(3) == options.width else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "continuation dimensions must match the target H3 dimensions"
                )
            }
            guard !options.frameInputs.contains(where: { $0.frameIndex == 0 }),
                  options.firstFrameURL == nil else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "continuation already supplies the first target frame"
                )
            }
        }

        progressHandler?(.init(stage: .loadingTextEncoder, stepIndex: 0, totalSteps: options.steps - 1))
        let conditionerPreparationStarted = CFAbsoluteTimeGetCurrent()
        let tokenizer = try QwenTokenizer.load(from: resources.tokenizerURL, maxLengthOverride: 262_144)
        let presentation = try conditionerPresentation(
            tokenizer: tokenizer,
            options: options,
            continuationFrame: continuation.map(Self.boundaryFrame)
        )
        guard !presentation.tokenIDs.isEmpty else {
            throw MiniMaxH3GeneratorError.invalidOptions("prompt tokenized to zero rows")
        }
        guard presentation.tokenIDs.count <= 262_144 else {
            throw MiniMaxH3GeneratorError.invalidOptions("multimodal presentation exceeds 262144 tokens")
        }
        let inputIDs = MLXArray(presentation.tokenIDs.map(Int32.init)).reshaped(1, presentation.tokenIDs.count)
        let attentionMask = MLXArray.ones([1, presentation.tokenIDs.count], dtype: .int32)
        phaseProfileLogger?(String(
            format: "phase=conditioner_preparation seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - conditionerPreparationStarted
        ))
        let promptStates: MLXArray = try withMiniMaxH3AutoreleasePool {
            let conditionerLoadStarted = CFAbsoluteTimeGetCurrent()
            let encoder = try loadConditioner(
                resources: resources,
                configuration: configuration,
                progressHandler: { shard in
                    progressHandler?(.init(
                        stage: .loadingTextEncoder,
                        stepIndex: shard.shardIndex,
                        totalSteps: shard.shardCount
                    ))
                }
            )
            phaseProfileLogger?(String(
                format: "phase=conditioner_load seconds=%.3f",
                CFAbsoluteTimeGetCurrent() - conditionerLoadStarted
            ))
            progressHandler?(.init(stage: .encodingText, stepIndex: 0, totalSteps: options.steps - 1))
            let textEncodingStarted = CFAbsoluteTimeGetCurrent()
            guard let states = try encoder.forwardMultimodalActivationHiddenState(
                inputIds: inputIDs,
                attentionMask: attentionMask,
                images: presentation.images,
                activationLayer: 49
            ) else {
                throw MiniMaxH3GeneratorError.invalidOptions("text encoder did not return layer-50 states")
            }
            MLX.eval(states)
            phaseProfileLogger?(String(
                format: "phase=text_encoding seconds=%.3f",
                CFAbsoluteTimeGetCurrent() - textEncodingStarted
            ))
            return states
        }
        Memory.clearCache()

        let frameConditions = Self.frameConditions(options: options)
        let keyframeEncodingStarted = CFAbsoluteTimeGetCurrent()
        let continuationConditions = try continuation.map {
            try encodeContinuation(
                $0,
                resources: resources,
                progressHandler: progressHandler
            )
        }
        let keyframeRows = try encodeKeyframes(
            frameConditions.map(\.url),
            options: options,
            resources: resources,
            progressHandler: progressHandler
        )
        var conditionVideoRows = Self.concatenateRows([
            continuationConditions?.videoRows,
            keyframeRows,
        ])
        var conditionAudioRows = continuationConditions?.audioRows
        phaseProfileLogger?(String(
            format: "phase=keyframe_encoding seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - keyframeEncodingStarted
        ))
        Memory.clearCache()

        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: presentation.tokenTags,
            videoLatentFrames: latentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            audioLatentFrames: audioFrames,
            keyframeAnchors: (continuationConditions?.videoAnchors ?? [])
                + frameConditions.map(\.anchor),
            audioConditionAnchors: continuationConditions?.audioAnchors ?? []
        )
        if let currentConditions = conditionVideoRows {
            conditionVideoRows = MiniMaxH3ServingContract.noisedCondition(
                currentConditions,
                seed: options.seed
            )
        }
        if let currentConditions = conditionAudioRows {
            conditionAudioRows = MiniMaxH3ServingContract.noisedCondition(
                currentConditions,
                seed: options.seed &+ 1
            )
        }
        var video = MiniMaxH3ServingContract.targetVideoNoise(
            seed: options.seed,
            latentFrames: latentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth
        )
        var videoRows = MiniMaxH3Geometry.patchifyVideo(video)
        var audioRows = MiniMaxH3ServingContract.targetAudioNoise(
            seed: options.seed,
            latentFrames: audioFrames
        )
        let videoShift = options.adapterInferenceRecipe?.videoFlowShift ?? configuration.videoFlowShift
        let audioShift = options.adapterInferenceRecipe?.audioFlowShift ?? configuration.audioFlowShift
        let videoSchedule: MiniMaxH3Schedule
        let audioSchedule: MiniMaxH3Schedule
        if let baseSigmas = options.adapterInferenceRecipe?.baseDenoisingSigmas {
            videoSchedule = try MiniMaxH3Schedule(baseSigmas: baseSigmas, shift: videoShift)
            audioSchedule = try MiniMaxH3Schedule(baseSigmas: baseSigmas, shift: audioShift)
        } else {
            videoSchedule = try MiniMaxH3Schedule(pointCount: options.steps, shift: videoShift)
            audioSchedule = try MiniMaxH3Schedule(pointCount: options.steps, shift: audioShift)
        }

        progressHandler?(.init(stage: .loadingTransformer, stepIndex: 0, totalSteps: options.steps - 1))
        try withMiniMaxH3AutoreleasePool {
            let transformerPreparationStarted = CFAbsoluteTimeGetCurrent()
            let runtime = try loadDenoisingRuntime(
                resources: resources,
                configuration: configuration,
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule,
                sequenceLength: layout.sequenceLength,
                weightMode: options.transformerWeightMode,
                adapterURL: options.adapterURL,
                adapterStrength: options.adapterStrength,
                progressHandler: progressHandler
            )
            phaseProfileLogger?(String(
                format: "phase=transformer_preparation seconds=%.3f resident_bf16=%@",
                CFAbsoluteTimeGetCurrent() - transformerPreparationStarted,
                runtime.transformer.usesResidentBF16 ? "true" : "false"
            ))
            let denoisingStarted = CFAbsoluteTimeGetCurrent()
            (videoRows, audioRows) = try denoise(
                transformer: runtime.transformer,
                videoRows: videoRows,
                audioRows: audioRows,
                conditionVideoRows: conditionVideoRows,
                conditionAudioRows: conditionAudioRows,
                promptStates: promptStates,
                layout: layout,
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule,
                adaLNCache: runtime.adaLNCache,
                accelerationMode: options.accelerationMode,
                permitsCacheReuse: true,
                progressHandler: progressHandler
            )
            phaseProfileLogger?(String(
                format: "phase=denoising seconds=%.3f",
                CFAbsoluteTimeGetCurrent() - denoisingStarted
            ))
        }
        Memory.clearCache()
        video = MiniMaxH3Geometry.unpatchifyVideo(
            videoRows,
            frames: latentFrames,
            height: latentHeight,
            width: latentWidth
        )
        let audio = MiniMaxH3Geometry.unpackAudio(audioRows[0])
        progressHandler?(.init(stage: .decodingVideo, stepIndex: options.steps - 1, totalSteps: options.steps - 1))
        let videoDecodingStarted = CFAbsoluteTimeGetCurrent()
        let frames: MLXArray = try withMiniMaxH3AutoreleasePool {
            let vae = try loadVideoVAE(resources: resources)
            let decoded = Self.mediaFrames(from: vae.decode(video))
            let pixels = try MiniMaxH3FrameScaler.scaled(
                decoded,
                width: options.width,
                height: options.height
            )
            MLX.eval(pixels)
            return pixels
        }
        phaseProfileLogger?(String(
            format: "phase=video_decoding seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - videoDecodingStarted
        ))
        Memory.clearCache()
        progressHandler?(.init(stage: .decodingAudio, stepIndex: options.steps - 1, totalSteps: options.steps - 1))
        let audioDecodingStarted = CFAbsoluteTimeGetCurrent()
        let waveform: MLXArray = try withMiniMaxH3AutoreleasePool {
            let vae = try loadAudioVAE(resources: resources)
            let decoded = vae.decode(audio)
            MLX.eval(decoded)
            return decoded
        }
        phaseProfileLogger?(String(
            format: "phase=audio_decoding seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - audioDecodingStarted
        ))
        Memory.clearCache()
        phaseProfileLogger?(String(
            format: "phase=generation_total seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - generationStarted
        ))
        return MiniMaxH3GenerationResult(frames: frames, audio: waveform, seed: options.seed)
    }

}
