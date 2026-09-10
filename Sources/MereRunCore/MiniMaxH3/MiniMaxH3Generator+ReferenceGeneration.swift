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
    func generateRef2VA(
        options: MiniMaxH3GenerationOptions,
        resources: MiniMaxH3Resources,
        configuration: MiniMaxH3Configuration,
        continuation: MiniMaxH3ContinuationInput?,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)?,
        phaseProfileLogger: (@Sendable (String) -> Void)?,
        generationStarted: CFTimeInterval
    ) throws -> MiniMaxH3GenerationResult {
        guard options.firstFrameURL == nil,
              options.lastFrameURL == nil,
              options.frameInputs.isEmpty else {
            throw MiniMaxH3GeneratorError.invalidOptions("Ref2VA uses ordered references, not FL2VA keyframes")
        }
        guard !options.references.isEmpty else {
            throw MiniMaxH3GeneratorError.invalidOptions("Ref2VA requires at least one --reference")
        }
        let latentFrames = try MiniMaxH3Geometry.videoLatentFrameCount(for: options.numFrames)
        let latentHeight = options.internalHeight / 16
        let latentWidth = options.internalWidth / 16
        let audioFrames = MiniMaxH3Geometry.audioLatentFrameCount(for: options.numFrames)
        if let continuation {
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
        }
        let prepared = try preparedReferences(options: options)

        progressHandler?(.init(stage: .loadingTextEncoder, stepIndex: 0, totalSteps: options.steps - 1))
        let conditionerPreparationStarted = CFAbsoluteTimeGetCurrent()
        let tokenizer = try QwenTokenizer.load(from: resources.tokenizerURL, maxLengthOverride: 262_144)
        let presentation = try referenceConditionerPresentation(
            tokenizer: tokenizer,
            prompt: options.prompt,
            references: prepared,
            continuationFrame: continuation.map(Self.boundaryFrame)
        )
        guard !presentation.tokenIDs.isEmpty, presentation.tokenIDs.count <= 262_144 else {
            throw MiniMaxH3GeneratorError.invalidOptions("Ref2VA presentation must contain 1...262144 tokens")
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

        let referenceEncodingStarted = CFAbsoluteTimeGetCurrent()
        let referenceRows = try encodeReferences(
            prepared,
            options: options,
            resources: resources,
            progressHandler: progressHandler
        )
        let continuationConditions = try continuation.map {
            try encodeContinuation(
                $0,
                resources: resources,
                progressHandler: progressHandler
            )
        }
        var videoConditionSpans = referenceRows.video
        if let continuationVideo = continuationConditions?.videoRows {
            videoConditionSpans.insert(continuationVideo, at: 0)
        }
        var audioConditionSpans = referenceRows.audio
        if let continuationAudio = continuationConditions?.audioRows {
            audioConditionSpans.insert(continuationAudio, at: 0)
        }
        phaseProfileLogger?(String(
            format: "phase=reference_encoding seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - referenceEncodingStarted
        ))
        Memory.clearCache()

        let layout = try MiniMaxH3Geometry.buildRef2VA(
            textTokenTags: presentation.tokenTags,
            references: prepared.map(\.geometry),
            videoLatentFrames: latentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            audioLatentFrames: audioFrames,
            keyframeAnchors: continuationConditions?.videoAnchors ?? [],
            audioConditionAnchors: continuationConditions?.audioAnchors ?? []
        )
        let noisedVideoConditions = Self.concatenateRows(videoConditionSpans.map {
            MiniMaxH3ServingContract.noisedCondition($0, seed: options.seed)
        })
        let noisedAudioConditions = Self.concatenateRows(audioConditionSpans.map {
            MiniMaxH3ServingContract.noisedCondition($0, seed: options.seed &+ 1)
        })
        var videoRows = MiniMaxH3Geometry.patchifyVideo(
            MiniMaxH3ServingContract.targetVideoNoise(
                seed: options.seed,
                latentFrames: latentFrames,
                latentHeight: latentHeight,
                latentWidth: latentWidth
            )
        )
        var audioRows = MiniMaxH3ServingContract.targetAudioNoise(
            seed: options.seed,
            latentFrames: audioFrames
        )
        let videoSchedule = try MiniMaxH3Schedule(
            pointCount: options.steps,
            shift: options.adapterInferenceRecipe?.videoFlowShift ?? configuration.videoFlowShift
        )
        let audioSchedule = try MiniMaxH3Schedule(
            pointCount: options.steps,
            shift: options.adapterInferenceRecipe?.audioFlowShift ?? configuration.audioFlowShift
        )

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
                conditionVideoRows: noisedVideoConditions,
                conditionAudioRows: noisedAudioConditions,
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
        let video = MiniMaxH3Geometry.unpatchifyVideo(
            videoRows,
            frames: latentFrames,
            height: latentHeight,
            width: latentWidth
        )
        let audio = MiniMaxH3Geometry.unpackAudio(audioRows[0])
        progressHandler?(.init(stage: .decodingVideo, stepIndex: options.steps - 1, totalSteps: options.steps - 1))
        let videoDecodingStarted = CFAbsoluteTimeGetCurrent()
        let frames: MLXArray = try withMiniMaxH3AutoreleasePool {
            let decoded = Self.mediaFrames(
                from: try loadVideoVAE(resources: resources).decode(video)
            )
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
            let decoded = try loadAudioVAE(resources: resources).decode(audio)
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
