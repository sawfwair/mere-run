import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

public struct LTXTextToAudioGuidance: Sendable, Hashable {
    public let classifierFreeScale: Float
    public let spatioTemporalScale: Float
    public let rescale: Float
    public let spatioTemporalBlocks: Set<Int>
    public let skipStep: Int

    public init(
        classifierFreeScale: Float = 7,
        spatioTemporalScale: Float = 1,
        rescale: Float = 0.7,
        spatioTemporalBlocks: Set<Int> = [28],
        skipStep: Int = 0
    ) {
        self.classifierFreeScale = classifierFreeScale
        self.spatioTemporalScale = spatioTemporalScale
        self.rescale = rescale
        self.spatioTemporalBlocks = spatioTemporalBlocks
        self.skipStep = skipStep
    }

    func combine(
        conditioned: MLXArray,
        negativeText: MLXArray,
        perturbed: MLXArray
    ) -> MLXArray {
        let dtype = conditioned.dtype
        let conditioned32 = conditioned.asType(.float32)
        var prediction = conditioned32
            + MLXArray(classifierFreeScale - 1)
                * (conditioned32 - negativeText.asType(.float32))
            + MLXArray(spatioTemporalScale)
                * (conditioned32 - perturbed.asType(.float32))
        if rescale != 0 {
            let factor = sampleStandardDeviation(conditioned32)
                / sampleStandardDeviation(prediction)
            prediction = prediction
                * (MLXArray(rescale) * factor + MLXArray(1 - rescale))
        }
        return prediction.asType(dtype)
    }

    func shouldSkip(step: Int) -> Bool {
        skipStep > 0 && !step.isMultiple(of: skipStep + 1)
    }
}

public struct LTXTextToAudioGenerationOptions: Sendable {
    public let prompt: String
    public let negativePrompt: String
    public let numFrames: Int
    public let fps: Double
    public let seed: Int
    public let inferenceSteps: Int
    public let maxTextLength: Int
    public let guidance: LTXTextToAudioGuidance
    public let sigmas: [Float]?
    public let loras: [LTXLoRAConfiguration]

    public init(
        prompt: String,
        negativePrompt: String = LTXUnifiedAVGenerationOptions.defaultNegativePrompt,
        numFrames: Int,
        fps: Double = 24,
        seed: Int,
        inferenceSteps: Int = 30,
        maxTextLength: Int = 1_024,
        guidance: LTXTextToAudioGuidance = LTXTextToAudioGuidance(),
        sigmas: [Float]? = nil,
        loras: [LTXLoRAConfiguration] = []
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.numFrames = numFrames
        self.fps = fps
        self.seed = seed
        self.inferenceSteps = inferenceSteps
        self.maxTextLength = maxTextLength
        self.guidance = guidance
        self.sigmas = sigmas
        self.loras = loras
    }
}

public struct LTXTextToAudioGenerationResult: @unchecked Sendable {
    public let audioLatents: MLXArray
    public let audioWaveform: MLXArray
    public let audioSampleRate: Int
    public let timings: LTXGenerationTimings

    public init(
        audioLatents: MLXArray,
        audioWaveform: MLXArray,
        audioSampleRate: Int,
        timings: LTXGenerationTimings = LTXGenerationTimings()
    ) {
        self.audioLatents = audioLatents
        self.audioWaveform = audioWaveform
        self.audioSampleRate = audioSampleRate
        self.timings = timings
    }
}

struct LTXUnifiedGenerationOutput {
    let frames: MLXArray
    let hdrOutput: LTXHDROutputFrames?
    let videoLatents: MLXArray
    let audioLatents: MLXArray
    let audioWaveform: MLXArray?
    let audioSampleRate: Int?
    let generatedKeyframeLatents: MLXArray?
    let generatedKeyframeIndices: [Int]
    let playbackFPS: Double
    let timings: LTXGenerationTimings
}
