import Foundation
import MLX

/// The upstream LTX-2.3 audio-to-video guidance defaults.
public struct LTXAudioToVideoGuidance: Sendable, Hashable {
    public var classifierFreeScale: Float
    public var spatioTemporalScale: Float
    public var rescale: Float
    public var audioToVideoScale: Float
    public var spatioTemporalBlocks: Set<Int>

    public init(
        classifierFreeScale: Float = 3,
        spatioTemporalScale: Float = 1,
        rescale: Float = 0.7,
        audioToVideoScale: Float = 3,
        spatioTemporalBlocks: Set<Int> = [28]
    ) {
        self.classifierFreeScale = classifierFreeScale
        self.spatioTemporalScale = spatioTemporalScale
        self.rescale = rescale
        self.audioToVideoScale = audioToVideoScale
        self.spatioTemporalBlocks = spatioTemporalBlocks
    }

    func combine(
        conditioned: MLXArray,
        negativeText: MLXArray,
        perturbed: MLXArray,
        isolatedAudio: MLXArray
    ) -> MLXArray {
        let originalDType = conditioned.dtype
        let conditioned32 = conditioned.asType(.float32)
        let negative32 = negativeText.asType(.float32)
        let perturbed32 = perturbed.asType(.float32)
        let isolated32 = isolatedAudio.asType(.float32)
        var prediction = conditioned32
            + MLXArray(classifierFreeScale - 1) * (conditioned32 - negative32)
            + MLXArray(spatioTemporalScale) * (conditioned32 - perturbed32)
            + MLXArray(audioToVideoScale - 1) * (conditioned32 - isolated32)

        if rescale != 0 {
            let factor = sampleStandardDeviation(conditioned32) / sampleStandardDeviation(prediction)
            prediction = prediction * (MLXArray(rescale) * factor + MLXArray(1 - rescale))
        }
        return prediction.asType(originalDType)
    }
}

/// Guidance for one stream of the full LTX 2.3 joint audio/video denoiser.
public struct LTXMultiModalGuidance: Sendable, Hashable {
    public let classifierFreeScale: Float
    public let spatioTemporalScale: Float
    public let rescale: Float
    public let modalityScale: Float
    public let spatioTemporalBlocks: Set<Int>
    public let skipStep: Int

    public init(
        classifierFreeScale: Float,
        spatioTemporalScale: Float = 1,
        rescale: Float = 0.7,
        modalityScale: Float = 3,
        spatioTemporalBlocks: Set<Int> = [28],
        skipStep: Int = 0
    ) {
        self.classifierFreeScale = classifierFreeScale
        self.spatioTemporalScale = spatioTemporalScale
        self.rescale = rescale
        self.modalityScale = modalityScale
        self.spatioTemporalBlocks = spatioTemporalBlocks
        self.skipStep = skipStep
    }

    func combine(
        conditioned: MLXArray,
        negativeText: MLXArray,
        perturbed: MLXArray,
        isolatedModality: MLXArray
    ) -> MLXArray {
        let originalDType = conditioned.dtype
        let conditioned32 = conditioned.asType(.float32)
        let negative32 = negativeText.asType(.float32)
        let perturbed32 = perturbed.asType(.float32)
        let isolated32 = isolatedModality.asType(.float32)
        var prediction = conditioned32
            + MLXArray(classifierFreeScale - 1) * (conditioned32 - negative32)
            + MLXArray(spatioTemporalScale) * (conditioned32 - perturbed32)
            + MLXArray(modalityScale - 1) * (conditioned32 - isolated32)

        if rescale != 0 {
            let factor = sampleStandardDeviation(conditioned32) / sampleStandardDeviation(prediction)
            prediction = prediction * (MLXArray(rescale) * factor + MLXArray(1 - rescale))
        }
        return prediction.asType(originalDType)
    }

    func shouldSkip(step: Int) -> Bool {
        skipStep > 0 && !step.isMultiple(of: skipStep + 1)
    }
}


/// Exact scheduler used by the full LTX-2.3 checkpoint in stage one.
public enum LTX2DiffusionScheduler {
    public static func sigmas(
        steps: Int,
        tokenCount: Int = 4_096,
        baseShift: Double = 0.95,
        maxShift: Double = 2.05,
        terminal: Double = 0.1
    ) -> [Float] {
        precondition(steps > 0, "LTX-2 diffusion requires at least one step.")
        let baseAnchor = 1_024.0
        let maxAnchor = 4_096.0
        let slope = (maxShift - baseShift) / (maxAnchor - baseAnchor)
        let intercept = baseShift - slope * baseAnchor
        let shift = Double(tokenCount) * slope + intercept
        let exponentialShift = exp(shift)

        var values = (0...steps).map { index -> Double in
            let sigma = 1 - Double(index) / Double(steps)
            guard sigma != 0 else { return 0 }
            return exponentialShift / (exponentialShift + (1 / sigma - 1))
        }

        let lastNonzeroIndex = steps - 1
        let scaleFactor = (1 - values[lastNonzeroIndex]) / (1 - terminal)
        for index in 0...lastNonzeroIndex {
            values[index] = 1 - (1 - values[index]) / scaleFactor
        }
        return values.map(Float.init)
    }
}

func sampleStandardDeviation(_ array: MLXArray) -> MLXArray {
    let count = array.shape.reduce(1, *)
    precondition(count > 1, "Guidance rescaling requires more than one latent value.")
    let mean = MLX.mean(array)
    let centered = array - mean
    return MLX.sqrt(MLX.sum(centered * centered) / MLXArray(Float(count - 1)))
}
