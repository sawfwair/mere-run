import Foundation
import MLX

public enum MiniMaxH3AdaLNCacheError: LocalizedError, Sendable {
    case incompatible(String)

    public var errorDescription: String? {
        switch self {
        case .incompatible(let reason):
            return "MiniMax-H3 AdaLN cache is incompatible: \(reason)"
        }
    }
}

package struct MiniMaxH3AdaLNCache {
    package static let filename = "adaln_cache.safetensors"
    package static let schemaVersion = "2"

    package let timeEmbeddings: MLXArray
    package let blockModulations: [MLXArray]
    package let finalModulations: MLXArray
    package let videoSigmas: [Float]
    package let audioSigmas: [Float]
    package let sourceIdentity: String

    package init(
        timeEmbeddings: MLXArray,
        blockModulations: [MLXArray],
        finalModulations: MLXArray,
        videoSigmas: [Float],
        audioSigmas: [Float],
        sourceIdentity: String
    ) {
        self.timeEmbeddings = timeEmbeddings
        self.blockModulations = blockModulations
        self.finalModulations = finalModulations
        self.videoSigmas = videoSigmas
        self.audioSigmas = audioSigmas
        self.sourceIdentity = sourceIdentity
    }

    package var stepCount: Int { videoSigmas.count - 1 }

    package func step(at index: Int) -> MiniMaxH3AdaLNStep {
        precondition(index >= 0 && index < stepCount)
        return MiniMaxH3AdaLNStep(
            timeEmbedding: timeEmbeddings[index],
            blockModulations: blockModulations.map { $0[index] },
            finalModulation: finalModulations[index]
        )
    }

    package func isCompatible(
        configuration: MiniMaxH3TransformerConfiguration,
        videoSchedule: MiniMaxH3Schedule,
        audioSchedule: MiniMaxH3Schedule
    ) -> Bool {
        isStructurallyCompatible(configuration: configuration)
            && videoSigmas == videoSchedule.sigmas
            && audioSigmas == audioSchedule.sigmas
    }

    package func isStructurallyCompatible(
        configuration: MiniMaxH3TransformerConfiguration
    ) -> Bool {
        videoSigmas.count >= 2
            && videoSigmas.count == audioSigmas.count
            && blockModulations.count == configuration.layerCount
            && timeEmbeddings.shape == [stepCount, 3, configuration.timeEmbeddingDimension]
            && finalModulations.shape == [stepCount, 3, 2 * configuration.hiddenSize]
            && blockModulations.allSatisfy {
                $0.shape == [stepCount, 3 * 3, 6 * configuration.hiddenSize]
            }
    }

    /// Rebuilds the small inference table for another sampler schedule without
    /// restoring the 13B-parameter AdaLN branch. AdaLN is a smooth function of
    /// timestep, and every stored video/audio point contains all three modality
    /// rows. Combining both source schedules therefore gives up to twice the
    /// sampling density of either schedule alone.
    package func resampled(
        configuration: MiniMaxH3TransformerConfiguration,
        videoSchedule: MiniMaxH3Schedule,
        audioSchedule: MiniMaxH3Schedule
    ) throws -> MiniMaxH3AdaLNCache {
        guard isStructurallyCompatible(configuration: configuration) else {
            throw MiniMaxH3AdaLNCacheError.incompatible("source tensor geometry does not match")
        }
        guard videoSchedule.timesteps.count == audioSchedule.timesteps.count else {
            throw MiniMaxH3AdaLNCacheError.incompatible("target video/audio schedules disagree")
        }
        if isCompatible(
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule
        ) {
            return self
        }

        let samples = sourceSamples()
        let targetTimesteps = videoSchedule.timesteps.indices.map { index in
            [
                videoSchedule.timesteps[index],
                audioSchedule.timesteps[index],
                max(videoSchedule.timesteps[index], 0.999),
            ]
        }

        let resampledTimeEmbeddings = try MLX.stacked(targetTimesteps.map { step in
            try MLX.stacked(step.map { timestep in
                try interpolatedValue(timestep: timestep, samples: samples) { sample in
                    timeEmbeddings[sample.stepIndex, sample.timestepIndex, 0...]
                }
            }, axis: 0)
        }, axis: 0)
        MLX.eval(resampledTimeEmbeddings)

        var resampledBlocks: [MLXArray] = []
        resampledBlocks.reserveCapacity(blockModulations.count)
        for source in blockModulations {
            let resampled = try MLX.stacked(targetTimesteps.map { step in
                try MLX.concatenated(step.map { timestep in
                    try interpolatedValue(timestep: timestep, samples: samples) { sample in
                        let start = sample.timestepIndex * 3
                        return source[sample.stepIndex, start..<(start + 3), 0...]
                    }
                }, axis: 0)
            }, axis: 0)
            MLX.eval(resampled)
            resampledBlocks.append(resampled)
        }

        let resampledFinal = try MLX.stacked(targetTimesteps.map { step in
            try MLX.stacked(step.map { timestep in
                try interpolatedValue(timestep: timestep, samples: samples) { sample in
                    finalModulations[sample.stepIndex, sample.timestepIndex, 0...]
                }
            }, axis: 0)
        }, axis: 0)
        MLX.eval(resampledFinal)

        return MiniMaxH3AdaLNCache(
            timeEmbeddings: resampledTimeEmbeddings,
            blockModulations: resampledBlocks,
            finalModulations: resampledFinal,
            videoSigmas: videoSchedule.sigmas,
            audioSigmas: audioSchedule.sigmas,
            sourceIdentity: sourceIdentity
        )
    }

}

private extension MiniMaxH3AdaLNCache {
    struct SourceSample {
        let timestep: Float
        let stepIndex: Int
        /// 0 = video, 1 = audio, 2 = condition. Block caches hold three
        /// modality rows per timestep; final/time tables hold one row.
        let timestepIndex: Int
    }

    struct Interpolation {
        let lower: SourceSample
        let upper: SourceSample
        let fraction: Float
    }

    func sourceSamples() -> [SourceSample] {
        var values: [SourceSample] = []
        values.reserveCapacity(stepCount * 3)
        for index in 0..<stepCount {
            let videoTimestep = 1 - videoSigmas[index]
            values.append(.init(timestep: videoTimestep, stepIndex: index, timestepIndex: 0))
            values.append(.init(
                timestep: 1 - audioSigmas[index],
                stepIndex: index,
                timestepIndex: 1
            ))
            values.append(.init(
                timestep: max(videoTimestep, 0.999),
                stepIndex: index,
                timestepIndex: 2
            ))
        }
        values.sort { $0.timestep < $1.timestep }
        return values.reduce(into: []) { unique, sample in
            if unique.last?.timestep != sample.timestep {
                unique.append(sample)
            }
        }
    }

    func interpolation(
        for timestep: Float,
        samples: [SourceSample]
    ) throws -> Interpolation {
        guard let first = samples.first, let last = samples.last,
              timestep >= first.timestep, timestep <= last.timestep else {
            throw MiniMaxH3AdaLNCacheError.incompatible(
                "target timestep \(timestep) is outside the cached curve"
            )
        }
        let upperIndex = samples.firstIndex { $0.timestep >= timestep } ?? (samples.count - 1)
        let upper = samples[upperIndex]
        if upper.timestep == timestep || upperIndex == 0 {
            return Interpolation(lower: upper, upper: upper, fraction: 0)
        }
        let lower = samples[upperIndex - 1]
        let fraction = (timestep - lower.timestep) / (upper.timestep - lower.timestep)
        return Interpolation(lower: lower, upper: upper, fraction: fraction)
    }

    func interpolatedValue(
        timestep: Float,
        samples: [SourceSample],
        value: (SourceSample) -> MLXArray
    ) throws -> MLXArray {
        let interpolation = try interpolation(for: timestep, samples: samples)
        let lower = value(interpolation.lower)
        guard interpolation.fraction != 0 else { return lower }
        let upper = value(interpolation.upper)
        return lower * (1 - interpolation.fraction) + upper * interpolation.fraction
    }
}
