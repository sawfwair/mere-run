import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor

extension MiniMaxH3Transformer {
    package func precomputeAdaLN(
        videoSchedule: MiniMaxH3Schedule,
        audioSchedule: MiniMaxH3Schedule,
        sourceIdentity: String,
        progressHandler: (@Sendable (Int, Int) -> Void)? = nil
    ) -> MiniMaxH3AdaLNCache {
        precondition(videoSchedule.timesteps.count == audioSchedule.timesteps.count)
        let stepTimeEmbeddings = videoSchedule.timesteps.indices.map { index in
            let timesteps = MLXArray([
                videoSchedule.timesteps[index],
                audioSchedule.timesteps[index],
                max(videoSchedule.timesteps[index], 0.999),
            ])
            let value = embedTimesteps(timesteps)
            MLX.eval(value)
            return value
        }
        let timeEmbeddings = MLX.stacked(stepTimeEmbeddings, axis: 0)
        MLX.eval(timeEmbeddings)

        let progressTotal = blocks.count + 2
        progressHandler?(1, progressTotal)
        var blockModulations: [MLXArray] = []
        blockModulations.reserveCapacity(blocks.count)
        for (index, block) in blocks.enumerated() {
            let stepModulations = stepTimeEmbeddings.map { timeEmbedding in
                let value = block.precomputeModulation(timeEmbedding: timeEmbedding)
                MLX.eval(value)
                return value
            }
            let modulation = MLX.stacked(stepModulations, axis: 0)
            MLX.eval(modulation)
            blockModulations.append(modulation)
            progressHandler?(index + 2, progressTotal)
        }
        let finalModulations = MLX.stacked(stepTimeEmbeddings.map { timeEmbedding in
            let value = finalLayer.precomputeModulation(timeEmbedding: timeEmbedding)
            MLX.eval(value)
            return value
        }, axis: 0)
        MLX.eval(finalModulations)
        progressHandler?(progressTotal, progressTotal)
        return MiniMaxH3AdaLNCache(
            timeEmbeddings: timeEmbeddings,
            blockModulations: blockModulations,
            finalModulations: finalModulations,
            videoSigmas: videoSchedule.sigmas,
            audioSigmas: audioSchedule.sigmas,
            sourceIdentity: sourceIdentity
        )
    }

    package func precomputeAdaLNStep(timesteps: MLXArray) -> MiniMaxH3AdaLNStep {
        precondition(timesteps.shape == [3])
        let timeEmbedding = embedTimesteps(timesteps)
        MLX.eval(timeEmbedding)
        let blockModulations = blocks.map { block in
            let modulation = block.precomputeModulation(timeEmbedding: timeEmbedding)
            MLX.eval(modulation)
            return modulation
        }
        let finalModulation = finalLayer.precomputeModulation(
            timeEmbedding: timeEmbedding
        )
        MLX.eval(finalModulation)
        return MiniMaxH3AdaLNStep(
            timeEmbedding: timeEmbedding,
            blockModulations: blockModulations,
            finalModulation: finalModulation
        )
    }

    /// Releases the schedule-only projection weights after an exact modulation
    /// table has been built for the current run. The remaining dense core is
    /// the complete denoising transformer.
    package func discardAdaLNWeights() {
        guard adaLNWeightsAvailable else { return }
        compiledBlockRunner = nil
        compiledBlockForwards = nil
        update(modules: ModuleChildren.unflattened([
            ("time_embedder", MiniMaxH3TimeEmbedding(discarded: ())),
        ]))
        for block in blocks {
            block.discardAdaLNWeights()
        }
        finalLayer.discardAdaLNWeights()
        adaLNWeightsAvailable = false
    }

    func embedTimesteps(_ timesteps: MLXArray) -> MLXArray {
        guard adaLNWeightsAvailable, let timeEmbedder else {
            preconditionFailure("MiniMax-H3 AdaLN cache is required")
        }
        let arguments = timesteps.reshaped(-1, 1) * timeFrequencies.reshaped(1, -1)
        let sinusoidal = MLX.concatenated([MLX.cos(arguments), MLX.sin(arguments)], axis: -1)
        return timeEmbedder(sinusoidal.asType(.float32))
    }

    func rotaryEmbedding(positions: MLXArray) -> MiniMaxH3RotaryEmbedding {
        let frequencies = positions.asType(.float32).expandedDimensions(axis: -1)
            * inverseFrequencies.reshaped(1, 1, -1)
        let combined = frequencies.reshaped(positions.dim(0), 3 * configuration.ropeFrequencyCount)
        let angles = MLX.concatenated([combined, combined], axis: -1)
            .reshaped(1, positions.dim(0), 1, 6 * configuration.ropeFrequencyCount)
        return MiniMaxH3RotaryEmbedding(cosine: MLX.cos(angles), sine: MLX.sin(angles))
    }


}
