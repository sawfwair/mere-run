import Foundation
import MLX
import MLXNN
import MLXRandom

extension Flux2KleinLoRATrainer {
    // MARK: - SNR Weighting

    /// Compute SNR weight for a given sigma (on-device).
    static func computeSNRWeight(
        sigma: MLXArray,
        strategy: Flux2LossWeightingStrategy
    ) -> MLXArray {
        guard strategy != .none else { return MLXArray(1.0) }

        let s = maximum(sigma, 1e-6)
        let one = MLXArray(1.0)
        let snr = (one - s).square() / s.square()

        switch strategy {
        case .none:
            return one
        case .snr:
            return one / (snr + one)
        case .minSNR:
            let gamma = MLXArray(5.0)
            return minimum(snr, gamma) / snr
        }
    }
}

