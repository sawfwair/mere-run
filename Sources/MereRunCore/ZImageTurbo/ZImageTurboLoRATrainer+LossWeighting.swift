import Foundation
import MLX
import MLXNN
import MLXRandom

extension ZImageTurboLoRATrainer {
    // MARK: - SNR Weighting

    /// Compute SNR (Signal-to-Noise Ratio) weight for a given sigma.
    /// SNR = (1 - sigma)^2 / sigma^2
    static func computeSNRWeight(sigma: Float, strategy: LossWeightingStrategy) -> Float {
        guard strategy != .none else { return 1.0 }

        // Clamp sigma to avoid division by zero
        let s = max(sigma, 1e-6)
        let snr = pow(1.0 - s, 2) / pow(s, 2)

        switch strategy {
        case .none:
            return 1.0
        case .snr:
            // Weight inversely by SNR to balance high-noise and low-noise steps
            return 1.0 / (snr + 1.0)
        case .minSNR:
            // Min-SNR-gamma weighting (gamma=5) from the paper
            // "Efficient Diffusion Training via Min-SNR Weighting Strategy"
            let gamma: Float = 5.0
            return min(snr, gamma) / snr
        }
    }
}

