import Foundation
import MLX
import MLXNN
import MLXRandom

extension ZImageTurboLoRATrainer {
    // MARK: - Wavelet Loss

    /// Compute Haar wavelet decomposition of a 2D array.
    /// Returns (LL, LH, HL, HH) subbands.
    private static func haarWavelet2D(_ x: MLXArray) -> (ll: MLXArray, lh: MLXArray, hl: MLXArray, hh: MLXArray) {
        // x shape: [B, C, H, W]
        // Haar wavelet: average and difference of adjacent pixels
        // Step 1: Apply along columns (W dimension)
        let xEvenCols = x[0..., 0..., 0..., .stride(by: 2)]
        let xOddCols = x[0..., 0..., 0..., .stride(from: 1, by: 2)]
        let l = (xEvenCols + xOddCols) * 0.5  // Low-pass columns
        let h = (xEvenCols - xOddCols) * 0.5  // High-pass columns

        // Step 2: Apply along rows (H dimension)
        let lEvenRows = l[0..., 0..., .stride(by: 2), 0...]
        let lOddRows = l[0..., 0..., .stride(from: 1, by: 2), 0...]
        let hEvenRows = h[0..., 0..., .stride(by: 2), 0...]
        let hOddRows = h[0..., 0..., .stride(from: 1, by: 2), 0...]

        let ll = (lEvenRows + lOddRows) * 0.5  // Low-Low (approximation)
        let lh = (lEvenRows - lOddRows) * 0.5  // Low-High (horizontal detail)
        let hl = (hEvenRows + hOddRows) * 0.5  // High-Low (vertical detail)
        let hh = (hEvenRows - hOddRows) * 0.5  // High-High (diagonal detail)

        return (ll, lh, hl, hh)
    }

    /// Compute wavelet loss between prediction and target.
    /// Focuses on high-frequency detail preservation.
    static func computeWaveletLoss(prediction: MLXArray, target: MLXArray) -> MLXArray {
        let predWavelet = haarWavelet2D(prediction)
        let targetWavelet = haarWavelet2D(target)

        // Loss on high-frequency subbands (LH, HL, HH)
        let lhLoss = (predWavelet.lh - targetWavelet.lh).square().mean()
        let hlLoss = (predWavelet.hl - targetWavelet.hl).square().mean()
        let hhLoss = (predWavelet.hh - targetWavelet.hh).square().mean()

        return lhLoss + hlLoss + hhLoss
    }
}

