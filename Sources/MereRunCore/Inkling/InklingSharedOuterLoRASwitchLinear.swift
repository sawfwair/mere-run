import Foundation
import MLX
import MLXNN
import MLXRandom

/// Tinker-compatible LoRA for an expert-stacked projection.
///
/// The factor touching the model hidden dimension is shared across experts,
/// while the factor touching the expert intermediate dimension remains
/// expert-specific. This preserves per-expert adaptation without allocating a
/// full independent LoRA pair for all 256 routed experts.
final class InklingSharedOuterLoRASwitchLinear: Q35SwitchLinear, TrainableLoRALayer {
    enum SharedFactor: Sendable, Equatable {
        /// Gate/up projections: the input-side factor touches hidden size.
        case input
        /// Down projections: the output-side factor touches hidden size.
        case output
    }

    let loraRank: Int
    let loraAlpha: Float
    var loraDown: MLXArray
    var loraUp: MLXArray
    var isActive = true
    var role: LoRARole = .train

    var loraDownM: MLXArray?
    var loraDownV: MLXArray?
    var loraUpM: MLXArray?
    var loraUpV: MLXArray?

    let sharedFactor: SharedFactor
    private let scale: Float

    init(
        base: Q35SwitchLinear,
        rank: Int,
        alpha: Float,
        sharedFactor: SharedFactor,
        zeroInitUp: Bool
    ) {
        let expertCount = base.weight.dim(0)
        let outputDimensions = base.weight.dim(1)
        let inputDimensions: Int
        if let scales = base.scales {
            inputDimensions = scales.dim(scales.ndim - 1) * base.groupSize
        } else {
            inputDimensions = base.weight.dim(base.weight.ndim - 1)
        }

        self.loraRank = max(1, rank)
        self.loraAlpha = alpha
        self.scale = alpha / Float(max(1, rank))
        self.sharedFactor = sharedFactor

        let bound = 1.0 / sqrt(Float(inputDimensions))
        switch sharedFactor {
        case .input:
            self.loraDown = MLXRandom.uniform(
                low: -bound,
                high: bound,
                [self.loraRank, inputDimensions]
            ).asType(.float32)
            self.loraUp = zeroInitUp
                ? MLXArray.zeros([expertCount, outputDimensions, self.loraRank], dtype: .float32)
                : MLXRandom.normal([expertCount, outputDimensions, self.loraRank]).asType(.float32)
                    * (0.01 / sqrt(Float(self.loraRank)))
        case .output:
            self.loraDown = MLXRandom.uniform(
                low: -bound,
                high: bound,
                [expertCount, self.loraRank, inputDimensions]
            ).asType(.float32)
            self.loraUp = zeroInitUp
                ? MLXArray.zeros([outputDimensions, self.loraRank], dtype: .float32)
                : MLXRandom.normal([outputDimensions, self.loraRank]).asType(.float32)
                    * (0.01 / sqrt(Float(self.loraRank)))
        }

        super.init(
            weight: base.weight,
            scales: base.scales,
            biases: base.biases,
            bias: base.bias,
            groupSize: base.groupSize,
            bits: base.bits
        )
    }

    override func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let baseOutput = super.callAsFunction(x, indices: indices)
        guard isActive else { return baseOutput }

        let routedInput = expandForRoutes(x.asType(.float32), indices: indices)
        let contribution: MLXArray
        switch sharedFactor {
        case .input:
            let shared = MLX.matmul(x.asType(.float32), loraDown.T)
            let routedShared = expandForRoutes(shared, indices: indices)
            let selectedUp = loraUp.take(indices, axis: 0)
            contribution = (
                selectedUp * routedShared.expandedDimensions(axis: -2)
            ).sum(axis: -1)
        case .output:
            let selectedDown = loraDown.take(indices, axis: 0)
            let lowered = (
                selectedDown * routedInput.expandedDimensions(axis: -2)
            ).sum(axis: -1)
            contribution = MLX.matmul(lowered, loraUp.T)
        }

        return baseOutput + (contribution * MLXArray(scale)).asType(baseOutput.dtype)
    }

    private func expandForRoutes(_ values: MLXArray, indices: MLXArray) -> MLXArray {
        if values.ndim == 4 {
            return values
        }
        let expanded = values.expandedDimensions(axis: 2)
        return MLX.broadcast(
            expanded,
            to: [values.dim(0), values.dim(1), indices.dim(2), values.dim(2)]
        )
    }
}
