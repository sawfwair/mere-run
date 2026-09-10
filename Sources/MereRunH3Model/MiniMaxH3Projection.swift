import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor
#if DEBUG
import MLXRandom
#endif

@inline(__always)
func miniMaxH3Linear(_ linear: Linear, _ value: MLXArray) -> MLXArray {
    let activationDType = (linear as? QuantizedLinear)?.scales.dtype ?? linear.weight.dtype
    return linear(value.asType(activationDType))
}

package enum MiniMaxH3ExactKernelMode: String, Sendable, Equatable {
    case disabled
    case boundaryLayout = "boundary-layout"
    case affineQ8 = "affine-q8"
    case affineQ8MLP = "affine-q8-mlp"
    case fastH3Metal = "fasth3-metal"

    package var usesBoundaryLayout: Bool {
        self == .boundaryLayout || self == .affineQ8 || self == .fastH3Metal
    }

    package var usesAffineQ8FeedForward: Bool {
        self == .affineQ8 || self == .affineQ8MLP || self == .fastH3Metal
    }

    package var usesTiledAffineQ8FeedForward: Bool {
        self == .affineQ8MLP || self == .fastH3Metal
    }
}

package enum MiniMaxH3ExactKernelStage: String, CaseIterable, Sendable, Hashable {
    case attentionAdaLN = "k0-attention-adaln"
    case gateAdaLN = "k1-gate-adaln"
    case qkvLayout = "k2a-qkv-layout"
    case qkvProjection = "k2b-qkv-projection"
    case attentionOutput = "k3-attention-output"
    case feedForwardInput = "k4a-feed-forward-input"
    case feedForwardOutput = "k4b-feed-forward-output"
}

struct MiniMaxH3AffineQ8Weights {
    package let codes: MLXArray
    package let scales: MLXArray
    package let biases: MLXArray
}

package struct MiniMaxH3FastH3CompressionGate {
    package enum Storage: Sendable, Equatable {
        case dense
        case affineQ8(groupSize: Int, bits: Int)
    }

    package let storage: Storage
    package let parameters: [MLXArray]

    package init(weight: MLXArray) {
        self.storage = .dense
        self.parameters = [weight]
    }

    package init(
        codes: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        groupSize: Int,
        bits: Int
    ) {
        precondition(groupSize > 0 && bits == 8)
        self.storage = .affineQ8(groupSize: groupSize, bits: bits)
        self.parameters = [codes, scales, biases]
    }

    package func project(_ input: MLXArray) -> MLXArray {
        Self.project(input, storage: storage, parameters: parameters)
    }

    package static func project(
        _ input: MLXArray,
        storage: Storage,
        parameters: [MLXArray]
    ) -> MLXArray {
        switch storage {
        case .dense:
            precondition(parameters.count == 1)
            let weight = parameters[0]
            return MLX.matmul(input.asType(weight.dtype), weight.T)
        case .affineQ8(let groupSize, let bits):
            precondition(parameters.count == 3)
            return MLX.quantizedMM(
                input.asType(parameters[1].dtype),
                parameters[0],
                scales: parameters[1],
                biases: parameters[2],
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            )
        }
    }
}

func miniMaxH3AffineQ8Weights(
    _ linear: Linear
) -> MiniMaxH3AffineQ8Weights? {
    guard let quantized = linear as? QuantizedLinear,
          quantized.bits == 8,
          quantized.groupSize == 64,
          quantized.mode == .affine,
          quantized.bias == nil,
          quantized.globalScale == nil,
          let biases = quantized.biases else {
        return nil
    }
    return MiniMaxH3AffineQ8Weights(
        codes: quantized.weight,
        scales: quantized.scales,
        biases: biases
    )
}

package func miniMaxH3SplitProjectedQKV(
    _ projected: MLXArray,
    heads: Int,
    headDimension: Int
) -> [MLXArray] {
    precondition(projected.dim(-1) == heads * 3 * headDimension)
    return MLX.split(projected, parts: 3, axis: -1).map {
        $0.reshaped(projected.dim(0), projected.dim(1), heads, headDimension)
    }
}
