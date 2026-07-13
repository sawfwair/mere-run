import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

struct Trellis2SparseDecoderOutput {
    let tensor: Trellis2SparseTensor
    let subdivisions: [Trellis2Subdivision]
}

/// Functional MLX port of TRELLIS.2's sparse ConvNeXt decoder. Both shape and
/// PBR decoders use this graph; the shape checkpoint predicts its subdivision
/// tree and the texture checkpoint follows that same tree.
struct Trellis2SparseDecoder {
    enum Kind {
        case shape
        case texture

        var outputChannels: Int {
            switch self {
            case .shape: 7
            case .texture: 6
            }
        }

        var predictsSubdivisions: Bool { self == .shape }
    }

    private static let modelChannels = [1_024, 512, 256, 128, 64]
    private static let blockCounts = [4, 16, 8, 4, 0]

    let kind: Kind
    private let weights: Trellis2WeightStore

    init(kind: Kind, checkpointURL: URL) throws {
        self.kind = kind
        self.weights = try Trellis2WeightStore(url: checkpointURL)
        try validateWeights()
    }

    init(kind: Kind, weights: [String: MLXArray]) throws {
        self.kind = kind
        self.weights = Trellis2WeightStore(values: weights)
        try validateWeights()
    }

    private func validateWeights() throws {
        _ = try weights.checkedTensor("from_latent.weight", shape: [1_024, 32])
        _ = try weights.checkedTensor(
            "output_layer.weight",
            shape: [kind.outputChannels, 64]
        )
        for level in 0..<Self.blockCounts.count {
            let channels = Self.modelChannels[level]
            for block in 0..<Self.blockCounts[level] {
                let prefix = "blocks.\(level).\(block)"
                _ = try weights.checkedTensor(
                    "\(prefix).conv.weight",
                    shape: [channels, 3, 3, 3, channels]
                )
                _ = try weights.checkedTensor(
                    "\(prefix).mlp.0.weight",
                    shape: [4 * channels, channels]
                )
                _ = try weights.checkedTensor(
                    "\(prefix).mlp.2.weight",
                    shape: [channels, 4 * channels]
                )
            }
            if level < Self.modelChannels.count - 1 {
                let prefix = "blocks.\(level).\(Self.blockCounts[level])"
                _ = try weights.checkedTensor(
                    "\(prefix).conv1.weight",
                    shape: [8 * Self.modelChannels[level + 1], 3, 3, 3, channels]
                )
                _ = try weights.checkedTensor(
                    "\(prefix).conv2.weight",
                    shape: [Self.modelChannels[level + 1], 3, 3, 3, Self.modelChannels[level + 1]]
                )
                if kind.predictsSubdivisions {
                    _ = try weights.checkedTensor(
                        "\(prefix).to_subdiv.weight",
                        shape: [8, channels]
                    )
                }
            }
        }
    }

    func decode(
        _ latent: Trellis2SparseTensor,
        guideSubdivisions: [Trellis2Subdivision]? = nil,
        maximumTokens: Int
    ) throws -> Trellis2SparseDecoderOutput {
        guard latent.channels == 32 else {
            throw Trellis2ModelError.invalidInputShape(
                expected: "[tokens, 32] structured latent",
                actual: latent.features.shape
            )
        }
        if kind == .texture, guideSubdivisions?.count != 4 {
            throw Trellis2ModelError.invalidInputShape(
                expected: "four shape subdivision guides",
                actual: [guideSubdivisions?.count ?? 0]
            )
        }

        var hidden = try latent.replacing(features: weights.linear(
            latent.features,
            prefix: "from_latent"
        ))
        var subdivisions = [Trellis2Subdivision]()
        subdivisions.reserveCapacity(4)

        for level in 0..<Self.modelChannels.count {
            for block in 0..<Self.blockCounts[level] {
                hidden = try convNeXtBlock(hidden, prefix: "blocks.\(level).\(block)")
                MLX.eval(hidden.features)
                Memory.clearCache()
            }
            if level < Self.modelChannels.count - 1 {
                let prefix = "blocks.\(level).\(Self.blockCounts[level])"
                let guide: Trellis2Subdivision
                if kind.predictsSubdivisions {
                    let logits = try weights.linear(hidden.features, prefix: "\(prefix).to_subdiv")
                    guide = try Trellis2Subdivision(logits: logits, topology: hidden.topology)
                    subdivisions.append(guide)
                } else {
                    guide = guideSubdivisions![level]
                }
                hidden = try channelToSpatialBlock(
                    hidden,
                    subdivision: guide,
                    prefix: prefix,
                    outputChannels: Self.modelChannels[level + 1],
                    maximumTokens: maximumTokens
                )
                MLX.eval(hidden.features)
                Memory.clearCache()
            }
        }

        let normalized = MLXFast.layerNorm(hidden.features.asType(.float32), eps: 1e-6)
            .asType(hidden.features.dtype)
        let output = try hidden.replacing(features: weights.linear(
            normalized,
            prefix: "output_layer"
        ))
        return Trellis2SparseDecoderOutput(tensor: output, subdivisions: subdivisions)
    }

    private func convNeXtBlock(
        _ input: Trellis2SparseTensor,
        prefix: String
    ) throws -> Trellis2SparseTensor {
        var hidden = try Trellis2SparseOperations.convolution(
            input,
            weights: weights,
            prefix: "\(prefix).conv"
        )
        hidden = try hidden.replacing(features: affineLayerNorm(
            hidden.features,
            prefix: "\(prefix).norm"
        ))
        var features = try weights.linear(hidden.features, prefix: "\(prefix).mlp.0")
        features = silu(features)
        features = try weights.linear(features, prefix: "\(prefix).mlp.2")
        return try input.replacing(features: input.features + features)
    }

    private func channelToSpatialBlock(
        _ input: Trellis2SparseTensor,
        subdivision: Trellis2Subdivision,
        prefix: String,
        outputChannels: Int,
        maximumTokens: Int
    ) throws -> Trellis2SparseTensor {
        let normalized = affineLayerNorm(input.features, prefix: "\(prefix).norm1")
        var hidden = try input.replacing(features: silu(normalized))
        hidden = try Trellis2SparseOperations.convolution(
            hidden,
            weights: weights,
            prefix: "\(prefix).conv1"
        )
        hidden = try Trellis2SparseOperations.channelToSpatial(
            hidden,
            subdivision: subdivision,
            maximumTokens: maximumTokens
        )

        var residual = try Trellis2SparseOperations.channelToSpatial(
            input,
            subdivision: subdivision,
            maximumTokens: maximumTokens
        )
        let repeatCount = outputChannels / residual.channels
        residual = try residual.replacing(features: MLX.repeated(
            residual.features,
            count: repeatCount,
            axis: 1
        ))

        var features = MLXFast.layerNorm(hidden.features.asType(.float32), eps: 1e-6)
            .asType(hidden.features.dtype)
        features = silu(features)
        hidden = try hidden.replacing(features: features)
        hidden = try Trellis2SparseOperations.convolution(
            hidden,
            weights: weights,
            prefix: "\(prefix).conv2"
        )
        return try hidden.replacing(features: hidden.features + residual.features)
    }

    private func affineLayerNorm(_ input: MLXArray, prefix: String) -> MLXArray {
        MLXFast.layerNorm(
            input.asType(.float32),
            weight: weights.values["\(prefix).weight"]!.asType(.float32),
            bias: weights.values["\(prefix).bias"]!.asType(.float32),
            eps: 1e-6
        ).asType(input.dtype)
    }
}
